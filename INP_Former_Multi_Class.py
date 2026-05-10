from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch, evaluation_batch_with_seg, evaluation_batch_tiled, stitch_tiles, fit_evt_null, evt_threshold, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger

# Dataset-Related Modules
from dataset import MVTecDataset, RealIADDataset, TiledImageFolder, TiledMVTecDataset
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

# Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former, SegHead
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings("ignore")


def save_heatmaps(model, dataloader, device, save_dir, item, crop_size, seg_head=None, top_percent=None, min_score=None, evt_params=None, evt_fdr=0.01):
    from utils import cal_anomaly_maps, get_gaussian_kernel, denormalize, min_max_norm
    from models.uad import compute_residual
    model.eval()
    if seg_head is not None:
        seg_head.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, desc=f'Saving maps: {item}', ncols=80):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, crop_size)
            anomaly_map = gaussian_kernel(anomaly_map)

            # Seg head prediction if available
            seg_pred = None
            if seg_head is not None:
                residual = compute_residual(en, de)
                seg_pred = seg_head(residual, out_size=(crop_size, crop_size))

            for i in range(img.shape[0]):
                fname = os.path.splitext(os.path.basename(img_path[i]))[0]
                defect_type = img_path[i].replace('\\', '/').split('/')[-2]
                out_dir = os.path.join(save_dir, item, defect_type)
                os.makedirs(out_dir, exist_ok=True)
                input_img = denormalize(img[i].cpu().numpy())
                raw_amap = anomaly_map[i, 0].cpu().numpy()
                np.save(os.path.join(out_dir, f'{fname}_heatmap_raw.npy'), raw_amap)
                amap = (raw_amap - raw_amap.min()) / (raw_amap.max() - raw_amap.min() + 1e-8)
                plt.imsave(os.path.join(out_dir, f'{fname}_input.png'), input_img)
                plt.imsave(os.path.join(out_dir, f'{fname}_heatmap.png'), amap, cmap='jet')
                amap_color = (plt.cm.jet(amap)[:, :, :3] * 255).astype(np.uint8)
                overlay = cv2.addWeighted(input_img, 0.5, amap_color, 0.5, 0)
                plt.imsave(os.path.join(out_dir, f'{fname}_overlay.png'), overlay)

                # Binary mask
                raw_amap = anomaly_map[i, 0].cpu().numpy()
                if evt_params is not None:
                    pred_mask = evt_threshold(raw_amap, evt_params, fdr=evt_fdr)
                elif min_score is not None and raw_amap.max() < min_score:
                    pred_mask = np.zeros_like(amap, dtype=np.uint8)
                elif top_percent is not None:
                    threshold = np.percentile(amap, 100 - top_percent)
                    pred_mask = ((amap >= threshold) * 255).astype(np.uint8)
                else:
                    amap_uint8 = (amap * 255).astype(np.uint8)
                    _, pred_mask = cv2.threshold(amap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                plt.imsave(os.path.join(out_dir, f'{fname}_binary.png'), pred_mask, cmap='gray')

                # Seg head binary mask (threshold at 0.5)
                if seg_pred is not None:
                    seg_map = seg_pred[i, 0].cpu().numpy()
                    binary_seg = ((seg_map >= 0.5) * 255).astype(np.uint8)
                    plt.imsave(os.path.join(out_dir, f'{fname}_binary_seg.png'), binary_seg, cmap='gray')
                    plt.imsave(os.path.join(out_dir, f'{fname}_seg_heatmap.png'), seg_map, cmap='jet')

                if label[i] == 1:
                    gt_map = gt[i, 0].cpu().numpy()
                    plt.imsave(os.path.join(out_dir, f'{fname}_gt.png'), gt_map, cmap='gray')

                    # Comparison overlay: green=TP, red=FN, blue=FP
                    gt_binary = (gt_map > 0.5).astype(np.uint8)
                    pred_binary = (pred_mask > 127).astype(np.uint8)
                    h, w = gt_binary.shape
                    comp = input_img.copy()
                    tp = (gt_binary == 1) & (pred_binary == 1)
                    fn = (gt_binary == 1) & (pred_binary == 0)
                    fp = (gt_binary == 0) & (pred_binary == 1)
                    comp[tp] = (comp[tp] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)   # green = correct detection
                    comp[fn] = (comp[fn] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)   # red = missed anomaly
                    comp[fp] = (comp[fp] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)   # blue = false alarm
                    plt.imsave(os.path.join(out_dir, f'{fname}_comparison.png'), comp)
                plt.close('all')


def save_heatmaps_tiled(model, dataloader, device, save_dir, item, crop_size, top_percent=None, min_score=None, evt_params=None, evt_fdr=0.01):
    """Save stitched heatmaps from tiled test images."""
    from utils import cal_anomaly_maps, get_gaussian_kernel
    import ast
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    image_tiles = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Saving tiled maps: {item}', ncols=80):
            tile_img, tile_gt, label, img_path, tile_idx, h, w, tile_h, tile_w, n_tiles, positions_str, margin_x, margin_y = batch
            tile_img = tile_img.to(device)
            output = model(tile_img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, tile_img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            for i in range(tile_img.shape[0]):
                path = img_path[i]
                tidx = tile_idx[i].item()
                nt = n_tiles[i].item()
                if path not in image_tiles:
                    positions = ast.literal_eval(positions_str[i])
                    image_tiles[path] = {
                        'maps': [None] * nt, 'gts': [None] * nt,
                        'label': label[i].item(),
                        'h': h[i].item(), 'w': w[i].item(),
                        'tile_h': tile_h[i].item(), 'tile_w': tile_w[i].item(),
                        'positions': positions,
                        'margin_x': margin_x[i].item(), 'margin_y': margin_y[i].item()
                    }
                image_tiles[path]['maps'][tidx] = anomaly_map[i, 0].cpu().numpy()
                image_tiles[path]['gts'][tidx] = tile_gt[i, 0].numpy()

    for path, data in image_tiles.items():
        fname = os.path.splitext(os.path.basename(path))[0]
        defect_type = path.replace('\\', '/').split('/')[-2]
        out_dir = os.path.join(save_dir, item, defect_type)
        os.makedirs(out_dir, exist_ok=True)

        mx, my = data['margin_x'], data['margin_y']
        amap = stitch_tiles(data['maps'], data['h'], data['w'], data['tile_h'], data['tile_w'], data['positions'], mx, my)

        # Resize to save size before saving (keep full-res for metrics)
        save_size = 512
        amap_save = cv2.resize(amap, (save_size, save_size))
        amap_save = (amap_save - amap_save.min()) / (amap_save.max() - amap_save.min() + 1e-8)

        # Save heatmap
        plt.imsave(os.path.join(out_dir, f'{fname}_heatmap.png'), amap_save, cmap='jet')
        np.save(os.path.join(out_dir, f'{fname}_heatmap_raw.npy'), amap)

        # Binary mask (compute on full-res, then resize)
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        if evt_params is not None:
            pred_mask = evt_threshold(amap, evt_params, fdr=evt_fdr)
        elif min_score is not None and amap.max() < min_score:
            pred_mask = np.zeros_like(amap_norm, dtype=np.uint8)
        elif top_percent is not None:
            threshold = np.percentile(amap_norm, 100 - top_percent)
            pred_mask = ((amap_norm >= threshold) * 255).astype(np.uint8)
        else:
            amap_uint8 = (amap_norm * 255).astype(np.uint8)
            _, pred_mask = cv2.threshold(amap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pred_mask_save = cv2.resize(pred_mask, (save_size, save_size), interpolation=cv2.INTER_NEAREST)
        plt.imsave(os.path.join(out_dir, f'{fname}_binary.png'), pred_mask_save, cmap='gray')

        if data['label'] == 1:
            # Load original GT directly — no tiling/stitching needed
            gt_defect = path.replace('\\', '/').split('/')[-2]
            gt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))), 'ground_truth', gt_defect)
            gt_name = os.path.splitext(os.path.basename(path))[0] + '_mask.png'
            gt_path_full = os.path.join(gt_dir, gt_name)
            if os.path.exists(gt_path_full):
                gt_orig = cv2.imread(gt_path_full, cv2.IMREAD_GRAYSCALE)
                gt_save = cv2.resize(gt_orig, (save_size, save_size), interpolation=cv2.INTER_NEAREST)
                plt.imsave(os.path.join(out_dir, f'{fname}_gt.png'), gt_save, cmap='gray')
        plt.close('all')


def save_scores_csv(model, dataloader, device, save_dir, item, crop_size, max_ratio=0.01, metrics=None, top_percent=None):
    from utils import cal_anomaly_maps, get_gaussian_kernel
    import csv
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    rows = []
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, crop_size)
            anomaly_map = gaussian_kernel(anomaly_map)
            anomaly_map_flat = anomaly_map.flatten(1)
            sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][:, :int(anomaly_map_flat.shape[1] * max_ratio)]
            sp_score = sp_score.mean(dim=1)
            for i in range(img.shape[0]):
                score = sp_score[i].item()
                all_scores.append(score)
                all_labels.append(label[i].item())

                # Per-image pixel-level metrics
                amap = anomaly_map[i, 0].cpu().numpy()
                amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
                if top_percent is not None:
                    threshold = np.percentile(amap_norm, 100 - top_percent)
                    pred_binary = (amap_norm >= threshold).astype(int).flatten()
                else:
                    amap_uint8 = (amap_norm * 255).astype(np.uint8)
                    _, pred_mask = cv2.threshold(amap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    pred_binary = (pred_mask > 127).astype(int).flatten()

                gt_map = gt[i, 0].numpy()
                gt_binary = (gt_map > 0.5).astype(int).flatten()
                tp = (pred_binary * gt_binary).sum()
                fp = (pred_binary * (1 - gt_binary)).sum()
                fn = ((1 - pred_binary) * gt_binary).sum()
                px_precision = tp / (tp + fp + 1e-8)
                px_recall = tp / (tp + fn + 1e-8)
                seg_f1 = 2 * px_precision * px_recall / (px_precision + px_recall + 1e-8)
                anomaly_area = gt_binary.sum() / len(gt_binary)
                fp_area = fp / len(pred_binary)

                rows.append({
                    'filename': os.path.basename(img_path[i]),
                    'defect_type': img_path[i].replace('\\', '/').split('/')[-2],
                    'anomaly_score': score,
                    'ground_truth': 'anomaly' if label[i] == 1 else 'normal',
                    'seg_f1': f'{seg_f1:.4f}',
                    'px_precision': f'{px_precision:.4f}',
                    'px_recall': f'{px_recall:.4f}',
                    'anomaly_area': f'{anomaly_area:.4f}',
                    'fp_area': f'{fp_area:.4f}',
                })
    # find threshold that maximizes F1
    from sklearn.metrics import precision_recall_curve
    precs, recs, thrs = precision_recall_curve(all_labels, all_scores)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    best_thr = thrs[np.argmax(f1s[:-1])]
    for row in rows:
        row['predicted'] = 'anomaly' if row['anomaly_score'] >= best_thr else 'normal'
    out_dir = os.path.join(save_dir, 'scores')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'{item}_scores.csv')
    with open(csv_path, 'w', newline='') as f:
        if metrics:
            f.write(f"# Metrics: {', '.join(f'{k}={v:.4f}' for k, v in metrics.items())}\n")
        writer = csv.DictWriter(f, fieldnames=['filename', 'defect_type', 'anomaly_score', 'ground_truth', 'predicted',
                                                'seg_f1', 'px_precision', 'px_recall', 'anomaly_area', 'fp_area'])
        writer.writeheader()
        writer.writerows(rows)


def build_model(args, device):
    """Build a fresh INP-Former model."""
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")

    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim)) for _ in range(1)])
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
                       decoder=INP_Guided_Decoder, target_layers=target_layers,
                       remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                       fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    return model.to(device), embed_dim, Bottleneck, INP_Guided_Decoder, INP_Extractor, INP


def train_one_category(args, item, data_transform, gt_transform, device, use_tiling, tile_overlap):
    """Train a model for a single category."""
    model, embed_dim, Bottleneck, INP_Guided_Decoder, INP_Extractor, INP = build_model(args, device)

    train_path = os.path.join(args.data_path, item, 'train')
    test_path = os.path.join(args.data_path, item)

    if use_tiling:
        train_data = TiledImageFolder(root=train_path, transform=data_transform, overlap=tile_overlap, target_tile=args.target_tile)
        test_data = TiledMVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test", overlap=tile_overlap, target_tile=args.target_tile)
    else:
        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.samples = [(s[0], 0) for s in train_data.samples]
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Initialize
    trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4,
                                       total_iters=args.total_epochs * len(train_dataloader), warmup_iters=100)

    print_fn(f'=== Training {item} === ({len(train_data)} samples)')
    for epoch in range(args.total_epochs):
        model.train()
        loss_list = []
        for img, _ in tqdm(train_dataloader, ncols=80, desc=f'{item} [{epoch+1}/{args.total_epochs}]'):
            img = img.to(device)
            en, de, g_loss = model(img)
            loss = global_cosine_hm_adaptive(en, de, y=3)
            loss = loss + 0.2 * g_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()
        print_fn(f'{item}: epoch [{epoch+1}/{args.total_epochs}], loss:{np.mean(loss_list):.4f}')

    # Save model per category
    cat_save_dir = os.path.join(args.save_dir, args.save_name, item)
    os.makedirs(cat_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cat_save_dir, 'model.pth'))
    print_fn(f'{item}: model saved to {cat_save_dir}/model.pth')

    # Evaluate
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    if use_tiling:
        results = evaluation_batch_tiled(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
    else:
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
    print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
        item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
    return results


def test_one_category(args, item, data_transform, gt_transform, device, use_tiling, tile_overlap, embed_dim):
    """Test a model for a single category."""
    model, embed_dim, *_ = build_model(args, device)

    cat_save_dir = os.path.join(args.save_dir, args.save_name, item)
    model.load_state_dict(torch.load(os.path.join(cat_save_dir, 'model.pth')), strict=True)
    model.eval()

    test_path = os.path.join(args.data_path, item)
    if use_tiling:
        test_data = TiledMVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test", overlap=tile_overlap, target_tile=args.target_tile)
    else:
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    # Seg head
    seg_head_model = None
    if args.seg_head:
        seg_head_path = os.path.join(cat_save_dir, 'seg_head.pth')
        if os.path.exists(seg_head_path):
            seg_head_model = SegHead(in_channels=embed_dim).to(device)
            seg_head_model.load_state_dict(torch.load(seg_head_path, map_location=device))
            seg_head_model.eval()
            print_fn(f'{item}: loaded seg head')

    # EVT
    evt_params = None
    if args.evt:
        train_path = os.path.join(args.data_path, item, 'train')
        evt_data = ImageFolder(root=train_path, transform=data_transform)
        evt_dl = torch.utils.data.DataLoader(evt_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        evt_params = fit_evt_null(model, evt_dl, device)
        print_fn(f'{item}: EVT fitted (shape={evt_params[0]:.4f}, loc={evt_params[1]:.6f}, scale={evt_params[2]:.6f})')

    # Evaluate
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    if use_tiling:
        results = evaluation_batch_tiled(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
    elif seg_head_model is not None:
        results = evaluation_batch_with_seg(model, seg_head_model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
    else:
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
    print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
        item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

    # Save maps
    if args.save_maps:
        map_dir = os.path.join(args.save_dir, args.save_name, 'heatmaps')
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        if use_tiling:
            save_heatmaps_tiled(model, test_dataloader, device, map_dir, item, args.crop_size, top_percent=args.top_percent, min_score=args.min_score, evt_params=evt_params, evt_fdr=args.evt_fdr)
        else:
            save_heatmaps(model, test_dataloader, device, map_dir, item, args.crop_size, seg_head=seg_head_model, top_percent=args.top_percent, min_score=args.min_score, evt_params=evt_params, evt_fdr=args.evt_fdr)
        print_fn(f'{item}: heatmaps saved')

    # Save scores
    if args.save_scores:
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        save_scores_csv(model, test_dataloader, device, os.path.join(args.save_dir, args.save_name), item, args.crop_size,
                        metrics={'I-AUROC': auroc_sp, 'I-AP': ap_sp, 'I-F1': f1_sp,
                                 'P-AUROC': auroc_px, 'P-AP': ap_px, 'P-F1': f1_px, 'P-AUPRO': aupro_px},
                        top_percent=args.top_percent)
        print_fn(f'{item}: scores saved')

    return results


def validate_one_category(args, item, data_transform, device, use_tiling, tile_overlap):
    """Run model on validation/good images and save heatmaps only (no metrics)."""
    from utils import cal_anomaly_maps, get_gaussian_kernel

    model, embed_dim, *_ = build_model(args, device)
    cat_save_dir = os.path.join(args.save_dir, args.save_name, item)
    model.load_state_dict(torch.load(os.path.join(cat_save_dir, 'model.pth')), strict=True)
    model.eval()

    val_path = os.path.join(args.data_path, item, 'validation', 'good')
    if not os.path.isdir(val_path):
        print_fn(f'{item}: no validation/good dir found, skipping')
        return

    val_data = ImageFolder(root=os.path.join(args.data_path, item, 'validation'), transform=data_transform)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    # Save to val_heatmaps/ (separate from test heatmaps/)
    out_dir = os.path.join(args.save_dir, args.save_name, 'val_heatmaps', item, 'good')
    os.makedirs(out_dir, exist_ok=True)

    sample_idx = 0
    with torch.no_grad():
        for imgs, _ in tqdm(val_dataloader, desc=f'Validation maps: {item}', ncols=80):
            imgs = imgs.to(device)
            output = model(imgs)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, args.crop_size)
            anomaly_map = gaussian_kernel(anomaly_map)

            for i in range(imgs.shape[0]):
                fpath = val_data.samples[sample_idx][0]
                fname = os.path.splitext(os.path.basename(fpath))[0]
                raw_amap = anomaly_map[i, 0].cpu().numpy()
                np.save(os.path.join(out_dir, f'{fname}_heatmap_raw.npy'), raw_amap)
                sample_idx += 1

    print_fn(f'{item}: {sample_idx} validation heatmaps saved to {out_dir}')


def main(args):
    setup_seed(1)

    lighting_aug = getattr(args, 'lighting_aug', False) and args.phase == 'train'
    lighting_intensity = (args.lighting_min, args.lighting_max)
    use_tiling = getattr(args, 'tiling', False)
    tile_overlap = getattr(args, 'tile_overlap', 0.2)
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size, lighting_aug=lighting_aug, lighting_intensity=lighting_intensity, lighting_prob=args.lighting_prob, tiling=use_tiling)

    # Determine which categories to process
    if args.item:
        items_to_process = [args.item]
    else:
        items_to_process = args.item_list

    # Encoder info for embed_dim
    if 'small' in args.encoder:
        embed_dim = 384
    elif 'base' in args.encoder:
        embed_dim = 768
    elif 'large' in args.encoder:
        embed_dim = 1024

    all_results = []
    for item in items_to_process:
        print_fn(f'\n{"="*20} {item} {"="*20}')
        if args.phase == 'train':
            results = train_one_category(args, item, data_transform, gt_transform, device, use_tiling, tile_overlap)
            all_results.append(results)
        elif args.phase == 'test':
            results = test_one_category(args, item, data_transform, gt_transform, device, use_tiling, tile_overlap, embed_dim)
            all_results.append(results)
        elif args.phase == 'validation':
            validate_one_category(args, item, data_transform, device, use_tiling, tile_overlap)

    # Print mean across all categories
    if all_results and len(all_results) > 1:
        mean_results = np.mean(all_results, axis=0)
        print_fn('\nMean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(*mean_results))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'E:\IMSN-LW\dataset\mvtec_anomaly_detection') # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='INP-Former-Multi-Class')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)

    # training info
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--save_maps', action='store_true', help='Save anomaly heatmaps during test phase')
    parser.add_argument('--save_scores', action='store_true', help='Save per-image anomaly scores as CSV')
    parser.add_argument('--seg_head', action='store_true', help='Use segmentation head during test (requires seg_head.pth)')
    parser.add_argument('--top_percent', type=float, default=None, help='Top X%% of pixels marked as anomalous (e.g. 5). If not set, uses Otsu.')
    parser.add_argument('--min_score', type=float, default=None, help='Min raw anomaly score to trigger masking. Below this, output all black.')
    parser.add_argument('--evt', action='store_true', help='Use Extreme Value Theory for thresholding (fits GEV to training scores)')
    parser.add_argument('--evt_fdr', type=float, default=0.01, help='FDR rate for EVT-based BH thresholding (default 0.01)')
    parser.add_argument('--lighting_aug', action='store_true', help='Apply random lighting augmentation during training')
    parser.add_argument('--lighting_min', type=float, default=0.08, help='Min lighting augmentation intensity (default 0.08)')
    parser.add_argument('--lighting_max', type=float, default=0.2, help='Max lighting augmentation intensity (default 0.2)')
    parser.add_argument('--lighting_prob', type=float, default=0.5, help='Probability of applying lighting augmentation (default 0.5)')
    parser.add_argument('--tiling', action='store_true', help='Use 2x2 overlapping tiling for train and test')
    parser.add_argument('--tile_overlap', type=float, default=0.2, help='Tile overlap ratio (default 0.2)')
    parser.add_argument('--target_tile', type=int, default=1000, help='Target tile size in pixels (default 1000). Lower = more tiles = more detail.')
    parser.add_argument('--item', type=str, default=None, help='Train/test a single category (e.g. can). If not set, runs all categories.')

    args = parser.parse_args()
    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # category info
    if args.dataset == 'MVTec-AD':
        # args.data_path = 'E:\IMSN-LW\dataset\mvtec_anomaly_detection' # '/path/to/dataset/MVTec-AD/'
        args.item_list = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    elif args.dataset == 'VisA':
        # args.data_path = r'E:\IMSN-LW\dataset\VisA_pytorch\1cls'  # '/path/to/dataset/VisA/'
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset == 'Real-IAD':
        # args.data_path = 'E:\IMSN-LW\dataset\Real-IAD'  # '/path/to/dataset/Real-IAD/'
        args.item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    main(args)
