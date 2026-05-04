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
from utils import evaluation_batch, evaluation_batch_with_seg, evaluation_batch_tiled, stitch_tiles, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger

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


def save_heatmaps(model, dataloader, device, save_dir, item, crop_size, seg_head=None, top_percent=None, min_score=None):
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
                amap = anomaly_map[i, 0].cpu().numpy()
                amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
                plt.imsave(os.path.join(out_dir, f'{fname}_input.png'), input_img)
                plt.imsave(os.path.join(out_dir, f'{fname}_heatmap.png'), amap, cmap='jet')
                amap_color = (plt.cm.jet(amap)[:, :, :3] * 255).astype(np.uint8)
                overlay = cv2.addWeighted(input_img, 0.5, amap_color, 0.5, 0)
                plt.imsave(os.path.join(out_dir, f'{fname}_overlay.png'), overlay)

                # Binary mask
                raw_max = anomaly_map[i, 0].cpu().numpy().max()
                if min_score is not None and raw_max < min_score:
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


def save_heatmaps_tiled(model, dataloader, device, save_dir, item, crop_size, top_percent=None, min_score=None):
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

        # Binary mask (compute on full-res, then resize)
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        if min_score is not None and amap.max() < min_score:
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


def main(args):
    # Fixing the Random Seed
    setup_seed(1)

    # Data Preparation
    lighting_aug = getattr(args, 'lighting_aug', False) and args.phase == 'train'
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size, lighting_aug=lighting_aug)

    use_tiling = getattr(args, 'tiling', False)
    tile_overlap = getattr(args, 'tile_overlap', 0.5)

    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA':
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):
            train_path = os.path.join(args.data_path, item, 'train')
            test_path = os.path.join(args.data_path, item)

            if use_tiling:
                train_data = TiledImageFolder(root=train_path, transform=data_transform, overlap=tile_overlap)
                test_data = TiledMVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test", overlap=tile_overlap)
            else:
                train_data = ImageFolder(root=train_path, transform=data_transform)
                train_data.classes = item
                train_data.class_to_idx = {item: i}
                train_data.samples = [(sample[0], i) for sample in train_data.samples]
                test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)
        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif args.dataset == 'Real-IAD' :
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):
            train_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                        gt_transform=gt_transform,
                                        phase='train')
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            test_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                       gt_transform=gt_transform,
                                       phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)

        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                       drop_last=True)
    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder info
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(args.INP_num, embed_dim))
                     for _ in range(1)])

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    model = model.to(device)

    if args.phase == 'train':
        # Model Initialization
        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=args.total_epochs*len(train_dataloader),
                                           warmup_iters=100)
        print_fn('train image number:{}'.format(len(train_data)))

        # Train
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80):
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
            print_fn('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list)))
            if (epoch + 1) % args.total_epochs == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(args.item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                                  num_workers=4)
                    if use_tiling:
                        results = evaluation_batch_tiled(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    else:
                        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)
                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))
                model.train()
    elif args.phase == 'test':
        # Test
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, 'model.pth')), strict=True)

        # Load seg head if requested
        seg_head_model = None
        if args.seg_head:
            seg_head_path = os.path.join(args.save_dir, args.save_name, 'seg_head.pth')
            seg_head_model = SegHead(in_channels=embed_dim).to(device)
            seg_head_model.load_state_dict(torch.load(seg_head_path, map_location=device))
            seg_head_model.eval()
            print_fn(f'Loaded seg head from {seg_head_path}')

        auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
        auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
        model.eval()
        map_dir = os.path.join(args.save_dir, args.save_name, 'heatmaps')
        for item, test_data in zip(args.item_list, test_data_list):
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=4)
            if use_tiling:
                results = evaluation_batch_tiled(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            elif seg_head_model is not None:
                results = evaluation_batch_with_seg(model, seg_head_model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            else:
                results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            f1_sp_list.append(f1_sp)
            auroc_px_list.append(auroc_px)
            ap_px_list.append(ap_px)
            f1_px_list.append(f1_px)
            aupro_px_list.append(aupro_px)
            print_fn(
                '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
            if args.save_maps:
                test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                              num_workers=4)
                if use_tiling:
                    save_heatmaps_tiled(model, test_dataloader, device, map_dir, item, args.crop_size, top_percent=args.top_percent, min_score=args.min_score)
                else:
                    save_heatmaps(model, test_dataloader, device, map_dir, item, args.crop_size, seg_head=seg_head_model, top_percent=args.top_percent, min_score=args.min_score)
                print_fn(f'{item}: heatmaps saved to {map_dir}/{item}/')
            if args.save_scores:
                test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                              num_workers=4)
                scores_dir = os.path.join(args.save_dir, args.save_name, 'scores')
                save_scores_csv(model, test_dataloader, device, os.path.join(args.save_dir, args.save_name), item, args.crop_size,
                                metrics={'I-AUROC': auroc_sp, 'I-AP': ap_sp, 'I-F1': f1_sp,
                                         'P-AUROC': auroc_px, 'P-AP': ap_px, 'P-F1': f1_px, 'P-AUPRO': aupro_px},
                                top_percent=args.top_percent)
                print_fn(f'{item}: scores saved to {scores_dir}/{item}_scores.csv')

        print_fn(
            'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))


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
    parser.add_argument('--lighting_aug', action='store_true', help='Apply random lighting augmentation during training')
    parser.add_argument('--tiling', action='store_true', help='Use 2x2 overlapping tiling for train and test')
    parser.add_argument('--tile_overlap', type=float, default=0.2, help='Tile overlap ratio (default 0.2)')

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
