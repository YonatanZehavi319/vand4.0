"""
Ensemble: Run INP-Former with tiling + without tiling (whole image),
combine their heatmaps by averaging.

Usage:
  # Train both models for all categories
  python ensemble.py --phase train --data_path ../mvtec --total_epochs 40

  # Train for one category
  python ensemble.py --phase train --data_path ../mvtec --total_epochs 40 --item can

  # Test and save combined heatmaps
  python ensemble.py --phase test --data_path ../mvtec --save_maps

  # Test one category with EVT
  python ensemble.py --phase test --data_path ../mvtec --save_maps --item can --evt --evt_fdr 0.3
"""
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import (evaluation_batch, evaluation_batch_tiled, cal_anomaly_maps,
                   get_gaussian_kernel, fit_evt_null, evt_threshold,
                   WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger)

from dataset import MVTecDataset, TiledImageFolder, TiledMVTecDataset
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def build_model(args, device):
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
    return model.to(device), Bottleneck, INP_Guided_Decoder, INP_Extractor, INP


def train_model(args, item, data_transform, gt_transform, device, use_tiling, tile_overlap, model_tag):
    """Train a single model (tiled or whole)."""
    model, Bottleneck, INP_Guided_Decoder, INP_Extractor, INP = build_model(args, device)

    train_path = os.path.join(args.data_path, item, 'train')
    if use_tiling:
        train_data = TiledImageFolder(root=train_path, transform=data_transform, overlap=tile_overlap)
    else:
        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.samples = [(s[0], 0) for s in train_data.samples]

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

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

    print_fn(f'=== Training {item} [{model_tag}] === ({len(train_data)} samples)')
    for epoch in range(args.total_epochs):
        model.train()
        loss_list = []
        for img, _ in tqdm(train_dataloader, ncols=80, desc=f'{item}[{model_tag}] [{epoch+1}/{args.total_epochs}]'):
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
        print_fn(f'{item}[{model_tag}]: epoch [{epoch+1}/{args.total_epochs}], loss:{np.mean(loss_list):.4f}')

    cat_save_dir = os.path.join(args.save_dir, args.save_name, item)
    os.makedirs(cat_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cat_save_dir, f'model_{model_tag}.pth'))
    print_fn(f'{item}[{model_tag}]: saved to {cat_save_dir}/model_{model_tag}.pth')


def get_heatmaps_whole(model, test_data, device, crop_size, batch_size):
    """Get per-image heatmaps from whole-image model. Returns dict: img_path -> anomaly_map (numpy)."""
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    heatmaps = {}
    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80, desc='Whole-image heatmaps'):
            img = img.to(device)
            en, de = model(img)[0], model(img)[1]
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)
            for i in range(img.shape[0]):
                heatmaps[img_path[i]] = anomaly_map[i, 0].cpu().numpy()
    return heatmaps


def get_heatmaps_tiled(model, test_data, device, crop_size, batch_size):
    """Get per-image stitched heatmaps from tiled model. Returns dict: img_path -> anomaly_map (numpy)."""
    from utils import stitch_tiles
    import ast
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    image_tiles = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=80, desc='Tiled heatmaps'):
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
                        'maps': [None] * nt,
                        'h': h[i].item(), 'w': w[i].item(),
                        'tile_h': tile_h[i].item(), 'tile_w': tile_w[i].item(),
                        'positions': positions,
                        'margin_x': margin_x[i].item(), 'margin_y': margin_y[i].item()
                    }
                image_tiles[path]['maps'][tidx] = anomaly_map[i, 0].cpu().numpy()

    heatmaps = {}
    for path, data in image_tiles.items():
        mx, my = data['margin_x'], data['margin_y']
        stitched = stitch_tiles(data['maps'], data['h'], data['w'], data['tile_h'], data['tile_w'], data['positions'], mx, my)
        heatmaps[path] = stitched
    return heatmaps


def normalize_map(amap):
    return (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)


def test_ensemble(args, item, data_transform, gt_transform, device):
    """Test ensemble: load both models, get heatmaps, combine, save."""
    cat_save_dir = os.path.join(args.save_dir, args.save_name, item)
    test_path = os.path.join(args.data_path, item)

    # Load whole-image model
    model_whole, *_ = build_model(args, device)
    model_whole.load_state_dict(torch.load(os.path.join(cat_save_dir, 'model_whole.pth')), strict=True)
    model_whole.eval()
    test_data_whole = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    # Load tiled model
    model_tiled, *_ = build_model(args, device)
    model_tiled.load_state_dict(torch.load(os.path.join(cat_save_dir, 'model_tiled.pth')), strict=True)
    model_tiled.eval()
    test_data_tiled = TiledMVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform,
                                         phase="test", overlap=args.tile_overlap)

    # EVT (fit on whole-image model's training data)
    evt_params = None
    if args.evt:
        train_path = os.path.join(args.data_path, item, 'train')
        evt_data = ImageFolder(root=train_path, transform=data_transform)
        evt_dl = DataLoader(evt_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        evt_params = fit_evt_null(model_whole, evt_dl, device)
        print_fn(f'{item}: EVT fitted')

    # Get heatmaps from both models
    heatmaps_whole = get_heatmaps_whole(model_whole, test_data_whole, device, args.crop_size, args.batch_size)
    heatmaps_tiled = get_heatmaps_tiled(model_tiled, test_data_tiled, device, args.crop_size, args.batch_size)

    # Combine and save
    map_dir = os.path.join(args.save_dir, args.save_name, 'heatmaps')
    save_size = 512

    for img_path in heatmaps_whole:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        defect_type = img_path.replace('\\', '/').split('/')[-2]
        out_dir = os.path.join(map_dir, item, defect_type)
        os.makedirs(out_dir, exist_ok=True)

        # Get both heatmaps, resize to common size
        amap_whole = heatmaps_whole[img_path]
        amap_whole_resized = cv2.resize(normalize_map(amap_whole), (save_size, save_size))

        if img_path in heatmaps_tiled:
            amap_tiled = heatmaps_tiled[img_path]
            amap_tiled_resized = cv2.resize(normalize_map(amap_tiled), (save_size, save_size))
            combined = (amap_whole_resized + amap_tiled_resized) / 2
        else:
            combined = amap_whole_resized

        # Save combined heatmap
        plt.imsave(os.path.join(out_dir, f'{fname}_heatmap.png'), combined, cmap='jet')

        # Binary mask
        if evt_params is not None:
            # EVT on the raw combined scores (un-normalized)
            raw_combined = cv2.resize(amap_whole, (save_size, save_size))
            if img_path in heatmaps_tiled:
                raw_tiled = cv2.resize(amap_tiled, (save_size, save_size))
                raw_combined = (raw_combined + raw_tiled) / 2
            pred_mask = evt_threshold(raw_combined, evt_params, fdr=args.evt_fdr)
        elif args.top_percent is not None:
            threshold = np.percentile(combined, 100 - args.top_percent)
            pred_mask = ((combined >= threshold) * 255).astype(np.uint8)
        else:
            combined_uint8 = (combined * 255).astype(np.uint8)
            _, pred_mask = cv2.threshold(combined_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        plt.imsave(os.path.join(out_dir, f'{fname}_binary.png'), pred_mask, cmap='gray')

        # GT
        idx = list(test_data_whole.img_paths).index(img_path) if img_path in test_data_whole.img_paths else -1
        if idx >= 0 and test_data_whole.labels[idx] == 1:
            gt_path = test_data_whole.gt_paths[idx]
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_resized = cv2.resize(gt, (save_size, save_size), interpolation=cv2.INTER_NEAREST)
            plt.imsave(os.path.join(out_dir, f'{fname}_gt.png'), gt_resized, cmap='gray')

        plt.close('all')

    print_fn(f'{item}: ensemble heatmaps saved to {map_dir}/{item}/')


def main(args):
    setup_seed(1)
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size,
                                                        lighting_aug=(args.lighting_aug and args.phase == 'train'))

    items = [args.item] if args.item else args.item_list

    for item in items:
        print_fn(f'\n{"="*20} {item} {"="*20}')
        if args.phase == 'train':
            # Train whole-image model
            train_model(args, item, data_transform, gt_transform, device,
                       use_tiling=False, tile_overlap=0, model_tag='whole')
            # Train tiled model
            train_model(args, item, data_transform, gt_transform, device,
                       use_tiling=True, tile_overlap=args.tile_overlap, model_tag='tiled')
        elif args.phase == 'test':
            test_ensemble(args, item, data_transform, gt_transform, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble: tiled + whole image INP-Former')
    parser.add_argument('--dataset', type=str, default='MVTec-AD')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='INP-Former-Ensemble')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--total_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--item', type=str, default=None)
    parser.add_argument('--save_maps', action='store_true')
    parser.add_argument('--lighting_aug', action='store_true')
    parser.add_argument('--tile_overlap', type=float, default=0.0)
    parser.add_argument('--top_percent', type=float, default=None)
    parser.add_argument('--evt', action='store_true')
    parser.add_argument('--evt_fdr', type=float, default=0.01)

    args = parser.parse_args()
    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_INP_num={args.INP_num}'

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'MVTec-AD':
        args.item_list = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    elif args.dataset == 'VisA':
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                          'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

    main(args)
