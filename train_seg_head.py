"""
Stage 2: Train segmentation head on pseudo-anomalies (INP-Former++ style).
The reconstruction model (encoder, bottleneck, INP extractor, decoder) is frozen.
Only the seg head is trained with Dice loss.
"""
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import numpy as np
import os
import random
import glob
import cv2
import argparse
from functools import partial
from tqdm import tqdm
from torch.nn.init import trunc_normal_

from optimizers import StableAdamW
from utils import WarmCosineScheduler, dice_loss, setup_seed, get_logger
from dataset import get_data_transforms
from models import vit_encoder
from models.uad import INP_Former, SegHead, compute_residual
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from synthesis import generate_perlin_mask

import warnings
warnings.filterwarnings("ignore")


# ---- Pseudo-anomaly generation (CPR-style: 50% DTD texture, 50% self-sourced) ----

def structure_source(img, grid_size=16):
    """Create a structure anomaly source by shuffling grid patches of the image."""
    h, w = img.shape[:2]
    assert h % grid_size == 0 and w % grid_size == 0
    gh, gw = h // grid_size, w // grid_size
    # Split into grid patches
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patches.append(img[i*gh:(i+1)*gh, j*gw:(j+1)*gw].copy())
    random.shuffle(patches)
    # Reassemble
    rows = []
    for i in range(grid_size):
        row = np.concatenate(patches[i*grid_size:(i+1)*grid_size], axis=1)
        rows.append(row)
    result = np.concatenate(rows, axis=0).astype(np.float32)
    # Apply random color augmentation
    aug_type = random.choice(['brightness', 'contrast', 'hue', 'none'])
    if aug_type == 'brightness':
        factor = random.uniform(0.7, 1.3)
        result = np.clip(result * factor, 0, 255)
    elif aug_type == 'contrast':
        mean = result.mean()
        factor = random.uniform(0.5, 1.5)
        result = np.clip((result - mean) * factor + mean, 0, 255)
    elif aug_type == 'hue':
        shift = random.randint(-30, 30)
        result = np.clip(result + shift, 0, 255)
    return result


def generate_anomaly(img_np, dtd_images, transparency_range=(0.5, 1.0)):
    """Generate a pseudo-anomaly from a normal image.
    50% chance DTD texture, 50% chance self-sourced structure."""
    h, w = img_np.shape[:2]
    # Generate Perlin noise mask
    mask = generate_perlin_mask(h, w)

    # Choose anomaly source
    if random.random() < 0.5 and dtd_images:
        # DTD texture
        dtd_path = random.choice(dtd_images)
        texture = cv2.imread(dtd_path)
        if texture is None:
            texture = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        else:
            texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            texture = cv2.resize(texture, (w, h))
        source = texture.astype(np.float32)
    else:
        # Self-sourced structure (grid shuffle)
        source = structure_source(img_np)

    # Blend
    factor = random.uniform(*transparency_range)
    mask_3c = np.stack([mask] * 3, axis=-1)
    anomalous = img_np.astype(np.float32) * (1 - mask_3c * factor) + source * mask_3c * factor
    anomalous = np.clip(anomalous, 0, 255).astype(np.uint8)

    return anomalous, mask


class AnomalyTrainDataset(torch.utils.data.Dataset):
    """Dataset that loads normal images and generates pseudo-anomalies on-the-fly."""
    def __init__(self, data_path, item_list, dtd_path, data_transform, gt_transform, input_size):
        self.data_transform = data_transform
        self.gt_transform = gt_transform
        self.input_size = input_size

        # Collect all normal training images
        self.image_paths = []
        for item in item_list:
            train_dir = os.path.join(data_path, item, 'train', 'good')
            paths = sorted(glob.glob(os.path.join(train_dir, '*.png')) +
                          glob.glob(os.path.join(train_dir, '*.JPG')) +
                          glob.glob(os.path.join(train_dir, '*.bmp')))
            self.image_paths.extend(paths)

        # Collect DTD texture images
        self.dtd_images = []
        if dtd_path and os.path.isdir(dtd_path):
            self.dtd_images = sorted(glob.glob(os.path.join(dtd_path, 'images', '*', '*')))
        if not self.dtd_images:
            print(f"Warning: No DTD images found at {dtd_path}, using self-sourced only")

        print(f"AnomalyTrainDataset: {len(self.image_paths)} normal images, {len(self.dtd_images)} DTD textures")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil.resize((self.input_size, self.input_size)))

        # Generate pseudo-anomaly
        anomalous_np, mask_np = generate_anomaly(img_np, self.dtd_images)

        # Convert to PIL for transforms
        anomalous_pil = Image.fromarray(anomalous_np)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

        # Apply transforms
        anomalous_tensor = self.data_transform(anomalous_pil)
        mask_tensor = self.gt_transform(mask_pil)

        return anomalous_tensor, mask_tensor


def build_model(args, device):
    """Build INP-Former model with same config as Multi-Class training."""
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

    model = INP_Former(
        encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
        decoder=INP_Guided_Decoder, target_layers=target_layers,
        remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP
    )
    return model, embed_dim


def main(args):
    setup_seed(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Build and load pre-trained model
    model, embed_dim = build_model(args, device)
    model = model.to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded pre-trained model from {args.model_path}")

    # Freeze the entire reconstruction model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Create seg head
    seg_head = SegHead(in_channels=embed_dim).to(device)

    # Initialize seg head weights
    for m in seg_head.modules():
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    # Dataset
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    dataset = AnomalyTrainDataset(
        data_path=args.data_path,
        item_list=args.item_list,
        dtd_path=args.dtd_path,
        data_transform=data_transform,
        gt_transform=gt_transform,
        input_size=args.input_size,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    # Optimizer (only seg head parameters)
    optimizer = StableAdamW(
        [{'params': seg_head.parameters()}],
        lr=args.seg_lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10
    )
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=args.seg_lr, final_value=args.seg_lr * 0.1,
        total_iters=args.seg_epochs * len(dataloader), warmup_iters=50
    )

    # Training loop
    print(f"Training seg head for {args.seg_epochs} epochs on {len(dataset)} images")
    for epoch in range(args.seg_epochs):
        seg_head.train()
        loss_list = []
        for anomalous_img, gt_mask in tqdm(dataloader, ncols=80, desc=f'Epoch {epoch+1}/{args.seg_epochs}'):
            anomalous_img = anomalous_img.to(device)
            gt_mask = gt_mask.to(device)
            gt_mask = (gt_mask > 0.5).float()

            # Forward through frozen model
            with torch.no_grad():
                en, de, _ = model(anomalous_img)
                residual = compute_residual(en, de)

            # Seg head prediction
            pred = seg_head(residual, out_size=(args.crop_size, args.crop_size))

            # Dice loss
            loss = dice_loss(pred, gt_mask).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(seg_head.parameters(), max_norm=0.1)
            optimizer.step()
            lr_scheduler.step()
            loss_list.append(loss.item())

        print_fn(f'Epoch [{epoch+1}/{args.seg_epochs}], Dice Loss: {np.mean(loss_list):.4f}')

    # Save seg head
    save_path = os.path.join(os.path.dirname(args.model_path), 'seg_head.pth')
    torch.save(seg_head.state_dict(), save_path)
    print_fn(f'Seg head saved to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation head (stage 2)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained INP-Former model.pth')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset root (e.g. ../mvtec_anomaly_detection)')
    parser.add_argument('--dtd_path', type=str, required=True, help='Path to DTD dataset root')
    parser.add_argument('--dataset', type=str, default='MVTec-AD')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seg_epochs', type=int, default=50)
    parser.add_argument('--seg_lr', type=float, default=1e-3)

    args = parser.parse_args()

    # Category info
    if args.dataset == 'MVTec-AD':
        args.item_list = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    elif args.dataset == 'VisA':
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                          'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

    save_dir = os.path.dirname(args.model_path)
    logger = get_logger('seg_head_train', save_dir)
    print_fn = logger.info

    main(args)
