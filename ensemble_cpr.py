"""
Ensemble: INP-Former (tiled) + CPR heatmaps.

Both models run independently. This script combines their saved heatmaps.

Usage:
  # Step 1: Run INP-Former with tiling and save maps
  python INP_Former_Multi_Class.py --phase test --data_path ../mvtec --save_maps --tiling --item can

  # Step 2: Run CPR and save maps (from CPR repo)
  python test.py --save-maps --save-dir ./saved_results ...

  # Step 3: Combine heatmaps
  python ensemble_cpr.py \
    --inp_dir ./saved_results/INP-Former-.../heatmaps \
    --cpr_dir /workspace/CPR/saved_results/heatmaps \
    --data_dir ../mvtec \
    --out_dir ./saved_results/ensemble_cpr/heatmaps

  # Step 4: Evaluate
  python seg_f1.py ./saved_results/ensemble_cpr ../mvtec
"""
import numpy as np
import cv2
import os
import sys
import argparse
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def normalize_map(amap):
    return (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)


def _build_jet_reverse_lut():
    """Build a lookup table to reverse jet colormap: RGB -> scalar value."""
    jet_lut = (plt.cm.jet(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    # Create reverse LUT: for each possible RGB, find the closest jet index
    # Use a quantized approach for speed
    reverse = {}
    for i in range(256):
        r, g, b = jet_lut[i]
        reverse[(r, g, b)] = i
    return jet_lut, reverse

def load_heatmap(path):
    """Load a saved heatmap PNG as grayscale (fallback when .npy not available)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32) / 255.0


def load_heatmap_npy(path):
    """Load raw heatmap from .npy file if available."""
    if os.path.exists(path):
        return np.load(path)
    return None


def main(args):
    categories = sorted(os.listdir(args.inp_dir))
    if args.item:
        categories = [c for c in categories if c == args.item]

    save_size = args.save_size
    all_categories_results = []

    for category in categories:
        inp_cat_dir = os.path.join(args.inp_dir, category)
        cpr_cat_dir = os.path.join(args.cpr_dir, category)

        if not os.path.isdir(inp_cat_dir):
            print(f"  Skipping {category}: INP dir not found")
            continue

        sub_dirs = sorted(os.listdir(inp_cat_dir))
        n_combined = 0

        for sub_dir in sub_dirs:
            inp_sub = os.path.join(inp_cat_dir, sub_dir)
            cpr_sub = os.path.join(cpr_cat_dir, sub_dir)
            out_sub = os.path.join(args.out_dir, category, sub_dir)
            os.makedirs(out_sub, exist_ok=True)

            # Find INP-Former heatmaps
            inp_heatmaps = sorted(glob(os.path.join(inp_sub, '*_heatmap.png')))

            for inp_path in inp_heatmaps:
                fname = os.path.basename(inp_path).replace('_heatmap.png', '')

                # Load INP-Former heatmap
                inp_map = load_heatmap(inp_path)
                if inp_map is None:
                    continue

                # Try to load CPR heatmap (matching filename)
                cpr_path = os.path.join(cpr_sub, f'{fname}_heatmap.png')
                cpr_npy = os.path.join(cpr_sub, f'{fname}_heatmap_raw.npy')
                cpr_map = load_heatmap_npy(cpr_npy)
                if cpr_map is None:
                    cpr_map = load_heatmap(cpr_path)

                # Resize both to common size
                inp_resized = cv2.resize(inp_map, (save_size, save_size))
                inp_norm = normalize_map(inp_resized)

                if cpr_map is not None:
                    cpr_resized = cv2.resize(cpr_map, (save_size, save_size))
                    cpr_norm = normalize_map(cpr_resized)
                    # Weighted average (configurable)
                    combined = args.inp_weight * inp_norm + args.cpr_weight * cpr_norm
                    combined = combined / (args.inp_weight + args.cpr_weight)
                    n_combined += 1
                else:
                    combined = inp_norm

                # Save combined heatmap
                plt.imsave(os.path.join(out_sub, f'{fname}_heatmap.png'), combined, cmap='jet')

                # Binary mask
                if args.top_percent is not None:
                    threshold = np.percentile(combined, 100 - args.top_percent)
                    pred_mask = ((combined >= threshold) * 255).astype(np.uint8)
                else:
                    combined_uint8 = (combined * 255).astype(np.uint8)
                    _, pred_mask = cv2.threshold(combined_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                plt.imsave(os.path.join(out_sub, f'{fname}_binary.png'), pred_mask, cmap='gray')

                # Copy GT if exists
                gt_path = os.path.join(args.data_dir, category, 'ground_truth', sub_dir, f'{fname}_mask.png')
                if os.path.exists(gt_path):
                    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    gt_resized = cv2.resize(gt, (save_size, save_size), interpolation=cv2.INTER_NEAREST)
                    plt.imsave(os.path.join(out_sub, f'{fname}_gt.png'), gt_resized, cmap='gray')

                plt.close('all')

        print(f"  {category}: {n_combined} images combined (INP + CPR)")

    print(f"\nEnsemble heatmaps saved to {args.out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble INP-Former (tiled) + CPR heatmaps')
    parser.add_argument('--inp_dir', type=str, required=True, help='Path to INP-Former heatmaps dir')
    parser.add_argument('--cpr_dir', type=str, required=True, help='Path to CPR heatmaps dir')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset (for GT masks)')
    parser.add_argument('--out_dir', type=str, default='./saved_results/ensemble_cpr/heatmaps', help='Output dir')
    parser.add_argument('--item', type=str, default=None, help='Single category')
    parser.add_argument('--save_size', type=int, default=512, help='Output image size')
    parser.add_argument('--inp_weight', type=float, default=1.0, help='Weight for INP-Former heatmap')
    parser.add_argument('--cpr_weight', type=float, default=1.0, help='Weight for CPR heatmap')
    parser.add_argument('--top_percent', type=float, default=None, help='Top X%% threshold. If not set, uses Otsu.')

    args = parser.parse_args()
    main(args)
