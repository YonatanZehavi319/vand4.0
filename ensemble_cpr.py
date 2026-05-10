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
from scipy.stats import genextreme


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


def combine_heatmaps(inp_dir, cpr_dir, fname, save_size, inp_weight, cpr_weight, global_stats=None):
    """Load and combine INP-Former + CPR heatmaps for a single image.
    global_stats: if provided, dict with 'inp_min','inp_max','cpr_min','cpr_max' for
                  global normalization (preserves cross-image differences).
                  If None, uses per-image normalization.
    Returns the combined (weighted average) heatmap, or None if INP heatmap not found."""
    # Load INP-Former heatmap (.npy preferred, PNG fallback)
    inp_npy = os.path.join(inp_dir, f'{fname}_heatmap_raw.npy')
    inp_map = load_heatmap_npy(inp_npy)
    if inp_map is None:
        inp_path = os.path.join(inp_dir, f'{fname}_heatmap.png')
        inp_map = load_heatmap(inp_path)
    if inp_map is None:
        return None, False

    inp_resized = cv2.resize(inp_map, (save_size, save_size))
    if global_stats is not None:
        inp_val = (inp_resized - global_stats['inp_min']) / (global_stats['inp_max'] - global_stats['inp_min'] + 1e-8)
    else:
        inp_val = normalize_map(inp_resized)

    # Load CPR heatmap (.npy preferred, PNG fallback)
    cpr_npy = os.path.join(cpr_dir, f'{fname}_heatmap_raw.npy')
    cpr_map = load_heatmap_npy(cpr_npy)
    if cpr_map is None:
        cpr_path = os.path.join(cpr_dir, f'{fname}_heatmap.png')
        cpr_map = load_heatmap(cpr_path)

    if cpr_map is not None:
        cpr_resized = cv2.resize(cpr_map, (save_size, save_size))
        if global_stats is not None:
            cpr_val = (cpr_resized - global_stats['cpr_min']) / (global_stats['cpr_max'] - global_stats['cpr_min'] + 1e-8)
        else:
            cpr_val = normalize_map(cpr_resized)
        combined = (inp_weight * inp_val + cpr_weight * cpr_val) / (inp_weight + cpr_weight)
        return combined, True
    else:
        return inp_val, False


def compute_global_stats(inp_val_dir, cpr_val_dir, categories, save_size):
    """Compute global min/max per model across all validation heatmaps."""
    inp_min, inp_max = float('inf'), float('-inf')
    cpr_min, cpr_max = float('inf'), float('-inf')

    for category in categories:
        inp_good = os.path.join(inp_val_dir, category, 'good')
        cpr_good = os.path.join(cpr_val_dir, category, 'good')

        for npy_path in sorted(glob(os.path.join(inp_good, '*_heatmap_raw.npy'))):
            amap = np.load(npy_path)
            amap = cv2.resize(amap, (save_size, save_size))
            inp_min = min(inp_min, amap.min())
            inp_max = max(inp_max, amap.max())

        for npy_path in sorted(glob(os.path.join(cpr_good, '*_heatmap_raw.npy'))):
            amap = np.load(npy_path)
            amap = cv2.resize(amap, (save_size, save_size))
            cpr_min = min(cpr_min, amap.min())
            cpr_max = max(cpr_max, amap.max())

    print(f"  Global stats — INP: [{inp_min:.4f}, {inp_max:.4f}], CPR: [{cpr_min:.4f}, {cpr_max:.4f}]")
    return {'inp_min': inp_min, 'inp_max': inp_max, 'cpr_min': cpr_min, 'cpr_max': cpr_max}


def fit_evt_from_validation(inp_val_dir, cpr_val_dir, category, save_size, inp_weight, cpr_weight, global_stats=None):
    """Fit GEV distribution on combined validation/good heatmaps for a category.
    Returns (shape, loc, scale) EVT params."""
    inp_val_good = os.path.join(inp_val_dir, category, 'good')
    cpr_val_good = os.path.join(cpr_val_dir, category, 'good')

    if not os.path.isdir(inp_val_good):
        print(f"  WARNING: No INP validation heatmaps for {category} at {inp_val_good}")
        return None

    # Find all validation heatmaps (.npy only)
    inp_npy_files = sorted(glob(os.path.join(inp_val_good, '*_heatmap_raw.npy')))

    all_pixel_scores = []
    for npy_path in inp_npy_files:
        fname = os.path.basename(npy_path).replace('_heatmap_raw.npy', '')
        combined, _ = combine_heatmaps(inp_val_good, cpr_val_good, fname, save_size, inp_weight, cpr_weight, global_stats=global_stats)
        if combined is not None:
            all_pixel_scores.append(combined.flatten())

    if not all_pixel_scores:
        print(f"  WARNING: No validation heatmaps combined for {category}")
        return None

    all_pixel_scores = np.concatenate(all_pixel_scores)
    # Fit GEV to the tail (top 5% of normal scores), sample max 50k for speed
    tail_threshold = np.percentile(all_pixel_scores, 95)
    tail_scores = all_pixel_scores[all_pixel_scores >= tail_threshold]
    if len(tail_scores) > 500000:
        tail_scores = np.random.choice(tail_scores, 500000, replace=False)
    print(f'  {category}: fitting GEV on {len(tail_scores)} tail samples...')
    shape, loc, scale = genextreme.fit(tail_scores)
    print(f'  {category}: EVT fit: shape={shape:.4f}, loc={loc:.6f}, scale={scale:.6f}')
    return shape, loc, scale


def evt_threshold(combined_map, evt_params, fdr=0.01):
    """Apply EVT-based thresholding. Pixels with p-value < fdr are anomalous."""
    shape, loc, scale = evt_params
    p_values = 1 - genextreme.cdf(combined_map, shape, loc=loc, scale=scale)
    return ((p_values < fdr) * 255).astype(np.uint8)


def main(args):
    categories = sorted(os.listdir(args.inp_dir))
    if args.item:
        categories = [c for c in categories if c == args.item]

    save_size = args.save_size

    # Compute global normalization stats and fit EVT from validation heatmaps
    global_stats = None
    evt_params_per_cat = {}
    if args.evt:
        if not args.inp_val_dir or not args.cpr_val_dir:
            print("ERROR: --evt requires --inp_val_dir and --cpr_val_dir")
            sys.exit(1)
        print("Computing global normalization stats from validation...")
        global_stats = compute_global_stats(args.inp_val_dir, args.cpr_val_dir, categories, save_size)
        print("Fitting EVT from validation heatmaps...")
        for category in categories:
            params = fit_evt_from_validation(
                args.inp_val_dir, args.cpr_val_dir, category,
                save_size, args.inp_weight, args.cpr_weight, global_stats=global_stats)
            if params is not None:
                evt_params_per_cat[category] = params

    for category in categories:
        inp_cat_dir = os.path.join(args.inp_dir, category)
        cpr_cat_dir = os.path.join(args.cpr_dir, category)

        if not os.path.isdir(inp_cat_dir):
            print(f"  Skipping {category}: INP dir not found")
            continue

        sub_dirs = sorted(os.listdir(inp_cat_dir))
        n_combined = 0
        cat_evt = evt_params_per_cat.get(category)

        for sub_dir in sub_dirs:
            inp_sub = os.path.join(inp_cat_dir, sub_dir)
            cpr_sub = os.path.join(cpr_cat_dir, sub_dir)
            out_sub = os.path.join(args.out_dir, category, sub_dir)
            os.makedirs(out_sub, exist_ok=True)

            # Find INP-Former heatmaps (exclude seg_heatmap files)
            inp_heatmaps = sorted(glob(os.path.join(inp_sub, '*_heatmap.png')))
            inp_heatmaps = [p for p in inp_heatmaps if '_seg_heatmap.png' not in p]

            for inp_path in inp_heatmaps:
                fname = os.path.basename(inp_path).replace('_heatmap.png', '')

                combined, was_combined = combine_heatmaps(
                    inp_sub, cpr_sub, fname, save_size, args.inp_weight, args.cpr_weight,
                    global_stats=global_stats if cat_evt else None)
                if combined is None:
                    continue
                if was_combined:
                    n_combined += 1

                # Save combined heatmap
                plt.imsave(os.path.join(out_sub, f'{fname}_heatmap.png'), combined, cmap='jet')

                # Binary mask
                if cat_evt is not None:
                    pred_mask = evt_threshold(combined, cat_evt, fdr=args.evt_fdr)
                elif args.top_percent is not None:
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

        print(f"  {category}: {n_combined} images combined (INP + CPR)" +
              (f" [EVT fdr={args.evt_fdr}]" if cat_evt else ""))

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
    parser.add_argument('--evt', action='store_true', help='Use EVT thresholding (fit on validation heatmaps)')
    parser.add_argument('--evt_fdr', type=float, default=0.01, help='FDR rate for EVT thresholding (default 0.01)')
    parser.add_argument('--inp_val_dir', type=str, default=None, help='Path to INP-Former validation heatmaps dir')
    parser.add_argument('--cpr_val_dir', type=str, default=None, help='Path to CPR validation heatmaps dir')

    args = parser.parse_args()
    main(args)
