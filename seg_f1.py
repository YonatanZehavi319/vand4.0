import cv2 as cv
import numpy as np
import os
import sys
from glob import glob

def compute_seg_f1(binary_path, gt_path, resize=None):
    binary = cv.imread(binary_path, cv.IMREAD_GRAYSCALE)
    gt = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
    # Auto-detect size from binary if resize not specified
    if resize is None:
        resize = binary.shape[0]
    gt = cv.resize(gt, (resize, resize))
    binary = cv.resize(binary, (resize, resize))
    pred = (binary > 127).astype(int).flatten()
    gt = (gt > 127).astype(int).flatten()
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <saved_results_dir> <data_dir> [--resize 320] [--bh] [--seg] [--per-image] [--item can]")
        print(f"  e.g. python {sys.argv[0]} ./saved_results ./data/mvtec")
        print(f"       python {sys.argv[0]} ./saved_results ./data/mvtec --item can --per-image")
        sys.exit(1)

    results_dir = sys.argv[1]
    data_dir = sys.argv[2]
    resize = None
    use_bh = '--bh' in sys.argv
    use_seg = '--seg' in sys.argv
    per_image = '--per-image' in sys.argv
    single_item = None
    if '--item' in sys.argv:
        single_item = sys.argv[sys.argv.index('--item') + 1]
    if '--resize' in sys.argv:
        resize = int(sys.argv[sys.argv.index('--resize') + 1])

    if use_seg:
        suffix = '_binary_seg.png'
        method_name = 'SegHead'
    elif use_bh:
        suffix = '_binary_bh.png'
        method_name = 'BH'
    else:
        suffix = '_binary.png'
        method_name = 'Otsu'
    print(f"Method: {method_name}")

    heatmaps_dir = os.path.join(results_dir, 'heatmaps')
    categories = sorted(os.listdir(heatmaps_dir))
    if single_item:
        categories = [c for c in categories if c == single_item]
        if not categories:
            print(f"Category '{single_item}' not found in {heatmaps_dir}")
            sys.exit(1)

    all_f1s = []
    all_precisions = []
    all_recalls = []

    for category in categories:
        cat_dir = os.path.join(heatmaps_dir, category)
        anomaly_types = [d for d in sorted(os.listdir(cat_dir)) if d != 'good']

        cat_tp, cat_fp, cat_fn = 0, 0, 0
        cat_results = []

        for anomaly_type in anomaly_types:
            binary_files = sorted(glob(os.path.join(cat_dir, anomaly_type, '*' + suffix)))

            for binary_path in binary_files:
                image_name = os.path.basename(binary_path).replace(suffix, '')
                gt_path = os.path.join(data_dir, category, 'ground_truth', anomaly_type, f'{image_name}_mask.png')

                if not os.path.exists(gt_path):
                    print(f"  Warning: GT not found for {category}/{anomaly_type}/{image_name}")
                    continue

                p, r, f1 = compute_seg_f1(binary_path, gt_path, resize)
                cat_results.append((anomaly_type, image_name, p, r, f1))

                binary = cv.imread(binary_path, cv.IMREAD_GRAYSCALE)
                match_size = binary.shape[0] if resize is None else resize
                binary = cv.resize(binary, (match_size, match_size))
                gt = cv.resize(cv.imread(gt_path, cv.IMREAD_GRAYSCALE), (match_size, match_size))
                pred = (binary > 127).astype(int).flatten()
                gt_flat = (gt > 127).astype(int).flatten()
                cat_tp += (pred * gt_flat).sum()
                cat_fp += (pred * (1 - gt_flat)).sum()
                cat_fn += ((1 - pred) * gt_flat).sum()

        cat_precision = cat_tp / (cat_tp + cat_fp + 1e-8)
        cat_recall = cat_tp / (cat_tp + cat_fn + 1e-8)
        cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall + 1e-8)
        all_f1s.append(cat_f1)
        all_precisions.append(cat_precision)
        all_recalls.append(cat_recall)

        print(f"  {category:<15} SegF1={cat_f1:.4f}  P={cat_precision:.4f}  R={cat_recall:.4f}")
        if per_image:
            for atype, name, p, r, f1 in sorted(cat_results, key=lambda x: x[4]):
                print(f"    {atype}/{name:<20} F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

    print(f"\n  {'Average':<15} SegF1={np.mean(all_f1s):.4f}  P={np.mean(all_precisions):.4f}  R={np.mean(all_recalls):.4f}  ({len(all_f1s)} categories)")


if __name__ == '__main__':
    main()
