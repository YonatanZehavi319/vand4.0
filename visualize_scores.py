"""
Visualize per-image scores across all categories.
Two rows per category:
  Top: bad images — Y = SegF1, dot = TP, X = FN
  Bottom: good images — Y = FP area (% pixels flagged), dot = TN, X = FP
"""
import cv2 as cv
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from glob import glob


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <saved_results_dir> <data_dir> [--resize 392] [--seg] [--out_dir ./plots]")
        sys.exit(1)

    results_dir = sys.argv[1]
    data_dir = sys.argv[2]
    resize = 392
    use_seg = '--seg' in sys.argv
    if '--resize' in sys.argv:
        resize = int(sys.argv[sys.argv.index('--resize') + 1])
    out_dir = './plots'
    if '--out_dir' in sys.argv:
        out_dir = sys.argv[sys.argv.index('--out_dir') + 1]
    os.makedirs(out_dir, exist_ok=True)

    suffix = '_binary_seg.png' if use_seg else '_binary.png'
    method_name = 'SegHead' if use_seg else 'Otsu'

    heatmaps_dir = os.path.join(results_dir, 'heatmaps')
    categories = sorted(os.listdir(heatmaps_dir))
    group_sizes = {'vial': 7, 'fruit_jelly': 4}

    # Collect data for all categories
    cat_bad = {}
    cat_good = {}
    for category in categories:
        cat_dir = os.path.join(heatmaps_dir, category)
        sub_dirs = sorted(os.listdir(cat_dir))
        bad_entries = []
        good_entries = []

        for sub_dir in sub_dirs:
            binary_files = sorted(glob(os.path.join(cat_dir, sub_dir, '*' + suffix)))
            is_good = (sub_dir == 'good')

            for binary_path in binary_files:
                image_name = os.path.basename(binary_path).replace(suffix, '')
                binary = cv.imread(binary_path, cv.IMREAD_GRAYSCALE)

                if is_good:
                    if binary is not None:
                        fp_area = (binary > 127).sum() / binary.size
                    else:
                        fp_area = 0.0
                    label = 'FP' if fp_area > 0 else 'TN'
                    good_entries.append((image_name, fp_area, label))
                else:
                    gt_path = os.path.join(data_dir, category, 'ground_truth', sub_dir, f'{image_name}_mask.png')
                    if not os.path.exists(gt_path):
                        continue
                    gt = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
                    sz = binary.shape[0]
                    gt = cv.resize(gt, (sz, sz))
                    pred = (binary > 127).astype(int).flatten()
                    gt_flat = (gt > 127).astype(int).flatten()
                    tp = (pred * gt_flat).sum()
                    fp = (pred * (1 - gt_flat)).sum()
                    fn = ((1 - pred) * gt_flat).sum()
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    label = 'TP' if f1 > 0 else 'FN'
                    bad_entries.append((image_name, f1, label))

        cat_bad[category] = bad_entries
        cat_good[category] = good_entries

    n_cats = len(categories)
    fig, axes = plt.subplots(n_cats * 2, 1, figsize=(20, 2.5 * n_cats * 2), sharex=False)

    for i, category in enumerate(categories):
        ax_bad = axes[i * 2]
        ax_good = axes[i * 2 + 1]
        gs = group_sizes.get(category, 6)

        # --- Bad images (SegF1) ---
        bad = cat_bad[category]
        if bad:
            names_b = [e[0] for e in bad]
            f1s = [e[1] for e in bad]
            lbls_b = [e[2] for e in bad]
            x_b = np.arange(len(names_b))
            for j in range(len(names_b)):
                if lbls_b[j] == 'TP':
                    ax_bad.plot(x_b[j], f1s[j], 'o', color='blue', markersize=5)
                else:
                    ax_bad.plot(x_b[j], f1s[j], 'x', color='orange', markersize=5, markeredgewidth=1.5)
            max_f1 = max(f1s) if max(f1s) > 0 else 0.1
            ax_bad.set_ylim(-0.02 * max_f1, max_f1 * 1.1)
            for g in range(0, len(names_b), gs):
                if (g // gs) % 2 == 1:
                    ax_bad.axvspan(g - 0.5, min(g + gs - 0.5, len(names_b) - 0.5), alpha=0.08, color='gray')
            ax_bad.set_xticks(x_b)
            ax_bad.set_xticklabels(names_b, rotation=90, fontsize=4)
            tp_count = lbls_b.count('TP')
            fn_count = lbls_b.count('FN')
        else:
            tp_count, fn_count = 0, 0
        ax_bad.set_ylabel('SegF1')
        ax_bad.set_title(f'{category} — BAD images ({method_name}) | TP={tp_count} FN={fn_count}')
        ax_bad.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        # --- Good images (FP area) ---
        good = cat_good[category]
        if good:
            names_g = [e[0] for e in good]
            fp_areas = [e[1] for e in good]
            lbls_g = [e[2] for e in good]
            x_g = np.arange(len(names_g))
            for j in range(len(names_g)):
                if lbls_g[j] == 'TN':
                    ax_good.plot(x_g[j], fp_areas[j], 'o', color='green', markersize=5)
                else:
                    ax_good.plot(x_g[j], fp_areas[j], 'x', color='red', markersize=5, markeredgewidth=1.5)
            max_fp = max(fp_areas) if max(fp_areas) > 0 else 0.01
            ax_good.set_ylim(-0.02 * max_fp, max_fp * 1.1)
            for g in range(0, len(names_g), gs):
                if (g // gs) % 2 == 1:
                    ax_good.axvspan(g - 0.5, min(g + gs - 0.5, len(names_g) - 0.5), alpha=0.08, color='gray')
            ax_good.set_xticks(x_g)
            ax_good.set_xticklabels(names_g, rotation=90, fontsize=4)
            fp_count = lbls_g.count('FP')
            tn_count = lbls_g.count('TN')
        else:
            fp_count, tn_count = 0, 0
        ax_good.set_ylabel('FP area %')
        ax_good.set_title(f'{category} — GOOD images ({method_name}) | TN={tn_count} FP={fp_count}')
        ax_good.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='TP (detected anomaly)'),
        Line2D([0], [0], marker='x', color='orange', markersize=8, markeredgewidth=2, linestyle='None', label='FN (missed anomaly)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='TN (correct normal)'),
        Line2D([0], [0], marker='x', color='red', markersize=8, markeredgewidth=2, linestyle='None', label='FP (false alarm)'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')

    plt.suptitle(f'Per-Image Scores — {method_name}', fontsize=14, y=1.005)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'all_categories_{method_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {out_dir}/all_categories_{method_name}.png")


if __name__ == '__main__':
    main()
