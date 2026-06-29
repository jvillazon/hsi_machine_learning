"""
Prediction Analysis Script

Iterates over a directory of subdirectories, each containing:
  - A *_predictions.csv  (image grid of predicted class names, no header/index)
  - A *_probabilities.tif (shape: n_classes x H x W, class labels in ImageJ metadata)

For each subdirectory this script:
  1. Counts the number of pixels assigned to each predicted class.
  2. Computes two multiclass performance metrics for a user-specified correct class.
     Both metrics exclude pixels where p_correct == 0 or any No Match slice has p > 0.

     a) Mean log-loss  : mean(-log(p_correct))  — lower is better; 0 = perfect certainty.
     b) Mean Pearson r : mean per-pixel Pearson correlation between the one-hot ground-truth
                        vector and the predicted probability vector (No Match slices excluded
                        from both). Range [-1, 1]; 1 = perfect alignment.

Usage:
    python analyze_predictions.py <input_dir> <correct_class>
    python analyze_predictions.py <input_dir> <correct_class> --save_summary

Arguments:
    input_dir      Directory containing per-image subdirectories with prediction outputs.
    correct_class  Name of the ground-truth / expected class to evaluate confidence against.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tifffile


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def count_predictions_from_csv(csv_path: str) -> dict:
    """
    Read a predictions CSV (no header, no index) and return {class_name: pixel_count}.
    """
    df = pd.read_csv(csv_path, header=None, dtype=str)
    counts = df.stack().value_counts().to_dict()
    return counts


def load_probability_stack(tif_path: str, class_map: dict | None = None):
    """
    Load a probability TIFF and return (data, labels).

    Parameters
    ----------
    tif_path  : path to the probability TIFF
    class_map : optional dict mapping slice index (int) -> class name (str).
                When provided, overrides any labels found in the TIFF metadata.

    Returns
    -------
    data   : ndarray of shape (n_classes, H, W)
    labels : list of str, one per class slice
    """
    with tifffile.TiffFile(tif_path) as tif:
        data = tif.asarray()  # (n_classes, H, W)
        meta_labels = None
        if tif.imagej_metadata and "Labels" in tif.imagej_metadata:
            meta_labels = tif.imagej_metadata["Labels"]

    n_slices = data.shape[0]

    if class_map is not None:
        # Build labels from the provided mapping; fall back to slice index string for
        # any slice not covered by the map.
        labels = [class_map.get(i, class_map.get(str(i), str(i))) for i in range(n_slices)]
    elif meta_labels is not None:
        labels = list(meta_labels)
    else:
        raise ValueError(
            f"No class labels found in TIFF metadata: {tif_path}\n"
            "Provide a --class_map or ensure the TIFF was written with ImageJ "
            "metadata containing a 'Labels' key."
        )

    return data, labels


def find_class_index(labels: list, class_name: str) -> int:
    """
    Case-insensitive search for class_name in labels.
    Raises ValueError if not found.
    """
    target = class_name.lower()
    for i, label in enumerate(labels):
        if label.lower() == target:
            return i
    raise ValueError(
        f"Class '{class_name}' not found in probability TIFF labels.\n"
        f"Available classes: {labels}"
    )


def compute_metrics(tif_path: str, correct_class: str, class_map: dict | None = None):
    """
    Compute two multiclass performance metrics against a known correct class.

    Pixels are excluded when:
      - p_correct == 0, OR
      - Any 'No Match' slice has p > 0.

    Metric 1 — Mean log-loss
        mean(-log(p_correct + eps))  Lower is better; 0 = perfect certainty.

    Metric 2 — Mean Pearson r
        Per-pixel Pearson correlation between the one-hot ground-truth vector
        (1 at correct_class, 0 elsewhere) and the predicted probability vector.
        No Match slices are excluded from both vectors before computing.
        Range [-1, 1]; 1 = perfect alignment.

    Returns
    -------
    mean_log_loss : float or None
    mean_pearson  : float or None
    mean_p        : float or None   Mean p_correct over valid pixels
    n_evaluated   : int
    labels        : list of str     All class labels in the TIFF
    no_match_idxs : list of int     Detected No Match slice indices
    """
    data, labels = load_probability_stack(tif_path, class_map=class_map)
    class_idx = find_class_index(labels, correct_class)
    n_classes = data.shape[0]

    no_match_idxs = [
        i for i, lbl in enumerate(labels)
        if lbl.lower().replace(" ", "").replace("-", "") == "nomatch"
    ]

    p_correct = data[class_idx].astype(np.float64)  # (H, W)

    # Valid pixel mask
    valid_mask = p_correct > 0
    for nm_idx in no_match_idxs:
        valid_mask &= data[nm_idx].astype(np.float64) == 0

    n_evaluated = int(np.sum(valid_mask))
    if n_evaluated == 0:
        return None, None, None, 0, labels, no_match_idxs

    p_valid = p_correct[valid_mask]  # (N,)

    # --- Metric 1: Mean log-loss ---
    eps = 1e-12
    mean_log_loss = float(np.mean(-np.log(p_valid + eps)))

    # --- Metric 2: Mean Pearson cross-correlation ---
    # Build probability matrix excluding No Match slices: shape (N, C_valid)
    valid_class_indices = [i for i in range(n_classes) if i not in set(no_match_idxs)]
    prob_matrix = np.stack(
        [data[i][valid_mask] for i in valid_class_indices], axis=1
    ).astype(np.float64)  # (N, C_valid)

    # One-hot ground truth vector for the correct class
    gt_pos = valid_class_indices.index(class_idx)
    one_hot = np.zeros(len(valid_class_indices), dtype=np.float64)
    one_hot[gt_pos] = 1.0

    # Vectorised Pearson: r = (p_centered @ oh_centered) / (||p_centered|| * ||oh_centered||)
    oh_centered = one_hot - one_hot.mean()          # (C_valid,)
    oh_norm = np.sqrt(np.dot(oh_centered, oh_centered))  # scalar

    p_means = prob_matrix.mean(axis=1, keepdims=True)   # (N, 1)
    p_centered = prob_matrix - p_means                  # (N, C_valid)
    p_norms = np.sqrt(np.einsum("nc,nc->n", p_centered, p_centered))  # (N,)

    denom = p_norms * oh_norm  # (N,)
    numerator = p_centered @ oh_centered  # (N,)

    with np.errstate(invalid="ignore"):
        pearson_r = np.where(denom > 1e-12, numerator / denom, np.nan)  # (N,)

    valid_r = pearson_r[~np.isnan(pearson_r)]
    mean_pearson = float(np.mean(valid_r)) if valid_r.size > 0 else None

    return mean_log_loss, mean_pearson, float(np.mean(p_valid)), n_evaluated, labels, no_match_idxs


# ---------------------------------------------------------------------------
# Per-folder processing
# ---------------------------------------------------------------------------

def process_folder(subdir: str, correct_class: str, class_map: dict | None = None) -> dict | None:
    """
    Process one subfolder. Prints results and returns a summary dict.
    """
    folder_name = os.path.basename(subdir)

    csv_files = sorted(f for f in os.listdir(subdir) if f.endswith("_predictions.csv"))
    tif_files = sorted(f for f in os.listdir(subdir) if f.endswith("_probabilities.tif"))

    if not csv_files:
        print(f"\n  [{folder_name}] No *_predictions.csv found — skipping.")
        return None
    if not tif_files:
        print(f"\n  [{folder_name}] No *_probabilities.tif found — skipping.")
        return None

    csv_path = os.path.join(subdir, csv_files[0])
    tif_path = os.path.join(subdir, tif_files[0])

    # --- 1. Prediction counts ---
    counts = count_predictions_from_csv(csv_path)
    total_pixels = sum(counts.values())

    print(f"\n[{folder_name}]")
    print(f"  File: {csv_files[0]}")
    print(f"  Prediction counts  (total pixels: {total_pixels:,})")
    for cls, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * cnt / total_pixels
        print(f"    {cls:<30s} {cnt:>8,}  ({pct:5.1f}%)")

    # --- 2. Classification metrics ---
    print(f"\n  Classification metrics for correct class '{correct_class}'")
    print(f"  File: {tif_files[0]}")
    try:
        mean_log_loss, mean_pearson, mean_p, n_evaluated, labels, no_match_idxs = compute_metrics(
            tif_path, correct_class, class_map=class_map
        )

        if no_match_idxs:
            print(f"    'No Match' slice(s) at index(es) {no_match_idxs} — those pixels excluded.")

        if mean_log_loss is None:
            print(f"    No valid pixels found for '{correct_class}' (all zero or No Match).")
            row = {
                "folder": folder_name,
                "total_pixels": total_pixels,
                "mean_log_loss": None,
                "mean_pearson_r": None,
                "mean_p": None,
                "n_pixels_evaluated": 0,
            }
        else:
            pct_evaluated = 100.0 * n_evaluated / total_pixels if total_pixels else 0
            print(f"    Mean p (correct class)  : {mean_p:.6f}")
            print(f"    Mean log-loss           : {mean_log_loss:.6f}  (lower = more certain)")
            pearson_str = f"{mean_pearson:.6f}" if mean_pearson is not None else "N/A"
            print(f"    Mean Pearson r          : {pearson_str}  (range [-1, 1]; higher = better)")
            print(f"    Pixels evaluated        : {n_evaluated:,}  ({pct_evaluated:.1f}% of total)")
            row = {
                "folder": folder_name,
                "total_pixels": total_pixels,
                "mean_log_loss": mean_log_loss,
                "mean_pearson_r": mean_pearson,
                "mean_p": mean_p,
                "n_pixels_evaluated": n_evaluated,
            }
    except ValueError as exc:
        print(f"    WARNING: {exc}")
        row = {
            "folder": folder_name,
            "total_pixels": total_pixels,
            "mean_log_loss": None,
            "mean_pearson_r": None,
            "mean_p": None,
            "n_pixels_evaluated": 0,
        }

    row.update({f"count_{k}": v for k, v in counts.items()})
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze classifier prediction CSVs and probability TIFF stacks.\n"
            "Reports per-class pixel counts, mean log-loss, and mean Pearson r "
            "for a specified correct class."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing per-image subdirectories with classifier outputs.",
    )
    parser.add_argument(
        "correct_class",
        type=str,
        help="Ground-truth class name to evaluate confidence against.",
    )
    parser.add_argument(
        "--class_map",
        type=str,
        default=None,
        help=(
            "Optional: mapping of slice index to class name, overriding TIFF metadata labels. "
            "Accepts either a JSON string (e.g. '{\"0\": \"PC\", \"1\": \"PE\", \"2\": \"No Match\"}') "
            "or a path to a JSON file with the same structure."
        ),
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        default=True,
        help="Save a summary CSV to <input_dir>/prediction_analysis_summary.csv (default: True).",
    )
    args = parser.parse_args()

    input_dir = os.path.normpath(args.input_dir)
    correct_class = args.correct_class

    # Parse class_map from JSON string or file
    class_map = None
    if args.class_map is not None:
        if os.path.isfile(args.class_map):
            with open(args.class_map, "r") as f:
                raw = json.load(f)
        else:
            raw = json.loads(args.class_map)
        # Normalise keys to int so dict lookup works with integer indices
        class_map = {int(k): str(v) for k, v in raw.items()}

    if not os.path.isdir(input_dir):
        print(f"ERROR: input_dir does not exist or is not a directory: {input_dir}")
        return

    subdirs = sorted(
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    )

    if not subdirs:
        print(f"No subdirectories found in: {input_dir}")
        return

    print(f"Input directory : {input_dir}")
    print(f"Subdirectories  : {len(subdirs)}")
    print(f"Correct class   : '{correct_class}'")
    if class_map is not None:
        print(f"Class map       : {class_map}")
    print("=" * 80)

    summary_rows = []
    for subdir in subdirs:
        row = process_folder(subdir, correct_class, class_map=class_map)
        if row is not None:
            summary_rows.append(row)

    # --- Summary table ---
    if summary_rows:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        summary_df = pd.DataFrame(summary_rows).set_index("folder")

        # Print compact view: score columns + count columns
        score_cols = ["total_pixels", "mean_p", "mean_log_loss", "mean_pearson_r", "n_pixels_evaluated"]
        count_cols = sorted(c for c in summary_df.columns if c.startswith("count_"))

        print("\n--- Classification metrics ---")
        print(summary_df[score_cols].to_string())

        print("\n--- Class counts per folder ---")
        print(summary_df[count_cols].fillna(0).astype(int).to_string())

        if args.save_summary:
            summary_path = os.path.join(input_dir, "prediction_analysis_summary.csv")
            summary_df.to_csv(summary_path)
            print(f"\nSummary saved to: {summary_path}")
    else:
        print("\nNo valid folders were processed.")


if __name__ == "__main__":
    main()
