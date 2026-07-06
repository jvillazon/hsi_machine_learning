#!/opt/miniconda3/envs/hsi_machine_learning/bin/python3
"""
compare_peak_ratios.py
----------------------
Quantitative comparison of peak-height ratios across spectra from a 2-column-pairs CSV.

This script follows the same style and processing flow as plot_spectra_comparison.py:
- loads spectra from CSV
- optional molecule matching by partial/case-insensitive query
- optional baseline removal
- crop + min-max normalization in the selected region

Then, for each selected spectrum, it computes configured peak ratios:
ratio = peak_height(numerator_peak) / peak_height(denominator_peak)
where each peak height is the maximum intensity inside a local search window.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags, csc_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.special import expit

# ============================================================================
# CONFIG -- edit these values before running
# ============================================================================

# Path relative to the repository root (same style as plot_spectra_comparison.py)
CSV_PATH = "/Volumes/ShiLab_SSD1/Kidney_spectra.csv"

# Selection (partial/case-insensitive strings)
MOLECULES = [
    "1_2",
    "5_2",
    "9_1",
    "15_1",
    "tag 16",
    "dag 16",
    "tag mix",
    "cer 18:1-18:0",
]

# Selection behavior
USE_ALL_SPECTRA = False

# Processing parameters
WN_MIN = 1200.0
WN_MAX = 3100.0
BASELINE_METHOD = "als"  # "als" | "drpls" | None

# Peak ratio definitions:
# ratio_name: (numerator_peak_cm-1, denominator_peak_cm-1)
RATIO_DEFINITIONS: dict[str, tuple[float, float]] = {
    "1300cm-1/1440cm-1": (1300.0, 1440.0),
    "1460cm-1/1440cm-1": (1460.0, 1440.0),
    "2930cm-1/2850cm-1": (2930.0, 2850.0),
    "3015cm-1/2880cm-1": (3015.0, 2880.0),


}

# Local peak search around each target position
PEAK_SEARCH_WINDOW_CM = 10.0
MIN_DENOMINATOR = 1e-8

# Output
SAVE_TABLE_PATH = "task_specific/plots/peak_ratio_table.csv"  # None to skip
SAVE_PLOT_PATH = "task_specific/plots/peak_ratio_comparison.png"  # None to skip
SHOW_PLOT = True

# ============================================================================


def resolve_path(rel_path: str) -> str:
    """Resolve path relative to repo root."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(script_dir, ".."))
    return os.path.join(repo_root, rel_path)


def load_spectra_csv(csv_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, header=0, dtype=str)
    df = df.dropna(axis="columns", how="all")
    headers = list(df.columns)

    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_molecules = len(headers) // 2

    for i in range(n_molecules):
        name = headers[2 * i].strip()
        wn_col = pd.to_numeric(df.iloc[:, 2 * i], errors="coerce")
        int_col = pd.to_numeric(df.iloc[:, 2 * i + 1], errors="coerce")
        mask = wn_col.notna() & int_col.notna()
        wn = wn_col[mask].to_numpy(dtype=float)
        intensity = int_col[mask].to_numpy(dtype=float)

        if len(wn) == 0:
            continue
        order = np.argsort(wn)
        spectra[name] = (wn[order], intensity[order])

    return spectra


def match_molecule_names(query_list: list[str], available: list[str]) -> list[str]:
    matched: list[str] = []
    for query in query_list:
        q = query.strip().lower()
        hits = [name for name in available if q in name.lower()]
        if not hits:
            print(f"  [!] No match found for: '{query}'")
        elif len(hits) == 1:
            matched.append(hits[0])
        else:
            exact = [h for h in hits if h.lower() == q]
            if exact:
                matched.append(exact[0])
            else:
                print(f"  [?] Multiple matches for '{query}': {hits}. Using '{hits[0]}'")
                matched.append(hits[0])
    return list(dict.fromkeys(matched))


def _als_baseline(intensity: np.ndarray, lam: float = 1e5, p: float = 0.001, n_iter: int = 10) -> np.ndarray:
    n = len(intensity)
    d = diags([1, -2, 1], (0, 1, 2), shape=(n - 2, n)).toarray()
    d = csc_matrix(d)
    dtd = d.T @ d
    w = np.ones(n)
    baseline = intensity.copy()
    for _ in range(n_iter):
        w_mat = diags(w, 0, format="csc")
        baseline = spsolve(w_mat + lam * dtd, w * intensity)
        w = np.where(intensity > baseline, p, 1 - p)
    return np.asarray(baseline, dtype=float)


def _drpls_baseline(intensity: np.ndarray, lam: float = 1e5, eta: float = 0.5, n_iter: int = 50) -> np.ndarray:
    n = len(intensity)
    d = diags([1, -2, 1], (0, 1, 2), shape=(n - 2, n)).toarray()
    d = csc_matrix(d)
    dtd = d.T @ d
    i_n = eye(n, format="csc")
    eps = 1e-6 * lam
    w = np.ones(n)
    baseline = intensity.copy()
    for _ in range(n_iter):
        w_mat = diags(w, 0, format="csc")
        z = w_mat + lam * (i_n - eta * w_mat) @ dtd + eps * i_n
        baseline = spsolve(z, w * intensity)
        residual = intensity - baseline
        d_neg = residual[residual < 0]
        sigma = np.std(d_neg) if len(d_neg) > 1 else 1.0
        w = 0.5 * expit(-residual / sigma)
    return np.asarray(baseline, dtype=float)


def remove_baseline(intensity: np.ndarray, method: str = "drpls") -> np.ndarray:
    if method == "als":
        return intensity - _als_baseline(intensity)
    if method == "drpls":
        return intensity - _drpls_baseline(intensity)
    return intensity


def crop_and_normalize(
    wn: np.ndarray,
    intensity: np.ndarray,
    wn_min: float,
    wn_max: float,
    baseline_method: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (wn >= wn_min) & (wn <= wn_max)
    if not any(mask):
        raise ValueError(f"No points in range [{wn_min}, {wn_max}]")
    w, i = wn[mask], intensity[mask]
    if baseline_method:
        i = remove_baseline(i, method=baseline_method)
    rng = float(i.max() - i.min())
    if rng == 0.0:
        return w, i * 0.0
    return w, (i - i.min()) / rng


def local_peak_height(wn: np.ndarray, intensity: np.ndarray, target_wn: float, window_cm: float) -> float:
    """Return local maximum intensity in [target-window, target+window]."""
    mask = (wn >= target_wn - window_cm) & (wn <= target_wn + window_cm)
    if not any(mask):
        return np.nan
    return float(np.max(intensity[mask]))


def compute_ratio_table(processed: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for name, (wn, intensity) in processed.items():
        row: dict[str, float | str] = {"spectrum": name}

        for ratio_name, (num_peak, den_peak) in RATIO_DEFINITIONS.items():
            num_h = local_peak_height(wn, intensity, num_peak, PEAK_SEARCH_WINDOW_CM)
            den_h = local_peak_height(wn, intensity, den_peak, PEAK_SEARCH_WINDOW_CM)

            if np.isnan(num_h) or np.isnan(den_h):
                ratio = np.nan
            elif abs(den_h) < MIN_DENOMINATOR:
                ratio = np.nan
            else:
                ratio = float(num_h / den_h)

            row[ratio_name] = ratio
            row[f"{ratio_name}__num_height"] = num_h
            row[f"{ratio_name}__den_height"] = den_h

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.set_index("spectrum")
    return out


def plot_ratio_table(df: pd.DataFrame):
    ratio_cols = [c for c in df.columns if not c.endswith("__num_height") and not c.endswith("__den_height")]
    if not ratio_cols:
        print("No ratio columns to plot.")
        return

    plot_df = df[ratio_cols].copy()
    n_spectra = len(plot_df.index)
    n_ratios = len(ratio_cols)

    fig_w = max(10, 0.8 * n_spectra + 4)
    fig_h = max(5, 0.9 * n_ratios + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(n_spectra)
    width = 0.8 / max(n_ratios, 1)

    for idx, col in enumerate(ratio_cols):
        offset = (idx - (n_ratios - 1) / 2) * width
        ax.bar(x + offset, plot_df[col].values, width=width, label=col, alpha=0.9, edgecolor="black")

    ax.set_title(f"Peak Ratio Comparison ({WN_MIN} to {WN_MAX} cm-1)", fontweight="bold", fontsize=20)
    ax.set_ylabel("Ratio", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=45, ha="right", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)

    plt.tight_layout()


def main():
    print("=" * 60)
    print("  Quantitative Peak Ratio Comparison")
    print("=" * 60)

    resolved_csv = resolve_path(CSV_PATH)
    print(f"Loading: {resolved_csv}")

    all_spectra = load_spectra_csv(resolved_csv)
    available = list(all_spectra.keys())

    if USE_ALL_SPECTRA:
        selected_raw = all_spectra
        print(f"Selected all spectra: {len(selected_raw)}")
    else:
        matched = match_molecule_names(MOLECULES, available)
        if not matched:
            print("No molecules matched configuration. Exiting.")
            return
        print(f"Selected ({len(matched)}): {matched}")
        selected_raw = {name: all_spectra[name] for name in matched}

    processed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (wn, intensity) in selected_raw.items():
        try:
            wn_c, int_n = crop_and_normalize(wn, intensity, WN_MIN, WN_MAX, BASELINE_METHOD)
            processed[name] = (wn_c, int_n)
        except Exception as exc:
            print(f"Skipping {name}: {exc}")

    if not processed:
        print("No spectra were processable in selected range.")
        return

    ratio_df = compute_ratio_table(processed)
    if ratio_df.empty:
        print("No ratio data generated.")
        return

    ratio_cols = [c for c in ratio_df.columns if not c.endswith("__num_height") and not c.endswith("__den_height")]
    print("\nRatio values:")
    print(ratio_df[ratio_cols].to_string(float_format=lambda v: f"{v:.4f}"))

    if SAVE_TABLE_PATH:
        out_csv = resolve_path(SAVE_TABLE_PATH)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        ratio_df.to_csv(out_csv)
        print(f"\nSaved table: {out_csv}")

    if SHOW_PLOT or SAVE_PLOT_PATH:
        plot_ratio_table(ratio_df)

        if SAVE_PLOT_PATH:
            out_png = resolve_path(SAVE_PLOT_PATH)
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            plt.savefig(out_png, dpi=160, bbox_inches="tight")
            print(f"Saved plot: {out_png}")

        if SHOW_PLOT:
            plt.show()


if __name__ == "__main__":
    main()
