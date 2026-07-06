#!/opt/miniconda3/envs/hsi_machine_learning/bin/python3
"""
plot_spectra_comparison.py
──────────────────────────
Loads a molecule spectra CSV and plots a cropped, area-normalized
comparison. Refactored to use in-script configuration and feature 
additions like cosine similarity and peak annotations.

CSV format expected:
-------------------
  - Two columns per molecule: [wavenumber, intensity]
  - Molecule names in headers of wavenumber columns
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.sparse import diags, csc_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.special import expit

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these values before running
# ══════════════════════════════════════════════════════════════════════════════

# Path relative to the repository root
CSV_PATH = "/Volumes/ShiLab_SSD1/Kidney_spectra.csv"

# Selection (partial/case-insensitive strings)
MOLECULES = [
    "15_1",
    "5_2",
    "tag 16",
    "dag 16",
    "tag mix",
    "cer 18:1-18:0",
]

# Analysis parameters
WN_MIN = 1200.0
WN_MAX = 1800.0
BASELINE_METHOD = "als"   # "als" | "drpls" | None

COMPUTE_SIMILARITY = True   # Shows pairwise cosine similarity matrix
SAMPLE_NAME = MOLECULES[0] # "" if no specific sample
# SAVE_PATH = f"/Users/jorgevillazon/Documents/GitHub/hsi_machine_learning/task_specific/plots/{SAMPLE_NAME}_{WN_MIN}_{WN_MAX}.png"            # e.g., "plots/comparison.png" (None to just show)
SAVE_PATH = None

# Selection / filtering behavior
USE_ALL_SPECTRA = False
SIMILARITY_METRIC = "weighted_rmse"  # "cosine" | "peak_f1" | "weighted_rmse"
FILTER_TO_HIGH_SIMILARITY_OUTLIERS = True
SIMILARITY_ZSCORE_THRESHOLD = 1  # Keep spectra above mean + (this * std) similarity score
# If set, compute similarity against this spectrum (partial/case-insensitive match).
# If empty or no match, per-spectrum mean pairwise similarity is used instead.
SIMILARITY_REFERENCE_QUERY = SAMPLE_NAME

# Weighted-RMSE parameters (used when SIMILARITY_METRIC == "weighted_rmse")
RMSE_PEAK_WEIGHT_EXPONENT = 3.0
RMSE_WEIGHT_FLOOR = 0.05

# Peak similarity parameters (used when SIMILARITY_METRIC == "peak_f1")
PEAK_MIN_HEIGHT = 0.05
PEAK_MIN_PROMINENCE = 0.2
PEAK_MIN_DISTANCE_CM = 10.0
PEAK_MATCH_TOLERANCE_CM = 10.0

# Peak / band annotations: { wavenumber : "label" }
PEAK_ANNOTATIONS: dict[float, str] = {
    1265.0: "Amide III",
    1300.0: "CH₂ twist",
    1440.0: "CH₂ / CH₃ def.",
    1460.0: "CH₂ scissor",

    2850.0: "CH₂ sym. stretch",
    2880.0: "CH₂ asym. stretch",
    2930.0: "CH₃ sym. stretch",
    2960.0: "CH₃ asym. stretch",
    3015.0: "=CH stretch",
}

# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

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
        wn_col = pd.to_numeric(df.iloc[:, 2 * i],     errors="coerce")
        int_col = pd.to_numeric(df.iloc[:, 2 * i + 1], errors="coerce")
        mask = wn_col.notna() & int_col.notna()
        wn  = wn_col[mask].to_numpy(dtype=float)
        intensity = int_col[mask].to_numpy(dtype=float)

        if len(wn) == 0: continue
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
    return list(dict.fromkeys(matched)) # Deduplicate preserving order

# ─────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────

def _als_baseline(intensity: np.ndarray, lam: float = 1e5, p: float = 0.001, n_iter: int = 10) -> np.ndarray:
    n = len(intensity)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n)).toarray()
    D = csc_matrix(D)
    DTD = D.T @ D
    w = np.ones(n)
    baseline = intensity.copy()
    for _ in range(n_iter):
        W = diags(w, 0, format="csc")
        baseline = spsolve(W + lam * DTD, w * intensity)
        w = np.where(intensity > baseline, p, 1 - p)
    return baseline

def _drpls_baseline(intensity: np.ndarray, lam: float = 1e5, eta: float = 0.5, n_iter: int = 50) -> np.ndarray:
    n = len(intensity)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n)).toarray()
    D = csc_matrix(D)
    DTD = D.T @ D
    I_n = eye(n, format="csc")
    eps = 1e-6 * lam
    w = np.ones(n)
    baseline = intensity.copy()
    for _ in range(n_iter):
        W = diags(w, 0, format="csc")
        Z = W + lam * (I_n - eta * W) @ DTD + eps * I_n
        baseline = spsolve(Z, w * intensity)
        residual = intensity - baseline
        d_neg = residual[residual < 0]
        sigma = np.std(d_neg) if len(d_neg) > 1 else 1.0
        w = 0.5 * expit(-residual / sigma)
    return baseline

def remove_baseline(intensity: np.ndarray, method: str = "drpls") -> np.ndarray:
    if method == "als":
        return intensity - _als_baseline(intensity)
    elif method == "drpls":
        return intensity - _drpls_baseline(intensity)
    return intensity

def crop_and_normalize(wn: np.ndarray, intensity: np.ndarray, wn_min: float, wn_max: float, baseline_method: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    mask = (wn >= wn_min) & (wn <= wn_max)
    if not any(mask):
        raise ValueError(f"No points in range [{wn_min}, {wn_max}]")
    w, i = wn[mask], intensity[mask]
    if baseline_method:
        i = remove_baseline(i, method=baseline_method)
    rng = i.max() - i.min()
    if rng == 0: return w, i * 0.0
    return w, (i - i.min()) / rng

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def detect_peaks(wn: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect relevant peaks on an interpolated spectrum."""
    if len(wn) < 3:
        return np.array([], dtype=float), np.array([], dtype=float)

    spacing = float(np.median(np.diff(wn))) if len(wn) > 1 else 1.0
    min_distance_pts = max(1, int(round(PEAK_MIN_DISTANCE_CM / max(spacing, 1e-8))))

    peak_idx, _ = find_peaks(
        intensity,
        height=PEAK_MIN_HEIGHT,
        prominence=PEAK_MIN_PROMINENCE,
        distance=min_distance_pts,
    )
    return wn[peak_idx], intensity[peak_idx]

def peak_f1_similarity(wn1: np.ndarray, v1: np.ndarray, wn2: np.ndarray, v2: np.ndarray) -> float:
    """Peak-overlap F1 score with penalties for missing/extra non-overlapping peaks."""
    p1_w, p1_h = detect_peaks(wn1, v1)
    p2_w, p2_h = detect_peaks(wn2, v2)

    n1 = len(p1_w)
    n2 = len(p2_w)
    if n1 == 0 and n2 == 0:
        return 1.0
    if n1 == 0 or n2 == 0:
        return 0.0

    used2: set[int] = set()
    match_quality_sum = 0.0

    for i in range(n1):
        deltas = np.abs(p2_w - p1_w[i])
        candidate_idx = [j for j in np.where(deltas <= PEAK_MATCH_TOLERANCE_CM)[0] if j not in used2]
        if not candidate_idx:
            continue

        best_j = min(candidate_idx, key=lambda j: deltas[j])
        used2.add(best_j)

        # Position mismatch is penalized strongest, with smaller amplitude penalty.
        pos_score = 1.0 - float(deltas[best_j] / PEAK_MATCH_TOLERANCE_CM)
        amp_score = float(min(p1_h[i], p2_h[best_j]) / max(p1_h[i], p2_h[best_j], 1e-8))
        match_quality_sum += 0.8 * pos_score + 0.2 * amp_score

    precision = match_quality_sum / n2
    recall = match_quality_sum / n1
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))

def weighted_rmse_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similarity from intensity-weighted RMSE, emphasizing mismatches at strong peaks."""
    if len(v1) == 0 or len(v2) == 0:
        return 0.0

    # Weight errors more where either spectrum has strong intensity.
    weights = np.maximum(np.maximum(v1, v2), RMSE_WEIGHT_FLOOR) ** RMSE_PEAK_WEIGHT_EXPONENT
    weighted_mse = float(np.sum(weights * (v1 - v2) ** 2) / np.sum(weights))
    weighted_rmse = np.sqrt(max(weighted_mse, 0.0))

    # Spectra are min-max normalized in [0,1], so RMSE is bounded in [0,1].
    return float(np.clip(1.0 - weighted_rmse, 0.0, 1.0))

def compute_similarity_scores(
    processed: dict[str, tuple[np.ndarray, np.ndarray]],
    wn_min: float,
    wn_max: float,
    metric: str,
    reference_query: str = "",
) -> dict[str, float]:
    """Compute one similarity score per spectrum on a shared cropped grid."""
    names = list(processed.keys())
    if not names:
        return {}

    common_wn = np.linspace(wn_min, wn_max, 1000)
    vectors: dict[str, np.ndarray] = {}
    for name, (w, i) in processed.items():
        f = interp1d(w, i, bounds_error=False, fill_value=0.0)
        vectors[name] = f(common_wn)

    if metric == "cosine":
        def sim_fn(a: np.ndarray, b: np.ndarray) -> float:
            return cosine_similarity(a, b)
    elif metric == "weighted_rmse":
        def sim_fn(a: np.ndarray, b: np.ndarray) -> float:
            return weighted_rmse_similarity(a, b)
    else:
        def sim_fn(a: np.ndarray, b: np.ndarray) -> float:
            return peak_f1_similarity(common_wn, a, common_wn, b)

    if len(names) == 1:
        return {names[0]: 1.0}

    ref_name = None
    if reference_query:
        q = reference_query.strip().lower()
        hits = [n for n in names if q in n.lower()]
        if hits:
            ref_name = hits[0]

    scores: dict[str, float] = {}
    if ref_name is not None:
        ref_vec = vectors[ref_name]
        for name in names:
            scores[name] = sim_fn(ref_vec, vectors[name])
        print(f"Using reference spectrum for {metric} scores: '{ref_name}'")
        return scores

    # Fallback: each spectrum gets mean pairwise similarity vs all others.
    for name_i in names:
        sims = [sim_fn(vectors[name_i], vectors[name_j]) for name_j in names if name_j != name_i]
        scores[name_i] = float(np.mean(sims)) if sims else 1.0

    print(f"Using mean pairwise {metric} score per spectrum (no reference match found).")
    return scores

# ─────────────────────────────────────────────────────────────
# Annotation and Plotting
# ─────────────────────────────────────────────────────────────

def annotate_peaks(ax, wn_min, wn_max):
    """Draw staggered peak annotations."""
    if not PEAK_ANNOTATIONS: return
    
    # Filter annotations in range
    valid = {k: v for k, v in PEAK_ANNOTATIONS.items() if wn_min <= k <= wn_max}
    if not valid: return

    # Stagger heights to avoid label collision (3 levels)
    heights = np.linspace(1.02, 1.15, 3)
    
    sorted_keys = sorted(valid.keys())
    for idx, wn in enumerate(sorted_keys):
        label = valid[wn]
        h = heights[idx % len(heights)]
        
        # Vertical line
        ax.axvline(wn, color="black", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
        
        # Label at top
        ax.text(wn, h, label, rotation=90, va="bottom", ha="center", 
                fontsize=11,fontweight="bold", color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                          edgecolor="gray", alpha=0.9, linewidth=0.5),
                zorder=5, clip_on=True)

def plot_comparison(selected_raw: dict[str, tuple[np.ndarray, np.ndarray]]):
    names = list(selected_raw.keys())
    colors = [cm.tab10(i % 10) for i in range(len(names))]
    
    # Process all selected
    processed = {}
    for name in names:
        try:
            w, i = selected_raw[name]
            wn_c, int_n = crop_and_normalize(w, i, WN_MIN, WN_MAX, BASELINE_METHOD)
            processed[name] = (wn_c, int_n)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            
    if not processed:
        print("No spectra to plot.")
        return

    if FILTER_TO_HIGH_SIMILARITY_OUTLIERS:
        scores = compute_similarity_scores(
            processed,
            wn_min=WN_MIN,
            wn_max=WN_MAX,
            metric=SIMILARITY_METRIC,
            reference_query=SIMILARITY_REFERENCE_QUERY,
        )
        values = np.array(list(scores.values()), dtype=float)
        mean_score = float(values.mean()) if values.size else 0.0
        std_score = float(values.std()) if values.size else 0.0
        raw_threshold = mean_score + SIMILARITY_ZSCORE_THRESHOLD * std_score
        threshold = min(1.0, raw_threshold)

        print(f"\n{SIMILARITY_METRIC} Similarity Scores (mean: {mean_score:.4f}, std: {std_score:.4f}):")
        for name, score in scores.items():
            print(f"  {name}: {score:.4f}")

        keep_names = [n for n in processed if scores.get(n, 0.0) >= threshold]

        designated_matches = match_molecule_names(MOLECULES, list(processed.keys()))
        has_extra_designations = any(
            q.strip().lower() != SAMPLE_NAME.strip().lower() for q in MOLECULES
        )
        forced_names = [n for n in designated_matches if n != SAMPLE_NAME] if has_extra_designations else []
        final_names = list(dict.fromkeys(keep_names + forced_names))

        print(
            f"Similarity filter ({SIMILARITY_METRIC}): mean={mean_score:.4f}, std={std_score:.4f}, "
            f"threshold(mean + {SIMILARITY_ZSCORE_THRESHOLD:.1f}*std)={threshold:.4f}"
        )
        if raw_threshold > 1.0:
            print("Requested threshold exceeded score bounds [0, 1]; clamped to 1.0.")
        print(f"Kept {len(keep_names)}/{len(processed)} spectra above threshold.")
        if forced_names:
            print(f"Also forced from MOLECULES: {forced_names}")
        print(f"Final displayed spectra: {len(final_names)}")

        if final_names:
            processed = {name: processed[name] for name in final_names}
        else:
            print("No spectra matched the filter criteria; nothing to plot.")
            return

    # Setup layout
    if COMPUTE_SIMILARITY and len(processed) > 1:
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.5], hspace=0.4)
        ax_spectra = fig.add_subplot(gs[0])
        ax_sim = fig.add_subplot(gs[1])
    else:
        fig, ax_spectra = plt.subplots(figsize=(12, 6))
        ax_sim = None

    # 1. Main Spectra Plot
    for (name, (w, i)), color in zip(processed.items(), colors):
        ax_spectra.plot(w, i, label=name, color=color, linewidth=1.8, alpha=0.9)
        ax_spectra.fill_between(w, i, alpha=0.1, color=color)

    ax_spectra.set_title(f"Spectra Comparison ({WN_MIN} - {WN_MAX} cm⁻¹)", fontsize=14, weight="bold")
    ax_spectra.set_xlabel("Wavenumber (cm⁻¹)")
    ax_spectra.set_ylabel("Normalized Intensity")
    ax_spectra.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    ax_spectra.grid(True, linestyle=":", alpha=0.6)
    

    # Annotations
    annotate_peaks(ax_spectra, WN_MIN, WN_MAX)
    ax_spectra.set_ylim(-0.05, 2) # Space for annotations
    
    # 2. Similarity Matrix
    if ax_sim:
        # Interpolate for calculation
        common_wn = np.linspace(WN_MIN, WN_MAX, 1000)
        vectors = []
        for name in processed:
            w, i = processed[name]
            f = interp1d(w, i, bounds_error=False, fill_value=0.0)
            vectors.append(f(common_wn))
            
        n = len(vectors)
        sim_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if SIMILARITY_METRIC == "cosine":
                    sim_mat[i, j] = cosine_similarity(vectors[i], vectors[j])
                elif SIMILARITY_METRIC == "weighted_rmse":
                    sim_mat[i, j] = weighted_rmse_similarity(vectors[i], vectors[j])
                else:
                    sim_mat[i, j] = peak_f1_similarity(common_wn, vectors[i], common_wn, vectors[j])
        
        im = ax_sim.imshow(sim_mat, cmap="viridis", vmin=0, vmax=1)
        ax_sim.set_xticks(range(n))
        ax_sim.set_yticks(range(n))
        ax_sim.set_xticklabels([f"{n[:15]}..." if len(n)>15 else n for n in processed.keys()], rotation=45, ha="right", fontsize=8)
        ax_sim.set_yticklabels([f"{n[:15]}..." if len(n)>15 else n for n in processed.keys()], fontsize=8)
        ax_sim.set_title(f"Pairwise Similarity ({SIMILARITY_METRIC})", fontsize=11, pad=10)
        plt.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)
        
        # Add values inside cells
        for k in range(n):
            for m in range(n):
                ax_sim.text(m, k, f"{sim_mat[k,m]:.2f}", ha="center", va="center", color="white" if sim_mat[k,m] < 0.7 else "black", fontsize=8)

    plt.tight_layout()
    if SAVE_PATH:
        os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
        plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved: {SAVE_PATH}")
    else:
        plt.show()

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Spectra Comparison & Similarity Tool")
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
    
    plot_comparison(selected_raw)

if __name__ == "__main__":
    main()
