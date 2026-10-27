"""
integrated_pipeline
===================
Consolidated toolkit for:
1. Converting segmentation masks (binary/multi-slice TIFF) and Xenium CSV metadata into GUI-compatible Vertices_JSON ROI Parquet tables.
2. High-speed multi-modal pixel/geometry quantification (MALDI MSI, SRS/Raman ratio maps, Collagen).
3. Non-parametric group statistics (Kruskal-Wallis, Dunn's + BH FDR, Cliff's Delta, Van Elteren stratified Wilcoxon).
4. Publication-grade visualization (Violin/bar plots, sample-level strip plots, automated significance brackets).
"""

from .roi_converter import mask_to_gui_rois, cells_to_gui_rois
from .group_stats_pipeline import run_group_stats_pipeline, load_table, load_json_config
from .calculate_group_statistics import run_group_statistics_pipeline
