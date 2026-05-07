#!/usr/bin/env python3
"""
M05_SegmentationStats_threeway.py

Summarize segmentation-quality metrics and three-way voxel
comparisons (original CT, manual combined, automated combined).

Reads:
    BASE_DEV/all_specimens_segmentation_summary.csv  (from M03_Metrics)

Outputs (in BASE_DEV/StatsComparison):
    segm_threeway_summary_metrics.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DEV = Path(r"C:\\Users\\jeichwal\\Documents\\A_DocTech\\P1_Segmentation\\Data")
SUMMARY_CSV = BASE_DEV / "all_specimens_segmentation_summary.csv"
OUT_DIR = BASE_DEV / "StatsComparison"


def median_iqr(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan, np.nan
    med = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return med, (q3 - q1)


def summarize_metric(df, col, metric_label, compartment, comparison_type):
    if col not in df.columns:
        print(f"[WARN] Missing segmentation metric column: {col}")
        return None

    s = df[col]
    mean = s.mean()
    sd = s.std(ddof=1)
    med, iqr = median_iqr(s)

    return {
        "metric": metric_label,
        "compartment": compartment,
        "comparison_type": comparison_type,
        "column": col,
        "n": int(s.notna().sum()),
        "mean": mean,
        "sd": sd,
        "median": med,
        "IQR": iqr,
    }


def main():
    if not SUMMARY_CSV.is_file():
        print(f"[ERROR] Missing summary CSV: {SUMMARY_CSV}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Reading {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)

    rows = []

    # ------------------------------------------------------------------
    # 1) Manual vs semi-auto segmentation metrics (per compartment)
    # ------------------------------------------------------------------
    metric_sets = [
        # col_name, metric_label, compartment
        ("DSC_cortical", "Dice", "cortical"),
        ("DSC_trabecular", "Dice", "trabecular"),
        ("Jaccard_cortical", "Jaccard", "cortical"),
        ("Jaccard_trabecular", "Jaccard", "trabecular"),
        ("Sensitivity_cortical", "Sensitivity", "cortical"),
        ("Sensitivity_trabecular", "Sensitivity", "trabecular"),
        ("Specificity_cortical", "Specificity", "cortical"),
        ("Specificity_trabecular", "Specificity", "trabecular"),
        ("Precision_cortical", "Precision", "cortical"),
        ("Precision_trabecular", "Precision", "trabecular"),
        ("F1_cortical", "F1", "cortical"),
        ("F1_trabecular", "F1", "trabecular"),
        ("VolumeSimilarity_cortical", "VolumeSimilarity", "cortical"),
        ("VolumeSimilarity_trabecular", "VolumeSimilarity", "trabecular"),
        ("FPE_cortical", "FPE", "cortical"),
        ("FPE_trabecular", "FPE", "trabecular"),
        ("FNE_cortical", "FNE", "cortical"),
        ("FNE_trabecular", "FNE", "trabecular"),
        ("CohenKappa_cortical", "CohenKappa", "cortical"),
        ("CohenKappa_trabecular", "CohenKappa", "trabecular"),
        ("MSD_mm_cortical", "MSD_mm", "cortical"),
        ("MSD_mm_trabecular", "MSD_mm", "trabecular"),
        ("HD95_mm_cortical", "HD95_mm", "cortical"),
        ("HD95_mm_trabecular", "HD95_mm", "trabecular"),
        ("HD_max_mm_cortical", "HD_max_mm", "cortical"),
        ("HD_max_mm_trabecular", "HD_max_mm", "trabecular"),
    ]

    for col, metric_label, comp in metric_sets:
        row = summarize_metric(df, col, metric_label, comp, "manual_vs_auto")
        if row is not None:
            rows.append(row)

    # ------------------------------------------------------------------
    # 2) Three-way voxel-count comparison (trabecular, combined masks)
    #    Columns present in CSV:
    #      Original binary mask voxel count_trabecular
    #      Automated combined voxel count_trabecular
    #      Voxel count difference_trabecular   (auto - original)
    #      Manual combined voxel count_trabecular
    # ------------------------------------------------------------------
    orig_col = "Original binary mask voxel count_trabecular"
    auto_col = "Automated combined voxel count_trabecular"
    manual_col = "Manual combined voxel count_trabecular"

    if all(c in df.columns for c in [orig_col, auto_col, manual_col]):
        orig = df[orig_col].astype(float)
        auto = df[auto_col].astype(float)
        manual = df[manual_col].astype(float)

        # Auto vs original: absolute and percent difference
        auto_diff = auto - orig
        auto_diff_pct = auto_diff / orig.replace(0, np.nan) * 100.0

        # Manual vs original
        man_diff = manual - orig
        man_diff_pct = man_diff / orig.replace(0, np.nan) * 100.0

        # Auto vs manual
        auto_man_diff = auto - manual
        auto_man_diff_pct = auto_man_diff / manual.replace(0, np.nan) * 100.0

        df_vox = pd.DataFrame({
            "auto_minus_orig": auto_diff,
            "auto_minus_orig_pct": auto_diff_pct,
            "manual_minus_orig": man_diff,
            "manual_minus_orig_pct": man_diff_pct,
            "auto_minus_manual": auto_man_diff,
            "auto_minus_manual_pct": auto_man_diff_pct,
        })

        voxel_metrics = [
            ("auto_minus_orig_pct", "VoxelDiff_pct", "trabecular", "auto_vs_original"),
            ("manual_minus_orig_pct", "VoxelDiff_pct", "trabecular", "manual_vs_original"),
            ("auto_minus_manual_pct", "VoxelDiff_pct", "trabecular", "auto_vs_manual"),
        ]

        for col, metric_label, comp, comp_type in voxel_metrics:
            s = df_vox[col]
            mean = s.mean()
            sd = s.std(ddof=1)
            med, iqr = median_iqr(s)
            rows.append({
                "metric": metric_label,
                "compartment": comp,
                "comparison_type": comp_type,
                "column": col,
                "n": int(s.notna().sum()),
                "mean": mean,
                "sd": sd,
                "median": med,
                "IQR": iqr,
            })

    else:
        print("[WARN] Some voxel-count columns missing; skipping three-way voxel analysis.")

    if not rows:
        print("[WARN] No segmentation metrics summarized; check column names.")
        return

    out_df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "segm_threeway_summary_metrics.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[DONE] Three-way segmentation metrics summary saved to {out_csv}")


if __name__ == "__main__":
    main()