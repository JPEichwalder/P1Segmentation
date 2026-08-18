#!/usr/bin/env python3
"""
analysis_results.py

Post-processes your refactored M01–M05 outputs into:
- Clean segmentation summary table for the paper
- Morphometric agreement table for the paper
- Delta summary table
- Core figures (overlap metrics, distance metrics, scatter, Bland–Altman)

Inputs (expected after running run_m01..run_m05):
- BASE_DEV/all_specimens_segmentation_summary.csv
- BASE_DEV/StatsComparison/all_specimens_with_deltas.csv
- BASE_DEV/StatsComparison/morpho_agreement_summary_all_params.csv
- BASE_DEV/StatsComparison/segm_threeway_summary_metrics.csv

Outputs:
- BASE_DEV/StatsComparison/table_segmentation_metrics_for_paper.csv
- BASE_DEV/StatsComparison/table_morphometric_agreement_for_paper.csv
- BASE_DEV/StatsComparison/delta_summary.csv
- BASE_DEV/StatsComparison/fig_overlap_metrics.png
- BASE_DEV/StatsComparison/fig_distance_metrics.png
- BASE_DEV/StatsComparison/fig_manual_vs_automated_scatter.png
- BASE_DEV/StatsComparison/fig_bland_altman.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# 1. Paths (match your existing BASE_DEV)
# ---------------------------------------------------------------------
BASE_DEV = Path(r"C:\Users\jeichwal\Documents\A_DocTech\P1_Segmentation\Data")
STATS_DIR = BASE_DEV / "StatsComparison"
STATS_DIR.mkdir(parents=True, exist_ok=True)

SEG_SUMMARY_CSV = BASE_DEV / "all_specimens_segmentation_summary.csv"
DELTAS_CSV = STATS_DIR / "all_specimens_with_deltas.csv"
MORPHO_AGREE_CSV = STATS_DIR / "morpho_agreement_summary_all_params.csv"
SEGM_THREEWAY_CSV = STATS_DIR / "segm_threeway_summary_metrics.csv"

# ---------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------
def mean_sd(x):
    x = pd.Series(x).dropna()
    return x.mean(), x.std(ddof=1)

def median_iqr(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return np.nan, np.nan
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    return x.median(), (q3 - q1)

def bland_altman_stats(manual, auto):
    manual = pd.Series(manual).astype(float)
    auto = pd.Series(auto).astype(float)
    diff = auto - manual
    bias = diff.mean()
    sd_diff = diff.std(ddof=1)
    loa_lower = bias - 1.96 * sd_diff
    loa_upper = bias + 1.96 * sd_diff
    return {
        "bias": bias,
        "sd_diff": sd_diff,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
    }

def icc_2_1(data):
    """
    Two-way random effects, absolute agreement, single measurement ICC(2,1).
    data: n x 2 array (manual, auto)
    """
    arr = np.asarray(data, dtype=float)
    n, k = arr.shape
    if n < 2:
        return np.nan

    mean_rows = np.mean(arr, axis=1)
    mean_cols = np.mean(arr, axis=0)
    grand_mean = np.mean(arr)

    ss_rows = k * np.sum((mean_rows - grand_mean) ** 2)
    ss_cols = n * np.sum((mean_cols - grand_mean) ** 2)
    ss_total = np.sum((arr - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc = (ms_rows - ms_error) / (
        ms_rows + (k - 1) * ms_error + (k * (ms_cols - ms_error) / n)
    )
    return icc

def summarise_metric(df, value_col, label, compartment):
    x = df[value_col].dropna()
    mean, sd = mean_sd(x)
    median, iqr = median_iqr(x)
    return {
        "metric": label,
        "compartment": compartment,
        "n": int(x.shape[0]),
        "mean": mean,
        "sd": sd,
        "median": median,
        "IQR": iqr,
    }

# ---------------------------------------------------------------------
# 3. Load data
# ---------------------------------------------------------------------
if not SEG_SUMMARY_CSV.is_file():
    raise FileNotFoundError(f"Missing {SEG_SUMMARY_CSV}")
if not DELTAS_CSV.is_file():
    raise FileNotFoundError(f"Missing {DELTAS_CSV}")
if not MORPHO_AGREE_CSV.is_file():
    raise FileNotFoundError(f"Missing {MORPHO_AGREE_CSV}")
if not SEGM_THREEWAY_CSV.is_file():
    print(f"[WARN] {SEGM_THREEWAY_CSV} not found; continuing without three-way stats.")

segm_all = pd.read_csv(SEG_SUMMARY_CSV)
deltas = pd.read_csv(DELTAS_CSV)
morpho_agree = pd.read_csv(MORPHO_AGREE_CSV)

# ---------------------------------------------------------------------
# 4. Segmentation summary table (from segm_all)
# ---------------------------------------------------------------------
segm_rows = []
segm_rows.append(summarise_metric(segm_all, "DSC_cortical", "Dice", "cortical"))
segm_rows.append(summarise_metric(segm_all, "DSC_trabecular", "Dice", "trabecular"))
segm_rows.append(summarise_metric(segm_all, "Jaccard_cortical", "Jaccard", "cortical"))
segm_rows.append(summarise_metric(segm_all, "Jaccard_trabecular", "Jaccard", "trabecular"))
segm_rows.append(summarise_metric(segm_all, "HD95_mm_cortical", "HD95_mm", "cortical"))
segm_rows.append(summarise_metric(segm_all, "HD95_mm_trabecular", "HD95_mm", "trabecular"))
segm_rows.append(summarise_metric(segm_all, "MSD_mm_cortical", "MSD_mm", "cortical"))
segm_rows.append(summarise_metric(segm_all, "MSD_mm_trabecular", "MSD_mm", "trabecular"))

segm_table = pd.DataFrame(segm_rows)
for col in ["mean", "sd", "median", "IQR"]:
    segm_table[col] = segm_table[col].round(3)

segm_out = STATS_DIR / "table_segmentation_metrics_for_paper.csv"
segm_table.to_csv(segm_out, index=False)
print(f"[DONE] Segmentation metrics table → {segm_out}")

# ---------------------------------------------------------------------
# 5. Morphometric agreement table (recomputed from deltas)
# ---------------------------------------------------------------------
parameter_pairs = {
    "BVTV_trab": ("BVTV_trab_manual", "BVTV_trab_auto"),
    "TbTh_trab": ("TbTh_trab_manual", "TbTh_trab_auto"),
    "TbSp_trab": ("TbSp_trab_manual", "TbSp_trab_auto"),
    "TbN_trab": ("TbN_trab_manual", "TbN_trab_auto"),
    "CtTh_cort": ("CtTh_cort_manual", "CtTh_cort_auto"),
    "CtAr_cort": ("CtAr_cort_manual", "CtAr_cort_auto"),
    "CtAr_TtAr": ("CtAr_TtAr_manual", "CtAr_TtAr_auto"),
    "TtAr_total": ("TtAr_total_manual", "TtAr_total_auto"),
}

agreement_rows = []
for param, (man_col, auto_col) in parameter_pairs.items():
    if man_col not in deltas.columns or auto_col not in deltas.columns:
        print(f"[WARN] Missing columns for {param}; skipping.")
        continue
    tmp = deltas[[man_col, auto_col]].dropna().copy()
    tmp.columns = ["manual", "auto"]
    if tmp.shape[0] < 2:
        print(f"[WARN] Too few specimens for {param}; skipping.")
        continue

    ba = bland_altman_stats(tmp["manual"], tmp["auto"])
    diff = tmp["auto"] - tmp["manual"]

    row = {
        "parameter": param,
        "n": int(tmp.shape[0]),
        "pearson_r": float(tmp["manual"].corr(tmp["auto"], method="pearson")),
        "ICC2_1": float(icc_2_1(tmp[["manual", "auto"]].values)),
        "BA_bias": ba["bias"],
        "BA_sd_diff": ba["sd_diff"],
        "BA_LOA_lower": ba["loa_lower"],
        "BA_LOA_upper": ba["loa_upper"],
        "manual_mean": tmp["manual"].mean(),
        "manual_sd": tmp["manual"].std(ddof=1),
        "manual_median": tmp["manual"].median(),
        "manual_IQR": tmp["manual"].quantile(0.75) - tmp["manual"].quantile(0.25),
        "auto_mean": tmp["auto"].mean(),
        "auto_sd": tmp["auto"].std(ddof=1),
        "auto_median": tmp["auto"].median(),
        "auto_IQR": tmp["auto"].quantile(0.75) - tmp["auto"].quantile(0.25),
        "diff_mean": diff.mean(),
        "diff_sd": diff.std(ddof=1),
        "diff_median": diff.median(),
        "diff_IQR": diff.quantile(0.75) - diff.quantile(0.25),
    }
    agreement_rows.append(row)

agreement_df = pd.DataFrame(agreement_rows)
for col in agreement_df.columns:
    if col not in ["parameter", "n"]:
        agreement_df[col] = agreement_df[col].round(3)

agree_out = STATS_DIR / "table_morphometric_agreement_for_paper.csv"
agreement_df.to_csv(agree_out, index=False)
print(f"[DONE] Morphometric agreement table → {agree_out}")

# ---------------------------------------------------------------------
# 6. Delta summary table (from all_specimens_with_deltas)
# ---------------------------------------------------------------------
delta_cols = [c for c in deltas.columns if c.startswith("d_")]
delta_summary_rows = []
for col in delta_cols:
    x = deltas[col].dropna()
    med, iqr = median_iqr(x)
    delta_summary_rows.append({
        "delta_variable": col,
        "n": int(x.shape[0]),
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "median": med,
        "IQR": iqr,
        "min": x.min(),
        "max": x.max(),
    })

delta_summary_df = pd.DataFrame(delta_summary_rows)
for col in ["mean", "sd", "median", "IQR", "min", "max"]:
    delta_summary_df[col] = delta_summary_df[col].round(4)

delta_out = STATS_DIR / "delta_summary.csv"
delta_summary_df.to_csv(delta_out, index=False)
print(f"[DONE] Delta summary table → {delta_out}")

# ---------------------------------------------------------------------
# 7. Plot styling
# ---------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# ---------------------------------------------------------------------
# 8. Figure 1: Overlap metrics boxplots
# ---------------------------------------------------------------------
plot1 = pd.DataFrame({
    "specimen_id": segm_all["specimen_id"],
    "Dice cortical": segm_all["DSC_cortical"],
    "Dice trabecular": segm_all["DSC_trabecular"],
    "Jaccard cortical": segm_all["Jaccard_cortical"],
    "Jaccard trabecular": segm_all["Jaccard_trabecular"],
}).melt(id_vars="specimen_id", var_name="metric", value_name="value")

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=plot1, x="metric", y="value", ax=ax, color="#c7dcef", fliersize=0)
sns.stripplot(data=plot1, x="metric", y="value", ax=ax, color="black", size=4, alpha=0.7)
ax.set_xlabel("")
ax.set_ylabel("Overlap metric")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig1_out = STATS_DIR / "fig_overlap_metrics.png"
plt.savefig(fig1_out)
plt.close()
print(f"[DONE] Overlap metrics figure → {fig1_out}")

# ---------------------------------------------------------------------
# 9. Figure 2: Surface distance metrics boxplots
# ---------------------------------------------------------------------
plot2 = pd.DataFrame({
    "specimen_id": segm_all["specimen_id"],
    "HD95 cortical": segm_all["HD95_mm_cortical"],
    "HD95 trabecular": segm_all["HD95_mm_trabecular"],
    "MSD cortical": segm_all["MSD_mm_cortical"],
    "MSD trabecular": segm_all["MSD_mm_trabecular"],
}).melt(id_vars="specimen_id", var_name="metric", value_name="value")

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=plot2, x="metric", y="value", ax=ax, color="#f3d19c", fliersize=0)
sns.stripplot(data=plot2, x="metric", y="value", ax=ax, color="black", size=4, alpha=0.7)
ax.set_xlabel("")
ax.set_ylabel("Distance (mm)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig2_out = STATS_DIR / "fig_distance_metrics.png"
plt.savefig(fig2_out)
plt.close()
print(f"[DONE] Distance metrics figure → {fig2_out}")

# ---------------------------------------------------------------------
# 10. Figure 3: Manual vs automated scatter plots
# ---------------------------------------------------------------------
params_for_scatter = ["BVTV_trab", "TbSp_trab", "TbN_trab", "CtAr_cort"]

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

for ax, param in zip(axes, params_for_scatter):
    man_col = f"{param}_manual"
    auto_col = f"{param}_auto"
    if man_col not in deltas.columns or auto_col not in deltas.columns:
        ax.set_visible(False)
        continue

    tmp = deltas[[man_col, auto_col]].dropna()
    x = tmp[man_col]
    y = tmp[auto_col]

    sns.scatterplot(x=x, y=y, ax=ax, s=45, color="#2c7fb8")
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1)
    ax.set_title(param)
    ax.set_xlabel("Manual")
    ax.set_ylabel("Automated")

plt.tight_layout()
fig3_out = STATS_DIR / "fig_manual_vs_automated_scatter.png"
plt.savefig(fig3_out)
plt.close()
print(f"[DONE] Scatter plots → {fig3_out}")

# ---------------------------------------------------------------------
# 11. Figure 4: Bland–Altman plots
# ---------------------------------------------------------------------
params_for_ba = ["BVTV_trab", "TbSp_trab", "TbN_trab", "CtAr_cort"]

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

for ax, param in zip(axes, params_for_ba):
    man_col = f"{param}_manual"
    auto_col = f"{param}_auto"
    if man_col not in deltas.columns or auto_col not in deltas.columns:
        ax.set_visible(False)
        continue

    tmp = deltas[[man_col, auto_col]].dropna().copy()
    tmp["mean"] = (tmp[man_col] + tmp[auto_col]) / 2
    tmp["diff"] = tmp[auto_col] - tmp[man_col]

    ba = bland_altman_stats(tmp[man_col], tmp[auto_col])

    sns.scatterplot(data=tmp, x="mean", y="diff", ax=ax, s=45, color="#d95f0e")
    ax.axhline(ba["bias"], color="black", linestyle="-", linewidth=1)
    ax.axhline(ba["loa_lower"], color="gray", linestyle="--", linewidth=1)
    ax.axhline(ba["loa_upper"], color="gray", linestyle="--", linewidth=1)
    ax.axhline(0, color="lightgray", linestyle=":", linewidth=1)
    ax.set_title(param)
    ax.set_xlabel("Mean of manual and automated")
    ax.set_ylabel("Automated - Manual")

plt.tight_layout()
fig4_out = STATS_DIR / "fig_bland_altman.png"
plt.savefig(fig4_out)
plt.close()
print(f"[DONE] Bland–Altman plots → {fig4_out}")