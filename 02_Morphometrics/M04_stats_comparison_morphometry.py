#!/usr/bin/env python3
"""
M04_MorphometryStats.py

Morphometry comparison between manual and semi-automatic segmentations.

This script:
1) Reads per-specimen morphometry from M03:
       all_specimens_segmentation_summary.csv
2) Computes per-specimen absolute and percent differences (auto - manual)
   for all key morphometric parameters.
3) Generates simple bias and scatter plots for selected parameters.
4) Optionally merges with whole-bone morphometry (BV/TV context).
5) Computes agreement statistics (r, ICC(2,1), Bland–Altman, mean/SD,
   median/IQR) for all morphometric parameters and saves a single
   summary CSV + per-parameter plots.

Inputs:
    BASE_DEV/all_specimens_segmentation_summary.csv  (from M03)
    BASE_DEV/StatsComparison/whole_bone_morphometry.csv  (from script08)

Outputs (in BASE_DEV/StatsComparison):
    all_specimens_with_deltas.csv
    bvtv_comparisons.csv
    bias_BVTV_trab_pct_bar.png
    bias_TbTh_trab_pct_bar.png
    bias_CtTh_cort_pct_bar.png
    scatter_BVTV_trab_manual_vs_auto.png
    scatter_CtTh_cort_manual_vs_auto.png

    morpho_agreement_summary_all_params.csv
    morpho_scatter_<param>_manual_vs_auto_ICC.png
    morpho_bland_altman_<param>.png
"""

from pathlib import Path
from math import isnan

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DEV = Path(r"C:\Users\jeichwal\Documents\A_DocTech\P1_Segmentation\Data")
INPUT_CSV = BASE_DEV / "all_specimens_segmentation_summary.csv"
OUT_DIR = BASE_DEV / "StatsComparison"

WHOLE_BONE_CSV = OUT_DIR / "whole_bone_morphometry.csv"  # from script08


# -------------------------------------------------------------------
# Part 1: deltas and simple plots (old M04)
# -------------------------------------------------------------------
def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute absolute and percent differences (auto - manual) for
    all key morphometric parameters and return enriched DataFrame.
    """
    pairs = [
        ("BVTV_trab", "BVTV_trab_manual", "BVTV_trab_auto"),
        ("TbTh_trab", "TbTh_trab_manual", "TbTh_trab_auto"),
        ("TbSp_trab", "TbSp_trab_manual", "TbSp_trab_auto"),
        ("TbN_trab", "TbN_trab_manual", "TbN_trab_auto"),
        ("CtTh_cort", "CtTh_cort_manual", "CtTh_cort_auto"),
        ("CtAr_cort", "CtAr_cort_manual", "CtAr_cort_auto"),
        ("CtAr_TtAr", "CtAr_TtAr_manual", "CtAr_TtAr_auto"),
        ("TtAr_total", "TtAr_total_manual", "TtAr_total_auto"),
    ]

    out = df.copy()

    for base_name, man_col, auto_col in pairs:
        if man_col not in out.columns or auto_col not in out.columns:
            print(f"[WARN] Missing columns for {base_name}: {man_col}, {auto_col}")
            continue

        dcol = f"d_{base_name}"       # absolute difference (auto - manual)
        pcol = f"d_{base_name}_pct"   # percent difference relative to manual

        out[dcol] = out[auto_col] - out[man_col]
        out[pcol] = out[dcol] / out[man_col].replace({0: pd.NA}) * 100.0

    return out


def barplot_percent_bias(out: pd.DataFrame, param_base: str, ylabel: str, filename: str):
    pcol = f"d_{param_base}_pct"
    if pcol not in out.columns:
        print(f"[WARN] Cannot plot {param_base}: missing column {pcol}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = out["specimen_id"].astype(str)
    y = out[pcol]

    ax.bar(x, y, color="steelblue", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Specimen")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Percent bias (semi-auto vs manual) in {param_base}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig_path = OUT_DIR / filename
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {fig_path}")


def scatter_manual_vs_auto(out: pd.DataFrame, param_base: str,
                           xlabel: str, ylabel: str, filename: str):
    man_col = f"{param_base}_manual"
    auto_col = f"{param_base}_auto"
    if man_col not in out.columns or auto_col not in out.columns:
        print(f"[WARN] Cannot plot {param_base}: missing {man_col} or {auto_col}")
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    x = out[man_col]
    y = out[auto_col]

    ax.scatter(x, y, color="darkorange", alpha=0.8)
    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{param_base}: semi-auto vs manual")
    plt.tight_layout()

    fig_path = OUT_DIR / filename
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved {fig_path}")


def compute_bvtv_comparisons(df_with_deltas: pd.DataFrame):
    """
    Combine morphometry (with BVTV_trab_manual/auto) and whole-bone BVTV
    to compute several BV/TV comparison metrics.
    """
    if not WHOLE_BONE_CSV.is_file():
        print(f"[WARN] Skipping BVTV comparisons (missing whole-bone CSV: {WHOLE_BONE_CSV})")
        return

    df_morph = df_with_deltas
    df_whole = pd.read_csv(WHOLE_BONE_CSV)

    needed_cols = ["specimen_id", "BVTV_trab_manual", "BVTV_trab_auto"]
    missing = [c for c in needed_cols if c not in df_morph.columns]
    if missing:
        print(f"[ERROR] Missing columns in morphometry data for BVTV comparisons: {missing}")
        return

    df_m = df_morph[needed_cols].copy()
    df_w = df_whole.copy()

    df = pd.merge(df_m, df_w, on="specimen_id", how="inner")
    if df.empty:
        print("[WARN] No overlapping specimen_id between morphometry and whole-bone CSV.")
        return

    # 1) Manual trabecular vs semi-auto trabecular
    df["d_BVTV_trab_auto_manual"] = df["BVTV_trab_auto"] - df["BVTV_trab_manual"]
    df["d_BVTV_trab_auto_manual_pct"] = (
        df["d_BVTV_trab_auto_manual"]
        / df["BVTV_trab_manual"].replace({0: pd.NA})
        * 100.0
    )

    # 2) Semi-auto trabecular vs auto whole-bone
    df["d_BVTV_trab_auto_vs_whole_auto"] = df["BVTV_trab_auto"] - df["BVTV_whole_auto"]
    df["d_BVTV_trab_auto_vs_whole_auto_pct"] = (
        df["d_BVTV_trab_auto_vs_whole_auto"]
        / df["BVTV_whole_auto"].replace({0: pd.NA})
        * 100.0
    )

    # 3) Manual trabecular vs manual whole-bone
    df["d_BVTV_trab_manual_vs_whole_manual"] = df["BVTV_trab_manual"] - df["BVTV_whole_manual"]
    df["d_BVTV_trab_manual_vs_whole_manual_pct"] = (
        df["d_BVTV_trab_manual_vs_whole_manual"]
        / df["BVTV_whole_manual"].replace({0: pd.NA})
        * 100.0
    )

    out_csv = OUT_DIR / "bvtv_comparisons.csv"
    df.to_csv(out_csv, index=False)
    print(f"[DONE] BV/TV comparison table saved to {out_csv}")


# -------------------------------------------------------------------
# Part 2: agreement stats (old M05)
# -------------------------------------------------------------------
def icc_2_1(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan

    data = np.vstack([x, y]).T
    n, k = data.shape

    mean_per_target = np.mean(data, axis=1, keepdims=True)
    mean_per_rater = np.mean(data, axis=0, keepdims=True)
    grand_mean = np.mean(data)

    ss_between_targets = k * np.sum((mean_per_target - grand_mean) ** 2)
    ss_error = np.sum((data - mean_per_target - mean_per_rater + grand_mean) ** 2)

    df_between_targets = n - 1
    df_error = (n - 1) * (k - 1)

    ms_between_targets = ss_between_targets / df_between_targets
    ms_error = ss_error / df_error

    icc_val = (ms_between_targets - ms_error) / (ms_between_targets + (k - 1) * ms_error)
    return float(icc_val)


def bland_altman_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return dict(bias=np.nan, sd_diff=np.nan, loa_lower=np.nan, loa_upper=np.nan)

    diff = y - x
    bias = np.mean(diff)
    sd_diff = np.std(diff, ddof=1) if diff.size > 1 else np.nan
    loa_lower = bias - 1.96 * sd_diff if not isnan(sd_diff) else np.nan
    loa_upper = bias + 1.96 * sd_diff if not isnan(sd_diff) else np.nan
    return dict(bias=bias, sd_diff=sd_diff, loa_lower=loa_lower, loa_upper=loa_upper)


def median_iqr(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan, np.nan
    med = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return med, (q3 - q1)


def agreement_for_param(df, man_col, auto_col, name, out_prefix):
    if man_col not in df.columns or auto_col not in df.columns:
        print(f"[WARN] Missing columns for {name}: {man_col}, {auto_col}")
        return None

    x = df[man_col].to_numpy(dtype=float)
    y = df[auto_col].to_numpy(dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        print(f"[WARN] Not enough data for {name}")
        return None

    r = np.corrcoef(x, y)[0, 1]
    icc_val = icc_2_1(x, y)
    ba = bland_altman_stats(x, y)

    diff = y - x
    man_s = pd.Series(x)
    auto_s = pd.Series(y)
    diff_s = pd.Series(diff)

    man_mean, man_sd = man_s.mean(), man_s.std(ddof=1)
    man_med, man_iqr = median_iqr(man_s)

    auto_mean, auto_sd = auto_s.mean(), auto_s.std(ddof=1)
    auto_med, auto_iqr = median_iqr(auto_s)

    diff_mean, diff_sd = diff_s.mean(), diff_s.std(ddof=1)
    diff_med, diff_iqr = median_iqr(diff_s)

    # Scatter
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(x, y, color="tab:blue", alpha=0.8)
    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)
    ax.set_xlabel(f"{name} manual")
    ax.set_ylabel(f"{name} semi-auto")
    ax.set_title(f"{name}: r={r:.2f}, ICC={icc_val:.2f}")
    plt.tight_layout()
    scatter_path = OUT_DIR / f"morpho_scatter_{out_prefix}_manual_vs_auto_ICC.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved scatter plot to {scatter_path}")

    # Bland–Altman
    mean_vals = (x + y) / 2.0
    diff_vals = diff
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(mean_vals, diff_vals, color="tab:orange", alpha=0.8)
    ax.axhline(ba["bias"], color="red", linestyle="--", label=f"Bias={ba['bias']:.3g}")
    if not isnan(ba["loa_lower"]) and not isnan(ba["loa_upper"]):
        ax.axhline(ba["loa_lower"], color="gray", linestyle="--", label="LOA")
        ax.axhline(ba["loa_upper"], color="gray", linestyle="--")
    ax.set_xlabel(f"Mean of manual and semi-auto {name}")
    ax.set_ylabel(f"Semi-auto − manual {name}")
    ax.set_title(f"Bland–Altman: {name}")
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    ba_path = OUT_DIR / f"morpho_bland_altman_{out_prefix}.png"
    plt.savefig(ba_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Bland–Altman plot to {ba_path}")

    return {
        "parameter": name,
        "n": int(x.size),
        "pearson_r": r,
        "ICC_2_1": icc_val,
        "BA_bias": ba["bias"],
        "BA_sd_diff": ba["sd_diff"],
        "BA_LOA_lower": ba["loa_lower"],
        "BA_LOA_upper": ba["loa_upper"],
        "manual_mean": man_mean,
        "manual_sd": man_sd,
        "manual_median": man_med,
        "manual_IQR": man_iqr,
        "auto_mean": auto_mean,
        "auto_sd": auto_sd,
        "auto_median": auto_med,
        "auto_IQR": auto_iqr,
        "diff_mean": diff_mean,
        "diff_sd": diff_sd,
        "diff_median": diff_med,
        "diff_IQR": diff_iqr,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    if not INPUT_CSV.is_file():
        print(f"[ERROR] Could not find input CSV: {INPUT_CSV}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Reading {INPUT_CSV}")
    df_raw = pd.read_csv(INPUT_CSV)

    # 1) deltas
    out = compute_deltas(df_raw)
    out_csv = OUT_DIR / "all_specimens_with_deltas.csv"
    out.to_csv(out_csv, index=False)
    print(f"[INFO] Saved CSV with deltas to {out_csv}")

    # 2) simple plots
    barplot_percent_bias(
        out,
        param_base="BVTV_trab",
        ylabel="Percent bias in BV/TV (semi-auto − manual) [%]",
        filename="bias_BVTV_trab_pct_bar.png",
    )
    barplot_percent_bias(
        out,
        param_base="TbTh_trab",
        ylabel="Percent bias in Tb.Th (semi-auto − manual) [%]",
        filename="bias_TbTh_trab_pct_bar.png",
    )
    barplot_percent_bias(
        out,
        param_base="CtTh_cort",
        ylabel="Percent bias in Ct.Th (semi-auto − manual) [%]",
        filename="bias_CtTh_cort_pct_bar.png",
    )

    scatter_manual_vs_auto(
        out,
        param_base="BVTV_trab",
        xlabel="BV/TV trab manual",
        ylabel="BV/TV trab semi-auto",
        filename="scatter_BVTV_trab_manual_vs_auto.png",
    )
    scatter_manual_vs_auto(
        out,
        param_base="CtTh_cort",
        xlabel="Ct.Th cort manual [mm]",
        ylabel="Ct.Th cort semi-auto [mm]",
        filename="scatter_CtTh_cort_manual_vs_auto.png",
    )

    # 3) BV/TV vs whole-bone
    compute_bvtv_comparisons(out)

    # 4) agreement stats for all parameters
    params = [
        "BVTV_trab",
        "TbTh_trab",
        "TbSp_trab",
        "TbN_trab",
        "CtTh_cort",
        "CtAr_cort",
        "CtAr_TtAr",
        "TtAr_total",
    ]
    rows = []
    for p in params:
        man_col = f"{p}_manual"
        auto_col = f"{p}_auto"
        print(f"[INFO] Agreement analysis for {p}")
        row = agreement_for_param(out, man_col, auto_col, name=p, out_prefix=p)
        if row is not None:
            rows.append(row)

    if rows:
        summary_df = pd.DataFrame(rows)
        out_summary = OUT_DIR / "morpho_agreement_summary_all_params.csv"
        summary_df.to_csv(out_summary, index=False)
        print(f"[DONE] Morphometry agreement summary saved to {out_summary}")
    else:
        print("[WARN] No agreement rows computed; check morphometry columns.")


if __name__ == "__main__":
    main()