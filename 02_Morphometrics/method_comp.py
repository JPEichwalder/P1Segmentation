from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# Paths and settings
# ------------------------------------------------
BONEJ_XLSX = Path(
    r"T:\TMMI Shared\Projects\Puck_Validation\Workflow\Development\MorphStudy\DataReliance.xlsx"
)

BASE_DIR = Path(
    r"T:\TMMI Shared\Projects\Puck_Validation\Workflow\Development"
)

OUT_ROOT = Path(
    r"T:\TMMI Shared\Projects\Puck_Validation\Workflow\Development\MorphStudy\Results"
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OVERALL_OUT = OUT_ROOT / "overall"
OVERALL_OUT.mkdir(parents=True, exist_ok=True)

# specimen bases (numeric part only)
SPECIMEN_BASES = ["23162",
"23162",
"23167",
"23168",
"23169",
"23170",
"23175",]
# extend, e.g. ["23162", "23167", "23170"]

METRIC_NAMES = [
    "BV_mm^3",
    "TV_mm^3",
    "BVTV",
    "Tb.Th_mean_mm",
    "Tb.Th_std_mm",
    "Tb.Th_max_mm",
    "Tb.Sp_mean_mm",
    "Tb.Sp_std_mm",
    "Tb.Sp_max_mm",
    "Euler_chi",
    "Connectivity",
    "ConnD_mm^-3",
]

CUBE_COL = "cube_file"
SIDE_COL = "specimen_side"


def load_bonej_table(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0)


def load_python_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def merge_tables(bonej_side: pd.DataFrame, py: pd.DataFrame) -> pd.DataFrame:
    bonej_side[CUBE_COL] = bonej_side[CUBE_COL].astype(str)
    py[CUBE_COL] = py[CUBE_COL].astype(str)

    merged = pd.merge(
        bonej_side,
        py,
        on=[SIDE_COL, CUBE_COL],
        how="inner",
        suffixes=("_BoneJ", "_Py"),
    )
    return merged


def analyse_metric(merged: pd.DataFrame, name: str):
    x = merged[f"{name}_BoneJ"].astype(float).values
    y = merged[f"{name}_Py"].astype(float).values

    diff = y - x
    ratio = np.where(x != 0, y / x, np.nan)

    corr = np.corrcoef(x, y)[0, 1]
    stats = {
        "pearson_r": corr,
        "ratio_mean": float(np.nanmean(ratio)),
        "ratio_std": float(np.nanstd(ratio)),
        "diff_mean": float(np.nanmean(diff)),
        "diff_std": float(np.nanstd(diff)),
    }
    return stats, x, y


def plot_metric(name: str, x, y, stats: dict, out_dir: Path, title_prefix: str = ""):
    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, edgecolor="k", alpha=0.8)
    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    plt.plot([lo, hi], [lo, hi], "r--", label="Identity")
    plt.xlabel(f"{name} (BoneJ)")
    plt.ylabel(f"{name} (Python)")
    plt.title(
        f"{title_prefix}{name}: r={stats['pearson_r']:.3f}, "
        f"scale≈{stats['ratio_mean']:.3f}"
    )
    plt.tight_layout()
    out = out_dir / f"scatter_{name}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def process_specimen_side(bonej: pd.DataFrame, specimen_base: str, side: str):
    side_id = f"{specimen_base}{side}"
    pair_folder = f"{specimen_base}R&L"
    py_csv = (
        BASE_DIR
        / pair_folder
        / side_id
        / "Cube"
        / f"{side_id}_cube_microstructure_python.csv"
    )

    if not py_csv.is_file():
        print(f"[WARN] Python CSV not found for {side_id}: {py_csv}")
        return pd.DataFrame()

    bonej_side = bonej[bonej[SIDE_COL] == side_id].copy()
    if bonej_side.empty:
        print(f"[WARN] No BoneJ rows for {side_id} in {BONEJ_XLSX}")
        return pd.DataFrame()

    py = load_python_table(py_csv)
    merged = merge_tables(bonej_side, py)

    if merged.empty:
        print(f"[WARN] No matching cubes for {side_id} (specimen_side + cube_file)")
        return pd.DataFrame()

    out_dir = OUT_ROOT / f"{specimen_base}{side}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] {side_id}: merged rows = {len(merged)}")

    summary = []

    for name in METRIC_NAMES:
        if f"{name}_BoneJ" not in merged.columns or f"{name}_Py" not in merged.columns:
            print(f"[WARN] missing columns for {name} in {side_id}")
            continue

        stats, x, y = analyse_metric(merged, name)
        plot_path = plot_metric(name, x, y, stats, out_dir, title_prefix=f"{side_id} ")

        print(
            f"  {name}: r={stats['pearson_r']:.3f}, "
            f"mean(Python/BoneJ)={stats['ratio_mean']:.3f}"
        )

        summary.append(
            {
                "specimen_side": side_id,
                "metric": name,
                "pearson_r": stats["pearson_r"],
                "mean_ratio_Py_div_BoneJ": stats["ratio_mean"],
                "ratio_std": stats["ratio_std"],
                "mean_diff_Py_minus_BoneJ": stats["diff_mean"],
                "diff_std": stats["diff_std"],
                "scatter_png": plot_path.name,
            }
        )

    merged.to_csv(out_dir / f"merged_per_cube_{side_id}.csv", index=False)
    pd.DataFrame(summary).to_csv(
        out_dir / f"summary_{side_id}_BoneJ_vs_Python.csv", index=False
    )

    return merged


def analyse_metric_overall(df: pd.DataFrame, name: str):
    x = df[f"{name}_BoneJ"].astype(float).values
    y = df[f"{name}_Py"].astype(float).values

    diff = y - x
    ratio = np.where(x != 0, y / x, np.nan)

    corr = np.corrcoef(x, y)[0, 1]
    stats = {
        "pearson_r": corr,
        "ratio_mean": float(np.nanmean(ratio)),
        "ratio_std": float(np.nanstd(ratio)),
        "diff_mean": float(np.nanmean(diff)),
        "diff_std": float(np.nanstd(diff)),
    }
    return stats


def plot_metric_overall(name: str, df: pd.DataFrame, stats: dict, out_dir: Path):
    plt.figure(figsize=(5, 5))

    side_ids = df[SIDE_COL].astype(str)
    specimen_bases = side_ids.str[:-1]
    sides = side_ids.str[-1]

    unique_specimens = sorted(specimen_bases.unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_specimens)))
    color_map = dict(zip(unique_specimens, colors))

    for spec in unique_specimens:
        for side in ["R", "L"]:
            mask = (specimen_bases == spec) & (sides == side)
            if not mask.any():
                continue

            x = df.loc[mask, f"{name}_BoneJ"].astype(float).values
            y = df.loc[mask, f"{name}_Py"].astype(float).values

            marker = "o" if side == "R" else "s"
            plt.scatter(
                x,
                y,
                color=color_map[spec],
                marker=marker,
                edgecolor="k",
                alpha=0.8,
                label=f"{spec}{side}",
            )

    x_all = df[f"{name}_BoneJ"].astype(float).values
    y_all = df[f"{name}_Py"].astype(float).values
    lo = min(np.nanmin(x_all), np.nanmin(y_all))
    hi = max(np.nanmax(x_all), np.nanmax(y_all))
    plt.plot([lo, hi], [lo, hi], "r--", label="Identity")

    plt.xlabel(f"{name} (BoneJ)")
    plt.ylabel(f"{name} (Python)")
    plt.title(
        f"All samples {name}: r={stats['pearson_r']:.3f}, "
        f"scale≈{stats['ratio_mean']:.3f}"
    )
    plt.legend(fontsize=6, ncol=2, frameon=True)
    plt.tight_layout()
    out = out_dir / f"scatter_all_{name}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def main():
    bonej = load_bonej_table(BONEJ_XLSX)

    all_merged = []

    for specimen_base in SPECIMEN_BASES:
        for side in ("R", "L"):
            merged = process_specimen_side(bonej, specimen_base, side)
            if not merged.empty:
                all_merged.append(merged)

    # overall results across all specimens/sides
    if all_merged:
        merged_all = pd.concat(all_merged, ignore_index=True)
        merged_all_path = OUT_ROOT / "merged_all_specimens_sides.csv"
        merged_all.to_csv(merged_all_path, index=False)

        overall_summary = []

        for name in METRIC_NAMES:
            if (
                f"{name}_BoneJ" not in merged_all.columns
                or f"{name}_Py" not in merged_all.columns
            ):
                print(f"[WARN] missing columns for {name} in overall merged data")
                continue

            stats = analyse_metric_overall(merged_all, name)
            plot_path = plot_metric_overall(name, merged_all, stats, OVERALL_OUT)

            overall_summary.append(
                {
                    "metric": name,
                    "pearson_r": stats["pearson_r"],
                    "mean_ratio_Py_div_BoneJ": stats["ratio_mean"],
                    "ratio_std": stats["ratio_std"],
                    "mean_diff_Py_minus_BoneJ": stats["diff_mean"],
                    "diff_std": stats["diff_std"],
                    "scatter_png": plot_path.name,
                }
            )

        pd.DataFrame(overall_summary).to_csv(
            OVERALL_OUT / "summary_all_specimens_sides_BoneJ_vs_Python.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
