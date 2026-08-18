#!/usr/bin/env python3
"""
MC01_standard_algorithms_23162L.py

Development script for specimen 23162L.

Compares manual reference segmentations against three standard approaches:
- Kohler: cortical + trabecular masks
- Herbst: cortical + trabecular TIFF-slice folders
- Klintstroem: trabecular TIFF-slice folder only

The comparison domain is the full reconstructed 3D volume, matching the
existing M01/M02 analysis logic. The script calculates the same voxel-level,
surface-distance, and global morphometry outcomes, then merges these results
with the existing manual vs probability-method master summary into long and
wide five-method overview tables.

Important interpretation:
- Manual masks are the expert reference, not absolute ground truth.
- Klintstroem is intentionally trabecular-only; cortical outputs are NaN.
- This script does not rerun or alter the existing probability-method analysis.

Expected existing manual files:
BASE_DIR/23162R&L/23162L/segmentation_manual/
    23162L_cortical_tiff/23162L_cortical_manual_full.tiff
    23162L_trabecular_tiff/23162L_trabecular_manual_full.tiff

Expected standard-method files:
Kohler/
    23162L_cortical_mask_final.tiff
    23162L_trabecular_mask_final.tiff
DragonflyHerbst/
    Cortical_Final/*.tiff
    Trabecular_Final/*.tiff
3DSlicerKlintstrom/
    Trabecular_tiff/*.tiff

Outputs:
BASE_DIR/23162R&L/23162L/PaperComparison/StandardMethodsComparison/
    23162L_standard_methods_long.csv
    23162L_standard_methods_morphometry_long.csv
    23162L_standard_methods_overview_long.csv
    23162L_all_five_methods_overview_long.csv
    23162L_all_five_methods_overview_wide.csv
    <specimen>_<method>_<compartment>_metrics.csv
    <specimen>_<method>_morphometry.csv
    <specimen>_<method>_voxel_surface_metrics.txt

Requirements:
    pip install numpy pandas scipy scikit-learn tifffile SimpleITK porespy
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from sklearn.metrics import cohen_kappa_score
import SimpleITK as sitk
import porespy as ps

# ---------------------------------------------------------------------
# Configuration: development specimen only
# ---------------------------------------------------------------------

BASE_DIR = Path(r"T:\TMMI Shared\Projects\J_SegmentationPaper\Comparison")
PAIR_FOLDER = "23162R&L"
SPECIMEN_ID = "23162L"
VOXEL_SIZE_MM = 0.032

MANUAL_BASE_DIR = Path(
    r"C:\Users\jeichwal\Documents\A_DocTech\P1_Segmentation\Data"
)

MANUAL_ROOT = (
    MANUAL_BASE_DIR
    / PAIR_FOLDER
    / SPECIMEN_ID
    / "segmentation_manual"
)
MANUAL_CORTICAL = (
    MANUAL_ROOT
    / f"{SPECIMEN_ID}_cortical_tiff"
    / f"{SPECIMEN_ID}_cortical_manual_full.tiff"
)
MANUAL_TRABECULAR = (
    MANUAL_ROOT
    / f"{SPECIMEN_ID}_trabecular_tiff"
    / f"{SPECIMEN_ID}_trabecular_manual_full.tiff"
)

METHOD_INPUTS = {
    "kohler": {
        "cortical": BASE_DIR / PAIR_FOLDER / SPECIMEN_ID / "Kohler" / f"{SPECIMEN_ID}_cortical_mask_final.tiff",
        "trabecular": BASE_DIR / PAIR_FOLDER / SPECIMEN_ID / "Kohler" / f"{SPECIMEN_ID}_trabecular_mask_final.tiff",
    },
    "herbst": {
        "cortical": BASE_DIR / PAIR_FOLDER / SPECIMEN_ID / "DragonflyHerbst" / "Cortical_Final",
        "trabecular": BASE_DIR / PAIR_FOLDER / SPECIMEN_ID / "DragonflyHerbst" / "Trabecular_Final",
    },
    "klintstroem": {
        "cortical": None,
        "trabecular": BASE_DIR / PAIR_FOLDER / SPECIMEN_ID / "3DSlicerKlintstrom" / "Trabecular_tiff",
    },
}

SPECIMEN_DIR = BASE_DIR / PAIR_FOLDER / SPECIMEN_ID
OUT_DIR = SPECIMEN_DIR / "PaperComparison" / "StandardMethodsComparison"
EXISTING_MASTER_SUMMARY = (
    MANUAL_BASE_DIR / "all_specimens_segmentation_summary.csv"
)

# ---------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------


def read_3d_tiff(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")
    image = sitk.ReadImage(str(path))
    return (sitk.GetArrayFromImage(image) > 0).astype(bool)


def read_tiff_slice_folder(folder: Path) -> np.ndarray:
    if not folder.is_dir():
        raise FileNotFoundError(f"TIFF slice folder not found: {folder}")

    paths = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    )
    if not paths:
        raise FileNotFoundError(f"No TIFF slices found in: {folder}")

    slices = []
    shape_2d: Optional[tuple[int, int]] = None
    for path in paths:
        img = tifffile.imread(path)
        img = np.squeeze(img)
        if img.ndim != 2:
            raise ValueError(f"Expected 2D TIFF slice; got shape {img.shape} in {path}")
        if shape_2d is None:
            shape_2d = img.shape
        elif img.shape != shape_2d:
            raise ValueError(
                f"Inconsistent TIFF slice dimensions in {folder}: expected {shape_2d}, got {img.shape} ({path.name})"
            )
        slices.append(img > 0)

    volume = np.stack(slices, axis=0).astype(bool)
    print(f"[INFO] Loaded {len(paths)} slices from {folder}")
    return volume


def read_mask_source(source: Path) -> np.ndarray:
    if source.is_file():
        return read_3d_tiff(source)
    if source.is_dir():
        return read_tiff_slice_folder(source)
    raise FileNotFoundError(f"Mask source does not exist: {source}")


def validate_shape(name: str, mask: np.ndarray, reference: np.ndarray) -> None:
    if mask.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch for {name}: {mask.shape} versus manual reference {reference.shape}. "
            "All masks must have identical dimensions, orientation, voxel grid, and crop."
        )

# ---------------------------------------------------------------------
# Voxel and surface metrics: matched to M01
# ---------------------------------------------------------------------


def voxel_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    ref = reference.astype(bool).ravel()
    pred = prediction.astype(bool).ravel()

    tp = int(np.sum(ref & pred))
    tn = int(np.sum(~ref & ~pred))
    fp = int(np.sum(~ref & pred))
    fn = int(np.sum(ref & ~pred))
    eps = 1e-9

    sensitivity = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)

    if (tp + tn + fp + fn) > 0 and (tp + fn) > 0 and (tp + fp) > 0:
        kappa = cohen_kappa_score(ref.astype(int), pred.astype(int))
    else:
        kappa = np.nan

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "DSC": float((2 * tp) / (2 * tp + fp + fn + eps)),
        "Jaccard": float(tp / (tp + fp + fn + eps)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(tn / (tn + fp + eps)),
        "Precision": float(precision),
        "F1": float((2 * precision * sensitivity) / (precision + sensitivity + eps)),
        "VolumeSimilarity": float(1.0 - abs((tp + fn) - (tp + fp)) / ((tp + fn) + (tp + fp) + eps)),
        "FPE": float(fp / (tp + fp + eps)),
        "FNE": float(fn / (tp + fn + eps)),
        "CohenKappa": float(kappa),
        "ReferenceVoxelCount": int(ref.sum()),
        "MethodVoxelCount": int(pred.sum()),
        "VoxelDifference": int(pred.sum() - ref.sum()),
        "VoxelDifference_pct": float((pred.sum() - ref.sum()) / (ref.sum() + eps) * 100.0),
    }


def surface_from_mask(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3, 3), dtype=bool)
    eroded = ndi.binary_erosion(mask.astype(bool), structure=structure)
    return mask.astype(bool) & ~eroded


def surface_metrics(reference: np.ndarray, prediction: np.ndarray, voxel_size_mm: float) -> dict:
    ref = reference.astype(bool)
    pred = prediction.astype(bool)
    if not ref.any() or not pred.any():
        return {"MSD_mm": np.nan, "HD95_mm": np.nan, "HD_max_mm": np.nan}

    ref_surface = surface_from_mask(ref)
    pred_surface = surface_from_mask(pred)
    if not ref_surface.any() or not pred_surface.any():
        return {"MSD_mm": np.nan, "HD95_mm": np.nan, "HD_max_mm": np.nan}

    distance_to_pred_surface = ndi.distance_transform_edt(~pred_surface) * voxel_size_mm
    distance_to_ref_surface = ndi.distance_transform_edt(~ref_surface) * voxel_size_mm
    distances = np.concatenate([
        distance_to_pred_surface[ref_surface],
        distance_to_ref_surface[pred_surface],
    ])

    return {
        "MSD_mm": float(np.mean(distances)),
        "HD95_mm": float(np.percentile(distances, 95)),
        "HD_max_mm": float(np.max(distances)),
    }

# ---------------------------------------------------------------------
# Morphometry: matched to M02
# ---------------------------------------------------------------------


def compute_bvtv(trabecular_mask: np.ndarray) -> float:
    return float(trabecular_mask.sum() / trabecular_mask.size) if trabecular_mask.size else np.nan


def local_thickness_mean(mask: np.ndarray, voxel_size_mm: float) -> float:
    binary = mask.astype(bool)
    if not binary.any():
        return np.nan
    thickness = ps.filters.local_thickness(im=binary, method="dt")
    values = thickness[binary].astype(np.float64) * voxel_size_mm
    return float(values.mean()) if values.size else np.nan


def compute_tb_spacing(trabecular_mask: np.ndarray, voxel_size_mm: float) -> float:
    void = ~trabecular_mask.astype(bool)
    if not void.any():
        return np.nan
    thickness = ps.filters.local_thickness(im=void, method="dt")
    values = thickness[void].astype(np.float64) * voxel_size_mm
    return float(values.mean()) if values.size else np.nan


def compute_ct_areas(cortical_mask: np.ndarray, total_mask: np.ndarray, voxel_size_mm: float) -> tuple[float, float, float]:
    length_mm = cortical_mask.shape[0] * voxel_size_mm
    if length_mm <= 0:
        return np.nan, np.nan, np.nan
    voxel_volume_mm3 = voxel_size_mm ** 3
    ct_ar = cortical_mask.sum() * voxel_volume_mm3 / length_mm
    tt_ar = total_mask.sum() * voxel_volume_mm3 / length_mm
    ratio = ct_ar / tt_ar if tt_ar > 0 else np.nan
    return float(ct_ar), float(tt_ar), float(ratio)


def morphometry_row(
    specimen_id: str,
    method: str,
    cortical: Optional[np.ndarray],
    trabecular: np.ndarray,
) -> dict:
    bvtv = compute_bvtv(trabecular)
    tb_th = local_thickness_mean(trabecular, VOXEL_SIZE_MM)
    tb_sp = compute_tb_spacing(trabecular, VOXEL_SIZE_MM)
    tb_n = bvtv / tb_th if np.isfinite(tb_th) and tb_th > 0 else np.nan

    result = {
        "specimen_id": specimen_id,
        "method": method,
        "BVTV_trab": bvtv,
        "TbTh_trab": tb_th,
        "TbSp_trab": tb_sp,
        "TbN_trab": tb_n,
        "CtTh_cort": np.nan,
        "CtAr_cort": np.nan,
        "CtAr_TtAr": np.nan,
        "TtAr_total": np.nan,
        "TrabecularVoxelCount": int(trabecular.sum()),
        "CorticalVoxelCount": np.nan,
        "TotalBoneVoxelCount": np.nan,
    }

    if cortical is not None:
        cortical = cortical.astype(bool)
        total = cortical | trabecular.astype(bool)
        ct_th = local_thickness_mean(cortical, VOXEL_SIZE_MM)
        ct_ar, tt_ar, ratio = compute_ct_areas(cortical, total, VOXEL_SIZE_MM)
        result.update({
            "CtTh_cort": ct_th,
            "CtAr_cort": ct_ar,
            "CtAr_TtAr": ratio,
            "TtAr_total": tt_ar,
            "CorticalVoxelCount": int(cortical.sum()),
            "TotalBoneVoxelCount": int(total.sum()),
        })

    return result

# ---------------------------------------------------------------------
# Reporting and merge helpers
# ---------------------------------------------------------------------


def save_text_report(path: Path, specimen_id: str, method: str, results: pd.DataFrame) -> None:
    lines = [
        f"Specimen: {specimen_id}",
        f"Method: {method}",
        "Reference: expert manual segmentation",
        "Comparison domain: full reconstructed 3D volume",
        "",
    ]
    for _, row in results.iterrows():
        lines.append(f"{row['compartment'].capitalize()} metrics")
        for col, value in row.items():
            if col in {"specimen_id", "method", "compartment"}:
                continue
            lines.append(f"{col}: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def existing_probability_rows(master_csv: Path, specimen_id: str) -> pd.DataFrame:
    if not master_csv.is_file():
        print(f"[WARN] Existing master summary not found; five-method merge skipped: {master_csv}")
        return pd.DataFrame()

    source = pd.read_csv(master_csv)
    source = source[source["specimen_id"].astype(str) == specimen_id].copy()
    if source.empty:
        print(f"[WARN] No existing probability/manual results for {specimen_id} in {master_csv}")
        return pd.DataFrame()

    row = source.iloc[0]
    records = []
    metric_names = [
        "DSC", "Jaccard", "Sensitivity", "Specificity", "Precision", "F1",
        "VolumeSimilarity", "FPE", "FNE", "CohenKappa", "MSD_mm", "HD95_mm", "HD_max_mm",
    ]
    for compartment in ("cortical", "trabecular"):
        record = {"specimen_id": specimen_id, "method": "probability", "compartment": compartment}
        for metric in metric_names:
            column = f"{metric}_{compartment}"
            record[metric] = row[column] if column in row.index else np.nan
        records.append(record)

    return pd.DataFrame(records)


def existing_manual_probability_morphometry(master_csv: Path, specimen_id: str) -> pd.DataFrame:
    if not master_csv.is_file():
        return pd.DataFrame()
    source = pd.read_csv(master_csv)
    source = source[source["specimen_id"].astype(str) == specimen_id].copy()
    if source.empty:
        return pd.DataFrame()

    row = source.iloc[0]
    fields = ["BVTV_trab", "TbTh_trab", "TbSp_trab", "TbN_trab", "CtTh_cort", "CtAr_cort", "CtAr_TtAr", "TtAr_total"]
    records = []
    for method, suffix in [("manual", "manual"), ("probability", "auto")]:
        record = {"specimen_id": specimen_id, "method": method}
        for field in fields:
            column = f"{field}_{suffix}"
            record[field] = row[column] if column in row.index else np.nan
        records.append(record)
    return pd.DataFrame(records)


def make_wide_overview(long_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["specimen_id", "method", "compartment"]
    value_cols = [c for c in long_df.columns if c not in id_cols]
    if long_df.empty or not value_cols:
        return pd.DataFrame()
    wide = long_df.pivot_table(
        index="specimen_id",
        columns=["method", "compartment"],
        values=value_cols,
        aggfunc="first",
    )
    wide.columns = ["__".join(str(part) for part in col if part != "") for col in wide.columns.to_flat_index()]
    return wide.reset_index()

# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Development specimen: {SPECIMEN_ID}")
    print(f"[INFO] Output directory: {OUT_DIR}")
    print("[INFO] Loading manual reference masks")
    manual_cortical = read_3d_tiff(MANUAL_CORTICAL)
    manual_trabecular = read_3d_tiff(MANUAL_TRABECULAR)
    validate_shape("manual trabecular", manual_trabecular, manual_cortical)

    all_metric_rows = []
    all_morphometry_rows = []

    for method, inputs in METHOD_INPUTS.items():
        print(f"\n[INFO] Processing {method}")
        method_trabecular = read_mask_source(inputs["trabecular"])
        validate_shape(f"{method} trabecular", method_trabecular, manual_trabecular)

        method_cortical = None
        if inputs["cortical"] is not None:
            method_cortical = read_mask_source(inputs["cortical"])
            validate_shape(f"{method} cortical", method_cortical, manual_cortical)

            overlap = method_cortical & method_trabecular
            if overlap.any():
                print(
                    f"[WARN] {method}: cortical and trabecular masks overlap in {int(overlap.sum())} voxels. "
                    "Metrics are retained as supplied; inspect the source masks before final analysis."
                )

        compartment_pairs = [("trabecular", manual_trabecular, method_trabecular)]
        if method_cortical is not None:
            compartment_pairs.insert(0, ("cortical", manual_cortical, method_cortical))

        method_rows = []
        for compartment, reference, prediction in compartment_pairs:
            metrics = voxel_metrics(reference, prediction)
            metrics.update(surface_metrics(reference, prediction, VOXEL_SIZE_MM))
            metrics.update({
                "specimen_id": SPECIMEN_ID,
                "method": method,
                "compartment": compartment,
            })
            method_rows.append(metrics)
            all_metric_rows.append(metrics)

            per_compartment_csv = OUT_DIR / f"{SPECIMEN_ID}_{method}_{compartment}_metrics.csv"
            pd.DataFrame([metrics]).to_csv(per_compartment_csv, index=False)
            print(f"[DONE] Saved {per_compartment_csv.name}")

        method_metrics_df = pd.DataFrame(method_rows)
        report_path = OUT_DIR / f"{SPECIMEN_ID}_{method}_voxel_surface_metrics.txt"
        save_text_report(report_path, SPECIMEN_ID, method, method_metrics_df)

        morphometry = morphometry_row(SPECIMEN_ID, method, method_cortical, method_trabecular)
        all_morphometry_rows.append(morphometry)
        morphometry_csv = OUT_DIR / f"{SPECIMEN_ID}_{method}_morphometry.csv"
        pd.DataFrame([morphometry]).to_csv(morphometry_csv, index=False)
        print(f"[DONE] Saved {morphometry_csv.name}")

    standard_metrics_df = pd.DataFrame(all_metric_rows)
    standard_morphometry_df = pd.DataFrame(all_morphometry_rows)
    standard_metrics_csv = OUT_DIR / f"{SPECIMEN_ID}_standard_methods_long.csv"
    standard_morphometry_csv = OUT_DIR / f"{SPECIMEN_ID}_standard_methods_morphometry_long.csv"
    standard_metrics_df.to_csv(standard_metrics_csv, index=False)
    standard_morphometry_df.to_csv(standard_morphometry_csv, index=False)

    standard_overview = standard_metrics_df.merge(
        standard_morphometry_df,
        on=["specimen_id", "method"],
        how="left",
    )
    standard_overview_csv = OUT_DIR / f"{SPECIMEN_ID}_standard_methods_overview_long.csv"
    standard_overview.to_csv(standard_overview_csv, index=False)
    print(f"[DONE] Saved {standard_overview_csv.name}")

    probability_metrics = existing_probability_rows(EXISTING_MASTER_SUMMARY, SPECIMEN_ID)
    manual_probability_morphometry = existing_manual_probability_morphometry(EXISTING_MASTER_SUMMARY, SPECIMEN_ID)

    five_method_metrics = pd.concat([probability_metrics, standard_metrics_df], ignore_index=True, sort=False)
    five_method_overview = five_method_metrics.merge(
        standard_morphometry_df,
        on=["specimen_id", "method"],
        how="left",
    )

    # Add current manual/probability morphometry to every corresponding compartment row.
    if not manual_probability_morphometry.empty:
        five_method_overview = five_method_overview.merge(
            manual_probability_morphometry,
            on=["specimen_id", "method"],
            how="left",
            suffixes=("", "_from_existing"),
        )
        for field in ["BVTV_trab", "TbTh_trab", "TbSp_trab", "TbN_trab", "CtTh_cort", "CtAr_cort", "CtAr_TtAr", "TtAr_total"]:
            existing_field = f"{field}_from_existing"
            if existing_field in five_method_overview.columns:
                five_method_overview[field] = five_method_overview[field].combine_first(five_method_overview[existing_field])
                five_method_overview = five_method_overview.drop(columns=existing_field)

    five_method_long_csv = OUT_DIR / f"{SPECIMEN_ID}_all_five_methods_overview_long.csv"
    five_method_overview.to_csv(five_method_long_csv, index=False)
    print(f"[DONE] Saved {five_method_long_csv.name}")

    five_method_wide = make_wide_overview(five_method_overview)
    five_method_wide_csv = OUT_DIR / f"{SPECIMEN_ID}_all_five_methods_overview_wide.csv"
    five_method_wide.to_csv(five_method_wide_csv, index=False)
    print(f"[DONE] Saved {five_method_wide_csv.name}")

    print("\n[COMPLETE] Development analysis finished.")
    print("[NEXT] Inspect the three standard-method masks and CSV outputs for 23162L.")
    print("[NEXT] After validation, replace the single-specimen configuration with a specimen/side loop.")


if __name__ == "__main__":
    main()