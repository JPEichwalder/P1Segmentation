#!/usr/bin/env python3
"""
M02_global_morphometry.py

Compute global morphometric parameters for cortical and trabecular bone
for BOTH manual and semi-automated segmentations.

Adapted to match the updated M01 path logic:
- works with segmentation_final_V* or segmentation_final
- works whether masks are inside an optional Segmentation/ subfolder or directly
  inside segmentation_final*

Outputs one CSV per specimen/side with rows:
- method = "manual"
- method = "semi_auto"
"""

from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import porespy as ps

VOXEL_SIZE_MM = 0.032
BASE_DIR = Path(r"C:\Users\jeichwal\Documents\A_DocTech\P1_Segmentation\Data")

SPECIMEN_BASES = [
    "23162R&L",
    "23167R&L",
    "23168R&L",
    "23169R&L",
    "23170R&L",
    "23175R&L",
]

CORT_MAN_NAME = "{specimen_id}_cortical_manual_full.tiff"
TRAB_MAN_NAME = "{specimen_id}_trabecular_manual_full.tiff"


def read_mask(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return (arr > 0).astype(np.uint8)


def compute_bvtv(mask: np.ndarray) -> float:
    """BV/TV: bone volume / total volume (dimensionless)."""
    bv = int(mask.sum())
    tv = int(mask.size)
    return (bv / tv) if tv > 0 else 0.0


def compute_tb_th(mask: np.ndarray, voxel_size_mm: float) -> float:
    """Mean trabecular thickness (mm) via local thickness in bone."""
    bone = mask.astype(bool)
    if not bone.any():
        return 0.0
    lt = ps.filters.local_thickness(im=bone, method="dt")
    vals = lt[bone].astype(np.float64) * voxel_size_mm
    return float(vals.mean()) if vals.size > 0 else 0.0


def compute_tb_sp(mask: np.ndarray, voxel_size_mm: float) -> float:
    """Mean trabecular spacing (mm) via local thickness in void."""
    bone = mask.astype(bool)
    void = ~bone
    if not void.any():
        return 0.0
    lt_void = ps.filters.local_thickness(im=void, method="dt")
    vals = lt_void[void].astype(np.float64) * voxel_size_mm
    return float(vals.mean()) if vals.size > 0 else 0.0


def compute_tb_n(bvtv: float, tb_th: float) -> float:
    """Tb.N from BV/TV and Tb.Th, using standard relation: Tb.N ≈ BV/TV / Tb.Th (1/mm)."""
    if tb_th <= 0.0:
        return 0.0
    return float(bvtv / tb_th)


def compute_ct_th(cort_mask: np.ndarray, voxel_size_mm: float) -> float:
    """Approximate mean cortical thickness (mm) using local thickness in cortex."""
    cortex = cort_mask.astype(bool)
    if not cortex.any():
        return 0.0
    lt = ps.filters.local_thickness(im=cortex, method="dt")
    vals = lt[cortex].astype(np.float64) * voxel_size_mm
    return float(vals.mean()) if vals.size > 0 else 0.0


def compute_ct_areas(cort_mask: np.ndarray, total_mask: np.ndarray, voxel_size_mm: float):
    """
    Compute Ct.Ar, Tt.Ar, and Ct.Ar/Tt.Ar as volume-based surrogates:
    - Ct.Ar = cortical volume / length
    - Tt.Ar = total cross-sectional volume / length
    - Ratio = Ct.Ar / Tt.Ar
    """
    z = cort_mask.shape[0]
    length_mm = z * voxel_size_mm

    vox_vol = voxel_size_mm ** 3
    ct_vol_mm3 = int(cort_mask.sum()) * vox_vol
    tt_vol_mm3 = int(total_mask.sum()) * vox_vol

    if length_mm <= 0:
        return 0.0, 0.0, 0.0

    ct_ar = ct_vol_mm3 / length_mm
    tt_ar = tt_vol_mm3 / length_mm
    ratio = (ct_ar / tt_ar) if tt_ar > 0 else 0.0
    return float(ct_ar), float(tt_ar), float(ratio)


def find_auto_seg_folder(side_dir: Path, specimen_id: str):
    """Find folder containing semi-auto masks, same logic as updated M01."""
    versioned = sorted(side_dir.glob("segmentation_final_V*"))
    candidates = versioned if versioned else [side_dir / "segmentation_final"]

    for seg_root in reversed(candidates):
        if not seg_root.is_dir():
            continue

        possible_folders = [seg_root / "Segmentation", seg_root]
        for folder in possible_folders:
            cort = folder / f"{specimen_id}_cortical_mask_final.tiff"
            trab = folder / f"{specimen_id}_trabecular_mask_final.tiff"
            if cort.is_file() and trab.is_file():
                return folder

    return None


def process_side_for_morphometry(pair_folder: Path, specimen_base: str, side: str):
    side = side.upper()
    specimen_id = f"{specimen_base}{side}"
    side_dir = pair_folder / specimen_id

    print(f"\n[INFO] Morphometry for {specimen_id}")
    print(f" Side folder: {side_dir}")

    if not side_dir.is_dir():
        print(f" [SKIP] Missing specimen folder: {side_dir}")
        return

    seg_folder = find_auto_seg_folder(side_dir, specimen_id)
    if seg_folder is None:
        print(f" [SKIP] Could not find semi-auto masks for {specimen_id}")
        print("        Looked in segmentation_final_V*/Segmentation, segmentation_final_V*,")
        print("        segmentation_final/Segmentation, and segmentation_final")
        return

    auto_cort_path = seg_folder / f"{specimen_id}_cortical_mask_final.tiff"
    auto_trab_path = seg_folder / f"{specimen_id}_trabecular_mask_final.tiff"

    man_root = side_dir / "segmentation_manual"
    cort_folder = man_root / f"{specimen_id}_cortical_tiff"
    trab_folder = man_root / f"{specimen_id}_trabecular_tiff"
    man_cort_path = cort_folder / CORT_MAN_NAME.format(specimen_id=specimen_id)
    man_trab_path = trab_folder / TRAB_MAN_NAME.format(specimen_id=specimen_id)

    if not man_cort_path.is_file() or not man_trab_path.is_file():
        print(f" [SKIP] Missing manual full volumes for {specimen_id}")
        print(f"        Expected: {man_cort_path}")
        print(f"        Expected: {man_trab_path}")
        return

    print(f" Semi-auto cortical   : {auto_cort_path}")
    print(f" Semi-auto trabecular : {auto_trab_path}")
    print(f" Manual cortical      : {man_cort_path}")
    print(f" Manual trabecular    : {man_trab_path}")

    auto_cort = read_mask(auto_cort_path)
    auto_trab = read_mask(auto_trab_path)
    man_cort = read_mask(man_cort_path)
    man_trab = read_mask(man_trab_path)

    if auto_cort.shape != man_cort.shape or auto_trab.shape != man_trab.shape:
        print(" [WARN] Shape mismatch between manual and semi-auto masks; skipping.")
        return

    auto_total = ((auto_cort > 0) | (auto_trab > 0)).astype(np.uint8)
    man_total = ((man_cort > 0) | (man_trab > 0)).astype(np.uint8)

    rows = []

    def add_row(method: str, cort_mask: np.ndarray, trab_mask: np.ndarray, total_mask: np.ndarray):
        tb_mask = trab_mask.astype(np.uint8)
        bvtv = compute_bvtv(tb_mask)
        tb_th = compute_tb_th(tb_mask, VOXEL_SIZE_MM)
        tb_sp = compute_tb_sp(tb_mask, VOXEL_SIZE_MM)
        tb_n = compute_tb_n(bvtv, tb_th)

        ct_mask = cort_mask.astype(np.uint8)
        ct_th = compute_ct_th(ct_mask, VOXEL_SIZE_MM)
        ct_ar, tt_ar, ratio = compute_ct_areas(ct_mask, total_mask.astype(np.uint8), VOXEL_SIZE_MM)

        rows.append({
            "specimen_id": specimen_id,
            "method": method,
            "compartment": "global",
            "BVTV_trab": bvtv,
            "TbTh_trab": tb_th,
            "TbSp_trab": tb_sp,
            "TbN_trab": tb_n,
            "CtTh_cort": ct_th,
            "CtAr_cort": ct_ar,
            "CtAr_TtAr": ratio,
            "TtAr_total": tt_ar,
        })

    add_row("manual", man_cort, man_trab, man_total)
    add_row("semi_auto", auto_cort, auto_trab, auto_total)

    out_dir = side_dir / "PaperComparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{specimen_id}_global_morphometry.csv"

    df = pd.DataFrame(rows, columns=[
        "specimen_id",
        "method",
        "compartment",
        "BVTV_trab",
        "TbTh_trab",
        "TbSp_trab",
        "TbN_trab",
        "CtTh_cort",
        "CtAr_cort",
        "CtAr_TtAr",
        "TtAr_total",
    ])
    df.to_csv(out_csv, index=False)
    print(f" → Saved morphometry to {out_csv}")


def main():
    for base_name in SPECIMEN_BASES:
        pair_folder = BASE_DIR / base_name
        if not pair_folder.is_dir():
            print(f"[SKIP] Missing pair folder: {pair_folder}")
            continue

        specimen_base = "".join(ch for ch in base_name if ch.isdigit())
        for side in ("L", "R"):
            process_side_for_morphometry(pair_folder, specimen_base, side)


if __name__ == "__main__":
    main()