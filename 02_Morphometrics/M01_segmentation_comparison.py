#!/usr/bin/env python3
"""
M01_segmentation_comparison.py

Compare semi-automatic vs manual segmentation using full 3D manual volumes.

Expected folder structure:

BASE_DEV/
  <pair_folder> (e.g. 23162R&L)/
    <specimen_id> (e.g. 23162R)/
      segmentation_final_V*  OR  segmentation_final
        [optional] Segmentation/
          <specimen_id>_cortical_mask_final.tiff
          <specimen_id>_trabecular_mask_final.tiff
          <specimen_id>_bone_mask_binary.tiff
        OR directly inside segmentation_final*
          <specimen_id>_cortical_mask_final.tiff
          <specimen_id>_trabecular_mask_final.tiff
          <specimen_id>_bone_mask_binary.tiff
      segmentation_manual/
        <specimen_id>_cortical_tiff/
          <specimen_id>_cortical_manual_full.tiff
        <specimen_id>_trabecular_tiff/
          <specimen_id>_trabecular_manual_full.tiff
      PaperComparison/
        SegmentationComparison/
          (all outputs from this script)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from scipy import ndimage as ndi
from sklearn.metrics import cohen_kappa_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# EDIT THESE TO MATCH YOUR DATA LOCATION / NAMING
# -------------------------------------------------------------------

BASE_DEV = Path(r"C:\Users\jeichwal\Documents\A_DocTech\P1_Segmentation\Data")

SPECIMEN_BASES = [
    #"23162R&L",
    #"23167R&L",
    #"23168R&L",
    #"23169R&L",
    "23170R&L",
    "23175R&L",
]

CORT_FULL_NAME = "{specimen_id}_cortical_manual_full.tiff"
TRAB_FULL_NAME = "{specimen_id}_trabecular_manual_full.tiff"

DEFAULT_VOXEL_SIZE_UM = 32.0


# ------------------------ metrics helpers --------------------------

def voxel_metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    ref = ref.astype(bool).ravel()
    pred = pred.astype(bool).ravel()

    TP = int(np.sum(ref & pred))
    TN = int(np.sum(~ref & ~pred))
    FP = int(np.sum(~ref & pred))
    FN = int(np.sum(ref & ~pred))

    eps = 1e-9
    dsc = (2 * TP) / (2 * TP + FP + FN + eps)
    jaccard = TP / (TP + FP + FN + eps)
    sens = TP / (TP + FN + eps)
    spec = TN / (TN + FP + eps)
    prec = TP / (TP + FP + eps)
    f1 = (2 * prec * sens) / (prec + sens + eps)
    vol_sim = 1.0 - abs((TP + FN) - (TP + FP)) / ((TP + FN) + (TP + FP) + eps)
    fpe = FP / (TP + FP + eps)
    fne = FN / (TP + FN + eps)

    if (TP + TN + FP + FN) > 0 and (TP + FN) > 0 and (TP + FP) > 0:
        kappa = cohen_kappa_score(ref.astype(int), pred.astype(int))
    else:
        kappa = float("nan")

    return dict(
        TP=TP, TN=TN, FP=FP, FN=FN,
        DSC=float(dsc), Jaccard=float(jaccard),
        Sensitivity=float(sens), Specificity=float(spec),
        Precision=float(prec), F1=float(f1),
        VolumeSimilarity=float(vol_sim),
        FPE=float(fpe), FNE=float(fne),
        CohenKappa=float(kappa),
    )


def surface_from_mask(mask: np.ndarray) -> np.ndarray:
    struct = np.ones((3, 3, 3), dtype=bool)
    eroded = ndi.binary_erosion(mask.astype(bool), structure=struct)
    return mask.astype(bool) & ~eroded


def compute_distance_map(mask: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    surface = surface_from_mask(mask)
    dist_vox = ndi.distance_transform_edt(~surface)
    return dist_vox * voxel_size_mm


def surface_metrics(ref: np.ndarray, pred: np.ndarray, voxel_size_mm: float) -> dict:
    ref_bool = ref.astype(bool)
    pred_bool = pred.astype(bool)

    if not ref_bool.any() or not pred_bool.any():
        nan = float("nan")
        return dict(MSD_mm=nan, HD95_mm=nan, HD_max_mm=nan)

    ref_surf = surface_from_mask(ref_bool)
    pred_surf = surface_from_mask(pred_bool)

    if not ref_surf.any() or not pred_surf.any():
        nan = float("nan")
        return dict(MSD_mm=nan, HD95_mm=nan, HD_max_mm=nan)

    dist_pred_map = compute_distance_map(pred_bool, voxel_size_mm)
    d_ref_to_pred = dist_pred_map[ref_surf]

    dist_ref_map = compute_distance_map(ref_bool, voxel_size_mm)
    d_pred_to_ref = dist_ref_map[pred_surf]

    all_distances = np.concatenate([d_ref_to_pred, d_pred_to_ref])

    return dict(
        MSD_mm=float(np.mean(all_distances)),
        HD95_mm=float(np.percentile(all_distances, 95)),
        HD_max_mm=float(np.max(all_distances)),
    )


def slicewise_dsc(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    n_slices = ref.shape[0]
    dscs = np.zeros(n_slices, dtype=np.float32)

    for k in range(n_slices):
        r = ref[k].astype(bool).ravel()
        p = pred[k].astype(bool).ravel()
        tp = int(np.sum(r & p))
        denom = int(np.sum(r)) + int(np.sum(p))
        dscs[k] = (2 * tp / denom) if denom > 0 else float("nan")

    return dscs


def build_disagreement_volume(ref_cort, ref_trab, pred_cort, pred_trab):
    ref_label = np.zeros(ref_cort.shape, dtype=np.uint8)
    pred_label = np.zeros(pred_cort.shape, dtype=np.uint8)

    ref_label[ref_trab.astype(bool)] = 1
    ref_label[ref_cort.astype(bool)] = 2

    pred_label[pred_trab.astype(bool)] = 1
    pred_label[pred_cort.astype(bool)] = 2

    out = np.where(ref_label == pred_label, ref_label, np.uint8(3)).astype(np.uint8)
    return out, ref_label, pred_label


# ------------------------- path helpers ----------------------------

def find_auto_seg_folder(side_folder: Path, specimen_id: str) -> Path | None:
    """
    Find the folder that actually contains the automatic mask TIFFs.

    Supports both:
      specimen/segmentation_final/Segmentation/*.tiff
      specimen/segmentation_final/*.tiff
      specimen/segmentation_final_V*/Segmentation/*.tiff
      specimen/segmentation_final_V*/*.tiff
    """
    versioned = sorted(side_folder.glob("segmentation_final_V*"))
    candidates = versioned if versioned else [side_folder / "segmentation_final"]

    for seg_root in reversed(candidates):
        if not seg_root.is_dir():
            continue

        possible_folders = [
            seg_root / "Segmentation",
            seg_root,
        ]

        for folder in possible_folders:
            cortical_path = folder / f"{specimen_id}_cortical_mask_final.tiff"
            trabecular_path = folder / f"{specimen_id}_trabecular_mask_final.tiff"

            if cortical_path.is_file() and trabecular_path.is_file():
                return folder

    return None


# ------------------------- core logic ------------------------------

def process_side(pair_folder: Path, specimen_base: str, side: str, voxel_size_um: float):
    side = side.upper()
    specimen_id = f"{specimen_base}{side}"
    side_folder = pair_folder / specimen_id

    if not side_folder.is_dir():
        print(f"[SKIP] Missing specimen folder: {side_folder}")
        return

    paper_root = side_folder / "PaperComparison"
    paper_root.mkdir(parents=True, exist_ok=True)
    print(f"PaperComparison root: {paper_root}")

    seg_folder = find_auto_seg_folder(side_folder, specimen_id)
    if seg_folder is None:
        print(f"[SKIP] Could not find automatic masks for {specimen_id}")
        print("       Looked in segmentation_final_V*/Segmentation, segmentation_final_V*,")
        print("       segmentation_final/Segmentation, and segmentation_final")
        return

    cortical_path = seg_folder / f"{specimen_id}_cortical_mask_final.tiff"
    trabecular_path = seg_folder / f"{specimen_id}_trabecular_mask_final.tiff"
    binary_path = seg_folder / f"{specimen_id}_bone_mask_binary.tiff"

    # Check if binary mask exists
    if not binary_path.is_file():
        print(f"[WARN] Missing binary mask: {binary_path}")
        print(f"       Will skip binary mask comparisons for {specimen_id}")
        binary_exists = False
    else:
        binary_exists = True

    man_root = side_folder / "segmentation_manual"
    cort_folder = man_root / f"{specimen_id}_cortical_tiff"
    trab_folder = man_root / f"{specimen_id}_trabecular_tiff"

    man_cort_path = cort_folder / CORT_FULL_NAME.format(specimen_id=specimen_id)
    man_trab_path = trab_folder / TRAB_FULL_NAME.format(specimen_id=specimen_id)

    if not man_cort_path.is_file() or not man_trab_path.is_file():
        print(f"[SKIP] Missing manual full volumes for {specimen_id}")
        print(f"       Expected: {man_cort_path}")
        print(f"       Expected: {man_trab_path}")
        return

    print(f"\n[INFO] Comparing {specimen_id}")
    print(f"  Auto seg folder : {seg_folder}")
    print(f"  Auto cortical   : {cortical_path}")
    print(f"  Auto trabecular : {trabecular_path}")
    if binary_exists:
        print(f"  Binary mask     : {binary_path}")
    print(f"  Man cortical    : {man_cort_path}")
    print(f"  Man trabecular  : {man_trab_path}")

    auto_cort = imread(str(cortical_path))
    auto_trab = imread(str(trabecular_path))
    man_cort = imread(str(man_cort_path))
    man_trab = imread(str(man_trab_path))

    if binary_exists:
        binary_mask = imread(str(binary_path))

    if auto_cort.shape != man_cort.shape:
        print(f"[WARN] Shape mismatch cortical {auto_cort.shape} vs {man_cort.shape}")
        return

    if auto_trab.shape != man_trab.shape:
        print(f"[WARN] Shape mismatch trabecular {auto_trab.shape} vs {man_trab.shape}")
        return

    if binary_exists and binary_mask.shape != auto_cort.shape:
        print(f"[WARN] Shape mismatch binary mask {binary_mask.shape} vs {auto_cort.shape}")
        binary_exists = False

    auto_cort_b = auto_cort > 0
    auto_trab_b = auto_trab > 0
    man_cort_b = man_cort > 0
    man_trab_b = man_trab > 0

    if binary_exists:
        binary_mask_b = binary_mask > 0

    voxel_size_mm = voxel_size_um / 1000.0

    out_dir = paper_root / "SegmentationComparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Comparison outputs → {out_dir}")

    # Original comparisons: manual vs automated
    vm_cort = voxel_metrics(man_cort_b, auto_cort_b)
    vm_trab = voxel_metrics(man_trab_b, auto_trab_b)

    sm_cort = surface_metrics(man_cort_b, auto_cort_b, voxel_size_mm)
    sm_trab = surface_metrics(man_trab_b, auto_trab_b, voxel_size_mm)

    # NEW: Combined segmentation comparisons vs binary mask
    if binary_exists:
        # Automated combined (cortical OR trabecular)
        auto_combined_b = auto_cort_b | auto_trab_b
        vm_auto_combined = voxel_metrics(binary_mask_b, auto_combined_b)
        sm_auto_combined = surface_metrics(binary_mask_b, auto_combined_b, voxel_size_mm)

        # Manual combined (cortical OR trabecular)
        man_combined_b = man_cort_b | man_trab_b
        vm_man_combined = voxel_metrics(binary_mask_b, man_combined_b)
        sm_man_combined = surface_metrics(binary_mask_b, man_combined_b, voxel_size_mm)

    dsc_cort = slicewise_dsc(man_cort_b, auto_cort_b)
    dsc_trab = slicewise_dsc(man_trab_b, auto_trab_b)

    pd.DataFrame({
        "slice": np.arange(len(dsc_cort)),
        "DSC_cortical": dsc_cort
    }).to_csv(out_dir / f"{specimen_id}_cortical_slicewise_dsc.csv", index=False)

    pd.DataFrame({
        "slice": np.arange(len(dsc_trab)),
        "DSC_trabecular": dsc_trab
    }).to_csv(out_dir / f"{specimen_id}_trabecular_slicewise_dsc.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    idx = np.arange(len(dsc_cort))

    axes[0].plot(idx, dsc_cort, "b-", lw=0.8)
    axes[0].axhline(np.nanmean(dsc_cort), color="r", ls="--")
    axes[0].set_ylabel("DSC")
    axes[0].set_title(f"{specimen_id} cortical slice-wise DSC")

    axes[1].plot(idx, dsc_trab, "g-", lw=0.8)
    axes[1].axhline(np.nanmean(dsc_trab), color="r", ls="--")
    axes[1].set_ylabel("DSC")
    axes[1].set_xlabel("Slice index")
    axes[1].set_title(f"{specimen_id} trabecular slice-wise DSC")

    plt.tight_layout()
    plt.savefig(out_dir / f"{specimen_id}_slicewise_dsc.png", dpi=150)
    plt.close()

    disagree_vol, _, _ = build_disagreement_volume(
        man_cort_b, man_trab_b, auto_cort_b, auto_trab_b
    )
    imwrite(
        out_dir / f"{specimen_id}_disagreement_volume.tiff",
        disagree_vol.astype(np.uint8)
    )

    with open(out_dir / f"{specimen_id}_voxel_surface_metrics.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MANUAL vs AUTOMATED SEGMENTATION COMPARISON\n")
        f.write("=" * 70 + "\n\n")

        f.write("Cortical voxel metrics (Manual as reference):\n")
        for k, v in vm_cort.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTrabecular voxel metrics (Manual as reference):\n")
        for k, v in vm_trab.items():
            f.write(f"{k}: {v}\n")

        f.write("\nCortical surface metrics (Manual as reference):\n")
        for k, v in sm_cort.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTrabecular surface metrics (Manual as reference):\n")
        for k, v in sm_trab.items():
            f.write(f"{k}: {v}\n")

        if binary_exists:
            f.write("\n" + "=" * 70 + "\n")
            f.write("COMBINED SEGMENTATION vs ORIGINAL BINARY MASK COMPARISON\n")
            f.write("=" * 70 + "\n\n")

            f.write("Automated combined (cortical+trabecular) vs original binary mask:\n")
            f.write(f"  Original binary mask voxel count: {int(np.sum(binary_mask_b))}\n")
            f.write(f"  Automated combined voxel count:  {int(np.sum(auto_combined_b))}\n")
            f.write(f"  Voxel count difference:           {int(np.sum(auto_combined_b)) - int(np.sum(binary_mask_b))}\n\n")
            for k, v in vm_auto_combined.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n  Surface metrics:\n")
            for k, v in sm_auto_combined.items():
                f.write(f"  {k}: {v}\n")

            f.write("\nManual combined (cortical+trabecular) vs original binary mask:\n")
            f.write(f"  Original binary mask voxel count: {int(np.sum(binary_mask_b))}\n")
            f.write(f"  Manual combined voxel count:      {int(np.sum(man_combined_b))}\n")
            f.write(f"  Voxel count difference:           {int(np.sum(man_combined_b)) - int(np.sum(binary_mask_b))}\n\n")
            for k, v in vm_man_combined.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n  Surface metrics:\n")
            for k, v in sm_man_combined.items():
                f.write(f"  {k}: {v}\n")

    print(f"  → Saved metrics to {out_dir}")


def main(voxel_size_um: float):
    for base_name in SPECIMEN_BASES:
        pair_folder = BASE_DEV / base_name
        if not pair_folder.is_dir():
            print(f"[SKIP] Missing pair folder: {pair_folder}")
            continue

        specimen_base = "".join(ch for ch in base_name if ch.isdigit())
        print(f"\n=== Pair folder: {pair_folder.name} (base {specimen_base}) ===")

        for side in ("L", "R"):
            process_side(pair_folder, specimen_base, side, voxel_size_um)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel_size_um", type=float, default=DEFAULT_VOXEL_SIZE_UM)
    args = parser.parse_args()
    main(args.voxel_size_um)
