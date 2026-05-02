# script01_calibration_js.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from sklearn.linear_model import LinearRegression
from scipy import ndimage as ndi

from config_js import TRAB_MIN_DEFAULT
from utils_io_js import (
    make_ids_and_paths,
    make_versioned_seg_dir,
    load_tiff_stack,
    create_bone_mask,
)


def fit_calibration_from_three_means(excel_file: Path,
                                     phantom_cols,
                                     phantom_densities):
    df = pd.read_excel(excel_file, dtype=str)
    gray_means = []
    for col in phantom_cols:
        col_series = df[col].dropna().astype(str)
        mean_val = None
        for cell in col_series:
            s = cell.strip()
            if s.lower().startswith("mean:"):
                num_str = s.split(":", 1)[1].strip().replace(",", "")
                mean_val = float(num_str)
                break
        if mean_val is None:
            raise ValueError(f"No 'Mean:' entry found in column {col}")
        gray_means.append(mean_val)

    gray_arr = np.array(gray_means, dtype=np.float64).reshape(-1, 1)
    dens_arr = np.array(phantom_densities, dtype=np.float64)

    model = LinearRegression()
    model.fit(gray_arr, dens_arr)

    a = float(model.coef_[0])
    b = float(model.intercept_)
    print("Calibration points (gray -> mgHA):")
    for g, d in zip(gray_means, phantom_densities):
        print(f" {g:.2f} -> {d:.2f} mgHA")
    print(f"Fitted calibration: mgHA = {a:.6f} * gray + {b:.6f}")
    return a, b


def apply_calibration(volume: np.ndarray, a: float, b: float) -> np.ndarray:
    v = volume.astype(np.float32)
    mgha = a * v + b
    return mgha.astype(np.float32)


def keep_largest_component_3d(mask: np.ndarray, connectivity: int = 3) -> np.ndarray:
    mask_bool = mask.astype(bool)
    struct = ndi.generate_binary_structure(3, connectivity)
    labeled, n = ndi.label(mask_bool, structure=struct)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask_bool, labeled, index=range(1, n + 1))
    largest_label = int(1 + np.argmax(sizes))
    largest = (labeled == largest_label)
    return largest.astype(np.uint8)


def run_script_01(specimen_base: str, side: str, bone_min_mgha: float = 200.0):
    from config_js import PROJECT_ROOT  # just for logging

    specimen_id, pair_folder, abs_pair_path, abs_side_path = make_ids_and_paths(specimen_base, side)

    ct_folder = abs_side_path / f"{specimen_id}_Tiff"
    excel_file = abs_pair_path / f"CoversionValues_{specimen_base}.xlsx"
    out_dir = make_versioned_seg_dir(abs_side_path)
    print(f"Using segmentation output folder: {out_dir}")

    calibrated_tiff_name = f"{specimen_id}_calibrated_mgHA.tiff"
    bone_mask_tiff_name = f"{specimen_id}_bone_mask_binary.tiff"
    meta_json_name = f"{specimen_id}_calibration_meta.json"

    phantom_columns = ['Phantom1', 'Phantom2', 'Phantom3']
    phantom_densities_mgha = [1200.0, 800.0, 200.0]

    print("=== Script 01: calibration + bone mask ===")
    print(f"Specimen_base: {specimen_base}, side: {side}, specimen_id: {specimen_id}")
    print(f"CT folder: {ct_folder}")
    print(f"Excel file: {excel_file}")
    print(f"Output dir: {out_dir}")

    print("=== Load bone volume ===")
    vol = load_tiff_stack(ct_folder)
    print(f"Volume shape {vol.shape}, dtype={vol.dtype}")

    print("=== Apply 3D Gaussian filter (sigma=0.5) to gray-scale CT ===")
    vol_smoothed = ndi.gaussian_filter(vol, sigma=0.5)
    print(f"Smoothed volume shape {vol_smoothed.shape}, dtype={vol_smoothed.dtype}")

    print("=== Fit calibration from Excel ===")
    a, b = fit_calibration_from_three_means(
        excel_file,
        phantom_columns,
        phantom_densities_mgha
    )

    print("=== Apply calibration ===")
    mgha_vol = apply_calibration(vol_smoothed, a, b)
    print(f"mgHA range: [{mgha_vol.min():.2f}, {mgha_vol.max():.2f}]")

    hist, edges = np.histogram(mgha_vol[mgha_vol > 0], bins=100)
    print("Top 5 histogram peaks (mgHA):")
    peak_indices = hist.argsort()[-5:][::-1]
    for idx in peak_indices:
        center = 0.5 * (edges[idx] + edges[idx + 1])
        print(f" peak at ~{center:.1f} mgHA with count {hist[idx]}")

    calib_path = out_dir / calibrated_tiff_name
    print(f"Saving calibrated mgHA volume to: {calib_path}")
    imwrite(str(calib_path), mgha_vol, dtype=np.float32)

    print("=== Create binary bone mask for Script 2 ===")
    # bone_min_mgha is passed as parameter (default: 200.0 from TRAB_MIN_DEFAULT)
    bone_mask = create_bone_mask(mgha_vol, bone_min_mgha)


    print("=== Keep only largest 26-connected component ===")
    bone_mask = keep_largest_component_3d(bone_mask, connectivity=3)

    bone_vox = int(bone_mask.sum())
    total_vox = int(np.prod(bone_mask.shape))
    print(
        f"Bone voxels >= {bone_min_mgha:.1f} mgHA (largest 26-connected component): "
        f"{bone_vox} ({bone_vox / total_vox * 100:.2f} %)"
    )
    if bone_vox < 1e5:
        print("WARNING: very few bone voxels; check calibration/threshold for this specimen.")

    mask_path = out_dir / bone_mask_tiff_name
    print(f"Saving binary bone mask to: {mask_path}")
    imwrite(str(mask_path), (bone_mask * 255).astype(np.uint8))

    meta = {
        "calibration": {
            "a": a,
            "b": b,
            "excel_file": str(excel_file),
            "phantom_columns": phantom_columns,
            "phantom_densities_mgha": phantom_densities_mgha,
        },
        "bone_threshold": {"bone_min_mgha": bone_min_mgha},
        "inputs": {"ct_folder": str(ct_folder)},
        "outputs": {
            "calibrated_mgHA": str(calib_path),
            "bone_mask_binary": str(mask_path),
            "segmentation_dir": str(out_dir),
        },
        "gaussian_filter": {
            "applied": True,
            "sigma": 0.5,
        },
        "largest_component_filter": {
            "applied": True,
            "connectivity": 26,
        },
    }

    meta_path = out_dir / meta_json_name
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta to: {meta_path}")
    print("Done: smoothed grayscale calibrated volume + largest-component binary bone mask created.")