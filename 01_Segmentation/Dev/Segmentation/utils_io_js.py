# utils_io_js.py

import json
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite
from scipy import ndimage as ndi

from config_js import PROJECT_ROOT


def make_ids_and_paths(specimen_base: str, side: str):
    side = side.upper()
    if side not in ("L", "R"):
        raise ValueError("SIDE must be 'L' or 'R'")

    specimen_id = f"{specimen_base}{side}"
    pair_folder = f"{specimen_base}R&L"
    abs_pair_path = PROJECT_ROOT / pair_folder
    abs_side_path = abs_pair_path / specimen_id
    return specimen_id, pair_folder, abs_pair_path, abs_side_path


def make_versioned_seg_dir(abs_side_path: Path) -> Path:
    """
    Return a segmentation_final directory with automatic versioning:
    segmentation_final, segmentation_final_V1, segmentation_final_V2, ...
    """
    base = abs_side_path / "segmentation_final"
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base

    i = 1
    while True:
        cand = abs_side_path / f"segmentation_final_V{i}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        i += 1


def find_meta(abs_side_path: Path, specimen_id: str) -> Path:
    # 1) Collect all versioned folders that contain the meta file
    versioned = []
    i = 1
    while True:
        cand_dir = abs_side_path / f"segmentation_final_V{i}"
        cand_meta = cand_dir / f"{specimen_id}_calibration_meta.json"
        if cand_meta.is_file():
            versioned.append(cand_meta)
        if not cand_dir.exists():
            break
        i += 1

    # If any versioned meta exists, return the last (highest Vn)
    if versioned:
        return versioned[-1]

    # 2) Fall back to unversioned folder
    base_meta = abs_side_path / "segmentation_final" / f"{specimen_id}_calibration_meta.json"
    if base_meta.is_file():
        return base_meta

    raise FileNotFoundError(
        f"Calibration meta not found in any segmentation_final* folder for {specimen_id}"
    )


def load_tiff_stack(folder: Path) -> np.ndarray:
    tiffs = sorted([f for f in folder.iterdir()
                    if f.suffix.lower() in ('.tif', '.tiff')])
    if not tiffs:
        raise FileNotFoundError(f"No TIFF files in {folder}")
    if len(tiffs) == 1:
        vol = imread(str(tiffs[0]))
        if vol.ndim == 2:
            vol = vol[None, ...]
        return vol
    slices = [imread(str(f)) for f in tiffs]
    return np.stack(slices, axis=0)


def create_bone_mask(mgha_vol: np.ndarray, bone_min_mgha: float) -> np.ndarray:
    mask = (mgha_vol >= bone_min_mgha)
    return mask.astype(np.uint8)


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
