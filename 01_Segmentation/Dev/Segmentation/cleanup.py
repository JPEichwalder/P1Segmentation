#!/usr/bin/env python3
import shutil
from pathlib import Path
import json


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

    if versioned:
        return versioned[-1]

    # 2) Fall back to unversioned folder
    base_meta = abs_side_path / "segmentation_final" / f"{specimen_id}_calibration_meta.json"
    if base_meta.is_file():
        return base_meta

    raise FileNotFoundError(
        f"Calibration meta not found in any segmentation_final* folder for {specimen_id}"
    )


def main(base_dir: str, specimen_id: str):
    base = Path(base_dir)
    spec_dir = base / specimen_id

    # Use the same segmentation directory as the main pipeline
    meta_path = find_meta(spec_dir, specimen_id)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    seg_dir = Path(meta["outputs"]["segmentation_dir"])

    if not seg_dir.is_dir():
        raise FileNotFoundError(f"segmentation_final folder not found: {seg_dir}")

    seg_target = seg_dir / "Segmentation"
    add_target = seg_dir / "AdditionalFiles"
    seg_target.mkdir(parents=True, exist_ok=True)
    add_target.mkdir(parents=True, exist_ok=True)

    cortical_pattern = f"{specimen_id}_cortical_mask_final.tiff"
    trab_pattern = f"{specimen_id}_trabecular_mask_final.tiff"
    params_txt = f"{specimen_id}_segmentation_params.txt"
    params_json = f"{specimen_id}_segmentation_params.json"

    print("=== Cleanup: sorting output files ===")
    print(f"Source folder: {seg_dir}")
    print(f"Segmentation folder: {seg_target}")
    print(f"Additional files folder: {add_target}")

    for f in seg_dir.iterdir():
        if f.is_dir():
            if f.name not in ("Segmentation", "AdditionalFiles"):
                print(f"Skipping directory: {f.name}")
            continue

        name = f.name

        # Files that should live in Segmentation
        if (
            name == cortical_pattern
            or name == trab_pattern
            or name == params_txt
            or name == params_json
        ):
            dest = seg_target / name
        else:
            dest = add_target / name

        if dest == f:
            continue

        print(f"Moving {name} -> {dest.parent.name}")
        shutil.move(str(f), str(dest))

    print("Cleanup done.")