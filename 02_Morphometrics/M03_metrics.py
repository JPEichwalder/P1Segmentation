#!/usr/bin/env python3
"""
M03_Metrics.py

Aggregate per-specimen voxel/surface metrics (M01) and global
morphometric metrics (M02) into:

1) Per-specimen summary CSV:
   .../PaperComparison/SegmentationComparison/{specimen_id}_summary.csv

2) Master CSV across all specimens:
   BASE_DEV/all_specimens_segmentation_summary.csv

Extended to also include voxel counts from the final masks:
   - Original binary mask voxel count_trabecular
   - Automated combined voxel count_trabecular
   - Automated combined voxel count_cortical
   - Manual combined voxel count_trabecular
   - Manual combined voxel count_cortical
"""

from pathlib import Path
import pandas as pd
import numpy as np
import imageio.v3 as iio  # for reading binary masks


# ----------------- config (match other scripts) -----------------

BASE_DEV = Path(r"C:\Users\jeichwal\Documents\A_DocTech\P1_Segmentation\Data")

# pair folders to process (same list as M01/M02)
SPECIMEN_BASES = [
    "23162R&L",
    "23167R&L",
    "23168R&L",
    "23169R&L",
    "23170R&L",
    "23175R&L",
]


# ----------------- helpers -----------------

def parse_voxel_surface_metrics(txt_path: Path) -> dict:
    """
    Parse *_voxel_surface_metrics.txt into a flat dict with keys like:
    DSC_cortical, DSC_trabecular, HD95_mm_cortical, HD95_mm_trabecular, ...
    Cortical/trabecular and voxel/surface blocks are distinguished
    by the section headers in the text file.
    """
    data = {}
    current_block = None

    with txt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # detect block headers
            if line.startswith("Cortical voxel metrics"):
                current_block = "cort_vox"
                continue
            if line.startswith("Trabecular voxel metrics"):
                current_block = "trab_vox"
                continue
            if line.startswith("Cortical surface metrics"):
                current_block = "cort_surf"
                continue
            if line.startswith("Trabecular surface metrics"):
                current_block = "trab_surf"
                continue

            if ":" in line and current_block is not None:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    v = float(val)
                except ValueError:
                    continue

                # suffix to encode cortical/trabecular
                if current_block in ("cort_vox", "cort_surf"):
                    suffix = "cortical"
                else:
                    suffix = "trabecular"

                col_name = f"{key}_{suffix}"
                data[col_name] = v

    return data


def count_nonzero_voxels(mask_path: Path) -> int | None:
    """
    Read a binary mask image and return the number of non-zero voxels.
    Returns None if the file does not exist.
    """
    if not mask_path.is_file():
        print(f"[WARN] Missing mask: {mask_path}")
        return None
    arr = iio.imread(mask_path)
    return int(np.count_nonzero(arr))


# ----------------- main aggregation -----------------

def main():
    all_rows = []

    for base_name in SPECIMEN_BASES:
        pair_folder = BASE_DEV / base_name
        if not pair_folder.is_dir():
            print(f"[SKIP] Missing pair folder: {pair_folder}")
            continue

        # specimen_base = digits only (e.g. "23162" from "23162R&L")
        specimen_base = "".join(ch for ch in base_name if ch.isdigit())

        for side in ("L", "R"):
            specimen_id = f"{specimen_base}{side}"
            side_dir = pair_folder / specimen_id
            if not side_dir.is_dir():
                continue

            pc_root = side_dir / "PaperComparison"
            segcomp_dir = pc_root / "SegmentationComparison"
            morpho_csv = pc_root / f"{specimen_id}_global_morphometry.csv"
            metrics_txt = segcomp_dir / f"{specimen_id}_voxel_surface_metrics.txt"

            if not morpho_csv.is_file() or not metrics_txt.is_file():
                print(f"[SKIP] Missing inputs for {specimen_id}")
                continue

            print(f"[INFO] Aggregating {specimen_id}")

            # read morphometry (manual + semi_auto)
            df_morph = pd.read_csv(morpho_csv)

            # split manual vs semi_auto rows
            row_manual = df_morph[df_morph["method"] == "manual"].iloc[0]
            row_auto = df_morph[df_morph["method"] == "semi_auto"].iloc[0]

            # parse voxel/surface metrics (pairwise manual vs semi_auto)
            vs_dict = parse_voxel_surface_metrics(metrics_txt)

            # ------------------------------------------------------------------
            # New: voxel counts from final masks
            # ------------------------------------------------------------------
            seg_final_dir = side_dir / "segmentation_final"
            seg_manual_dir = side_dir / "segmentation_manual"

            # automatic masks
            auto_trab_mask = seg_final_dir / f"{specimen_id}_trabecular_mask_final.tiff"
            auto_cort_mask = seg_final_dir / f"{specimen_id}_cortical_mask_final.tiff"
            ct_bone_mask = seg_final_dir / f"{specimen_id}_bone_mask_binary.tiff"

            # manual masks
            man_cort_mask = (
                seg_manual_dir
                / f"{specimen_id}_cortical_tiff"
                / f"{specimen_id}_cortical_manual_full.tiff"
            )
            man_trab_mask = (
                seg_manual_dir
                / f"{specimen_id}_trabecular_tiff"
                / f"{specimen_id}_trabecular_manual_full.tiff"
            )

            orig_trab_vox = count_nonzero_voxels(ct_bone_mask)
            auto_trab_vox = count_nonzero_voxels(auto_trab_mask)
            auto_cort_vox = count_nonzero_voxels(auto_cort_mask)
            man_trab_vox = count_nonzero_voxels(man_trab_mask)
            man_cort_vox = count_nonzero_voxels(man_cort_mask)

            voxel_counts = {
                "Original binary mask voxel count_trabecular": orig_trab_vox,
                "Automated combined voxel count_trabecular": auto_trab_vox,
                "Automated combined voxel count_cortical": auto_cort_vox,
                "Manual combined voxel count_trabecular": man_trab_vox,
                "Manual combined voxel count_cortical": man_cort_vox,
            }

            # build one combined row for this specimen side
            out_row = {
                "specimen_id": specimen_id,
                "compartment": row_manual.get("compartment", "global"),

                # manual morphometry
                "BVTV_trab_manual": row_manual.get("BVTV_trab", None),
                "TbTh_trab_manual": row_manual.get("TbTh_trab", None),
                "TbSp_trab_manual": row_manual.get("TbSp_trab", None),
                "TbN_trab_manual": row_manual.get("TbN_trab", None),
                "CtTh_cort_manual": row_manual.get("CtTh_cort", None),
                "CtAr_cort_manual": row_manual.get("CtAr_cort", None),
                "CtAr_TtAr_manual": row_manual.get("CtAr_TtAr", None),
                "TtAr_total_manual": row_manual.get("TtAr_total", None),

                # semi-auto morphometry
                "BVTV_trab_auto": row_auto.get("BVTV_trab", None),
                "TbTh_trab_auto": row_auto.get("TbTh_trab", None),
                "TbSp_trab_auto": row_auto.get("TbSp_trab", None),
                "TbN_trab_auto": row_auto.get("TbN_trab", None),
                "CtTh_cort_auto": row_auto.get("CtTh_cort", None),
                "CtAr_cort_auto": row_auto.get("CtAr_cort", None),
                "CtAr_TtAr_auto": row_auto.get("CtAr_TtAr", None),
                "TtAr_total_auto": row_auto.get("TtAr_total", None),
            }

            # add pairwise segmentation metrics
            out_row.update(vs_dict)

            # add voxel counts
            out_row.update(voxel_counts)

            # per-specimen summary CSV (now one row per specimen_id)
            segcomp_dir.mkdir(parents=True, exist_ok=True)
            summary_df = pd.DataFrame([out_row])
            summary_path = segcomp_dir / f"{specimen_id}_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f" → Saved per-specimen summary to {summary_path}")

            all_rows.append(out_row)

    if all_rows:
        master_df = pd.DataFrame(all_rows)
        out_master = BASE_DEV / "all_specimens_segmentation_summary.csv"
        master_df.to_csv(out_master, index=False)
        print(f"[DONE] Master summary saved to {out_master}")
    else:
        print("[WARN] No rows aggregated; check that M01 and M02 have been run.")


if __name__ == "__main__":
    main()