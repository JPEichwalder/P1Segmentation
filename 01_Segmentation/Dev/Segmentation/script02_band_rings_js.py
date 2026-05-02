# script02_band_rings_js.py

import json
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite
from scipy import ndimage as ndi

from config_js import (
    GROW_RADIUS_2D,
    VOXEL_SIZE_UM,
    BAND_DEPTH_MM,
    BAND_DEPTH_VOX,
    OUTER_RING_THICKNESS_PX,
    INNER_RING_THICKNESS_PX,
)
from utils_io_js import make_ids_and_paths, find_meta


def make_solid_blob(bone: np.ndarray, grow_radius_2d: int) -> np.ndarray:
    z, y, x = bone.shape
    solid = np.zeros_like(bone, dtype=np.uint8)

    struct2d = ndi.generate_binary_structure(2, 1)
    if grow_radius_2d > 0:
        struct2d = ndi.iterate_structure(struct2d, grow_radius_2d)

    print(f"=== Slice-wise outside flood fill (grow_radius_2d={grow_radius_2d}) ===")
    for k in range(z):
        sl = bone[k, :, :].astype(bool)

        if grow_radius_2d > 0 and sl.any():
            sl = ndi.binary_dilation(sl, structure=struct2d)

        bg = ~sl
        labeled_bg, n_bg = ndi.label(bg)
        if n_bg == 0:
            solid[k, :, :] = 1
        else:
            border_mask = np.zeros_like(bg, dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True

            border_labels = np.unique(labeled_bg[border_mask & bg])
            outside = np.isin(labeled_bg, border_labels)
            inside_holes = bg & (~outside)

            solid_slice = sl | inside_holes
            solid[k, :, :] = solid_slice.astype(np.uint8)

        if k % 20 == 0 or k == z - 1:
            print(f" Slice {k+1}/{z} processed")

    return solid


def run_script_02(
    specimen_base: str,
    side: str,
    voxel_um: float = 32.0,
    band_mm: float = 2.0,
    grow_radius: int = 20,
    outer_ring_thickness: int = 2,
    inner_ring_thickness: int = 2,
):
    specimen_id, pair_folder, abs_pair_path, abs_side_path = make_ids_and_paths(
        specimen_base, side
    )

    meta_path = find_meta(abs_side_path, specimen_id)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    seg_dir = Path(meta["outputs"]["segmentation_dir"])
    print(f"Using segmentation output folder: {seg_dir}")

    input_bone_mask = seg_dir / f"{specimen_id}_bone_mask_binary.tiff"

    print("=== Script 02: solid blob + band + rings ===")
    print(f"Specimen: {specimen_id}")
    print(f"Input bone mask: {input_bone_mask}")
    print(f"Output dir: {seg_dir}")

    print("=== Load binary bone mask ===")
    bone_raw = imread(str(input_bone_mask))
    bone = (bone_raw > 0).astype(np.uint8)
    print(f"Bone mask shape {bone.shape}, voxels: {int(bone.sum())}")

    solid = make_solid_blob(bone, grow_radius)
    print(f"Solid blob voxels: {int(solid.sum())}")

    blob_path = seg_dir / f"{specimen_id}_bone_mask_solid_blob.tiff"
    print(f"Saving solid blob mask to: {blob_path}")
    imwrite(str(blob_path), (solid * 255).astype(np.uint8))

    solid_bool = solid.astype(bool)

    print("=== Distance transform inside solid blob ===")
    dist = ndi.distance_transform_edt(solid_bool)
    inside_vals = dist[solid_bool]
    print(f"Distance stats (voxels): min={inside_vals.min():.2f}, max={inside_vals.max():.2f}")

    if BAND_DEPTH_VOX is not None:
        band_depth_vox = float(BAND_DEPTH_VOX)
    else:
        band_depth_vox = band_mm * 1000.0 / voxel_um

    print(f"Cortical band depth: {band_depth_vox:.2f} voxels (from surface, ≈ {BAND_DEPTH_MM} mm)")

    band_mask = (dist <= band_depth_vox) & solid_bool
    struct3d = ndi.generate_binary_structure(3, 1)
    band_mask = ndi.binary_dilation(band_mask, structure=struct3d)

    band_mask_u8 = band_mask.astype(np.uint8)
    print(f"Cortical band voxels: {int(band_mask_u8.sum())}")

    band_path = seg_dir / f"{specimen_id}_cortical_band_mask.tiff"
    print(f"Saving cortical band mask to: {band_path}")
    imwrite(str(band_path), (band_mask_u8 * 255).astype(np.uint8))

    z, y, x = solid.shape

    print(f"=== Create outer ring: {outer_ring_thickness} px in-plane ===")
    outer_ring = np.zeros_like(solid, dtype=np.uint8)
    struct2d_out = ndi.generate_binary_structure(2, 1)
    se2d_out = ndi.iterate_structure(struct2d_out, outer_ring_thickness)


    for k in range(z):
        sl = solid[k, :, :].astype(bool)
        if sl.any():
            eroded_sl = ndi.binary_erosion(sl, structure=se2d_out)
            ring_sl = sl & (~eroded_sl)
            outer_ring[k, :, :] = ring_sl.astype(np.uint8)
        if k % 20 == 0 or k == z - 1:
            print(f" Outer ring slice {k+1}/{z} done")

    print(f"Outer ring voxels: {int(outer_ring.sum())}")
    outer_path = seg_dir / f"{specimen_id}_cortical_outer_ring.tiff"
    print(f"Saving outer ring to: {outer_path}")
    imwrite(str(outer_path), (outer_ring * 255).astype(np.uint8))

    print(f"=== Create inner ring: {inner_ring_thickness} px around inner band boundary ===")
    inner_ring = np.zeros_like(solid, dtype=np.uint8)
    struct2d_in = ndi.generate_binary_structure(2, 1)
    se2d_in = ndi.iterate_structure(struct2d_in, inner_ring_thickness)


    core = (dist > band_depth_vox) & solid_bool

    for k in range(z):
        sl_core = core[k, :, :].astype(bool)
        if sl_core.any():
            eroded_core = ndi.binary_erosion(sl_core, structure=se2d_in)
            ring_sl = sl_core & (~eroded_core)
            inner_ring[k, :, :] = ring_sl.astype(np.uint8)
        if k % 20 == 0 or k == z - 1:
            print(f" Inner ring slice {k+1}/{z} done")

    print(f"Inner ring voxels: {int(inner_ring.sum())}")
    inner_path = seg_dir / f"{specimen_id}_cortical_inner_ring.tiff"
    print(f"Saving inner ring to: {inner_path}")
    imwrite(str(inner_path), (inner_ring * 255).astype(np.uint8))

    print("Done: solid blob, cortical band, and inner/outer rings saved.")
