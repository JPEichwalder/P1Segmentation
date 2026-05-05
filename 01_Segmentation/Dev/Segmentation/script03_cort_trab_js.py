#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite
from scipy import ndimage as ndi

from config_js import (
    TRAB_MIN_DEFAULT,
    CORT_MIN_DEFAULT,
    CORT_MAX_DEFAULT,
    WRITE_LABEL_VOLUME,
    BAND_DEPTH_MM,
)
from utils_io_js import make_ids_and_paths, find_meta

TRAB_MIN = TRAB_MIN_DEFAULT
CORT_MIN = CORT_MIN_DEFAULT
CORT_MAX = CORT_MAX_DEFAULT


def largest_component(mask: np.ndarray) -> np.ndarray:
    struct_26 = np.ones((3, 3, 3), dtype=bool)
    labeled, n = ndi.label(mask, structure=struct_26)
    if n == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == counts.argmax()


def keep_components_min_size(mask: np.ndarray, min_size: int, structure=None) -> np.ndarray:
    if structure is None:
        structure = ndi.generate_binary_structure(3, 1)
    labeled, n = ndi.label(mask, structure=structure)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = np.where(sizes >= min_size)[0]
    if keep.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, keep)


def identify_cortical_shell(
    cortex_candidate: np.ndarray,
    solid_bool: np.ndarray,
    min_component_size: int = 3000,
) -> np.ndarray:
    struct_6 = ndi.generate_binary_structure(3, 1)
    labeled, n = ndi.label(cortex_candidate, structure=struct_6)

    if n == 0:
        return cortex_candidate & solid_bool

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep_labels = np.where(sizes >= min_component_size)[0]

    if keep_labels.size == 0:
        cortical_shell = labeled == sizes.argmax()
        print(f"No cortical components >= {min_component_size} voxels; keeping largest only.")
    else:
        cortical_shell = np.isin(labeled, keep_labels)

    print(f"Connected cortical components found: {n}")
    print(f"Keeping {len(keep_labels)} cortical components >= {min_component_size} voxels")
    print(f"Final cortical shell voxels kept: {int(cortical_shell.sum())}")
    return cortical_shell & solid_bool


def masked_uniform_mean(values: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    values = values.astype(np.float32)
    mask_f = mask.astype(np.float32)
    num = ndi.uniform_filter(values * mask_f, size=size, mode="nearest")
    den = ndi.uniform_filter(mask_f, size=size, mode="nearest")
    out = np.zeros_like(values, dtype=np.float32)
    valid = den > 1e-6
    out[valid] = num[valid] / den[valid]
    return out


def run_script_03(
    specimen_base: str,
    side: str,
    trab_min: float = 200.0,
    cort_min: float = 600.0,
    cort_max: float = 1200.0,
    min_trab_cluster_size: int = 300,
    min_band_trab_size: int = 50,
    medullary_dilation_iterations: int = 2,
    min_cortex_voxels: int = 3000,
    pore_min_size: int = 20,
    fill_unassigned: bool = False,
):
    specimen_id, pair_folder, abs_pair_path, abs_side_path = make_ids_and_paths(
        specimen_base, side
    )

    meta_path = find_meta(abs_side_path, specimen_id)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    out_dir = Path(meta["outputs"]["segmentation_dir"])
    print(f"Using segmentation output folder: {out_dir}")

    input_mgha = out_dir / f"{specimen_id}_calibrated_mgHA.tiff"
    input_band = out_dir / f"{specimen_id}_cortical_band_mask.tiff"
    input_outer = out_dir / f"{specimen_id}_cortical_outer_ring.tiff"
    input_inner = out_dir / f"{specimen_id}_cortical_inner_ring.tiff"

    print("=== Script 03: reliable 2-output cortical/trabecular segmentation ===")
    print(f"Specimen: {specimen_id}")
    print(f"mgHA file: {input_mgha}")
    print(f"Band file: {input_band}")
    print(f"Outer ring: {input_outer}")
    print(f"Inner ring: {input_inner}")
    print(f"Output dir: {out_dir}")

    mg = imread(str(input_mgha)).astype(np.float32)
    band = imread(str(input_band)) > 0
    outer = imread(str(input_outer)) > 0
    inner = imread(str(input_inner)) > 0

    bone_mask = imread(str(out_dir / f"{specimen_id}_bone_mask_binary.tiff")) > 0
    solid_bool = imread(str(out_dir / f"{specimen_id}_bone_mask_solid_blob.tiff")) > 0

    if not (mg.shape == band.shape == outer.shape == inner.shape == bone_mask.shape == solid_bool.shape):
        raise ValueError("All input volumes must have identical shape.")

    struct_6 = ndi.generate_binary_structure(3, 1)

    print("\n--- Step 1: Initial density classes ---")
    high_cort = (mg >= cort_min) & (mg <= cort_max) & bone_mask
    high_trab = (mg >= trab_min) & (mg < cort_min) & bone_mask
    gray = bone_mask & (~high_cort) & (~high_trab)

    print(f"High-density cortical voxels: {int(high_cort.sum())}")
    print(f"Trabecular-density voxels: {int(high_trab.sum())}")
    print(f"Gray-zone voxels: {int(gray.sum())}")

    print("\n--- Step 2: Strong cortical and trabecular seeds ---")
    cortex_seed = (high_cort & band & solid_bool) | (high_cort & outer)
    cortex_seed = identify_cortical_shell(
        cortex_seed,
        solid_bool,
        min_component_size=min_cortex_voxels,
    )
    cortex_seed &= band

    trab_seed = (high_trab & (~band)) | (high_trab & inner) | (gray & (~band))
    trab_seed = keep_components_min_size(
        trab_seed,
        max(10, min_trab_cluster_size // 2),
        structure=struct_6,
    )
    trab_seed = largest_component(trab_seed)

    trab_seed_dil = ndi.binary_dilation(
        trab_seed,
        structure=struct_6,
        iterations=medullary_dilation_iterations,
    )
    cortex_seed_dil = ndi.binary_dilation(
        cortex_seed,
        structure=struct_6,
        iterations=2,
    )

    print(f"Cortical seed voxels: {int(cortex_seed.sum())}")
    print(f"Trabecular seed voxels: {int(trab_seed.sum())}")

    print("\n--- Step 3: Local band features for reliable assignment ---")
    local_mean_mg = masked_uniform_mean(mg, bone_mask, size=5)
    local_bone_frac = ndi.uniform_filter(bone_mask.astype(np.float32), size=5, mode="nearest")
    local_void_frac = 1.0 - local_bone_frac

    dist_to_surface = ndi.distance_transform_edt(solid_bool)
    norm_depth = np.zeros_like(mg, dtype=np.float32)
    if np.any(band):
        max_band_depth = dist_to_surface[band].max() + 1e-6
        norm_depth[band] = dist_to_surface[band] / max_band_depth

    print("Computed local mean mgHA, local bone fraction, local void fraction, and normalized band depth.")

    print("\n--- Step 4: Internal 3-class decision inside cortical band ---")
    compact_score = (
        1.8 * (local_mean_mg >= cort_min).astype(np.float32)
        + 0.9 * local_bone_frac
        + 0.7 * cortex_seed_dil.astype(np.float32)
        + 0.4 * outer.astype(np.float32)
        - 0.8 * trab_seed_dil.astype(np.float32)
        - 0.5 * local_void_frac
    )

    trab_score = (
        1.3 * (local_mean_mg < cort_min).astype(np.float32)
        + 0.9 * trab_seed_dil.astype(np.float32)
        + 0.5 * inner.astype(np.float32)
        + 0.5 * norm_depth
        + 0.6 * local_void_frac
        - 0.6 * cortex_seed_dil.astype(np.float32)
    )

    score_diff = compact_score - trab_score

    compact_band = band & bone_mask & (score_diff > 0.9)
    trab_band = band & bone_mask & (score_diff < -0.2)
    transition_band = band & bone_mask & (~compact_band) & (~trab_band)

    compact_band |= cortex_seed
    trab_band |= (high_trab & inner & (~compact_band))
    transition_band &= (~compact_band) & (~trab_band)

    print(f"Compact band voxels: {int(compact_band.sum())}")
    print(f"Transition band voxels: {int(transition_band.sum())}")
    print(f"Trabecular band voxels: {int(trab_band.sum())}")

    print("\n--- Step 5: Reassign transitional voxels reliably to final cortex or trabecula ---")
    transition_to_cortex = transition_band & (
        (local_mean_mg >= (0.85 * cort_min))
        & (local_bone_frac >= 0.55)
        & (cortex_seed_dil | (~trab_seed_dil))
    )

    remaining_transition = transition_band & (~transition_to_cortex)

    transition_to_trab = remaining_transition & (
        trab_seed_dil
        | (local_void_frac > 0.35)
        | (local_mean_mg < 0.85 * cort_min)
    )

    unresolved_transition = remaining_transition & (~transition_to_trab)

    fallback_to_cortex = unresolved_transition & (score_diff >= 0)
    fallback_to_trab = unresolved_transition & (~fallback_to_cortex)

    cortical_band_final = compact_band | transition_to_cortex | fallback_to_cortex
    trab_band_final = trab_band | transition_to_trab | fallback_to_trab

    print(f"Transition -> cortex: {int(transition_to_cortex.sum())}")
    print(f"Transition -> trabecula: {int(transition_to_trab.sum())}")
    print(f"Fallback -> cortex: {int(fallback_to_cortex.sum())}")
    print(f"Fallback -> trabecula: {int(fallback_to_trab.sum())}")

    print("\n--- Step 6: Assemble preliminary two compartments ---")
    trab_outside = ((high_trab | gray) & (~band) & bone_mask) | trab_seed
    trab_outside = keep_components_min_size(trab_outside, min_trab_cluster_size, structure=struct_6)

    cortex_final = cortical_band_final & band & solid_bool
    trab_final = (trab_outside | trab_band_final) & bone_mask
    trab_final &= (~cortex_final)

    if fill_unassigned:
        remaining_core = (~band) & bone_mask & (~cortex_final) & (~trab_final)
        trab_final[remaining_core] = True

    print(f"Cortical voxels before trabecular-region reassignment: {int(cortex_final.sum())}")
    print(f"Trabecular voxels before trabecular-region reassignment: {int(trab_final.sum())}")

    # -------------------------------------------------------------------------
    # NEW STEP 7: Reassign existing bone voxels in real trabecular region
    # (outside cortical band) to trabecula, without geometric growth.
    # -------------------------------------------------------------------------
    print("\n--- Step 7: Reassign existing bone voxels in real trabecular region ---")

    trab_region = (~band) & bone_mask & (~cortex_final)
    trab_existing = trab_final & trab_region

    trab_neighbor_count = ndi.convolve(
        trab_existing.astype(np.uint8),
        np.ones((3, 3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )

    trab_reassign = trab_region & (~trab_existing) & (
        (local_mean_mg >= trab_min * 0.9)
        | (trab_neighbor_count >= 4)
    )

    trab_final[trab_reassign] = True

    print(f"Reassigned existing medullary bone voxels to trabecula: {int(trab_reassign.sum())}")
    print(f"Trabecular voxels after real-trabecular-region reassignment: {int(trab_final.sum())}")

    print("\n--- Step 8: Final cleanup ---")
    cortex_final = keep_components_min_size(cortex_final, min_cortex_voxels, structure=struct_6)
    cortex_final = largest_component(cortex_final)

    trab_in_band = trab_final & band
    trab_out_band = trab_final & (~band)

    trab_in_band = keep_components_min_size(trab_in_band, min_band_trab_size, structure=struct_6)
    trab_out_band = keep_components_min_size(trab_out_band, min_trab_cluster_size, structure=struct_6)

    trab_final = trab_in_band | trab_out_band
    trab_final = largest_component(trab_final)

    cortex_final &= bone_mask
    trab_final &= bone_mask
    trab_final &= (~cortex_final)

    print(f"Cortical voxels final: {int(cortex_final.sum())}")
    print(f"Trabecular voxels final: {int(trab_final.sum())}")

    print("\n--- Step 9: Intracortical pore detection ---")
    intracortical_void = band & solid_bool & (~bone_mask)
    pore_labels, n_pores = ndi.label(intracortical_void, structure=struct_6)
    intracortical_pores = np.zeros_like(intracortical_void, dtype=bool)

    if n_pores > 0:
        pore_sizes = np.bincount(pore_labels.ravel())
        pore_sizes[0] = 0
        keep_pores = np.where(pore_sizes >= pore_min_size)[0]
        if keep_pores.size > 0:
            intracortical_pores = np.isin(pore_labels, keep_pores)

    print(f"Intracortical pore voxels detected: {int(intracortical_pores.sum())}")

    cortical_path = out_dir / f"{specimen_id}_cortical_mask_final.tiff"
    trabecular_path = out_dir / f"{specimen_id}_trabecular_mask_final.tiff"
    label_path = out_dir / f"{specimen_id}_bone_compartments_label.tiff"

    print(f"Saving cortical mask to: {cortical_path}")
    print(f"Saving trabecular mask to: {trabecular_path}")
    print(f"Saving label volume to: {label_path}")

    imwrite(str(cortical_path), (cortex_final * 255).astype(np.uint8))
    imwrite(str(trabecular_path), (trab_final * 255).astype(np.uint8))

    if WRITE_LABEL_VOLUME:
        labels = np.zeros_like(mg, dtype=np.uint8)
        labels[trab_final] = 1
        labels[cortex_final] = 2
        labels[intracortical_pores] = 3
        imwrite(str(label_path), labels.astype(np.uint8))

    all_parameters = {
        "specimen_id": specimen_id,
        "cortical_band_thickness_mm": BAND_DEPTH_MM,
        "trab_min_mgha": trab_min,
        "cort_min_mgha": cort_min,
        "cort_max_mgha": cort_max,
        "min_trab_cluster_size": min_trab_cluster_size,
        "min_band_trab_size": min_band_trab_size,
        "medullary_dilation_iterations": medullary_dilation_iterations,
        "min_cortex_voxels": min_cortex_voxels,
        "pore_min_size": pore_min_size,
        "fill_unassigned": fill_unassigned,
        "band_voxels": int(band.sum()),
        "bone_mask_voxels": int(bone_mask.sum()),
        "cortex_voxels_final": int(cortex_final.sum()),
        "trab_voxels_final": int(trab_final.sum()),
        "intracortical_pore_voxels": int(intracortical_pores.sum()),
        "method": "2-output segmentation with internal 3-class band assignment and label-only reassignment in real trabecular region",
    }

    json_path = out_dir / f"{specimen_id}_segmentation_params.json"
    with open(json_path, "w") as f:
        json.dump(all_parameters, f, indent=2)

    print(f"Saved JSON parameters to: {json_path}")
    print("=== Script 03 complete ===")