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
    largest_label = counts.argmax()
    return labeled == largest_label


def auto_trab_cort_thresholds(
        mg: np.ndarray,
        bone_mask: np.ndarray,
        default_trab_min=200.0,
        default_cort_min=600.0,
        default_cort_max=1200.0,
):
    vals = mg[bone_mask > 0].ravel().astype(np.float32)
    if vals.size < 1000:
        print("Too few voxels for auto trab/cort thresholds, using defaults.")
        return default_trab_min, default_cort_min, default_cort_max

    hist, edges = np.histogram(vals, bins=256)
    mg_min, mg_max = 50.0, 2000.0
    mask = (edges[:-1] >= mg_min) & (edges[1:] <= mg_max)
    if not np.any(mask):
        print("No suitable histogram range for trab/cort; using defaults.")
        return default_trab_min, default_cort_min, default_cort_max

    sub_hist = hist[mask]
    sub_edges = edges[:-1][mask]

    peak_indices = sub_hist.argsort()[-2:]
    peak_indices.sort()
    trab_peak = float(sub_edges[peak_indices[0]])
    cort_peak = float(sub_edges[peak_indices[1]])

    mid = 0.5 * (trab_peak + cort_peak)
    trab_min = max(100.0, trab_peak * 0.5)
    cort_min = mid
    cort_max = cort_peak * 1.3

    print(f"Auto trab/cort peaks: trab_peak~{trab_peak:.1f}, cort_peak~{cort_peak:.1f}")
    print(
        f"Auto thresholds -> TRAB_MIN={trab_min:.1f}, "
        f"CORT_MIN={cort_min:.1f}, CORT_MAX={cort_max:.1f}"
    )
    return trab_min, cort_min, cort_max


def soft_box(x, lo, hi, margin):
    return np.clip(
        np.minimum(
            (x - (lo - margin)) / margin,
            ((hi + margin) - x) / margin,
        ),
        0.0,
        1.0,
    )


def identify_cortical_shell(cortex_candidate: np.ndarray, solid_bool: np.ndarray) -> np.ndarray:
    """
    Identify the main connected cortical shell by:
    1. Finding the largest connected component of high-confidence cortex
    2. Preserving this as the 'true' cortical shell

    This prevents isolated high-density trabecular voxels from being classified as cortex.
    """
    struct_6 = ndi.generate_binary_structure(3, 1)
    labeled, n = ndi.label(cortex_candidate, structure=struct_6)

    if n == 0:
        return cortex_candidate

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    if sizes.size == 0:
        return cortex_candidate

    largest_label = np.argmax(sizes)
    cortical_shell = labeled == largest_label

    print(f"Identified cortical shell with {int(cortical_shell.sum())} voxels")
    print(f"Removed {int(cortex_candidate.sum()) - int(cortical_shell.sum())} isolated voxels")

    return cortical_shell


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
):

    specimen_id, pair_folder, abs_pair_path, abs_side_path = make_ids_and_paths(
        specimen_base, side
    )

    # Rest of your code starts here...

    meta_path = find_meta(abs_side_path, specimen_id)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    out_dir = Path(meta["outputs"]["segmentation_dir"])
    print(f"Using segmentation output folder: {out_dir}")

    input_mgha = out_dir / f"{specimen_id}_calibrated_mgHA.tiff"
    input_band = out_dir / f"{specimen_id}_cortical_band_mask.tiff"
    input_outer = out_dir / f"{specimen_id}_cortical_outer_ring.tiff"

    print(
        "=== Script 03: cortex / trabecula segmentation "
        "(density-based + connectivity-filtered) ==="
    )
    print(f"Specimen: {specimen_id}")
    print(f"mgHA file: {input_mgha}")
    print(f"Band file: {input_band}")
    print(f"Outer ring: {input_outer}")
    print(f"Output dir: {out_dir}")

    print("=== Load data ===")
    mg = imread(str(input_mgha))
    band_raw = imread(str(input_band))
    outer_raw = imread(str(input_outer))

    print(f"mgHA shape: {mg.shape}, dtype: {mg.dtype}")
    print(f"Cortical band shape: {band_raw.shape}, dtype: {band_raw.dtype}")
    print(f"Outer ring shape: {outer_raw.shape}, dtype: {outer_raw.dtype}")

    if mg.shape != band_raw.shape or mg.shape != outer_raw.shape:
        raise ValueError("mgHA, band, and outer ring volumes must have the same shape.")

    band = band_raw > 0
    outer = outer_raw > 0  # QC only
    print(f"Band voxels: {int(band.sum())}")
    print(f"Outer voxels (QC only): {int(outer.sum())}")

    mg = mg.astype(np.float32)

    bone_mask_path = out_dir / f"{specimen_id}_bone_mask_binary.tiff"
    bone_raw = imread(str(bone_mask_path))
    bone_mask = bone_raw > 0
    print(f"Bone mask voxels (for Script 03): {int(bone_mask.sum())}")

    solid_path = out_dir / f"{specimen_id}_bone_mask_solid_blob.tiff"
    solid_raw = imread(str(solid_path))
    solid_bool = solid_raw > 0
    print(f"Solid blob voxels: {int(solid_bool.sum())}")

    bone_mask_for_thresholds = band

    # Use GUI-provided values directly (no auto-detection)
    print(f"\nUsing density thresholds from GUI:")
    print(f"  Trab min: {trab_min}")
    print(f"  Cort min: {cort_min}")
    print(f"  Cort max: {cort_max}")

    print("=== Hierarchical cortical / trabecular classification ===")

    # ============================================================================
    # STEP 1: Identify candidate regions based on DENSITY ALONE (no geometry yet)
    # ============================================================================
    print("\n--- Density-based classification ---")

    # High-confidence cortical density
    is_high_cort_density = (mg >= cort_min) & (mg <= cort_max)
    high_cort_voxels = int(is_high_cort_density.sum())
    print(f"Voxels with cortical density ({cort_min:.1f}-{cort_max:.1f} mgHA): {high_cort_voxels}")

    # High-confidence trabecular density
    is_high_trab_density = (mg >= trab_min) & (mg < cort_min)
    high_trab_voxels = int(is_high_trab_density.sum())
    print(f"Voxels with trabecular density ({trab_min:.1f}-{cort_min:.1f} mgHA): {high_trab_voxels}")

    # Gray zone (ambiguous density)
    is_gray_zone = ~(is_high_cort_density | is_high_trab_density) & bone_mask
    gray_voxels = int(is_gray_zone.sum())
    print(f"Voxels in gray zone (ambiguous density): {gray_voxels}")

    # ============================================================================
    # STEP 2: Apply GEOMETRY constraint - cortex must be in band
    # ============================================================================
    print("\n--- Geometry constraint (cortical band) ---")

    # Cortical candidates: in band + cortical density
    cortex_candidate = is_high_cort_density & band & bone_mask
    print(f"Cortical candidates (density + band + bone): {int(cortex_candidate.sum())}")

    # Trabecular candidates: in band + trabecular density
    trab_candidate_in_band = is_high_trab_density & band & bone_mask
    print(f"Trabecular in band (density + band + bone): {int(trab_candidate_in_band.sum())}")

    # ============================================================================
    # STEP 3: Filter cortex using CONNECTIVITY to remove isolated high-density voxels
    # ============================================================================
    print("\n--- Connectivity filtering (cortical shell identification) ---")

    cortex_candidate_opened = ndi.binary_opening(cortex_candidate)
    is_cortex = identify_cortical_shell(cortex_candidate_opened, solid_bool)

    print(f"Cortical shell voxels (after connectivity filter): {int(is_cortex.sum())}")

    # ============================================================================
    # STEP 4: Trabecular = everything else in bone (in band OR outside band)
    # ============================================================================
    print("\n--- Trabecular assignment ---")

    # Start with all bone that's not cortex
    is_trab = bone_mask & (~is_cortex)

    print(f"Initial trabecular voxels: {int(is_trab.sum())}")

    # Include high-density trabecular voxels that are in the band
    # (They are allowed to exist in the cortical band if density is trabecular)
    is_trab_in_band = is_trab & trab_candidate_in_band
    print(f"Trabecular voxels in cortical band: {int(is_trab_in_band.sum())}")

    print(f"Total trabecular voxels: {int(is_trab.sum())}")

    # ============================================================================
    # STEP 5: Remove periosteal (outside solid blob) cortex
    # ============================================================================
    print("\n--- Removing periosteal cortex ---")
    periosteal_side = ~solid_bool
    is_cortex[periosteal_side] = False

    # Recompute trabecula after removing periosteal cortex
    is_trab = bone_mask & (~is_cortex)

    print(f"Cortex voxels (after periosteal removal): {int(is_cortex.sum())}")
    print(f"Trabecular voxels (after periosteal removal): {int(is_trab.sum())}")

    # ============================================================================
    # STEP 6: Conservative cleanup (preserve trabecular structure in band)
    # ============================================================================
    print("\n=== Tissue cleanup (conservative) ===")
    struct_6 = ndi.generate_binary_structure(3, 1)
    z, y, x = mg.shape

    print("--- Cortical cleanup: mild morphology (no aggressive closing) ---")
    # Light opening: remove only very small noise, preserve main structure
    cortex_clean = ndi.binary_opening(is_cortex, structure=struct_6)

    # Preserve cortical end slices if they contain cortex
    if z > 0:
        if cortex_clean[0].sum() == 0 and is_cortex[0].sum() > 0:
            cortex_clean[0] = ndi.binary_opening(is_cortex[0])
        if cortex_clean[-1].sum() == 0 and is_cortex[-1].sum() > 0:
            cortex_clean[-1] = ndi.binary_opening(is_cortex[-1])

    # Remove only VERY small cortical debris (keep structure)
    labeled_cort, n_cort = ndi.label(cortex_clean, structure=struct_6)
    if n_cort > 0:
        sizes = np.bincount(labeled_cort.ravel())
        sizes[0] = 0
        keep_labels = np.where(sizes >= min_cortex_voxels)[0]
        cortex_clean = np.isin(labeled_cort, keep_labels)

    print(f"Cortical voxels (after cleanup): {int(cortex_clean.sum())}")

    print("--- Trabecular cleanup: light opening (preserve trabeculae in band) ---")
    # Very light opening for trabecula - just remove absolute noise
    trab_clean = ndi.binary_opening(is_trab, structure=struct_6)

    print(f"Trabecular voxels (after cleanup): {int(trab_clean.sum())}")

    # ============================================================================
    # STEP 6b: Remove small trabecular speckles in the cortex (keep only main medullary space)
    # ============================================================================
    print("\n--- Removing small trabecular speckles in the cortex ---")
    trab_labels, n_trab = ndi.label(trab_clean, structure=struct_6)
    sizes = np.bincount(trab_labels.ravel())
    sizes[0] = 0

    if n_trab > 0:
        # Find the largest trabecular cluster (assumed to be the medullary space)
        largest_trab_label = np.argmax(sizes)
        main_trab = trab_labels == largest_trab_label

        # Minimum size threshold for trabecular clusters to keep (in voxels)
        # Adjust this value based on your data - smaller = keep more, larger = faster

        # Identify small cluster labels efficiently (vectorized)
        small_cluster_labels = np.where((sizes > 0) & (sizes < min_trab_cluster_size))[0]
        small_cluster_labels = small_cluster_labels[small_cluster_labels != largest_trab_label]

        if small_cluster_labels.size > 0:
            # Create a single mask for all small clusters (one operation, not looping)
            small_trab_mask = np.isin(trab_labels, small_cluster_labels)

            # Reassign small clusters in the band to cortex (vectorized)
            cortex_clean[small_trab_mask & band] = True

            # Remove all small clusters from trabecular (vectorized)
            trab_clean[small_trab_mask] = False

    print(f"Trabecular voxels after speckle removal: {int(trab_clean.sum())}")
    print(f"Cortical voxels after speckle removal: {int(cortex_clean.sum())}")

    # ============================================================================
    # STEP 6c: Aggressive removal of isolated trabecular voxels in cortical band
    # ============================================================================
    print("\n--- Removing isolated trabecular voxels in cortical band ---")

    trab_in_band = trab_clean & band

    if trab_in_band.sum() > 0:
        # Label all remaining trabecular voxels in the band
        trab_band_labels, n_trab_band = ndi.label(trab_in_band, structure=struct_6)

        if n_trab_band > 0:
            sizes_in_band = np.bincount(trab_band_labels.ravel())

            # Identify tiny/isolated clusters to remove
            tiny_band_labels = np.where((sizes_in_band > 0) & (sizes_in_band < min_band_trab_size))[0]

            if tiny_band_labels.size > 0:
                # Create mask for all tiny clusters
                tiny_band_mask = np.isin(trab_band_labels, tiny_band_labels)

                # Remove from trabecular, reassign to cortex
                trab_clean[tiny_band_mask] = False
                cortex_clean[tiny_band_mask] = True

                removed_count = int(tiny_band_mask.sum())
                print(f"Removed {removed_count} isolated trabecular voxels from band")

    # ============================================================================
    # STEP 6d: Remove trabecular in band NOT connected to main medullary space
    # ============================================================================
    print("\n--- Removing unconnected trabecular in cortical band ---")

    # Identify the main medullary trabecular space (largest cluster)
    trab_labels_all, n_trab_all = ndi.label(trab_clean, structure=struct_6)
    sizes_all = np.bincount(trab_labels_all.ravel())
    sizes_all[0] = 0

    if n_trab_all > 0:
        # Find largest trabecular cluster (the real medullary space)
        largest_trab_label = np.argmax(sizes_all)
        main_medullary = trab_labels_all == largest_trab_label

        # Dilate the main medullary space to define "connected to medullary"
        # This allows some tolerance for connectivity
        struct_dilate = ndi.generate_binary_structure(3, 1)
        medullary_dilated = ndi.binary_dilation(main_medullary, structure=struct_dilate, iterations=medullary_dilation_iterations)


        # Get trabecular in the band
        trab_in_band = trab_clean & band

        # Find which trabecular in band is connected to (dilated) medullary space
        connected_to_medullary = trab_in_band & medullary_dilated

        # Remove trabecular in band that is NOT connected to medullary
        unconnected_in_band = trab_in_band & (~medullary_dilated)

        # Remove unconnected structures
        trab_clean[unconnected_in_band] = False

        removed_unconnected = int(unconnected_in_band.sum())
        print(f"Removed {removed_unconnected} trabecular voxels NOT connected to main medullary space")
        print(f"Kept {int(connected_to_medullary.sum())} trabecular voxels connected to medullary space")

    print(f"Trabecular voxels after medullary connectivity filter: {int(trab_clean.sum())}")

    # ============================================================================
    # STEP 7: Identify and preserve intracortical pores
    # ============================================================================

    print("\n=== Intracortical pore detection ===")
    void_inside = solid_bool & (~bone_mask)
    void_labels, n_void = ndi.label(void_inside, structure=struct_6)

    intracortical_pores = np.zeros_like(void_inside, dtype=bool)
    if n_void > 0:
        # Border voxels for voids (touching volume boundary)
        border_mask = np.zeros_like(void_inside, dtype=bool)
        border_mask[0, :, :] |= void_inside[0, :, :]
        border_mask[-1, :, :] |= void_inside[-1, :, :]
        border_mask[:, 0, :] |= void_inside[:, 0, :]
        border_mask[:, -1, :] |= void_inside[:, -1, :]
        border_mask[:, :, 0] |= void_inside[:, :, 0]
        border_mask[:, :, -1] |= void_inside[:, :, -1]

        border_labels = np.unique(void_labels[border_mask])
        all_labels = np.arange(1, n_void + 1)
        pore_labels = np.setdiff1d(all_labels, border_labels)

        if pore_labels.size > 0:

            # Calculate sizes for all pores using ndi.sum
            sizes = ndi.sum(
                void_inside.astype(np.uint8),
                void_labels,
                index=pore_labels,
            )

            # Keep only pores above minimum size
            keep = pore_labels[np.array(sizes) >= pore_min_size]

            # Create intracortical pores mask using loop (memory-efficient)
            # instead of np.isin() which allocates huge temporary arrays
            intracortical_pores = np.zeros_like(void_inside, dtype=bool)
            if keep.size > 0:
                for pore_label in keep:
                    intracortical_pores |= (void_labels == pore_label)

    print(f"Intracortical pore voxels detected: {int(intracortical_pores.sum())}")

    # Ensure pores remain empty
    cortex_clean[intracortical_pores] = False
    trab_clean[intracortical_pores] = False

    print(f"\n=== Final segmentation ===")
    print(f"Cortical voxels (final): {int(cortex_clean.sum())}")
    print(f"Trabecular voxels (final): {int(trab_clean.sum())}")
    print(f"Intracortical pores (final): {int(intracortical_pores.sum())}")

    if cortex_clean.sum() == 0:
        print("WARNING: no cortical voxels detected.")
    if trab_clean.sum() == 0:
        print("WARNING: no trabecular voxels detected.")

    # ============================================================================
    # STEP 7b: Keep only largest connected island for cortex and trabecula
    # ============================================================================
    print("\n=== Keeping only largest connected islands ===")

    # Keep only largest cortical island
    if cortex_clean.sum() > 0:
        cortex_labels, n_cortex = ndi.label(cortex_clean, structure=struct_6)
        if n_cortex > 0:
            sizes_cortex = np.bincount(cortex_labels.ravel())
            sizes_cortex[0] = 0
            largest_cortex_label = np.argmax(sizes_cortex)
            cortex_before = int(cortex_clean.sum())
            cortex_clean = cortex_labels == largest_cortex_label
            cortex_after = int(cortex_clean.sum())
            removed_cortex = cortex_before - cortex_after
            print(f"Cortex: kept largest island ({cortex_after} voxels), removed {removed_cortex} isolated voxels")

    # Keep only largest trabecular island
    if trab_clean.sum() > 0:
        trab_labels_final, n_trab = ndi.label(trab_clean, structure=struct_6)
        if n_trab > 0:
            sizes_trab = np.bincount(trab_labels_final.ravel())
            sizes_trab[0] = 0
            largest_trab_label = np.argmax(sizes_trab)
            trab_before = int(trab_clean.sum())
            trab_clean = trab_labels_final == largest_trab_label
            trab_after = int(trab_clean.sum())
            removed_trab = trab_before - trab_after
            print(f"Trabecula: kept largest island ({trab_after} voxels), removed {removed_trab} isolated voxels")

    print(f"\nCortical voxels (largest island only): {int(cortex_clean.sum())}")
    print(f"Trabecular voxels (largest island only): {int(trab_clean.sum())}")


    # ============================================================================
    # STEP 8: Save outputs
    # ============================================================================
    cortex_u8 = cortex_clean.astype(np.uint8) * 255
    trab_u8 = trab_clean.astype(np.uint8) * 255

    cortex_path = out_dir / f"{specimen_id}_cortical_mask_final.tiff"
    trab_path = out_dir / f"{specimen_id}_trabecular_mask_final.tiff"

    # Delete existing files to ensure overwrite
    import os
    if cortex_path.exists():
        os.remove(cortex_path)
    if trab_path.exists():
        os.remove(trab_path)

    print(f"\nSaving cortical mask to: {cortex_path}")
    imwrite(str(cortex_path), cortex_u8)

    print(f"Saving trabecular mask to: {trab_path}")
    imwrite(str(trab_path), trab_u8)

    if WRITE_LABEL_VOLUME:
        labels = np.zeros(mg.shape, dtype=np.uint8)
        labels[trab_clean] = 1
        labels[cortex_clean] = 2
        label_path = out_dir / f"{specimen_id}_bone_compartments_label.tiff"
        if label_path.exists():
            os.remove(label_path)
        print(f"Saving label volume to: {label_path}")
        imwrite(str(label_path), labels)

    # ============================================================================
    # STEP 9: Save comprehensive parameters summary
    # ============================================================================
    band_voxels = int(band.sum())
    bone_mask_voxels = int(bone_mask.sum())
    cortex_voxels_final = int(cortex_clean.sum())
    trab_voxels_final = int(trab_clean.sum())

    all_parameters = {
        "specimen_id": specimen_id,
        "cortical_band_thickness_mm": BAND_DEPTH_MM,
        "bone_threshold_mgha": 200.0,
        "trab_min_mgha": trab_min,
        "cort_min_mgha": cort_min,
        "cort_max_mgha": cort_max,
        "min_trab_cluster_size": min_trab_cluster_size,
        "min_band_trab_size": min_band_trab_size,
        "medullary_dilation_iterations": medullary_dilation_iterations,
        "min_cortex_voxels": min_cortex_voxels,
        "pore_min_size": pore_min_size,
        "write_label_volume": WRITE_LABEL_VOLUME,
        "band_voxels": band_voxels,
        "bone_mask_voxels": bone_mask_voxels,
        "cortex_voxels_final": cortex_voxels_final,
        "trab_voxels_final": trab_voxels_final,
        "intracortical_pore_voxels": int(intracortical_pores.sum()),
        "method": "density-based + connectivity-filtered + medullary connectivity",
    }

    # Save as JSON
    json_path = out_dir / f"{specimen_id}_segmentation_params.json"
    with open(json_path, "w") as f:
        json.dump(all_parameters, f, indent=2)

    # Save as comprehensive parameters table (TXT format)
    params_table_path = out_dir / f"{specimen_id}_parameters_used.txt"
    with open(params_table_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"SEGMENTATION PARAMETERS USED FOR SPECIMEN: {specimen_id}\n")
        f.write("=" * 80 + "\n\n")
        f.write("PARAMETER NAME\t\t\t\tVALUE\n")
        f.write("-" * 80 + "\n")

        for param_name, param_value in all_parameters.items():
            f.write(f"{param_name:<40}\t{param_value}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("METHOD DESCRIPTION:\n")
        f.write("=" * 80 + "\n")
        f.write("This segmentation uses density-based classification combined with:\n")
        f.write("1. Connectivity filtering to identify main cortical shell\n")
        f.write("2. Medullary connectivity to remove isolated trabecular artifacts\n")
        f.write("3. Periosteal edge cleaning to remove resolution artifacts\n")
        f.write("=" * 80 + "\n")

    # Save as simple TSV table (easier to import into Excel/Calc)
    params_tsv_path = out_dir / f"{specimen_id}_parameters_used.tsv"
    with open(params_tsv_path, "w") as f:
        f.write("Parameter_Name\tValue\n")
        for param_name, param_value in all_parameters.items():
            f.write(f"{param_name}\t{param_value}\n")

    # Also save the old format for backwards compatibility
    txt_path = out_dir / f"{specimen_id}_segmentation_params.txt"
    with open(txt_path, "w") as f:
        f.write(
            "specimen_id\t"
            "cortical_band_thickness_mm\t"
            "bone_threshold_mgha\t"
            "trab_min_mgha\t"
            "cort_min_mgha\t"
            "cort_max_mgha\t"
            "band_voxels\t"
            "bone_mask_voxels\t"
            "cortex_voxels_final\t"
            "trab_voxels_final\t"
            "intracortical_pore_voxels\t"
            "method\n"
        )
        f.write(
            f"{specimen_id}\t"
            f"{BAND_DEPTH_MM:.3f}\t"
            f"200.0\t"
            f"{trab_min:.3f}\t"
            f"{cort_min:.3f}\t"
            f"{cort_max:.3f}\t"
            f"{band_voxels}\t"
            f"{bone_mask_voxels}\t"
            f"{cortex_voxels_final}\t"
            f"{trab_voxels_final}\t"
            f"{int(intracortical_pores.sum())}\t"
            f"density-based + connectivity-filtered + medullary connectivity\n"
        )

    print(f"Saved JSON parameters to: {json_path}")
    print(f"Saved parameters table to: {params_table_path}")
    print(f"Saved TSV parameters to: {params_tsv_path}")
    print(f"Saved segmentation params to: {txt_path}")
    print("=== Script 03 complete ===")

