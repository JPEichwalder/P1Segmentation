#!/usr/bin/env python3

"""
master_js.py

Master driver to run the full 3‑step segmentation pipeline
(Script 01, 02, 03) for multiple specimens and sides, with
all key parameters defined here in one place.
"""

import itertools
import time
from pathlib import Path

# --- 1) SPECIMENS AND SIDES TO RUN (EDIT HERE) ---

SPECIMENS = [
 #  "23162",
    "23167",
    "23168",
    "23169",
    "23170",
    "23175",
]

SIDES = [
    "L",
    "R",
]

# --- 2) PARAMETER OVERRIDES (EDIT HERE IF YOU WANT) ---

# If you leave these as None, the values from config_js.py
# will be used (GROW_RADIUS_2D, BAND_DEPTH_MM, etc.). [file:3]
# If you set a number here, that value will override config_js
# at runtime for ALL specimens in this run. [file:3]

OVERRIDE_GROW_RADIUS_2D = None      # e.g. 20
OVERRIDE_VOXEL_SIZE_UM = None       # e.g. 32.0
OVERRIDE_BAND_DEPTH_MM = None       # e.g. 2.0
OVERRIDE_BAND_DEPTH_VOX = None      # e.g. 60

OVERRIDE_OUTER_RING_THICKNESS_PX = None   # e.g. 2
OVERRIDE_INNER_RING_THICKNESS_PX = None   # e.g. 2

OVERRIDE_TRAB_MIN = None             # e.g. 200.0
OVERRIDE_CORT_MIN = None             # e.g. 600.0
OVERRIDE_CORT_MAX = None             # e.g. 1200.0
OVERRIDE_WRITE_LABEL_VOLUME = None   # e.g. True or False

from config_js import (
    GROW_RADIUS_2D,
    VOXEL_SIZE_UM,
    BAND_DEPTH_MM,
    BAND_DEPTH_VOX,
    OUTER_RING_THICKNESS_PX,
    INNER_RING_THICKNESS_PX,
    TRAB_MIN_DEFAULT,
    CORT_MIN_DEFAULT,
    CORT_MAX_DEFAULT,
    WRITE_LABEL_VOLUME,
)

from script01_calibration_js import run_script_01
from script02_band_rings_js import run_script_02
from script03_cort_trab_js import run_script_03

RUNTIME_LOG_PATH = Path("segmentation_runtime_log.txt")

def apply_overrides():
    """
    Apply master‑script overrides to the imported module‑level
    variables so Script 02 and 03 pick them up. [file:6][file:2]
    """
    global GROW_RADIUS_2D, VOXEL_SIZE_UM, BAND_DEPTH_MM, BAND_DEPTH_VOX
    global OUTER_RING_THICKNESS_PX, INNER_RING_THICKNESS_PX
    global TRAB_MIN_DEFAULT, CORT_MIN_DEFAULT, CORT_MAX_DEFAULT
    global WRITE_LABEL_VOLUME

    if OVERRIDE_GROW_RADIUS_2D is not None:
        GROW_RADIUS_2D = OVERRIDE_GROW_RADIUS_2D
    if OVERRIDE_VOXEL_SIZE_UM is not None:
        VOXEL_SIZE_UM = OVERRIDE_VOXEL_SIZE_UM
    if OVERRIDE_BAND_DEPTH_MM is not None:
        BAND_DEPTH_MM = OVERRIDE_BAND_DEPTH_MM
    if OVERRIDE_BAND_DEPTH_VOX is not None:
        BAND_DEPTH_VOX = OVERRIDE_BAND_DEPTH_VOX

    if OVERRIDE_OUTER_RING_THICKNESS_PX is not None:
        OUTER_RING_THICKNESS_PX = OVERRIDE_OUTER_RING_THICKNESS_PX
    if OVERRIDE_INNER_RING_THICKNESS_PX is not None:
        INNER_RING_THICKNESS_PX = OVERRIDE_INNER_RING_THICKNESS_PX

    if OVERRIDE_TRAB_MIN is not None:
        TRAB_MIN_DEFAULT = OVERRIDE_TRAB_MIN
    if OVERRIDE_CORT_MIN is not None:
        CORT_MIN_DEFAULT = OVERRIDE_CORT_MIN
    if OVERRIDE_CORT_MAX is not None:
        CORT_MAX_DEFAULT = OVERRIDE_CORT_MAX

    if OVERRIDE_WRITE_LABEL_VOLUME is not None:
        WRITE_LABEL_VOLUME = OVERRIDE_WRITE_LABEL_VOLUME


def run_pipeline_for_specimen(base: str, side: str):
    """
    Run Script 01, 02, 03 in sequence for one specimen+side,
    and record how long it took.
    """
    specimen_id = f"{base}{side}"

    print("=" * 80)
    print(f"Running full pipeline for specimen {base}, side {side}")
    print("=" * 80)

    start = time.time()

    # Script 01: calibration, bone mask, etc.
    print("\n--- Script 01 (calibration) ---")
    run_script_01(base, side)

    # Script 02: solid blob, cortical band, inner/outer rings
    print("\n--- Script 02 (band + rings) ---")
    run_script_02(base, side)

    # Script 03: cortex/trabecular segmentation
    print("\n--- Script 03 (cortex + trabecula) ---")
    run_script_03(base, side)

    end = time.time()
    elapsed_sec = end - start
    elapsed_min = elapsed_sec / 60.0

    print(f"\nDone specimen {base} {side}")
    print(f"Total runtime: {elapsed_sec:.1f} s ({elapsed_min:.2f} min)\n")

    # Optional: append to a small log file for later reference
    try:
        with RUNTIME_LOG_PATH.open("a") as f:
            f.write(
                f"{specimen_id}\t{elapsed_sec:.1f}\t{elapsed_min:.2f}\n"
            )
    except Exception as e:
        print(f"WARNING: could not write runtime log ({e})")


def main():
    # Apply global overrides once before running anything
    apply_overrides()

    # Loop through all specimen/side combinations
    for base, side in itertools.product(SPECIMENS, SIDES):
        run_pipeline_for_specimen(base, side)


if __name__ == "__main__":
    main()