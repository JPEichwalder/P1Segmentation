#!/usr/bin/env python3

import argparse

from config_js import SPECIMEN_BASE_DEFAULT, SIDE_DEFAULT
from script01_calibration_js import run_script_01
from script02_band_rings_js import run_script_02
from script03_cort_trab_js import run_script_03


def main():
    parser = argparse.ArgumentParser(
        description="Bone segmentation pipeline (classic js01 logic, split into 3 scripts)."
    )
    parser.add_argument(
        "--specimen_base",
        default=SPECIMEN_BASE_DEFAULT,
        help='Specimen base ID, e.g. "23162"',
    )
    parser.add_argument(
        "--side",
        default=SIDE_DEFAULT,
        help='Side, "L" or "R"',
    )
    parser.add_argument(
        "--step",
        choices=["01", "02", "03", "all"],
        default="all",
        help="Which step(s) to run: 01, 02, 03, or all (01→02→03).",
    )
    args = parser.parse_args()

    base = args.specimen_base
    side = args.side

    if args.step in ("01", "all"):
        run_script_01(base, side)
    if args.step in ("02", "all"):
        run_script_02(base, side)
    if args.step in ("03", "all"):
        run_script_03(base, side)


if __name__ == "__main__":
    main()
