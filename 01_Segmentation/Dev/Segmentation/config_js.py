# config_js.py

from pathlib import Path

# Global specimen defaults
SPECIMEN_BASE_DEFAULT = "23162"
SIDE_DEFAULT = "L"

# Root path
PROJECT_ROOT = Path(r"C:\Users\jeichwal\Documents\A_DocTech\P1_Segmentation\Data")

# Script 02 parameters
GROW_RADIUS_2D = 20
VOXEL_SIZE_UM = 32.0
BAND_DEPTH_MM = 2.0
BAND_DEPTH_VOX = None
OUTER_RING_THICKNESS_PX = 2
INNER_RING_THICKNESS_PX = 2

# Script 03 parameters
TRAB_MIN_DEFAULT = 200.0
CORT_MIN_DEFAULT = 600.0
CORT_MAX_DEFAULT = 1200.0
WRITE_LABEL_VOLUME = True

