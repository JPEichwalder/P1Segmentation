import numpy as np
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
from skimage.measure import euler_number
import porespy as ps  # local thickness


# ---------------------------------------
# User settings
# ---------------------------------------
VOXEL_SIZE_MM = 0.032  # 32 µm
BASE_DIR = Path(r"T:\TMMI Shared\Projects\Puck_Validation\Workflow\Development")

# Specimen base folders to process (pair folders)
SPECIMEN_BASES = [
    "23162R&L",
    "23167R&L",
    "23168R&L",
    "23169R&L",
    "23170R&L",
    "23175R&L",
]


# ---------------------------------------
# Core computations (raw physical units)
# ---------------------------------------

def load_cube(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    return (arr > 0).astype(np.uint8)


def compute_volumes(mask: np.ndarray, voxel_size_mm: float):
    # BV and TV in mm^3; BV/TV dimensionless
    bv_vox = int(mask.sum())
    tv_vox = int(mask.size)
    vox_vol_mm3 = voxel_size_mm ** 3
    bv_mm3 = bv_vox * vox_vol_mm3
    tv_mm3 = tv_vox * vox_vol_mm3
    bvtv = bv_mm3 / tv_mm3 if tv_mm3 > 0 else 0.0
    return bv_mm3, tv_mm3, bvtv


def compute_thickness_stats(mask: np.ndarray, voxel_size_mm: float):
    # Trabecular thickness in mm using distance-transform-based local thickness
    mask_bool = mask.astype(bool)
    lt_bone = ps.filters.local_thickness(im=mask_bool, method="dt")
    vals_th = lt_bone[mask_bool].astype(np.float64)
    if vals_th.size == 0:
        return 0.0, 0.0, 0.0

    vals_th_mm = vals_th * voxel_size_mm
    return (
        float(vals_th_mm.mean()),
        float(vals_th_mm.std()),
        float(vals_th_mm.max()),
    )


def compute_spacing_stats(mask: np.ndarray, voxel_size_mm: float):
    # Trabecular spacing in mm: local thickness of the void phase
    void = ~mask.astype(bool)
    lt_void = ps.filters.local_thickness(im=void, method="dt")
    vals_sp = lt_void[void].astype(np.float64)
    if vals_sp.size == 0:
        return 0.0, 0.0, 0.0

    vals_sp_mm = vals_sp * voxel_size_mm
    return (
        float(vals_sp_mm.mean()),
        float(vals_sp_mm.std()),
        float(vals_sp_mm.max()),
    )


def compute_euler_and_connectivity(mask: np.ndarray, voxel_size_mm: float):
    # Euler characteristic (dimensionless) and connectivity density (mm^-3)
    chi = euler_number(mask.astype(bool), connectivity=3)
    conn = -chi  # BoneJ-style connectivity definition
    tv_mm3 = mask.size * (voxel_size_mm ** 3)
    conn_d_mm3 = conn / tv_mm3 if tv_mm3 > 0 else 0.0
    return int(chi), float(conn), float(conn_d_mm3)


def process_side(specimen_root: Path, specimen_base: str, side: str):
    side_id = f"{specimen_base}{side}"
    side_dir = specimen_root / side_id
    cube_dir = side_dir / "Cube"

    if not cube_dir.is_dir():
        print(f"[SKIP] No Cube folder: {cube_dir}")
        return

    out_csv = cube_dir / f"{side_id}_cube_microstructure_python.csv"
    print(f"\n[INFO] Processing cubes for {side_id}")
    print(f" Cube folder: {cube_dir}")
    print(f" Output CSV:  {out_csv}")

    rows = []
    cube_paths = sorted(cube_dir.glob("cube_*.tiff"))
    if not cube_paths:
        print("  No cube_*.tiff files found.")
        return

    for cube_path in cube_paths:
        print(f"  -> {cube_path.name}")
        mask = load_cube(cube_path)

        # BV, TV, BV/TV (mm^3, mm^3, dimensionless)
        bv_mm3, tv_mm3, bvtv = compute_volumes(mask, VOXEL_SIZE_MM)

        # Thickness stats (mm)
        tb_th_mean_mm, tb_th_std_mm, tb_th_max_mm = compute_thickness_stats(
            mask, VOXEL_SIZE_MM
        )

        # Spacing stats (mm)
        tb_sp_mean_mm, tb_sp_std_mm, tb_sp_max_mm = compute_spacing_stats(
            mask, VOXEL_SIZE_MM
        )

        # Euler, connectivity, connectivity density (dimensionless, -, mm^-3)
        euler_chi, connectivity, conn_d_mm3 = compute_euler_and_connectivity(
            mask, VOXEL_SIZE_MM
        )

        # Δ(x): BoneJ-specific anisotropy measure; NA for now
        delta_x = np.nan

        rows.append(
            {
                "specimen_side": side_id,
                "cube_file": cube_path.name,
                "BV_mm^3": bv_mm3,
                "TV_mm^3": tv_mm3,
                "BVTV": bvtv,
                "Tb.Th_mean_mm": tb_th_mean_mm,
                "Tb.Th_std_mm": tb_th_std_mm,
                "Tb.Th_max_mm": tb_th_max_mm,
                "Tb.Sp_mean_mm": tb_sp_mean_mm,
                "Tb.Sp_std_mm": tb_sp_std_mm,
                "Tb.Sp_max_mm": tb_sp_max_mm,
                "Euler_chi": euler_chi,
                "Delta_x": delta_x,
                "Connectivity": connectivity,
                "ConnD_mm^-3": conn_d_mm3,
            }
        )

    cols = [
        "specimen_side",
        "cube_file",
        "BV_mm^3",
        "TV_mm^3",
        "BVTV",
        "Tb.Th_mean_mm",
        "Tb.Th_std_mm",
        "Tb.Th_max_mm",
        "Tb.Sp_mean_mm",
        "Tb.Sp_std_mm",
        "Tb.Sp_max_mm",
        "Euler_chi",
        "Delta_x",
        "Connectivity",
        "ConnD_mm^-3",
    ]

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Saved metrics for {len(rows)} cubes to {out_csv}")


def main():
    for base_name in SPECIMEN_BASES:
        specimen_root = BASE_DIR / base_name
        if not specimen_root.is_dir():
            print(f"[SKIP] No specimen base: {specimen_root}")
            continue

        specimen_id = base_name.replace("R&L", "").replace("L&R", "")
        for side in ("L", "R"):
            process_side(specimen_root, specimen_id, side)


if __name__ == "__main__":
    main()
