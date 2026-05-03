#!/usr/bin/env python3
"""
GUI.py

Interactive GUI for bone segmentation pipeline with FULL parameter control
Allows setting ALL tunable parameters and running segmentation
"""

import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

from config_js import PROJECT_ROOT, SPECIMEN_BASE_DEFAULT, SIDE_DEFAULT
from script01_calibration_js import run_script_01
from script02_band_rings_js import run_script_02
from script03_cort_trab_js import run_script_03


def run_pipeline(
    base_dir: str,
    specimen_base: str,
    side: str,
    grow_radius: int,
    outer_ring_thickness: int,
    inner_ring_thickness: int,
    voxel_um: float,
    band_mm: float,
    trab_min: float,
    cort_min: float,
    cort_max: float,
    min_trab_cluster_size: int,
    min_band_trab_size: int,
    medullary_dilation_iterations: int,
    min_cortex_voxels: int,
    pore_min_size: int,
    run_s1: bool,
    run_s2: bool,
    run_s3: bool,
    log_widget: tk.Text,
    stop_flag,
):
    """
    Run the segmentation pipeline in a separate thread.
    base_dir is the project/data directory selected in the GUI.
    """
    try:
        specimen_id = f"{specimen_base}{side}"
        log = log_widget

        log.insert(tk.END, "DATA DIRECTORY:\n")
        log.insert(tk.END, f"  GUI-selected directory: {base_dir}\n\n")

        log.insert(tk.END, "DENSITY & GEOMETRY PARAMETERS:\n")
        log.insert(tk.END, f"  Voxel size:                     {voxel_um} µm\n")
        log.insert(tk.END, f"  Cortical band depth:            {band_mm} mm\n")
        log.insert(tk.END, f"  Trabecular min threshold:       {trab_min} mgHA\n")
        log.insert(tk.END, f"  Cortical min threshold:         {cort_min} mgHA\n")
        log.insert(tk.END, f"  Cortical max threshold:         {cort_max} mgHA\n\n")

        log.insert(tk.END, "SEGMENTATION REFINEMENT PARAMETERS:\n")
        log.insert(tk.END, f"  Grow radius:                    {grow_radius} voxels\n")
        log.insert(tk.END, f"  Outer ring thickness:           {outer_ring_thickness} voxels\n")
        log.insert(tk.END, f"  Inner ring thickness:           {inner_ring_thickness} voxels\n")
        log.insert(tk.END, f"  Min trabecular cluster size:    {min_trab_cluster_size} voxels\n")
        log.insert(tk.END, f"  Min band trabecular size:       {min_band_trab_size} voxels\n")
        log.insert(tk.END, f"  Medullary dilation iterations:  {medullary_dilation_iterations}\n")
        log.insert(tk.END, f"  Min cortex voxels:              {min_cortex_voxels} voxels\n")
        log.insert(tk.END, f"  Pore min size:                  {pore_min_size} voxels\n\n")

        log.insert(tk.END, "SCRIPTS TO RUN:\n")
        log.insert(tk.END, f"  Script 01 (Calibration):        {run_s1}\n")
        log.insert(tk.END, f"  Script 02 (Band + Rings):       {run_s2}\n")
        log.insert(tk.END, f"  Script 03 (Cortex/Trabecula):   {run_s3}\n\n")
        log.see(tk.END)

        # --- Script 01: Calibration ---
        if run_s1:
            if stop_flag.cancelled:
                log.insert(tk.END, "⚠ Cancelled before Script 01.\n")
                log.see(tk.END)
                return

            log.insert(tk.END, "━" * 80 + "\n")
            log.insert(tk.END, "[01/03] Calibration + bone mask...\n")
            log.insert(tk.END, "━" * 80 + "\n")
            log.see(tk.END)

            try:
                run_script_01(specimen_base, side, base_dir=base_dir)
                log.insert(tk.END, "✓ Script 01 completed successfully.\n\n")
            except TypeError:
                run_script_01(specimen_base, side)
                log.insert(tk.END, "✓ Script 01 completed successfully (no base_dir argument).\n\n")
            except Exception as e:
                log.insert(tk.END, f"✗ Script 01 failed: {e}\n\n")
                raise
            log.see(tk.END)
        else:
            log.insert(tk.END, "Skipping Script 01 (unchecked in GUI).\n\n")

        # --- Script 02: Band and rings ---
        if run_s2:
            if stop_flag.cancelled:
                log.insert(tk.END, "⚠ Cancelled before Script 02.\n")
                log.see(tk.END)
                return

            log.insert(tk.END, "━" * 80 + "\n")
            log.insert(tk.END, "[02/03] Cortical band + rings detection...\n")
            log.insert(tk.END, "━" * 80 + "\n")
            log.see(tk.END)

            try:
                run_script_02(
                    specimen_base,
                    side,
                    voxel_um=voxel_um,
                    band_mm=band_mm,
                    grow_radius=grow_radius,
                    outer_ring_thickness=outer_ring_thickness,
                    inner_ring_thickness=inner_ring_thickness,
                    base_dir=base_dir,
                )
                log.insert(tk.END, "✓ Script 02 completed successfully.\n\n")
            except TypeError:
                run_script_02(
                    specimen_base,
                    side,
                    voxel_um=voxel_um,
                    band_mm=band_mm,
                    grow_radius=grow_radius,
                    outer_ring_thickness=outer_ring_thickness,
                    inner_ring_thickness=inner_ring_thickness,
                )
                log.insert(tk.END, "✓ Script 02 completed successfully (no base_dir argument).\n\n")
            except Exception as e:
                log.insert(tk.END, f"✗ Script 02 failed: {e}\n\n")
                raise
            log.see(tk.END)
        else:
            log.insert(tk.END, "Skipping Script 02 (unchecked in GUI).\n\n")

        # --- Script 03: Cortex / Trabecula segmentation ---
        if run_s3:
            if stop_flag.cancelled:
                log.insert(tk.END, "⚠ Cancelled before Script 03.\n")
                log.see(tk.END)
                return

            log.insert(tk.END, "━" * 80 + "\n")
            log.insert(tk.END, "[03/03] Cortex / Trabecula segmentation...\n")
            log.insert(tk.END, "━" * 80 + "\n")
            log.see(tk.END)

            try:
                run_script_03(
                    specimen_base,
                    side,
                    trab_min=trab_min,
                    cort_min=cort_min,
                    cort_max=cort_max,
                    min_trab_cluster_size=min_trab_cluster_size,
                    min_band_trab_size=min_band_trab_size,
                    medullary_dilation_iterations=medullary_dilation_iterations,
                    min_cortex_voxels=min_cortex_voxels,
                    pore_min_size=pore_min_size,
                    base_dir=base_dir,
                )
                log.insert(tk.END, "✓ Script 03 completed successfully.\n\n")
            except TypeError:
                run_script_03(
                    specimen_base,
                    side,
                    trab_min=trab_min,
                    cort_min=cort_min,
                    cort_max=cort_max,
                    min_trab_cluster_size=min_trab_cluster_size,
                    min_band_trab_size=min_band_trab_size,
                    medullary_dilation_iterations=medullary_dilation_iterations,
                    min_cortex_voxels=min_cortex_voxels,
                    pore_min_size=pore_min_size,
                )
                log.insert(tk.END, "✓ Script 03 completed successfully (no base_dir argument).\n\n")
            except Exception as e:
                log.insert(tk.END, f"✗ Script 03 failed: {e}\n\n")
                raise
            log.see(tk.END)
        else:
            log.insert(tk.END, "Skipping Script 03 (unchecked in GUI).\n\n")

        if not stop_flag.cancelled:
            log.insert(tk.END, "=" * 80 + "\n")
            log.insert(tk.END, "✓ SEGMENTATION PIPELINE COMPLETED SUCCESSFULLY\n")
            log.insert(tk.END, "=" * 80 + "\n")
            log.see(tk.END)
            messagebox.showinfo(
                "Success",
                f"Segmentation completed successfully for {specimen_id}!\n\n"
                f"Output saved to segmentation_final/ folder.",
            )
        else:
            log.insert(tk.END, "Pipeline cancelled by user.\n")
            log.see(tk.END)

    except Exception as e:
        log.insert(tk.END, f"\n✗ ERROR: {str(e)}\n")
        log.see(tk.END)
        messagebox.showerror("Segmentation error", f"Pipeline failed:\n\n{e}")


def main():
    root = tk.Tk()
    root.title("Bone Segmentation Pipeline - Full Parameter Control")
    root.geometry("1300x900")

    root.grid_rowconfigure(4, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    stop_flag = SimpleNamespace(cancelled=False)
    worker_thread = SimpleNamespace(thread=None)

    # ========================================================================
    # SECTION 1: BASIC SPECIMEN INFO
    # ========================================================================
    section1 = ttk.LabelFrame(root, text="Specimen Information", padding=10)
    section1.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

    tk.Label(section1, text="Specimen Base ID:", font=("Arial", 10, "bold")).grid(
        row=0, column=0, sticky="w", padx=5, pady=5
    )
    spec_var = tk.StringVar(value=SPECIMEN_BASE_DEFAULT)
    tk.Entry(section1, textvariable=spec_var, width=20, font=("Arial", 10)).grid(
        row=0, column=1, sticky="w", padx=5, pady=5
    )

    tk.Label(section1, text="Side:", font=("Arial", 10, "bold")).grid(
        row=0, column=2, sticky="w", padx=5, pady=5
    )
    side_var = tk.StringVar(value=SIDE_DEFAULT)
    side_menu = tk.OptionMenu(section1, side_var, "L", "R")
    side_menu.config(width=5, font=("Arial", 10))
    side_menu.grid(row=0, column=3, sticky="w", padx=5, pady=5)

    # ========================================================================
    # SECTION 2: DENSITY & GEOMETRY PARAMETERS
    # ========================================================================
    section2 = ttk.LabelFrame(root, text="Density & Geometry Parameters", padding=10)
    section2.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)

    params_density = [
        ("Voxel Size (µm):", "32.0"),
        ("Cortical Band Depth (mm):", "2.0"),
        ("Trabecular Min (mgHA):", "200.0"),
        ("Cortical Min (mgHA):", "600.0"),
        ("Cortical Max (mgHA):", "1200.0"),
    ]

    density_vars = {}
    for idx, (label, default) in enumerate(params_density):
        tk.Label(section2, text=label, font=("Arial", 10)).grid(
            row=idx, column=0, sticky="w", padx=5, pady=5
        )
        var = tk.StringVar(value=default)
        tk.Entry(section2, textvariable=var, width=20, font=("Arial", 10)).grid(
            row=idx, column=1, sticky="w", padx=5, pady=5
        )
        density_vars[label] = var

    voxel_var = density_vars["Voxel Size (µm):"]
    band_var = density_vars["Cortical Band Depth (mm):"]
    trab_var = density_vars["Trabecular Min (mgHA):"]
    cort_min_var = density_vars["Cortical Min (mgHA):"]
    cort_max_var = density_vars["Cortical Max (mgHA):"]

    # ========================================================================
    # SECTION 3: SEGMENTATION REFINEMENT PARAMETERS
    # ========================================================================
    section3 = ttk.LabelFrame(root, text="Segmentation Refinement Parameters", padding=10)
    section3.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)

    params_refinement = [
        ("Grow Radius (voxels):", "20",
         "Radius for growing regions in band/rings detection"),
        ("Outer Ring Thickness (voxels):", "2",
         "Thickness of outer ring in band/rings detection"),
        ("Inner Ring Thickness (voxels):", "2",
         "Thickness of inner ring in band/rings detection"),
        ("Min Trabecular Cluster Size (voxels):", "300",
         "Remove trabecular clusters smaller than this"),
        ("Min Band Trabecular Size (voxels):", "50",
         "Remove isolated trabecular in band smaller than this"),
        ("Medullary Dilation Iterations:", "2",
         "Iterations for medullary space dilation (connectivity tolerance)"),
        ("Min Cortex Voxels (voxels):", "3000",
         "Remove cortical clusters smaller than this"),
        ("Pore Min Size (voxels):", "20",
         "Remove intracortical pores smaller than this"),
    ]

    refinement_vars = {}
    for idx, (label, default, tooltip) in enumerate(params_refinement):
        tk.Label(section3, text=label, font=("Arial", 10)).grid(
            row=idx, column=0, sticky="w", padx=5, pady=5
        )
        var = tk.StringVar(value=default)
        entry = tk.Entry(section3, textvariable=var, width=20, font=("Arial", 10))
        entry.grid(row=idx, column=1, sticky="w", padx=5, pady=5)
        tk.Label(section3, text=tooltip, font=("Arial", 8, "italic"), fg="gray").grid(
            row=idx, column=2, sticky="w", padx=5, pady=5
        )
        refinement_vars[label] = var

    grow_radius_var = refinement_vars["Grow Radius (voxels):"]
    outer_ring_var = refinement_vars["Outer Ring Thickness (voxels):"]
    inner_ring_var = refinement_vars["Inner Ring Thickness (voxels):"]
    trab_cluster_var = refinement_vars["Min Trabecular Cluster Size (voxels):"]
    band_trab_var = refinement_vars["Min Band Trabecular Size (voxels):"]
    medullary_iter_var = refinement_vars["Medullary Dilation Iterations:"]
    cortex_voxels_var = refinement_vars["Min Cortex Voxels (voxels):"]
    pore_var = refinement_vars["Pore Min Size (voxels):"]

    # ========================================================================
    # SECTION 4: DATA DIRECTORY + SCRIPT SELECTION
    # ========================================================================
    section4 = ttk.LabelFrame(root, text="Data Directory & Script Selection", padding=10)
    section4.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

    tk.Label(section4, text="Project/Data Directory:", font=("Arial", 10, "bold")).grid(
        row=0, column=0, sticky="w", padx=5, pady=5
    )

    data_dir_var = tk.StringVar(value=str(PROJECT_ROOT))

    data_dir_entry = tk.Entry(
        section4,
        textvariable=data_dir_var,
        width=70,
        font=("Arial", 9),
    )
    data_dir_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

    def browse_data_dir():
        selected_dir = filedialog.askdirectory(
            title="Select Data Directory",
            initialdir=data_dir_var.get() if Path(data_dir_var.get()).exists() else str(PROJECT_ROOT),
        )
        if selected_dir:
            data_dir_var.set(selected_dir)

    browse_btn = tk.Button(
        section4,
        text="Browse...",
        command=browse_data_dir,
        font=("Arial", 9, "bold"),
        padx=10,
    )
    browse_btn.grid(row=0, column=2, sticky="w", padx=5, pady=5)

    section4.grid_columnconfigure(1, weight=1)

    scripts_frame = ttk.LabelFrame(section4, text="Scripts to Run", padding=5)
    scripts_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=(10, 5))

    run_s1_var = tk.BooleanVar(value=True)
    run_s2_var = tk.BooleanVar(value=True)
    run_s3_var = tk.BooleanVar(value=True)

    tk.Checkbutton(
        scripts_frame,
        text="Run Script 01 (Calibration + bone mask)",
        variable=run_s1_var,
        font=("Arial", 9),
    ).grid(row=0, column=0, sticky="w", padx=5, pady=2)

    tk.Checkbutton(
        scripts_frame,
        text="Run Script 02 (Cortical band + rings)",
        variable=run_s2_var,
        font=("Arial", 9),
    ).grid(row=1, column=0, sticky="w", padx=5, pady=2)

    tk.Checkbutton(
        scripts_frame,
        text="Run Script 03 (Cortex / Trabecula segmentation)",
        variable=run_s3_var,
        font=("Arial", 9),
    ).grid(row=2, column=0, sticky="w", padx=5, pady=2)

    # ========================================================================
    # SECTION 5: LOG WINDOW
    # ========================================================================
    tk.Label(root, text="Pipeline Output Log:", font=("Arial", 10, "bold")).grid(
        row=3, column=0, sticky="w", padx=10, pady=(10, 5)
    )

    log_frame = tk.Frame(root)
    log_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

    log = tk.Text(log_frame, wrap="word", font=("Courier", 9), height=15)
    log.grid(row=0, column=0, sticky="nsew")

    y_scroll = tk.Scrollbar(log_frame, orient="vertical", command=log.yview)
    y_scroll.grid(row=0, column=1, sticky="ns")
    log.configure(yscrollcommand=y_scroll.set)

    log_frame.grid_rowconfigure(0, weight=1)
    log_frame.grid_columnconfigure(0, weight=1)

    # ========================================================================
    # CONTROL BUTTONS
    # ========================================================================
    button_frame = tk.Frame(root)
    button_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    def on_run():
        specimen_base = spec_var.get().strip()
        side = side_var.get().strip()
        data_dir = Path(data_dir_var.get()).expanduser().resolve()

        if not specimen_base:
            messagebox.showwarning("Missing input", "Please provide specimen base ID (e.g., 23162).")
            return
        if side not in ("L", "R"):
            messagebox.showwarning("Invalid side", "Side must be 'L' or 'R'.")
            return
        if not data_dir.exists() or not data_dir.is_dir():
            messagebox.showwarning("Invalid directory", "Please select a valid existing data directory.")
            return

        try:
            trab_min = float(trab_var.get())
            cort_min = float(cort_min_var.get())
            cort_max = float(cort_max_var.get())
            band_mm = float(band_var.get())
            voxel_um = float(voxel_var.get())
            grow_radius = int(grow_radius_var.get())
            outer_ring_thickness = int(outer_ring_var.get())
            inner_ring_thickness = int(inner_ring_var.get())
            min_trab_cluster = int(trab_cluster_var.get())
            min_band_trab = int(band_trab_var.get())
            medullary_iter = int(medullary_iter_var.get())
            min_cortex_vox = int(cortex_voxels_var.get())
            pore_min = int(pore_var.get())
        except ValueError:
            messagebox.showwarning(
                "Invalid input",
                "Please enter valid numeric values for all parameters."
            )
            return

        if voxel_um <= 0 or band_mm <= 0 or grow_radius < 0 or outer_ring_thickness < 0 or inner_ring_thickness < 0:
            messagebox.showwarning(
                "Invalid input",
                "Voxel size, band depth, and ring thicknesses must be non-negative."
            )
            return

        log.delete("1.0", tk.END)
        stop_flag.cancelled = False

        t = threading.Thread(
            target=run_pipeline,
            args=(
                str(data_dir),
                specimen_base,
                side,
                grow_radius,
                outer_ring_thickness,
                inner_ring_thickness,
                voxel_um,
                band_mm,
                trab_min,
                cort_min,
                cort_max,
                min_trab_cluster,
                min_band_trab,
                medullary_iter,
                min_cortex_vox,
                pore_min,
                run_s1_var.get(),
                run_s2_var.get(),
                run_s3_var.get(),
                log,
                stop_flag,
            ),
            daemon=True,
        )
        t.start()
        worker_thread.thread = t

    def on_cancel():
        stop_flag.cancelled = True
        log.insert(tk.END, "\n⚠ Cancellation requested. Waiting for current step to finish...\n")
        log.see(tk.END)

    run_btn = tk.Button(
        button_frame,
        text="▶ Run Segmentation",
        command=on_run,
        font=("Arial", 11, "bold"),
        bg="#4CAF50",
        fg="white",
        padx=20,
        pady=10,
    )
    run_btn.pack(side=tk.LEFT, padx=5)

    cancel_btn = tk.Button(
        button_frame,
        text="⊗ Cancel",
        command=on_cancel,
        font=("Arial", 11, "bold"),
        bg="#f44336",
        fg="white",
        padx=20,
        pady=10,
    )
    cancel_btn.pack(side=tk.LEFT, padx=5)

    # ========================================================================
    # FOOTER
    # ========================================================================
    footer_frame = tk.Frame(root, bg="white", relief="sunken", bd=1)
    footer_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=0, pady=0)

    tk.Label(
        footer_frame,
        text="Bone Segmentation Pipeline v2.0 - Full Parameter Control",
        font=("Arial", 9, "bold"),
        bg="white",
    ).pack(side=tk.LEFT, padx=10, pady=5)

    tk.Label(
        footer_frame,
        text="Johannes Eichwalder | jeichwal@uwaterloo.ca",
        font=("Arial", 8),
        fg="gray50",
        bg="white",
    ).pack(side=tk.RIGHT, padx=10, pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()