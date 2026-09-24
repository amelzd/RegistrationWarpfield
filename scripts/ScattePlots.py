#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:17:27 2026

@author: zidoum
"""
import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table

## import localizations (pixel 3D localizations)



file1 = ("/mnt/grey/DATA/users/zidoum/Pyhim_dev/Register_3D_deformations/TestMerfish/TestLocalizations/localize_3d/data/"
    "localizations_3D_barcode_unregistered.ecsv")

file2 = ("/mnt/grey/DATA/users/zidoum/Pyhim_dev/Register_3D_deformations/TestMerfish/TestLocalizations/localize_3d/data/localizations_3D_barcode.ecsv")

## import traces (in x,y,z microns)

# read tables as pd
df1 = Table.read(file1, format="ascii.ecsv").to_pandas()
df2 = Table.read(file2, format="ascii.ecsv").to_pandas()
columns = [
    "Buid",
    "ROI #",
    "CellID #",
    "Barcode #",
    "id",
    "zcentroid",
    "xcentroid",
    "ycentroid",
    "snr",
    "spot_pixel_percentage",
    "skew",
    "patch_size",
    "object_class",
    "mean_intensity",
    "flux",
    "roundness",
]

# Keep only Buids present in both tables (normally identical number)
common_buids = np.intersect1d(df1["Buid"].unique(),df2["Buid"].unique())

df1 = df1[df1["Buid"].isin(common_buids)].copy()
df2 = df2[df2["Buid"].isin(common_buids)].copy()
#index same order
df1 = df1.sort_values("Buid").reset_index(drop=True)
df2 = df2.sort_values("Buid").reset_index(drop=True)

# Check matching
assert np.array_equal(
    df1["Buid"].values,
    df2["Buid"].values
), "Buid matching failed"



x1 = df1["xcentroid"].values
y1 = df1["ycentroid"].values
z1 = df1["zcentroid"].values

x2 = df2["xcentroid"].values
y2 = df2["ycentroid"].values
z2 = df2["zcentroid"].values


# for each cell, cor each barcode 
for cell_id in sorted(df1["CellID #"].unique()):

    for barcode in sorted(df1.loc[df1["CellID #"] == cell_id, "Barcode #"].unique()):

        mask = (
            (df1["CellID #"] == cell_id) &
            (df1["Barcode #"] == barcode)
        )

        if not np.any(mask):
            continue

        # Coordinates for this cell + barcode
        xx1 = x1[mask]
        yy1 = y1[mask]
        zz1 = z1[mask]

        xx2 = x2[mask]
        yy2 = y2[mask]
        zz2 = z2[mask]

        # plot
        fig = plt.figure(figsize=(16, 12))

        ax_xy = fig.add_subplot(221)
        ax_xz = fig.add_subplot(222)
        ax_yz = fig.add_subplot(223)
        ax_3d = fig.add_subplot(224, projection="3d")


        ax_xy.scatter(
            xx1, yy1,
            s=15,
            alpha=0.7,
            label="Unregistered",
        )

        ax_xy.scatter(
            xx2, yy2,
            s=15,
            alpha=0.7,
            label="Registered",
        )

        # Draw displacement between corresponding spots
        for i in range(len(xx1)):
            ax_xy.plot(
                [xx1[i], xx2[i]],
                [yy1[i], yy2[i]],
                alpha=0.3,
                linewidth=0.5,
            )

        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_title("XY")
        ax_xy.legend()



        ax_xz.scatter(
            xx1, zz1,
            s=15,
            alpha=0.7,
        )

        ax_xz.scatter(
            xx2, zz2,
            s=15,
            alpha=0.7,
        )

        for i in range(len(xx1)):
            ax_xz.plot(
                [xx1[i], xx2[i]],
                [zz1[i], zz2[i]],
                alpha=0.3,
                linewidth=0.5,
            )

        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_title("XZ")



        ax_yz.scatter(
            yy1, zz1,
            s=15,
            alpha=0.7,
        )

        ax_yz.scatter(
            yy2, zz2,
            s=15,
            alpha=0.7,
        )

        for i in range(len(xx1)):
            ax_yz.plot(
                [yy1[i], yy2[i]],
                [zz1[i], zz2[i]],
                alpha=0.3,
                linewidth=0.5,
            )

        ax_yz.set_xlabel("Y")
        ax_yz.set_ylabel("Z")
        ax_yz.set_title("YZ")


        ax_3d.scatter(
            xx1, yy1, zz1,
            s=15,
            alpha=0.7,
            label="Unregistered",
        )

        ax_3d.scatter(
            xx2, yy2, zz2,
            s=15,
            alpha=0.7,
            label="Registered",
        )

        for i in range(len(xx1)):
            ax_3d.plot(
                [xx1[i], xx2[i]],
                [yy1[i], yy2[i]],
                [zz1[i], zz2[i]],
                alpha=0.3,
                linewidth=0.5,
            )

        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("3D")


        fig.suptitle(
            f"CellID #{cell_id} — Barcode #{barcode}\n"
            f"{np.sum(mask)} matched spots",
            fontsize=14
        )

        plt.tight_layout()
        plt.show()
        


# RT28 = reference
# If an RT spot is within  of ANY RT28 spot
# in 3D, it is considered overlapping.
#
# RT28-only spots     -> blue
# RT-only spots       -> orange
# Overlapping spots   -> orange + blue
reference_barcode = 28
overlap_threshold = 0.05
pixelSizeXY =0.1
pixelSizeZ=0.25
for cell_id in sorted(df2["CellID #"].unique()):

    # Get RT28 spots for this cell

    ref_mask = (
        (df2["CellID #"] == cell_id) &
        (df2["Barcode #"] == reference_barcode)
    )

    if not np.any(ref_mask):
        continue

    ref = df2.loc[ref_mask]

    x_ref = ref["xcentroid"].values
    y_ref = ref["ycentroid"].values
    z_ref = ref["zcentroid"].values

    # Coordinates of RT28 spots
    ref_xyz = np.column_stack([
        x_ref,
        y_ref,
        z_ref
    ])

    for barcode in sorted(
        df2.loc[
            df2["CellID #"] == cell_id,
            "Barcode #"
        ].unique()
    ):

        if barcode == reference_barcode:
            continue

        current_mask = (
            (df2["CellID #"] == cell_id) &
            (df2["Barcode #"] == barcode)
        )

        current = df2.loc[current_mask]

        if len(current) == 0:
            continue

        x_cur = current["xcentroid"].values
        y_cur = current["ycentroid"].values
        z_cur = current["zcentroid"].values

        cur_xyz = np.column_stack([
            x_cur,
            y_cur,
            z_cur
        ])


        # Calculate 3D distance from every RT spot
        # to every RT28 spot
        
        dx = (cur_xyz[:, None, 0]- ref_xyz[None, :, 0]) * pixelSizeXY
        
        dy = ( cur_xyz[:, None, 1]- ref_xyz[None, :, 1] ) * pixelSizeXY
        
        dz = ( cur_xyz[:, None, 2] - ref_xyz[None, :, 2]) * pixelSizeZ
        
        distances = np.sqrt(
            dx**2 +
            dy**2 +
            dz**2)

        # For each RT spot:
        # minimum distance to any RT28 spot
        min_distance = np.min(distances, axis=1)

        # RT spots that overlap RT28 within 50 nm
        overlap_cur = min_distance < overlap_threshold


        min_distance_ref = np.min(distances, axis=0)

        overlap_ref = min_distance_ref < overlap_threshold


        fig = plt.figure(figsize=(16, 12))

        ax_xy = fig.add_subplot(221)
        ax_xz = fig.add_subplot(222)
        ax_yz = fig.add_subplot(223)
        ax_3d = fig.add_subplot(224, projection="3d")

        # XY

        # RT28 spots that do NOT overlap
        ax_xy.scatter(
            x_ref[~overlap_ref],
            y_ref[~overlap_ref],
            s=25,
            alpha=0.7,
            label="RT28",
        )

        # RT spots that do NOT overlap
        ax_xy.scatter(
            x_cur[~overlap_cur],
            y_cur[~overlap_cur],
            s=25,
            alpha=0.7,
            label=f"RT{barcode}",
        )

        # Overlapping spots
        ax_xy.scatter(
            x_ref[overlap_ref],
            y_ref[overlap_ref],
            s=50,
            alpha=0.9,
            color="orange",
            edgecolor="blue",
            linewidth=1.5,
            label="<50 nm overlap",
        )

        ax_xy.set_xlabel("X ")
        ax_xy.set_ylabel("Y ")
        ax_xy.set_title("XY")
        ax_xy.legend()

        # XZ
        ax_xz.scatter(
            x_ref[~overlap_ref],
            z_ref[~overlap_ref],
            s=25,
            alpha=0.7,
            label="RT28",
        )

        ax_xz.scatter(
            x_cur[~overlap_cur],
            z_cur[~overlap_cur],
            s=25,
            alpha=0.7,
            label=f"RT{barcode}",
        )

        ax_xz.scatter(
            x_ref[overlap_ref],
            z_ref[overlap_ref],
            s=50,
            alpha=0.9,
            color="orange",
            edgecolor="blue",
            linewidth=1.5,
            label="< 50 nm overlap",
        )

        ax_xz.set_xlabel("X ")
        ax_xz.set_ylabel("Z ")
        ax_xz.set_title("XZ")
        ax_xz.legend()

        # YZ
        ax_yz.scatter(
            y_ref[~overlap_ref],
            z_ref[~overlap_ref],
            s=25,
            alpha=0.7,
            label="RT28",
        )

        ax_yz.scatter(
            y_cur[~overlap_cur],
            z_cur[~overlap_cur],
            s=25,
            alpha=0.7,
            label=f"RT{barcode}",
        )

        ax_yz.scatter(
            y_ref[overlap_ref],
            z_ref[overlap_ref],
            s=50,
            alpha=0.9,
            color="orange",
            edgecolor="blue",
            linewidth=1.5,
            label="<  overlap",
        )

        ax_yz.set_xlabel("Y ")
        ax_yz.set_ylabel("Z")
        ax_yz.set_title("YZ")
        ax_yz.legend()

        # 3D
        ax_3d.scatter(
            x_ref[~overlap_ref],
            y_ref[~overlap_ref],
            z_ref[~overlap_ref],
            s=25,
            alpha=0.7,
            label="RT28",
        )

        ax_3d.scatter(
            x_cur[~overlap_cur],
            y_cur[~overlap_cur],
            z_cur[~overlap_cur],
            s=25,
            alpha=0.7,
            label=f"RT{barcode}",
        )

        ax_3d.scatter(
            x_ref[overlap_ref],
            y_ref[overlap_ref],
            z_ref[overlap_ref],
            s=60,
            alpha=0.9,
            color="orange",
            edgecolor="blue",
            linewidth=1.5,
            label="< overlap",
        )

        ax_3d.set_xlabel("X ")
        ax_3d.set_ylabel("Y ")
        ax_3d.set_zlabel("Z ")
        ax_3d.set_title("3D")
        ax_3d.legend()


        n_overlap_rt = np.sum(overlap_cur)
        n_overlap_ref = np.sum(overlap_ref)

        fig.suptitle(
            f"CellID #{cell_id} — RT{barcode} vs RT28\n"
            f"RT28: {len(x_ref)} spots | "
            f"RT{barcode}: {len(x_cur)} spots | "
            f"Overlap < 50nm : {n_overlap_rt} RT spots",
            fontsize=14
        )

        plt.tight_layout()
        plt.show()
        
        # Calculate overlap counts and percentages
       
        n_overlap_rt = np.sum(overlap_cur)
        n_overlap_ref = np.sum(overlap_ref)
        
        # Percentage of RT spots that overlap RT28
        overlap_percent_rt = (
            n_overlap_rt / len(x_cur) * 100
            if len(x_cur) > 0 else 0
        )
        
        # Percentage of RT28 spots that overlap RT
        overlap_percent_ref = (
            n_overlap_ref / len(x_ref) * 100
            if len(x_ref) > 0 else 0
        )
        
        fig.suptitle(
            f"CellID #{cell_id} — RT{barcode} vs RT28\n"
            f"RT28: {len(x_ref)} spots | "
            f"RT{barcode}: {len(x_cur)} spots\n"
            f"RT{barcode} overlap: {n_overlap_rt} "
            f"({overlap_percent_rt:.2f}%) | "
            f"RT28 overlap: {n_overlap_ref} "
            f"({overlap_percent_ref:.2f}%)",
            fontsize=14
        )
        
        # Print overlap information

        print(
            f"Cell {cell_id} | "
            f"RT28 vs RT{barcode} | "
            f"RT28 spots = {len(x_ref)} | "
            f"RT{barcode} spots = {len(x_cur)} | "
            f"RT spots < 50 nm from RT28 = {n_overlap_rt} "
            f"({overlap_percent_rt:.2f}%) | "
            f"RT28 spots < 50 nm from RT = {n_overlap_ref} "
            f"({overlap_percent_ref:.2f}%)"
        )


"""
#### Comparison to barcode 28 (groundtruth RT)
reference_barcode = 28


for cell_id in sorted(df2["CellID #"].unique()):

    # RT28 
    ref_mask = (
        (df2["CellID #"] == cell_id) &
        (df2["Barcode #"] == reference_barcode)
    )

    if not np.any(ref_mask):
        continue

    ref = df2.loc[ref_mask]

    x_ref = ref["xcentroid"].values
    y_ref = ref["ycentroid"].values
    z_ref = ref["zcentroid"].values

    # Loop over every other RT barcode

    for barcode in sorted(
        df2.loc[
            df2["CellID #"] == cell_id,
            "Barcode #"
        ].unique()
    ):

        # Skip RT28 itself
        if barcode == reference_barcode:
            continue

        current_mask = (
            (df2["CellID #"] == cell_id) &
            (df2["Barcode #"] == barcode)
        )

        current = df2.loc[current_mask]

        if len(current) == 0:
            continue

        x_cur = current["xcentroid"].values
        y_cur = current["ycentroid"].values
        z_cur = current["zcentroid"].values


        fig = plt.figure(figsize=(16, 12))

        ax_xy = fig.add_subplot(221)
        ax_xz = fig.add_subplot(222)
        ax_yz = fig.add_subplot(223)
        ax_3d = fig.add_subplot(224, projection="3d")


        # XY
        ax_xy.scatter(
            x_ref,
            y_ref,
            s=25,
            alpha=0.7,
            label="RT28",
        )

        ax_xy.scatter(
            x_cur,
            y_cur,
            s=25,
            alpha=0.7,
            label=f"RT{barcode}",
        )

        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_title("XY")
        ax_xy.legend()

        # XZ
        ax_xz.scatter(
            x_ref,
            z_ref,
            s=25,
            alpha=0.7,
            label="RT28",
        )

        ax_xz.scatter(
            x_cur,
            z_cur,
            s=25,
            alpha=0.7,
            label=f"RT{barcode}",
        )

        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_title("XZ")
        ax_xz.legend()


        # YZ
        ax_yz.scatter(
            y_ref,
            z_ref,
            s=25,
            alpha=0.7,
            label="RT28",
        )

        ax_yz.scatter(
            y_cur,
            z_cur,
            s=25,
            alpha=0.7,
            label=f"RT{barcode}",
        )

        ax_yz.set_xlabel("Y")
        ax_yz.set_ylabel("Z")
        ax_yz.set_title("YZ")
        ax_yz.legend()

        # 3D
        ax_3d.scatter(
            x_ref,
            y_ref,
            z_ref,
            s=25,
            alpha=0.7,
            label="RT28",
        )

        ax_3d.scatter(
            x_cur,
            y_cur,
            z_cur,
            s=25,
            alpha=0.7,
            label=f"RT{barcode}",
            )
            
        plt.tight_layout()
        plt.show()
"""
