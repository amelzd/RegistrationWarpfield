#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 08:51:42 2026

@author: zidoum

compute warpfield function edited from merfish3d/utils/registration.py 

usage : python Warpfield_register3D.py --reference <reference fiducial image> --moving <fiducial to move> --output <output path>


usage example :
python Warpfield_register3D.py \
    --reference /mnt/grey/DATA/users/zidoum/Pyhim_dev/Register_3D_deformations/TestMerfish/Test_cycle/024/unregistered/scan_001_RT28_024_ROI_converted_decon_ch00.tif \
    --moving /mnt/grey/DATA/users/zidoum/Pyhim_dev/Register_3D_deformations/TestMerfish/Test_cycle/024/unregistered/scan_001_RT284_024_ROI_converted_decon_ch00.tif \
    --zbin 2 \
    --xybin 1 \
    --output ./
    


"""



import gc
import argparse
import numpy as np
import cupy as cp
import os
import time
import matplotlib.pyplot as plt
import tifffile as tiff

from skimage import exposure
from numpy.typing import ArrayLike
from scipy.ndimage import shift as shift_image
from scipy.ndimage import zoom

#import warpfield
from warpfield.warp import warp_volume
from warpfield import Recipe, register_volumes
from warpfield.register import WarpMap

import sys
import subprocess

'''
Install : 
conda create -n warpfield python=3.11
conda activate warpfield
conda install -c conda-forge cupy cuda-version=12 
pip install warpfield
python -m pip install matplotlib tifffile
'''


def compute_warpfield(
    img_ref: ArrayLike, 
    img_trg: ArrayLike,
    tomove_image: ArrayLike,
    h5_path: str,
    gpu_id: int = 0
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike | None] :
    """
    Compute the warpfield to warp a target image to a reference image. Applies warp_map to tomoveimage.
    """

    cp.cuda.Device(gpu_id).use()

    recipe = ( Recipe() )  # initialized with a translation level, followed by an affine registration level
    recipe.pre_filter.clip_thresh = 0  # clip DC background, if present
    
    recipe.pre_filter.soft_edge = [4,33,33]


    # affine level properties
    recipe.levels[-1].repeats = 0

    # add non-rigid registration levels:
    recipe.add_level(block_size=[32,64,64]) 
    recipe.levels[-1].block_stride = 0.8
    recipe.levels[-1].smooth.sigmas = [1.0,1.0,1.0]
    recipe.levels[-1].repeats = 2

    recipe.add_level(block_size=[15, 32, 32]) # adjust block_size to make blocks roughly isotropic in real space. 
    recipe.levels[-1].block_stride = 0.8
    recipe.levels[-1].smooth.sigmas = [1.0, 1.0, 1.0] 
    recipe.levels[-1].smooth.long_range_ratio = 0.1
    recipe.levels[-1].repeats = 5
        
    recipe.add_level(block_size=[9,14,14])
    recipe.levels[-1].block_stride = 0.8
    recipe.levels[-1].smooth.sigmas = [1.0,1.0,1.0] 
    recipe.levels[-1].smooth.long_range_ratio = 0.1 # Long range ratio for double gaussian kernel.
    recipe.levels[-1].repeats = 5

    #register moving volume
    warped_image, warp_map, _ = register_volumes(
        ref=img_ref,
        vol=img_trg,
        recipe=recipe )

    # save warpfield as h5 
    warp_map.to_h5(
        h5_path,
        group="warp_map",
        compression="gzip",
        overwrite=True,
        )
    # save as np
    """
    warped_image = cp.asnumpy(warped_image).astype(np.float32)
    warp_field = cp.asnumpy(warp_map.warp_field).astype(np.float32)
    block_size = cp.asnumpy(warp_map.block_size).astype(np.float32)
    block_stride = cp.asnumpy(warp_map.block_stride).astype(np.float32)
    """
    tomove_registered = None
    
    # apply warp to other channel.s
    if tomove_image is not None:
        """
        offset = -(block_size/ block_stride/ 2)
        tomove_registered_cp = warp_volume(
            cp.asarray(tomove_image, dtype=cp.float32),
            cp.asarray(warp_field),
            cp.asarray(block_stride),
            cp.asarray(offset, dtype=cp.float32)
            )
        """
        tomove_registered_cp = warp_map.apply(cp.asarray(tomove_image, dtype=cp.float32))

        
        tomove_registered = cp.asnumpy(tomove_registered_cp).astype(np.float32)
        del tomove_registered_cp 

    
    del warp_map
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return (warped_image, warp_field, tomove_registered)
    
#### From pyhim_tools
class BothImgRbgFile:
    def __init__(self, image1, image2, tag='', title=''):
        self.image1 = image1
        self.image2 = image2
        self.tag = tag
        if title is None:
            self.title = tag  # gets title from tag
        else:
            self.title = title  # New attribute to hold the title

    def save(self, folder_path, basename):
        self.folder_path = folder_path
        self.basename = f"{basename}_{self.tag}_overlay"
        self.path_name = os.path.join( self.folder_path, self.basename + ".png")

        def normalize_contrast(img, low_percentile=2, high_percentile=99):
            """
            Normalize image using percentile-based contrast stretching.
            """

            img = img.astype(np.float32)
            low = np.percentile(img, low_percentile)
            high = np.percentile(img, high_percentile)

            # Avoid division by zero
            if high <= low:
                return np.zeros_like(img)

            img = (img - low) / (high - low)
            return np.clip(img, 0, 1)

        # Contrast enhancement
        img_1 = normalize_contrast(self.image1)
        img_2 = normalize_contrast(self.image2)

        # RGB overlay
        null_image = np.zeros_like(img_1)
        rgb = np.dstack([
            img_1,          # Red = image 1
            img_2,          # Green = image 2
            null_image      # Blue = 0
            ])

        fig, ax1 = plt.subplots()
        fig.set_size_inches((30, 30))
        ax1.imshow(rgb)
        ax1.axis("off")
        ax1.set_title(self.title)
        fig.savefig( self.path_name, bbox_inches="tight", pad_inches=0, dpi=150  )
        plt.close(fig)


def compute_intensity_np(displacement_field, z_plane):
    dx = displacement_field[0, z_plane, :, :]
    dy = displacement_field[1, z_plane, :, :]
    dz = displacement_field[2, z_plane, :, :]

    intensity = np.sqrt(dx**2 + dy**2 + dz**2)
    return [intensity, dx, dy, dz]

def plot_deformation_intensity(displacement_field, z_plane, output_prefix):
    intensity,_,_,_ = compute_intensity_np(displacement_field, z_plane)
    plt.figure(figsize=(10, 8))
    plt.imshow(intensity, cmap='Reds')
    plt.colorbar(label='Vector Field Intensity')
    plt.title(f'Intensity of Vector Field at Z-plane {z_plane}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(f"{output_prefix}_intensity_z{z_plane}.png")
    plt.close()

def plot_deformation_intensity_xyz(displacement_field, z_plane, output_prefix):
    data = compute_intensity_np(displacement_field, z_plane)
    titles = ["magnitude", "dx", "dy", "dz"]

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches((10, 10))
    ax = axes.ravel()

    for axis, img, title in zip(ax, data, titles):
        vmin, vmax = 0, np.max(img)
        cmap="YlOrRd"
        im = axis.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)    
        axis.set_xlabel('X-axis')
        axis.set_ylabel('Y-axis')
        cbar1 = fig.colorbar(im, ax=axis, shrink=0.5)
        cbar1.set_label('pixels')

    fig.tight_layout()
    fig.suptitle(f'Intensity of Vector Field at Z-plane {z_plane}')

    fig.savefig(f"{output_prefix}_DF_intensity_z{z_plane}.png")


def compute_direction_np(warp, z_plane):
    dx = warp[0, z_plane, :, :]
    dy = warp[1, z_plane, :, :]
    
    # Compute the direction in the XY plane (arctangent of dy/dx)
    direction = np.arctan2(dy, dx)
    
    # Normalize the direction to the range [0, 1] for color mapping
    norm = plt.Normalize(-np.pi, np.pi)
    direction_normalized = norm(direction)
    
    return direction_normalized

def plot_deformation_direction(displacement_field, z_plane, output_prefix):
    direction = compute_direction_np(displacement_field, z_plane)
    plt.figure(figsize=(10, 8))
    plt.imshow(direction, cmap='twilight', alpha=0.9, norm=plt.Normalize(-np.pi, np.pi))
    cbar = plt.colorbar(ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    cbar.set_label('Vector Field Direction (radians)')
    plt.title(f'Direction of Vector Field at Z-plane {z_plane}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(f"{output_prefix}_DF_direction_z{z_plane}.png")
    plt.close()

def process_batch(
    reference_path,
    input_dir,
    output_dir,
    gpu_id=0,
    xybin=1,
    zbin=2,
    shift=(0, 0, 0),
):
    """
    Register every *_ch00.tif file in input_dir against reference_path,
    excluding the reference image itself.
    """

    os.makedirs(output_dir, exist_ok=True)

    reference_path = os.path.abspath(reference_path)
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    # Find all files ending exactly with ch00.tif
    moving_files = sorted(
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.endswith("ch00.tif")
    )

    # Exclude reference
    moving_files = [
        path for path in moving_files
        if os.path.abspath(path) != reference_path
    ]

    if not moving_files:
        print(f"No files ending in 'ch00.tif' found in:")
        print(f"  {input_dir}")
        return

    print(f"\nReference:")
    print(f"  {reference_path}")

    print(f"\nFound {len(moving_files)} moving images:")

    for path in moving_files:
        print(f"  {path}")

    print()

    for i, moving_path in enumerate(moving_files, start=1):

        print("\n" + "=" * 80)
        print(f"Processing {i}/{len(moving_files)}")
        print(f"Moving image:")
        print(f"  {moving_path}")
        print("=" * 80)

        cmd = [
            sys.executable,
            os.path.abspath(__file__),

            "--reference",
            reference_path,

            "--moving",
            moving_path,

            "--zbin",
            str(zbin),

            "--xybin",
            str(xybin),

            "--output",
            output_dir,

            "--gpu",
            str(gpu_id),

            "--shift",
            str(shift[0]),
            str(shift[1]),
            str(shift[2]),
        ]

        print("\nRunning:")
        print(" ".join(cmd))
        print()

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print("\nWARNING: registration failed!")
            print(f"Image: {moving_path}")
            print(f"Return code: {result.returncode}")
        else:
            print("\nSuccessfully processed:")
            print(f"  {moving_path}")

    print("\n" + "=" * 80)
    print("Batch processing finished.")
    print("=" * 80)

  
def main():
    parser = argparse.ArgumentParser( description="Apply GPU deformation correction (warpfield optical flow).")

    parser.add_argument("--reference", required=True, help="Reference image ")
    parser.add_argument( "--moving", default=None,help="Moving image. Used for single-image registration.")
    parser.add_argument("--input_dir",default=None,help="Directory containing *_ch00.tif images to register.")
    
    parser.add_argument("--tomove", default = None, help="Image to apply same correction")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--xybin", type=int, default=1)
    parser.add_argument("--zbin", type=int, default=2)
    parser.add_argument("--shift", nargs=3, type=float, default=[0,0,0])
    parser.add_argument('--lower_threshold', type=float, default=0.9, help='Lower threshold for intensity adjustment in preprocessing.')
    parser.add_argument('--higher_threshold', type=float, default=0.9999999, help='Higher threshold for intensity adjustment in preprocessing.')

    args = parser.parse_args()
    
    start_time = time.time() 
    
    #### BATCH mode ####
    if args.moving is None and args.input_dir is None:
        parser.error("You must provide either --moving or --input_dir.")
    if args.moving is not None and args.input_dir is not None:
        parser.error("Use either --moving or --input_dir, not both.")
        
    if args.input_dir is not None:
        process_batch(reference_path=args.reference,input_dir=args.input_dir, output_dir=args.output,gpu_id=args.gpu,xybin=args.xybin, zbin=args.zbin,shift=args.shift,
                      )
        return
    
    #### single image ####
    # Load images
    print("Loading images.") 
    moving = tiff.imread(args.moving)
    moving_dtype = moving.dtype

    original_shape=moving.shape
    reference = tiff.imread(args.reference)
    
    tomove_dtype = None
    tomove_image = None
    if args.tomove is not None:
        tomove_image = tiff.imread(args.tomove)
        tomove_dtype = tomove_image.dtype

   
    # Apply (Register Global) Shifts
    if args.shift != [0,0,0]:
        shift = np.zeros((3))
        shift[0], shift[1], shift[2] = args.shift[2], args.shift[0], args.shift[1]
        moving = shift_image(moving, shift)

    # binning
    if args.xybin > 1 or args.zbin > 1 :
        moving = zoom(moving, (1.0 / args.zbin, 1.0 / args.xybin, 1.0 / args.xybin), order=1)
        reference = zoom(reference, (1.0 / args.zbin,  1.0 / args.xybin,  1.0 / args.xybin), order=1)
        
        if args.tomove is not None :
            tomove_image = zoom(tomove_image, ( 1.0 / args.zbin, 1.0 / args.xybin, 1.0 / args.xybin), order=1)
    

    base = os.path.splitext(os.path.basename(args.moving))[0]
    #base = filename.removesuffix(".nii.gz")
    h5_path = os.path.join(args.output, base +".h5")
    
    print("Correcting deformations.")
    moving_registered, warp_field, tomove_registered = compute_warpfield(
        reference,
        moving,
        tomove_image,
        h5_path,
        gpu_id=args.gpu )
    

    # RGB overlay
    os.makedirs(args.output, exist_ok=True)
    overlay = BothImgRbgFile(reference.max(axis=0), moving.max(axis=0), tag='reference_original')
    overlay.save(args.output, f"{base}_registered")
    overlay = BothImgRbgFile(reference.max(axis=0), moving_registered.max(axis=0), tag='reference_aligned')
    overlay.save(args.output,  f"{base}_registered")
    
    # Plot the intensity and direction of the deformation field at the center z-plane
    z_plane = warp_field.shape[1] // 2 # (3,z,x,y)
    plot_deformation_intensity_xyz( warp_field, z_plane, os.path.join(args.output, base))
    plot_deformation_direction( warp_field,  z_plane, os.path.join(args.output, base))

    
    
    # Upsample back to the original shape if binning was applied
    if args.zbin > 1 or args.xybin > 1 :
        zoom_factors = [original_shape[0] / moving_registered.shape[0],  # Z upsampling
                        original_shape[1] / moving_registered.shape[1],  # Y upsampling
                        original_shape[2] / moving_registered.shape[2]]  # X upsampling

        print(f"Zoom factors: {zoom_factors}")
        moving_registered = zoom(moving_registered, zoom_factors, order=1)
        if tomove_registered is not None :
            tomove_registered = zoom(tomove_registered, zoom_factors, order=1)
            
    # Restore moving image dtype
    if np.issubdtype(moving_dtype, np.integer):
        info = np.iinfo(moving_dtype)
        moving_registered = np.clip(moving_registered,info.min,  info.max).astype(moving_dtype)
    else:
        moving_registered = moving_registered.astype(moving_dtype)

    # Restore tomove image dtype
    if tomove_registered is not None and tomove_dtype is not None:
        if np.issubdtype(tomove_dtype, np.integer):
            info = np.iinfo(tomove_dtype)
            tomove_registered = np.clip(tomove_registered,info.min, info.max).astype(tomove_dtype)
        else:
            tomove_registered = tomove_registered.astype(tomove_dtype) 

    # saving outputs 
    print(f"Saving images in: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    tiff.imwrite(os.path.join(args.output, f"{base}_registered.tif"), moving_registered)
    
    if tomove_registered is not None and args.tomove is not None:
        base_2 = os.path.splitext(os.path.basename(args.tomove))[0]
        tiff.imwrite(os.path.join(args.output, f"{base_2}_registered.tif"),tomove_registered)
        
    #np.save(os.path.join(args.output, f"{base}_warp_field.npy"), warp_field)
    #saving tif as unsampled for later use
    #tiff.imwrite(os.path.join(args.output, f"{base}_warpfield.tiff"), warp_field.astype(np.float32) )

    elapsed_time = time.time() - start_time
    print(f'$ Script done in {elapsed_time} s')


if __name__ == "__main__":
    main()
