#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script shifts a 3D image in the XYZ directions by user-provided amounts.

Usage:
    python shift_3d_image.py -F input_file --shift_x shift_x --shift_y shift_y --shift_z shift_z

Example:
    python shift_3d_image.py -F input_image.tif --shift_x 10 --shift_y 20 --shift_z 5

Arguments:
    -F, --input           Input file (TIFF or HDF5 format).
    --shift_x             Shift in the x direction (number of pixels).
    --shift_y             Shift in the y direction (number of pixels).
    --shift_z             Shift in the z direction (number of pixels).

Installation:
    conda create -y -n image_processing python==3.11
    pip install numpy scipy h5py tifffile scikit-image matplotlib

Created on Sat Sep 10 10:42:15 2022

Author: marcnol

Updated by OpenAI's GPT-4

"""

import os, argparse, sys
from skimage import io
from tifffile import imwrite
import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import shift as shift_image
import h5py

def read_image(file_path):
    """Read an image from a file (TIFF or HDF5)."""
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            image = f['image'][:]
    else:
        image = io.imread(file_path).squeeze()
    return image

def write_image(image, file_path):
    """Write an image to a file (TIFF or HDF5)."""
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('image', data=image, compression='gzip')
    else:
        imwrite(file_path, image)

def parseArguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--input", help="Input file (TIFF or HDF5 format)", required=True)
    parser.add_argument("--shift_x", help="Shift image in x direction (number of pixels)", type=float, default=0.0)
    parser.add_argument("--shift_y", help="Shift image in y direction (number of pixels)", type=float, default=0.0)
    parser.add_argument("--shift_z", help="Shift image in z direction (number of pixels)", type=float, default=0.0)
    
    args = parser.parse_args()

    p = {
        "file": args.input,
        "shift_x": args.shift_x,
        "shift_y": args.shift_y,
        "shift_z": args.shift_z,
    }
    
    return p

if __name__ == "__main__":
    # parameters
    p = parseArguments()
    print("This algorithm shifts a 3D volume in TIF or HDF5 format by a user-provided amount.")
    print("Use the shift values as produced by pyHiM or our other registration methods.")

    file = p["file"]
    
    # Load image file
    print(f"$ Loading file: {file}")
    imageRaw = read_image(file)
    
    im_size_out = imageRaw.shape
    print(f"$ Output image size: {im_size_out}")
    
    # Shift image
    shift = np.zeros((3))
    shift[0], shift[1], shift[2] = p["shift_z"], p["shift_x"], p["shift_y"]
    print(f"$ Shifts that will be applied (ZXY): {shift}")
    im = shift_image(imageRaw, shift)
    
    # Save output data
    output_ext = 'h5' if file.endswith('.h5') else 'tif'
    output_file = file.split('.')[0] + '_shifted.' + output_ext
    
    print(f"$ Output file: {output_file}")
    
    write_image(im, output_file)
    
    plt.imshow(np.sum(100 * im, axis=0), cmap='Reds')
    plt.show()
