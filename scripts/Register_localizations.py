#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:05:27 2026

Apply deformation corrections to localizations based on barcode identity and deformation fields.

Usage:
    Warpfield_register_localizations.py --localizations localizations_3D_barcode_unregistered.dat --deformation_folder /path/to/deformation_folder --output localizations_3D_barcode.dat

Installation environment:

pip install astropy numpy simpleITK h5py

"""

import argparse
import os
import re
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
import h5py
from astropy.table import Table
from tqdm import tqdm

def read_localizations(file_path):
    """Read a localizations table from a file."""
    return Table.read(file_path, format='ascii.ecsv')

def write_localizations(table, file_path):
    """Write a localizations table to a file."""
    table.write(file_path, format='ascii.ecsv', overwrite=True)

def read_deformation_field(file_path):
    """Read a deformation field and return it as a NumPy array."""

    if file_path.endswith(('.tif', '.tiff', '.nii', '.nii.gz')):
        displacement_field = sitk.ReadImage(file_path)
        displacement_array = sitk.GetArrayFromImage(displacement_field)

    elif file_path.endswith(('.h5', '.hdf5')):
        with h5py.File(file_path, 'r') as h5_file:
            
            # format for warpfield.py DF
            displacement_array = h5_file['warp_map/warp_field'][:]
            """
            block_size = tuple(h5_file['warp_map/block_size'][:])
            block_stride = tuple(h5_file['warp_map/block_stride'][:])
            mov_shape = tuple(h5_file['warp_map/mov_shape'][:])
            ref_shape = tuple(h5_file['warp_map/ref_shape'][:])
        
        print("Original warp field shape:", displacement_array.shape)
        print("block_size:", block_size)
        print("block_stride:", block_stride)
        print("mov_shape:", mov_shape)
        print("ref_shape:", ref_shape)"""
        
        
        """# HDF5: (3, Z, Y, X) warpfield format 
        # NumPy/SimpleITK vector image: (Z, Y, X, 3)
        #displacement_array = np.moveaxis(displacement_array, 0, -1)"""
        
        #print("Displacement field shape:", displacement_array.shape)

    else:
        raise ValueError("Unsupported file format for deformation field.")

    return displacement_array


def apply_deformation(localizations, deformation_field, 
                      barcode_id,
                      z_binning=2.03125,
                      xy_binning=1, 
                      toleranceDrift_XY=3200, 
                      toleranceDrift_Z = 65):
    """Apply deformation corrections to localizations.
    Localization coordinates are assumed to be in coordinates : 
        65/2, 3200, 3200 
        
    warpfield was generated on :
        65 x 1600 x 1600
        
    original image size : 
        65 x 3200 x 3200
        
    Therefore :
        x = X_localization / xy_binning
        y = Y_localization / xy_binning
        z = Z_localization * z_binning
"""
    print("Unbinning the deformation field.")
    
    # original binned / deformation field block
    
    # for RAMM binned in xy + zbinned (pyhim)
    #original_shape = 32, 1600, 1600
    original_shape = 32, 3200, 3200
    #stride = 4,7,7
    
    # stride = ?,0.8,0.8
    
    scale = (
        original_shape[0] / deformation_field.shape[1],
        original_shape[1] / deformation_field.shape[2],
        original_shape[2] / deformation_field.shape[3])


    # 4D deformation field : (3,z,y,x)
    u0 = zoom(deformation_field[0], scale, order=3)
    u1 = zoom(deformation_field[1], scale, order=3)
    u2 = zoom(deformation_field[2], scale, order=3)
    
    print("deformation shape after unbinning:",u1.shape[1],u1.shape[2],u1.shape[0]) 
    # shape is 32,1600,1600
    counter=0
    counterspots=0
    for row in tqdm(localizations, desc="Processing localizations"):

        barcode_number=int(row["Barcode #"])
        if barcode_id == barcode_number:
            print(f"$ Registering barcode: {barcode_number}")    
            counterspots+=1
            
            # coordinates resized for deformation field
            #x, y, z = int(row['xcentroid']/ xy_binning),int( row['ycentroid']/xy_binning), int(row['zcentroid'])
            x, y, z = int(row['xcentroid']), int(row['ycentroid']),int((row['zcentroid']))

            
            if 0 <= x < u1.shape[2] and 0 <= y < u1.shape[1] and 0 <= z < u1.shape[0]:
                
                # displcement in X at (z,y,x)
                d0 = u0[z, y, x]
                # displacement in Y at (z,y,x)
                d1 = u1[z, y, x]
                # displcaement in Z at (z,y,x)
                d2 = u2[z, y, x]

                # checks that the DF is not trying to correct more than the tolerance allowsloc
                #if np.abs(dx)<toleranceDrift_XY and np.abs(dy)<toleranceDrift_XY and np.abs(dz)<toleranceDrift_Z:
                    
                row['xcentroid'] += -d0
                row['ycentroid'] += -d1 #*xy_binning
                row['zcentroid'] += -d2
                
                    #print(f"AFTER: x={row['xcentroid']}", f"y={row['ycentroid']}", f"z={row['zcentroid']}")
                    #print(f"BEFORE: x={row['xcentroid']}", f"y={row['ycentroid']}", f"z={row['zcentroid']}")
                    #print(f"$ Correcting {row['Buid']} barcode: {row['Barcode #']}| rxyz-final = ({row['xcentroid']},{row['ycentroid']},{row['zcentroid']}, dxyz={dx},{dy},{dz})")
                counter+=1
                    
            
    print(f"$ Corrected {counter} localizations out of {counterspots}.")
    
    return localizations


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply deformation corrections to localizations based on barcode identity and deformation fields.")
    parser.add_argument('--localizations', required=True, help='Path to the localizations file (Astropy format).')
    parser.add_argument('--deformation_folder', required=True, help='Path to the folder containing deformation fields.')
    parser.add_argument('--output', required=True, help='Path to the output file for corrected localizations.')
    parser.add_argument('--channel', default = 'ch00', help='Channel of the fiducial barcodes. Default = ch00')
    parser.add_argument('--zBinning', type=float, default=2.0, help='zBinning used in pyHiM. default=2.')
    parser.add_argument('--xyBinning', type=float, default=2.03125, help='xyBinning used in warpfield regustration. default=2.')
    parser.add_argument('--toleranceDrift_XY', default = 100 , type= float, help='Deformation field XY tolerance correction. Default = 1px')
    parser.add_argument('--toleranceDrift_Z', default = 50 , type= float, help='Deformation field Z tolerance correction. Default = 3px')    


    args = parser.parse_args()

    # Read the localizations
    localizations = read_localizations(args.localizations)

    # Regular expression to match the deformation files
    #regex_pattern = r'scan_(?P<runNumber>[0-9]+)_(?P<cycle>RT[0-9]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>ch00)_DF\.(tif|nii|nii.gz|h5|hdf5)'
    regex_pattern = fr'scan_(?P<runNumber>[0-9]+)_(?P<cycle>RT[0-9]+)_(?P<roi>[0-9]+)_ROI_converted_decon_(?P<channel>{args.channel})\.(tif|nii|nii.gz|h5|hdf5)'

    deformation_files = [f for f in os.listdir(args.deformation_folder) if re.match(regex_pattern, f)]

    if len(deformation_files) == 0:
        print(f"! Could not find deformation files in: {args.deformation_folder}")
        return
    else: 
        print(f"$ Found these files to process: {deformation_files}")
     
    for deformation_file in deformation_files:
        print("-"*80)
        match = re.match(regex_pattern, deformation_file)
        if match:
            cycle = match.group('cycle')
            channel = match.group('channel')
            print(f"$ Analyzing file: {deformation_file}")
            print("$ Decoded barcode: {} channel of DF: {}".format(int(cycle.split('RT')[1]), channel))
            
            if channel == args.channel and 'RT' in cycle:
                barcode_id = int(cycle.split('RT')[1])
                deformation_file_path = os.path.join(args.deformation_folder, deformation_file)
                
                if os.path.exists(deformation_file_path):
                    print(f"Applying deformation field: {deformation_file_path} for barcode ID: {barcode_id}")
                    deformation_field = read_deformation_field(deformation_file_path)
                    localizations = apply_deformation(localizations,\
                        deformation_field,\
                        barcode_id,\
                        z_binning=args.zBinning,\
                        xy_binning=args.xyBinning,\
                        toleranceDrift_XY=args.toleranceDrift_XY,\
                        toleranceDrift_Z = args.toleranceDrift_Z)
                    
                else:
                    print(f"Deformation field not found: {deformation_file_path}")

    # Save the corrected localizations
    write_localizations(localizations, args.output)
    print(f"Corrected localizations saved to {args.output}")

if __name__ == "__main__":
    main()
