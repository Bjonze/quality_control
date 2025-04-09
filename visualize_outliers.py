import os
import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vtk
from plot_meshes import nifisdf2vtk, create_mesh_subplot
import json
import pandas as pd
from tqdm import tqdm

def main(sdf_dir, outlier_txt, out_dir):
    #load the txt file 
    with open(outlier_txt, 'r') as f:
        outliers = f.readlines()
    outliers = [x.strip() for x in outliers]
    out_vtk_dir = os.path.join(out_dir, "meshes")
    os.makedirs(out_vtk_dir, exist_ok=True)
    out_image_dir = os.path.join(out_dir, "images")
    os.makedirs(out_image_dir, exist_ok=True)
    for sdf in tqdm(outliers, total=len(outliers)):
        sdf_no_ext = sdf.split(".")[0]
        mesh_file_path = os.path.join(out_vtk_dir, sdf_no_ext + ".vtk")
        output_filename = os.path.join(out_image_dir, sdf_no_ext + ".png")
        nifisdf2vtk(os.path.join(sdf_dir, sdf), mesh_file_path)
        create_mesh_subplot(mesh_file_path, output_filename)
if __name__ == '__main__':
    if not os.environ.get('XDG_RUNTIME_DIR'):
        os.environ['XDG_RUNTIME_DIR'] = '/tmp'
    sdf_dir = "/work3/bmsha/sdf_lq"
    outlier_txt = "/work3/bmsha/quality_control/outlier_filenames.txt"
    out_dir = "/work3/bmsha/quality_control/figures/outlier_visual/"
    main(sdf_dir, outlier_txt, out_dir)