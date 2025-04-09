import os
import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vtk
from skimage.measure import marching_cubes
from plot_meshes import nifisdf2vtk, create_mesh_subplot
import json
import pandas as pd
from tqdm import tqdm
import time
import shutil

def main(sdf_dir, json_dir, out_dir, q_low, q_high):
    #we first need to actually calculate the outliers
    #load the json first:
    with open(json_dir) as f_in:
        json_file = json.load(f_in)
        
    df = pd.DataFrame(json_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    df_norm = df.copy()
    
    for col in numeric_cols:
        if col == "estimated_bifurcations":
            # Skip normalization for this column; leave as-is.
            continue
        # Combine the column values from both dataframes
        combined = df[col]
        lower = combined.quantile(q_low)
        upper = combined.quantile(q_high)
        
        # Avoid division by zero in case upper equals lower
        if upper == lower:
            df[col] = 0.0
        else:
            # Apply min–max normalization and clip the results to [0, 1]
            df_norm[col] = ((df[col] - lower) / (upper - lower)).clip(0, 1)
    # Convert back to list of dictionaries
    norm_list = df_norm.to_dict(orient='records')
    # Cache to store the computed mesh image path for each file.
    computed_images = {}

    # Process only records that are flagged as an outlier in at least one metric.
    for record in tqdm(norm_list, total=len(norm_list), desc="Creating outlier images"): #TODO: If more than 2 outliers are present (1 outlier is too few)
        filename = record["filename"]
        name_no_ext = os.path.splitext(filename)[0]
        
        # Check if this record is an outlier in any numeric key (except 'estimated_bifurcations')
        is_record_outlier = False
        for col in numeric_cols:
            if col == "estimated_bifurcations":
                continue
            if record[col] < 0.01 or record[col] > 0.99:
                is_record_outlier = True
                break
        if not is_record_outlier:
            continue  # Skip this record entirely if no outlier condition is met.
        
        nii_sdf_path = os.path.join(sdf_dir, filename)
        # Create the VTK mesh only once for the record if needed.
        out_vtk_folder = os.path.join(out_dir, "meshes")
        os.makedirs(out_vtk_folder, exist_ok=True)
        out_vtk_path = os.path.join(out_vtk_folder, f"{name_no_ext}.vtk")
        if not os.path.exists(out_vtk_path):
            nifisdf2vtk(nii_sdf_path, out_vtk_path)
        
        # Process each numeric key for outlier conditions.
        for col in numeric_cols:
            if col == "estimated_bifurcations":
                continue
            value = record[col]
            if value < 0.01:
                category = "outlier"#f"{col}_small"
            elif value > 0.99:
                category = "outlier"#f"{col}_big"
            else:
                continue
            
            img_save_folder = os.path.join(out_dir, category)
            os.makedirs(img_save_folder, exist_ok=True)
            mesh_img_save_path = os.path.join(img_save_folder, f"{name_no_ext}.png")
            
            if not os.path.exists(mesh_img_save_path):
                if name_no_ext in computed_images:
                    continue
                    # try:
                    #     shutil.copy(computed_images[name_no_ext], mesh_img_save_path)
                    #     print(f"Copied image for {filename} from {computed_images[name_no_ext]} to {mesh_img_save_path} for category {category}")
                    # except Exception as e:
                    #     print(f"Error copying image for {filename}: {e}")
                else:
                    create_mesh_subplot(out_vtk_path, mesh_img_save_path)
                    computed_images[name_no_ext] = mesh_img_save_path

if __name__ == '__main__':
    if not os.environ.get('XDG_RUNTIME_DIR'):
        os.environ['XDG_RUNTIME_DIR'] = '/tmp'
    sdf_dir = "/work3/bmsha/sdf_lq"
    json_dir = "/work3/bmsha/data/shape_measures_lq_unNorm.json"
    out_dir = "/work3/bmsha/images/quality_control"
    main(sdf_dir, json_dir, out_dir, q_low=0.025, q_high=0.975)