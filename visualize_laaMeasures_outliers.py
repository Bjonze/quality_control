import os 
import json 
import numpy as np 
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
import SimpleITK as sitk
import trimesh
from tqdm import tqdm

def main(json_dir, sdf_dir, save_dir):
    # Load first json file
    with open(json_dir, 'r') as f:
        data1 = json.load(f)

    tortuosity_outliers_small = []
    tortuosity_outliers_big = []
    centerline_length_outliers_small = []
    centerline_length_outliers_big = []
    max_geodesic_distance_outliers_small = []
    max_geodesic_distance_outliers_big = []
    volume_outliers_small = []
    volume_outliers_big = []
    normalized_shape_index_outliers_small = []
    normalized_shape_index_outliers_big = []
    for d in data1:
        if d["tortuosity"] < 0.01:
            tortuosity_outliers_small.append(d["filename"])
        if d["tortuosity"] > 0.99:
            tortuosity_outliers_big.append(d["filename"])
        if d["centerline_length"] < 0.01:
            centerline_length_outliers_small.append(d["filename"])
        if d["centerline_length"] > 0.99:
            centerline_length_outliers_big.append(d["filename"])
        if d["max_geodesic_distance"] < 0.01:
            max_geodesic_distance_outliers_small.append(d["filename"])
        if d["max_geodesic_distance"] > 0.99:
            max_geodesic_distance_outliers_big.append(d["filename"])
        if d["volume"] < 0.01:
            volume_outliers_small.append(d["filename"])
        if d["volume"] > 0.99:
            volume_outliers_big.append(d["filename"])
        if d["normalized_shape_index"] < 0.01:
            normalized_shape_index_outliers_small.append(d["filename"])
        if d["normalized_shape_index"] > 0.99:
            normalized_shape_index_outliers_big.append(d["filename"])

    for name in tqdm(tortuosity_outliers_small, total=len(tortuosity_outliers_small), desc="Creating meshes for small tortuosity outliers"):
        file_path = os.path.join(sdf_dir, name)
        sdf = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        #save the sdf as a mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        new_name = name.split(".")[0]
        out_path = os.path.join(save_dir, "tortuosity_outlier_small")
        os.makedirs(out_path, exist_ok=True)
        mesh.export(os.path.join(out_path, new_name + "_tortuosity_outlier.obj"))
    for name in tqdm(tortuosity_outliers_big, total=len(tortuosity_outliers_big), desc="Creating meshes for big tortuosity outliers"):
        file_path = os.path.join(sdf_dir, name)
        sdf = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        #save the sdf as a mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        new_name = name.split(".")[0]
        out_path = os.path.join(save_dir, "tortuosity_outlier_big")
        os.makedirs(out_path, exist_ok=True)
        mesh.export(os.path.join(out_path, new_name + "_tortuosity_outlier.obj"))
    for name in tqdm(centerline_length_outliers_small, total=len(centerline_length_outliers_small), desc="Creating meshes for small centerline length outliers"):
        file_path = os.path.join(sdf_dir, name)
        sdf = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        #save the sdf as a mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        new_name = name.split(".")[0]
        out_path = os.path.join(save_dir, "centerline_length_outlier_small")
        os.makedirs(out_path, exist_ok=True)
        mesh.export(os.path.join(out_path, new_name + "_centerline_length_outlier.obj"))
    for name in tqdm(centerline_length_outliers_big, total=len(centerline_length_outliers_big), desc="Creating meshes for big centerline length outliers"):
        file_path = os.path.join(sdf_dir, name)
        sdf = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        #save the sdf as a mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        new_name = name.split(".")[0]
        out_path = os.path.join(save_dir, "centerline_length_outlier_big")
        os.makedirs(out_path, exist_ok=True)
        mesh.export(os.path.join(out_path, new_name + "_centerline_length_outlier.obj"))
    for name in tqdm(max_geodesic_distance_outliers_small, total=len(max_geodesic_distance_outliers_small), desc="Creating meshes for small max geodesic distance outliers"):
        file_path = os.path.join(sdf_dir, name)
        sdf = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        #save the sdf as a mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        new_name = name.split(".")[0]
        out_path = os.path.join(save_dir, "max_geodesic_distance_outlier_small")
        os.makedirs(out_path, exist_ok=True)
        mesh.export(os.path.join(out_path, new_name + "_max_geodesic_distance_outlier.obj"))
    for name in tqdm(max_geodesic_distance_outliers_big, total=len(max_geodesic_distance_outliers_big), desc="Creating meshes for big max geodesic distance outliers"):
        file_path = os.path.join(sdf_dir, name)
        sdf = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        #save the sdf as a mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        new_name = name.split(".")[0]
        out_path = os.path.join(save_dir, "max_geodesic_distance_outlier_big")
        os.makedirs(out_path, exist_ok=True)
        mesh.export(os.path.join(out_path, new_name + "_max_geodesic_distance_outlier.obj"))
    for name in tqdm(volume_outliers_small, total=len(volume_outliers_small), desc="Creating meshes for small volume outliers"):
        file_path = os.path.join(sdf_dir, name)
        sdf = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        #save the sdf as a mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        new_name = name.split(".")[0]
        out_path = os.path.join(save_dir, "volume_outlier_small")
        os.makedirs(out_path, exist_ok=True)
        mesh.export(os.path.join(out_path, new_name + "_volume_outlier.obj"))
    for name in tqdm(volume_outliers_big, total=len(volume_outliers_big), desc="Creating meshes for big volume outliers"):
        file_path = os.path.join(sdf_dir, name)
        sdf = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        #save the sdf as a mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        new_name = name.split(".")[0]
        out_path = os.path.join(save_dir, "volume_outlier_big")
        os.makedirs(out_path, exist_ok=True)
        mesh.export(os.path.join(out_path, new_name + "_volume_outlier.obj"))
    for name in tqdm(normalized_shape_index_outliers_small, total=len(normalized_shape_index_outliers_small), desc="Creating meshes for small normalized shape index outliers"):
        file_path = os.path.join(sdf_dir, name)
        sdf = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        #save the sdf as a mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        new_name = name.split(".")[0]
        out_path = os.path.join(save_dir, "normalized_shape_index_outlier_small")
        os.makedirs(out_path, exist_ok=True)
        mesh.export(os.path.join(out_path, new_name + "_normalized_shape_index_outlier.obj"))
    for name in tqdm(normalized_shape_index_outliers_big, total=len(normalized_shape_index_outliers_big), desc="Creating meshes for big normalized shape index outliers"):
        file_path = os.path.join(sdf_dir, name)
        sdf = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        verts, faces, _, _ = marching_cubes(sdf, level=0)
        #save the sdf as a mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        new_name = name.split(".")[0]
        out_path = os.path.join(save_dir, "normalized_shape_index_outlier_big")
        os.makedirs(out_path, exist_ok=True)
        mesh.export(os.path.join(out_path, new_name + "_normalized_shape_index_outlier.obj"))

if __name__ == "__main__":
    json_dir = "/work3/bmsha/data/shape_measures_lq.json"
    sdf_dir = "/work3/bmsha/sdf_lq/"
    save_dir = "/work3/bmsha/meshes/laa_measures_check/lq"
    os.makedirs(save_dir, exist_ok=True)
    main(json_dir, sdf_dir, save_dir)