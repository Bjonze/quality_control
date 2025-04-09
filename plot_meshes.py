import os
import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vtk
from skimage.measure import marching_cubes

 
def nifisdf2vtk(nii_sdf_path, out_vtk_path):
    sdf = sitk.GetArrayFromImage(sitk.ReadImage(nii_sdf_path))
    # class 8 is foreground and rest is background
    sdf = np.where(sdf == 8, 1, 0)
    # Convert the SDF to a binary volume
    verts, faces, _, _ = marching_cubes(sdf, level=0)
    #save the sdf as a .vtk mesh 
    points = vtk.vtkPoints()
    for v in verts:
        # InsertNextPoint expects the (x, y, z) coordinate.
        points.InsertNextPoint(v[0], v[1], v[2])
    
    # Create a vtkCellArray to store the triangle faces.
    triangles = vtk.vtkCellArray()
    for f in faces:
        # Each face from marching_cubes is a triangle represented by 3 indices.
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, int(f[0]))
        triangle.GetPointIds().SetId(1, int(f[1]))
        triangle.GetPointIds().SetId(2, int(f[2]))
        triangles.InsertNextCell(triangle)
    
    # Create a vtkPolyData object and set its points and triangle faces.
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    
    # Write the polydata to a VTK file using vtkPolyDataWriter.
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_vtk_path)
    writer.SetInputData(polydata)
    writer.Write()

def create_mesh_subplot(mesh_file_path, output_filename):
    """
    Load a mesh from the given file, create a 3x3 subplot image displaying the mesh from various angles,
    and save the resulting image to the output directory.
    
    Parameters:
        mesh_file_path (str): Path to the input mesh file (e.g., .vtk, .stl, etc.)
        output_dir (str): Directory where the output image will be saved.
    """
    pv.start_xvfb()
    # Load the mesh using PyVista
    mesh = pv.read(mesh_file_path)
    
    # Ensure the output directory exists
    #os.makedirs(output_dir, exist_ok=True)
    
    # Create a Plotter with a 3x3 grid layout and a defined window size
    plotter = pv.Plotter(shape=(3, 3), window_size=(900, 900), off_screen=True)
    
    # Compute the center of the mesh (used as the camera's focal point)
    center = mesh.center
    
    # Calculate a suitable radius for the camera based on the mesh's bounding box diagonal
    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    diag = np.sqrt((bounds[1] - bounds[0])**2 +
                   (bounds[3] - bounds[2])**2 +
                   (bounds[5] - bounds[4])**2)
    radius = diag * 1.2  # a bit larger than the diagonal for a comfortable view
    
    # Define 3 elevations and 3 azimuths (in degrees) to obtain 9 unique viewpoints.
    elevations = [30, 0, -30]  # top view, side view, bottom view
    azimuths = [0, 90, 180]     # different rotations around the mesh
    
    # Loop over the subplot grid to set each view
    for i, elev in enumerate(elevations):
        for j, azim in enumerate(azimuths):
            # Select the subplot at row i, column j
            plotter.subplot(i, j)
            
            # Add the mesh to the subplot (display edges for clarity)
            plotter.add_mesh(mesh, interpolate_before_map=False, style='surface', show_edges=False, lighting=True, color=(1.0, 0.9, 0.8))
            
            # Convert angles from degrees to radians for computing camera position
            azim_rad = np.deg2rad(azim)
            elev_rad = np.deg2rad(elev)
            
            # Compute the camera position using spherical coordinates around the mesh center
            cam_x = center[0] + radius * np.cos(elev_rad) * np.cos(azim_rad)
            cam_y = center[1] + radius * np.cos(elev_rad) * np.sin(azim_rad)
            cam_z = center[2] + radius * np.sin(elev_rad)
            camera_position = [(cam_x, cam_y, cam_z), center, (0, 0, 1)]
            
            # Set the camera position for the current subplot
            plotter.camera_position = camera_position
            
            # Optionally, add a text annotation to indicate the viewing angles
            plotter.add_text(f"Az: {azim}°, Elev: {elev}°", font_size=10, position="upper_left")
    
    # Define the output file path for the screenshot
    #output_file = os.path.join(output_dir, "mesh_subplot.png")
    
    # Render the subplots and save a screenshot; auto_close ensures the window is closed after saving
    plotter.show(screenshot=output_filename, auto_close=True, interactive=False)
    
if __name__ == "__main__":
    pv.start_xvfb()
    # Example usage:
    nii_sdf_path = "/work3/bmsha/sdf_lq/Aorta-2_7439_SERIES0010_sdf.nii"
    mesh_name = os.path.splitext(os.path.basename(nii_sdf_path))[0].split(".")[0]
    out_vtk_path = os.path.join("/work3/bmsha/images/quality_control/", mesh_name + ".vtk")
    nifisdf2vtk(nii_sdf_path, out_vtk_path)
    
    mesh_name = os.path.splitext(os.path.basename(out_vtk_path))[0].split(".")[0]
    output_dir = os.path.join("/work3/bmsha/images/quality_control/", mesh_name)
    os.makedirs(output_dir, exist_ok=True)
    create_mesh_subplot(out_vtk_path, output_dir)