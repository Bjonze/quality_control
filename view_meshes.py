import os
import argparse
import pyvista as pv

class MeshViewer:
    def __init__(self, mesh_folder, output_file):
        self.mesh_folder = mesh_folder
        self.output_file = output_file
        # List and sort all .vtk files in the folder
        self.mesh_files = sorted([f for f in os.listdir(mesh_folder) if f.lower().endswith('.vtk')])
        if not self.mesh_files:
            raise ValueError(f"No .vtk files found in folder: {mesh_folder}")
        self.current_index = 0
        self.plotter = pv.Plotter()
        self.setup_callbacks()
        self.update_mesh()

    def update_mesh(self):
        """Load and display the current mesh."""
        self.plotter.clear()  # Remove any existing mesh
        mesh_path = os.path.join(self.mesh_folder, self.mesh_files[self.current_index])
        try:
            mesh = pv.read(mesh_path)
        except Exception as e:
            print(f"Error loading {mesh_path}: {e}")
            return
        # Add the mesh with edges highlighted for better visibility
        self.plotter.add_mesh(mesh, color="yellow", style="surface" , show_edges=True, edge_color="black", line_width=1, lighting=True)
        # Overlay text showing current index and filename
        overlay_text = f"[{self.current_index+1}/{len(self.mesh_files)}] {self.mesh_files[self.current_index]}"
        self.plotter.add_text(overlay_text, position="upper_left", font_size=12, color="black")
        self.plotter.render()

    def next_mesh(self):
        """Move to the next mesh."""
        self.current_index = (self.current_index + 1) % len(self.mesh_files)
        self.update_mesh()

    def prev_mesh(self):
        """Move to the previous mesh."""
        self.current_index = (self.current_index - 1) % len(self.mesh_files)
        self.update_mesh()

    def select_current(self):
        """Append the current mesh name (without extension) to the output file."""
        filename_no_ext = os.path.splitext(self.mesh_files[self.current_index])[0]
        with open(self.output_file, 'a') as f:
            f.write(filename_no_ext + "\n")
        print(f"Selected: {filename_no_ext}")
        # Optionally, automatically move to the next mesh after selection:
        self.next_mesh()

    def setup_callbacks(self):
        """Set up key bindings for navigation and selection."""
        self.plotter.add_key_event("Left", self.prev_mesh)
        self.plotter.add_key_event("Right", self.next_mesh)
        self.plotter.add_key_event("space", self.select_current)
        self.plotter.add_key_event("Escape", self.plotter.close)

    def start(self):
        self.plotter.show()

def main():
    parser = argparse.ArgumentParser(description="Mesh Viewer for .VTK files")
    parser.add_argument("mesh_folder", type=str, help="Folder containing .vtk files")
    parser.add_argument("--output", type=str, default="selected_meshes.txt",
                        help="Output text file for selected mesh names")
    args = parser.parse_args()

    try:
        viewer = MeshViewer(args.mesh_folder, args.output)
    except ValueError as e:
        print(e)
        return

    viewer.start()

if __name__ == "__main__":
    main()
