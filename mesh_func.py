import sys
import meshio

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(
        points=points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]}
    )
    return out_mesh

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mesh_func.py <filename>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    try:
        mesh_from_file = meshio.read(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Create line mesh and write to "mf.xdmf"
    line_mesh = create_mesh(mesh_from_file, "line", prune_z=True)
    meshio.write("mesh/mf.xdmf", line_mesh)

    # Create triangle mesh and write to "mesh.xdmf"
    triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
    meshio.write("mesh/mesh.xdmf", triangle_mesh)

    print(f"Meshes written to 'mf.xdmf' and 'mesh.xdmf' from '{input_file}'.")
