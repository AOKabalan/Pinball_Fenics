from dolfin import *
import numpy as np

# Read the mesh and mesh functions
mesh = Mesh()
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile("mesh/mesh.xdmf") as infile:
    infile.read(mesh)
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile("mesh/mf.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# Create velocity function space
V = VectorFunctionSpace(mesh, "Lagrange", 2)  # Using order 2 as in your original setup

# Read the original velocity
u_original = Function(V)
filename_velocity_checkpoint = f'velocity_checkpoint.xdmf'
with XDMFFile(filename_velocity_checkpoint) as infile:
    infile.read_checkpoint(u_original, "u_out", 0)

P = FunctionSpace(mesh, "Lagrange", 2)
ux = project(u_original.sub(0), P)
uy = project(u_original.sub(1), P)

def find_closest_point(target_x, target_y, coordinates):
    distances = np.sqrt((coordinates[:, 0] - target_x)**2 +
                       (coordinates[:, 1] - target_y)**2)
    return np.argmin(distances)

# Get ALL dof coordinates from P
P_dofs = P.tabulate_dof_coordinates()
gdim = mesh.geometry().dim()
P_dofs = P_dofs.reshape((-1, gdim))

# Create functions for swapped components
u_x_swapped = Function(P)
u_y_swapped = Function(P)


# Copy original vectors to ensure we have all values
u_x_swapped.vector()[:] = ux.vector()
u_y_swapped.vector()[:] = uy.vector()

# Now do the mirroring for ALL points
for i in range(len(P_dofs)):
    x = P_dofs[i, 0]
    y = P_dofs[i, 1]
    
    # Find conjugate point (x, -y)
    conjugate_idx = find_closest_point(x, -y, P_dofs)
    
    # Copy velocities from conjugate point
    u_x_swapped.vector()[conjugate_idx] = ux.vector()[i]
    u_y_swapped.vector()[conjugate_idx] = uy.vector()[i]

# Create vector function and assign components
u_swapped = Function(V)
assigner = FunctionAssigner(V, [P, P])
assigner.assign(u_swapped, [u_x_swapped, u_y_swapped])

# Verify the results for both components
n_components = mesh.geometry().dim()
dof_range = len(u_swapped.vector()) // n_components

# Save the swapped field as XDMF
filename_swapped = 'velocity_swapped.xdmf'
with XDMFFile(filename_swapped) as outfile:
    outfile.write_checkpoint(u_swapped, "u_out", 0, XDMFFile.Encoding.HDF5, False)
