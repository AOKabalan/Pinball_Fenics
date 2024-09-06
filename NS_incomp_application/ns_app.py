from dolfin import *
import json
from ns_forms import solve_unsteady_navier_stokes
from ns_forms import solve_steady_navier_stokes
import numpy as np
from visualize_benchmark import visualize_bench
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class BoundaryFunction(UserExpression):
    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    # def eval(self, values, x):
    #     U = 1.5*sin(pi*self.t/8)
    #     values[0] = 4*U*x[1]*(0.41-x[1])/pow(0.41, 2)
    #     values[1] = 0
    def eval(self, values, x):
        U = 1.5
        
        values[0] = 4*U*x[1]*(0.41-x[1])/pow(0.41, 2)
        values[1] = 0

    def value_shape(self):
        return (2,)

# Visualize the mesh and boundaries with different colors
def plot_mesh_and_boundaries(mesh, mf):
    plt.figure()
    
    # Get the unique boundary IDs
    boundary_ids = np.unique(mf.array())
    
    # Create a colormap
    cmap = plt.get_cmap("tab20", len(boundary_ids))
    
    fig, ax = plt.subplots()
 

    # Plot each boundary with a different color
    for i, boundary_id in enumerate(boundary_ids):
        color = cmap(i / len(boundary_ids))
        
        # Extract facets with this boundary id
        boundary_facets = [facet for facet in cpp.mesh.facets(mesh) if mf[facet.index()] == boundary_id]
        
        for facet in boundary_facets:
            coords = facet.entities(0)
            pts = mesh.coordinates()[coords]
            polygon = plt.Polygon(pts, closed=True, fill=None, edgecolor=color, linewidth=2)
            ax.add_patch(polygon)
        legend_patch = plt.Line2D([0], [0], color=color, lw=2, label=f'Boundary ID {boundary_id}')
        ax.add_line(legend_patch)
    plt.legend()
    plt.title("Mesh with Boundaries")
    plt.show()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

# Load configuration from JSON file
config = load_config('inputs2.json')

# Access configuration values

mesh_file = config['mesh_file']
mesh_function_file = config['mesh_function_file']
boundary_conditions = config['boundary_conditions']


theta = 0.5



mesh = Mesh()
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile(mesh_file) as infile:
    infile.read(mesh)
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile(mesh_function_file) as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)


# # Call the visualization function
# plot_mesh_and_boundaries(mesh, mf)

V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W_element = MixedElement(V_element, Q_element) # Taylor-Hood
W = FunctionSpace(mesh, W_element)

#Function for vorticity 
Q = FunctionSpace(mesh, "Lagrange", 1)

    # Define boundary conditions
#inlet is 2
#outlet is 3
#walls are 4
#cylinder is 5

# Function to dynamically create boundary conditions
def apply_boundary_conditions(W, mf, boundary_conditions,U_inlet):
    bcs = []
    for bc in boundary_conditions:
        if bc['type'] == 'Dirichlet':
            sub_space = W.sub(bc['sub_space'])
            boundary_id = bc['boundary_id']
            if bc['value'] == "BoundaryFunction":
                
                value = U_inlet
            else:
                value = Constant(tuple(bc['value']))
            bcs.append(DirichletBC(sub_space, value, mf, boundary_id))
    return bcs

# Apply boundary conditions
t = 0
U_inlet = BoundaryFunction(t)
bcs = apply_boundary_conditions(W, mf, boundary_conditions,U_inlet)




# Prepare surface measure on the three cylinders used for drag and lift
ds_circle_4 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
ds_circle_5 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)
ds_circle_6 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=6)

ds_circle = ds_circle_4 + ds_circle_5 + ds_circle_6

#ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4,5,6)

n1 = -FacetNormal(mesh) #Normal pointing out of obstacle


#Solve and Save results
if config['steady_solver']:

    solve_steady_navier_stokes(W=W,  
    Q=Q,
    nu=config['nu'],
    bcs=bcs,
    ds_circle = ds_circle,
    n1 = n1,
    flag_drag_lift=config.get('flag_drag_lift', False),
    flag_initial_u=config.get('flag_initial_u', False),
    u0_file = config.get('u0_file', "results/velocity.xdmf"),
    flag_write_checkpoint =config.get('flag_write_checkpoint', False), 
    flag_save_vorticity =config.get('flag_save_vorticity', False), 
    results_dir=config.get('results_dir', "results/"))


else:
    solve_unsteady_navier_stokes(
    W=W,  
    nu=config['nu'],
    bcs=bcs,  
    T=config['final_time'],
    dt=config['time_step'],
    time_integration_method=config['time_integration'],
    theta=theta,
    ds_circle = ds_circle,
    n1 = n1,
    U_inlet= U_inlet,
    write_velocity=config.get('write_velocity', True),
    write_pressure=config.get('write_pressure', False),
    flag_drag_lift=config.get('flag_drag_lift', False),
    flag_initial_u=config.get('flag_initial_u', False),
    u0_file = config.get('u0_file', "results/velocity.xdmf"),
    flag_write_checkpoint =config.get('flag_write_checkpoint', False), 
    flag_save_vorticity =config.get('flag_save_vorticity', False), 
    results_dir=config.get('results_dir', "results/")
    )

flag_drag_lift=config.get('flag_drag_lift', False),
# if flag_drag_lift :

#     os.makedirs(config['figures_dir'], exist_ok=True)
#     visualize_bench(config['time_integration'],config['results_dir'])
