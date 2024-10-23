from dolfin import *
import json
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from datetime import datetime
from ns_forms import solve_unsteady_navier_stokes, solve_steady_navier_stokes
from visualize_benchmark import visualize_bench
import shutil
# ============================================================================
# Boundary Condition Class
# ============================================================================
class BoundaryFunction(UserExpression):
    """
    Defines the inlet velocity profile as a parabolic function.
    For unsteady flow, can be modified to include time dependency.
    """
    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    def eval(self, values, x):
        # Parabolic profile with maximum velocity U
        U = 1.5
        values[0] = 4*U*x[1]*(0.41-x[1])/pow(0.41, 2)
        values[1] = 0

    def value_shape(self):
        return (2,)

# ============================================================================
# Utility Functions
# ============================================================================
def plot_mesh_and_boundaries(mesh, mf):
    """
    Visualizes mesh and boundaries with different colors for each boundary ID.
    
    Args:
        mesh: FEniCS mesh object
        mf: Mesh function containing boundary markers
    """
    plt.figure()
    boundary_ids = np.unique(mf.array())
    cmap = plt.get_cmap("tab20", len(boundary_ids))
    fig, ax = plt.subplots()

    for i, boundary_id in enumerate(boundary_ids):
        color = cmap(i / len(boundary_ids))
        boundary_facets = [facet for facet in cpp.mesh.facets(mesh) 
                         if mf[facet.index()] == boundary_id]
        
        for facet in boundary_facets:
            coords = facet.entities(0)
            pts = mesh.coordinates()[coords]
            polygon = plt.Polygon(pts, closed=True, fill=None, 
                                edgecolor=color, linewidth=2)
            ax.add_patch(polygon)
        legend_patch = plt.Line2D([0], [0], color=color, lw=2, 
                                 label=f'Boundary ID {boundary_id}')
        ax.add_line(legend_patch)
    
    plt.legend()
    plt.title("Mesh with Boundaries")
    plt.show()

def load_config(config_file):
    """
    Loads configuration from JSON file.
    
    Args:
        config_file (str): Path to JSON configuration file
    Returns:
        dict: Configuration parameters
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def save_input_parameters(config, results_dir):
    """
    Saves input parameters to a text file in results directory.
    
    Args:
        config (dict): Configuration dictionary
        results_dir (str): Path to results directory
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_log_path = os.path.join(results_dir, f'input_parameters_{timestamp}.txt')
    
    # Write the JSON contents to a text file
    with open(input_log_path, 'w') as f:
        f.write("Input Parameters:\n")
        f.write("="*20 + "\n\n")
        f.write(f"Original config file: inputs2.json\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(json.dumps(config, indent=4))


def apply_boundary_conditions(W, mf, boundary_conditions, U_inlet):
    """
    Creates boundary conditions based on configuration.
    
    Args:
        W: Function space
        mf: Mesh function with boundary markers
        boundary_conditions: List of boundary condition specifications
        U_inlet: Inlet velocity profile function
    Returns:
        list: FEniCS DirichletBC objects
    """
    bcs = []
    for bc in boundary_conditions:
        if bc['type'] == 'Dirichlet':
            sub_space = W.sub(bc['sub_space'])
            boundary_id = bc['boundary_id']
            value = U_inlet if bc['value'] == "BoundaryFunction" else Constant(tuple(bc['value']))
            bcs.append(DirichletBC(sub_space, value, mf, boundary_id))
    return bcs

# ============================================================================
# Main Execution
# ============================================================================
def main():
    # Load configuration
    config = load_config('inputs2.json')
    results_dir = config.get('results_dir', "results/")

    # Clean directory before saving
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            filepath = os.path.join(results_dir, file)
            try:
                if os.path.isfile(filepath):
                    os.unlink(filepath)
            except Exception as e:
                print(f"Error: {e}")

    save_input_parameters(config, results_dir)

    # Setup mesh and boundary conditions
    mesh = Mesh()
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
    with XDMFFile(config['mesh_file']) as infile:
        infile.read(mesh)
        infile.read(mvc, "name_to_read")
    cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
    with XDMFFile(config['mesh_function_file']) as infile:
        infile.read(mvc, "name_to_read")
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # Setup finite elements and function spaces
    V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W_element = MixedElement(V_element, Q_element)  # Taylor-Hood
    W = FunctionSpace(mesh, W_element)
    Q = FunctionSpace(mesh, "Lagrange", 1)  # For vorticity

    # Setup boundary conditions
    t = 0
    U_inlet = BoundaryFunction(t)
    bcs = apply_boundary_conditions(W, mf, config['boundary_conditions'], U_inlet)

    # Setup measures for force calculations - keeping original implementation
    ds_circle_4 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
    ds_circle_5 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)
    ds_circle_6 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=6)

    ds_circle = ds_circle_4 + ds_circle_5 + ds_circle_6

    n1 = -FacetNormal(mesh)  # Normal pointing out of obstacle

    # Solve problem
    solver_params = {
        'W': W,
        'Q': Q,
        'nu': config['nu'],
        'bcs': bcs,
        'ds_circle': ds_circle,
        'n1': n1,
        'flag_drag_lift': config.get('flag_drag_lift', False),
        'flag_initial_u': config.get('flag_initial_u', False),
        'u0_file': config.get('u0_file', "results/velocity.xdmf"),
        'flag_write_checkpoint': config.get('flag_write_checkpoint', False),
        'flag_save_vorticity': config.get('flag_save_vorticity', False),
        'results_dir': results_dir
    }

    if config['steady_solver']:
        solve_steady_navier_stokes(**solver_params)
    else:
        unsteady_params = {
            'W': W,
            'nu': config['nu'],
            'bcs': bcs,
            'T': config['final_time'],
            'dt': config['time_step'],
            'time_integration_method': config['time_integration'],
            'theta': 0.5,
            'ds_circle': ds_circle,
            'n1': n1,
            'U_inlet': U_inlet,
            'write_velocity': config.get('write_velocity', True),
            'write_pressure': config.get('write_pressure', False),
            'flag_drag_lift': config.get('flag_drag_lift', False),
            'flag_initial_u': config.get('flag_initial_u', False),
            'u0_file': config.get('u0_file', "results/velocity.xdmf"),
            'flag_write_checkpoint': config.get('flag_write_checkpoint', False),
            'flag_save_vorticity': config.get('flag_save_vorticity', False),
            'results_dir': results_dir
        }
        solve_unsteady_navier_stokes(**unsteady_params)

if __name__ == "__main__":
    main()


  