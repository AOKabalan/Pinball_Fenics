from dolfin import *
import json
from ns_forms import solve_unsteady_navier_stokes
from visualize_benchmark import visualize_bench
import os

class BoundaryFunction(UserExpression):
    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    def eval(self, values, x):
        U = 1.5*sin(pi*self.t/8)
        values[0] = 4*U*x[1]*(0.41-x[1])/pow(0.41, 2)
        values[1] = 0

    def value_shape(self):
        return (2,)



def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

# Load configuration from JSON file
config = load_config('inputs.json')

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


V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W_element = MixedElement(V_element, Q_element) # Taylor-Hood
W = FunctionSpace(mesh, W_element)


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




# Prepare surface measure on cylinder used for drag and lift
ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)

n1 = -FacetNormal(mesh) #Normal pointing out of obstacle


#Solve and Save results

# Call the function with specific arguments from the JSON configuration
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
    flag_drag_lift=config.get('flag_drag_lift', True),
    results_dir=config.get('results_dir', "results/")
    )

os.makedirs(config['figures_dir'], exist_ok=True)
visualize_bench(config['time_integration'],config['results_dir'])
