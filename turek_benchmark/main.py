from dolfin import *
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
from solver import solve_steady_navier_stokes, solve_unsteady_navier_stokes


  

class MeshHandler:
    @staticmethod
    def load_mesh(mesh_file, boundary_file):
        mesh = Mesh()
        mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
        
        with XDMFFile(mesh_file) as infile:
            infile.read(mesh)
            infile.read(mvc, "name_to_read")
        
        cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
        
        mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
        with XDMFFile(boundary_file) as infile:
            infile.read(mvc, "name_to_read")
        
        mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
        return mesh, mf, cf

class SolverSetup:
    @staticmethod
    def create_function_spaces(mesh):
        # Create function spaces for velocity and pressure
        V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        W_element = MixedElement(V_element, Q_element)
        W = FunctionSpace(mesh, W_element)
        Q = FunctionSpace(mesh, "Lagrange", 1)
        return W, Q

    @staticmethod
    def setup_boundary_conditions(W, mf, config, U_inlet=None):
        bcs = []
        
        # Time-dependent inlet condition using Expression
        U_inlet = Expression(
            ('4*1.5*sin(pi*t/8)*x[1]*(0.41-x[1])/pow(0.41, 2)', '0'),
            t=0.0, degree=2
        ) if U_inlet is None else U_inlet

        for bc in config['boundary_conditions']:
            # Handle different boundary condition types
            if bc['type'] == 'Dirichlet':
                sub_space = W.sub(bc['sub_space'])
                
                # Inlet boundary (ID = 1)
                if bc['value'] == "BoundaryFunction":
                    value = U_inlet
                
                # No-slip walls (ID = 3) and cylinder (ID = 4)
                elif bc.get('bc_type') == 'cylinder' or bc['boundary_id'] == 3:
                    value = Constant((0.0, 0.0))
                    
                else:
                    value = Constant(tuple(bc['value']))
                    
                bcs.append(DirichletBC(sub_space, value, mf, bc['boundary_id']))
                
            # Pressure boundary condition at outflow (ID = 2)
            elif bc['type'] == 'Dirichlet_p':
                p_space = W.sub(1)  # Pressure subspace
                value = Constant(bc['value'][0])
                bcs.append(DirichletBC(p_space, value, mf, bc['boundary_id']))

        return bcs, U_inlet
class ConfigHandler:
    @staticmethod
    def load_config(config_file):
        with open(config_file, 'r') as file:
            return json.load(file)
    
    @staticmethod
    def save_parameters(config, results_dir):
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(results_dir, f'input_parameters_{timestamp}.txt')
        
        with open(log_path, 'w') as f:
            f.write("Input Parameters:\n" + "="*20 + "\n\n")
            f.write(f"Original config file: inputs2.json\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(json.dumps(config, indent=4))
    
    @staticmethod
    def clean_results_directory(directory):
        if os.path.exists(directory):
            for file in os.listdir(directory):
                try:
                    filepath = os.path.join(directory, file)
                    if os.path.isfile(filepath):
                        os.unlink(filepath)
                except Exception as e:
                    print(f"Error cleaning directory: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='inputs2.json')
    args = parser.parse_args()
    
    # Load and setup configuration
    config = ConfigHandler.load_config(args.config)
    results_dir = config.get('results_dir', "results/")
    ConfigHandler.clean_results_directory(results_dir)
    ConfigHandler.save_parameters(config, results_dir)
    
    # Setup mesh and function spaces
    mesh, mf, cf = MeshHandler.load_mesh(config['mesh_file'], 
                                       config['mesh_function_file'])
    W, Q = SolverSetup.create_function_spaces(mesh)
    
    # Setup boundary conditions

    bcs, U_inlet = SolverSetup.setup_boundary_conditions(W, mf, config)
    
    # Setup measures for force calculations
    ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)


    n1 = -FacetNormal(mesh)
    

    configuration = {
        'W': W,  # Your function space
        'nu': config['nu'],
        'bcs': bcs,
        'time_step': config['time_step'],
        'final_time': config['final_time'],
        'time_integration': config['time_integration'],
        'write_velocity': config.get('write_velocity', True),
        'write_pressure': config.get('write_pressure', False),
        'flag_drag_lift': config.get('flag_drag_lift', False),
        'ds_circle': ds_circle,
        'n1': n1,
        'results_dir': results_dir,
        'flag_initial_u': config.get('flag_initial_u', False),
        'flag_write_checkpoint': config.get('flag_write_checkpoint', False),
        'u0_file': config.get('u0_file', "results/velocity.xdmf"),
        'U_inlet': U_inlet
    }

    if config['steady_solver']:
        solve_steady_navier_stokes(**configuration)
    else:
        solve_unsteady_navier_stokes(**configuration)

if __name__ == "__main__":
    main()