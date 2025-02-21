from dolfin import *
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
from solver import solve_steady_navier_stokes, solve_unsteady_navier_stokes

class BoundaryConditions:
    class InletProfile(UserExpression):
        def __init__(self, t, max_velocity=1.5, height=0.41, **kwargs):
            super().__init__(**kwargs)
            self.t = t
            self.U = max_velocity
            self.H = height
            
        def eval(self, values, x):
            # Parabolic inlet profile
            values[0] = 4 * self.U * x[1] * (self.H - x[1]) / (self.H ** 2)
            values[1] = 0
            
        def value_shape(self):
            return (2,)
    

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
        V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        W_element = MixedElement(V_element, Q_element)
        W = FunctionSpace(mesh, W_element)
        Q = FunctionSpace(mesh, "Lagrange", 1)
        return W, Q

    @staticmethod
    def setup_boundary_conditions(W, mf, config, amp, U_inlet=None):
        bcs = []
        for bc in config['boundary_conditions']:
            if bc['type'] != 'Dirichlet':
                continue
            sub_space = W.sub(bc['sub_space'])
            
            if bc.get('bc_type') == 'cylinder':
                if bc['cylinder_type'] == 'front':
                    value = Expression(('0', '0'), degree=1)
                elif bc['cylinder_type'] == 'top':
                    value = Expression(('amp*sin(atan2(x[1]-cy, x[0]-cx))',
                                     '-amp*cos(atan2(x[1]-cy, x[0]-cx))'),
                                    degree=1,
                                    amp=amp,
                                    cx=bc['cylinder_center'][0],
                                    cy=bc['cylinder_center'][1])
                elif bc['cylinder_type'] == 'bottom':
                    value = Expression(('-amp*sin(atan2(x[1]-cy, x[0]-cx))',
                                     'amp*cos(atan2(x[1]-cy, x[0]-cx))'),
                                    degree=1,
                                    amp=amp,
                                    cx=bc['cylinder_center'][0],
                                    cy=bc['cylinder_center'][1])
            else:
                value = (U_inlet if bc['value'] == "BoundaryFunction" 
                        else Constant(tuple(bc['value'])))
                
            bcs.append(DirichletBC(sub_space, value, mf, bc['boundary_id']))
        return bcs
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
    if config['continuation']:
        simple_continuation(config)
    else:

        
        # Setup mesh and function spaces
        mesh, mf, cf = MeshHandler.load_mesh(config['mesh_file'], 
                                        config['mesh_function_file'])
        W, Q = SolverSetup.create_function_spaces(mesh)
        amp = config['amplitude']
        # Setup boundary conditions
        U_inlet = BoundaryConditions.InletProfile(t=0)

        # Setup measures for force calculations
        ds_circle_4 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
        ds_circle_5 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)
        ds_circle_6 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=6)

        ds_circle = ds_circle_4 + ds_circle_5 + ds_circle_6

        n1 = -FacetNormal(mesh)
    

        bcs = SolverSetup.setup_boundary_conditions(W, mf, config,amp, U_inlet)
    
        configuration = {
            'W': W, 
            'Q': Q,
            'nu': config['nu'],
            'bcs': bcs,
            'time_step': config['time_step'],
            'final_time': config['final_time'],
            'time_integration': config['time_integration'],
            'write_velocity': config.get('write_velocity', True),
            'write_pressure': config.get('write_pressure', False),
            'picard': config.get('picard', False),
            'flag_write_checkpoint': config.get('flag_write_checkpoint', False),
            'write_vorticity': config.get('write_vorticity', False),
            'flag_drag_lift': config.get('flag_drag_lift', False),
            'ds_circle': ds_circle,
            'n1': n1,
            'results_dir': results_dir,
            'flag_initial_u': config.get('flag_initial_u', False),
            'u0_file': config.get('u0_file', "results/velocity.xdmf")
        }
        if config['steady_solver']:
            
            solve_steady_navier_stokes(**configuration)
        else:
            solve_unsteady_navier_stokes(**configuration)

import numpy as np

def simple_continuation(config):
    # Setup mesh and function spaces
    mesh, mf, cf = MeshHandler.load_mesh(config['mesh_file'], 
                                    config['mesh_function_file'])
    W, Q = SolverSetup.create_function_spaces(mesh)
    
    # Setup boundary conditions
    U_inlet = BoundaryConditions.InletProfile(t=0)
    # Setup measures for force calculations
    ds_circle_4 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
    ds_circle_5 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)
    ds_circle_6 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=6)
    ds_circle = ds_circle_4 + ds_circle_5 + ds_circle_6
    n1 = -FacetNormal(mesh)

    # Define the amplitude range and step size
    amplitudes = np.linspace(0, 2, 40)
    print(amplitudes)
    
    # Initialize prev_sol and prev_sol2
    prev_sol = Function(W)
    prev_sol2 = Function(W)  # Solution from two steps ago
    u0_xdmf_file = XDMFFile(config.get('u0_file', "results/velocity.xdmf"))
    V = W.sub(0).collapse()  # Extract the velocity subspace
    u_initial = Function(V)
    u0_xdmf_file.read_checkpoint(u_initial, "u_out", 0)
    w_initial = Function(W)
    assign(w_initial.sub(0), u_initial)
    prev_sol.vector()[:] = w_initial.vector()
    prev_sol2.vector()[:] = w_initial.vector()  # Initialize prev_sol2
    
    for idx, amplitude in enumerate(amplitudes):
        # Create the new results directory name
        run_dir = f"trial_cont/run{idx + 1}"
        ConfigHandler.clean_results_directory(run_dir)
        bcs = SolverSetup.setup_boundary_conditions(W, mf, config, amplitude, U_inlet)
        configuration = {
            'W': W, 
            'Q': Q,
            'nu': config['nu'],
            'bcs': bcs,
            'time_step': config['time_step'],
            'final_time': config['final_time'],
            'time_integration': config['time_integration'],
            'write_velocity': config.get('write_velocity', True),
            'write_pressure': config.get('write_pressure', False),
            'picard': config.get('picard', False),
            'flag_write_checkpoint': config.get('flag_write_checkpoint', False),
            'write_vorticity': config.get('write_vorticity', False),
            'flag_drag_lift': config.get('flag_drag_lift', False),
            'ds_circle': ds_circle,
            'n1': n1,
            'results_dir': run_dir,
            'flag_initial_u': config.get('flag_initial_u', False),
            'prev_sol': prev_sol,
            'u0_file': config.get('u0_file', "results/velocity.xdmf"),
            'continuation': config.get('continuation', False)
        }

        try:
            # Attempt to solve the steady Navier-Stokes equations
            sol = solve_steady_navier_stokes(**configuration)
            
            # Update prev_sol2 and prev_sol
            prev_sol2.assign(prev_sol)  # Store the current prev_sol as prev_sol2
            prev_sol.assign(sol)       # Update prev_sol with the new solution
            
            print(f"Solved successfully for amplitude {amplitude}")
        except RuntimeError as e:
            # Handle the error: log it, revert prev_sol to prev_sol2, and continue
            print(f"RuntimeError encountered for amplitude {amplitude}: {e}")
            print("SKipping to next")
if __name__ == "__main__":
    main()