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
    
    # class CylinderBC(UserExpression):
    #     def __init__(self, center, cylinder_type, amplitude, **kwargs):
    #         super().__init__(**kwargs)
    #         self.center = np.array(center)
    #         self.type = cylinder_type
    #         self.amplitude = amplitude
        
    #     def eval(self, value, x):
    #         pos = np.array([x[0], x[1]]) - self.center
    #         theta = np.arctan2(pos[1], pos[0])
            
    #         if self.type == "front":
    #             value[0] = value[1] = 0.0
    #         elif self.type == "top":

    #             value[0] = self.amplitude * np.sin(theta)
    #             value[1] = -self.amplitude * np.cos(theta)
    #         elif self.type == "bottom":
    #             value[0] = -self.amplitude * np.sin(theta)
    #             value[1] = self.amplitude * np.cos(theta)

                
    #     def value_shape(self):
    #         return (2,)

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

# class SolverSetup:
#     @staticmethod
#     def create_function_spaces(mesh):
#         V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
#         Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#         W_element = MixedElement(V_element, Q_element)
        
#         W = FunctionSpace(mesh, W_element)
#         Q = FunctionSpace(mesh, "Lagrange", 1)
#         return W, Q

#     @staticmethod
#     def setup_boundary_conditions(W, mf, config, U_inlet=None):
#         bcs = []
#         for bc in config['boundary_conditions']:
#             if bc['type'] != 'Dirichlet':
#                 continue
                
#             sub_space = W.sub(bc['sub_space'])
#             value = None
            
#             if bc.get('bc_type') == 'cylinder':
#                 value = BoundaryConditions.CylinderBC(
#                     bc['cylinder_center'],
#                     bc['cylinder_type'],
#                     bc['amplitude'],
#                     degree=1
#                 )
#             else:
#                 value = (U_inlet if bc['value'] == "BoundaryFunction" 
#                         else Constant(tuple(bc['value'])))
            
#             bcs.append(DirichletBC(sub_space, value, mf, bc['boundary_id']))
#         return bcs
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
    def setup_boundary_conditions(W, mf, config, U_inlet=None):
        bcs = []
        for bc in config['boundary_conditions']:
            if bc['type'] != 'Dirichlet':
                continue
            sub_space = W.sub(bc['sub_space'])
            
            if bc.get('bc_type') == 'cylinder':

                value = Expression(('0', '0'), degree=1)

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
    
    # Setup mesh and function spaces
    mesh, mf, cf = MeshHandler.load_mesh(config['mesh_file'], 
                                       config['mesh_function_file'])
    W, Q = SolverSetup.create_function_spaces(mesh)
    
    # Setup boundary conditions
    U_inlet = BoundaryConditions.InletProfile(t=0)
    bcs = SolverSetup.setup_boundary_conditions(W, mf, config, U_inlet)
    
    # Setup measures for force calculations
    ds_circle_4 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4)
    ds_circle_5 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=5)
    ds_circle_6 = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=6)

    ds_circle = ds_circle_4 + ds_circle_5 + ds_circle_6

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
        'u0_file': config.get('u0_file', "results/velocity.xdmf")
    }

    if config['steady_solver']:
        solve_steady_navier_stokes(**configuration)
    else:
        solve_unsteady_navier_stokes(**configuration)

if __name__ == "__main__":
    main()