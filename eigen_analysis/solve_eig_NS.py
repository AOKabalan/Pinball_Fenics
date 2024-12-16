from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from solver import solve_steady_navier_stokes
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
    def setup_boundary_conditions(W, mf, config, U_inlet=None):
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
                                    amp=bc['amplitude'],
                                    cx=bc['cylinder_center'][0],
                                    cy=bc['cylinder_center'][1])
                elif bc['cylinder_type'] == 'bottom':
                    value = Expression(('-amp*sin(atan2(x[1]-cy, x[0]-cx))',
                                     'amp*cos(atan2(x[1]-cy, x[0]-cx))'),
                                    degree=1,
                                    amp=bc['amplitude'],
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
class StabilityAnalysis:
    def __init__(self, mesh, W, bcs, config):
        self.mesh = mesh
        self.W = W
        self.bcs = bcs
        self.nu = config['nu']
        self.N_eig = config.get('N_eig', 10)  # Number of eigenvalues to compute
        
        # Create trial and test functions for the stability analysis
        self.solution = Function(W)
        self.dup_e = TrialFunction(W)
        self.v_e = TestFunction(W)
        
        # Split functions for velocity and pressure
        self.du_e, self.dp_e = split(self.dup_e)
        self.v_e, self.q_e = split(self.v_e)
        
        # Initialize storage for eigenvalues
        self.eigenvalue_r = [[] for _ in range(self.N_eig)]
        self.eigenvalue_c = [[] for _ in range(self.N_eig)]
        
    def setup_base_solution(self, y_hf, p_hf):
        """Set up the base solution for stability analysis"""
        self.y_hf = y_hf
        self.p_hf = p_hf
        
    def compute_stability_matrices(self, mu):
        """Compute the stability matrices G and B"""
        # Generalized eigenvalue problem matrices
        G_form = (mu*(inner(grad(self.y_hf), grad(self.v_e))*dx) 
                 - self.q_e*div(self.y_hf)*dx 
                 - self.p_hf*div(self.v_e)*dx 
                 + inner(grad(self.y_hf)*self.y_hf, self.v_e)*dx)
        
        # Compute derivative for linearized system
        G_form_der = derivative(G_form, self.solution, self.dup_e)
        
        # Assemble matrices
        G = PETScMatrix()
        assemble(G_form_der, tensor=G)
        
        B_form = inner(grad(self.du_e), grad(self.v_e))*dx + inner(self.dp_e, self.q_e)*dx
        B = PETScMatrix()
        assemble(B_form, tensor=B)
        
        return G, B
    
    def solve_eigenvalue_problem(self, mu):
        """Solve the eigenvalue problem for a given mu"""
        G, B = self.compute_stability_matrices(mu)
        
        # Setup eigenvalue solver
        eigensolver = SLEPcEigenSolver(G, B)
        eigensolver.parameters['problem_type'] = 'gen_non_hermitian'
        eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
        eigensolver.parameters['spectral_shift'] = 1e-4
        eigensolver.parameters['spectrum'] = 'target real'
        
        # Solve eigenvalue problem
        eigensolver.solve(self.N_eig)
        
        # Extract eigenpairs
        for ne in range(self.N_eig):
            rx, cx = Function(self.W), Function(self.W)
            r, c, rx1, cx1 = eigensolver.get_eigenpair(rx, cx, ne)
            self.eigenvalue_r[ne].append(r)
            self.eigenvalue_c[ne].append(c)
            
        return self.eigenvalue_r, self.eigenvalue_c
    
    def compute_errors(self, y_ro, p_ro):
        """Compute errors between high-fidelity and reduced-order solutions"""
        err_l2_vel = sqrt(assemble(inner(self.y_hf-y_ro, self.y_hf-y_ro)*dx))/sqrt(assemble(inner(self.y_hf, self.y_hf)*dx))
        err_h1_vel = sqrt(assemble(inner(grad(self.y_hf-y_ro), grad(self.y_hf-y_ro))*dx))/sqrt(assemble(inner(grad(self.y_hf), grad(self.y_hf))*dx))
        
        err_l2_pre = sqrt(assemble(inner(self.p_hf-p_ro, self.p_hf-p_ro)*dx))/sqrt(assemble(inner(self.p_hf, p_ro)*dx))
        err_h1_pre = sqrt(assemble(inner(grad(self.p_hf-p_ro), grad(self.p_hf-p_ro))*dx))/sqrt(assemble(inner(grad(self.p_hf), grad(self.p_hf))*dx))
        
        return {
            'l2_velocity': err_l2_vel,
            'h1_velocity': err_h1_vel,
            'l2_pressure': err_l2_pre,
            'h1_pressure': err_h1_pre
        }
    
    def plot_eigenvalues(self, save_path='results/eigenvalues.png'):
        """Plot eigenvalues in complex plane"""
        plt.figure("Eigenvalue analysis")
        for ne in range(self.N_eig):
            plt.scatter(self.eigenvalue_r[ne], self.eigenvalue_c[ne])
        
        plt.xlabel('Re(λ)')
        plt.ylabel('Im(λ)')
        plt.title("Eigenvalue Distribution")
        plt.grid(True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        plt.close()

def main():

    # Load configuration and setup mesh from existing code
    config = ConfigHandler.load_config('inputs2.json')
    mesh, mf, cf = MeshHandler.load_mesh(config['mesh_file'], 
                                       config['mesh_function_file'])
    W, Q = SolverSetup.create_function_spaces(mesh)
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
        'results_dir': 'eig_res',
        'flag_initial_u': config.get('flag_initial_u', False),
        'u0_file': config.get('u0_file', "results/velocity.xdmf")
    }
    # Initialize stability analysis
    stability = StabilityAnalysis(mesh, W, bcs, config)
    
    # Compute base solution (you need to implement this based on your solver)
    y_hf, p_hf = solve_steady_navier_stokes(**configuration)
    stability.setup_base_solution(y_hf, p_hf)
    
    # Analyze stability for different mu values
    mu_values = np.linspace(0.1, 2.0, 20)  # Adjust range as needed
    for mu in mu_values:
        eigenvalue_r, eigenvalue_c = stability.solve_eigenvalue_problem(mu)
        
        # If you have a reduced order solution, compute errors
        if 'reduced_order_solution' in config:
            y_ro, p_ro = load_reduced_order_solution(config)
            errors = stability.compute_errors(y_ro, p_ro)
            print(f"\nErrors for mu = {mu}:")
            for error_type, value in errors.items():
                print(f"{error_type}: {value}")
    
    # Plot final eigenvalue distribution
    stability.plot_eigenvalues()

if __name__ == "__main__":
    main()