from dolfin import *
from rbnics import *
from rbnics.backends.online import OnlineFunction
from rbnics.backends import assign, export, import_
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from utils import *

def perform_bifurcation_analysis(problem, reduced_problem, config):
    """Perform bifurcation analysis and generate plots"""
    print("Starting bifurcation analysis...")
    mesh_manager = MeshManager()
    
    # Create output directory if it doesn't exist
    os.makedirs(config["bifurcation"]["output_dir"], exist_ok=True)
    
    # Setup bifurcation analysis parameters with proper tuple format
    Re_range = np.linspace(
        config["bifurcation"]["Re_start"],
        config["bifurcation"]["Re_end"],
        config["bifurcation"]["Re_num"]
    )
    mu_range = 1 / Re_range
    
    # Initialize output arrays
    hf_output = []  # High-fidelity output
    rb_output = []  # Reduced basis output
    flag_bifurcation = config["bifurcation"]["enabled"]
    
    # Perform analysis
    for i, Re in enumerate(Re_range):
        print(f"Processing Re = {Re:.2f} ({i+1}/{len(Re_range)})")
        
        mu = 1 / Re
        online_mu = (mu,)  # Maintain tuple format
        
        # High-fidelity solution
        problem.set_mu(online_mu)
        solution = problem.solve()
        problem.export_solution(
            config["simulation_name"],
            "online_solution_hf",
            suffix=i
        )
        hf_output.append(calculate_lift(solution, mu, mesh_manager.n1, mesh_manager.ds_circle))
        
        # Reduced basis solution
        reduced_problem.set_mu(online_mu)
        reduced_solution = reduced_problem.solve(flag_bifurcation=flag_bifurcation)
        Z = reduced_problem.basis_functions * reduced_solution
        reduced_problem.export_solution(
            config["simulation_name"],
            "online_solution_ro",
            suffix=i
        )
        rb_output.append(calculate_lift(Z, mu, mesh_manager.n1, mesh_manager.ds_circle))
    
    # Save numerical results
    results_file = os.path.join(config["bifurcation"]["output_dir"], "bifurcation_data.csv")
    data_to_save = np.column_stack((Re_range, hf_output, rb_output))
    np.savetxt(
        results_file,
        data_to_save,
        delimiter=',',
        header='Re,HF_output,RB_output',
        comments=''
    )
    print(f"Numerical results saved to {results_file}")
    
    # Generate and save plot
    plt.figure("Bifurcation analysis")
    plt.plot(Re_range, hf_output, "-r", linewidth=2, label="HF output")
    plt.plot(Re_range, rb_output, "--b", linewidth=2, label="RB output")
    plt.xlabel('Re')
    plt.ylabel('$C_L$')
    plt.title("Bifurcation Diagram")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_file = os.path.join(config["bifurcation"]["output_dir"], "bifurcation_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bifurcation plot saved to {plot_file}")
    
    return {
        "Re_range": Re_range.tolist(),
        "hf_output": hf_output,
        "rb_output": rb_output
    }

class MeshManager:
    """Singleton class to manage mesh and related functions"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MeshManager, cls).__new__(cls)
            cls._instance.mesh = None
            cls._instance.boundaries = None
            cls._instance.subdomains = None
            cls._instance.ds_circle = None
            cls._instance.n1 = None
        return cls._instance
    
    def initialize_mesh(self, config):
        """Initialize mesh and related functions"""
        self.mesh = Mesh()
        mvc = MeshValueCollection("size_t", self.mesh, self.mesh.topology().dim())
        with XDMFFile(config["mesh"]["file"]) as infile:
            infile.read(self.mesh)
            infile.read(mvc, "name_to_read")
        self.subdomains = cpp.mesh.MeshFunctionSizet(self.mesh, mvc)

        mvc = MeshValueCollection("size_t", self.mesh, self.mesh.topology().dim()-1)
        with XDMFFile(config["mesh"]["function_file"]) as infile:
            infile.read(mvc, "name_to_read")
        self.boundaries = cpp.mesh.MeshFunctionSizet(self.mesh, mvc)
        
        # Surface measures
        ds_circle_4 = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries, subdomain_id=4)
        ds_circle_5 = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries, subdomain_id=5)
        ds_circle_6 = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries, subdomain_id=6)
        self.ds_circle = ds_circle_4 + ds_circle_5 + ds_circle_6
        
        self.n1 = -FacetNormal(self.mesh)
    
    def get_function_space(self):
        """Create and return the function space"""
        element_u = VectorElement("Lagrange", self.mesh.ufl_cell(), 2)
        element_p = FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        element = MixedElement(element_u, element_p)
        return FunctionSpace(self.mesh, element, components=[["u", "s"], "p"])

@DEIM("online")
@ExactParametrizedFunctions("offline")
class Pinball(NavierStokesProblem):
    def __init__(self, V, **kwargs):
        self.config = kwargs.pop("config")
        NavierStokesProblem.__init__(self, V, **kwargs)
        
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        dup = TrialFunction(V)
        (self.du, self.dp) = split(dup)
        (self.u, _) = split(self._solution)
        vq = TestFunction(V)
        (self.v, self.q) = split(vq)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        self.inlet = Expression(("20*(x[1] - 2.5)*(5 - x[1])", "0."), degree=2)
        self.f = Constant((0.0, 0.0))
        self.g = Constant(0.0)
        self._solution_prev = Function(V)
        
        self._nonlinear_solver_parameters.update({
            "linear_solver": "umfpack",
            "maximum_iterations": 20,
            "report": True
        })

    def _compute_initial_state(self):
        mesh_manager = MeshManager()
        element_u = VectorElement("Lagrange", mesh_manager.mesh.ufl_cell(), 2)
        element_p = FiniteElement("Lagrange", mesh_manager.mesh.ufl_cell(), 1)
        element = MixedElement(element_u, element_p)
        V = FunctionSpace(mesh_manager.mesh, element, components=[["u", "s"], "p"])
        w_initial = Function(V)
        
        xdmf_file = XDMFFile(self.config["initial_conditions"]["velocity_file"])
        u_initial = Function(FunctionSpace(mesh_manager.mesh, element_u))
        xdmf_file.read_checkpoint(u_initial, "u_out", 0)
        assign(w_initial.sub(0), u_initial)
        
        return w_initial

    def name(self):
        return self.config["simulation_name"]

    @compute_theta_for_derivatives
    @compute_theta_for_supremizers
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            return (mu[0],)  
        elif term in ("b", "bt"):
            return (1.,)  
        elif term == "c":
            return (1.,)  
        elif term == "f":
            return (1.,)  
        elif term == "g":
            return (1.,)  
        elif term == "dirichlet_bc_u":
            return (1.,)  
        else:
            raise ValueError("Invalid term for compute_theta().")

    @assemble_operator_for_derivatives
    @assemble_operator_for_supremizers
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            u = self.du
            v = self.v
            return (inner(grad(u), grad(v))*dx,)  
        elif term == "b":
            u = self.du
            q = self.q
            return (- q*div(u)*dx,)  
        elif term == "bt":
            p = self.dp
            v = self.v
            return (- p*div(v)*dx,)  
        elif term == "c":
            u = self.u
            v = self.v
            return (inner(grad(u)*u, v)*dx,)  
        elif term == "f":
            v = self.v
            return (inner(self.f, v)*dx,)  
        elif term == "g":
            q = self.q
            return (self.g*q*dx,)  
        elif term == "dirichlet_bc_u":
            bc0 = [DirichletBC(self.V.sub(0), Constant((1.0, 0.0)), self.boundaries, 1),
                   DirichletBC(self.V.sub(0), Constant((1.0, 0.0)), self.boundaries, 3),
                   DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 4),
                   DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 5),
                   DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 6)]
            return (bc0,)  
        elif term == "inner_product_u":
            u = self.du
            v = self.v
            return (inner(grad(u), grad(v))*dx,)  
        elif term == "inner_product_p":
            p = self.dp
            q = self.q
            return (inner(p, q)*dx,)  
        else:
            raise ValueError("Invalid term for assemble_operator().")

    def _solve(self, **kwargs):
        self._solution_prev = self._compute_initial_state()
        assign(self._solution, self._solution_prev)
        NavierStokesProblem._solve(self, **kwargs)
        assign(self._solution_prev, self._solution)

@CustomizeReducedProblemFor(NavierStokesProblem)
def CustomizeReducedNavierStokes(ReducedNavierStokes_Base):
    class ReducedNavierStokes(ReducedNavierStokes_Base):
        def __init__(self, truth_problem, **kwargs):
            ReducedNavierStokes_Base.__init__(self, truth_problem, **kwargs)
            self._solution_prev = None
            self._nonlinear_solver_parameters.update({
                "report": True,
                "line_search": "wolfe",
                "maximum_iterations": 20
            })
            self._projected_initial_state = None  # Cache for projected initial state
            self._cached_N = None  # Store N for which the projection was done
            self.flag = False

        def _compute_initial_state(self):
            mesh_manager = MeshManager()
            element_u = VectorElement("Lagrange", mesh_manager.mesh.ufl_cell(), 2)
            element_p = FiniteElement("Lagrange", mesh_manager.mesh.ufl_cell(), 1)
            element = MixedElement(element_u, element_p)
            V = FunctionSpace(mesh_manager.mesh, element, components=[["u", "s"], "p"])
            w_initial = Function(V)
            
            xdmf_file = XDMFFile(self.truth_problem.config["initial_conditions"]["velocity_file"])
            u_initial = Function(FunctionSpace(mesh_manager.mesh, element_u))
            xdmf_file.read_checkpoint(u_initial, "u_out", 0)
            assign(w_initial.sub(0), u_initial)
            return w_initial

       # def _project_initial_state(self, initial_state, N):
        #    return self.project(initial_state, min(N.values()))
        def _project_initial_state(self, initial_state, N):
    # Check if we already have a projection for this N
            if self._projected_initial_state is not None and self._cached_N == N:
                print("Using cached projected initial state")
                new_solution = OnlineFunction(N)
                assign(new_solution, self._projected_initial_state)
                return new_solution
    
            print("Computing new projected initial state")
            projected = self.project(initial_state, min(N.values()))
            self._projected_initial_state = OnlineFunction(N)
            assign(self._projected_initial_state, projected)
            self._cached_N = N
            return projected
        
        def _solve(self, N, **kwargs):
            flag_bifurcation = kwargs.get('flag_bifurcation')
            timer = kwargs.pop('timer',None) 
            if self.flag:
                assign(self._solution, self._solution_prev)
            
            if not flag_bifurcation:
                try:
                    initial_state = self._compute_initial_state()
                    print("Initial state computed successfully")
                  
                    self._solution = self._project_initial_state(initial_state, N)
                    print("Initial state projected successfully")
                   
                except Exception as e:
                    print(f"Error in solve:")
                    print(f"- Stage: {'project' if 'initial_state' in locals() else 'compute initial'}")
                    print(f"- Error: {str(e)}")
                    self._solution = None
                    raise
            
            ReducedNavierStokes_Base._solve(self, N, **kwargs)
            
            if flag_bifurcation:
                self._solution_prev = OnlineFunction(N)
                assign(self._solution_prev, self._solution)
                self.flag = True
            
    return ReducedNavierStokes

def calculate_lift(w, nu, n1, ds_circle):
    u, p = w.split()
    u_t = inner(as_vector((n1[1], -n1[0])), u)
    lift = assemble(-2/(1.)*(Constant(nu)*inner(grad(u_t), n1)*n1[0] + p*n1[1])*ds_circle)
    return lift

def DEIM_convergence(reduction_method, config):
    N_DEIM_c = reduction_method.reduced_problem.truth_problem.DEIM_approximations['c'][0].N
    N_DEIM_dc = reduction_method.reduced_problem.truth_problem.DEIM_approximations['dc'][0].N
    
    # Use the minimum between the two terms
    N_DEIM_max = min(N_DEIM_c, N_DEIM_dc)
    N_DEIM_min = 1
    errors_u = []
    errors_p = []
    N_values = list(range(N_DEIM_min, N_DEIM_max))
    components = ['u', 'p']
    reduction_method.reduced_problem._solution_cache.clear()
    
    # Fix: Properly format the online_mu parameter as a tuple
    online_mu = (0.0125,)
    reduction_method.reduced_problem.set_mu(online_mu)

    for N in N_values:
        print(f'Solving for N ={N}')
        reduction_method.reduced_problem.solve(DEIM=N, flag_bifurcation=False)
        error = reduction_method.reduced_problem.compute_error()
        errors_u.append(error[components[0]])
        errors_p.append(error[components[1]])

    # Create the semi-log plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(N_values, errors_u, 'b-o', label='DEIM Error (u)')
    plt.semilogy(N_values, errors_p, 'r-s', label='DEIM Error (p)')
    plt.grid(True)
    plt.xlabel('N (DEIM basis functions)')
    plt.ylabel('Error (log scale)')
    plt.title('DEIM Convergence Analysis')
    plt.legend()
    plt.savefig(f'{config["simulation_name"]}/DEIM_convergence.png')
    plt.close()
    return {'u': errors_u, 'p': errors_p}

def run_simulation(config):
    """Core simulation function"""
    # Initialize mesh manager
    mesh_manager = MeshManager()
    mesh_manager.initialize_mesh(config)
    
    # Get function space
    V = mesh_manager.get_function_space()
    
    # Setup parameter ranges - maintaining specific tuple formats
    mu_range = [(1./config["parameters"]["Re_range"][0], 1./config["parameters"]["Re_range"][1])]
    lifting_mu = (config["parameters"]["lifting_mu"],)
    online_mu = (1./config["parameters"]["online_Re"],)
    
    # Initialize problem
    problem = Pinball(V, config=config, subdomains=mesh_manager.subdomains, boundaries=mesh_manager.boundaries)
    problem.set_mu_range(mu_range)

    # Setup reduction method
    reduction_method = PODGalerkin(problem)
    reduction_method.set_Nmax(
        config["max_basis"]["rom"],
        DEIM=config["max_basis"]["deim"]
    )
    #reduction_method.set_tolerance(
    #    config["tolerances"]["rom"],
    #    DEIM=config["tolerances"]["deim"]
    #)

    # Initialize training with proper lifting_mu tuple
    problem.set_mu(lifting_mu)
    reduction_method.initialize_training_set(
        config["snapshots"]["training"],
        DEIM=config["snapshots"]["deim"],
        sampling=EquispacedDistribution()
    )

    # Initialize testing set
    reduction_method.initialize_testing_set(
        config["snapshots"]["testing"],
        DEIM = config["snapshots"]["testing_DEIM"],
        sampling=EquispacedDistribution()
    )
    
    # Perform offline phase
    reduced_problem = reduction_method.offline()
    flag_bifurcation = config["bifurcation"]["enabled"]
    N_max = 12
    # Online solve with proper online_mu tuple
    reduced_problem.set_mu(online_mu)
    print(f'The bifurcation flag before solve is: {flag_bifurcation}')
    reduced_solution = reduced_problem.solve(flag_bifurcation=flag_bifurcation)
    reduced_problem.export_solution(config["simulation_name"], "online_solution")
    
    # Calculate lift
    Z = reduced_problem.basis_functions * reduced_solution
    lift_value = calculate_lift(Z, online_mu[0], mesh_manager.n1, mesh_manager.ds_circle)
    N_DEIM_c = reduction_method.reduced_problem.truth_problem.DEIM_approximations['c'][0].N
    N_DEIM_dc = reduction_method.reduced_problem.truth_problem.DEIM_approximations['dc'][0].N
    
    # Use the minimum between the two terms
    N_DEIM_max = min(N_DEIM_c, N_DEIM_dc)


    # Prepare results
    results = {
        "lift_coefficient": float(lift_value),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if config["analysis"]["deim_convergence"]:
        print("Starting DEIM convergence study...")
        err = DEIM_convergence(reduction_method, config)
       

    # Perform bifurcation analysis if enabled
    if config["bifurcation"]["enabled"]:
        bifurcation_results = perform_bifurcation_analysis(
            problem,
            reduced_problem,
            config
        )
        results["bifurcation_analysis"] = bifurcation_results
    reduction_method.error_analysis()
    if config["analysis"]["error_analysis"]:
        print("Starting error analysis...")
        error_analysis_pinball(reduction_method, N_max,  filename="error_analysis")
       # results["error_analysis"] = error_results
    
    # Perform speedup analysis if enabled
    if config["analysis"]["speedup_analysis"]:
        print("Starting speedup analysis...")
        speedup_analysis_pinball(reduction_method, N_max, filename="speedup_analysis")
        #results["speedup_analysis"] = speedup_results

    return results
