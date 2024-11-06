from dolfin import *
from rbnics import *
from rbnics.backends.online import OnlineFunction
from rbnics.backends import assign, export, import_
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from datetime import datetime
import json
import os

# Configuration dictionary
config = {
    "simulation_name": "PinballDEIM",
    "tolerances": {
        "rom": 1e-15,  # Reduced Order Model tolerance
        "deim": 1e-15  # DEIM tolerance
    },
    "max_basis": {
        "rom": 20,    # Maximum number of ROM basis functions
        "deim": 50    # Maximum number of DEIM basis functions
    },
    "snapshots": {
        "training": 100,  # Number of training snapshots
        "testing": 50,   # Number of testing snapshots
        "deim": 144,
        "testing_DEIM": 55      # Number of DEIM snapshots
    },
    "mesh": {
        "file": "data2/mesh.xdmf",
        "function_file": "data2/mf.xdmf"
    },
    "initial_conditions": {
        "velocity_file": "velocity_checkpoint.xdmf"
    },
    "parameters": {
        "mu_range": [(0.017, 0.01)],
        "lifting_mu": (0.017,),
        "online_mu": (0.0125,)
    },
    "bifurcation": {
        "enabled": False,
        "Re_start": 55,   # Corresponds to mu_start = 0.03
        "Re_end": 85,     # Corresponds to mu_end = 0.017
        "Re_num": 100,    # Number of points for bifurcation analysis
        "output_dir": "bifurcation_results"
    }
}

def save_run_info(config, results):
    """Save configuration and results to a timestamped file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config["simulation_name"], exist_ok=True)
    filename = f"{config['simulation_name']}/run_info_{timestamp}.txt"
    
    output_data = {
        "configuration": config,
        "results": results
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Run information saved to {filename}")

def perform_bifurcation_analysis(problem, reduced_problem, config, n1, ds_circle):
    """Perform bifurcation analysis and generate plots"""
    print("Starting bifurcation analysis...")
    
    os.makedirs(config["bifurcation"]["output_dir"], exist_ok=True)
    
    Re_range = np.linspace(
        config["bifurcation"]["Re_start"],
        config["bifurcation"]["Re_end"],
        config["bifurcation"]["Re_num"]
    )
    mu_range = 1 / Re_range
    
    hf_output = []  # High-fidelity output
    rb_output = []  # Reduced basis output
    flag_bifurcation = config["bifurcation"]["enabled"]
    
    for i, Re in enumerate(Re_range):
        print(f"Processing Re = {Re:.2f} ({i+1}/{len(Re_range)})")
        
        mu = 1 / Re
        online_mu = (mu,)
        
        # High-fidelity solution
        problem.set_mu(online_mu)
        solution = problem.solve()
        problem.export_solution(
            config["simulation_name"],
            "online_solution_hf",
            suffix=i
        )
        hf_output.append(calculate_lift(solution, mu, n1, ds_circle))
        
        # Reduced basis solution
        reduced_problem.set_mu(online_mu)
        reduced_solution = reduced_problem.solve(flag_bifurcation=flag_bifurcation)
        Z = reduced_problem.basis_functions * reduced_solution
        reduced_problem.export_solution(
            config["simulation_name"],
            "online_solution_ro",
            suffix=i
        )
        rb_output.append(calculate_lift(Z, mu, n1, ds_circle))
    
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
    
    # Generate and save plot
    plt.figure("Bifurcation analysis")
    plt.plot(Re_range, hf_output, "-r", linewidth=2, label="HF output")
    plt.plot(Re_range, rb_output, "--b", linewidth=2, label="RB output")
    plt.xlabel('Re')
    plt.ylabel('$C_L$')
    plt.title("Bifurcation Diagram")
    plt.legend()
    plt.grid(True)
    
    plot_file = os.path.join(config["bifurcation"]["output_dir"], "bifurcation_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "Re_range": Re_range.tolist(),
        "hf_output": hf_output,
        "rb_output": rb_output
    }

# Load mesh and mesh functions
mesh = Mesh()
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile(config["mesh"]["file"]) as infile:
    infile.read(mesh)
    infile.read(mvc, "name_to_read")
subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile(config["mesh"]["function_file"]) as infile:
    infile.read(mvc, "name_to_read")
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# Surface measures
ds_circle_4 = Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=4)
ds_circle_5 = Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=5)
ds_circle_6 = Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=6)
ds_circle = ds_circle_4 + ds_circle_5 + ds_circle_6

n1 = -FacetNormal(mesh)

@DEIM("online")
@ExactParametrizedFunctions("offline")
class Pinball(NavierStokesProblem):
    def __init__(self, V, **kwargs):
        NavierStokesProblem.__init__(self, V, **kwargs)
        
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
       
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
        element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        element = MixedElement(element_u, element_p)
        V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])
        w_initial = Function(V)
        
        xdmf_file = XDMFFile(config["initial_conditions"]["velocity_file"])
        u_initial = Function(FunctionSpace(mesh, element_u))
        xdmf_file.read_checkpoint(u_initial, "u_out", 0)
        assign(w_initial.sub(0), u_initial)
        
        return w_initial

    def name(self):
        return config["simulation_name"]

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
            self.flag = False

        def _compute_initial_state(self):
            element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
            element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
            element = MixedElement(element_u, element_p)
            V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])
            w_initial = Function(V)
            
            xdmf_file = XDMFFile(config["initial_conditions"]["velocity_file"])
            u_initial = Function(FunctionSpace(mesh, element_u))
            xdmf_file.read_checkpoint(u_initial, "u_out", 0)
            assign(w_initial.sub(0), u_initial)
            return w_initial

        def _project_initial_state(self, initial_state, N):
            return self.project(initial_state, min(N.values()))

        def _solve(self, N, **kwargs):
            flag_bifurcation = kwargs.get('flag_bifurcation')
            
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

def run_simulation(config):
    # Setup function spaces
    element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_u, element_p)
    V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])

    # Initialize problem
    problem = Pinball(V, subdomains=subdomains, boundaries=boundaries)
    problem.set_mu_range(config["parameters"]["mu_range"])

    # Setup reduction method
    reduction_method = PODGalerkin(problem)
    reduction_method.set_Nmax(
        config["max_basis"]["rom"],
        DEIM=config["max_basis"]["deim"]
    )
    reduction_method.set_tolerance(
        config["tolerances"]["rom"],
        DEIM=config["tolerances"]["deim"]
    )

    # Initialize training
    problem.set_mu(config["parameters"]["lifting_mu"])
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

    # Online solve
    reduced_problem.set_mu(config["parameters"]["online_mu"])
    print(f'The bifurcation flag before solve is: {flag_bifurcation}')
    reduced_solution = reduced_problem.solve(flag_bifurcation=flag_bifurcation)
    reduced_problem.export_solution(config["simulation_name"], "online_solution")
    
    # Calculate lift
    Z = reduced_problem.basis_functions * reduced_solution
    lift_value = calculate_lift(Z, config["parameters"]["online_mu"][0], n1, ds_circle)
    
    # Prepare results
    results = {
        "lift_coefficient": float(lift_value),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Perform bifurcation analysis if enabled
    if config["bifurcation"]["enabled"]:
        bifurcation_results = perform_bifurcation_analysis(
            problem,
            reduced_problem,
            config,
            n1,
            ds_circle
        )
        results["bifurcation_analysis"] = bifurcation_results
    
    return results

if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs(config["simulation_name"], exist_ok=True)
        if config["bifurcation"]["enabled"]:
            os.makedirs(config["bifurcation"]["output_dir"], exist_ok=True)
        
        # Run simulation and save results
        print("Starting simulation...")
        results = run_simulation(config)
        save_run_info(config, results)
        print("Simulation completed successfully!")
        
        # Print summary
        print("\nSimulation Summary:")
        print(f"- Lift coefficient: {results['lift_coefficient']:.6f}")
        print(f"- Results saved at: {results['timestamp']}")
        if config["bifurcation"]["enabled"]:
            print("- Bifurcation analysis completed and saved")
            
    except Exception as e:
        print(f"\nError during simulation execution:")
        print(f"- Type: {type(e).__name__}")
        print(f"- Message: {str(e)}")
        raise