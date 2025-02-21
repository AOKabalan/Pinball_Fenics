from dolfin import *
from rbnics import *
from rbnics.backends.online import OnlineFunction
from rbnics.backends import assign
import numpy as np

# Embedded configuration
CONFIG = {
    "simulation_name": "Pinball_rotation_trial",
    "tolerances": {"rom": 0.},
    "max_basis": {"rom": 20},
    "snapshots": {"training": 200, "testing": 100},
    "mesh": {
        "file": "data2/mesh.xdmf",
        "function_file": "data2/mf.xdmf"
    },
    "initial_conditions": {
        "velocity_file": "velocity_checkpoint.xdmf"
    },
    "parameters": {
        "Re_range": [55, 100],
        "lifting_mu": 0.017,
        "online_Re": 80
    },
    "bifurcation": {
        "enabled": False,
        "Re_start": 55.,
        "Re_end": 85.,
        "Re_num": 100.,
        "output_dir": "bifurcation_results"
    },
    "analysis": {
        "error_analysis": False,
        "speedup_analysis": False
    }
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

        mvc = MeshValueCollection("size_t", self.mesh, self.mesh.topology().dim() - 1)
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


@ExactParametrizedFunctions()
class Pinball(NavierStokesProblem):
    def __init__(self, V, **kwargs):
        self.config = kwargs.pop("config")
        NavierStokesProblem.__init__(self, V, **kwargs)
        mesh_manager = MeshManager()
        self.subdomains = mesh_manager.subdomains
        self.boundaries = mesh_manager.boundaries

        dup = TrialFunction(V)
        (self.du, self.dp) = split(dup)
        (self.u, _) = split(self._solution)
        vq = TestFunction(V)
        (self.v, self.q) = split(vq)

        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)

        self.f = Constant((0.0, 0.0))
        self.g = Constant(0.0)

        self._solution_prev = Function(V)
        self._nonlinear_solver_parameters.update({
            "linear_solver": "umfpack",
            "maximum_iterations": 20,
            "report": True
        })
        self.top_rot = Expression(('sin(atan2(x[1]-cy, x[0]-cx))',
                                     '-cos(atan2(x[1]-cy, x[0]-cx))'),
                                    degree=1,                                    
                                    cx=0.0,
                                    cy=0.75)
        self.bottom_rot = Expression(('-sin(atan2(x[1]-cy, x[0]-cx))',
                                     'cos(atan2(x[1]-cy, x[0]-cx))'),
                                    degree=1,                                    
                                    cx=0.0,
                                    cy=-0.75) 

    def _compute_initial_state(self):
        """Compute initial state for the problem"""
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
            theta_bc0 = 1.
            theta_bc1 = mu[1]
            return (theta_bc0, theta_bc1)
        else:
            raise ValueError("Invalid term for compute_theta().")


    # Simple continuation method to reconstruct a branch of the bifurcation diagram
    def _solve(self, **kwargs):
        assign(self._solution, self._solution_prev)
        NavierStokesProblem._solve(self, **kwargs)
        assign(self._solution_prev, self._solution)

    @assemble_operator_for_derivatives
    @assemble_operator_for_supremizers
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            u = self.du
            v = self.v
            return (inner(grad(u), grad(v)) * dx,)
        elif term == "b":
            u = self.du
            q = self.q
            return (-q * div(u) * dx,)
        elif term == "bt":
            p = self.dp
            v = self.v
            return (-p * div(v) * dx,)
        elif term == "c":
            u = self.u
            v = self.v
            return (inner(grad(u) * u, v) * dx,)
        elif term == "f":
            v = self.v
            return (inner(self.f, v) * dx,)
        elif term == "g":
            q = self.q
            return (self.g * q * dx,)
        elif term == "dirichlet_bc_u":
            bc0 = [
                DirichletBC(self.V.sub(0), Constant((1.0, 0.0)), self.boundaries, 1),
                DirichletBC(self.V.sub(0), Constant((1.0, 0.0)), self.boundaries, 3),
                DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 4),
                DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 5),
                DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 6)
            ]
            bc1 = [
                DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 1),
                DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 3),
                DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 4),
                DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 5),
                DirichletBC(self.V.sub(0), Constant((0.0, 0.0)), self.boundaries, 6)
            ]
            return (bc0, bc1)
        elif term == "inner_product_u":
            u = self.du
            v = self.v
            return (inner(grad(u), grad(v)) * dx,)
        elif term == "inner_product_p":
            p = self.dp
            q = self.q
            return (inner(p, q) * dx,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


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
            """Compute initial state for the reduced problem"""
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

        def _project_initial_state(self, initial_state, N):
            return self.project(initial_state, min(N.values()))

        def _solve(self, N, **kwargs):
            flag_bifurcation = kwargs.get('flag_bifurcation')
            if self.flag:
                assign(self._solution, self._solution_prev)

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


def run_simulation():
    """Core simulation function"""
    # Initialize mesh manager
    mesh_manager = MeshManager()
    mesh_manager.initialize_mesh(CONFIG)

    # Get function space
    V = mesh_manager.get_function_space()

    # Setup parameter ranges
    mu_range = [(1. / CONFIG["parameters"]["Re_range"][0], 1. / CONFIG["parameters"]["Re_range"][1]), (2.0, -2.0)]
    lifting_mu = (CONFIG["parameters"]["lifting_mu"], 2.0)
    online_mu = (1. / CONFIG["parameters"]["online_Re"], 0)

    # Initialize problem
    problem = Pinball(V, config=CONFIG, subdomains=mesh_manager.subdomains, boundaries=mesh_manager.boundaries)
    problem.set_mu_range(mu_range)

    # Setup reduction method
    reduction_method = PODGalerkin(problem)
    reduction_method.set_Nmax(CONFIG["max_basis"]["rom"])
    reduction_method.set_tolerance(CONFIG["tolerances"]["rom"])

    # Initialize training
    problem.set_mu(lifting_mu)
    reduction_method.initialize_training_set(
        CONFIG["snapshots"]["training"],
        sampling=EquispacedDistribution()
    )

    # Perform offline phase
    reduced_problem = reduction_method.offline()

    # Online solve
    reduced_problem.set_mu(online_mu)
    reduced_solution = reduced_problem.solve(flag_bifurcation=CONFIG["bifurcation"]["enabled"])
    reduced_problem.export_solution(CONFIG["simulation_name"], "online_solution")

    return {"status": "Simulation completed successfully"}


if __name__ == "__main__":
    run_simulation()