
from dolfin import *
from rbnics import *
from rbnics.backends.online import OnlineFunction
from rbnics.backends import assign
import numpy as np
import matplotlib.pyplot as plt
from utils import *

mesh_file = 'data2/mesh.xdmf'
mesh_function_file = 'data2/mf.xdmf'

mesh = Mesh()
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile(mesh_file) as infile:
    infile.read(mesh)
    infile.read(mvc, "name_to_read")
subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile(mesh_function_file) as infile:
    infile.read(mvc, "name_to_read")
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

flag_initial_u = 'True'
u0_file= "velocity_checkpoint.xdmf"

@ExactParametrizedFunctions()
class Pinball(NavierStokesProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        NavierStokesProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
       
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u_initial= kwargs["u_initial"]
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
        
        w_initial = Function(V)
        assign(w_initial.sub(0), self.u_initial)
        self._solution.vector()[:] = w_initial.vector()
    
        # Customize nonlinear solver parameters
        self._nonlinear_solver_parameters.update({
            "linear_solver": "umfpack",
            "maximum_iterations": 20,
            "report": True
        })

    # Return custom problem name
    def name(self):
        return "FluidicPinball"

    # Return theta multiplicative terms of the affine expansion of the problem.
    @compute_theta_for_derivatives
    @compute_theta_for_supremizers
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = mu[0]
            return (theta_a0,)
        elif term in ("b", "bt"):
            theta_b0 = 1.
            return (theta_b0,)
        elif term == "c":
            theta_c0 = 1.
            return (theta_c0,)
        elif term == "f":
            theta_f0 = 1.
            return (theta_f0,)
        elif term == "g":
            theta_g0 = 1.
            return (theta_g0,)
        elif term == "dirichlet_bc_u":
            theta_bc00 = 1.
            return (theta_bc00,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    @assemble_operator_for_derivatives
    @assemble_operator_for_supremizers
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            u = self.du
            v = self.v
            a0 = inner(grad(u), grad(v))*dx
            return (a0,)
        elif term == "b":
            u = self.du
            q = self.q
            b0 = - q*div(u)*dx
            return (b0,)
        elif term == "bt":
            p = self.dp
            v = self.v
            bt0 = - p*div(v)*dx
            return (bt0,)
        elif term == "c":
            u = self.u
            v = self.v
            c0 = inner(grad(u)*u, v)*dx
            return (c0,)
        elif term == "f":
            v = self.v
            f0 = inner(self.f, v)*dx
            return (f0,)
        elif term == "g":
            q = self.q
            g0 = self.g*q*dx
            return (g0,)
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
            x0 = inner(grad(u), grad(v))*dx
            return (x0,)
        elif term == "inner_product_p":
            p = self.dp
            q = self.q
            x0 = inner(p, q)*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")

    # Simple continuation method to reconstruct a branch of the bifurcation diagram
    def _solve(self, **kwargs):
        
        assign(self._solution, self._solution_prev)
        
        NavierStokesProblem._solve(self, **kwargs)
        assign(self._solution_prev, self._solution)



# Customize the resulting reduced problem to enable simple continuation at reduced level
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

        def _solve(self, N, **kwargs):
            if self.flag:
                assign(self._solution, self._solution_prev)
            ReducedNavierStokes_Base._solve(self, N, **kwargs)
            self._solution_prev = OnlineFunction(N)
            assign(self._solution_prev, self._solution)
            self.flag = True
    return ReducedNavierStokes

# mesh = Mesh("data2/mesh.xdmf")
# subdomains = MeshFunction("size_t", mesh, "data/channel_physical_region.xml")
# boundaries = MeshFunction("size_t", mesh, "data/channel_facet_region.xml")


element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(element_u, element_p)
V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])

if flag_initial_u:
    xdmf_file = XDMFFile(u0_file)

    u_initial = Function(FunctionSpace(mesh,element_u))
    xdmf_file.read_checkpoint(u_initial, "u_out", 0)


problem = Pinball(V, subdomains=subdomains, boundaries=boundaries,u_initial= u_initial)

mu_range = [(0.15, 0.0125)]
problem.set_mu_range(mu_range)

reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(20)
reduction_method.set_tolerance(1e-8)


lifting_mu = (0.0125,)
problem.set_mu(lifting_mu)
solution = problem.solve()
problem.export_solution("FluidicPinball", "test_sol")
# reduction_method.initialize_training_set(51, sampling=EquispacedDistribution())
# reduced_problem = reduction_method.offline()


# online_mu = (0.0125, )
# reduced_problem.set_mu(online_mu)
# reduced_solution = reduced_problem.solve()


# # reduction_method.initialize_training_set(16, sampling=EquispacedDistribution())
# # reduction_method.error_analysis()
# plt.figure()


# plot(reduced_solution, reduced_problem=reduced_problem, component="u")    

# plt.show()