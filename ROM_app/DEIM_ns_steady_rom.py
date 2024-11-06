
from dolfin import *
from rbnics import *
from rbnics.backends.online import OnlineFunction
from rbnics.backends import assign, export, import_

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


# Prepare surface measure on the three cylinders used for drag and lift
ds_circle_4 = Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=4)
ds_circle_5 = Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=5)
ds_circle_6 = Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=6)

ds_circle = ds_circle_4 + ds_circle_5 + ds_circle_6

#ds_circle = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=4,5,6)

n1 = -FacetNormal(mesh) #Normal pointing out of obstacle

@DEIM("online")
@ExactParametrizedFunctions("offline")
class Pinball(NavierStokesProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        NavierStokesProblem.__init__(self, V, **kwargs)
        
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
       
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        #self.u_initial= kwargs["u_initial"]
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
        
 
        # Customize nonlinear solver parameters
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
        u0_file= "velocity_checkpoint.xdmf"
        xdmf_file = XDMFFile(u0_file)
        
        u_initial = Function(FunctionSpace(mesh,element_u))
        xdmf_file.read_checkpoint(u_initial, "u_out", 0)

        #import_(w_initial, ".", "velocity_checkpoint",suffix=0, component=None)
        assign(w_initial.sub(0), u_initial)

        
        return w_initial
    # Return custom problem name
    def name(self):
        return "FluidicPinballDEIM"

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
        self._solution_prev = self._compute_initial_state()

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



def calculate_lift(w,nu,n1,ds_circle):
       
    u,p = w.split()
    
    u_t = inner(as_vector((n1[1], -n1[0])), u)

    lift = assemble(-2/(1.)*(Constant(nu)*inner(grad(u_t), n1)*n1[0] + p*n1[1])*ds_circle)

    return lift


element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(element_u, element_p)
V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])


problem = Pinball(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.017, 0.01)]
problem.set_mu_range(mu_range)

reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(20, DEIM=20)
reduction_method.set_tolerance(0, DEIM=0)
# hf_output = list()

lifting_mu = (0.017,)
problem.set_mu(lifting_mu)
reduction_method.initialize_training_set(100, DEIM=144, sampling=EquispacedDistribution())
reduced_problem = reduction_method.offline()


# reduction_method.initialize_testing_set(51, sampling=EquispacedDistribution())
# N_max = min(reduced_problem.N.values())

#error_analysis_pinball(reduction_method, N_max, filename="error_analysis")
#speedup_analysis_pinball(reduction_method, N_max, filename="speedup_analysis2")


online_mu = (0.012,)
reduced_problem.set_mu(online_mu)
reduced_solution = reduced_problem.solve()
reduced_problem.export_solution("FluidicPinballDEIM", "test_sol4")
# Z = reduced_problem.basis_functions * reduced_solution
# print((calculate_lift(Z, mu_on, n1, ds_circle)))

flag_bifurcation = False
if flag_bifurcation:
    # Quantities for the bifurcation analysis
    Re_start_bif = 55  # Corresponds to mu_start_bif = 0.03
    Re_end_bif = 85   # Corresponds to mu_end_bif = 0.017
    Re_num_bif = 100
    Re_range_bif = np.linspace(Re_start_bif, Re_end_bif, Re_num_bif)
    mu_range_bif = 1 / Re_range_bif  # Calculate mu from Re

    # Quantities for the bifurcation diagram
    hf_output = []
    rb_output = []

    for (i, Re) in enumerate(Re_range_bif):
        mu = 1 / Re
        online_mu = (mu,)
        problem.set_mu(online_mu)
        solution = problem.solve()
        problem.export_solution("FluidicPinball", "online_solution_hf", suffix=i)
        hf_output.append(calculate_lift(solution, mu, n1, ds_circle))

        reduced_problem.set_mu(online_mu)
        reduced_solution = reduced_problem.solve()
        Z = reduced_problem.basis_functions * reduced_solution
        reduced_problem.export_solution("FluidicPinball", "online_solution_ro", suffix=i)
        rb_output.append(calculate_lift(Z, mu, n1, ds_circle))

        # Save data to file
    data_to_save = np.column_stack((Re_range_bif, hf_output, rb_output))
    np.savetxt('bifurcation_data.csv', data_to_save, delimiter=',', 
               header='Re,HF_output,RB_output', comments='')
    print("Data saved to bifurcation_data.csv")

    plt.figure("Bifurcation analysis")
    plt.plot(Re_range_bif, hf_output, "-r", linewidth=2, label="HF output")
    plt.plot(Re_range_bif, rb_output, "--b", linewidth=2, label="RB output")
    plt.xlabel('Re')
    plt.ylabel('$C_L$')
    plt.title("Bifurcation Diagram")
    plt.legend()
    plt.grid(True)
    plt.show()




