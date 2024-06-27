
# nstinc : navier stokes incompressible solver


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import dolfin
from dolfin import *



mesh = Mesh()
mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
with XDMFFile("mesh/mesh.xdmf") as infile:
    infile.read(mesh)
    infile.read(mvc, "name_to_read")
cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile("mesh/mf.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)


cmap = colors.ListedColormap(['blue', 'yellow', 'green', 'red'])
idx = np.unique(mf.array())
#print(idx)
labels = dict()


# Define function spaces
# V_element = VectorFunctionSpace(mesh, 'P', 2)
# Q_element = FunctionSpace(mesh, 'P', 1)
# W_element = MixedElement([V_element, Q_element]) # Taylor-Hood
# W = FunctionSpace(mesh, W_element)

V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W_element = MixedElement(V_element, Q_element) # Taylor-Hood
W = FunctionSpace(mesh, W_element)


# Set parameter values
dt = 0.01
T = 100  
theta = 0.5 # Crank Nicolson


Re = 40.
u_bar = 1.
# u_in = Expression(("1.5*u_bar*4/(0.41*0.41)*x[1]*(0.41 - x[1])", "0."), u_bar=u_bar, degree=2)

u_in = Constant((1., 0.))

rho = 1.0
diam = 1.5
nu = Constant(u_bar*diam*rho/Re) # obtained from the definition of Re = u_bar * diam / nu. In our case diam = 0.1.
mu = rho * nu

f = Constant((0., 0.))
# Define inflow profile


# Define boundary conditions
bcu_inflow = DirichletBC(W.sub(0), u_in, mf, 1)
bcu_walls = DirichletBC(W.sub(0), Constant((1.0, 0)), mf, 3)
bcu_cylinder1 = DirichletBC(W.sub(0), Constant((0, 0)), mf,4)
bcu_cylinder2 = DirichletBC(W.sub(0), Constant((0, 0)), mf,5)
bcu_cylinder3 = DirichletBC(W.sub(0), Constant((0, 0)), mf,6)
bcp_outflow = DirichletBC(W.sub(1), Constant(0), mf, 2)
bcs = [bcu_inflow, bcu_walls, bcu_cylinder1, bcu_cylinder2, bcu_cylinder3]
bcp = [bcp_outflow]






def solve_navier_stokes2(W, nu, bcs):
    """Solve steady Navier-Stokes and return the solution"""
    
    # Define variational forms
    v, q = TestFunctions(W)
    delta_up = TrialFunction(W) # Trial function in the mixed space 
    w = Function(W)
    u, p = split(w)
    # Residual
    F = (   nu*inner(grad(u), grad(v))*dx
        + inner(grad(u)*u, v)*dx
        - div(v)*p*dx
        + div(u)*q*dx
        - inner(f, v)*dx
        )
    # Jacobian
    J = derivative(F, w, delta_up)

    snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": True}}
    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters.update(snes_solver_parameters)
    with pipes() as (out, err):
        solver.solve()
    print(out.read())

    return w



def solve_unsteady_navier_stokes_theta(W, nu, bcs, T, dt, theta):
    """Solver unsteady Navier-Stokes and write results
    to file"""

    # Current and old solution
    w = Function(W)
    u, p = split(w)

    w_old = Function(W)
    u_old, p_old = split(w_old)

    # Define variational forms
    v, q = TestFunctions(W)
    F = ( Constant(1/dt)*dot(u - u_old, v)
          + Constant(theta)*nu*inner(grad(u), grad(v))
          + Constant(theta)*dot(dot(grad(u), u), v)
          + Constant(1-theta)*nu*inner(grad(u_old), grad(v))
          + Constant(1-theta)*dot(dot(grad(u_old), u_old), v)
          - Constant(theta)*p*div(v)
          - Constant(1-theta)*p_old*div(v)
          - q*div(u)
        )*dx

    # F = ( Constant(1/dt)*dot(u - u_old, v)
    #       + Constant(theta)*nu*inner(grad(u), grad(v))
    #       + Constant(theta)*dot(dot(grad(u), u), v)
    #       + Constant(1-theta)*nu*inner(grad(u), grad(v))
    #       + Constant(1-theta)*dot(dot(grad(u_old), u_old), v)
    #       - p*div(v)
    #       - q*div(u)
    #     )*dx
    # SUPG / PSPG
    # sigma = 2.*mu*sym(grad(u)) - p*Identity(len(u))
    # # Strong formulation:
    # res_strong = rho*dot(u, grad(u)) - div(sigma)

    # res_strong = -nu*laplacian(u) + dot(u, grad(u)) - grad(p) - div
   
    # + inner(
    #             - nu*laplacian(u) + grad(p),
    #             - rho*delta_u*nu*laplacian(v) + delta_p*grad(q)
    #         )*dx
   
   
    # Cinv = Constant(16*Re) # --> 16*Re is rather high, but solver diverges for lower values
    # vnorm = sqrt(dot(u, u))
    # tau_SUPG = Min(h**2/(Cinv*nu), h/(2.*vnorm))
    # F_SUPG = inner(tau_SUPG*res_strong, rho*dot(grad(v),u))*dx2 # Includes PSPG
    
    #F = F + F_SUPG


    J = derivative(F, w)

    # Create solver
    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'

    f = XDMFFile('velocity_unsteady_navier_stokes2.xdmf')
    u, p = w.split()

    # Perform time-stepping
    t = 0
    while t < T:
        w_old.vector()[:] = w.vector()
        solver.solve()
        t += dt
        f.write(u, t)
        print(f"Time step {t} completed")



def solve_unsteady_navier_stokes_bdf3(W, nu, bcs, T, dt, theta):
    """Solver unsteady Navier-Stokes and write results
    to file"""

    # Current and old solution
    w = Function(W)
    u, p = split(w)

    f = Constant((0., 0.))

    w_old = Function(W)
    u_old, p_old = split(w_old)
    
    w_1 = Function(W)
    u_1, p_1 = split(w_1)

    w_2 = Function(W)
    u_2, p_2 = split(w_2)

    w_3 = Function(W)
    u_3, p_3 = split(w_3)

    # Define variational forms
    v, q = TestFunctions(W)
    F1 = (Constant(11.0)*inner(u, v)/Constant(6*dt)*dx 
      - Constant(18.0)*inner(u_1, v)/Constant(6*dt)*dx 
      + Constant(9.0)*inner(u_2, v)/Constant(6*dt)*dx 
      - Constant(2.0)*inner(u_3, v)/Constant(6*dt)*dx 
      + nu*inner(grad(u), grad(v))*dx
      + inner(grad(u)*u, v)*dx
      - div(v)*p*dx
      + div(u)*q*dx
      - inner(f, v)*dx
    )
 
    J1 = derivative(F1, w)


    F2 = (   inner(u, v)/Constant(dt)*dx # Implit Euler discretization
        - inner(u_old, v)/Constant(dt)*dx 
        + nu*inner(grad(u), grad(v))*dx
        + inner(grad(u)*u, v)*dx
        - div(v)*p*dx
        + div(u)*q*dx
        - inner(f, v)*dx
        )

    J2 = derivative(F2, w)

    # Create solver
    problem1 = NonlinearVariationalProblem(F1, w, bcs, J1)
    solver1 = NonlinearVariationalSolver(problem1)
    solver1.parameters['newton_solver']['linear_solver'] = 'mumps'

    problem2 = NonlinearVariationalProblem(F2, w, bcs, J2)
    solver2 = NonlinearVariationalSolver(problem2)
    solver2.parameters['newton_solver']['linear_solver'] = 'mumps'
    
    file1 = XDMFFile('results/velocity_unsteady_navier_stokes2.xdmf')
    file2 = XDMFFile('results/pressure_unsteady_navier_stokes2.xdmf')
    u, p = w.split()

    # Perform time-stepping
    t = 0
    for i in range(3):
            w_old.vector()[:] = w.vector()
            solver2.solve()
            t += dt
            if i == 0:
                w_3.vector()[:] = w.vector()
            if i == 1:
                w_2.vector()[:] = w.vector()
            if i == 2:
                w_1.vector()[:] = w.vector()
            file1.write(u, t)

    while t < T:
        
        
        solver1.solve()
        
        w_3.vector()[:] = w_2.vector()   
        w_2.vector()[:] = w_1.vector() 
        w_1.vector()[:] = w.vector()
        
        t += dt
        file1.write(u, t)
        #file2.write(p, t)
        print(f"Time step {t} completed")
        

      

# w = solve_stokes(W, nu, bcs)
# save_and_plot(w, 'stokes')

solve_unsteady_navier_stokes_bdf3(W, nu, bcs, T, dt, theta)
# save_and_plot(w, 'navier_stokes_picard')
# plt.show()
