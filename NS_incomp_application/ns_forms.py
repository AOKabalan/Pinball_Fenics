import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colors
import dolfin
from dolfin import *

def compute_vorticity(u):
    return curl(u)


def initialize_history(method, w, W, bcs, get_variational_form, U_inlet, t, dt, nu,write_velocity_func,write_pressure_func,flag_initial_u,u0_file):
    """Initialize history variables for higher-order methods like BDF2 and BDF3"""
    u, p = split(w)
    w_1 = Function(W)
    u_1, p_1 = split(w_1)
    w_2 = None
    w_3 = None 
    
    if flag_initial_u:
        xdmf_file = XDMFFile(u0_file)
        V = W.sub(0).collapse()  # Extract the velocity subspace
        u_initial = Function(V)
        xdmf_file.read_checkpoint(u_initial, "u_out", 0)
        w_initial = Function(W)
        assign(w_initial.sub(0), u_initial)

    v, q = TestFunctions(W)
    if method == "bdf2":
        w_2 = Function(W)
        u_2,_ = split(w_2)
        u_3 = None
        
        
        F = get_variational_form("bdf1", u, u_1, p, v, q, dt, nu,p_1, u_2, u_3)
        problem = NonlinearVariationalProblem(F, w, bcs, derivative(F, w))
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['linear_solver'] = 'mumps'
        u, p = w.split()
        if flag_initial_u:
            w_1.vector()[:] = w_initial.vector()

        for _ in range(2):

            solver.solve()
            t += dt
            U_inlet.t = t
            w_2.vector()[:] = w_1.vector()
            w_1.vector()[:] = w.vector()
            write_velocity_func(u, t)
            write_pressure_func(p, t)
            print(f"Time step {t} completed")

    
    elif method == "bdf3":
        w_2 = Function(W)
        u_2,_ = split(w_2)
        w_3 = Function(W)
        u_3,_ = split(w_3)

        F = get_variational_form("bdf1", u, u_1, p, v, q, dt, nu, p_1, u_2, u_3)
        problem = NonlinearVariationalProblem(F, w, bcs, derivative(F, w))
        solver = NonlinearVariationalSolver(problem)
        u, p = w.split()
        if flag_initial_u:
            w_1.vector()[:] = w_initial.vector()
        for _ in range(3):
            
            
            solver.solve()
            t += dt
            U_inlet.t = t
            w_3.vector()[:] = w_2.vector()
            w_2.vector()[:] = w_1.vector()
            w_1.vector()[:] = w.vector()
            write_velocity_func(u, t)
            write_pressure_func(p, t)
            print(f"Time step {t} completed")




    return w_1, w_2, w_3, t

def solve_steady_navier_stokes(W,Q,nu,bcs,ds_circle,n1,flag_drag_lift,flag_initial_u,u0_file,flag_write_checkpoint,flag_save_vorticity,results_dir):

    filename_velocity = f'{results_dir}/velocity_steady_navier_stokes.xdmf'
    filename_pressure = f'{results_dir}/pressure_steady_navier_stokes.xdmf'
    f_velocity = XDMFFile(filename_velocity)
    f_pressure = XDMFFile(filename_pressure)
    v, q = TestFunctions(W)
    delta_up = TrialFunction(W) # Trial function in the mixed space 
    w = Function(W)
    u, p = split(w)
    f = Constant((0., 0.))
    filename_velocity_checkpoint = f'{results_dir}/velocity_checkpoint_asymmetric.xdmf'
    f_velocity_checkpoint = XDMFFile(filename_velocity_checkpoint)
   
    if flag_initial_u:
        xdmf_file = XDMFFile(u0_file)
        V = W.sub(0).collapse()  # Extract the velocity subspace
        u_initial = Function(V)
        xdmf_file.read_checkpoint(u_initial, "u_out", 0)
        w_initial = Function(W)
        assign(w_initial.sub(0), u_initial) 
    # Residual
    F = (   nu*inner(grad(u), grad(v))*dx
        + inner(grad(u)*u, v)*dx
        - div(v)*p*dx
        + div(u)*q*dx
        - inner(f, v)*dx
        )
    # Jacobian
    J = derivative(F, w, delta_up)
    u, p = w.split()
    if flag_initial_u:
        w.vector()[:] = w_initial.vector()

    snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": True}}
    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters.update(snes_solver_parameters)
    solver.solve()
    f_velocity.write(u)
    f_pressure.write(p)

    if flag_drag_lift:
        u_t, c_ds, c_ls, ts = initialize_drag_lift(w, nu, ds_circle, n1, n_steps=1)
        save_drag_lift(c_ds, c_ls, ts, results_dir)
    
    if flag_write_checkpoint:
        f_velocity_checkpoint.write_checkpoint(u, "u_out", 0, XDMFFile.Encoding.HDF5, False)

    if flag_save_vorticity:
        vortex = curl(u)
        vor = Function(Q)
        vor = project(vortex,Q)
        vorticity_file = XDMFFile(f"{results_dir}/vorticity.xdmf")
        vorticity_file.write(vor)
        vorticity_file.close()


def solve_unsteady_navier_stokes(W, nu, bcs, T, dt, time_integration_method, theta=0.5, ds_circle=None, n1=None, U_inlet=None, write_velocity=True, write_pressure=False, flag_drag_lift=False, flag_initial_u= False,u0_file="results/velocity.xdmf",flag_write_checkpoint=False,flag_save_vorticity=False,results_dir="results/"):
    """Solve unsteady Navier-Stokes and write results to file"""

    # Current and old solution
    w = Function(W)
    u, p = split(w)


    filename_velocity_checkpoint = f'{results_dir}/velocity_checkpoint.xdmf'
    f_velocity_checkpoint = XDMFFile(filename_velocity_checkpoint)
   


    f = Constant((0., 0.))
    t = 0
    os.makedirs(results_dir, exist_ok=True)

    filename_velocity = f'{results_dir}/velocity_unsteady_navier_stokes_{time_integration_method}.xdmf'
    filename_pressure = f'{results_dir}/pressure_unsteady_navier_stokes_{time_integration_method}.xdmf'

    def write_nothing(*args):
        pass

    if write_velocity:
        f_velocity = XDMFFile(filename_velocity)
        write_velocity_func = lambda u, t: f_velocity.write(u, t)
    else:
        write_velocity_func = write_nothing

    if write_pressure:
        f_pressure = XDMFFile(filename_pressure)
        write_pressure_func = lambda p, t: f_pressure.write(p, t)
    else:
        write_pressure_func = write_nothing

    # Define noop function for drag and lift
    def noop_drag_lift(*args):
        return [], [], []


    if flag_drag_lift:
        # Calculate number of time steps
        n_steps = int(T/dt) + 1
        # Initialize arrays with correct size
        u_t, c_ds, c_ls, ts = initialize_drag_lift(w, nu, ds_circle, n1, t, n_steps)
        
        # Keep track of current step
        current_step = 1
        calculate_drag_lift_func = lambda: calculate_drag_lift(
            nu, u_t, p, n1, ds_circle, ts, c_ds, c_ls, t, current_step)
        save_drag_lift_func = lambda: save_drag_lift(c_ds, c_ls, ts, results_dir)
    else:
        calculate_drag_lift_func = noop_drag_lift
        save_drag_lift_func = write_nothing

    # Define variational forms
    v, q = TestFunctions(W)

    w_1, w_2, w_3, t = initialize_history(time_integration_method, w, W, bcs, get_variational_form, U_inlet, t, dt, nu,write_velocity_func,write_pressure_func,flag_initial_u,u0_file) if time_integration_method in ["bdf2", "bdf3"] else (Function(W), None, None, t)
    
    
    u_1, p_1 = split(w_1)

    if time_integration_method == "bdf2":
        
        u_2, p_2 = split(w_2)
        u_3 = None
     

    elif time_integration_method == "bdf3":
        
        u_2, p_2 = split(w_2)
        
        u_3, p_3 = split(w_3)
     
    else:
        u_2 = None
        u_3 = None

    F = get_variational_form(time_integration_method, u, u_1, p, v, q, dt, nu,p_1,u_2, u_3)

    J = derivative(F, w)
    # Create solver
    problem = NonlinearVariationalProblem(F, w, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'

    u, p = w.split()
    
    while t < T - DOLFIN_EPS:
        t += dt
        # U_inlet.t = t   
        solver.solve()
        if time_integration_method == "bdf2":
            w_2.assign(w_1)
            w_1.assign(w)
        elif time_integration_method == "bdf3":
            w_3.assign(w_2)
            w_2.assign(w_1)
            w_1.assign(w)
        else:
            w_1.assign(w)
                                            
        c_ds, c_ls,  ts = calculate_drag_lift_func()
        current_step += 1

        write_velocity_func(u, t)
        write_pressure_func(p, t)


        

        print(f"Time step {t} completed")
    
    save_drag_lift( c_ds, c_ls, ts,results_dir)
    if flag_write_checkpoint:
        f_velocity_checkpoint.write_checkpoint(u, "u_out", 0, XDMFFile.Encoding.HDF5, False)

    if write_velocity:
        f_velocity.close()
    
    if write_pressure:
        f_pressure.close()

def get_variational_form(method, u, u_1, p, v, q, dt, nu,p_1, u_2=0, u_3=0):
    if method == "theta":
        theta =0.5
        F = ( Constant(1/dt)*dot(u - u_1, v)
          + Constant(theta)*nu*inner(grad(u), grad(v))
          + Constant(theta)*dot(dot(grad(u), u), v)
          + Constant(1-theta)*nu*inner(grad(u_1), grad(v))
          + Constant(1-theta)*dot(dot(grad(u_1), u_1), v)
          - Constant(theta)*p*div(v)
          - Constant(1-theta)*p_1*div(v)
          - q*div(u)
        )*dx
        
    elif method == "bdf1":
        F = (   inner(u,      v)/Constant(dt)*dx # Implit Euler discretization
        - inner(u_1, v)/Constant(dt)*dx # Implit Euler discretization
        + nu*inner(grad(u), grad(v))*dx
        + inner(grad(u)*u, v)*dx
        - div(v)*p*dx
        - div(u)*q*dx
        )
    elif method == "bdf2":
        F = (Constant(1/dt)*(Constant(1.5)*inner(u,v) - Constant(2.0)*inner(u_1,v) + Constant(0.5)*inner(u_2, v))*dx 
        + nu*inner(grad(u), grad(v))*dx
        + inner(grad(u)*u, v)*dx
        - div(v)*p*dx
        - div(u)*q*dx
     
    )


    elif method == "bdf3":
        F = (Constant(11.0)*inner(u, v)/Constant(6*dt)*dx 
            - Constant(18.0)*inner(u_1, v)/Constant(6*dt)*dx 
            + Constant(9.0)*inner(u_2, v)/Constant(6*dt)*dx 
            - Constant(2.0)*inner(u_3, v)/Constant(6*dt)*dx 
            + nu*inner(grad(u), grad(v))*dx
            + inner(grad(u)*u, v)*dx
            - div(v)*p*dx
            - div(u)*q*dx
            )
    return F



def initialize_drag_lift(w, nu, ds_circle, n1, t=0, n_steps=1):
    """Initialize drag and lift calculations with pre-allocated arrays
    
    Parameters:
        w: Function
        nu: float
        ds_circle: Measure
        n1: Vector
        t: float, initial time
        n_steps: int, number of time steps (default=1 for steady case)
    """
    u, p = w.split()
    u_t = inner(as_vector((n1[1], -n1[0])), u)
    
    # Pre-allocate arrays
    c_ds = np.zeros(n_steps)
    c_ls = np.zeros(n_steps)
    ts = np.zeros(n_steps)
    
    # Calculate initial values
    c_ds[0] = assemble(2/(3*1.5)*(nu*inner(grad(u_t), n1)*n1[1] - p*n1[0])*ds_circle)
    c_ls[0] = assemble(-2/(3*1.5)*(nu*inner(grad(u_t), n1)*n1[0] + p*n1[1])*ds_circle)
    ts[0] = t
    
    return u_t, c_ds, c_ls, ts


def calculate_drag_lift(nu, u_t, p, n1, ds_circle, ts, c_ds, c_ls, t=0, step_idx=1):
    """Calculate drag and lift coefficients and store at the specified index
    
    Parameters:
        step_idx: int, current time step index in the arrays
    """
    c_ds[step_idx] = assemble(2/(3*1.5)*(nu*inner(grad(u_t), n1)*n1[1] - p*n1[0])*ds_circle)
    c_ls[step_idx] = assemble(-2/(3*1.5)*(nu*inner(grad(u_t), n1)*n1[0] + p*n1[1])*ds_circle)
    ts[step_idx] = t
    
    return c_ds, c_ls, ts



def save_drag_lift(c_ds, c_ls, ts, results_dir="results/"):
    """Save the pre-allocated arrays to file"""
    np.savez(f"{results_dir}/drag_lift_results",
        CD=c_ds, CL=c_ls,
        t=ts)