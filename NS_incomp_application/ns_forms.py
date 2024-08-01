import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colors
import dolfin
from dolfin import *

def initialize_history(method, w, W, bcs, get_variational_form, U_inlet, t, dt, nu,write_velocity_func,write_pressure_func):
    """Initialize history variables for higher-order methods like BDF2 and BDF3"""
    u, p = split(w)
    w_1 = Function(W)
    u_1, p_1 = split(w_1)
    w_2 = None
    w_3 = None
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

def solve_steady_navier_stokes(W,nu,bcs,results_dir):

    filename_velocity = f'{results_dir}/velocity_steady_navier_stokes.xdmf'
    filename_pressure = f'{results_dir}/pressure_steady_navier_stokes.xdmf'
    f_velocity = XDMFFile(filename_velocity)
    f_pressure = XDMFFile(filename_pressure)
    v, q = TestFunctions(W)
    delta_up = TrialFunction(W) # Trial function in the mixed space 
    w = Function(W)
    u, p = split(w)
    f = Constant((0., 0.))
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


def solve_unsteady_navier_stokes(W, nu, bcs, T, dt, time_integration_method, theta=0.5, ds_circle=None, n1=None, U_inlet=None, write_velocity=True, write_pressure=False, flag_drag_lift=False, results_dir="results/"):
    """Solve unsteady Navier-Stokes and write results to file"""

    # Current and old solution
    w = Function(W)
    u, p = split(w)


     


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
        return [], [], [], []


    if flag_drag_lift:
        u_t, c_ds, c_ls, ts, p_diffs = initialize_drag_lift(w, nu, ds_circle, t, n1)

        calculate_drag_lift_func = lambda: calculate_drag_lift(nu, u_t, p, n1, ds_circle, ts, c_ds, c_ls, p_diffs, t)
        save_drag_lift_func = lambda: save_drag_lift(p_diffs, c_ds, c_ls, ts, results_dir)
    else:
        calculate_drag_lift_func = noop_drag_lift
        save_drag_lift_func = write_nothing



    # Define variational forms
    v, q = TestFunctions(W)

    w_1, w_2, w_3, t = initialize_history(time_integration_method, w, W, bcs, get_variational_form, U_inlet, t, dt, nu,write_velocity_func,write_pressure_func) if time_integration_method in ["bdf2", "bdf3"] else (Function(W), None, None, t)
    
    
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
                                            
        c_ds, c_ls, p_diffs, ts = calculate_drag_lift_func()
        write_velocity_func(u, t)
        write_pressure_func(p, t)


        

        print(f"Time step {t} completed")

    save_drag_lift(p_diffs, c_ds, c_ls, ts,results_dir)
    
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



  
def initialize_drag_lift(w, nu, ds_circle, t, n1):
        # Compute reference quantities
   
    u,p = w.split()
    
    u_t = inner(as_vector((n1[1], -n1[0])), u)
    drag = assemble(2/0.1*(nu*inner(grad(u_t), n1)*n1[1] - p*n1[0])*ds_circle)
    lift = assemble(-2/0.1*(nu*inner(grad(u_t), n1)*n1[0] + p*n1[1])*ds_circle)
    p_diffs = [p(0.15,0.2)-p(0.25,0.2)]
    #p_diffs = [p(Point(0.15, 0.2)) - p(Point(0.25, 0.2))]
    
    c_ds = [drag]
    c_ls = [lift]
    ts = [t]
    return u_t, c_ds, c_ls, ts, p_diffs

def calculate_drag_lift(nu,u_t,p,n1,ds_circle,ts,c_ds, c_ls, p_diffs,t):

    c_ds.append(assemble(
        2/0.1*(nu*inner(grad(u_t), n1)*n1[1] - p*n1[0])*ds_circle))
    c_ls.append(assemble(
        -2/0.1*(nu*inner(grad(u_t), n1)*n1[0] + p*n1[1])*ds_circle))

 
    p_diffs.append(p(0.15,0.2)-p(0.25,0.2))
   

    ts.append(t)
    
    return c_ds, c_ls, p_diffs, ts

def save_drag_lift(p_diffs,c_ds,c_ls,ts, results_dir="results/"):
    np.savez(results_dir+"drag_lift_results", dp=np.array(p_diffs),
        CD=np.array(c_ds), CL=np.array(c_ls),
        t=ts)



