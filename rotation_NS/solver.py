from dolfin import *
import numpy as np
import os
from pathlib import Path

def write_nothing(*args): pass

class NavierStokesProblem:
    def __init__(self, config):
        self.config = config
        self.setup_parameters()
        self.setup_writers()
        self.setup_force_logging()

    def setup_parameters(self):
        self.W = self.config['W']
        self.Q = self.config['Q']
        self.nu = self.config['nu']
        self.bcs = self.config['bcs']
        self.results_dir = self.config.get('results_dir', 'results/')
        self.ds_circle = self.config.get('ds_circle')
        self.n1 = self.config.get('n1')
        self.method = self.config['time_integration']
        self.u0_file = self.config.get('u0_file', "results/velocity.xdmf")
        self.picard = self.config.get('picard', False)
        Path(self.results_dir).mkdir(exist_ok=True)
        
    def setup_writers(self):
        self.writers = {'velocity': write_nothing, 'pressure': write_nothing, 'forces': write_nothing,'vorticity': write_nothing,'energy': write_nothing}
        
        if self.config.get('write_velocity', True):
            velocity_file = XDMFFile(f'{self.results_dir}/velocity_{self.get_prefix()}.xdmf')
            self.writers['velocity'] = lambda u, t: velocity_file.write(u, t)
            
        if self.config.get('write_pressure', False):
            pressure_file = XDMFFile(f'{self.results_dir}/pressure_{self.get_prefix()}.xdmf')
            self.writers['pressure'] = lambda p, t: pressure_file.write(p, t)
            
        if self.config.get('flag_drag_lift', False):
            self.forces = []
            self.c_ds = []
            self.c_ls = []
            self.ts = []
            self.writers['forces'] = self.calculate_forces

        if self.config.get('flag_energy',False):
            def calculate_energy(self,w):
                (u, p) = split(w)
                j = assemble(inner(u, u)*dx)
                return j

            self.writers['energy'] = self.calculate_energy

        if self.config.get('flag_initial_u', False):
            
            u0_xdmf_file = XDMFFile(self.u0_file)
            V = self.W.sub(0).collapse()  # Extract the velocity subspace
            u_initial = Function(V)
            u0_xdmf_file.read_checkpoint(u_initial, "u_out", 0)
            self.w_initial = Function(self.W)
            assign(self.w_initial.sub(0), u_initial)

        if self.config.get('write_vorticity', False):
            vorticity_file = XDMFFile(f'{self.results_dir}/vorticity_{self.get_prefix()}.xdmf')
            def write_vorticity(u, t):
                vortex = curl(u)
                vor = Function(self.Q)
                vor = project(vortex, self.Q) # Extract z-component
                vorticity_file.write(vor, t)
            self.writers['vorticity'] = write_vorticity
    def write_checkpoint(self, u):
        """Write velocity checkpoint to file"""
        if self.config.get('flag_write_checkpoint', False):
            checkpoint_file = XDMFFile(f'{self.results_dir}/velocity_checkpoint_one.xdmf')
            checkpoint_file.write_checkpoint(u, "u_out", 0, XDMFFile.Encoding.HDF5, True)
            checkpoint_file.close()
            print(f'CHECKPOINT WRITTEN')
            
    def setup_force_logging(self):
        """Setup force logging directories and files"""
        if self.config.get('flag_drag_lift', False):
            # Create forces directory inside results
            self.forces_dir = Path(self.results_dir) / 'forces'
            self.forces_dir.mkdir(exist_ok=True)
            
            # Initialize force logging file
            self.force_log_file = self.forces_dir / f'forces_{self.get_prefix()}.csv'
            with open(self.force_log_file, 'w') as f:
                f.write('time,drag,lift\n')
            
            # Initialize lists for collecting force data
            self.forces = []
            self.c_ds = []
            self.c_ls = []
            self.ts = []
   



    def calculate_forces(self, u, p, t):
        """
        Calculate non-dimensionalized drag and lift coefficients
        according to Turek benchmark specifications.
        
        Parameters:
            - Mean inlet velocity Um = 1
            - Cylinder diameter D = 0.1
            - Density ρ = 1
            - Reference values: L_ref = D = 0.1, U_ref = Um = 1
        
        Coefficients:
            CD = 2*FD/(ρ*U_ref^2*D)
            CL = 2*FL/(ρ*U_ref^2*D)
        """
        # Parameters for non-dimensionalization
        D = 1.0  # Cylinder diameter
        rho = 1.0  # Density
        U_ref = 1.0  # Reference velocity (mean inlet velocity)
        
        # Create tangential vector using normal
        n = self.n1
        # Tangential vector (-n_y, n_x)---- change to ny,-nx
        u_t = inner(as_vector((n[1], -n[0])), u)
        
        # Calculate dimensional forces
        F_D = assemble(
            (self.nu*inner(grad(u_t), n)*n[1] - p*n[0])*self.ds_circle
        )
        
        F_L = assemble(
            -(self.nu*inner(grad(u_t), n)*n[0] + p*n[1])*self.ds_circle
        )
        
        # Calculate non-dimensional coefficients
        # Factor 2 comes from Turek benchmark definition
        C_D = 2.0 * F_D / (rho * U_ref**2 * D)
        C_L = 2.0 * F_L / (rho * U_ref**2 * D)
        
        # Store forces
        self.forces.append((t, C_D, C_L))
        self.c_ds.append(C_D)
        self.c_ls.append(C_L)
        self.ts.append(t)
        
        # Log forces to file
        with open(self.force_log_file, 'a') as f:
            f.write(f'{t},{C_D},{C_L}\n')    
    def get_prefix(self):
        return 'steady' if isinstance(self, SteadyNavierStokes) else f'unsteady_{self.method}'

class SteadyNavierStokes(NavierStokesProblem):
    def solve(self):
        if not self.picard:
            w = Function(self.W)
            v, q = TestFunctions(self.W)
            u, p = split(w)
            
            F = (self.nu*inner(grad(u), grad(v))*dx +
                inner(grad(u)*u, v)*dx -
                div(v)*p*dx +
                div(u)*q*dx)
            
            J = derivative(F, w)
            problem = NonlinearVariationalProblem(F, w, self.bcs, J)
            solver = NonlinearVariationalSolver(problem)
            solver.parameters.update({
                "nonlinear_solver": "snes",
                "snes_solver": {
                    "linear_solver": "mumps",
                    "maximum_iterations": 20,
                    "report": True,
                    "error_on_nonconvergence": True
                }
            })
            if self.config.get('continuation',False):
                prev_sol = self.config.get('prev_sol')
                w.vector()[:] = prev_sol.vector()
            else:
                if self.config.get('flag_initial_u', False):
                    w.vector()[:] = self.w_initial.vector()
                
            solver.solve()
            u, p = w.split()
            self.writers['velocity'](u, 0)
            self.writers['pressure'](p, 0)
            self.writers['forces'](u, p, 0)
            self.writers['vorticity'](u, 0)
            print(f'ENERGY: {self.writers['energy'](w)}')
            self.write_checkpoint(u)
            return w
        else:
            w = TrialFunction(self.W)
            v, q = TestFunctions(self.W)
            u, p = split(w)
            w_k = Function(self.W)
            u_k,p_k = split(w_k)
            f = Constant((0., 0.))
            if self.config.get('flag_initial_u', False):
                w_k.vector()[:] = self.w_initial.vector()            

            F = (self.nu*inner(grad(u), grad(v))*dx +
                inner(grad(u)*u_k, v)*dx -
                div(v)*p*dx +
                div(u)*q*dx)

            
            rhs = inner(f,v)*dx
            converged = False
            max_iter = 15
            tol = 1e-7
            w = Function(self.W)
            for i in range(max_iter):
                solve(F==rhs,w,self.bcs)
                diff = norm(w.vector()-w_k.vector(), 'L2')/norm(w,'L2')
                                # Check for convergence
                print(f"Iteration {i+1}: Error = {diff}")
                print(f"Iteration {i+1}: L2 norm of solution = {norm(w, 'L2')}, L2 norm of difference = {norm(w.vector() - w_k.vector(), 'L2')}")
                if diff < tol:
                    print("Picard iteration converged.")
                    converged = True
                    break

                # Update the previous solution
                w_k.assign(w)


            if not converged:
                print("Warning: Picard iteration did not fully converge. Switching to Newton.")

            # Switch to Newton's method
            w_newton = Function(self.W)
            w_newton.assign(w)  # Use Picard result as initial guess for Newton
            v, q = TestFunctions(self.W)
            u_newton, p_newton = split(w_newton)
            
            # Weak form for Newton's method
            F_newton = (self.nu * inner(grad(u_newton), grad(v)) * dx +
                        inner(grad(u_newton) * u_newton, v) * dx -
                        div(v) * p_newton * dx +
                        div(u_newton) * q * dx)
            
            J_newton = derivative(F_newton, w_newton)
            problem_newton = NonlinearVariationalProblem(F_newton, w_newton, self.bcs, J_newton)
            solver_newton = NonlinearVariationalSolver(problem_newton)
            solver_newton.parameters.update({
                "nonlinear_solver": "snes",
                "snes_solver": {
                    "linear_solver": "mumps",
                    "maximum_iterations": 20,
                    "report": True,
                    "error_on_nonconvergence": True
                }
            })
            
            solver_newton.solve()
            u, p = w_newton.split()
           
            
            self.writers['velocity'](u, 0)
            self.writers['pressure'](p, 0)
            self.writers['forces'](u, p, 0)
            self.writers['vorticity'](u, 0)
            self.write_checkpoint(u)
            return u, p            



class UnsteadyNavierStokes(NavierStokesProblem):
    def __init__(self, config):
        super().__init__(config)
        self.dt = config['time_step']
        self.T = config['final_time']
        self.method = config['time_integration']
        self.theta = config.get('theta', 0.5)

    def _get_form(self, method, u, u_1, p, u_2=None, u_3=None):
        v, q = TestFunctions(self.W)
        print(f"\nCreating form for {method}")
        dt = self.dt
        nu = self.nu
        
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
            + div(u)*q*dx
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
                + div(u)*q*dx
                )
        return F



    def setup_solver(self, w, w_1, p=None, w_2=None, w_3=None):
        u, p = split(w)
        u_1,p_1 = split(w_1)
        print("Setting up solver with method:", self.time_method)

        if self.time_method == "bdf2":
            u_2,p_2 = split(w_2)
            F = self._get_form(self.time_method, u, u_1, p, u_2)
        if self.time_method == "bdf3":
            u_2,p_2 = split(w_2)
            u_3,p_3 = split(w_3)
            F = self._get_form(self.time_method, u, u_1, p, u_2, u_3)

        if self.time_method == "bdf1":
            F = self._get_form(self.time_method, u, u_1, p)

        J = derivative(F, w)
        problem = NonlinearVariationalProblem(F, w, self.bcs, J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['linear_solver'] = 'mumps'
        return solver

    def initialize_history(self, w):
        t = 0
        w_1 = Function(self.W)
        w_2 = w_3 = None
        u, p = split(w)
        if self.config.get('flag_initial_u', False):
            w_1.vector()[:] = self.w_initial.vector()
            
          
        if self.method in ["bdf2", "bdf3"]:
            w_2 = Function(self.W)
            iterations = 2 if self.method == "bdf2" else 3
            if self.method == "bdf3":
                w_3 = Function(self.W)

            # Setup BDF1 solver once, reuse for all initial steps
            self.time_method = "bdf1"
            solver = self.setup_solver(w, w_1)
            self.time_method = self.method
            for _ in range(iterations):
                t += self.dt
                solver.solve()
                if self.method == "bdf3":
                    w_3.assign(w_2)
                w_2.assign(w_1)
                w_1.assign(w)
                u, p = w.split()
                self.writers['velocity'](u, t)
                self.writers['pressure'](p, t)
                self.writers['forces'](u, p, t)
               

        else:
            self.time_method = self.method
        return w_1, w_2, w_3, t

    def solve(self):
        w = Function(self.W)
        w_1, w_2, w_3, t = self.initialize_history(w)
        v, q = TestFunctions(self.W)

        solver2 = self.setup_solver(w, w_1, w_2=w_2, w_3=w_3)

        u, p = w.split()
        print('SOLVER IS READY')
        
            
        while t < self.T - DOLFIN_EPS:
            t += self.dt
            # U_inlet.t = t   
            solver2.solve()
            if w_3 is not None:
                w_3.assign(w_2)
            if w_2 is not None:
                w_2.assign(w_1)
            w_1.assign(w)
            
            self.writers['velocity'](u, t)
            self.writers['pressure'](p, t)
            print(f"Time step {t} completed")  
            self.writers['forces'](u, p, t)
            self.writers['vorticity'](u, t)  
            #return w.split()         

        self.write_checkpoint(u)
        
def solve_steady_navier_stokes(**kwargs):
    solver = SteadyNavierStokes(kwargs)
    return solver.solve()

def solve_unsteady_navier_stokes(**kwargs):
    solver = UnsteadyNavierStokes(kwargs)
    return solver.solve()