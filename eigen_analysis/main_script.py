"""
Navier-Stokes simulation for stability analysis with FEniCS legacy
"""

from dolfin import *
from slepc4py import SLEPc
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# Import the custom SNES utilities
from snes_utils import NonlinearVariationalProblem, NonlinearVariationalSolver # This is not necessary, it just provides more high level control over PETSc solver

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

class NS_simulation():
    def __init__(self):
        # Load mesh using MeshHandler
        self.mesh, self.mf, self.cf = MeshHandler.load_mesh("mesh/mesh.xdmf", "mesh/mf.xdmf")
        
        # Function spaces in FEniCS using mixed element
        V_element = VectorElement("Lagrange", self.mesh.ufl_cell(), 2)
        Q_element = FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
        W_element = MixedElement(V_element, Q_element)
        self.V = FunctionSpace(self.mesh, W_element)
        
        self.Re = Constant(0.0)
        self.sol = Function(self.V)

        self.sol_test = TestFunction(self.V)
        self.pvd = File("output2.pvd")  # Different VTK file syntax
        
        # FEniCS solver parameters
        self.solver_params = {
                                "mat_type": "aij",
                                'snes_type': 'newtonls',    
                                "snes_monitor": None,
                                "snes_linesearch_type": "basic",
                                "snes_max_it": 100,
                                "snes_atol": 1.0e-9,
                                "snes_rtol": 0.0,
                                "ksp_type": "preonly",
                                "pc_type": "lu",
                                "pc_factor_mat_solver_type": "mumps"
                            }

        # Get Boundary conditions
        self.boundary_conditions()
        
    def residual(self, sol, sol_test):
        f = Constant((0,0))
        (u,p) = split(sol)
        (v,q) = split(sol_test)

        F = -(
              (1/self.Re)*inner(grad(u), grad(v))*dx
            + inner(grad(u)*u, v)*dx
            - div(v)*p*dx
            - inner(f,v)*dx
            + q*div(u)*dx
            )
        return F

    def boundary_conditions(self):
        # Constants
        self.amp = Constant(0.0)  # Amplitude of the boundary condition
        inflow = Constant((1.0, 0.0))  # Inflow velocity
        noslip = Constant((0.0, 0.0))  # No-slip velocity
        
        # Define top and bottom boundary conditions using Expressions
        # Boundary ID 5 - Cylinder bottom
        bottom_expr = Expression(('-amp*sin(atan2(x[1]-cy, x[0]-cx))',
                                 'amp*cos(atan2(x[1]-cy, x[0]-cx))'),
                                 degree=2,
                                 amp=float(self.amp),
                                 cx=0.0,
                                 cy=-0.75)
        
        # Boundary ID 6 - Cylinder top
        top_expr = Expression(('amp*sin(atan2(x[1]-cy, x[0]-cx))',
                              '-amp*cos(atan2(x[1]-cy, x[0]-cx))'),
                              degree=2,
                              amp=float(self.amp),
                              cx=0.0,
                              cy=0.75)
        
        # Set up boundary conditions using mesh function
        self.bcs = [
            DirichletBC(self.V.sub(0), inflow, self.mf, 1),    # Inflow
            DirichletBC(self.V.sub(0), inflow, self.mf, 3),    # Far-field walls
            DirichletBC(self.V.sub(0), noslip, self.mf, 4),    # Cylinder parts
            DirichletBC(self.V.sub(0), bottom_expr, self.mf, 5),  # Bottom boundary
            DirichletBC(self.V.sub(0), top_expr, self.mf, 6)     # Top boundary
        ]

    def solve(self, Re):
        """
        Solve the Navier-Stokes equations for the given Reynolds number.
        
        Args:
            Re: Reynolds number for the simulation
        """
        # Set the Reynolds number
        self.Re.assign(Constant(Re))
        
        # Update the boundary expressions with current amplitude
        for bc in self.bcs:
            if hasattr(bc.value(), 'amp'):
                bc.value().amp = float(self.amp)
        
        # Define the nonlinear problem
        F = self.residual(self.sol, self.sol_test)
        J = derivative(F, self.sol)
        
        # Setup and solve the nonlinear problem
        problem = NonlinearVariationalProblem(F, self.sol, self.bcs, J)
        solver = NonlinearVariationalSolver(problem, solver_parameters=self.solver_params)
        solver.solve()
    # def solve(self,Re):
    #     """
    #     Solve the Navier-Stokes equations for the base flow
        
    #     Returns:
    #         sol: Solution containing velocity and pressure
    #     """
    #     self.Re.assign(Constant(Re))
    #     # Weak form
    #     F = self.residual(self.sol, self.sol_test)
        
    #     # Setup Jacobian and solver
    #     J = derivative(F, self.sol)
    #     problem = NonlinearVariationalProblem(F, self.sol, self.bcs, J)
    #     solver = NonlinearVariationalSolver(problem)
    #     solver.parameters.update({
    #         "nonlinear_solver": "snes",
    #         "snes_solver": {
    #             "linear_solver": "mumps",
    #             "maximum_iterations": 20,
    #             "report": True,
    #             "error_on_nonconvergence": True
    #         }
    #     })
        
    #     # Solve the problem
    #     solver.solve()

        print(f'Solved for Re = {float(Re)}, Amplitude = {float(self.amp)}')

    def save_solution(self):
        u, p = self.sol.split()
        u.rename("velocity", "velocity")
        p.rename("pressure", "pressure")
        self.pvd << u  # Changed Firedrake's write() to FEniCS's << operator
    

            
    def stability(self, num_eigenvalues=10):
        trial = TrialFunction(self.V)
        (u_t,p_t) = split(trial)
        (v,q) = split(self.sol_test)
        F = self.residual(self.sol, self.sol_test)
        J = derivative(F, self.sol, trial)

        # FEniCS way of assembling matrices
        A = assemble(J)
        for bc in self.bcs:
            bc.apply(A)

        massform = inner(u_t, v)*dx     
       
        M = assemble(massform)
        for bc in self.bcs:
            bc.apply(M)  # Apply boundary conditions to mass matrix
            bc.zero(M)   # Zero out rows and columns for Dirichlet BCs

        # Convert FEniCS matrices to PETSc
        A_petsc = as_backend_type(A).mat()
        M_petsc = as_backend_type(M).mat()

        # Solver options
        opts = PETSc.Options()
        
        parameters = {
             "eps_monitor_conv" : None,
             "eps_converged_reason": None,
             "eps_type": "krylovschur",
             "eps_nev" : num_eigenvalues,
             "eps_max_it": 200,
             "eps_tol" : 1e-10,
             "eps_which": "smallest_magnitude",
             "st_type": "sinvert",
             "st_ksp_type": "preonly",
             "st_pc_type": "lu",
             "st_pc_factor_mat_solver_type": "mumps",
             "st_ksp_max_it": 10,
             }
        
        for k in parameters:
            opts[k] = parameters[k]
        
        # Solve the eigenvalue problem using SLEPc
        eps = SLEPc.EPS()
        eps.create(comm=MPI.comm_world)
        eps.setOperators(A_petsc, M_petsc)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setFromOptions()
        eps.setTarget(0)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL) 
        print("### Solving eigenvalue problem ###")
        eps.solve()
        
        # Save eigenvalues
        eigenvalues = []
        self.eigenfunctions_R = []
        self.eigenfunctions_I = []
        
        # Create vectors for eigenvectors
        ev_re = A_petsc.createVecRight()
        ev_im = A_petsc.createVecRight()
        
        for i in range(eps.getConverged()):
            eigenvalues.append(eps.getEigenvalue(i))
            eps.getEigenpair(i, ev_re, ev_im)
            
            # Save the real part - convert PETSc Vec to FEniCS Function
            eigenfunction_R = Function(self.V, name="Eigenfunction_R")
            eigenfunction_I = Function(self.V, name="Eigenfunction_I")
            
            # FEniCS way to set Function from PETSc Vec
            eigenfunction_R.vector().set_local(ev_re.getArray())
            eigenfunction_I.vector().set_local(ev_im.getArray())
            
            self.eigenfunctions_R.append(eigenfunction_R)
            self.eigenfunctions_I.append(eigenfunction_I)
        
        print(norm(self.eigenfunctions_R[1].split()[0]))
        print(norm(self.eigenfunctions_I[1].split()[0]))
            
        return eigenvalues
    
    def compute_stability(self):
        # Critical = 45.4
        Re_array = np.linspace(10, 80, 100)
        Re_list = []
        eig_R_list = []
        eig_I_list = []
        for Re in Re_array:
            
            print("### Re = %f ###\n" % Re)
            
            self.solve(Re)
            eigenvalues = self.stability()
            eig_R = [eig.real for eig in eigenvalues]
            eig_I = [eig.imag for eig in eigenvalues]
            
            Re_list += [Re for i in range(len(eigenvalues))]
            eig_R_list += eig_R
            eig_I_list += eig_I
        
            eig_R_save = np.vstack((np.array(Re_list), np.array(eig_R_list))).transpose()
            np.savetxt("Stability/eig_real.csv", eig_R_save, delimiter=',')
            eig_I_save = np.vstack((np.array(Re_list), np.array(eig_I_list))).transpose()
            np.savetxt("Stability/eig_imag.csv", eig_I_save, delimiter=',')
            
            self.plot_stability()
            
    def plot_stability(self):
        eig_R = np.genfromtxt('Stability/eig_real.csv', delimiter=',')
        eig_I = np.genfromtxt('Stability/eig_imag.csv', delimiter=',')
        
        plt.close("all")
        fig = plt.figure(figsize=(8, 2.5))
        plt.subplot(1,2,1)
        plt.plot(eig_R[:,0], eig_R[:,1], '.')
        plt.axhline(0, xmin=0, xmax=1, color="r", linewidth=0.5)
        plt.title("Real")
        plt.xlabel("Re")
        
        plt.subplot(1,2,2)
        plt.plot(eig_I[:,0], eig_I[:,1], '.')
        plt.ylim(bottom=0)
        plt.title("Imaginary")
        plt.xlabel("Re")
        plt.savefig("stability.png", dpi=600, bbox_inches="tight")
 
    def plot_stability3(self):
        eig_R = np.genfromtxt('Stability/eig_real.csv', delimiter=',')
        eig_I = np.genfromtxt('Stability/eig_imag.csv', delimiter=',')
        
        # Find critical points
        unique_re = np.unique(eig_R[:,0])
        critical_points = []
        prev_eigs = None
        for re in unique_re:
            mask = (eig_R[:,0] == re)
            curr_eigs = np.sort(eig_R[mask,1])[::-1]
            
            if prev_eigs is not None:
                for i in range(len(curr_eigs)):
                    if i < len(prev_eigs):
                        if (prev_eigs[i] < 0 and curr_eigs[i] > 0):
                            re_crit = prev_re + (re - prev_re) * (0 - prev_eigs[i])/(curr_eigs[i] - prev_eigs[i])
                            critical_points.append(re_crit)
            prev_eigs = curr_eigs
            prev_re = re

        # Set publication quality params
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9,
            'figure.figsize': (10, 3.5)
        })
        
        fig = plt.figure()
        
        # Real eigenvalues subplot
        ax1 = plt.subplot(1,2,1)
        ax1.scatter(eig_R[:,0], eig_R[:,1], s=20, alpha=0.6, color='navy')
        ax1.axhline(0, xmin=0, xmax=1, color="red", linewidth=1, linestyle='--', alpha=0.5)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        for i, re_crit in enumerate(critical_points):
            ax1.axvline(re_crit, color="green", linestyle='--', alpha=0.7,
                    label=f'Re$_{{cr{i+1}}}$ = {re_crit:.2f}')
        
        ax1.set_title("Real Eigenvalues", pad=10)
        ax1.set_xlabel("Reynolds Number, Re")
        ax1.set_ylabel(r"Re($\lambda$)")
        ax1.legend(frameon=True, facecolor='white', edgecolor='none', 
                bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Imaginary eigenvalues subplot
        ax2 = plt.subplot(1,2,2)
        ax2.scatter(eig_I[:,0], eig_I[:,1], s=20, alpha=0.6, color='navy')
        ax2.set_ylim(bottom=0)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_title("Imaginary Eigenvalues", pad=10)
        ax2.set_xlabel("Reynolds Number, Re")
        ax2.set_ylabel(r"Im($\lambda$)")
        
        plt.tight_layout()
        plt.savefig("stability.png", dpi=600, bbox_inches="tight")

    def plot_complex_plane(self):
        eig_R = np.genfromtxt('Stability/eig_real.csv', delimiter=',')
        eig_I = np.genfromtxt('Stability/eig_imag.csv', delimiter=',')

        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9
        })

        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()

        # Create colormap based on Reynolds number
        unique_re = np.unique(eig_R[:,0])
        norm = plt.Normalize(unique_re.min(), unique_re.max())
        cmap = plt.cm.viridis

        # Plot eigenvalues on complex plane
        scatter = ax.scatter(eig_R[:,1], eig_I[:,1], c=eig_R[:,0], 
                            cmap=cmap, s=20, alpha=0.6)

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Reynolds Number')

        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel(r"Re($\lambda$)")
        ax.set_ylabel(r"Im($\lambda$)")
        ax.set_title("Eigenvalue Spectrum in Complex Plane")

        plt.tight_layout()
        plt.savefig("complex_plane.png", dpi=600, bbox_inches="tight")

        
    def plot_eigenvalues(self, Re, num_eigenvalues=10):
        """Plot eigenvalues in complex plane with publication quality."""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 14,
            'text.usetex': True,
            'axes.labelsize': 16,
            'axes.titlesize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'figure.figsize': (5, 4)
        })
        
        self.solve(Re)
        eigenvalues = self.stability(num_eigenvalues)[:num_eigenvalues]
        
        fig, ax = plt.subplots()
        
        # Plot eigenvalues
        scatter = ax.scatter([ev.real for ev in eigenvalues],
                            [ev.imag for ev in eigenvalues],
                            c='navy',
                            s=60,
                            alpha=0.8,
                            zorder=3)
        
        # Customize grid and spines
        ax.grid(True, linestyle=':', alpha=0.4, zorder=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        ax.set_xlabel('$\\Re(\\lambda)$')
        ax.set_ylabel('$\\Im(\\lambda)$')
        ax.set_title(f'Re = {Re}')
        
        # Show axes with refined styling
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5, zorder=2)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5, zorder=2)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
        
        plt.tight_layout()
        plt.savefig(f'second_eigenvalues_Re{Re}_amp{float(self.amp)}.pdf', format='pdf', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    # Create simulation object
    NS = NS_simulation()
    
    # Set amplitude
    NS.amp.assign(Constant(0.0))
    
    # Solve for Reynolds number 100
    NS.solve(100.0)
    
    # Compute and plot stability
    NS.plot_eigenvalues(100.0)
