"""
FEniCS SNES Utilities - Implements SNES solver to mimic Firedrake behavior

This module provides implementations of Firedrake-like NonlinearVariationalProblem
and NonlinearVariationalSolver classes for FEniCS.
"""

from dolfin import *
from petsc4py import PETSc
import ufl.algorithms
import numpy as np

# Initialize FEniCS and PETSc properly
# try:
#     set_log_level(ERROR)
# except AttributeError:
#     set_log_level(LogLevel.ERROR)

# Set optimization parameters for the form compiler
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

# Initialize PETSc explicitly to prevent deadlocks with parallel execution
SubSystemsManager.init_petsc()

# Disable divergence tolerance which can break deflation code
PETScOptions.set("snes_divergence_tolerance", -1)

# Custom SNES solver implementation to mimic Firedrake's interface
class SNUFLSolver(object):
    def __init__(self, problem, prefix="", solver_parameters={}, **kwargs):
        self.problem = problem
        u = problem.u
        self.u_dvec = as_backend_type(u.vector())
        self.u_pvec = self.u_dvec.vec()

        comm = u.function_space().mesh().mpi_comm()
        self.comm = comm
        snes = PETSc.SNES().create(comm=comm)
        snes.setOptionsPrefix(prefix)

        # Fix the worst defaults in PETSc
        opts = PETSc.Options()
        if "snes_linesearch_type" not in solver_parameters:
            opts[prefix + "snes_linesearch_type"] = "basic"
        if "snes_divergence_tolerance" not in solver_parameters:
            opts[prefix + "snes_divergence_tolerance"] = -1.0
        if "snes_stol" not in solver_parameters:
            opts[prefix + "snes_stol"] = 0.0

        # set the petsc options from the solver_parameters
        for k in solver_parameters:
            opts[prefix + k] = solver_parameters[k]

        (J, F, bcs, P) = (problem.J, problem.F, problem.bcs, problem.P)

        if hasattr(problem, 'problem'):
            self.ass = problem.problem.assembler(J, F, u, bcs)
            if P is not None:
                self.Pass = problem.problem.assembler(P, F, u, bcs)
        else:
            self.ass = SystemAssembler(J, F, bcs)
            if P is not None:
                self.Pass = SystemAssembler(P, F, bcs)

        self.b = self.init_residual()
        snes.setFunction(self.residual, self.b.vec())
        self.A = self.init_jacobian()
        self.P = self.init_preconditioner(self.A)
        snes.setJacobian(self.jacobian, self.A.mat(), self.P.mat())
        # why isn't this done in setJacobian?
        snes.ksp.setOperators(self.A.mat(), self.P.mat())

        # If the user wants to correctly compute dual norms, let them
        if hasattr(problem, 'problem') and "objective" in problem.problem.__class__.__dict__:
            snes.setObjective(self.objective)

        # Workaround for bug in Cray PETSc on ARCHER
        if "pc_type" in solver_parameters:
            snes.ksp.pc.setType(solver_parameters["pc_type"])

        dm = kwargs.get("dm", None)
        if dm is not None:
            snes.setDM(dm)
        snes.setFromOptions()
        self.snes = snes

    def init_jacobian(self):
        A = PETScMatrix(self.comm)
        self.ass.init_global_tensor(A, Form(self.problem.J))
        return A

    def init_residual(self):
        b = as_backend_type(
            Function(self.problem.u.function_space()).vector()
        )
        return b

    def init_preconditioner(self, A):
        if self.problem.P is None:
            return A
        P = PETScMatrix(self.comm)
        self.Pass.init_global_tensor(P, Form(self.problem.P))
        return P

    def update_x(self, x):
        """Given a PETSc Vec x, update the storage of our
           solution function u."""
        x.copy(self.u_pvec)
        self.u_dvec.update_ghost_values()

    def residual(self, snes, x, b):
        self.update_x(x)
        b_wrap = PETScVector(b)
        self.ass.assemble(b_wrap, self.u_dvec)

    def jacobian(self, snes, x, A, P):
        self.update_x(x)
        A_wrap = PETScMatrix(A)
        P_wrap = PETScMatrix(P)
        self.ass.assemble(A_wrap)
        if self.problem.P is not None:
            self.Pass.assemble(P_wrap)

    def objective(self, snes, b):
        F = self.problem.F
        v = ufl.algorithms.extract_arguments(F)[0]
        r = self.problem.problem.objective(self.problem.F, self.problem.u, v)
        return r

    def solve(self):
        # Need a copy for line searches etc. to work correctly.
        x = self.problem.u.copy(deepcopy=True)
        xv = as_backend_type(x.vector()).vec()

        try:
            self.snes.solve(None, xv)
        except:
            import traceback
            traceback.print_exc()
            pass

# NonlinearVariationalProblem class to mimic Firedrake's interface
class NonlinearVariationalProblem(object):
    def __init__(self, F, u, bcs=None, J=None, P=None):
        self.F = F
        self.u = u
        self.bcs = bcs or []
        self.J = J
        self.P = P

# NonlinearVariationalSolver class to mimic Firedrake's interface
class NonlinearVariationalSolver(object):
    def __init__(self, problem, solver_parameters=None):
        self.problem = problem
        self.solver_parameters = solver_parameters or {}
        
    def solve(self):
        # Create a SNES solver
        snes_solver = SNUFLSolver(self.problem, solver_parameters=self.solver_parameters)
        snes_solver.solve()
        
        # Copy solution back
        snes_solver.u_pvec.copy(self.problem.u.vector().vec())
        self.problem.u.vector().update_ghost_values()
