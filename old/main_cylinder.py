from dolfin import *
from space import build_space
from space import build_space2
from nstinc import solve_unsteady_navier_stokes_bdf3
from nstinc import solve_unsteady_navier_stokes_theta
from nstinc import solve_unsteady_navier_stokes_internet_theta
from nstinc import solve_unsteady_navier_stokes_bdf2
from nstinc import solve_unsteady_navier_stokes_bdf1

"""Solve unsteady Navier-Stokes to resolve
    Karman vortex street and save to file"""

# Problem data



u_in = Expression(("4.0*U*x[1]*(0.41 - x[1])/(0.41*0.41)", "0.0"),
                    degree=2, U=1)
nu = Constant(0.001)
T = 8

    # Discretization parameters
N_circle = 16*2
N_bulk = 64*2
theta = 0.5
dt = 0.05

# Prepare function space, BCs and measure on circle
W, bcs, ds_circle = build_space2(u_in)

# Solve unsteady Navier-Stokes
solve_unsteady_navier_stokes_bdf3(W, nu, bcs, T, dt, theta)
#solve_unsteady_navier_stokes_theta(W, nu, bcs, T, dt, theta)
#solve_unsteady_navier_stokes_internet_theta(W, nu, bcs, T, dt, theta)
#solve_unsteady_navier_stokes_bdf2(W, nu, bcs, T, dt, theta)
#solve_unsteady_navier_stokes_bdf1(W, nu, bcs, T, dt, theta)