# Pinball_Fenics
Solving the Navier Stokes equation using the finite element method on the fluidic pinball problem. 

Step1. Construct .msh file (using gmsh)

Step2. Get mesh.xdmf and mf.xdmf from mesh_func.py using the following syntax for running (python mesh_func.py <path/to/file.msh>)

Step3. Edit Inside nstinc.py the parameters ( Re, T, dt, boundary conditions, etc.) and run with (python nstinc.py)
