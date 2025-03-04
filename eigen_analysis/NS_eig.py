from dolfin import *
from slepc4py import SLEPc
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt
import json
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

def solve_base_flow(Re, mesh,mf, degree=2):
    """
    Solve for the base flow at given Reynolds number
    """
    # Function Spaces
    # V = VectorFunctionSpace(mesh, "P", degree)
    # Q = FunctionSpace(mesh, "P", degree-1)
    # W = V * Q  # Mixed space
    V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W_element = MixedElement(V_element, Q_element)
    W = FunctionSpace(mesh, W_element)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    # Trial and test functions
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    
    # Functions for solutions
    w = Function(W)
    w0 = Function(W)  # Previous iteration
    
    # Split mixed functions
    u_, p_ = split(w)
    u0, p0 = split(w0)
    
    # Viscosity
    nu = Constant(1.0/Re)
    
    # Boundary conditions
    inflow = Expression(('1.0', '0.0'), degree=2)
    noslip = Constant((0.0, 0.0))
    
    # Define boundary conditions
    bc_inflow = DirichletBC(W.sub(0), inflow, mf, 1 )
    bc_walls = DirichletBC(W.sub(0), inflow, mf, 3)
    bc_cyl_front = DirichletBC(W.sub(0), noslip, mf, 4)
    bc_cyl_bottom = DirichletBC(W.sub(0), noslip, mf, 5)
    bc_cyl_top = DirichletBC(W.sub(0), noslip, mf, 6)

    
    bcs = [bc_inflow, bc_walls, bc_cyl_front, bc_cyl_bottom, bc_cyl_top]
    
    # Weak form
    F = (nu*inner(grad(u_), grad(v))*dx 
         + inner(grad(u_)*u_, v)*dx
         - p_*div(v)*dx 
         + div(u_)*q*dx)
    
    J = derivative(F, w)
    problem = NonlinearVariationalProblem(F, w, bcs, J)
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
    solver.solve()
    u, p = w.split()
    velocity_file = XDMFFile(f'velocity_steady_Re_{Re}.xdmf')
    velocity_file.write(u,0)
    return w

def setup_eigenvalue_problem_explicitly(base_flow, Re, mesh, sigma=-1e-2, degree=2):
    """
    Setup linearized eigenvalue problem around base flow
    """
    # Function spaces
    V = VectorFunctionSpace(mesh, "P", degree)
    V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W_element = MixedElement(V_element, Q_element)
    W = FunctionSpace(mesh, W_element)
    
    # Extract base flow components
    u_base, p_base =split(base_flow)

    # Trial and test functions for perturbation
    u_trial, p_trial = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Viscosity
    nu = Constant(1.0/Re)
    
    # Linearized operator

    a = (nu*inner(grad(u_trial), grad(v))*dx
     + inner(dot(u_base, grad(u_trial)), v)*dx
     + inner(dot(u_trial, grad(u_base)), v)*dx
     - p_trial*div(v)*dx
     + div(u_trial)*q*dx)
    
    # Mass matrix

    m = inner(u_trial, v)*dx
    
    Assemble matrices
    A = PETScMatrix()
    M = PETScMatrix()
    assemble(a, tensor=A)
    assemble(m, tensor=M)
  

    inflow = Expression(('1.0', '0.0'), degree=2)
    noslip = Constant((0.0, 0.0))

    bc_inflow = DirichletBC(W.sub(0), inflow, mf, 1 )
    bc_walls = DirichletBC(W.sub(0), inflow, mf, 3)
    bc_cyl_front = DirichletBC(W.sub(0), noslip, mf, 4)
    bc_cyl_bottom = DirichletBC(W.sub(0), noslip, mf, 5)
    bc_cyl_top = DirichletBC(W.sub(0), noslip, mf, 6)


    bcs = [bc_inflow, bc_walls, bc_cyl_front, bc_cyl_bottom, bc_cyl_top]
    
    [bc.apply(A) for bc in bcs]
    [bc.apply(M) for bc in bcs]

    return A, M

def setup_eigenvalue_problem(base_flow, Re, mesh, degree=2):
    
     # Function spaces
    V = VectorFunctionSpace(mesh, "P", degree)
    V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W_element = MixedElement(V_element, Q_element)
    W = FunctionSpace(mesh, W_element)
    dup_e = TrialFunction(W)
    vq_e = TestFunction(W)
    up_e = Function(W)

    (du_e, dp_e) = split(dup_e)
    (v_e, q_e) = split(vq_e)
    (u_e, p_e) = split(up_e)
    # Extract base flow components
    y_hf, p_hf = split(base_flow)
    
    # Trial and test functions for perturbation
    # u_trial, p_trial = TrialFunctions(W)
    # v_e, q_e = TestFunctions(W)
    
    # Viscosity
    nu = Constant(1.0/Re)
    
    G_form = nu*(inner(grad(y_hf), grad(v_e))*dx) - q_e*div(y_hf)*dx - p_hf*div(v_e)*dx + inner(grad(y_hf)*y_hf, v_e)*dx	
    G_form_der = derivative(G_form, base_flow, dup_e)	
    G = PETScMatrix()	
    assemble(G_form_der, tensor=G)

    B_form = inner(grad(du_e), grad(v_e))*dx + inner(dp_e, q_e)*dx
    B = PETScMatrix()
    assemble(B_form, tensor=B)


    

    # Boundary conditions
    inflow = Expression(('1.0', '0.0'), degree=2)
    noslip = Constant((0.0, 0.0))

    # Define boundary conditions
    bc_inflow = DirichletBC(W.sub(0), inflow, mf, 1 )
    bc_walls = DirichletBC(W.sub(0), inflow, mf, 3)
    bc_cyl_front = DirichletBC(W.sub(0), noslip, mf, 4)
    bc_cyl_bottom = DirichletBC(W.sub(0), noslip, mf, 5)
    bc_cyl_top = DirichletBC(W.sub(0), noslip, mf, 6)

    # bc_outflow = DirichletBC(W.sub(1), Constant(0), "near(x[0], 5)")
    bcs = [bc_inflow, bc_walls, bc_cyl_front, bc_cyl_bottom, bc_cyl_top]
    
    [bc.apply(G) for bc in bcs]
    [bc.apply(B) for bc in bcs]

    return G, B


def compute_eigenvalues(A, M, num_eigenvalues=10):

    eigensolver = SLEPcEigenSolver(A, M)

    eigensolver.parameters['problem_type'] = 'gen_non_hermitian'
    eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
    eigensolver.parameters['spectral_shift'] = 1e-4
    eigensolver.parameters['spectrum'] = 'target real'
    eigensolver.solve(num_eigenvalues)
    

    # Solve the eigenvalue problem
    eigensolver.solve(num_eigenvalues)
    
    # Get number of converged eigenvalues
    nconv = eigensolver.get_number_converged()
    print(f"Number of converged eigenvalues: {nconv}")
    
    # Extract converged eigenvalues
    eigenvalues = []
    for i in range(nconv):
        lr, lc = eigensolver.get_eigenvalue(i)
        eigenvalues.append((lr, lc))

    return eigenvalues


        

# Main execution
if __name__ == "__main__":
    # Create mesh
    # mesh = RectangleMesh(Point(0.0, 0.0), Point(5.0, 1.0), 100, 20)
    config = ConfigHandler.load_config('inputs2.json')
    mesh, mf, cf = MeshHandler.load_mesh(config['mesh_file'], 
                                    config['mesh_function_file'])
    # Reynolds numbers to analyze
    reynolds_numbers = [10,15,20]
    all_eigenvalues = {}
    
    # Compute eigenvalues for each Reynolds number
    for Re in reynolds_numbers:
        print(f"Computing for Re = {Re}")
        
        # Get base flow
        base_flow = solve_base_flow(Re, mesh,mf)
        
        # Setup and solve eigenvalue problem
        A, M = setup_eigenvalue_problem(base_flow, Re, mesh)
        eigenvalues = compute_eigenvalues(A, M)
        all_eigenvalues[Re] = eigenvalues
        
        # Print leading eigenvalues
        print(f"Leading eigenvalues for Re = {Re}:")
        for i, eig in enumerate(eigenvalues[:5]):
            print(f"  λ_{i} = {eig.real:10.4f} + {eig.imag:10.4f}j")
    
    # Plot eigenvalue spectrum
    plt.figure(figsize=(10, 8))
    for Re in reynolds_numbers:
        eigs = all_eigenvalues[Re]
        real_parts = [e.real for e in eigs]
        imag_parts = [e.imag for e in eigs]
        plt.scatter(real_parts, imag_parts, label=f'Re = {Re}')
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Re(λ)')
    plt.ylabel('Im(λ)')
    plt.title('Eigenvalue Spectrum for Different Reynolds Numbers')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Save results
    np.save('eigenvalues.npy', all_eigenvalues)