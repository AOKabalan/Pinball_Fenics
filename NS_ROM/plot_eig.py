import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# Set up publication-quality plot formatting
plt.style.use('default')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('axes', labelsize=9)
rc('axes', titlesize=9)
rc('xtick', labelsize=8)
rc('ytick', labelsize=8)
rc('legend', fontsize=8)

# Set general plot parameters
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['figure.dpi'] = 300

def plot_eigenvalues(eigs, title, filename):
    """Plot eigenvalues with publication-quality formatting."""
    fig = plt.figure(figsize=(5, 4.5))
    
    # Create axis with specific margins
    ax = fig.add_subplot(111)
    
    # Create N array (1-based indexing for eigenvalues)
    N = np.arange(1, len(eigs) + 1)
    
    # Plot eigenvalues
    ax.semilogy(N, eigs, 'bo-', markersize=5, linewidth=1.2, markerfacecolor='none')
    
    ax.set_xlabel('$N$')
    ax.set_ylabel('Eigenvalue magnitude')
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_box_aspect(0.85)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
    plt.show()

def load_and_plot_all():
    # Load eigenvalues from files
    with open('runs/run_1_Pinball_20241209_162049_c7216192/post_processing/eigs_u.txt', 'r') as f:
        eigs_u = eval(f.read())
    with open('runs/run_1_Pinball_20241209_162049_c7216192/post_processing/eigs_p.txt', 'r') as f:
        eigs_p = eval(f.read())
    with open('runs/run_1_Pinball_20241209_162049_c7216192/post_processing/eigs_s.txt', 'r') as f:
        eigs_s = eval(f.read())
    
    # Plot each set of eigenvalues
    print("Plotting eigenvalues for u...")
    plot_eigenvalues(eigs_u, 'Eigenvalues for $u$', 'eigs_u_plot.pdf')
    
    print("\nPlotting eigenvalues for p...")
    plot_eigenvalues(eigs_p, 'Eigenvalues for $p$', 'eigs_p_plot.pdf')
    
    print("\nPlotting eigenvalues for s...")
    plot_eigenvalues(eigs_s, 'Eigenvalues for $s$', 'eigs_s_plot.pdf')

if __name__ == "__main__":
    load_and_plot_all()
