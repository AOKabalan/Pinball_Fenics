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
        # Set x-axis ticks at intervals of 5
    max_N = len(eigs)
    xticks = np.arange(0, max_N + 5, 5)  # +5 ensures we include the last tick if needed
    xticks = xticks[xticks <= max_N]  # Remove any ticks beyond the data range
    ax.set_xticks(xticks)
    
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
    with open('runs/run_5_PinballDEIM_20241209_140242_f45b5ae6/deim/bdda5f8ade07e9810d7b3060356477105d906763/post_processing/eigs.txt', 'r') as f:
        eigs1 = eval(f.read())
    with open('runs/run_5_PinballDEIM_20241209_140242_f45b5ae6/deim/ea57507283c4b6c25eaab9bb132a1d3a9fbca1f9/post_processing/eigs.txt', 'r') as f:
        eigs2 = eval(f.read())
    
    # Plot each set of eigenvalues
    print("Plotting eigenvalues for term 1...")
    plot_eigenvalues(eigs1, 'Eigenvalues for DEIM', 'eigs_DEIM_1.pdf')
    
    print("\nPlotting eigenvalues for p...")
    plot_eigenvalues(eigs2, 'Eigenvalues for DEIM', 'eigs_DEIM_2.pdf')
    

if __name__ == "__main__":
    load_and_plot_all()
