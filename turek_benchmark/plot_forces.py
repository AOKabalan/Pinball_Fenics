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

def plot_forces():
    # Read the CSV file with the correct path
    forces = pd.read_csv('study_results_upper_2/forces/forces_unsteady_bdf3.csv')
    
    # Create drag force plot
    fig1 = plt.figure(figsize=(10, 4.5))
    ax1 = fig1.add_subplot(111)
    ax1.plot(forces['time'], forces['drag'], 'bo-', label='Drag',
             markersize=0.1, linewidth=0.1, markerfacecolor='none')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Drag Force')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black',
               handletextpad=0.4, handlelength=1.5)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_box_aspect(0.4)
    plt.savefig('drag_force_plot.pdf', bbox_inches='tight', pad_inches=0.02)
    
    # Create lift force plot
    fig2 = plt.figure(figsize=(10, 4.5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(forces['time'], forces['lift'], 'rs-', label='Lift',
             markersize=0.1, linewidth=0.1, markerfacecolor='none')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Lift Force')
    ax2.legend(frameon=True, fancybox=False, edgecolor='black',
               handletextpad=0.4, handlelength=1.5)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_box_aspect(0.4)
    plt.savefig('lift_force_plot.pdf', bbox_inches='tight', pad_inches=0.02)
    
    plt.show()

# Generate the plots
print("Generating forces plots...")
plot_forces()