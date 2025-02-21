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
    # Read both CSV files
    forces_symm = pd.read_csv('study_results_symm/forces/forces_unsteady_bdf3.csv')
    # forces_lower = pd.read_csv('study_results_asymm_down/forces/forces_unsteady_bdf3.csv')
    # forces_upper = pd.read_csv('study_results_asymm_up/forces/forces_unsteady_bdf3.csv')
    
    # # Create drag force plot
    # fig1 = plt.figure(figsize=(10, 4.5))
    # ax1 = fig1.add_subplot(111)
    
    # ax1.plot(forces_symm['time'], forces_symm['drag'], 'bo-', 
    #          label='Drag (Symmetric)', markersize=0.1, linewidth=0.1, 
    #          markerfacecolor='none')
    # # ax1.plot(forces_lower['time'], forces_lower['drag'], 'ro-', 
    # #          label='Drag (Lower Asymm)', markersize=0.1, linewidth=0.1, 
    # #          markerfacecolor='none')
    # ax1.plot(forces_upper['time'], forces_upper['drag'], 'ro-', 
    #         label='Drag (Upper Asymm)', markersize=0.1, linewidth=0.1, 
    #         markerfacecolor='none')

    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Drag Force')
    # ax1.legend(frameon=True, fancybox=False, edgecolor='black',
    #           handletextpad=0.4, handlelength=1.5)
    # ax1.grid(True, which="both", ls="-", alpha=0.2)
    # ax1.set_box_aspect(0.4)
    # plt.savefig('drag_force_plot_comparison.pdf', bbox_inches='tight', 
    #             pad_inches=0.02)

    # Create lift force plot
    fig2 = plt.figure(figsize=(10, 4.5))
    ax2 = fig2.add_subplot(111)
    
    ax2.plot(forces_symm['time'], forces_symm['lift'], 'bs-', 
             label='Lift (Symmetric)', markersize=0.1, linewidth=0.1, 
             markerfacecolor='none')
    # ax2.plot(forces_lower['time'], forces_lower['lift'], 'rs-', 
    #          label='Lift (Lower Asymm)', markersize=0.1, linewidth=0.1, 
    #          markerfacecolor='none')
    # ax2.plot(forces_upper['time'], forces_upper['lift'], 'ko-', 
    #     label='Lift(Upper Asymm)', markersize=0.1, linewidth=0.1, 
    #     markerfacecolor='none')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Lift Force')
    ax2.legend(frameon=True, fancybox=False, edgecolor='black',
              handletextpad=0.4, handlelength=1.5)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_box_aspect(0.4)
    plt.savefig('lift_force_plot_comparison_100.pdf', bbox_inches='tight', 
                pad_inches=0.02)
    
    plt.show()

# Generate the plots
print("Generating forces comparison plots...")
plot_forces()
