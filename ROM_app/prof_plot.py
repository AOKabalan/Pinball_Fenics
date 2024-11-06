
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
# [Previous imports and settings remain the same...]

def plot_u_solution():
    df_u = pd.read_csv('FluidicPinball2/error_analysis/error_analysis/solution_u.csv', sep=';')
    
    fig = plt.figure(figsize=(10, 4.5))
  
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.35,
                         left=0.08, right=0.98, top=0.85, bottom=0.15)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot for errors
    ax1.semilogy(df_u['N'], df_u['gmean(error_u)'], 'bo-', label='Geometric Mean', 
                 markersize=5, linewidth=1.2, markerfacecolor='none')
    ax1.semilogy(df_u['N'], df_u['max(error_u)'], 'rs-', label='Maximum', 
                 markersize=5, linewidth=1.2, markerfacecolor='none')
    
    ax1.set_xlabel('$N$')
    ax1.set_ylabel('Error $\\epsilon_u$')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', 
              handletextpad=0.4, handlelength=1.5)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_xticks(df_u['N'])
    ax1.set_ylim(1e-5, 1)  # Set fixed y-axis limits
    ax1.set_box_aspect(0.85)
    
    # Plot for relative errors
    ax2.semilogy(df_u['N'], df_u['gmean(relative_error_u)'], 'bo-', label='Geometric Mean',
                 markersize=5, linewidth=1.2, markerfacecolor='none')
    ax2.semilogy(df_u['N'], df_u['max(relative_error_u)'], 'rs-', label='Maximum',
                 markersize=5, linewidth=1.2, markerfacecolor='none')
    
    ax2.set_xlabel('$N$')
    ax2.set_ylabel('Relative Error $\\epsilon_{u,rel}$')
    ax2.legend(frameon=True, fancybox=False, edgecolor='black', 
              handletextpad=0.4, handlelength=1.5)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_xticks(df_u['N'])
    ax2.set_ylim(1e-6, 1e-1)  # Set fixed y-axis limits
    ax2.set_box_aspect(0.85)
    
    plt.savefig('error_u_plots.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()
    
    # [Convergence rate calculations remain the same...]

def plot_p_solution():
    df_p = pd.read_csv('FluidicPinball2/error_analysis/error_analysis/solution_p.csv', sep=';')
    
    fig = plt.figure(figsize=(10, 4.5))
  
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.35,
                         left=0.08, right=0.98, top=0.85, bottom=0.15)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    ax1.semilogy(df_p['N'], df_p['gmean(error_p)'], 'bo-', label='Geometric Mean',
                 markersize=5, linewidth=1.2, markerfacecolor='none')
    ax1.semilogy(df_p['N'], df_p['max(error_p)'], 'rs-', label='Maximum',
                 markersize=5, linewidth=1.2, markerfacecolor='none')
    
    ax1.set_xlabel('$N$')
    ax1.set_ylabel('Error $\\epsilon_p$')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', 
              handletextpad=0.4, handlelength=1.5)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_xticks(df_p['N'])
    ax1.set_ylim(1e-5, 1)  # Set fixed y-axis limits
    ax1.set_box_aspect(0.85)
    
    ax2.semilogy(df_p['N'], df_p['gmean(relative_error_p)'], 'bo-', label='Geometric Mean',
                 markersize=5, linewidth=1.2, markerfacecolor='none')
    ax2.semilogy(df_p['N'], df_p['max(relative_error_p)'], 'rs-', label='Maximum',
                 markersize=5, linewidth=1.2, markerfacecolor='none')
    
    ax2.set_xlabel('$N$')
    ax2.set_ylabel('Relative Error $\\epsilon_{p,rel}$')
    ax2.legend(frameon=True, fancybox=False, edgecolor='black', 
              handletextpad=0.4, handlelength=1.5)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_xticks(df_p['N'])
    ax2.set_ylim(1e-6, 1e-1)  # Set fixed y-axis limits
    ax2.set_box_aspect(0.85)
    
    plt.savefig('error_p_plots.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()
    
    # [Convergence rate calculations remain the same...]

# Run both plots
print("Generating u solution plots...")
plot_u_solution()

print("\nGenerating p solution plots...")
plot_p_solution()