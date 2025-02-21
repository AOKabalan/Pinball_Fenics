import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# Set up publication-quality plot formatting
plt.style.use('default')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('axes', labelsize=18)  # Increased from 14
rc('axes', titlesize=18)  # Increased from 14
rc('xtick', labelsize=14)  # Increased from 11
rc('ytick', labelsize=14)  # Increased from 11
rc('legend', fontsize=12)  # Increased from 10
# Set general plot parameters
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['figure.dpi'] = 300

def cap_values(df, solution_type, cap_value=100):
    """Cap values in the dataframe to specified maximum."""
    df_capped = df.copy()
    columns_to_cap = [
        f'gmean(error_{solution_type})',
        f'max(error_{solution_type})',
        f'gmean(relative_error_{solution_type})',
        f'max(relative_error_{solution_type})'
    ]
    
    for col in columns_to_cap:
        # Replace inf values with cap_value
        df_capped[col] = df_capped[col].replace([np.inf, -np.inf], cap_value)
        # Cap remaining values
        df_capped[col] = df_capped[col].clip(upper=cap_value)
    
    return df_capped

def load_data(run_number, solution_type='u'):
    path = f'runs/run_{run_number}/speedup_analysis/speedup_analysis/speedup_solve.csv'
    
    # path = f'runs/run_{run_number}/error_analysis/error_analysis/solution_{solution_type}.csv'
    df = pd.read_csv(path, sep=';')
    return cap_values(df, solution_type)

def plot_comparison(solution_type='u'):
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1,
                         left=0.08, right=0.98, top=0.85, bottom=0.15)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
        
    colors = ['b', 'r', 'g', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h']
    labels = ['2 DEIM', '4 DEIM', '6 DEIM', '8 DEIM', '10 DEIM', '15 DEIM', '20 DEIM', '30 DEIM', '40 DEIM']
    numbers = ['01', '02', '03', '04', '1', '2', '3', '4', '5']
    for i in range(1, 7):
        df = load_data(numbers[i-1], solution_type)
        
        ax1.semilogy(df['N'][:12], df[f'gmean(error_{solution_type})'][:12], 
                    color=colors[i-1], marker=markers[i-1], linestyle='-',
                    label=f'{labels[i-1]}', markersize=5, linewidth=1.2, 
                    markerfacecolor='none')
        
        ax2.semilogy(df['N'][:12], df[f'gmean(relative_error_{solution_type})'][:12], 
                    color=colors[i-1], marker=markers[i-1], linestyle='-',
                    label=f'{labels[i-1]}', markersize=5, linewidth=1.2, 
                    markerfacecolor='none')
    
    ax1.set_xlabel('$N$')
    ax1.set_ylabel(f'Error $\\epsilon_{solution_type}$')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black',
              handletextpad=0.4, handlelength=1.5)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_xlim(0.5, 12.5)
    ax1.set_xticks(range(1, 13))
    ax1.set_box_aspect(0.85)
    ax1.set_ylim(3e-4, 100)
    
    ax2.set_xlabel('$N$')
    ax2.set_ylabel(f'Relative Error $\\epsilon_{{{solution_type},rel}}$')
    ax2.legend(frameon=True, fancybox=False, edgecolor='black',
              handletextpad=0.4, handlelength=1.5)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_xlim(0.5, 12.5)
    ax2.set_box_aspect(0.85)
    ax2.set_ylim(1e-5, 100)
    ax2.set_xticks(range(1, 13))
    
    plt.savefig(f'comparison_{solution_type}_errors_new.pdf', bbox_inches='tight', pad_inches=0.02)

def plot_comparison_p(solution_type='p'):
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1,
                         left=0.08, right=0.98, top=0.85, bottom=0.15)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    colors = ['b', 'r', 'g', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h']
    labels = ['2 DEIM', '4 DEIM', '6 DEIM', '8 DEIM', '10 DEIM', '15 DEIM', '20 DEIM', '30 DEIM', '40 DEIM']
    numbers = ['01', '02', '03', '04', '1', '2', '3', '4', '5']
    
    for i in range(1, 7):
        df = load_data(numbers[i-1], solution_type)
        
        ax1.semilogy(df['N'][:12], df[f'gmean(error_{solution_type})'][:12], 
                    color=colors[i-1], marker=markers[i-1], linestyle='-',
                    label=f'{labels[i-1]}', markersize=5, linewidth=1.2, 
                    markerfacecolor='none')
        
        ax2.semilogy(df['N'][:12], df[f'gmean(relative_error_{solution_type})'][:12], 
                    color=colors[i-1], marker=markers[i-1], linestyle='-',
                    label=f'{labels[i-1]}', markersize=5, linewidth=1.2, 
                    markerfacecolor='none')
    
    ax1.set_xlabel('$N$')
    ax1.set_ylabel(f'Error $\\epsilon_{solution_type}$')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black',
              handletextpad=0.4, handlelength=1.5)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_xlim(0.5, 12.5)
    ax1.set_box_aspect(0.85)
    ax1.set_ylim(3e-5, 100)
    ax1.set_xticks(range(1, 13))
    
    ax2.set_xlabel('$N$')
    ax2.set_ylabel(f'Relative Error $\\epsilon_{{{solution_type},rel}}$')
    ax2.legend(frameon=True, fancybox=False, edgecolor='black',
              handletextpad=0.4, handlelength=1.5)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_xlim(0.5, 12.5)
    ax2.set_box_aspect(0.85)
    ax2.set_ylim(5e-6, 100)
    ax2.set_xticks(range(1, 13))
    
    plt.savefig(f'comparison_{solution_type}_errors_new.pdf', bbox_inches='tight', pad_inches=0.02)

def plot_speedup_comparison():
    """
    Create a plot comparing speedup metrics across different DEIM configurations.
    """
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1,
                         left=0.08, right=0.98, top=0.85, bottom=0.15)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    colors = ['b', 'r', 'g', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h']
    labels = ['2 DEIM', '4 DEIM', '6 DEIM', '8 DEIM', '10 DEIM', '15 DEIM', '20 DEIM', '30 DEIM', '40 DEIM']
    numbers = ['01', '02', '03', '04', '1', '2', '3', '4', '5']
    
    for i in range(1, 7):
        path = f'runs/run_{numbers[i-1]}/speedup_analysis/speedup_analysis/speedup_solve.csv'
        df = pd.read_csv(path, sep=';')
        
        # Plot minimum and geometric mean speedup
        ax1.semilogy(df['N'][:12], df['min(speedup_solve)'][:12], 
                    color=colors[i-1], marker=markers[i-1], linestyle='-',
                    label=f'{labels[i-1]}', markersize=5, linewidth=1.2, 
                    markerfacecolor='none')
        
        ax2.semilogy(df['N'][:12], df['gmean(speedup_solve)'][:12], 
                    color=colors[i-1], marker=markers[i-1], linestyle='-',
                    label=f'{labels[i-1]}', markersize=5, linewidth=1.2, 
                    markerfacecolor='none')
    
    # Configure first subplot (minimum speedup)
    ax1.set_xlabel('$N$')
    ax1.set_ylabel('Minimum Speedup')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black',
              handletextpad=0.4, handlelength=1.5)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_xlim(0.5, 12.5)
    ax1.set_xticks(range(1, 13))
    ax1.set_box_aspect(0.85)
    ax1.set_ylim(1e-1, 2)
    
    # Configure second subplot (geometric mean speedup)
    ax2.set_xlabel('$N$')
    ax2.set_ylabel('Geometric Mean Speedup')
    ax2.legend(frameon=True, fancybox=False, edgecolor='black',
              handletextpad=0.4, handlelength=1.5)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_xlim(0.5, 12.5)
    ax2.set_box_aspect(0.85)
    ax2.set_ylim(1e-1, 2)
    ax2.set_xticks(range(1, 13))
    
    plt.savefig('comparison_speedup.pdf', bbox_inches='tight', pad_inches=0.02)
    print("Generated speedup comparison plot.")
# Plot comparisons for both u and p solutions
# print("Generating u solution comparison plots...")
# plot_comparison('u')

# print("\nGenerating p solution comparison plots...")
# plot_comparison_p('p')

print("Generating speedup comparison plots...")
plot_speedup_comparison()