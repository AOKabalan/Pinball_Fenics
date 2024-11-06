import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Plot for u solution
def plot_u_solution():
    # Read the CSV file
    df_u = pd.read_csv('FluidicPinball2/error_analysis/error_analysis/solution_u.csv', sep=';')
    
    plt.figure(figsize=(12, 8))
    
    # Plot for errors
    plt.subplot(2, 1, 1)
    plt.semilogy(df_u['N'], df_u['gmean(error_u)'], 'bo-', label='Geometric Mean Error')
    plt.semilogy(df_u['N'], df_u['max(error_u)'], 'rs-', label='Maximum Error')
    plt.grid(True)
    plt.legend()
    plt.title('Error u vs N')
    plt.xlabel('N')
    plt.ylabel('Error')
    
    # Plot for relative errors
    plt.subplot(2, 1, 2)
    plt.semilogy(df_u['N'], df_u['gmean(relative_error_u)'], 'bo-',
              label='Geometric Mean Relative Error')
    plt.semilogy(df_u['N'], df_u['max(relative_error_u)'], 'rs-',
              label='Maximum Relative Error')
    plt.grid(True)
    plt.legend()
    plt.title('Relative Error u vs N')
    plt.xlabel('N')
    plt.ylabel('Relative Error')
    
    plt.tight_layout()
    plt.savefig('error_u_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate convergence rates
    def calc_rate(N, error):
        rates = []
        for i in range(len(N)-1):
            rate = np.log(error[i]/error[i+1]) / np.log(N[i+1]/N[i])
            rates.append(rate)
        return np.mean(rates)
    
    print("\nU Solution Convergence Rates:")
    print(f"Geometric mean error rate: {calc_rate(df_u['N'], df_u['gmean(error_u)']):.2f}")
    print(f"Maximum error rate: {calc_rate(df_u['N'], df_u['max(error_u)']):.2f}")

# Plot for p solution
def plot_p_solution():
    # Read the CSV file
    df_p = pd.read_csv('FluidicPinball2/error_analysis/error_analysis/solution_p.csv', sep=';')
    
    plt.figure(figsize=(12, 8))
    
    # Plot for errors
    plt.subplot(2, 1, 1)
    plt.semilogy(df_p['N'], df_p['gmean(error_p)'], 'bo-', label='Geometric Mean Error')
    plt.semilogy(df_p['N'], df_p['max(error_p)'], 'rs-', label='Maximum Error')
    plt.grid(True)
    plt.legend()
    plt.title('Error p vs N')
    plt.xlabel('N')
    plt.ylabel('Error')
    
    # Plot for relative errors
    plt.subplot(2, 1, 2)
    plt.semilogy(df_p['N'], df_p['gmean(relative_error_p)'], 'bo-',
              label='Geometric Mean Relative Error')
    plt.semilogy(df_p['N'], df_p['max(relative_error_p)'], 'rs-',
              label='Maximum Relative Error')
    plt.grid(True)
    plt.legend()
    plt.title('Relative Error p vs N')
    plt.xlabel('N')
    plt.ylabel('Relative Error')
    
    plt.tight_layout()
    plt.savefig('error_p_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate convergence rates
    def calc_rate(N, error):
        rates = []
        for i in range(len(N)-1):
            rate = np.log(error[i]/error[i+1]) / np.log(N[i+1]/N[i])
            rates.append(rate)
        return np.mean(rates)
    
    print("\nP Solution Convergence Rates:")
    print(f"Geometric mean error rate: {calc_rate(df_p['N'], df_p['gmean(error_p)']):.2f}")
    print(f"Maximum error rate: {calc_rate(df_p['N'], df_p['max(error_p)']):.2f}")

# Run both plots
print("Generating u solution plots...")
plot_u_solution()

print("\nGenerating p solution plots...")
plot_p_solution()