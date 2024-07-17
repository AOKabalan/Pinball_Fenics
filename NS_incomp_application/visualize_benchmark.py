import numpy as np
import matplotlib.pyplot as plt

def visualize_bench(method,results_dir="results",res=0.015, lvl=4):
    # Load data from the results directory

    infile = np.load(f"{results_dir}/drag_lift_results.npz")

    # Load Turek benchmark data for comparison
    turek = np.loadtxt(f"bdforces_lv{lvl}")
    turek_p = np.loadtxt(f"pointvalues_lv{lvl}")

    # Plot the results
    
    fig = plt.figure(figsize=(25, 8))
    # Drag coefficient (CD) plot
   
    l1 = plt.plot(infile["t"][1:], infile["CD"][1:], label=f"FEniCS_{method}", linewidth=2)
    l2 = plt.plot(turek[1:, 1], turek[1:, 3], marker="x", markevery=50, linestyle="", markersize=4, label=f"Turek Lv {lvl}")
    plt.title("Drag coefficient")
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig("figures/drag_comparison_bdf3_finer.png")
    
    # Lift coefficient (CL) plot
    fig = plt.figure(figsize=(25, 8))
    l1 = plt.plot(infile["t"], infile["CL"], label=f"FEniCS_{method}" ,linewidth=2)
    l2 = plt.plot(turek[1:, 1], turek[1:, 4], marker="x", markevery=50,
                  linestyle="", markersize=4, label=f"Turek Lv {lvl}")
    plt.title("Lift coefficient")
    plt.grid()
    plt.legend()
    plt.savefig(f"figures/lift_comparison_{method}.png")   
    

    # Pressure difference (Δp) plot
    fig = plt.figure(figsize=(25, 8))
    l1 = plt.plot(infile["t"], infile["dp"], label=f"FEniCS_{method}", linewidth=2)
    l2 = plt.plot(turek[1:, 1], turek_p[1:, 6] - turek_p[1:, -1], marker="x", markevery=50,
                  linestyle="", markersize=4, label=f"Turek Lv {lvl}")
    plt.title("Pressure difference")
    plt.grid()
    plt.legend()
    plt.savefig(f"figures/pressure_comparison_{method}.png")
    

if __name__ == "__main__":
  
    
    visualize_bench('bdf3')
