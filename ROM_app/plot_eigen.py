import numpy as np
import matplotlib.pyplot as plt
import ast

def read_eigenvalues(filename):
    with open(filename, 'r') as f:
        content = f.read()
        return np.array(ast.literal_eval(content))

def plot_eigenvalues(filenames, labels, colors, Nmax):
    plt.figure(figsize=(12, 7))
    
    for filename, label, color in zip(filenames, labels, colors):
        eigenvalues = read_eigenvalues(filename)
        eigenvalues = eigenvalues[:Nmax]
        indices = np.arange(1, len(eigenvalues) + 1)
        
        plt.semilogy(indices, eigenvalues, 'o-', color=color, 
                    label=f'{label}', markersize=4)
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Mode number')
    plt.ylabel('Eigenvalue magnitude')
    plt.title(f'POD Eigenvalue Decay (First {Nmax} modes)')
    plt.xticks(np.arange(1, Nmax + 1))
    plt.legend()
    plt.tight_layout()
    plt.show()

# Use the function
Nmax = 10
filenames = ['FluidicPinball/post_processing/eigs_u.txt', 
             'FluidicPinball/post_processing/eigs_p.txt', 
             'FluidicPinball/post_processing/eigs_s.txt']
labels = ['Velocity', 'Pressure', 'supremizer']
colors = ['blue', 'red', 'green']
plot_eigenvalues(filenames, labels, colors, Nmax)