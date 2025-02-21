import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def edit_inputs_and_run_simulation(json_file_path, amplitudes):
    """
    Edits the inputs2.json file to change the output directory and amplitude,
    then runs the simulation for each amplitude value.
    
    Args:
        json_file_path (str): Path to the JSON input file.
        amplitudes (list): List of amplitude values to iterate over.
    """
    # Initialize the results directories
    results_dirs = []

    for idx, amplitude in enumerate(amplitudes):
        # Create the new results directory name
        run_dir = f"steady_states_runs/run{idx + 1}"
        results_dirs.append(run_dir)

        # Load the JSON configuration file
        with open(json_file_path, 'r') as f:
            config = json.load(f)

        # Update the results directory and amplitude in the JSON file
        config["results_dir"] = run_dir
        for bc in config["boundary_conditions"]:
            if "bc_type" in bc and bc["bc_type"] == "cylinder":
                bc["amplitude"] = amplitude

        # Save the updated JSON configuration file
        with open(json_file_path, 'w') as f:
            json.dump(config, f, indent=4)

        # Run the simulation (assuming you have a command to execute the simulation)
        print(f"Running simulation for amplitude {amplitude}...")
        os.system("python main.py")  # Replace "python main.py" with your actual simulation command

    return results_dirs

def extract_lift_and_plot(amplitudes,running_dir):
    """
    Extracts the lift coefficients from the forces files and plots them against the amplitudes.
    
    Args:
        amplitudes (list): List of amplitude values used in the simulations.
    """
    lift_values = []
    all_amplitudes = []

    for idx, amplitude in enumerate(amplitudes):
        # Create the new results directory name
        run_dir = f"{running_dir}/run{idx + 1}"
        
        # Check if the forces file exists
        forces_file = os.path.join(run_dir, "forces", "forces_steady.csv")
        if os.path.exists(forces_file):
            try:
                # Read the forces from the CSV file
                forces = pd.read_csv(forces_file)
                
                # Ensure the 'lift' column exists and has at least one value
                if 'lift' in forces.columns and not forces['lift'].empty:
                    lift = forces['lift'].iloc[-1]  # Assuming the last value is the steady-state lift
                    lift_values.append(lift)
                    all_amplitudes.append(amplitude)
                else:
                    print(f"Warning: No valid lift data found for amplitude {amplitude}. Skipping...")
            except Exception as e:
                print(f"Error processing forces file for amplitude {amplitude}: {e}")
        else:
            print(f"Warning: Forces file not found for amplitude {amplitude}. Skipping...")

    # Plot the lift values vs. amplitude
    if lift_values and all_amplitudes:
        plt.figure(figsize=(10, 6))
        plt.plot(all_amplitudes, lift_values, marker='o', linestyle='-', color='b')
        plt.title("Lift vs Amplitude")
        plt.xlabel("Amplitude")
        plt.ylabel("Lift")
        plt.grid(True)
        plt.savefig("liftvsamp.png")  # Save the plot as a PNG file
        plt.show()
    else:
        print("No valid data to plot.")
def main():
    """
    Main function to orchestrate the simulation and plotting process.
    """
    # Define the amplitude range and step size
    # amplitudes = [round(i / 10, 1) for i in range(-30, 31)]  # [-3.0, -2.9, ..., 3.0]
    amplitudes = np.linspace(0, 2, 40)
    # # Path to the JSON input file
    # json_file_path = "inputs2.json"

    # # Step 1: Edit the inputs and run the simulation
    # results_dirs = edit_inputs_and_run_simulation(json_file_path, amplitudes)
    running_dir = "trial_cont"
    # Step 2: Extract lift coefficients and plot them
    extract_lift_and_plot(amplitudes,running_dir)


if __name__ == "__main__":
    main()
