import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def modify_json(file_path, nu_value):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    data['nu'] = nu_value
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def run_program(program_path, input_json):
    subprocess.run(["python", program_path, input_json])

def extract_data_from_npz(npz_file, key):
    with np.load(npz_file) as data:
        return data[key]

def main():
    input_json = "inputs2.json"
    program_path = "ns_app.py"
    output_npz = "results_bdf3_pinball_steadydrag_lift_results.npz" 
    nu_values = [0.05,0.0375,0.025,0.024,0.0235,0.023,0.022,0.0214,0.0211,0.02,0.017,0.015]  # Example values, adjust as needed
    results = []
    re_values = []

    for nu in nu_values:
        modify_json(input_json, nu)
        run_program(program_path, input_json)
        result = extract_data_from_npz(output_npz, "CL")
        results.append(result)
        re_values.append(1.5 / nu)  # Calculate Reynolds number

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(re_values, results, marker='o')
    plt.xlabel("Reynolds number (Re)")
    plt.ylabel("CL")
    plt.title("Effect of Reynolds number on CL")
    plt.grid(True)

    # Set x-axis range
    plt.xlim(20, 110)

    # Set y-axis range
    plt.ylim(-0.04, 0.04)

    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='r', linestyle='--', linewidth=0.5)

    plt.savefig("figures/re_effect_graph.png")
    plt.show()

if __name__ == "__main__":
    main()
