import yaml
from config import load_base_config, generate_config_variants, save_config
from simulation_deim import run_simulation
import os
import sys
from datetime import datetime
from typing import Dict, Any
import logging
import uuid

def setup_logging(run_dir: str) -> None:
    """Setup logging for a simulation run"""
    # Create a log file specific to this run
    log_file = open(f"{run_dir}/simulation.log", 'w', buffering=1)
    
    class TeeOutput:
        def __init__(self, file):
            self.file = file
            self.stdout = sys.__stdout__
            self.stderr = sys.__stderr__

        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
            self.file.flush()
            self.stdout.flush()
            
        def flush(self):
            self.file.flush()
            self.stdout.flush()

    # Replace stdout and stderr
    sys.stdout = TeeOutput(log_file)
    sys.stderr = TeeOutput(log_file)
def run_multiple_configurations(parameter_variations: Dict[str, list]) -> Dict[str, Any]:
    """Run multiple configurations serially"""
    # Load base configuration 
    base_config = load_base_config('base_config.yaml')
    configs = generate_config_variants(base_config, parameter_variations)
    
    # Setup main output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"simulation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup main logging
    setup_logging(output_dir)
    
    print(f"=== Simulation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Total configurations to run: {len(configs)}")
    print(f"\nParameter variations:")
    for param, values in parameter_variations.items():
        print(f" {param}: {values}")
    print(f"\nOutput directory: {output_dir}")
    
    results = []
    successful_runs = 0
    failed_runs = 0
    
    print("\n=== Starting Serial Runs ===")
    # Run configurations sequentially
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}/{len(configs)} is requesting green light")
        print(f"Parameters:")
        print(f" ROM basis: {config['max_basis']['rom']}")
        print(f" Training snapshots: {config['snapshots']['training']}")
        print(f" DEIM Training snapshots: {config['snapshots']['deim']}")
        
        while True:
            try:
                response = input(f"\nDo you want to proceed with configuration {i}? (y/n): ").lower()
                if response in ['y', 'n']:
                    break
                print("Please enter 'y' for yes or 'n' for no.")
            except KeyboardInterrupt:
                print("\nUser interrupted the program")
                return results
        
        if response == 'n':
            print(f"⚠ Configuration {i} skipped by user")
            failed_runs += 1
            continue  # Skip to next configuration
            
        
        print(f"\nStarting configuration {i}/{len(configs)}")
        result = run_single_configuration(config, i)
        results.append(result)
        successful_runs += 1
        print(f"✓ Configuration {i} completed successfully")
    
    print(f"\n=== Final Summary ===")
    print(f"Total configurations: {len(configs)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    return results

def run_single_configuration(config: Dict[str, Any], run_number: int) -> Dict[str, Any]:
    """Run simulation with a single configuration"""
    #try:
        # Create unique output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    run_dir = f"runs/run_{run_number}_{config['simulation_name']}_{timestamp}_{unique_id}"
   
    # Ensure directory exists
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup logging for this run
    setup_logging(run_dir)
    
    print(f"\n{'='*50}")
    print(f"Starting configuration {run_number} in directory: {run_dir}")
    print(f"ROM basis: {config['max_basis']['rom']}")
    print(f"Training snapshots: {config['snapshots']['training']}")
    print(f"DEIM training snapshots: {config['snapshots']['deim']}")
    print(f"{'='*50}\n")
    
    # Save this configuration
    save_config(config, f"{run_dir}/config.yaml")
    
    # Modify config to use run-specific directories
    config['simulation_name'] = run_dir
    if config['bifurcation']['enabled']:
        config['bifurcation']['output_dir'] = f"{run_dir}/bifurcation_results"
    
    # Run the simulation
    print("\nStarting simulation...")
    results = run_simulation(config)
    print("Simulation completed successfully")
    
    # Add configuration parameters to results
    results['rom_tolerance'] = config['tolerances']['rom']
    results['rom_max_basis'] = config['max_basis']['rom']
    results['online_Re'] = config['parameters']['online_Re']
    results['run_number'] = run_number
    results['run_directory'] = run_dir
    
    return results
        
    # except Exception as e:
    #     print(f"\n{'!'*50}")
    #     print(f"ERROR in configuration {run_number}:")
    #     print(f"ROM basis: {config['max_basis']['rom']}")
    #     print(f"Training snapshots: {config['snapshots']['training']}")
    #     print(f"Error: {str(e)}")
    #     print(f"{'!'*50}\n")
    #     raise



def run_online_analysis(config: Dict[str, Any]) -> Dict[str, Any]:

    offline_dir = config["analysis"]["offline_directory"]
    
    if not os.path.exists(offline_dir):
        raise ValueError(f"Offline directory does not exist: {offline_dir}")
    
    print(f"=== Starting Offline Analysis ===")
    print(f"Using existing directory: {offline_dir}")
    
    # Setup logging in the existing directory
    setup_logging(f"{offline_dir}/offline_analysis.log")

    config['simulation_name'] = offline_dir
    results = run_simulation(base_config)
    return results

   
if __name__ == "__main__":
    # Define parameter variations to test
    parameter_variations = {
        "snapshots.training": [100],
        "snapshots.deim": [100]
    }
    base_config = load_base_config('base_config.yaml')
    # Run all configurations serially
    if config["analysis"]["run_online"]:
        results_df = run_online_analysis(base_config)
    else:
        results_df = run_multiple_configurations(parameter_variations)
    
    print("\nThanks")
