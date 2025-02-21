import yaml
from config import load_base_config, generate_config_variants, save_config
from simulation import run_simulation
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

def run_multiple_configurations(parameter_variations: Dict[str, list]) -> Any:
    """Run multiple configurations serially"""
    # Load base configuration
    base_config = load_base_config('base_config.yaml')
    
    # Generate all configuration variants
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
        print(f"  {param}: {values}")
    print(f"\nOutput directory: {output_dir}")
    
    results = []
    successful_runs = 0
    failed_runs = 0
    
    print("\n=== Starting Serial Runs ===")
    
    # Run configurations sequentially
    for i, config in enumerate(configs, 1):
        print(f"\nStarting configuration {i}/{len(configs)}")
        print(f"  ROM basis: {config['max_basis']['rom']}")
        print(f"  Training snapshots: {config['snapshots']['training']}")
        
        # try:
        result = run_single_configuration(config, i)
        results.append(result)
        successful_runs += 1
        print(f"✓ Configuration {i} completed successfully")
            
        # except Exception as e:
        #     failed_runs += 1
        #     print(f"✗ Configuration {i} failed: {str(e)}")
    
    print(f"\n=== Final Summary ===")
    print(f"Total configurations: {len(configs)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
   
    return results

def run_single_configuration(config: Dict[str, Any], run_number: int) -> Dict[str, Any]:
    """Run simulation with a single configuration"""
    # try:
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

if __name__ == "__main__":
    # Define parameter variations to test
    parameter_variations = {
        "snapshots.training": [100]
    }
    
    # Run all configurations serially
    results_df = run_multiple_configurations(parameter_variations)
    
    print("\nSimulation Summary:")
    if not results_df.empty:
        print("\nResults by ROM basis:")
        print(results_df.groupby('rom_max_basis')['run_number'].count())