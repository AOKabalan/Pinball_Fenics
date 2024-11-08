import yaml
from config import load_base_config, generate_config_variants, save_config
from simulation import run_simulation
import os
import sys
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any
import logging
import uuid
import multiprocessing

def setup_process_logging(run_dir: str) -> None:
    """Setup logging for a single process/configuration"""
    # Create a log file specific to this run
    log_file = open(f"{run_dir}/process.log", 'w')
    
    class TeeOutput:
        def __init__(self, file):
            self.file = file
            self.stdout = sys.__stdout__
            self.stderr = sys.__stderr__

        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
            self.file.flush()
            
        def flush(self):
            self.file.flush()

    # Replace stdout and stderr for this process
    sys.stdout = TeeOutput(log_file)
    sys.stderr = TeeOutput(log_file)

def run_multiple_configurations(parameter_variations: Dict[str, list]) -> pd.DataFrame:
    """Run multiple configurations in parallel"""
    # Load base configuration
    base_config = load_base_config('base_config.yaml')
    
    # Generate all configuration variants
    configs = generate_config_variants(base_config, parameter_variations)
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"simulation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup main process logging
    main_log = open(f"{output_dir}/main_process.log", 'w', buffering=1)
    
    class MainTeeOutput:
        def __init__(self, file):
            self.file = file
            self.stdout = sys.__stdout__
            
        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
            self.file.flush()
            self.stdout.flush()
            
        def flush(self):
            self.file.flush()
            self.stdout.flush()
    
    sys.stdout = MainTeeOutput(main_log)
    
    print(f"=== Simulation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Total configurations to run: {len(configs)}")
    print(f"\nParameter variations:")
    for param, values in parameter_variations.items():
        print(f"  {param}: {values}")
    print(f"\nOutput directory: {output_dir}")
    
    results = []
    successful_runs = 0
    failed_runs = 0
    
    print("\n=== Starting Parallel Runs ===")
    
    # Create process pool without initialization
    with ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), len(configs))) as executor:
        futures = []
        for i, config in enumerate(configs, 1):
            print(f"\nSubmitting configuration {i}/{len(configs)}")
            print(f"  ROM basis: {config['max_basis']['rom']}")
            print(f"  Training snapshots: {config['snapshots']['training']}")
            future = executor.submit(run_single_configuration, config)
            futures.append((i, config, future))
        
        print("\nAll configurations submitted. Processing results...")
        
        for i, config, future in futures:
            try:
                print(f"\nWaiting for configuration {i}/{len(configs)}...")
                result = future.result(timeout=None)
                results.append(result)
                successful_runs += 1
                print(f"✓ Configuration {i} completed successfully")
                
            except Exception as e:
                failed_runs += 1
                print(f"✗ Configuration {i} failed: {str(e)}")
    
    print(f"\n=== Final Summary ===")
    print(f"Total configurations: {len(configs)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    
    if results:
        df = pd.DataFrame(results)
        csv_path = f"{output_dir}/all_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    
    print(f"\n=== Simulation Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    return df if results else pd.DataFrame()

def run_single_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run simulation with a single configuration"""
    try:
        # Create unique output directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        run_dir = f"runs/{config['simulation_name']}_{timestamp}_{unique_id}"
        
        # Ensure directory exists
        os.makedirs(run_dir, exist_ok=True)
        
        # Setup logging with unbuffered output
        log_file = open(f"{run_dir}/process.log", 'w', buffering=1)
        
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

        # Set up output
        sys.stdout = TeeOutput(log_file)
        sys.stderr = TeeOutput(log_file)
        
        print(f"\n{'='*50}")
        print(f"Starting configuration in directory: {run_dir}")
        print(f"ROM basis: {config['max_basis']['rom']}")
        print(f"Training snapshots: {config['snapshots']['training']}")
        print(f"{'='*50}\n")
        
        # Save this configuration
        save_config(config, f"{run_dir}/config.yaml")
        print('lahon meshe l7al')
        
        # Modify config to use run-specific directories
        config['simulation_name'] = run_dir
        if config['bifurcation']['enabled']:
            config['bifurcation']['output_dir'] = f"{run_dir}/bifurcation_results"
        print('lahon meshe l7al 2')
        
        # Run the simulation
        print("\nStarting simulation...")
        results = run_simulation(config)
        print("Simulation completed successfully")
        
        # Add configuration parameters to results
        results['rom_tolerance'] = config['tolerances']['rom']
        results['rom_max_basis'] = config['max_basis']['rom']
        results['online_Re'] = config['parameters']['online_Re']
        
        return results
        
    except Exception as e:
        print(f"\n{'!'*50}")
        print(f"ERROR in configuration:")
        print(f"ROM basis: {config['max_basis']['rom']}")
        print(f"Training snapshots: {config['snapshots']['training']}")
        print(f"Error: {str(e)}")
        print(f"{'!'*50}\n")
        raise
    
if __name__ == "__main__":
    # Define parameter variations to test
    parameter_variations = {
        "snapshots.training": [4, 8],
        "max_basis.rom": [5, 10]
    }
    
    # # Run all configurations
    results_df = run_multiple_configurations(parameter_variations)
    # base_config = load_base_config('base_config.yaml')
    
   
    # results = run_single_configuration(base_config)
    # Print summary
    print("\nSimulation Summary:")
    #print(f"Total configurations run: {len(results_df)}")
    print("\nResults by ROM tolerance:")
    #print(results_df.groupby('config.rom_tolerance')['lift_coefficient'].describe())
