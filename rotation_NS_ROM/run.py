import yaml
from simulation import run_simulation  # Ensure this module is implemented
import os
import sys
from datetime import datetime

from typing import Dict, Any
import logging
import uuid
def load_config(config_file: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
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
def main():
    # Load the configuration file
    config = load_config('base_config.yaml')
    os.makedirs(config['simulation_name'], exist_ok=True)
    setup_logging(config['simulation_name'])
    # Run the simulation with the loaded configuration
    print("Starting simulation...")
    results = run_simulation(config)
    print("Simulation completed successfully.")
    
    # Optionally, print or save results
    if results:
        print("Simulation results:")
        print(results)

if __name__ == "__main__":
    main()