import numpy as np
import json
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
from concurrent.futures import ProcessPoolExecutor
import logging

def setup_logging(output_dir):
    logging.basicConfig(
        filename=output_dir / 'study.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def run_single_case(case_config_path):
    try:
        subprocess.run(['python', 'ns_app.py', '--config', str(case_config_path)], 
                      check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in {case_config_path}: {e.stderr.decode()}")
        return False

def run_parameter_study(Re_range, b_range, base_config='inputs2.json', max_workers=4):
    output_dir = Path(f"study_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True)
    setup_logging(output_dir)
    
    # Load base configuration
    with open(base_config) as f:
        base_config = json.load(f)
    
    # Prepare all configurations first
    cases = []
    for Re in Re_range:
        for b in b_range:
            case_name = f"Re{Re}_b{b}"
            case_dir = output_dir / case_name
            case_dir.mkdir()
            
            config = base_config.copy()
            config['nu'] = 1.0/Re
            config['results_dir'] = str(case_dir)
            
            for bc in config['boundary_conditions']:
                if bc.get('bc_type') == 'cylinder':
                    if bc['cylinder_type'] in ['top', 'bottom']:
                        bc['amplitude'] = b 
            
            config_path = case_dir / 'inputs2.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            cases.append(config_path)
            logging.info(f"Prepared case: Re={Re}, b={b}")
    
    # Run cases in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_case, cases))
    
    logging.info(f"Completed {sum(results)}/{len(results)} cases successfully")

if __name__ == "__main__":
    Re_range = np.linspace(15, 71, 8)
    b_range = np.linspace(0, 1, 5)
    run_parameter_study(Re_range, b_range)