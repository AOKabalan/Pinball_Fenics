# import numpy as np
# import json
# from pathlib import Path
# from datetime import datetime
# import subprocess
# import logging
# from time import sleep

# def run_parameter_study(Re_range, b_range, base_config='inputs2.json'):
#     # Setup output directory and logging
#     output_dir = Path(f"study_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
#     output_dir.mkdir(parents=True)
#     logging.basicConfig(
#         filename=output_dir / 'study.log',
#         level=logging.INFO,
#         format='%(asctime)s - %(message)s'
#     )

#     # Load base configuration
#     with open(base_config) as f:
#         base_config = json.load(f)

#     results = {}
#     total_cases = len(Re_range) * len(b_range)
#     current_case = 0

#     for Re in Re_range:
#         for b in b_range:
#             current_case += 1
#             case_name = f"Re{Re}_b{b}"
#             case_dir = output_dir / case_name
#             case_dir.mkdir()
            
#             # Prepare configuration
#             config = base_config.copy()
#             config['nu'] = 1.0/Re
#             config['results_dir'] = str(case_dir)
            
#             # Update cylinder rotations
#             for bc in config['boundary_conditions']:
#                 if bc.get('bc_type') == 'cylinder':
#                     if bc['cylinder_type'] in ['top', 'bottom']:
#                         bc['amplitude'] = b 
            
#             # Save case configuration
#             case_config = case_dir / 'inputs2.json'
#             with open(case_config, 'w') as f:
#                 json.dump(config, f, indent=4)
            
#             logging.info(f"Starting case {current_case}/{total_cases}: Re={Re}, b={b}")
            
#             try:
#                 # Run simulation
#                 with open(base_config, 'w') as f:
#                     json.dump(config, f, indent=4)
                
#                 subprocess.run(['python', 'ns_app.py'], check=True)
                
#                 results[case_name] = {'Re': Re, 'b': b, 'status': 'completed'}
#                 logging.info(f"Completed case {case_name}")
                
#             except Exception as e:
#                 results[case_name] = {'Re': Re, 'b': b, 'status': 'failed', 'error': str(e)}
#                 logging.error(f"Failed case {case_name}: {str(e)}")
            
#             # Save running results
#             with open(output_dir / 'results.json', 'w') as f:
#                 json.dump(results, f, indent=4)
            
#             # Small delay between runs
#             sleep(1)

#     logging.info("Study completed")
#     return results

# if __name__ == "__main__":
#     Re_range = np.linspace(15, 71, 3)
#     b_range = np.linspace(0, 1, 3)
#     run_parameter_study(Re_range, b_range)

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import subprocess
import os

def run_parameter_study(Re_range, b_range, config_file='inputs2.json'):
    # Get current working directory for module imports
    working_dir = os.getcwd()
    
    output_dir = Path(f"study_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True)

    with open(config_file) as f:
        base_config = json.load(f)

    total_cases = len(Re_range) * len(b_range)
    current_case = 0

    for Re in Re_range:
        for b in b_range:
            current_case += 1
            case_name = f"Re{Re}_b{b}"
            case_dir = output_dir / case_name
            case_dir.mkdir()
            
            config = base_config.copy()
            config['nu'] = 1.0/Re
            config['results_dir'] = str(case_dir)
            
            for bc in config['boundary_conditions']:
                if bc.get('bc_type') == 'cylinder':
                    if bc['cylinder_type'] in ['top', 'bottom']:
                        bc['amplitude'] = b if bc['cylinder_type'] == 'top' else -b
            
            print(f"\nStarting case {current_case}/{total_cases}: Re={Re}, b={b}")
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            subprocess.run(['python', 'ns_app.py'], check=True)
            
            print(f"Completed case {case_name}")

if __name__ == "__main__":
    Re_range = np.linspace(15, 71, 3)
    b_range = np.linspace(0, 1, 3)
    run_parameter_study(Re_range, b_range)