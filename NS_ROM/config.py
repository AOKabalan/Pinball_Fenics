# config.py
import yaml
from typing import Dict, List, Any
from itertools import product
import copy

def load_base_config(config_file: str) -> Dict[str, Any]:
    """Load base configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def generate_config_variants(base_config: Dict[str, Any], parameter_variations: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate multiple configurations based on parameter variations.
    
    Args:
        base_config: Base configuration dictionary
        parameter_variations: Dict of parameters to vary and their possible values
        Example: {
            "tolerances.rom": [1e-14, 1e-15, 1e-16],
            "max_basis.rom": [15, 20, 25],
            "parameters.online_mu": [(0.0125,), (0.0135,)]
        }
    """
    configs = []
    
    # Create all combinations of parameter values
    param_names = list(parameter_variations.keys())
    param_values = list(parameter_variations.values())
    
    for values in product(*param_values):
        # Create a deep copy of the base config
        config = copy.deepcopy(base_config)
        
        # Update the config with the current parameter combination
        for param_name, value in zip(param_names, values):
            # Handle nested parameters using dot notation
            keys = param_name.split('.')
            current = config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
            
        configs.append(config)
    
    return configs

def save_config(config: Dict[str, Any], filename: str) -> None:
    """Save configuration to YAML file"""
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
