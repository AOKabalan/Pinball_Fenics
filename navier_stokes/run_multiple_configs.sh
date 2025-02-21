#!/bin/bash

# Backup the original inputs2.json (optional but recommended)
cp inputs2.json inputs2.json.backup

# ----------------------------------------------------------------------
# Run 1: Original configuration (no changes)
# ----------------------------------------------------------------------
echo "Running with ORIGINAL configuration..."
python3 main.py

# ----------------------------------------------------------------------
# Run 2: Update to SYMM configuration
# ----------------------------------------------------------------------
echo "Updating to SYMM configuration..."
python3 - <<EOF
import json

with open("inputs2.json", "r") as f:
    data = json.load(f)

# Update keys for SYMM run
data["results_dir"] = "study_results_symm"
data["u0_file"] = "states/velocity_checkpoint_symm.xdmf"

with open("inputs2.json", "w") as f:
    json.dump(data, f, indent=4)
EOF

echo "Running with SYMM configuration..."
python3 main.py

# ----------------------------------------------------------------------
# Run 3: Update to ASYMM_DOWN configuration
# ----------------------------------------------------------------------
echo "Updating to ASYMM_DOWN configuration..."
python3 - <<EOF
import json

with open("inputs2.json", "r") as f:
    data = json.load(f)

# Update keys for ASYMM_DOWN run
data["results_dir"] = "study_results_asymm_down"
data["u0_file"] = "states/velocity_checkpoint_down.xdmf"

with open("inputs2.json", "w") as f:
    json.dump(data, f, indent=4)
EOF

echo "Running with ASYMM_DOWN configuration..."
python3 main.py

# Restore original configuration (optional)
# mv inputs2.json.backup inputs2.json
