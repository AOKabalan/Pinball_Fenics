#!/bin/bash
# Backup the original inputs2.json
cp inputs2.json inputs2.json.backup

# ----------------------------------------------------------------------
# Run 1: Original configuration (no changes)
# ----------------------------------------------------------------------
#echo "Running with ORIGINAL configuration..."
#python3 main.py

# ----------------------------------------------------------------------
# Run 2: Update amplitudes to 2.5
# ----------------------------------------------------------------------
echo "Updating amplitudes to 2.5..."
python3 - <<EOF
import json
with open("inputs2.json", "r") as f:
    data = json.load(f)
# Update results directory
data["results_dir"] = "study_results_25"

# Update amplitudes to 2.5 for all cylinders
for bc in data["boundary_conditions"]:
    if bc.get("bc_type") == "cylinder":
        bc["amplitude"] = 2.5

with open("inputs2.json", "w") as f:
    json.dump(data, f, indent=4)
EOF
echo "Running with -1.0 amplitude configuration..."
python3 main.py

# ----------------------------------------------------------------------
# Run 3: Update amplitudes to -2.5
# ----------------------------------------------------------------------
echo "Updating amplitudes to -2.5..."
python3 - <<EOF
import json
with open("inputs2.json", "r") as f:
    data = json.load(f)
# Update results directory
data["results_dir"] = "study_results_neg25"

# Update amplitudes to -2.5 for all cylinders
for bc in data["boundary_conditions"]:
    if bc.get("bc_type") == "cylinder":
        bc["amplitude"] = -2.5

with open("inputs2.json", "w") as f:
    json.dump(data, f, indent=4)
EOF
echo "Running with -2.5 amplitude configuration..."
python3 main.py

# Restore original configuration
mv inputs2.json.backup inputs2.json
