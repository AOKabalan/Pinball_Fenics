import numpy as np

# Load the .npz file
file_path = 'results_bdf3_pinball_steadydrag_lift_results.npz'
data = np.load(file_path)

# List the array names
print("Arrays in the .npz file:")
print(data.files)

# Access and print each array
for array_name in data.files:
    array = data[array_name]
    print(f"\nArray name: {array_name}")
    print(array)
