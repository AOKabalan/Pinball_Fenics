import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data files
forces = pd.read_csv('study_results_bdf3/forces/forces_unsteady_bdf3.csv')
bdforces = np.loadtxt('bdforces_lv3', skiprows=1)

# Create drag force comparison plot
fig1 = plt.figure(figsize=(10, 4.5))
ax1 = fig1.add_subplot(111)

# Plot your data
ax1.plot(forces['time'], forces['drag'], 'bo-', 
         label='Your Solution', 
         markersize=0.1, 
         linewidth=0.1, 
         markerfacecolor='none')
# Plot bdforces data
ax1.plot(bdforces[:, 1], bdforces[:, 3], 'ro-',
         label='Reference',
         markersize=0.1,
         linewidth=0.1,
         markerfacecolor='none')

ax1.set_xlabel('Time')
ax1.set_ylabel('Drag Force')
ax1.grid(True)
ax1.legend()

plt.tight_layout()
plt.savefig('drag_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create lift force comparison plot
fig2 = plt.figure(figsize=(10, 4.5))
ax2 = fig2.add_subplot(111)

# Plot your data
ax2.plot(forces['time'], forces['lift'], 'bo-',
         label='Your Solution',
         markersize=0.1,
         linewidth=0.1,
         markerfacecolor='none')

# Plot bdforces data
ax2.plot(bdforces[:, 1], bdforces[:, 4], 'ro-',
         label='Reference',
         markersize=0.1,
         linewidth=0.1,
         markerfacecolor='none')

ax2.set_xlabel('Time')
ax2.set_ylabel('Lift Force')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('lift_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
