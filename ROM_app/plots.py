import matplotlib.pyplot as plt
import numpy as np

# Data from the table
N = np.array([1, 2, 3, 4, 5, 6, 7])
gmean_error_u = np.array([0.4617504059835661, 0.3638302294505106, 0.07017657697653167, 0.02799976889985112, 0.01054827499380351, 0.002444880323409685, 0.0007775233967706826])
max_error_u = np.array([0.9991298536696258, 1.2349775851280311, 0.2508302808209966, 0.207063625956523, 0.1169315124081540, 0.01278296493597559, 0.01669946232514180])
gmean_relative_error_u = np.array([0.04338455670636939, 0.03418429743979225, 0.006593561649904459, 0.002630766634657237, 0.0009910813909264516, 0.0002297125240188856, 7.306315330630109e-05])
max_relative_error_u = np.array([0.08611508724519280, 0.1064427757076664, 0.02400703475245687, 0.02004784011595418, 0.01125663936500955, 0.001230576970539628, 0.001640794138872988])

gmean_error_p = np.array([0.1679391374549639, 0.04418936645042013, 0.01638320755306223, 0.004686078063459092, 0.002013925983442150, 0.0007433362609273507, 0.0002259801238238133])
max_error_p = np.array([0.4526915337760337, 0.2654325113192067, 0.04036232864250473, 0.01428903020599992, 0.006948668358403820, 0.002192323106462842, 0.0009638348627447624])
gmean_relative_error_p = np.array([0.05446263984697234, 0.01433068563801280, 0.005313072021344077, 0.001519694490117183, 0.0006531159274536926, 0.0002410638501399983, 7.328532397453225e-05])
max_relative_error_p = np.array([0.1463838620137119, 0.08583115258817067, 0.01308516987964930, 0.004620549672714328, 0.002246944848493502, 0.0007092306038810870, 0.0003069206071506658])

def plot_metrics(data, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title)

    for ax, (subplot_title, metrics) in zip([ax1, ax2], [
        ('Error', ['Geometric Mean Error', 'Max Error']),
        ('Relative Error', ['Geometric Mean Relative Error', 'Max Relative Error'])
    ]):
        for metric in metrics:
            ax.plot(N, data[metric], 'o-', label=metric)
        ax.set_title(subplot_title)
        ax.set_xlabel('N')
        ax.set_ylabel('Error')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# Plot u metrics
u_data = {
    'Geometric Mean Error': gmean_error_u,
    'Max Error': max_error_u,
    'Geometric Mean Relative Error': gmean_relative_error_u,
    'Max Relative Error': max_relative_error_u
}
plot_metrics(u_data, 'Error Metrics for u')

# Plot p metrics
p_data = {
    'Geometric Mean Error': gmean_error_p,
    'Max Error': max_error_p,
    'Geometric Mean Relative Error': gmean_relative_error_p,
    'Max Relative Error': max_relative_error_p
}
plot_metrics(p_data, 'Error Metrics for p')