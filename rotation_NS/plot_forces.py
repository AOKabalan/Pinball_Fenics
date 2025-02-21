import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy import signal
# Set up publication-quality plot formatting
plt.style.use('default')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('axes', labelsize=9)
rc('axes', titlesize=9)
rc('xtick', labelsize=8)
rc('ytick', labelsize=8)
rc('legend', fontsize=8)

# Set general plot parameters
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['figure.dpi'] = 300

def plot_forces():
    # Read both CSV files
    forces_symm = pd.read_csv('study_results_25/forces/forces_unsteady_bdf3.csv')
    # forces_lower = pd.read_csv('study_results_asymm_down/forces/forces_unsteady_bdf3.csv')
    # forces_upper = pd.read_csv('study_results_asymm_up/forces/forces_unsteady_bdf3.csv')

    # Create drag force plot
    fig1 = plt.figure(figsize=(10, 4.5))
    ax1 = fig1.add_subplot(111)

    # ax1.plot(forces_symm['time'],0.1*forces_symm['drag'], 'bo-',
    #          label='Drag (Symmetric)', markersize=0.1, linewidth=0.1,
    #          markerfacecolor='none')
    # ax1.plot(forces_lower['time'], forces_lower['drag'], 'ro-',
    #          label='Drag (Lower Asymm)', markersize=0.1, linewidth=0.1,
    #          markerfacecolor='none')
    # ax1.plot(forces_upper['time'], forces_upper['drag'], 'ro-',
    #         label='Drag (Upper Asymm)', markersize=0.1, linewidth=0.1,
    #         markerfacecolor='none')

    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Drag Force')
    # ax1.legend(frameon=True, fancybox=False, edgecolor='black',
    #           handletextpad=0.4, handlelength=1.5)
    # ax1.grid(True, which="both", ls="-", alpha=0.2)
    # ax1.set_box_aspect(0.4)
    # plt.savefig('drag_force_plot_comparison_Re_100_neg2.pdf', bbox_inches='tight',
    #             pad_inches=0.02)

    # Create lift force plot
    fig2 = plt.figure(figsize=(10, 4.5))
    ax2 = fig2.add_subplot(111)

    ax2.plot(forces_symm['time'], 0.1*forces_symm['lift'], 'bs-',
             label='Lift ', markersize=0.1, linewidth=0.1,
             markerfacecolor='none')
    # ax2.plot(forces_lower['time'], forces_lower['lift'], 'rs-',
    #          label='Lift (Lower Asymm)', markersize=0.1, linewidth=0.1,
    #          markerfacecolor='none')
    # ax2.plot(forces_upper['time'], forces_upper['lift'], 'ko-',
    #     label='Lift(Upper Asymm)', markersize=0.1, linewidth=0.1,
    #     markerfacecolor='none')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Lift Force')
    ax2.legend(frameon=True, fancybox=False, edgecolor='black',
              handletextpad=0.4, handlelength=1.5)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_box_aspect(0.4)
    plt.savefig('lift_force_plot_comparison_Re_100_25.pdf', bbox_inches='tight',
                pad_inches=0.02)

    plt.show()


def plot_psd():
    # Read the CSV file
    forces_symm = pd.read_csv('../navier_stokes/study_results_symm/forces/forces_unsteady_bdf3.csv')
        
    # Extract time and lift data
    time = forces_symm['time']
    lift = 0.1 * forces_symm['lift']
    
    # Filter data for time between 1000 and 1500
    mask = (time >= 500) & (time <= 1000)
    time_filtered = time[mask]
    lift_filtered = lift[mask]
    
    # Calculate sampling frequency
    dt = time_filtered.iloc[1] - time_filtered.iloc[0]  # time step
    fs = 1/dt  # sampling frequency
    
    # Get number of available points
    n_points = len(lift_filtered)
    print(f"Number of data points: {n_points}")
    
    # Use larger nperseg for better frequency resolution
    nperseg = 4096  # Using larger window since we have more data points
    noverlap = nperseg // 2  # 50% overlap
    
    print(f"Using nperseg = {nperseg}, noverlap = {noverlap}")
    
    # Calculate PSD using Welch's method with modified parameters
    frequencies, psd = signal.welch(lift_filtered, 
                                  fs=fs,
                                  window='hann',
                                  nperseg=nperseg,
                                  noverlap=noverlap,
                                  scaling='density',
                                  detrend='linear')
    
    # Filter frequencies between 0 and 1 Hz
    freq_mask = (frequencies >= 0) & (frequencies <= 1)
    frequencies = frequencies[freq_mask]
    psd = psd[freq_mask]
    
    # Create PSD plot
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    
    # Plot PSD with log scales and thicker line
    ax.semilogy(frequencies, psd, 'b-', linewidth=1.5,
                label='Lift Coefficient PSD')
    
    # Set x-axis limits explicitly
    ax.set_xlim(0, 1)
    
    # Set labels and title
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD ($(C_L)^2$/Hz)')
    ax.set_title('Power Spectral Density of Lift Coefficient (t = 1000-1500)')
    
    # Customize grid and legend
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(frameon=True, fancybox=False, edgecolor='black',
             handletextpad=0.4, handlelength=1.5)
    
    # Set aspect ratio
    ax.set_box_aspect(0.4)
    
    # Improved peak detection with more stringent criteria
    peak_indices = signal.find_peaks(psd, 
                                   height=np.max(psd)*0.2,
                                   distance=20,
                                   prominence=np.max(psd)*0.1
                                   )[0]
    
    # Annotate peaks
    for peak in peak_indices:
        freq = frequencies[peak]
        power = psd[peak]
        ax.annotate(f'{freq:.3f} Hz', 
                   xy=(freq, power),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=8,
                   arrowprops=dict(arrowstyle='->'))
        
        # Print the peak frequency and its power
        print(f"Peak frequency: {freq:.3f} Hz, Power: {power:.2e}")
    
    # Save the plot
    plt.savefig('lift_coefficient_psd_t1000-1500_f0-1_sharp.pdf', 
                bbox_inches='tight',
                pad_inches=0.02)
    plt.show()

    # Print dominant frequencies
    print("\nDominant frequencies between 0-1 Hz:")
    for peak in peak_indices:
        print(f"f = {frequencies[peak]:.3f} Hz")

def plot_lift_psd():
    # Read data
    forces_symm = pd.read_csv('../navier_stokes/study_results_symm/forces/forces_unsteady_bdf3.csv')
    time = forces_symm['time']
    cl = 0.1 * forces_symm['lift']  # Apply same scaling as time series plot
    mask = (time >= 500) & (time <= 1000)
    time_filtered = time[mask]
    lift_filtered = cl[mask]
    # Calculate sampling parameters
    dt = 0.1
    
    fs = 1/dt  # Sampling frequency

    # Compute PSD using Welch's method
    nperseg = 4096  # Increase window length
    noverlap = nperseg // 4  # Reduce overlap
    nfft = 8192  # Increase FFT points
    window = 'blackman'  # Use Blackman window
    # f, Pxx = signal.welch(lift_filtered, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft) 
    f, Pxx = signal.welch(cl, fs=fs, nperseg=1024, scaling='density')   
    # freq_mask = (f >= 0) & (f <= 1)
    # f = f[freq_mask]
    # Pxx = Pxx[freq_mask]
    # Create plot
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    
    # Plot PSD in log-linear format (common for spectral analysis)
    ax.semilogy(f, Pxx, 'b-', linewidth=0.8)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density [$C_L^2$/Hz]')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_xlim([0, 1])  # Show up to Nyquist frequency
    ax.set_box_aspect(0.4)
    
    plt.savefig('lift_psd_plot.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()
# Generate the plots
# print("Generating forces comparison plots...")
plot_forces()
# print("Generating PSD plot of lift coefficient...")
# plot_lift_psd()