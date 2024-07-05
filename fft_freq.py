import numpy as np
import matplotlib.pyplot as plt

# Define desired frequencies
frequencies = np.array([0, 5])  # Frequencies in Hz

# Sampling rate (choose a rate appropriate for your frequencies)
sampling_rate = 10000  # Samples per second

# Calculate number of samples based on desired duration (adjust as needed)
duration = 1  # Seconds
num_samples = int(sampling_rate * duration)

# Create time axis
t = np.linspace(0, duration, num_samples)

# Generate ideal frequency domain representation (replace with your spectrum)
# Here, we create impulses at the desired frequencies with amplitudes of 1
spectrum = np.zeros(num_samples)
spectrum[np.where(np.isin(np.fft.fftfreq(num_samples, d=1/sampling_rate), frequencies))] = 1

# Take real part for actual signal (if needed)
signal = np.real(1j * np.fft.fft(spectrum))

def set_axis_color(ax):
    ax.set_facecolor('#002b36')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white') 

# Print or plot the generated signal
fig = plt.figure(facecolor='#002b36', figsize=(10, 6))

plt.plot(t, signal)
ax = fig.gca()
set_axis_color(ax)

plt.xlabel("Time (s)")
plt.ylabel("Signal")
plt.title("Generated Signal", color = 'white')
plt.show()
