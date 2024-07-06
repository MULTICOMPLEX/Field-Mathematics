import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.ticker import ScalarFormatter
import phimagic_prng32

prng = phimagic_prng32.mxws(2)


def normalize_signal(signal):
    """Normalize a signal to the range [-1, 1]."""
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    # Scale to [0, 1]
    normalized_signal = (signal - min_val) / (max_val - min_val)
    
    # Scale to [-1, 1]
    normalized_signal = 2 * normalized_signal - 1
    
    return normalized_signal
    
def compute_twiddle_factors(N):
    """Compute the sine values for the twiddle factors and derive cosine values by rotating the sine array."""
    theta = -2 * np.pi * np.arange(N) / N
    sin_vals = np.sin(theta)
    cos_vals = np.roll(sin_vals, N // 4)  # Rotate by 90 degrees to get cosine values
    return cos_vals, sin_vals

def fft_recursive(x, cos_vals, sin_vals):
    """Compute the FFT of a sequence x using a recursive algorithm with precomputed sine and cosine values."""
    N = x.shape[0]
    
    if N <= 1:
        return x
    
    even = fft_recursive(x[0::2], cos_vals[::2], sin_vals[::2])
    odd = fft_recursive(x[1::2], cos_vals[::2], sin_vals[::2])
    
    combined = np.zeros(N, dtype=complex)
    
    for k in range(N // 2):
        t = complex(cos_vals[k] * odd[k].real - sin_vals[k] * odd[k].imag,
                    cos_vals[k] * odd[k].imag + sin_vals[k] * odd[k].real)
        combined[k] = even[k] + t
        combined[k + N // 2] = even[k] - t
    
    return combined

def func_approx(x, n):
    # Perform the Fourier Transform
    yf = np.fft.fft(x)
    # Truncate higher frequencies (approximation)
    num_components = int(n)# Adjust this to control the level of approximation
    yf_truncated = yf
    #yf_truncated = np.zeros(len(yf), dtype=np.complex128)
    #yf_truncated[num_components] = yf[num_components]
    #yf_truncated[-num_components] = yf[-num_components]
    yf_truncated[num_components:-num_components] = 0
    # Perform the Inverse Fourier Transform to get the approximated function
    y_approx = np.fft.ifft(yf_truncated)
    return y_approx.real

# Define desired frequencies
frequencies = 50  # Frequencies in Hz

# Sampling rate (choose a rate appropriate for your frequencies)
sampling_rate = 2**12  # Samples per second

# Calculate number of samples based on desired duration (adjust as needed)
duration = 1  # Seconds
num_samples = int(sampling_rate * duration)

spectrum = np.zeros(num_samples)
spectrum[frequencies] = 1

# Precompute the twiddle factors
#cos_vals, sin_vals = compute_twiddle_factors(num_samples)
sin_vals  = prng.sine_distribution(Enable_Random_Seed = 1,  Seed = 10, Ntrials =1000, Ncycles = frequencies,  N_Integrations = 10,   Nbins = num_samples)
sin_vals = normalize_signal(sin_vals)

# Find the index where the sine wave crosses zero near the beginning
zero_crossings = np.where(np.diff(np.sign(sin_vals)))[0]
first_zero_crossing = zero_crossings[0]
sin_vals= np.roll(sin_vals, -first_zero_crossing)

#cos_vals = -np.roll(sin_vals, num_samples // 4)  # Rotate by 90 degrees to get cosine values

# Compute the FFT using the recursive function with precomputed twiddle factors
#signal = np.real(1j * fft_recursive(spectrum, cos_vals, sin_vals))

signal = func_approx(sin_vals, len(sin_vals)/2)
signal = normalize_signal(signal)

#signal =  sin_vals 
#signal = np.real(1j * np.fft.fft(spectrum))

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
plt.title("Originald Signal", color = 'white')
t = np.linspace(0, duration, len(sin_vals))
plt.plot(t, sin_vals)
ax = fig.gca()
set_axis_color(ax)
plt.grid(True, which='both', alpha = 0.4)
plt.xlabel("Time (s)")
plt.ylabel("Signal")


fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
plt.title("FFT Originald Signal", color='white')
s, f = mlab.psd(sin_vals, NFFT=len(sin_vals))

plt.loglog(f * len(f), s)
formatter = ScalarFormatter()
formatter.set_useOffset(False)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlim(right = len(f) * 1.2)
ax = fig.gca()
set_axis_color(ax)
plt.grid(True, which='both', alpha = 0.4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (Unit**2/Hz)')

# Print or plot the generated signal
fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
plt.title("Generated Signal", color = 'white')
t = np.linspace(0, duration, len(signal))
plt.plot(t, signal)
ax = fig.gca()
set_axis_color(ax)
plt.grid(True, which='both', alpha = 0.4)
plt.xlabel("Time (s)")
plt.ylabel("Signal")


fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
plt.title("FFT Generated Signal", color='white')
s, f = mlab.psd(signal, NFFT=len(signal))

plt.loglog(f * len(f), s)
formatter = ScalarFormatter()
formatter.set_useOffset(False)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlim(right = len(f) * 1.2)
ax = fig.gca()
set_axis_color(ax)
plt.grid(True, which='both', alpha = 0.4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (Unit**2/Hz)')


plt.show()
