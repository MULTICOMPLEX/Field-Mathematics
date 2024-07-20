import sys
import os
from matplotlib import pylab as plt
from matplotlib import mlab
from matplotlib.ticker import ScalarFormatter
import numpy as np
import time
import phimagic_prng32
import phimagic_prng64


def sine_function(len, f):
    x = np.linspace(0, 2 *  np.pi, len) 
    return np.sin(x * f)
    
def func_approx(x, n, band = True):
    # Perform the Fourier Transform
    yf = np.fft.fft(x)
    # Truncate higher frequencies (approximation)
    num_components = int(n)# Adjust this to control the level of approximation
    yf_truncated = yf
    if(band == False):
        yf_truncated = np.zeros(len(yf), dtype=np.complex128)
        yf_truncated[num_components] = yf[num_components]
        yf_truncated[-num_components] = yf[-num_components]
    else:
        yf_truncated[num_components:-num_components] = 0
    # Perform the Inverse Fourier Transform to get the approximated function
    y_approx = np.fft.ifft(yf_truncated)
    return y_approx.real

def normalize_signal_to_range(signal, a, b):
    """Normalize a signal to the range [a, b]."""
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    # Scale to [0, 1]
    normalized_signal = (signal - min_val) / (max_val - min_val)
    
    # Scale to [a, b]
    normalized_signal = (b - a) * normalized_signal + a
    
    return normalized_signal

def calculate_snr(signal, noise):
    """
    Calculate the Signal-to-Noise Ratio (SNR).

    Parameters:
    signal (numpy.ndarray): The original signal array.
    noise (numpy.ndarray): The noise array.

    Returns:
    float: The SNR value in dB.
    """
    # Ensure the signal and noise arrays are numpy arrays
    
    # Calculate the power of the signal and noise
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate the SNR
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr
    
class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

# Create an instance of the custom PRNG
prng = phimagic_prng32.mxws()


Nbins = 2000
Ntrials = 100000
N_Integrations = 100
Ncycles1 = 17
Ncycles2 = 217
#Ncycles3 = 217
analytical = 0

if(analytical):
   Ntrials = 1 

#Time seed 
current_time_seconds = int(time.time())    

with Timer() as t:        
        s1, p1 = prng.sine(enable_seed = 1,  Seed = current_time_seconds, Ntrials = Ntrials, Ncycles = Ncycles1,  N_Integrations = N_Integrations,   Nbins = Nbins, Icycles = True)
        s2, p2 = prng.sine(enable_seed = 1,  Seed = current_time_seconds+1, Ntrials = Ntrials, Ncycles = Ncycles2,  N_Integrations = N_Integrations,   Nbins = Nbins, Icycles = True)
        #s3, p3 = prng.sine(enable_seed = 1,  Seed = current_time_seconds+1, Ntrials = Ntrials, Ncycles = Ncycles3,  N_Integrations = N_Integrations,   Nbins = Nbins, Icycles = False)

print(f"Elapsed time: {t.elapsed_time:.5g}", "\n")

fa1 = func_approx(s1, Ncycles1, False)
fa2 = func_approx(s2, Ncycles2, False)

s1 = normalize_signal_to_range(s1, 0, 1)
s2 = normalize_signal_to_range(s2, 0, 1)
s3 = np.zeros(len(s2)-len(s1)) 
s1 = np.concatenate((s1, s3), axis=None)

if(analytical ==True):
    s1 = sine_function(Nbins, Ncycles1)
    s2 = sine_function(Nbins, Ncycles2)
    s1 = normalize_signal_to_range(s1, 0, 1)
    s2 = normalize_signal_to_range(s2, 0, 1)


n1 = normalize_signal_to_range(s1, -1, 1)
n2 = normalize_signal_to_range(s2, -1, 1)
snr_db = calculate_snr(sine_function(len(n1), Ncycles1), n1)
print(f'SNR1: {snr_db}')
power_ratio = 10 ** (snr_db / 10)
print(f'POW1: {power_ratio}')
snr_db = calculate_snr(sine_function(len(n2), Ncycles2), n2)
print(f'SNR2: {snr_db}')
power_ratio = 10 ** (snr_db / 10)
print(f'POW2: {power_ratio}')

"""
s1 = sine_function(len(s1), Ncycles1)
s2 = sine_function(len(s2), Ncycles2)
s1 = normalize_signal_to_range(s1, s1, 0, 1)
s2 = normalize_signal_to_range(s2, s2, 0, 1)
s1 += prng.uniform(0.0, 0.5, len(s1))
s2 += prng.uniform(0.0, 0.5, len(s2))
"""

s3 = s1 *  np.sqrt(s2) 

#s3 = np.concatenate((s1, s2), axis=None)



def set_axis_color(ax):
    ax.set_facecolor('#002b36')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(which = 'major', colors='white')
    ax.tick_params(which = 'minor', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white') 

fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)
x = np.linspace(0,1,  len(s3)) 
plt.plot(x, s3, label = str(int((Ntrials * N_Integrations * p1[1] * Ncycles1) + (Ntrials * N_Integrations * p2[1] * Ncycles2)))+ " Trials") 
plt.title("Sine Distribution " + str(Ncycles1)+ " Hz * sqrt(" +str(Ncycles2)+ ") Hz", color = 'white')
plt.xlabel("Time", color = 'white')
plt.ylabel("Y", color = 'white')
plt.grid(alpha=0.4)
plt.legend()


fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)
x = np.linspace(0,1,  len(s1)) 
plt.plot(x, s1, label = str(int((Ntrials * N_Integrations * p1[1] * Ncycles1)))+ " Trials") 
plt.title("Sine Distribution1: " + str(Ncycles1)+ " Hz", color = 'white')
plt.xlabel("Time", color = 'white')
plt.ylabel("Y", color = 'white')
plt.grid(alpha=0.4)
plt.legend()

fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)
x = np.linspace(0,1,  len(s2)) 
plt.plot(x, s2, label = str(int((Ntrials * N_Integrations * p2[1] * Ncycles1)))+ " Trials") 
plt.title("Sine Distribution2: " + str(Ncycles2)+ " Hz", color = 'white')
plt.xlabel("Time", color = 'white')
plt.ylabel("Y", color = 'white')
plt.grid(alpha=0.4)
plt.legend()

fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)

s, f = mlab.psd(s3, NFFT=len(s3))

plt.loglog(f * len(f), s)
formatter = ScalarFormatter()
formatter.set_useOffset(False)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlim(right = len(f) * 1.2)
plt.grid(True, which='both', alpha = 0.4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (Unit**2/Hz)')

if(analytical ==False):
    plt.title("FFT Distribution", color='white')
else:
    plt.title("FFT Signal", color='white')
    
plt.grid(True)

plt.show()


