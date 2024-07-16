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

def normalize_signal_to_range(signal, signal2, a, b):
    """Normalize a signal to the range [a, b]."""
    min_val = np.min(signal2)
    max_val = np.max(signal2)
    
    # Scale to [0, 1]
    normalized_signal = (signal - min_val) / (max_val - min_val)
    
    # Scale to [a, b]
    normalized_signal = (b - a) * normalized_signal + a
    
    return normalized_signal
    
class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

# Create an instance of the custom PRNG
prng = phimagic_prng32.mxws()

# Generate random numbers
for _ in range(0):
    print(prng.rng())
    
 
 # Generate an array of random numbers
size = 100

v1 = np.sqrt(np.pi)
random_array = prng.uniform(-v1, v1, size=size)

# Print the array
#print("Generated random numbers:", random_array)

Nbins = 2000
Ntrials = 10000
Ncycles1 = 17
Ncycles2 = 217

start_time = time.perf_counter()

#Time seed 
current_time_seconds = int(time.time())   

with Timer() as t:        
    s1 = prng.sine(enable_seed = 1,  Seed = current_time_seconds, Ntrials = Ntrials, Ncycles = Ncycles1,  N_Integrations = 10,   Nbins = Nbins, Icycles = False)
    s2 = prng.sine(enable_seed = 1,  Seed = current_time_seconds, Ntrials = Ntrials, Ncycles = Ncycles2,  N_Integrations = 10,   Nbins = Nbins, Icycles = False)


print(f"Elapsed time: {t.elapsed_time:.5g}")

fa1 = func_approx(s1, Ncycles1, False)
fa2 = func_approx(s2, Ncycles2, False)
s1 = normalize_signal_to_range(s1, s1, 0, 1)
s2 = normalize_signal_to_range(s2, s2, 0, 1)

"""
s1 = sine_function(len(s1), Ncycles1)
s2 = sine_function(len(s2), Ncycles2)
s1 = normalize_signal_to_range(s1, s1, 0, 1)
s2 = normalize_signal_to_range(s2, s2, 0, 1)
s1 += prng.uniform(0.0, 0.5, len(s1))
s2 += prng.uniform(0.0, 0.5, len(s2))
"""

s3 = s1 * s2

#s3 = np.concatenate((s1, s2), axis=None)

s3 /= s3.sum()

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

plt.plot(x, s1) 
plt.title("Sine Distribution " + str(Ncycles1)+ " * " +str(Ncycles2)+ " Hz", color = 'white')
plt.xlabel("Time", color = 'white')
plt.ylabel("Y", color = 'white')
plt.grid(alpha=0.4)


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

plt.title("FFT Distribution", color='white')
plt.grid(True)

plt.show()


