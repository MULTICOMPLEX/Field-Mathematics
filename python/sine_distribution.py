import sys
import os
from matplotlib import pylab as plt
from matplotlib import mlab
from matplotlib.ticker import ScalarFormatter
import numpy as np
import time
import phimagic_prng32
import phimagic_prng64


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
Ntrials = 100000
Ncycles1 = 20
Ncycles2 = 217

#Time seed 
current_time_seconds = int(time.time())           
s1 = prng.sine(enable_seed = 1,  Seed = current_time_seconds, Ntrials = Ntrials, Ncycles = Ncycles1,  N_Integrations = 10,   Nbins = Nbins, Icycles = False)
s2 = prng.sine(enable_seed = 1,  Seed = current_time_seconds, Ntrials = Ntrials, Ncycles = Ncycles2,  N_Integrations = 10,   Nbins = Nbins, Icycles = False)

s3 = s1 * s2
s3 /= s3.sum()

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

x = np.linspace(0, len(s3),  len(s3)) 

plt.plot(x, s2) 
plt.title("sine distribution", color = 'white')
plt.xlabel("X", color = 'white')
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

plt.title("FFT", color='white')
plt.grid(True)

plt.show()