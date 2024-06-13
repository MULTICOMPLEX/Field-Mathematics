from matplotlib import mlab
from matplotlib import pylab as plt
import numpy as np

from numpy.fft import irfft, rfftfreq
from numpy.random import default_rng, Generator, RandomState

import math


def powerlaw_psd_gaussian(
        exponent, 
        samples
    ):
   
    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples) 
    
    # Build scaling factors for all frequencies
    s_scale = f
    s_scale[0] = 1
    s_scale = np.power(f, -exponent / 2.0)
    s_scale[0] = 0
    
    # Calculate theoretical output standard deviation from scaling
    sigma = 2 * np.sqrt(np.sum(s_scale**2)) / samples
      
    # prepare random number generator    
    rng = np.random.default_rng()
    
    # Generate scaled random power + phase
    v = np.sqrt(np.pi)
    sr = rng.uniform(-v, v, size=len(f))  # Independent standard uniform variables
    si = rng.uniform(-v, v, size=len(f))   # Independent standard uniform variables
    sr *= s_scale
    si *= s_scale   
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
     
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples) / sigma
    
    return y


def powerlaw_psd_gaussian_normal(exponent, samples):   
    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples) 
    
    print(f)
    
    # Build scaling factors for all frequencies
    s_scale = f
    s_scale = np.power(f[1::], -exponent / 2.0)
    s_scale[0] = 0
    
    # Calculate theoretical output standard deviation from scaling
    sigma = 2 * np.sqrt(np.sum(s_scale**2)) / samples
      
    # prepare random number generator    
    rng = np.random.default_rng()
    
    # Generate scaled random power + phase

    sr = rng.normal(size=len(f))  # Independent standard uniform variables
    si = rng.normal(size=len(f))   # Independent standard uniform variables
    sr *= s_scale
    si *= s_scale   
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
     
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples) / sigma
    
    return y


beta1 = 2 # the exponent
beta2 = 2 # the exponent

samples = 2**21# number of samples to generate
return_to_beginning = 1

if(return_to_beginning == 0):
    return_to_beginning = 2;

initial_n_bins = np.linspace(0, samples, int(samples/return_to_beginning)) 

y = powerlaw_psd_gaussian(beta1, samples)[:int(samples/return_to_beginning)]
plt.figure(figsize=(10, 6))
label = " (1/f)$\\beta$="
label += str(beta1)
plt.plot(initial_n_bins,  y, label=label)
plt.legend()
plt.grid(True)
# optionally plot the Power Spectral Density with Matplotlib
plt.figure(figsize=(10, 6))
s, f = mlab.psd(y, NFFT=2**13)
plt.loglog(f,s)
plt.title("FFT Colored Noise, (1/f)$\\beta$=" + str(beta1))
plt.grid(True)

y2 = powerlaw_psd_gaussian(beta2, samples)[:int(samples/return_to_beginning)]
plt.figure(figsize=(10, 6))
label = "(1/f)$\\beta$="
label += str(beta2)
plt.plot(initial_n_bins,  y2, label=label)
plt.legend()
plt.grid(True)
# optionally plot the Power Spectral Density with Matplotlib
plt.figure(figsize=(10, 6))
s, f = mlab.psd(y2, NFFT=2**13)
plt.loglog(f,s)
plt.title("FFT Colored Noise Normal, (1/f)$\\beta$=" + str(beta2))
plt.grid(True)



# Different colors
start_color = 'blue'  
end_color = 'orange'
path_color = 'gray'

plt.figure(figsize=(8, 8))

# Plot the path 
n = 512
plt.plot(y[::n], y2[::n], marker='o', markersize=2, linestyle='-', linewidth=0.5, color=path_color)  # Gray for path

# Plot the start point (green)
plt.plot(y[0], y2[0], marker='o', markersize=8, color=start_color, label='Start')

# Plot the end point (red)
plt.plot(y[-1], y2[-1], marker='o', markersize=8, color=end_color, label='End')

plt.title("Simulated 2D Colored Noise Motion "+ "(1/f)$\\beta$=" + str(beta1) + ", (1/f)$\\beta$=" + str(beta2) )
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()  # Show the legend for start/end points
plt.grid(alpha=0.4)
plt.axis('equal')


exponent = math.log2(samples)
text1 = "Number of samples " + f"{samples} = 2^{int(exponent)}"


def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

dif = distance(y[0],y2[0],y[-1],y2[-1])
text2 = "Δ Start-End              " + f"{dif:.6e}"

ax = plt.gca()
plt.text(0.015, 0.97, text1, fontsize=9, transform=ax.transAxes)
plt.text(0.015, 0.95, text2, fontsize=9, transform=ax.transAxes)

plt.legend(loc='upper right') 

print(text1)
print(text2)

plt.show()

