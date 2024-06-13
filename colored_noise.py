from matplotlib import mlab
from matplotlib import pylab as plt
import numpy as np

from typing import Union, Iterable, Optional
from numpy import sqrt, newaxis, integer
from numpy.fft import irfft, rfftfreq
from numpy.random import default_rng, Generator, RandomState
from numpy import sum as npsum

import math


def powerlaw_psd_gaussian(
        exponent: float, 
        samples:int, 
        fmin: float = 0.0
    ):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    samples: int 
        The number of samples in each time series

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. 
        
        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.

    Returns
    -------
    out : array
        The samples.


    Examples:
    ---------

    # generate 1/f noise == pink noise == flicker noise
    >>> y = powerlaw_psd_gaussian(1, 5)
    """
   
    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples) 
    
    # Build scaling factors for all frequencies
    s_scale = f 
    s_scale[0] = 1
    s_scale = np.power(f, -exponent / 2.0)
    s_scale[0] = 0
    
    # Calculate theoretical output standard deviation from scaling
    sigma = 2 * sqrt(npsum(s_scale**2)) / samples
      
    # prepare random number generator    
    rng = np.random.default_rng()
    
    # Generate scaled random power + phase
    sr = rng.normal(size=len(f))  # Independent standard normal variables
    si = rng.normal(size=len(f))  # Independent standard normal variables
    sr *= s_scale
    si *= s_scale
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
    
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples) / sigma
    
    return y


beta1 = 2 # the exponent
beta2 = 2 # the exponent

samples = 2**22# number of samples to generate
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
plt.title("FFT Colored Noise, (1/f)$\\beta$=" + str(beta2))
plt.grid(True)

def simulate_brownian_motion(num_terms=1000, spread = 0.001, seed = 10):
    """Simulates Brownian motion using a random Fourier series.

    Args:
        num_terms (int): Number of terms in the Fourier series.
        spread (float): difference between the start and end point.
        seed (int): seed for the RNG. 

    Returns:
        numpy.ndarray: Time points and corresponding Brownian motion values.
    """
    t = np.linspace(0, 2 * np.pi, num_terms)
    
    rng = np.random.default_rng(seed)
    xi = rng.uniform(-np.sqrt(np.pi), np.sqrt(np.pi), num_terms)  # Independent standard uniform variables

    B_t = xi[0] * t / np.sqrt(2 * np.pi) * spread
    B_t += sum(np.sin(n * t / 2) * xi[n] / n for n in range(1, num_terms)) * 2 / np.sqrt(np.pi)

    return t, B_t 

#t, B_t = simulate_brownian_motion(num_terms=8192,  spread = 0.0001, seed = None)

#plt.figure(figsize=(10, 6))
#s, f = mlab.psd(B_t, NFFT=2**13)
#plt.loglog(f,s)
#plt.title("Brownian motion using a random Fourier series.")
#plt.grid(True)


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

