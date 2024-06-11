import colorednoise as cn
from matplotlib import mlab
from matplotlib import pylab as plt
import numpy as np

beta = 1 # the exponent
samples = 2**13 # number of samples to generate

initial_n_bins = np.linspace(0, samples, samples) 

y = cn.powerlaw_psd_gaussian(beta, samples)


plt.figure(figsize=(10, 6))
plt.plot(initial_n_bins,  y, label='beta =1 = pink noise')
plt.legend()
plt.grid(True)


# optionally plot the Power Spectral Density with Matplotlib
plt.figure(figsize=(10, 6))
s, f = mlab.psd(y, NFFT=2**13)
plt.loglog(f,s)
plt.grid(True)

beta = 2 # the exponent
y = cn.powerlaw_psd_gaussian(beta, samples)

plt.figure(figsize=(10, 6))
plt.plot(initial_n_bins,  y, label='beta = 2 = red noise')
plt.legend()
plt.grid(True)


# optionally plot the Power Spectral Density with Matplotlib
plt.figure(figsize=(10, 6))
s, f = mlab.psd(y, NFFT=2**13)
plt.loglog(f,s)
plt.title("Colored Noise FFT")
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

t, B_t = simulate_brownian_motion(num_terms=samples,  spread = 0.001, seed = None)

plt.figure(figsize=(10, 6))
s, f = mlab.psd(B_t, NFFT=2**13)
plt.loglog(f,s)
plt.title("Brownian motion using a random Fourier series.")
plt.grid(True)

plt.show()