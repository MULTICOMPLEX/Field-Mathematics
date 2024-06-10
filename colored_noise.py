import colorednoise as cn
from matplotlib import mlab
from matplotlib import pylab as plt
import numpy as np

beta = 1 # the exponent
samples = 2**18 # number of samples to generate

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

beta = -1 # the exponent
y = cn.powerlaw_psd_gaussian(beta, samples)

plt.figure(figsize=(10, 6))
plt.plot(initial_n_bins,  y, label='beta = 2 = red noise')
plt.legend()
plt.grid(True)


# optionally plot the Power Spectral Density with Matplotlib
plt.figure(figsize=(10, 6))
s, f = mlab.psd(y, NFFT=2**13)
plt.loglog(f,s)
plt.grid(True)


plt.show()