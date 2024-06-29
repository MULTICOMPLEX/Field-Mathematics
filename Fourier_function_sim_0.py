import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt


def powerlaw_psd_gaussian(beta, samples, fmin, f, sr, si):
       
        # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples) # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")
    
    # Build scaling factors for all frequencies
    s_scale = f    
    ix   = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-beta/2.)
      
    sr *= s_scale
    si *= s_scale   
    
    si[0] = 0
    sr[0] *= np.sqrt(2)    # Fix magnitude
    
    # Calculate theoretical output standard deviation from scaling
    sigma = 2 * np.sqrt(np.sum(s_scale**2)) / samples
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
     
    # Transform to real time series & scale to unit variance
    y = np.fft.irfft(s, n=samples) / sigma
    
    return y
    

# Define the original function (square wave in this example)
def func2(x, period=2*np.pi, duty_cycle=0.5):
    return 0.5 + 0.5 * np.sign(np.sin(x / period * 2 * np.pi))

def func1(x, period=2*np.pi, duty_cycle=0.5):
    return np.sin(x) + np.cos(2*x)

def func3(x, period=2*np.pi, duty_cycle=0.5):
    return np.sinh(np.arctan(np.sin(2*X)+np.cos(5*X))) 


n = 2000

beta1 = 1 # the exponent
fmin = 0.0;  
f = np.fft.rfftfreq(n)   
v = np.sqrt(np.pi)
rng = np.random.default_rng()
sr1 = rng.uniform(-v, v, size=len(f))  # Independent standard uniform variables
si1 = rng.uniform(-v, v, size=len(f))   # Independent standard uniform variables
y = powerlaw_psd_gaussian(beta1, n, fmin, f, sr1, si1)

# Generate data points
x = np.linspace(-np.pi, np.pi, n)

# Perform the Fourier Transform
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*x[1]-x[0]), n//2)

# Truncate higher frequencies (approximation)
num_components = int(n/30)# Adjust this to control the level of approximation
yf_truncated = yf.copy()
yf_truncated[num_components:-num_components] = 0

# Perform the Inverse Fourier Transform to get the approximated function
y_approx = ifft(yf_truncated)


# Plot the results
fig = plt.figure(facecolor='#002b36', figsize = (10, 6))
ax = fig.gca()
ax.set_facecolor('#002b36')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(colors='white')
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white') 

plt.plot(x, y, label='Original function', linewidth=0.8)
plt.plot(x, y_approx.real, label=f'Approximation ({num_components} Components)')

plt.title('Function Approximation using Fourier Transform', color = 'white')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()
