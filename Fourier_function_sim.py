import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve, roc_auc_score, precision_recall_curve


def powerlaw_psd_gaussian(beta, samples, fmin, f_in, sr_in, si_in):
       
        # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples) # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")
    
    # Build scaling factors for all frequencies
    s_scale = f_in   
    ix   = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-beta/2.)
     
    sr = np.array(len(sr_in))    
    si = np.array(len(si_in))    
    
    sr = sr_in.copy()
    si = si_in.copy()
    
    sr *= s_scale
    si *= s_scale   
     
    # Calculate theoretical output standard deviation from scaling
    sigma = 2 * np.sqrt(np.sum(s_scale**2)) / samples
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
     
    # Transform to real time series & scale to unit variance
    y = np.fft.irfft(s, n=samples) / sigma
    
    return y
    

# Define the original function (square wave in this example)
def func2(x, period=2*np.pi,):
    return 0.5 + 0.5 * np.sign(np.sin(x / period * 2 * np.pi))

def func1(x):
    return np.sin(x) + np.cos(2*x)

def func3(x):
    return np.sinh(np.arctan(np.sin(2*x)+np.cos(5*x))) 

def func4(x):
    return x * 0 

def func_approx(x, n):
    # Perform the Fourier Transform
    yf = fft(x)
    # Truncate higher frequencies (approximation)
    num_components = int(n)# Adjust this to control the level of approximation
    yf_truncated = yf
    yf_truncated[num_components:-num_components] = 0
    # Perform the Inverse Fourier Transform to get the approximated function
    y_approx = ifft(yf_truncated)
    return y_approx

n = 20000

beta1 = 2 # the exponent
beta2 = 2 # the exponent
fmin = 0.0;  
num_components = n//512

f = np.fft.rfftfreq(n)   
v1 = np.sqrt(np.pi)
rng = np.random.default_rng()

sr1 = rng.uniform(-v1, v1, size=len(f))  # Independent standard uniform variables
si1 = rng.uniform(-v1, v1, size=len(f))   # Independent standard uniform variables
y1 = powerlaw_psd_gaussian(beta1, n, fmin, f, sr1, si1)
y_approx1 = func_approx(y1, num_components)

# Generate data points
#x = np.linspace(-np.pi, np.pi, len(f))
#sr1 = func3(x)
#si1 = func4(x)
#y = powerlaw_psd_gaussian(beta1, n, fmin, f, sr1, si1)

v2 = np.sqrt(np.pi) / 5
sr2 = rng.uniform(-v2, v2, size=len(f))  # Independent standard uniform variables
si2 = rng.uniform(-v2, v2, size=len(f))   # Independent standard uniform variables
y2 = powerlaw_psd_gaussian(beta2, n, fmin, f, sr1+sr2, si1+si2)
y_approx2 = func_approx(y2, num_components)

print("Δ Start-End              ", np.abs(y_approx1[0] - y_approx1[-1]))
#y_approx = fft(y_approx)

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

x = np.linspace(-np.pi, np.pi, n)
plt.plot(x, y1, label='Original function1', linewidth=0.8)
plt.plot(x, y2, label='Original function2', linewidth=0.8)
plt.plot(x, y_approx1.real, label=f'Approximation1 ({num_components} Components)')
plt.plot(x, y_approx2.real, label=f'Approximation2 ({num_components} Components)')

plt.title('Function Approximation using Fourier Transform', color = 'white')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

 
# Metrics
mse = mean_squared_error(y_approx1.real,  y_approx2.real)
mae = mean_absolute_error(y_approx1.real,  y_approx2.real)
r2 = r2_score(y_approx1.real,  y_approx2.real)
print(f'MSE: {mse:.10f}')
print(f'MAE: {mae:.10f}')
print(f'R^2: {r2:.10f}')



plt.show()


