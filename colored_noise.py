from matplotlib import mlab
from matplotlib import pylab as plt
import numpy as np
from numpy.fft import irfft, rfftfreq
from numpy.random import default_rng, Generator, RandomState
from matplotlib.ticker import ScalarFormatter
import math
from scipy.signal import bilinear
from scipy import signal 
import phimagic_prng32
import phimagic_prng64
import time


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
    y = irfft(s, n=samples) / sigma
    
    return y

def differentiate_once(f, p1):
    f = np.fft.ifft(1j*p1*np.fft.fft(f))
    return f
    
def differentiate_twice(f, p2):
    f = np.fft.ifft(-p2*np.fft.fft(f))
    return f


def func_approx(x, n):
    # Perform the Fourier Transform
    yf = np.fft.fft(x)
    # Truncate higher frequencies (approximation)
    num_components = int(n)# Adjust this to control the level of approximation
    yf_truncated = yf
    yf_truncated[num_components:-num_components] = 0
    # Perform the Inverse Fourier Transform to get the approximated function
    y_approx = np.fft.ifft(yf_truncated)
    return y_approx.real

samples = 2**17 # number of samples to generate
return_to_beginning = 1
beta1 = 2 # the exponent
beta2 = 2# the exponent
fmin = 0.0;
n = 1 #plot every n sample
derivative = 1
normal_input = 0
function_input = 0
Nfreq1 = 5
Nfreq2 = 30

# Calculate Frequencies (we asume a sample rate of one)
# Use fft functions for real output (-> hermitian spectrum)
f = rfftfreq(samples) 

# Generate scaled random power + phase

rng = np.random.default_rng()
v = np.sqrt(np.pi)
#sr1 = rng.uniform(-v, v, size=len(f))  # Independent standard uniform variables
#si1 = rng.uniform(-v, v, size=len(f))   # Independent standard uniform variables
#sr2 = rng.uniform(-v, v, size=len(f))  # Independent standard uniform variables
#si2 = rng.uniform(-v, v, size=len(f))   # Independent standard uniform variables

current_time_seconds = int(time.time())
prng = phimagic_prng32.mxws(current_time_seconds)

sr1 = prng.uniform(-v, v, size=len(f))  # Independent standard uniform variables
si1 = prng.uniform(-v, v, size=len(f))   # Independent standard uniform variables
sr2 = prng.uniform(-v, v, size=len(f))  # Independent standard uniform variables
si2 = prng.uniform(-v, v, size=len(f))   # Independent standard uniform variables


if normal_input:
    sr1 = rng.normal(size=len(f))  # Independent standard uniform variables
    si1 = rng.normal(size=len(f))   # Independent standard uniform variables
    sr2 = rng.normal(size=len(f))  # Independent standard uniform variables
    si2 = rng.normal(size=len(f))   # Independent standard uniform variables


# 2. Custom Function
def my_func1(x):
    return np.sin(x)**2 + 1 * x - 1
        
arr_custom1 = np.fromfunction(my_func1, (len(f),))  # Apply custom function

# 2. Custom Function
def my_func2(x):
    return np.cos(x)**2 + 2 * x - 1

arr_custom2 = np.fromfunction(my_func2, (len(f),))  # Apply custom function

# 2. Custom Function
def my_func3(x):
    return x**2 + 3 * x - 1
        
arr_custom3 = np.fromfunction(my_func3, (len(f),))  # Apply custom function

# 2. Custom Function
def my_func4(x):
    return x**2 + 4 * x - 1
 
def set_axis_color(ax):
    ax.set_facecolor('#002b36')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white') 
 
arr_custom4 = np.fromfunction(my_func4, (len(f),))  # Apply custom function

if function_input:
    sr1 = arr_custom1
    si1 = arr_custom2
    sr2 =  arr_custom3
    si2 = arr_custom4
    
    
#Frequencies for the derivative
x = 2 * np.pi * np.arange(0, samples, 1) / samples#-open-periodic domain    
dx = x[1] - x[0]
p1 = np.fft.fftfreq(samples, d = dx) * 2 * np.pi  #first order
p2 =  (1j * p1)**2         #second order

if(return_to_beginning == 0):
    return_to_beginning = 2;

initial_n_bins = np.linspace(0, samples, int(samples/return_to_beginning)) 

y1 = powerlaw_psd_gaussian(beta1, samples, fmin, f, sr1, si1)[:int(samples/return_to_beginning)]
if derivative:
    y1_dif = differentiate_once(y1, p1).real 

fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)

label = " (1/f)$\\beta$="
label += str(beta1)
plt.plot(initial_n_bins,  y1, label=label)
plt.legend()
plt.grid(True)


fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)
s, f = mlab.psd(y1, NFFT=len(y1))

plt.loglog(f * len(f), s)
formatter = ScalarFormatter()
formatter.set_useOffset(False)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlim(right = len(f) * 1.2)
plt.grid(True, which='both', alpha = 0.4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (Unit**2/Hz)')

plt.title("FFT Colored Noise, (1/f)$\\beta$=" + str(beta1), color='white')
plt.grid(True)

y2 = powerlaw_psd_gaussian(beta2, samples, fmin, f, sr2, si2)[:int(samples/return_to_beginning)]
if derivative:
    y2_dif = differentiate_once(y2, p1).real 

fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)

label = "(1/f)$\\beta$="
label += str(beta2)
plt.plot(initial_n_bins,  y2, label=label)
plt.legend()
plt.grid(True)
# optionally plot the Power Spectral Density with Matplotlib
fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)

s, f = mlab.psd(y2, NFFT=len(y2))

plt.loglog(f * len(f), s)
formatter = ScalarFormatter()
formatter.set_useOffset(False)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlim(right = len(f) * 1.2)
plt.grid(True, which='both', alpha = 0.4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (Unit**2/Hz)')

plt.title("FFT Colored Noise, (1/f)$\\beta$=" + str(beta2), color='white')
plt.grid(True)


# Different colors
start_color = 'blue'  
end_color = 'orange'
path_color = 'gray'

fig = plt.figure(facecolor='#002b36', figsize=(8, 8))
ax = fig.gca()
set_axis_color(ax)

# Plot the path 
plt.plot(y1[::n], y2[::n], marker='o', markersize=1, linestyle='-', linewidth=0.5, color=path_color)  # Gray for path
# Plot the start point (green)
plt.plot(y1[0], y2[0], marker='o', markersize=8, color=start_color, label='Start')
# Plot the end point (red)
plt.plot(y1[-1], y2[-1], marker='o', markersize=8, color=end_color, label='End')

plt.title("Simulated 2D Colored Noise Random Walk "+ "(1/f)$\\beta$=" + str(beta1) + ", (1/f)$\\beta$=" + str(beta2), color = 'white')
plt.xlabel("X", color = 'white')
plt.ylabel("Y", color = 'white')
plt.legend()  # Show the legend for start/end points
plt.grid(alpha=0.4)
plt.axis('equal')


exponent = math.log2(samples)
text1 = "Number of samples " + f"{samples} = 2^{int(exponent)}"


def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

dif = distance(y1[0],y2[0],y1[-1],y2[-1])
text2 = "Δ Start-End              " + f"{dif:.6e}"

ax = plt.gca()
plt.text(0.015, 0.97, text1, fontsize=9, transform=ax.transAxes, color = 'white')
plt.text(0.015, 0.95, text2, fontsize=9, transform=ax.transAxes, color = 'white')

plt.legend(loc='upper right') 

print(text1)
print(text2)


# Plot original and sorted points
plt.figure(facecolor='#002b36', figsize=(18, 6))
ax = plt.subplot(141, )
set_axis_color(ax)

label = " (1/f)$\\beta$="
label += str(beta1)

plt.title('Random Walk', color = 'white')
plt.plot(y1[::n], y2[::n], marker='o', markersize=0.5,  linewidth=0.5,  linestyle='-', color=path_color, label=label)
# Plot the start point (green)
plt.plot(y1[0], y2[0], marker='o', markersize=8, color=start_color, label='Start')
# Plot the end point (red)
plt.plot(y1[-1], y2[-1], marker='o', markersize=8, color=end_color, label='End')
plt.legend(loc='upper right') 
plt.grid(True)

ax = plt.subplot(142)
set_axis_color(ax)

# Sort by y-coordinate
sorted_indices = np.argsort(y1)
x_sorted = y2[sorted_indices]
y_sorted = y1[sorted_indices]

plt.title('Sorted by Y-Coordinate', color = 'white')
plt.plot(x_sorted[::n], -y_sorted[::n], marker='o', markersize=0.5,  linewidth=0.5,  linestyle='-', color=path_color, label=label)
# Plot the start point (green)
plt.plot(x_sorted[0], -y_sorted[0], marker='o', markersize=8, color=end_color, label='End')
# Plot the end point (red)
plt.plot(x_sorted[-1], -y_sorted[-1], marker='o', markersize=8, color=start_color, label='Start')
plt.legend(loc='lower right') 

ax = plt.subplot(143)
set_axis_color(ax)

plt.title('Random Walk Derivative', color = 'white')
plt.plot(y1_dif[::n], y2_dif[::n], marker='o', markersize=0.5,  linewidth=0.5,  linestyle='-', color=path_color, label=label)
# Plot the start point (green)
plt.plot(y1_dif[0], y2_dif[0], marker='o', markersize=8, color=start_color, label='Start')
# Plot the end point (red)
plt.plot(y1_dif[-1], y2_dif[-1], marker='o', markersize=8, color=end_color, label='End')
plt.legend(loc='upper right') 
plt.grid(True)

ax = plt.subplot(144)
set_axis_color(ax)

# Sort by y-coordinate
sorted_indices = np.argsort(y1_dif)
x_sorted = y2_dif[sorted_indices]
y_sorted = y1_dif[sorted_indices]

plt.title('Sorted by Y-Coordinate', color = 'white')
plt.plot(x_sorted[::n], -y_sorted[::n], marker='o', markersize=0.5,  linewidth=0.5,  linestyle='-', color=path_color, label=label)
# Plot the start point (green)
plt.plot(x_sorted[0], -y_sorted[0], marker='o', markersize=8, color=end_color, label='End')
# Plot the end point (red)
plt.plot(x_sorted[-1], -y_sorted[-1], marker='o', markersize=8, color=start_color, label='Start')
plt.legend(loc='lower right') 

# Plot original and sorted points

approx1 = func_approx(y1, Nfreq1)
approx2 = func_approx(y2, Nfreq2)

fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
plt.title("Random Walk Fourier Function Approximation", color = 'white')
label = "X vs Y"
plt.plot(approx1,  approx2, label=label)

plt.legend()
plt.grid(True, alpha = 0.4)
plt.xlabel("Approximation X= 0:" + str(Nfreq1) + " frequencies")
plt.ylabel("Approximation Y= 0:" + str(Nfreq2) + " frequencies")

ax = fig.gca()
set_axis_color(ax)

def is_closed_shape(x, y):
    """Checks if an array of (x, y) points represents a closed shape.

    Args:
        x: NumPy array of x-coordinates.
        y: NumPy array of y-coordinates.

    Returns:
        True if the shape is closed, False otherwise.
    """
    return (x[0] == x[-1]) and (y[0] == y[-1])


x = y1
y = y2
if is_closed_shape(x, y):
    print("Random Walk           is closed.")
else:
    print("Random Walk             is not closed.")
    
x = y1_dif
y = y2_dif
if is_closed_shape(x, y):
    print("Random Walk Derivative        is closed.")
else:
    print("Random Walk  Derivative is not closed.")

plt.show()






