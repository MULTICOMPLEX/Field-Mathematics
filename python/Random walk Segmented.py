from matplotlib import pylab as plt
import numpy as np
from numpy.fft import irfft, rfftfreq
from matplotlib.ticker import ScalarFormatter
import phimagic_prng32
import phimagic_prng64
import time


steps = 2**12 # number of steps to generate
return_to_beginning = 1
beta1 = 2 # the exponent
beta2 = 2# the exponent
fmin = 0.0;
n = 1 #plot every n sample
normal_input = 1
standard_dev = 1
function_input = 0
n_segments = 10

#Number of frequencies for approximation
Nfreq1 = 200
Nfreq2 = 25


# Different colors
start_color = 'blue'  
end_color = 'orange'
path_color = 'gray'

if(return_to_beginning == 0):
    return_to_beginning = 2;

def powerlaw_psd_gaussian(beta, steps, fmin, f, sr, si):
       
        # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./steps) # Low frequency cutoff
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
    
    # Calculate theoretical output standard deviation from scaling
    sigma = 2 * np.sqrt(np.sum(s_scale**2)) / steps
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
     
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=steps) / sigma
    
    return y


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


# Calculate Frequencies (we asume a sample rate of one)
# Use fft functions for real output (-> hermitian spectrum)
f = rfftfreq(steps) 

#Time seed 
current_time_seconds = int(time.time())
rng = np.random.default_rng(current_time_seconds)       #numpy PRNG
prng = phimagic_prng32.mxws(current_time_seconds)  #Phimagic fastest PRNG


# Independent standard normal variables
if normal_input:
    sr1 = rng.normal(0, standard_dev, size=len(f))  
    si1 = rng.normal(0, standard_dev, size=len(f))   
    sr2 = rng.normal(0, standard_dev, size=len(f)) 
    si2 = rng.normal(0, standard_dev, size=len(f))   

# Independent standard uniform variables
else:
    v = np.sqrt(np.pi)
    sr1 = prng.uniform(-v, v, size=len(f))  
    si1 = prng.uniform(-v, v, size=len(f))   
    sr2 = prng.uniform(-v, v, size=len(f))  
    si2 = prng.uniform(-v, v, size=len(f))  

 
def set_axis_color(ax):
    ax.set_facecolor('#002b36')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white') 
 
 
X = np.linspace(0, steps, int(steps/return_to_beginning)) 

y1 = powerlaw_psd_gaussian(beta1, steps, fmin, f, sr1, si1)
y2 = powerlaw_psd_gaussian(beta1, steps, fmin, f, sr2, si2)


approx1 = func_approx(y1, Nfreq1)
approx2 = func_approx(y2, Nfreq2)


def segment_curve(x, y, n_segments):
    """Segments a curve defined by x and y coordinates into n equal-length segments.

    Args:
        x: A 1D numpy array of x-coordinates.
        y: A 1D numpy array of y-coordinates.
        n_segments: The desired number of segments.

    Returns:
        A list of (x, y) tuples, where each tuple represents a segment. 
        If the curve cannot be divided equally, the last segment might be shorter.
    """

    # Calculate the cumulative distance along the curve
    distances = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    distances = np.insert(distances, 0, 0)  # prepend a 0 for the starting point

    # Calculate the target distance for each segment
    total_distance = distances[-1]
    target_distance = total_distance / n_segments


    segments = []
    start_index = 0
    for i in range(1, n_segments):
        # Find the index closest to the target distance for the current segment
        end_index = np.argmin(np.abs(distances - i * target_distance))  

        # Extract the segment
        x_segment = x[start_index:end_index+1]  #+1 to include the end point
        y_segment = y[start_index:end_index+1]
        segments.append((x_segment, y_segment))

        start_index = end_index


    # Add the last segment 
    x_segment = x[start_index:]
    y_segment = y[start_index:]
    segments.append((x_segment, y_segment))

    return segments


x = approx1[:int(steps/return_to_beginning)]
y = approx2[:int(steps/return_to_beginning)]

n_segments = n_segments# Number of segments
segments = segment_curve(x, y, n_segments)



# Plot the segmented curve:
plt.figure(figsize=(10,8))
for i, (x_seg, y_seg) in enumerate(segments):
    plt.plot(x_seg, y_seg, label=f"Segment {i+1}", marker='o', markersize=3)

# Plot the start point (green)
plt.plot(x[0], y[0], marker='o', markersize=8, color=start_color, label='Start')
# Plot the end point (red)
plt.plot(x[-1], y[-1], marker='o', markersize=8, color=end_color, label='End')

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Random Walk Segmented into {n_segments} Pieces")
plt.legend()
plt.grid(True)
plt.show()






