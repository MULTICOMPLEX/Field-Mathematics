from matplotlib import pylab as plt
import numpy as np
from numpy.fft import rfftfreq, irfft2
import progressbar
import matplotlib.animation as animation
import time
from scipy.stats import multivariate_normal


steps = 250# number of steps to generate

beta = 3.5 # the exponent

fmin = 0.0

#Number of frequencies for approximation
Nfreq = 10

function_approximation = False

#Number of animations
n = 500

cmap = 'Pastel1'
levels = 64
fps = 5

v = 1
vmin =  0
vmax = v 
 

def powerlaw_psd_gaussian_2D(real_part, imag_part, beta, size, fmin=0):

    # Validate fmin
    if not 0 <= fmin <= 0.5:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Dimensions (assuming square for simplicity)
    N = size
    # Generate 2D frequencies
    fx = np.fft.fftfreq(N)  # Note: fftfreq, not rfftfreq
    fy = np.fft.fftfreq(N)

    fx, fy = np.meshgrid(fx, fy) # 2D frequency grid

    f = np.sqrt(fx**2 + fy**2)   # Radial frequencies
    # Shift the zero frequency component to the center
   
   
    # Ensure fmin is not too small
    fmin = max(fmin, 1./N)  # Lower bound on fmin

    # Scaling factors
    s_scale = f.copy() #copy so we don't modify the original f
    s_scale[f < fmin] = fmin  # Avoid division by zero
    s_scale = s_scale**(-beta/2.)
    

    # Ensure conjugate symmetry for real signal
    sr = real_part * s_scale
    si = imag_part * s_scale
    #Enforce conjugate symmetry

    # Calculate theoretical output standard deviation from scaling
    sigma = 2 * np.sqrt(np.sum(s_scale**2)) / steps
        
    # Combine to create Fourier components
    s = sr + 1j * si

    # Inverse FFT
    y = np.fft.ifft2(s) / sigma 
    
    y= np.fft.fftshift(y)
    
    return y

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

def PDF_2D_Gaussian(X, Y):
   # Define the mean vector and covariance matrix for the 2D Gaussian
    mean = [0, 0]  # Mean for [X, Y]
    cov = [[0.5, 0.0], [0.0, 0.5]]  # Covariance matrix

    # Create a grid of (x, y) coordinates for the joint distribution
    pos = np.dstack((X, Y))

    # Compute the joint PDF for the 2D Gaussian
    rv_joint = multivariate_normal(mean, cov)
    pdf_joint =  np.fft.fftshift(rv_joint.pdf(pos)) 
    return pdf_joint

def func_approx_2D(x, n):
    """
    Performs 2D Fourier approximation.

    Args:
        x: 2D NumPy array (e.g., an image).
        n: Number of frequency components to keep in each dimension.

    Returns:
        The 2D approximated array (real part).
    """
    # 2D Fourier Transform
    yf = np.fft.fft2(x)

    # Truncate higher frequencies
    num_components = int(n)
    h, w = yf.shape
    yf_truncated = yf.copy() #Important to copy so you don't modify original yf
    yf_truncated[num_components:h-num_components,:] = 0 #rows, all cols
    yf_truncated[:,num_components:w-num_components] = 0 #all rows, cols

    # 2D Inverse Fourier Transform
    y_approx = np.fft.ifft2(yf_truncated)
    return y_approx

#Time seed 
current_time_seconds = int(time.time())
rng = np.random.default_rng(current_time_seconds)       #numpy PRNG


N = steps
real_part = rng.uniform(vmin, vmax, size=(N, N))
imag_part = rng.uniform(vmin, vmax, size=(N, N))


noise = powerlaw_psd_gaussian_2D(real_part, imag_part, beta, steps, fmin)

if function_approximation:
    noise =  np.abs(func_approx_2D(noise, Nfreq))
else:
    noise = np.abs(noise)


# Create X and Y coordinates
x = np.arange(noise.shape[1])  # x-coordinates (columns)
y = np.arange(noise.shape[0])  # y-coordinates (rows)
X, Y = np.meshgrid(x, y)


#noise = func_approx_2D(PDF_2D_Gaussian(X, Y), Nfreq)


fig = plt.figure(figsize=(8, 8), facecolor='#002b36')  
ax = fig.gca()
set_axis_color(ax)
plt.title('Random walk model of the atom', color = 'white')
plt.grid()
# Create the contour plot
ax.contour(X, Y, noise, levels=levels, cmap=cmap) #Or contourf
plt.show()


Ψ = np.zeros((n, *([steps] * 2)))
v = np.sqrt(np.pi) / 4


t0 = time.time()
bar = progressbar.ProgressBar()
for i in bar(range(n)):
    real_part += rng.uniform(vmin, vmax, size=(N, N))
    imag_part += rng.uniform(vmin, vmax, size=(N, N))
    tmp = powerlaw_psd_gaussian_2D(real_part, imag_part, beta, steps, fmin)
    if function_approximation:
        Ψ[i] =   np.abs(func_approx_2D(tmp, Nfreq))
    else:
        Ψ[i] = np.abs(tmp)
        
print("Took", time.time() - t0)

Ψ /= np.amax(np.abs(Ψ))

fig = plt.figure(figsize=(8, 8), facecolor='#002b36')  
ax = fig.gca()
set_axis_color(ax)
plt.title('Random walk model of the atom', color = 'white')
plt.grid()
# Create the contour plot
ax.contour(X, Y, Ψ[-1], levels=levels, cmap=cmap) #Or contourf
plt.show()


fig = plt.figure(figsize=(8, 8), facecolor='#002b36')  
ax = fig.gca()
set_axis_color(ax)
plt.title('Random walk model of the atom', color = 'white')
plt.grid()

# Set up the contour plot for the first frame
contour = ax.contour(X, Y, Ψ[0], levels=levels, cmap=cmap) # Initial frame

time_ax = ax.text(0.97,0.97, "",  color = "white",
                        transform=ax.transAxes, ha="right", va="top")
      
def animate(i):
    global contour  # Access and overwrite the existing contour object
    # Remove the previous contour by calling .remove() on each path
    contour.remove()  
     # Update frame counter text
    time_ax.set_text(f'Frame: {i+1}/{n}') # i+1 because i starts from 0
    contour = ax.contour(X, Y, Ψ[i], levels=levels, cmap=cmap)
    return 


# Create the animation

ani = animation.FuncAnimation(fig, animate, frames=n, interval= 1/fps * 1000, blit=False) # Adjust interval as needed (milliseconds)

writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=5000)  # Adjust bitrate here # kbps
ani.save('Random_walk2D.mp4', writer=writer)

#ani.save('Random_walk2D.gif', fps = fps, metadata = dict(artist = 'Me'))

