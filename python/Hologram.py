import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import skimage.draw  # You need to install scikit-image: pip install scikit-image
from matplotlib.colors import LinearSegmentedColormap
import colour  # Install with: pip install colour-science
import time
import porespy as ps
import phimagic_prng32


# Seed the random number generator
rng = np.random.default_rng(int(time.time())) 

# Create an instance of the custom PRNG
prng = phimagic_prng32.mxws() 

def generate_fresnel_hologram(object_field, z, wavelength, pixel_size, N):
    """
    Generates a Fresnel hologram of an object field.

    Args:
        object_field: NumPy array representing the object field (e.g., point source).
        z: Propagation distance.
        wavelength: Wavelength of light.
        pixel_size: Pixel size of the SLM/detector.
        N: Number of pixels in each dimension.

    Returns:
        NumPy array representing the hologram.
    """

    # Create spatial frequency grid
    fx = np.arange(-N/2, N/2) / (N * pixel_size)
    fy = np.arange(-N/2, N/2) / (N * pixel_size)
    Fx, Fy = np.meshgrid(fx, fy)

    # Fresnel propagator in Fourier domain
    H = np.exp(1j * 2 * np.pi * z / wavelength * np.sqrt(1 - (wavelength * Fx)**2 - (wavelength * Fy)**2))

    # Object field in Fourier domain
    Object_field_f = fft.fftshift(fft.fft2(object_field))

    # Propagate object field
    Hologram_field_f = Object_field_f * H

    # Hologram in spatial domain
    Hologram = fft.ifft2(fft.ifftshift(Hologram_field_f))

    return Hologram

def reconstruct_image(hologram, z, wavelength, pixel_size, N):
    """
    Reconstructs an image from a Fresnel hologram.

    Args:
        hologram: NumPy array representing the hologram.
        z: Propagation distance.
        wavelength: Wavelength of light.
        pixel_size: Pixel size of the SLM/detector.
        N: Number of pixels in each dimension.

    Returns:
        NumPy array representing the reconstructed image.
    """

    # Create spatial frequency grid
    fx = np.arange(-N/2, N/2) / (N * pixel_size)
    fy = np.arange(-N/2, N/2) / (N * pixel_size)
    Fx, Fy = np.meshgrid(fx, fy)

    # Fresnel propagator in Fourier domain (conjugate for reconstruction)
    H = np.exp(-1j * 2 * np.pi * z / wavelength * np.sqrt(1 - (wavelength * Fx)**2 - (wavelength * Fy)**2))

    # Hologram in Fourier domain
    Hologram_f = fft.fftshift(fft.fft2(hologram))

    # Propagate hologram
    Reconstructed_field_f = Hologram_f * H

    # Reconstructed image in spatial domain
    Reconstructed_image = np.abs(fft.ifft2(fft.ifftshift(Reconstructed_field_f)))**2

    return Reconstructed_image

def create_pentagram(N, radius, center):
    """Creates a binary image of a pentagram using scikit-image."""
    image = np.zeros((N, N), dtype=np.uint8)

    points = []
    for i in range(5):
        angle = 2 * np.pi * i / 5 - np.pi / 2  # Rotate to make one point upward
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        points.append((y, x))  # Note: skimage uses (row, col) indexing

    # Connect points to create the pentagram:
    # 0 -> 2 -> 4 -> 1 -> 3 -> 0
    pentagram_points = [points[0], points[2], points[4], points[1], points[3]]

    rr, cc = skimage.draw.polygon(
        [p[0] for p in pentagram_points], [p[1] for p in pentagram_points], shape=image.shape
    )
    image[rr, cc] = 255

    return image / 255.0

def create_pentagon(N, radius, center):
    """Creates a binary image of a pentagram using scikit-image."""
    image = np.zeros((N, N), dtype=np.uint8)

    points = []
    for i in range(5):
        angle = 2 * np.pi * i / 5 - np.pi / 2  # Rotate to make one point upward
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        points.append((y, x))  # Note: skimage uses (row, col) indexing

    rr, cc = skimage.draw.polygon(
        [p[0] for p in points], [p[1] for p in points], shape=image.shape
    )
    image[rr, cc] = 255
    
    return image / 255.0

def ordered_dithering(image, matrix_size=4):
    """Applies ordered dithering to an image."""
    # Bayer matrix for ordered dithering (4x4)
    bayer_matrix = np.array([
        [ 1,  9,  3, 11],
        [13,  5, 15,  7],
        [ 4, 12,  2, 10],
        [16,  8, 14,  6]
    ]) / (matrix_size * matrix_size + 1)

    height, width = image.shape
    dithered_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            threshold = bayer_matrix[y % matrix_size, x % matrix_size]
            dithered_image[y, x] = 1 if image[y, x] > threshold else 0
    return dithered_image

def random_dithering(image):
    """Applies random dithering to an image."""
    threshold = np.random.rand(*image.shape)
    dithered_image = (image > threshold).astype(float)
    return dithered_image

def floyd_steinberg_dithering(image):
    """Applies Floyd-Steinberg error diffusion dithering."""
    height, width = image.shape
    dithered_image = image.copy()
    for y in range(height - 1):
        for x in range(1, width - 1):
            old_pixel = dithered_image[y, x]
            new_pixel = round(old_pixel)  # Quantize to 0 or 1
            dithered_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            dithered_image[y    , x + 1] += quant_error * 7 / 16
            dithered_image[y + 1, x - 1] += quant_error * 3 / 16
            dithered_image[y + 1, x    ] += quant_error * 5 / 16
            dithered_image[y + 1, x + 1] += quant_error * 1 / 16
    return dithered_image

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

def set_axis_color(ax):
    """Sets the axis colors to white."""
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')


def powerlaw_psd_gaussian_2D(beta, size, rng, uniform_normal, stdev=1, fmin=0):
    """Generates a 2D Gaussian random field with a power-law PSD
       and scales the phase of the output to [0, 2*pi).
    """

    if not 0 <= fmin <= 0.5:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    N = size
    fx = np.fft.fftfreq(N)
    fy = np.fft.fftfreq(N)
    fx, fy = np.meshgrid(fx, fy)
    f = np.sqrt(fx**2 + fy**2)

    fmin = max(fmin, 1./N) 

    s_scale = f.copy()
    s_scale[f < fmin] = fmin
    s_scale = s_scale**(-beta/2.)
   
    if uniform_normal:
        #real_part = rng.uniform(0, 1, size=(N, N))
        #imag_part = rng.uniform(0, 1, size=(N, N))
        real_part = prng.uniform(0, 1, N, N)
        imag_part = prng.uniform(0, 1, N, N)
    else:
        real_part = rng.normal(loc=0, scale=stdev,  size=(N, N))
        imag_part = rng.normal(loc=0, scale=stdev,  size=(N, N))

    sr = real_part * s_scale
    si = imag_part * s_scale
    
    s = sr + 1j * si
    
    y = np.fft.fftshift(np.fft.ifftn(s))
    
    # Extract phase and scale it to [0, 2*pi)
    phase_y = np.angle(y)
    scaled_phase_y = (phase_y + np.pi) % (2 * np.pi)  # Shift to [0, 2*pi]

    return scaled_phase_y

def add_speckle_phase(field, speckle_size):
    """Adds a random phase to a field to simulate speckle."""
    N = field.shape[0]
   
    # Parameters
    beta = 2
    fmin = 0.0
    stdev = 4
    uniform_normal = 1

    # Generate the signal
    signal = powerlaw_psd_gaussian_2D(beta, N//speckle_size, rng, uniform_normal, stdev, fmin)
    
    signal = np.fft.fftshift(np.kron(signal,np.ones((speckle_size, speckle_size))))
    
    #speckle_phase = np.random.uniform(0, 2*np.pi, size=(N//speckle_size, N//speckle_size))
    #speckle_phase = np.fft.fftshift(np.kron(speckle_phase,np.ones((speckle_size, speckle_size))))
    
    return field * np.exp(1j * signal)
    #return field * np.exp(1j * speckle_phase)


# Parameters
N = 1024  # Number of pixels
# Define the wavelength of the HeNe laser
wavelength = 0.6328e-6  # Wavelength of HeNe laser (meters)
wavelength_nm = wavelength * 1e9

# Convert wavelength to RGB using CIE 1931 functions
xyz = colour.wavelength_to_XYZ(wavelength_nm)
if np.max(xyz) > 0:
    xyz /= np.max(xyz)  # Normalize XYZ to max value

rgb = colour.XYZ_to_sRGB(xyz)  # Convert XYZ to sRGB
rgb = np.clip(rgb, 0, 1)  # Ensure RGB values are in [0, 1]

# Create a custom colormap
colormap = LinearSegmentedColormap.from_list("Laser", [(0, "black"), (0.5, rgb), (1, "white")])

z = 0.2  # Propagation distance (meters)
pixel_size = 5e-7  # Pixel size (meters)    10e-6
speckle_size = N//128 # Control the size of the speckles, smaller values mean larger speckles

# Generate Point Source Object
object_field = np.zeros((N, N))
object_field[N//2, N//2] = 1  # Point source in the center

# --- Object Field Creation: Pentagram ---
pentagram_radius = N // 2
pentagram_center = (N // 2, N // 2)
object_field = create_pentagram(N, pentagram_radius, pentagram_center)
#object_field = func_approx_2D(object_field, N//3).real


# --- Add Speckle Phase ---
object_field_with_speckle = add_speckle_phase(object_field, speckle_size)

# --- Hologram Generation ---
hologram = generate_fresnel_hologram(object_field_with_speckle, z, wavelength, pixel_size, N)

# Display the Hologram
fig = plt.figure(figsize=(18, 5), facecolor='#002b36')
plt.tight_layout()

plt.subplot(1, 2, 1)
ax = fig.gca()
set_axis_color(ax)
plt.imshow(object_field, interpolation='bicubic', cmap='gray') # Display the colored reconstructed image
plt.title('Original Image', color='white')


# --- Image Reconstruction ---
reconstructed_image = reconstruct_image(hologram, z, wavelength, pixel_size, N)

# Display the Reconstructed Image with Colormap
plt.subplot(1, 2, 2)
ax = fig.gca()
set_axis_color(ax)
plt.imshow(reconstructed_image, interpolation='bicubic', cmap=colormap) # Display the colored reconstructed image
plt.title('Reconstructed Image', color='white')


colormap = 'seismic'

fig = plt.figure(figsize=(15, 8), facecolor='#002b36')
ax = fig.gca()
set_axis_color(ax)
im = plt.imshow(np.angle(hologram), interpolation='bicubic', cmap=colormap) # Use laser_colormap for the phase of the hologram
plt.title('Phase of Hologram', color='white')
# Add the colorbar and customize
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors='white')
cbar.set_label('Amplitude', color='white')


fig = plt.figure(figsize=(15, 8), facecolor='#002b36')
ax = fig.gca()
set_axis_color(ax)
im = plt.imshow(np.abs(hologram), interpolation='bicubic', cmap=colormap) # Use laser_colormap for the amplitude of the hologram
plt.title('Amplitude of Hologram', color='white')
# Add the colorbar and customize
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors='white')
cbar.set_label('Amplitude', color='white')

'''
# Parameters
beta = 2
fmin = 0.0
stdev = 4
uniform_normal = 1
# Generate the signal
hologram = np.exp(1j * powerlaw_psd_gaussian_2D(beta, N, rng, uniform_normal, stdev, fmin))
'''

image_array = np.angle(hologram)
# Binarize the image (thresholding)
binary_image = image_array > np.mean(image_array)

result = ps.metrics.boxcount(binary_image)
sizes = result.size
counts = result.count

# Step 3: Compute fractal dimension (slope of log-log plot)
log_sizes = np.log(sizes)
log_counts = np.log(counts)
slope, _ = np.polyfit(log_sizes, log_counts, 1)  # Linear fit

# Fractal dimension
fractal_dimension = -slope

fractal_dimension = -result.slope
print('fractal_dimension', fractal_dimension)

# Step 4: Plot the results
fig = plt.figure(figsize=(8, 6), facecolor='#002b36')
ax = fig.gca()
set_axis_color(ax)
plt.plot(log_sizes, log_counts, 'o-', label="Box-Counting Data")
plt.plot(log_sizes, np.polyval([slope, _], log_sizes), 'r--', label=f"Fit: slope = {slope:.2f}")
plt.xlabel("log(Box Size)")
plt.ylabel("log(Box Count)")
plt.title("Fractal Dimension via Box Counting", color='white')
plt.legend()

b = result

fig = plt.figure(figsize=(12, 6), facecolor='#002b36')
plt.tight_layout()

plt.subplot(1, 2, 1)
ax = fig.gca()
set_axis_color(ax)

ax.loglog(b.size, b.count, 'o-', )
ax.set_xlabel('box length')
ax.set_ylabel('number of partially filled boxes')

plt.subplot(1, 2, 2)
ax = fig.gca()
set_axis_color(ax)

ax.semilogx(b.size, b.slope, 'o-', )
ax.plot([0, 1000], [1.9, 1.9])
ax.set_xlabel('box length')
ax.set_ylabel('slope')
ax.set_ylim([0, 3]);

plt.show()