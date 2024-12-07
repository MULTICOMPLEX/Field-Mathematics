import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import skimage.draw  # You need to install scikit-image: pip install scikit-image
from matplotlib.colors import LinearSegmentedColormap
import colour  # Install with: pip install colour-science
import time
import porespy as ps
import phimagic_prng32
from PIL import Image



# Seed the random number generator
rng = np.random.default_rng(int(time.time())) 

# Create an instance of the custom PRNG
prng = phimagic_prng32.mxws(int(time.time())) 


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
    H = np.exp(1j * 2 * np.pi * z / wavelength * np.lib.scimath.sqrt(1 - (wavelength * Fx)**2 - (wavelength * Fy)**2))

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
    H = np.exp(-1j * 2 * np.pi * z / wavelength * np.lib.scimath.sqrt(1 - (wavelength * Fx)**2 - (wavelength * Fy)**2))

    # Hologram in Fourier domain
    Hologram_f = fft.fftshift(fft.fft2(hologram))

    # Propagate hologram
    Reconstructed_field_f = Hologram_f * H

    # Reconstructed image in spatial domain
    Reconstructed_image = np.abs(fft.ifft2(fft.ifftshift(Reconstructed_field_f)))

    return Reconstructed_image


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

def add_speckle_phase(field, speckle_sizespeckle_size, beta, fmin, stdev, uniform_normal):
    """Adds a random phase to a field to simulate speckle."""
    N = field.shape[0]

    # Generate the signal
    signal = powerlaw_psd_gaussian_2D(beta, N//speckle_size, rng, uniform_normal, stdev, fmin)
    
    signal = np.fft.fftshift(np.kron(signal,np.ones((speckle_size, speckle_size))))
    
    return field * np.exp(1j * signal)
    

def rescale_jpg_to_grayscale_array(image_path, new_width, new_height, normalize=False):
    """
    Rescales a JPG image to a grayscale NumPy array.

    Args:
        image_path: Path to the JPG image file.
        new_width: Desired width of the rescaled image.
        new_height: Desired height of the rescaled image.
        normalize: Whether to normalize pixel values to [0, 1] (default: False).

    Returns:
        NumPy array representing the rescaled grayscale image, or None if an error occurs.
    """
    try:
        # 1. Load image
        img = Image.open(image_path)

        # 2. Convert to grayscale

        r, g, b = img.split()
        
        # 3. Resize image
        resized_img1 = r.resize((new_width, new_height), Image.LANCZOS)
        resized_img2 = g.resize((new_width, new_height), Image.LANCZOS)
        resized_img3 = b.resize((new_width, new_height), Image.LANCZOS)

        # 4. Convert to NumPy array
        r = np.array(resized_img1)
        g = np.array(resized_img2)
        b = np.array(resized_img3)

        # 5. (Optional) Normalize
        if normalize:
            r = r / 255.0
            g = g / 255.0
            b = b / 255.0

        return [r,g,b]  # Return the img_array, not the original img object

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    
        

# Parameters
N = 1024  # Number of pixels
wavelength = 0.6328e-6  # Wavelength of HeNe laser (meters)
z = 0.2 # Propagation distance (meters)
pixel_size = 10e-7  # Pixel size (meters)    10e-6
speckle_size = 1 # Control the size of the speckles, smaller values mean larger speckles

# Noise Parameters
beta = 2
fmin = 0.0
stdev = 4
uniform_normal = 1

object_field_r = rescale_jpg_to_grayscale_array('IMG_20241202_154430799.jpg', new_width = N, new_height = N, normalize=True)[0]
object_field_g = rescale_jpg_to_grayscale_array('IMG_20241202_154430799.jpg', new_width = N, new_height = N, normalize=True)[1]
object_field_b = rescale_jpg_to_grayscale_array('IMG_20241202_154430799.jpg', new_width = N, new_height = N, normalize=True)[2]

# --- Add Speckle Phase ---
object_field_with_speckle_r = add_speckle_phase(object_field_r, speckle_size, beta, fmin, stdev, uniform_normal)
object_field_with_speckle_g = add_speckle_phase(object_field_g, speckle_size, beta, fmin, stdev, uniform_normal)
object_field_with_speckle_b = add_speckle_phase(object_field_b, speckle_size, beta, fmin, stdev, uniform_normal)

# --- Hologram Generation ---
hologram_r = generate_fresnel_hologram(object_field_with_speckle_r, z, wavelength, pixel_size, N)
hologram_g = generate_fresnel_hologram(object_field_with_speckle_g, z, wavelength, pixel_size, N)
hologram_b = generate_fresnel_hologram(object_field_with_speckle_b, z, wavelength, pixel_size, N)

# Display the Hologram
fig = plt.figure(figsize=(12, 5), facecolor='#002b36')
plt.tight_layout()

plt.subplot(1, 2, 1)
ax = fig.gca()
set_axis_color(ax)
img = Image.open('IMG_20241202_154430799.jpg')

plt.imshow(img.resize((N, N), Image.LANCZOS)) # Display the colored reconstructed image
plt.title('Original Image', color='white')


# --- Image Reconstruction ---
reconstructed_image_r = reconstruct_image(hologram_r, z, wavelength, pixel_size, N)
reconstructed_image_g = reconstruct_image(hologram_g, z, wavelength, pixel_size, N)
reconstructed_image_b = reconstruct_image(hologram_b, z, wavelength, pixel_size, N)


# Stack the arrays along a new axis (axis=2) to create the (height, width, 3) shape
rgb_array = np.stack([reconstructed_image_r, reconstructed_image_g, reconstructed_image_b], axis=2)

if rgb_array.dtype != np.uint8:
    # If the array has a different data type, normalize and scale to 0-255
    rgb_array = (rgb_array - rgb_array.min()) / (rgb_array.max() - rgb_array.min())  # Normalize to 0-1
    rgb_array = (rgb_array * 255).astype(np.uint8)

# Create the PIL image
reconstructed_image = Image.fromarray(rgb_array)

# Display the Reconstructed Image with Colormap
plt.subplot(1, 2, 2)
ax = fig.gca()
set_axis_color(ax)
plt.imshow(reconstructed_image) # Display the colored reconstructed image
plt.title('Reconstructed Image', color='white')


colormap = 'seismic'

fig = plt.figure(figsize=(12, 5), facecolor='#002b36')
plt.tight_layout()

plt.subplot(1, 2, 1)
ax = fig.gca()
set_axis_color(ax)

im = plt.imshow(np.angle(hologram_r), cmap=colormap) # Display the colored reconstructed image
plt.title('Phase of Hologram (red)', color='white')
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors='white')
cbar.set_label('Amplitude', color='white')

plt.subplot(1, 2, 2)
ax = fig.gca()
set_axis_color(ax)
im = plt.imshow(np.abs(hologram_r), cmap=colormap) # Display the colored reconstructed image
plt.title('Amplitude of Hologram (red)', color='white')
# Add the colorbar and customize
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors='white')
cbar.set_label('Amplitude (red)', color='white')

###

fig = plt.figure(figsize=(12, 5), facecolor='#002b36')
plt.tight_layout()

plt.subplot(1, 2, 1)
ax = fig.gca()
set_axis_color(ax)

im = plt.imshow(np.angle(hologram_g), cmap=colormap) # Display the colored reconstructed image
plt.title('Phase of Hologram (green)', color='white')
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors='white')
cbar.set_label('Amplitude', color='white')

plt.subplot(1, 2, 2)
ax = fig.gca()
set_axis_color(ax)
im = plt.imshow(np.abs(hologram_g), cmap=colormap) # Display the colored reconstructed image
plt.title('Amplitude of Hologram (green)', color='white')
# Add the colorbar and customize
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors='white')
cbar.set_label('Amplitude (green)', color='white')


###

fig = plt.figure(figsize=(12, 5), facecolor='#002b36')
plt.tight_layout()

plt.subplot(1, 2, 1)
ax = fig.gca()
set_axis_color(ax)

im = plt.imshow(np.angle(hologram_b), cmap=colormap) # Display the colored reconstructed image
plt.title('Phase of Hologram (blue)', color='white')
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors='white')
cbar.set_label('Amplitude (blue)', color='white')

plt.subplot(1, 2, 2)
ax = fig.gca()
set_axis_color(ax)
im = plt.imshow(np.abs(hologram_b), cmap=colormap) # Display the colored reconstructed image
plt.title('Amplitude of Hologram (blue)', color='white')
# Add the colorbar and customize
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(colors='white')
cbar.set_label('Amplitude (blue)', color='white')


plt.show()