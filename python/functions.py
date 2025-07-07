import numpy as np
from constants import *
import pyfftw
import multiprocessing
import time
import progressbar
from matplotlib import mlab
from matplotlib.ticker import ScalarFormatter
from numpy.fft import irfft, rfftfreq, ifft,  fftfreq, ifft2
#from numba import njit
import phimagic_prng32


#Time seed 
current_time_seconds = int(time.time())
rng = np.random.default_rng(current_time_seconds)       #numpy PRNG
prng = phimagic_prng32.mxws(current_time_seconds)  #Phimagic fastest PRNG

  # laplacian_operator
def fft_frequencies_1D(N, dx, hbar):
    p1 = np.fft.fftfreq(N, d=dx) * hbar * 2*np.pi
    return p1**2

# laplacian_operator
def fft_frequencies_2D(N, dx, hbar):
    p1 = np.fft.fftfreq(N, d = dx) * hbar  * 2*np.pi
    p2 = np.fft.fftfreq(N, d = dx) * hbar  * 2*np.pi
    p1, p2 = np.meshgrid(p1, p2)
    return p1**2 + p2**2

# Parameters
hbar = 1.0     # Reduced Planck's constant
m1 = 1.0       # Mass of oscillator 1
m2 = 1.0       # Mass of oscillator 2
g = 0.1        # Coupling constant

def fft_frequencies_2D_1(N, dx, hbar):
    # Compute the momentum components
    p = np.fft.fftfreq(N, d=dx) * hbar * 2 * np.pi
    p1, p2 = np.meshgrid(p, p, indexing='ij')
    return p1, p2

def kinetic_energy_2D(N, dx, hbar, m1, m2, g):
    # Compute the kinetic energy with coupling
    p1, p2 = fft_frequencies_2D_1(N, dx, hbar)
    kinetic_energy = (p1**2) / (2 * m1) + (p2**2) / (2 * m2) + g * p1 * p2
    return kinetic_energy


'''
# laplacian_operator
def fft_frequencies_3D(N, dx, hbar):
    p1 = np.fft.fftfreq(N, d = dx) * hbar  * 2*np.pi
    p2 = np.fft.fftfreq(N, d = dx) * hbar  * 2*np.pi
    p3 = np.fft.fftfreq(N, d = dx) * hbar  * 2*np.pi
    p1, p2, p3 = np.meshgrid(p1, p2, p3)
    return p1**2 + p2**2 + p3**2
'''
    
def fft_frequencies_ND(shape, spacing, hbar):
    freqs = [np.fft.fftfreq(n, d=dx) * hbar * 2*np.pi for n, dx in zip(shape, spacing)]
    grids = np.meshgrid(*freqs)
    return np.sum(np.square(grids), axis=0)


def norm(phi, dx):
    norm = np.linalg.norm(phi) * dx
    return (phi * np.sqrt(dx)) / norm

'''
def norm(phi, dx):
    return phi/np.sqrt(np.sum(np.square(np.abs(phi)) * dx))
'''

def normalize(T):
    norm = np.max(np.abs(T))
    return T  /  norm
    
# P = sum_i |psi_i><psi_i|
# method for projecting a vector onto a given subspace.
# orthogonal projection


def apply_projection(tmp, psi_list, dx):
    """
    Orthogonalizes a vector (tmp) against a list of vectors (psi_list)
    using the Gram-Schmidt process.

    Args:
        tmp: The vector to orthogonalize (NumPy array).
        psi_list: A list of orthogonal vectors (NumPy arrays).
        dx: Spatial discretization factor.

    Returns:
        The orthogonalized vector (NumPy array).
    """
    for i in range(len(psi_list)):
        psi = psi_list[i]
        coef = np.vdot(psi, tmp) * dx
        tmp -= coef * psi
    return tmp

def apply_projection2(tmp, psi_list, dx):
    for psi in psi_list:
        tmp -= np.vdot(psi, tmp) * psi * dx
    return tmp


'''   
def apply_projection(tmp, psi_list, dx):
    for psi in psi_list:
        tmp -= np.sum(tmp*psi.conj()) * psi * dx 
    return tmp
'''


def differentiate_twice(f, p2):
    f = np.fft.ifftn(-p2*np.fft.fftn(f))
    return f


'''

 # for differentiate_once or integrate_once
def fft_frequencies_1D_1(N, dx, hbar):
    p = np.fft.fftfreq(N, d=dx) * hbar * 2*np.pi * 1j
    return p

def differentiate_once(f, p):
    f = np.fft.ifft(p*np.fft.fft(f))
    return f

def integrate_twice(f, p2):
    F = np.fft.fftn(f)
    F[1:] /= p2[1:]
    F = np.fft.ifftn(F)
    F -= F[0]
    return F
    
def integrate_once(f, p):
    F = np.fft.fftn(f)
    F[1:] /= p[1:]
    return np.fft.ifftn(F)
'''


# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.interfaces.cache.enable()


def Split_Step_FFTW(Ψ, phi, dx, store_steps, Nt_per_store_step, Ur, Uk, ite):
    for i in range(store_steps):
        tmp = Ψ[i]
        for _ in range(Nt_per_store_step):
            c = pyfftw.interfaces.numpy_fft.fftn(Ur*tmp)
            tmp = Ur * pyfftw.interfaces.numpy_fft.ifftn(Uk*c)
            if (ite):
                tmp = norm(apply_projection(tmp, phi, dx), dx)
        Ψ[i+1] = tmp
    return Ψ


def Split_Step_NP(Ψ, phi, dx, store_steps, Nt_per_store_step, Ur, Uk, ite):
    for i in range(store_steps):
        tmp = Ψ[i]
        for _ in range(Nt_per_store_step):
            c = np.fft.fftn(Ur*tmp)
            tmp = Ur * np.fft.ifftn(Uk*c)
            if (ite):
                tmp = norm(apply_projection(tmp, phi, dx), dx)
        Ψ[i+1] = tmp
    return Ψ


def ground_state(psi_0, Ψ, dx, store_steps, Nt_per_store_step, Ur, Uk, ite, path_data, title, save):
    # Define the ground state wave function
    t0 = time.time()
    Ψ[0] = norm(psi_0, dx)
    phi = np.array([Ψ[0]])
    print("Computing Ground State...")
    Ψ = Split_Step_NP(Ψ, phi, dx, store_steps, Nt_per_store_step, Ur, Uk, ite)
    print("Took", time.time() - t0)
    if (save):
        title = path_data+title
        np.save(title, Ψ)
    return Ψ


def eigenvalues_exited_states(Ψ, phi, state, dx, store_steps, Nt_per_store_step, Ur, Uk, ite, path_data, save):
    # raising operators
    Split_Step_NP(Ψ, phi, dx, store_steps, Nt_per_store_step, Ur, Uk, ite)
    if (save):
        title = path_data+"Exited_State[{}].npy".format(state)
        print("Saving State...")
        np.save(title, Ψ[-1])
    phi = np.concatenate([phi, [Ψ[-1]]])
    return phi


def Energies(V, phi, p2, hbar, m):

    # Define the Hamiltonian operator
    def hamiltonian_operator(psi):
        # Calculate the kinetic energy part of the Hamiltonian
        KE = -(hbar**2 / 2*m) * differentiate_twice(psi, p2)
        # K = -(hbar^2 / 2m) * d^2/dx^2
        # KE = (hbar^2 / 2m) * |dpsi/dx|^2
        # Calculate the potential energy part of the Hamiltonian
        PE = V * psi
        # Combine the kinetic and potential energy parts to obtain the full Hamiltonian
        H = KE + PE
        return H

    def expectation_value(psi, operator):
        operator_values = operator(psi)
        expectation = np.vdot(psi, operator_values)  # E = <Ψ|H|Ψ>
        return expectation

    energies = np.array(
        [expectation_value(i, hamiltonian_operator) for i in phi])

    return energies


def set_axis_color(ax):
    ax.set_facecolor('#002b36')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white') 

def reverse_first_half_1d(arr):
    """
    Reverses the first half of a 1D NumPy array automatically.

    Parameters:
    - arr: 1D NumPy array

    Returns:
    - Modified array with the first half reversed
    """
    n = len(arr)
    half = n // 2
    first_half = arr[:half].copy()

    # Reverse the first half
    reversed_half = first_half[::-1]

    # Combine the reversed first half with the unchanged second half
    return np.concatenate((reversed_half, arr[half:]))

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



def powerlaw_psd_gaussian_2d(beta, steps_x, steps_y, fmin):
    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1. / min(steps_x, steps_y))  # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Create frequency grids for x and y dimensions
    freq_x = fftfreq(steps_x)
    freq_y = fftfreq(steps_y)
    fx, fy = np.meshgrid(freq_x, freq_y, indexing='ij')
    f = np.sqrt(fx**2 + fy**2)

    # Build scaling factors for all frequencies
    s_scale = f.copy()
    s_scale[s_scale < fmin] = fmin
    s_scale = (s_scale / np.sqrt(2)) ** (-beta)

    # Generate random components
    v = np.sqrt(np.pi)
    rng = np.random.default_rng()
    sr = rng.uniform(-v, v, size=(steps_x, steps_y))
    si = rng.uniform(-v, v, size=(steps_x, steps_y))

    sr *= s_scale
    si *= s_scale

    # Calculate theoretical output standard deviation from scaling
    sigma = 2 * np.sqrt(np.sum(s_scale**2)) / (steps_x * steps_y)

    # Combine power and phase to Fourier components
    s = sr + 1j * si

    # Transform to real space and scale to unit variance
    y = ifft2(s) / sigma

    return y

    
def powerlaw_psd_gaussian(beta, steps, fmin):
   
        # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./steps) # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")
    
    f = rfftfreq(steps)

    v = np.sqrt(np.pi)
    rng = np.random.default_rng(current_time_seconds)       #numpy PRNG
    sr = rng.uniform(-v, v, size=len(f))  
    si = rng.uniform(-v, v, size=len(f))
      
    # Build scaling factors for all frequencies
    s_scale = f    
    ix   = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = (s_scale/2)**(-beta)
      
    sr *= s_scale
    si *= s_scale   
    
    # Calculate theoretical output standard deviation from scaling
    sigma = 2 * np.sqrt(np.sum(s_scale**2)) / steps
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
     
    # Transform to real time series & scale to unit variance
    y = ifft(s, n=steps) / sigma
    
    return y

def second_order_diff_noise(num_samples, dx):
    num_samples += 2

    white_noise = prng.uniform(0, 1, num_samples)
    # First-order difference (differentiated white noise)
    first_order_diff = np.diff(white_noise)
    # Second-order difference (second-order differentiated white noise)
    second_order_diff = np.diff(first_order_diff)
    sr = second_order_diff

    white_noise = prng.uniform(0, 1, num_samples)
    # First-order difference (differentiated white noise)
    first_order_diff = np.diff(white_noise)
    # Second-order difference (second-order differentiated white noise)
    second_order_diff = np.diff(first_order_diff)
    si = 1j * second_order_diff

    psi_0 = norm(sr + si, dx)
    return psi_0

def first_order_diff_noise(num_samples, dx):
    num_samples += 1

    white_noise = prng.uniform(0, 1, num_samples)
    # First-order difference (differentiated white noise)
    first_order_diff = np.diff(white_noise)
    
    sr = first_order_diff

    white_noise = prng.uniform(0, 1, num_samples)
    # First-order difference (differentiated white noise)
    first_order_diff = np.diff(white_noise)
   
    si = 1j * first_order_diff

    psi_0 = norm(sr + si, dx)
    return psi_0



def madelung_transform(psi):
    R = np.abs(psi)  # Amplitude
    S = np.angle(psi) * hbar  # Phase
    return R, S

def States(Ψ, psi_0, dx, Nt_per_store_step, Ur, Uk, S):

    # Define the ground state wave function
    title = "Ground_State.npy"
    t0 = time.time()
    bar = progressbar.ProgressBar(maxval=1)
    for _ in bar(range(1)):
        Ψ = ground_state(psi_0, Ψ, dx, S["store steps"], Nt_per_store_step, Ur, Uk, S["imaginary time evolution"], S["path data"], title, False)
    print("Took", time.time() - t0)

    #Ψ[0] =  psi_0
    Ψ[0] = Ψ[-1]
    phi = np.array([Ψ[0]])

    nos = S["Number of States"]-1
    if (nos):
        t0 = time.time()
        bar = progressbar.ProgressBar(maxval=nos)
        # raising operators
        for i in bar(range(nos)):
            Ψ = Split_Step_NP(Ψ, phi, dx, S["store steps"], Nt_per_store_step, Ur, Uk, S["imaginary time evolution"])
            phi = np.concatenate([phi, [Ψ[-1]]])
        print("Took", time.time() - t0)
        
    return Ψ




'''    
#3D
def V_Coulomb_3D(X, Y, Z):
    q = 0.5
    ε = 1
    π = np.pi
    d = np.power(X**2 + Y**2 + Z**2, 0.5)
    return (q / (4 * π * ε)) * np.log(d)

#1D
def V_Coulomb_1D(X):
    q = 0.5
    ε = 1
    π = np.pi
    return (q / (2 * ε)) * np.log(X**2)
'''
