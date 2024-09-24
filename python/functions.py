import numpy as np
from constants import *
import pyfftw
import multiprocessing
import time
import progressbar
from matplotlib import mlab
from matplotlib.ticker import ScalarFormatter

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


from numba import njit

@njit
def apply_projection(tmp, psi_list, dx):
    for i in range(len(psi_list)):
        psi = psi_list[i]
        coef = np.vdot(psi, tmp)
        tmp -= coef * psi * dx
    return tmp

def apply_projection_V1(tmp, psi_list, dx):
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
