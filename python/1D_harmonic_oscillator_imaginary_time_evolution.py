import numpy as np
import time
import progressbar
import pyfftw
import multiprocessing

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import hsv_to_rgb

import phimagic_prng32

from visuals import *
from constants import *
from functions import *
from scipy.stats import multivariate_normal
from data import *

n = 2048*2

S = {
    "name": "Q0",
    "mode": "two tunnel+-",
    "total time": 4 * femtoseconds,
    "store steps": 20,
    "σ": 0.7 * Å,
    "v0": 60,  # T momentum
    "V0": 2,  # barrier voltage
    "initial offset": 0,
    "N": n,
    "dt": 0.5,
    "x0": 0,  # barrier x
    "x1": 3,
    "x2": 12,
    "extent": 20 * Å,  # 150, 30
    "extentN": -75 * Å,
    "extentP": +85 * Å,
    "Number of States": 13,
    "imaginary time evolution": True,
    "animation duration": 10,  # seconds
    "save animation": True,
    "fps": 30,
    "path data": "./data/",
    "path save": "./gifs/",
    "title": "1D harmonic oscillator imaginary time evolution"
}

X = np.linspace(-S["extent"]/2, S["extent"]/2, S["N"])
dx = X[1] - X[0]


# Define parameters
k = 1  # Harmonic force constant
alpha = 0.1  # Anharmonic coefficient (x^3 term)
beta = 0.01  # Anharmonic coefficient (x^4 term)

def V2(X):
# Calculate potential energy for harmonic and anharmonic cases
    V_harmonic = 0.5 * k * X**2
    V_anharmonic = 0.5 * k * X**2 + alpha * X**3 + beta * X**4
    return V_harmonic

#potential energy operator
def V(X):
    return 2 * m_e * (2 * np.pi / (0.6 * femtoseconds))**2 * X**2
    

#initial waveform
def 𝜓0_gaussian_wavepacket_1D(X, σ, v0, x0):
    v0 = v0 * Å / femtoseconds
    p_x0 = const["m_e"] * v0
    mean = x0
    cov = 4
    # Create the multivariate normal distribution object
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(X) 
    Z = Z * np.exp(1j*(p_x0*X))
    Zmax = np.amax(np.abs(Z))
    Z /= Zmax 
    return Z 

V = V(X)
#V = V / np.max(V)

Vmin = np.amin(V)
Vmax = np.amax(V)


dt_store = S["total time"]/S["store steps"]
Nt_per_store_step = int(np.round(dt_store / S["dt"]))

dt = dt_store/Nt_per_store_step

Ψ = np.zeros((S["store steps"] + 1, *([S["N"]])), dtype=np.complex128)

p2 = fft_frequencies_1D(S["N"], dx, const["hbar"])

if (S["imaginary time evolution"]):
    Ur = np.exp(-0.5*(dt/const["hbar"])*V)
    Uk = np.exp(-0.5*(dt/(const["m"]*const["hbar"]))*p2)

else:
    Ur = np.exp(-0.5j*(dt/const["hbar"])*V())
    Uk = np.exp(-0.5j*(dt/(const["m"]*const["hbar"]))*p2)


#tmp = pyfftw.empty_aligned(S["N"],  dtype='complex64')
#c = pyfftw.empty_aligned(S["N"], dtype='complex64')


print("store_steps", S["store steps"])
print("Nt_per_store_step", Nt_per_store_step)


psi_0 = norm(𝜓0_gaussian_wavepacket_1D(X, S["σ"], S["v0"], S["initial offset"]), dx)
#n = 1
#x = np.arange(len(X))
#Z = np.exp(1j * n/len(X) * 2 * np.pi * x)
#psi_0 = norm(Z, dx)

standard_dev = 1
#psi_0 = 1e-12j * norm(np.random.normal(0, standard_dev, size=len(X)), dx)
#psi_0 = 1e-12j * norm(np.random.uniform(0, 1, size=len(X)), dx) 

# Calculate Frequencies (we asume a sample rate of one)
# Use fft functions for real output (-> hermitian spectrum)


#Time seed 
current_time_seconds = int(time.time())
rng = np.random.default_rng(current_time_seconds)       #numpy PRNG
prng = phimagic_prng32.mxws(current_time_seconds)  #Phimagic fastest PRNG


f = rfftfreq(len(X)) 
v = np.sqrt(np.pi)
sr = prng.uniform(-v, v, size=len(f))  
si = prng.uniform(-v, v, size=len(f))  

fmin = 0
beta = - 4 # -2 = Violet noise, differentiated white noise
psi_0 = norm(powerlaw_psd_gaussian(beta, n, fmin, f, sr, si) + 1j * powerlaw_psd_gaussian(beta, n, fmin, f, sr, si), dx)    

#psi_0 = second_order_diff_noise( len(psi_0), dx) 
 
 
fig = plt.figure(facecolor='#002b36', figsize=(6, 6))
plt.title('psi_0', color = 'white') 

ax = fig.gca()
set_axis_color(ax)
plt.grid(True)

real_plot, = ax.plot(X/Å, np.real(psi_0), label='$Re|\\psi(0)|$')
imag_plot, = ax.plot(X/Å, np.imag(psi_0), label='$Im|\\psi(0)|$')
abs_plot, = ax.plot(X/Å, np.abs(psi_0), label='$|\\psi(0)|$')
leg = ax.legend(facecolor='#002b36', loc='lower left')
for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())


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


hbar = 1.054571817e-34    # Reduced Planck constant in J*s
m = 9.10938356e-31        # Mass of electron in kg
energies = Energies(V, Ψ, p2, hbar, m)
print("\nenergy =\n", energies.reshape(-1, 1))


Ψ /= np.amax(np.abs(Ψ))


fig = plt.figure(facecolor='#002b36', figsize=(6, 6))
plt.title('Quantum Ground State', color = 'white') 

ax = fig.gca()
set_axis_color(ax)
plt.grid(True)

index = 0

if Vmax-Vmin != 0:
        potential_plot = ax.plot(X/Å, (V + Vmin)/(Vmax-Vmin), label='$V(x)$')
else:
        potential_plot = ax.plot(X/Å, V, label='$V(x)$')  
real_plot, = ax.plot(X/Å, np.real(Ψ[index]), label='$Re|\\psi(x)|$')
imag_plot, = ax.plot(X/Å, np.imag(Ψ[index]), label='$Im|\\psi(x)|$')
abs_plot, = ax.plot(X/Å, np.abs(Ψ[index]), label='$|\\psi(x)|$')

ax.set_facecolor('#002b36')

leg = ax.legend(facecolor='#002b36', loc='lower left')
for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())

fig = plt.figure(facecolor='#002b36', figsize=(6, 6))
plt.title('Ψ[-1]', color = 'white') 

ax = fig.gca()
set_axis_color(ax)
plt.grid(True)

index = -1
if Vmax-Vmin != 0:
        potential_plot = ax.plot(X/Å, (V + Vmin)/(Vmax-Vmin), label='$V(x)$')
else:
        potential_plot = ax.plot(X/Å, V, label='$V(x)$')  
real_plot, = ax.plot(X/Å, np.real(Ψ[index]), label='$Re|\\psi(x)|$')
imag_plot, = ax.plot(X/Å, np.imag(Ψ[index]), label='$Im|\\psi(x)|$')
abs_plot, = ax.plot(X/Å, np.abs(Ψ[index]), label='$|\\psi(x)|$')

ax.set_facecolor('#002b36')

leg = ax.legend(facecolor='#002b36', loc='lower left')
for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
plt.show()



# visualize the time dependent simulation
animate_1D(Ψ = Ψ, X=X, V = V, Vmin = Vmin, Vmax = Vmax, xlim=[-S["extent"]/2/Å, 
S["extent"]/2/Å], ylim=[-1, 1.1], animation_duration=S["animation duration"], fps=S["fps"], 
total_time = S["total time"], store_steps = S["store steps"], energies = energies,
        save_animation=S["save animation"], title=S["title"]+" "+str(S["Number of States"])+" states", 
        path_save=S["path save"])
