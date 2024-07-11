import sys
import os
from matplotlib import pylab as plt
import numpy as np

# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import phimagic_prng32
import phimagic_prng64

print("Module imported successfully")

# Create an instance of the custom PRNG
prng = phimagic_prng32.mxws(2)

print("Instance created successfully")

# Generate random numbers
for _ in range(10):
    print(prng.rng())
    
 
 # Generate an array of random numbers
size = 10000000

v1 = np.sqrt(np.pi)
random_array = prng.uniform(-v1, v1, size=size)

# Print the array
print("Generated random numbers:", random_array)

            
s1 = prng.sine_distribution(enable_seed = 1,  Seed = 10, Ntrials = 10000000, Ncycles = 1,  N_Integrations = 10,   Nbins = 3000)
s2 = prng.sine_distribution(enable_seed = 1,  Seed = 10, Ntrials = 10000000, Ncycles = 1,  N_Integrations = 10,   Nbins = 3000)

s3 = np.concatenate((s1, s2), axis=None)

def set_axis_color(ax):
    ax.set_facecolor('#002b36')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white') 

fig = plt.figure(facecolor='#002b36', figsize=(8, 8))
ax = fig.gca()
set_axis_color(ax)

x = np.linspace(0, len(s3),  len(s3)) 

plt.plot(x, s3) 


plt.title("sine distribution", color = 'white')
plt.xlabel("X", color = 'white')
plt.ylabel("Y", color = 'white')

plt.grid(alpha=0.4)

plt.show()