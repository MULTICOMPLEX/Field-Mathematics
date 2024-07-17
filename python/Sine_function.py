import numpy as np
from scipy.special import gamma
import phimagic_prng32
import time

# Create an instance of the custom PRNG
prng = phimagic_prng32.mxws()

def gamma_monte_carlo(x, num_samples=100000000):
    # Generate uniform random samples
    current_time_seconds = int(time.time())
    U = prng.Gamma(enable_seed = True, Seed = current_time_seconds, Ntrials = num_samples, x = x)  
    return U

def sine_using_gamma(z):
    # Compute sine using the derived formula
    z /= np.pi
    sin_pi_z = np.pi / (gamma_monte_carlo(z) * gamma_monte_carlo(1 - z))
    #sin_pi_z = np.pi / (gamma(z) * gamma(1 - z))
    return sin_pi_z 

# Test values
z_values = [0.5, 1, 1.5, 2.3]

# Compute and print the results
for z in z_values:
    sin_value = sine_using_gamma(z)
    print('sine(z)  ', np.sin(z))
    print('sine_g(z)', sin_value, '\n')
