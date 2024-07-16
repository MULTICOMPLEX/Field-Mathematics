import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

import phimagic_prng32
import time

# Create an instance of the custom PRNG
prng = phimagic_prng32.mxws()

def gamma_monte_carlo(x, num_samples=1000000):
    # Generate uniform random samples
    current_time_seconds = int(time.time())
    U = prng.Gamma(enable_seed = True, Seed = current_time_seconds, Ntrials = num_samples, x = x)  
    return U

# Test values for x
test_values = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

# Calculate Gamma function using scipy and Monte Carlo method
scipy_gamma = [gamma(x) for x in test_values]
monte_carlo_gamma = [gamma_monte_carlo(x) for x in test_values]

# Print comparison
for x, g_scipy, g_monte_carlo in zip(test_values, scipy_gamma, monte_carlo_gamma):
    print(f"Gamma({x})")
    print(f"  SciPy:         {g_scipy}")
    print(f"  Monte Carlo:   {g_monte_carlo}")
    print(f"  Difference:    {abs(g_scipy - g_monte_carlo)}\n")

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

# Plot the results for visual comparison
fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)
plt.plot(test_values, scipy_gamma, label='SciPy Gamma', marker='o')
plt.plot(test_values, monte_carlo_gamma, label='Monte Carlo Gamma', marker='x')
plt.xlabel('x')
plt.ylabel('Gamma(x)')
plt.title('Gamma Function Comparison', color = 'white')
plt.legend()
plt.grid(True)
plt.show()
