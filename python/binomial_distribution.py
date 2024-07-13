import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import phimagic_prng32
import time

# Create an instance of the custom PRNG
prng = phimagic_prng32.mxws()

# Parameters
n = 100  # Number of rows (trials)
p = 0.7  # Probability of bouncing left or right
num_balls = 10000000  # Number of balls to drop
σ = np.sqrt(n*p*(1-p))  #Standard deviation

#Time seed 
current_time_seconds = int(time.time())
position_counts = prng.binomial(enable_seed = 1,  Seed = current_time_seconds, Ntrials = num_balls ,  n = n, p = p )


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

# Plotting the results
fig = plt.figure(facecolor='#002b36', figsize=(16, 9))
ax = fig.gca()
set_axis_color(ax)

x = np.arange(0, n + 1)

# Theoretical binomial distribution
binom_pmf = binom.pmf(x, n, p)
print('Binomial Distribution (Theoretical)', binom_pmf) 
plt.plot(x, binom_pmf, 'o-', label='Binomial Distribution (Theoretical)')

print('\nBinomial Distribution (Emperical)', position_counts)
plt.plot(x, position_counts, 'o-', label='Binomial Distribution (Emperical)')

plt.title('Galton Board: Binomial Distribution', color = 'white')
plt.xlabel('Number of Successes (Left Bounces)')
plt.ylabel('Probability')
plt.legend()
plt.show()
