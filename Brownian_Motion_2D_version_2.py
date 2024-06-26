import numpy as np
from scipy.stats import shapiro, linregress
import matplotlib.pyplot as plt

#https://www.johndcook.com/blog/2021/07/13/random-fourier-series/

def simulate_brownian_motion_2d(num_terms=1000, spread = 0.001, seed = 10):
    """Simulates 2D Brownian motion using random Fourier series.

    Args:
        num_terms (int): Number of terms in the Fourier series.
         spread (float): difference between the start and end point.
         seed (int): seed for the RNG. 

    Returns:
        numpy.ndarray: Time points, X-coordinates, and Y-coordinates.
    """
    rng = np.random.default_rng(seed)
    xi_x = rng.uniform(-np.sqrt(np.pi), np.sqrt(np.pi), num_terms)  # Independent standard  uniform for X
    xi_y = rng.uniform(-np.sqrt(np.pi), np.sqrt(np.pi), num_terms)  # Independent standard uniform for Y
    
    t = np.linspace(0, 2 * np.pi, num_terms)
    # Calculate X-coordinates
    
    B_t_x = xi_x[0] * t / np.sqrt(2 * np.pi) * spread
    B_t_x += sum(np.sin(n * t / 2) * xi_x[n] / n for n in range(1, num_terms)) * 2 / np.sqrt(np.pi) 

    # Calculate Y-coordinates
       
    B_t_y = xi_y[0] * t / np.sqrt(2 * np.pi) * spread
    B_t_y += sum(np.sin(n * t / 2) * xi_y[n] / n for n in range(1, num_terms)) * 2 / np.sqrt(np.pi) 

    return t, B_t_x, B_t_y

# Example usage
t, B_t_x, B_t_y = simulate_brownian_motion_2d(num_terms=10000, spread = 0.0001, seed = None)

plt.figure(figsize=(8, 8))

# Different colors
start_color = 'blue'  
end_color = 'orange'
path_color = 'gray'

# Different marker styles
start_marker = 's'  # Square
end_marker = '*'    # Star


# Plot the path 
plt.plot(B_t_x, B_t_y, marker='o', markersize=2, linestyle='-', linewidth=0.5, color=path_color)  # Gray for path

# Plot the start point (green)
plt.plot(B_t_x[0], B_t_y[0], marker='o', markersize=8, color=start_color, label='Start')

# Plot the end point (red)
plt.plot(B_t_x[-1], B_t_y[-1], marker='o', markersize=8, color=end_color, label='End')

plt.title("Simulated 2D Brownian Motion")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()  # Show the legend for start/end points
plt.grid(alpha=0.4)
plt.axis('equal')
plt.show()

# ... (your existing Brownian motion simulation code)

# 1. Test for independent increments
increments = np.diff(B_t_x)  # Calculate increments in x-direction (similar for y)
correlations = [np.corrcoef(increments[:-i], increments[i:])[0, 1] for i in range(1, len(increments) // 2)]
plt.plot(correlations)
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.title("Correlation of Increments vs. Lag")
plt.show()  # Should be close to zero for most lags

# 2. Test for normally distributed increments
plt.hist(increments, bins=50, density=True, alpha=0.6)
mu, sigma = np.mean(increments), np.std(increments)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2)), 'r')
plt.title("Histogram of Increments")
plt.xlabel("Increment")
plt.ylabel("Density")
plt.show()  # Should resemble a normal distribution

# Shapiro-Wilk test for normality
stat, p = shapiro(increments)
print('Shapiro-Wilk Test: Statistic=%.3f, p=%.3f' % (stat, p))  # If p > 0.05, we cannot reject normality

# 3. Test for linear variance scaling
variances = [np.var(B_t_x[i:]) for i in range(len(B_t_x))]  # Calculate variances for different time intervals
times = np.arange(len(B_t_x))
slope, intercept, r_value, p_value, std_err = linregress(times, variances)
plt.plot(times, variances)
plt.plot(times, slope * times + intercept, 'r')
plt.title("Variance vs. Time")
plt.xlabel("Time")
plt.ylabel("Variance")
plt.show()  # Should be approximately linear

# 4. Visual inspection of the path (already done in the simulation code)
