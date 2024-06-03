import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian_motion(num_terms=1000, interval=2*np.pi,  spread = 0.001, seed = 10):
    """Simulates Brownian motion using a random Fourier series.

    Args:
        num_terms (int): Number of terms in the Fourier series.
        interval (float): Length of the interval (default is 2π).

    Returns:
        numpy.ndarray: Time points and corresponding Brownian motion values.
    """
    
    t = np.linspace(0, interval, num_terms)
    
    rng = np.random.default_rng(seed)
    xi = rng.normal(0, 1, num_terms)  # Independent standard normal variables

    B_t = xi[0] * t
    for k in range(1, num_terms):
        B_t += (1.0 / spread) * np.sqrt(2) * xi[k] * np.sin(k * np.pi * t / interval) / (k * np.pi / interval)

    return t, B_t  * spread 

# Example usage
t, B_t = simulate_brownian_motion(spread = 0.1)

plt.figure(figsize=(10, 6))
plt.plot(t, B_t)
plt.title("Simulated Brownian Motion")
plt.xlabel("Time")
plt.ylabel("B(t)")
plt.grid(alpha=0.4)
plt.show()
