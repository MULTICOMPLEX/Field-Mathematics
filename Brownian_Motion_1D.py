import numpy as np
import matplotlib.pyplot as plt

#https://www.johndcook.com/blog/2021/07/13/random-fourier-series/

def simulate_brownian_motion(num_terms=1000, spread = 0.001, seed = 10):
    """Simulates Brownian motion using a random Fourier series.

    Args:
        num_terms (int): Number of terms in the Fourier series.
        spread (float): difference between the start and end point.
        seed (int): seed for the RNG. 

    Returns:
        numpy.ndarray: Time points and corresponding Brownian motion values.
    """
    
    t = np.linspace(0, 2 * np.pi, num_terms)
    
    rng = np.random.default_rng(seed)
    xi = rng.normal(0, 1, num_terms)  # Independent standard normal variables

    B_t = xi[0] * t / np.sqrt(2 * np.pi)
    B_t += sum((1.0 / spread) * np.sin(n * t) * xi[n] / n for n in range(1, num_terms)) * 2 / np.sqrt(np.pi)

    return t, B_t * spread 

# Example usage
t, B_t = simulate_brownian_motion(num_terms=1000, spread = 1)

plt.figure(figsize=(10, 6))
plt.plot(t, B_t)
plt.title("Simulated Brownian Motion")
plt.xlabel("Time")
plt.ylabel("B(t)")
plt.grid(alpha=0.4)
plt.show()
