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
    xi = rng.uniform(-np.sqrt(np.pi), np.sqrt(np.pi), num_terms)  # Independent standard normal variables

    B_t = xi[0] * t / np.sqrt(2 * np.pi) * spread
    B_t += sum(np.sin(n * t / 2) * xi[n] / n for n in range(1, num_terms)) * 2 / np.sqrt(np.pi)

    return t, B_t 

plt.figure(figsize=(10, 6))

# Example usage
for _ in range(10):
    t, B_t = simulate_brownian_motion(num_terms=1000, spread = 0.1, seed = None)
    plt.plot(t, B_t)

plt.plot(t, B_t)
plt.title("Simulated Brownian Motion, RNG Uniform")
plt.xlabel("Time")
plt.ylabel("B(t)")
plt.grid(alpha=0.4)
plt.show()

