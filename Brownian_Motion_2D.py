import numpy as np
import matplotlib.pyplot as plt

def simulate_brownian_motion_2d(num_terms=1000, interval=2 * np.pi):
    """Simulates 2D Brownian motion using random Fourier series.

    Args:
        num_terms (int): Number of terms in the Fourier series.
        interval (float): Length of the interval (default is 2π).

    Returns:
        numpy.ndarray: Time points, X-coordinates, and Y-coordinates.
    """
    t = np.linspace(0, interval, num_terms)
    xi_x = np.random.normal(0, 1, num_terms)  # Independent standard normal for X
    xi_y = np.random.normal(0, 1, num_terms)  # Independent standard normal for Y

    # Calculate X-coordinates
    B_t_x = xi_x[0] * t
    for k in range(1, num_terms):
        B_t_x += np.sqrt(2) * xi_x[k] * np.sin(k * np.pi * t / interval) / (k * np.pi / interval)

    # Calculate Y-coordinates
    B_t_y = xi_y[0] * t
    for k in range(1, num_terms):
        B_t_y += np.sqrt(2) * xi_y[k] * np.sin(k * np.pi * t / interval) / (k * np.pi / interval)

    return t, B_t_x, B_t_y

# Example usage
t, B_t_x, B_t_y = simulate_brownian_motion_2d()

plt.figure(figsize=(8, 8))

# Plot the path (excluding start and end points)
plt.plot(B_t_x[1:-1], B_t_y[1:-1], marker='o', markersize=2, linestyle='-', linewidth=0.5, color='gray')  # Gray for path

# Plot the start point (green)
plt.plot(B_t_x[0], B_t_y[0], marker='o', markersize=8, color='green', label='Start')

# Plot the end point (red)
plt.plot(B_t_x[-1], B_t_y[-1], marker='o', markersize=8, color='red', label='End')

plt.title("Simulated 2D Brownian Motion")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()  # Show the legend for start/end points
plt.grid(alpha=0.4)
plt.axis('equal')
plt.show()
