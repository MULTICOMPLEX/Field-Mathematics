import matplotlib.pyplot as plt
import numpy as np

def n_random_walks(n_walks, n_steps):
    """
    Simulates n random walks of +1, -1 steps over n samples.

    Args:
        n_walks: The number of random walks to simulate.
        n_steps: The number of steps in each random walk.

    Returns:
        A list of lists, where each inner list represents the positions of a
        single random walk at each step.
    """
    walks = []
    for _ in range(n_walks):
        walk = np.random.choice([-1, 1], size=n_steps).cumsum()
        walks.append(walk.tolist())
    return walks

# Simulate 5 random walks of 100 steps each
walks = n_random_walks(5, 1000000)

# Choose the interval for plotting (e.g., every 10 steps)
plot_interval = 1000

# Create a figure and axes
plt.figure(figsize=(10, 6))

# Plot each random walk, marking every nth step
for walk in walks:
    # Create x values for plotting (0, plot_interval, 2*plot_interval, etc.)
    x_values = range(0, len(walk), plot_interval)
    
    # Slice the walk to get the y values at those intervals
    y_values = walk[::plot_interval] 
    
    # Plot the points with markers (dots in this case)
    plt.plot(x_values, y_values)  

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Position')
plt.grid(True, which='both', alpha = 0.4)
plt.title(f'Random Walks (Plotted Every {plot_interval} Steps)')

# Show the plot
plt.show()
