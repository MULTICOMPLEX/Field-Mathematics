import matplotlib.pyplot as plt
import numpy as np

def n_random_walks(n_walks, n_steps):
    """Simulates n random walks of +1, -1 steps over n samples."""
    walks = np.random.choice([-1, 1], size=(n_walks, n_steps)).cumsum(axis=1)
    return walks

# Simulation parameters
n_walks = 8
n_steps = 10000000
plot_interval = 10000
window_size = 5000  

# Create figure and axes
plt.ion()  
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize empty lines
lines = []
for _ in range(n_walks):
    line, = ax.plot([], [])  
    lines.append(line)

# Initial plot limits (just for the x-axis)
ax.set_xlim([0, window_size])
ax.autoscale(enable=True, axis='y')  # Enable y-axis autoscaling

# Add labels and title
ax.set_xlabel('Steps')
ax.set_ylabel('Position')
ax.grid(True, which='both', alpha=0.4)
ax.set_title(f'Random Walks (Scrolling Window)')

# Main simulation loop
walks = n_random_walks(n_walks, n_steps)  

for i in range(0, n_steps, plot_interval):
    # Update line data
    for j, line in enumerate(lines):
        start_idx = max(0, i - window_size)
        end_idx = i + plot_interval
        x_data = list(range(start_idx, end_idx))  # Use full range
        y_data = walks[j][start_idx:end_idx]
        line.set_data(x_data, y_data)

    # Adjust x-axis limits for scrolling
    ax.set_xlim([max(0, i - window_size + plot_interval), i + plot_interval])
    
    # Rescale y-axis automatically
    ax.relim() 
    ax.autoscale_view()

    # Redraw the plot
    fig.canvas.draw()
    plt.pause(0.001)  # Adjust for smoother/faster animation

# Keep the plot open after the loop


# Create a histogram of the final positions
final_positions = [walk[-1] for walk in walks]
plt.figure()  # Create a new figure for the histogram
plt.hist(final_positions, bins=30, density=True, alpha=0.7, label='Final Positions')

# Overlay a normal distribution fit for comparison
mu, sigma = np.mean(final_positions), np.std(final_positions)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2)), 'r-', label='Normal Fit')

plt.xlabel('Final Position')
plt.ylabel('Density')
plt.title('Distribution of Final Positions')
plt.legend()
plt.ioff()  # Turn off interactive mode
plt.show()
