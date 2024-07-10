import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Simulation parameters
n_walks = 8
plot_interval = 10000
window_size = 5000

# Initialize the main window
root = tk.Tk()
root.title("Random Walk Simulation")

# Create the figure and axes for the plot
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
ax.set_title(str(n_walks) +  ' Random Walks')

# Embed the plot in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Initialize the starting positions and walks
positions = np.zeros((n_walks,), dtype=int)
walks = [[] for _ in range(n_walks)]

# Control flag for the simulation
running = False
current_step = 0

def reset_simulation():
    global positions, walks, current_step
    positions = np.zeros((n_walks,), dtype=int)
    walks = [[] for _ in range(n_walks)]
    current_step = 0
    for line in lines:
        line.set_data([], [])  # Clear the line data
    ax.set_xlim([0, window_size])
    ax.relim()
    ax.autoscale_view()
    canvas.draw()

def start_simulation():
    global running
    if not running:
        running = True
        simulate()

def stop_simulation():
    global running
    running = False

def simulate():
    global positions, walks, current_step

    if not running:
        return

    # Generate new steps for the interval
    new_steps = np.random.choice([-1, 1], size=(n_walks, plot_interval))

    # Update positions incrementally
    for j in range(n_walks):
        for step in new_steps[j]:
            positions[j] += step
            walks[j].append(positions[j])

        # Update line data
        start_idx = max(0, len(walks[j]) - window_size)
        end_idx = len(walks[j])
        x_data = list(range(start_idx, end_idx))
        y_data = walks[j][start_idx:end_idx]
        lines[j].set_data(x_data, y_data)

    # Adjust x-axis limits for scrolling
    ax.set_xlim([max(0, len(walks[0]) - window_size), len(walks[0])])

    # Rescale y-axis automatically
    ax.relim()
    ax.autoscale_view()

    # Redraw the plot
    canvas.draw()

    current_step += plot_interval
    root.after(1, simulate)  # Continue simulation after 1 millisecond

# Create Start, Stop, and Reset buttons
start_button = tk.Button(root, text="Start", command=start_simulation)
start_button.pack(side=tk.LEFT)

stop_button = tk.Button(root, text="Stop", command=stop_simulation)
stop_button.pack(side=tk.LEFT)

reset_button = tk.Button(root, text="Reset", command=reset_simulation)
reset_button.pack(side=tk.LEFT)

# Start the simulation automatically
start_simulation()

# Run the tkinter main loop
root.mainloop()
