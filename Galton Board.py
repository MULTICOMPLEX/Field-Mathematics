import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import norm  # Example: fitting a normal distribution
import phimagic_prng32

prng = phimagic_prng32.mxws(1)

# Simulation parameters
n_walks = 200
plot_interval = 1000
window_size = 5000
n_plotted_walks = 8

# Initialize the main window
root = tk.Tk()
root.title("Galton Board Simulation")
root.geometry("1200x900")  # Set the size of the main window

# Create a frame to hold the plot and buttons
frame = tk.Frame(root)
frame.pack(padx= 0, pady= 0, fill=tk.BOTH, expand=True)

# Create the figure and axes for the random walk plot
fig, (ax_walks, ax_hist) = plt.subplots(2, 1, figsize=(16, 5), facecolor='#002b36')  # Adjust the figure size if needed
plt.subplots_adjust(left=0.075, right=0.96, bottom=0.06, top=0.96, wspace=0.2, hspace=0.25)

# Initialize empty lines
lines = []
for _ in range(n_plotted_walks):  # Display only n walks on the plot
    line, = ax_walks.plot([], [],  linewidth=0.8)
    lines.append(line)

# Initial plot limits (just for the x-axis)
ax_walks.set_xlim([0, window_size])
ax_walks.autoscale(enable=True, axis='y')  # Enable y-axis autoscaling

# Add labels and title
ax_walks.set_xlabel('Steps')
ax_walks.set_ylabel('Position')
ax_walks.grid(True, which='both', alpha=0.4)
ax_walks.set_title(str(n_plotted_walks) + ' out of ' + str(n_walks) + ' Random Walks', color = 'white')

ax_walks.set_facecolor('#002b36')
ax_walks.xaxis.label.set_color('white')
ax_walks.yaxis.label.set_color('white')
ax_walks.tick_params(colors='white')
ax_walks.spines['left'].set_color('white')
ax_walks.spines['bottom'].set_color('white')
ax_walks.spines['top'].set_color('white')
ax_walks.spines['right'].set_color('white') 

ax_hist.set_facecolor('#002b36')
ax_hist.tick_params(colors='white')
ax_hist.spines['left'].set_color('white')
ax_hist.spines['bottom'].set_color('white')
ax_hist.spines['top'].set_color('white')
ax_hist.spines['right'].set_color('white') 


# Embed the plot in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=0)  # Adjust padx and pady as needed

# Create a frame for the buttons
button_frame = tk.Frame(frame)
button_frame.pack(side=tk.BOTTOM, pady=10)

# Initialize the starting positions and walks
positions = np.zeros((n_walks,), dtype=int)
walks = [[] for _ in range(n_plotted_walks)]  # Display only n walks on the plot
all_positions = []  # To store all positions over time

# Control flags for the simulation
running = False
current_step = 0
horizontal_scroll = True

def reset_simulation():
    global positions, walks, current_step, all_positions
    all_positions = [] 
    positions = np.zeros((n_walks,), dtype=int)
    walks = [[] for _ in range(8)]
    current_step = 0
    for line in lines:
        line.set_data([], [])  # Clear the line data
    if horizontal_scroll:
        ax_walks.set_xlim([0, window_size])
    else:
        ax_walks.set_ylim([0, window_size])
    ax_walks.relim()
    ax_walks.autoscale_view()
    ax_hist.clear()
    canvas.draw()

def start_simulation():
    global running
    if not running:
        running = True
        simulate()

def stop_simulation():
    global running
    running = False

def toggle_scroll_direction():
    global horizontal_scroll
    horizontal_scroll = not horizontal_scroll
    if horizontal_scroll:
        ax_walks.set_xlabel('Steps')
        ax_walks.set_ylabel('Position')
        ax_walks.set_xlim([0, window_size])
        ax_walks.set_ylim(auto=True)
    else:
        ax_walks.set_xlabel('Position')
        ax_walks.set_ylabel('Steps')
        ax_walks.set_xlim(auto=True)
        ax_walks.set_ylim([0, window_size])
    reset_simulation()

def simulate():
    global positions, walks, current_step, all_positions

    if not running:
        return

    # Generate new steps for the interval
    #new_steps = np.random.choice([-1, 1], size=(n_walks, plot_interval))
    new_steps = np.zeros((n_walks, plot_interval))
    for i in range(n_walks):
        new_steps[i] = prng.choice([-1, 1], size=plot_interval)
    # Update positions incrementally
    for j in range(n_walks):
        for step in new_steps[j]:
            positions[j] += step
            if j < n_plotted_walks:  # Update only the first nwalks for plotting
                walks[j].append(positions[j])
    # Accumulate all positions
    all_positions.extend(positions.tolist())

    # Update line data
    for j in range(min(n_plotted_walks, n_walks)):
        if horizontal_scroll:
            start_idx = max(0, len(walks[j]) - window_size)
            end_idx = len(walks[j])
            x_data = list(range(start_idx, end_idx))
            y_data = walks[j][start_idx:end_idx]
            lines[j].set_data(x_data, y_data)
        else:
            start_idx = max(0, len(walks[j]) - window_size)
            end_idx = len(walks[j])
            x_data = walks[j][start_idx:end_idx]
            y_data = list(range(start_idx, end_idx))
            y_data = y_data[::-1]
            lines[j].set_data(x_data, y_data)

    # Adjust axis limits for scrolling
    if horizontal_scroll:
        ax_walks.set_xlim([max(0, len(walks[0]) - window_size), len(walks[0])])
    else:
        ax_walks.set_ylim([max(0, len(walks[0]) - window_size), len(walks[0])])

    # Rescale axis automatically
    ax_walks.relim()
    ax_walks.autoscale_view()

    # Update the histogram
    ax_hist.clear()
    counts, bins, patches = ax_hist.hist(all_positions, bins=80, density=True, alpha=0.7, label='Positions')
    ax_hist.xaxis.label.set_color('white')
    ax_hist.yaxis.label.set_color('white')
    
    # Assuming you want to fit a normal distribution
    mu, std = norm.fit(all_positions)  # Find mean and standard deviation for the fit

    # Generate x-axis points for the fitted curve (covering bin centers)
    x_fit = np.linspace(min(bins), max(bins), len(bins))

    # Calculate the probability density function (PDF) for the fitted distribution
    y_fit = norm.pdf(x_fit, mu, std)
    
    # Plot the fitted curve on the same axes as the histogram
    ax_hist.plot(x_fit, y_fit, 'r-', label='Normal Distribution Fit')
    
    ax_hist.set_xlabel('Position')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Distribution of Positions', color = 'white')
    ax_hist.legend()

    # Redraw the plot
    canvas.draw()

    current_step += plot_interval
    root.after(1, simulate)  # Continue simulation after 1 millisecond

# Create Start, Stop, Reset, and Toggle Scroll Direction buttons in the button frame
start_button = tk.Button(button_frame, text="Start", command=start_simulation)
start_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(button_frame, text="Stop", command=stop_simulation)
stop_button.pack(side=tk.LEFT, padx=10)

reset_button = tk.Button(button_frame, text="Reset", command=reset_simulation)
reset_button.pack(side=tk.LEFT, padx=10)

toggle_button = tk.Button(button_frame, text="Toggle Scroll Direction", command=toggle_scroll_direction)
toggle_button.pack(side=tk.LEFT, padx=10)

# Start the simulation automatically
start_simulation()

# Run the tkinter main loop
root.mainloop()
