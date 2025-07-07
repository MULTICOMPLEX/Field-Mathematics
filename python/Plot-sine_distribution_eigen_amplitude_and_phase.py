import json
import matplotlib.pyplot as plt
import numpy as np

def hausdorff_dimension(data, min_box_size=1e-5, max_box_size=1e-1, num_scales=50):
    """
    Estimate the Hausdorff dimension of a 1D vector using the box-counting method.
    
    Parameters:
        data (np.ndarray): A 1D array of points.
        min_box_size (float): Minimum box size for box counting.
        max_box_size (float): Maximum box size for box counting.
        num_scales (int): Number of scales (box sizes) between min and max.
    
    Returns:
        float: The estimated Hausdorff dimension.
    """
    # Sort the data
    data = np.sort(data)
    
    # Generate box sizes (scales) in logarithmic space
    scales = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num_scales)
    counts = []

    # Iterate over each box size
    for scale in scales:
        # Compute the number of intervals (boxes) needed to cover the data
        min_val = data[0]
        max_val = data[-1]
        num_boxes = int(np.ceil((max_val - min_val) / scale))
        
        # Initialize a set to track which boxes contain at least one point
        covered_boxes = set()
        
        # Determine which boxes are covered by the points
        for point in data:
            box_index = int((point - min_val) / scale)
            covered_boxes.add(box_index)
        
        # Record the number of covered boxes
        counts.append(len(covered_boxes))

    # Fit a line to the log-log plot of box sizes versus counts
    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    hausdorff_dim = -coeffs[0]  # The slope of the line is the negative Hausdorff dimension
    
    return hausdorff_dim

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



def plot_json_data(filepath, x_key, y_key, title=None, xlabel=None, ylabel=None):
    """
    Plots data from a JSON file using matplotlib.

    Args:
        filepath: Path to the JSON file.
        x_key: Key in the JSON objects to use for the x-axis.
        y_key: Key in the JSON objects to use for the y-axis.
        title:  Optional title for the plot.
        xlabel: Optional label for the x-axis.
        ylabel: Optional label for the y-axis.
    """

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return

    x_values = [item[x_key] for item in data if x_key in item and y_key in item ] # handles missing keys gracefully
    y_values = [item[y_key] for item in data if x_key in item and y_key in item]


    if not x_values or not y_values:  # Check if any data was extracted after handling potential missing keys.
        print("Error: No valid data points found for the specified keys.")
        return
    
    dimension = hausdorff_dimension(y_values)

    if dimension is not None:
        print(ylabel, f"Estimated Hausdorff dimension: {dimension}")
    else:
        print("Not enough data points for Hausdorff dimension calculation.")
    
    fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
    ax = fig.gca()
    set_axis_color(ax)

    plt.step(x_values, y_values, linestyle='-')  # Added markers for better visualization


    if title:
        plt.title(title, color = 'white')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    n = 5
    every_nth = x_values[::n] 
    grid_points = every_nth
    ax.grid(color='gray')
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_ticks(grid_points)


f = 'phase_amplitude_values - 3100.json'

plot_json_data(f, 'Trial', 'Amplitude', title='Amplitude vs. Frequency', xlabel='Frequency', ylabel='Amplitude')
plot_json_data(f, 'Trial', 'Phase', title='Phase vs. Frequency', xlabel="Frequency", ylabel="Phase")

plt.show()