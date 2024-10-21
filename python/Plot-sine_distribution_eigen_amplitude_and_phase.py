import json
import matplotlib.pyplot as plt

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

    plt.grid(True)  # Add a grid for better readability


f = 'phase_amplitude_values - 3000.json'

plot_json_data(f, 'Trial', 'Amplitude', title='Amplitude vs. Trial', xlabel='Trial', ylabel='Amplitude')
plot_json_data(f, 'Trial', 'Phase', title="Phase vs. Trial", xlabel="Trial", ylabel="Phase")

plt.show()