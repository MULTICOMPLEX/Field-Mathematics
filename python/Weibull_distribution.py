import numpy as np
import matplotlib.pyplot as plt
import phimagic_prng64
import time

# Create an instance of the custom PRNG
prng = phimagic_prng64.mxws()

# Parameters for the Weibull distribution
shape_param = 1.9  # k
scale_param = 2.0  # lambda

# Number of samples
n_samples = 10000000
size = 3000
sp = 10

current_time_seconds = int(time.time())     
w = prng.Weibull(enable_seed = True, Seed = current_time_seconds, Ntrials = n_samples,  size = size, shape_param = shape_param,  scale_param=scale_param, sp = sp)


# Plot the PDF of the Weibull distribution
x = np.linspace(0,  sp,  len(w))
pdf = (shape_param / scale_param) * (x / scale_param)**(shape_param - 1) * np.exp(-(x / scale_param)**shape_param)

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

fig = plt.figure(figsize=(10, 6), facecolor='#002b36')
ax = fig.gca()
set_axis_color(ax)

x = np.linspace(0,  sp, len(w))
plt.step(x, w, label='Weibull PDF')

x = np.linspace(0,  sp, size)
plt.plot(x, pdf,  lw=1,  color = 'orange', label='Weibull PDF')
# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Weibull Distribution', color = 'white')
plt.legend()

# Show plot
plt.show()
