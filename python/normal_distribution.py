import numpy as np
import phimagic_prng32
import time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def normal_pdf(x, mu, sigma):
  """
  Calculates the probability density function (PDF) of a normal distribution.

  Args:
      x: The value at which to evaluate the PDF.
      mu: The mean of the normal distribution.
      sigma: The standard deviation of the normal distribution.

  Returns:
      The probability density of the normal distribution at x.
  """

  # Calculate the first term
  term1 = 1 / (np.sqrt(2 * np.pi) * sigma)

  # Calculate the second term (exponent)
  term2 = np.exp(-0.5 * np.power(x - mu, 2) / (sigma * sigma))

  # Combine terms and return the PDF value
  return term1 * term2

def set_axis_color(ax):
    ax.set_facecolor('#002b36')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white') 
    
def func_approx(x, n, band = True):
    # Perform the Fourier Transform
    yf = np.fft.fft(x)
    # Truncate higher frequencies (approximation)
    num_components = int(n)# Adjust this to control the level of approximation
    yf_truncated = yf
    if(band == False):
        yf_truncated = np.zeros(len(yf), dtype=np.complex128)
        yf_truncated[num_components] = yf[num_components]
        yf_truncated[-num_components] = yf[-num_components]
    else:
        yf_truncated[num_components:-num_components] = 0
    # Perform the Inverse Fourier Transform to get the approximated function
    y_approx = np.fft.ifft(yf_truncated)
    return y_approx.real


# Create an instance of the custom PRNG
prng = phimagic_prng32.mxws(2)

# Parameters
mu = 0  # Mean

sigma = 1.
trials = 3000  # Number of balls to drop

#Time seed 
current_time_seconds = int(time.time())
normal = prng.normal(trials = trials, mean = mu, dev = sigma, enable_seed= True, Seed = current_time_seconds)

fig = plt.figure(facecolor='#002b36', figsize=(16, 9))
ax = fig.gca()
set_axis_color(ax)

x = np.linspace(mu - 6 * 1, mu + 6 * 1, len(normal));

y = normal_pdf(x, mu, sigma)# Normal distribution probability density function

#normal = func_approx(normal, 3, True)
deltax = x[1] - x[0]

if(trials > 30000):
    plt.step(x, normal, label='Galton Board ' + str(trials) +' balls', color = 'blue')
else:
    plt.bar(x, normal, width=deltax, label='Galton Board ' + str(trials) +' balls', color = 'blue', edgecolor = '#002b36')
        
plt.plot(x, y, label=f"Normal Distribution (μ={mu}, σ={sigma})", linewidth=1, color = 'orange')

plt.title('Galton Board', color = 'white')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(alpha=0.2)
plt.show()
