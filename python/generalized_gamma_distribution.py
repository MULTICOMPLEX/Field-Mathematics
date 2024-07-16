import numpy as np
import matplotlib.pyplot as plt
import phimagic_prng64
import time

# Create an instance of the custom PRNG
prng = phimagic_prng64.mxws()

# Parameters for the Weibull distribution
shape_param = 2.9  # k
scale_param = 1.8  # lambda

ntrials = 100000
sp = 12


current_time_seconds = int(time.time())     



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

x = np.linspace(0,  sp, 2000)
#plt.step(x, w, label='Weibull PDF')


from scipy.stats import gamma, dweibull

#If p = 1, the generalised gamma becomes the gamma distribution.
tg = prng.GeneralizedGammaDistribution(enable_seed = True, seed = current_time_seconds, ntrials = ntrials, shape_d = shape_param, shape_p = 1.0,  scale = scale_param,  x=x)
plt.step(x, gamma.pdf(x, shape_param, scale=scale_param), lw=1, color = 'orange', label='Gamma PDF')

#If d = p , then the generalized gamma distribution becomes the Weibull distribution.
#tg = prng.GeneralizedGammaDistribution(shape_d = shape_param, shape_p = shape_param,  scale = scale_param,  x = x)  /  2
#plt.step(x, dweibull.pdf(x, shape_param, scale=scale_param), lw=1, color = 'orange', label='Weibull PDF')

plt.step(x,  tg, lw=1, color = 'green', label='Gen Gamma PDF')

# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Generalized Gamma Distribution', color = 'white')
plt.legend()

# Show plot
plt.show()
