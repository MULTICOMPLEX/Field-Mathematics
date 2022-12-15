
# Python program to Plot Rose curve with even number of petals
import numpy as np
from matplotlib import pyplot as plt

cycles = 7

theta = np.linspace(-np.pi, np.pi, 1000)


r = 3* np.sin(cycles* theta)

r2 = np.roll(r, -int(1000/(2 * cycles)))

fig = plt.figure()
ax = fig.add_subplot(projection='polar')

ax.plot(theta, r, 'r')
ax.plot(theta, r2, 'b')

plt.show()