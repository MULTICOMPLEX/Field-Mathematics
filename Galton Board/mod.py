
# Python program to Plot Rose curve with even number of petals
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 1000, 1000)

y = (((1/24 * (pow(-x,4) +20 * pow(x,3) -30 * pow(x,2) +20 * x - 5)) * 316 )% 15)

#r = np.roll(r, -250)

plt.plot(x, y, 'red')

plt.show()