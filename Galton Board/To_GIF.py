
import imageio.v3 as iio
import os
import matplotlib.pyplot as plt
import numpy as np
from pygifsicle import optimize
import glob

gif_path = "test.gif"
n = 100
frames = np.stack([iio.imread(f"{x}.png") for x in range(n)], axis=0)
iio.imwrite(gif_path, frames, duration=60, loop=0)

optimize(gif_path)

# Remove files
removing_files = glob.glob('*.png')
for i in removing_files:
    os.remove(i)