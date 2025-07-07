import sys
import os
from matplotlib import pylab as plt
from matplotlib import mlab
import numpy as np
import time
import phimagic_prng32
import phimagic_prng64
import json
from visuals import *


class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

# Create an instance of the custom PRNG
prng = phimagic_prng32.mxws()

Nbins = 3100
N_Integrations = 2


write_to_file = True

#Time seed 
current_time_seconds = int(time.time())    

# File to store results
output_file = 'phase_amplitude_values.json'

# List to accumulate results
results = []

fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)
        
for cycle in range(Ncycles_begin, Ncycles_end + 1):
        # Generate signal
        with Timer() as t:
            s1, p1 = prng.sine(enable_seed = 1,  Seed = current_time_seconds + cycle, Ntrials = Ntrials, Ncycles = cycle,  N_Integrations = N_Integrations,  Nbins = Nbins, Icycles = 1)
        
        print(f"Elapsed time: {t.elapsed_time:.4g}")
        print("cycle", cycle)
        
        # Step 1: Remove the mean (center the signal)
        DC = np.mean(s1)
        
        plt.plot(s1, label='DC')
        
        print('DC = ', DC)
        print('DC Calculated = ',((Ntrials / Nbins) * cycle) )
        print('Cycle from DC = ',DC /(Ntrials / Nbins))
        centered = s1 - DC

        # Step 2: Normalize to [-1, +1] range
        s1 = centered / np.max(np.abs(centered))  
      
        
        # Perform FFT
        N = len(s1)
        yf = np.fft.fft(s1)

        # print cycle length
        print("array length", N)

        # Extract phase and amplitude at dominant frequency
        phase = np.angle(yf[cycle])
        amplitude = np.abs(yf[cycle]) / N
        
        # Append the result as a dictionary to the list
        results.append({"Trial": cycle, "Phase": phase, "Amplitude": amplitude})

        # print progress
        print(f'cycle {cycle}: Phase = {phase:.10f}, Amplitude = {amplitude:.10f}')
        print('\n')

if write_to_file:
    # Write the accumulated results to the JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

print('Completed processing.')

DC = (Ntrials / Nbins) * Ncycles_end / N_Integrations

fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)
x = np.linspace(0,1,  len(s1)) 
plt.step(x, s1, label = str(int((Ntrials * N_Integrations * p1[1] * Ncycles_end)))+ " Trials") 
plt.title("Sine Distribution " + str(Ncycles_end), color = 'white')
plt.xlabel("Time", color = 'white')
plt.ylabel("Y", color = 'white')
plt.grid(alpha=0.4)
plt.legend()


fig = plt.figure(facecolor='#002b36', figsize=(10, 6))
ax = fig.gca()
set_axis_color(ax)

Fs = len(s1) # Example value, set this to the actual sampling rate of your signal
s, f = mlab.psd(s1, NFFT= Fs, Fs = Fs)
plt.loglog(f, s)
plt.grid(True, which='both', alpha = 0.4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (Unit**2/Hz)')


plt.title("FFT Distribution", color='white')

    
plt.grid(True)

plt.show()


