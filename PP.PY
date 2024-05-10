import numpy as np
import matplotlib.pyplot as plt

# Define the x values
K = np.linspace(2, 6000, 10000) 

# Calculate the y values
Number_of_Bins  = np.log(K * 6) * np.sqrt(np.pi)

Random_Number_Magnitude = K / np.sqrt(np.log2( K ))

# Plot the formula
plt.figure(figsize=(10, 6))
plt.plot(K,  Number_of_Bins, label='Number of Bins')
plt.plot(K, Random_Number_Magnitude, label='Random Number Magnitude')
plt.xlabel('K')
plt.ylabel('Random Number Magnitude, Number of Bins')
plt.title('Number of  Bins  =  log( 6K ) sqrt( π )\n Random Number Magnitude = K / sqrt( log2( K ))')
plt.legend()
plt.grid(True)
plt.show()


#Board_SIZE 34
#Board_size 9
#rn_range 15