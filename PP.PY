import numpy as np
import matplotlib.pyplot as plt

# Define the x values
K = np.linspace(2, 40, 200) 

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


BOARD_SIZE = 200
TRIALS = 1000000

Number_of_Bins = np.uint32(np.round(BOARD_SIZE -1 / np.sqrt(np.log2( BOARD_SIZE -1 ))))
K = np.linspace(0, BOARD_SIZE, 200) 
Random_Number_Magnitude = np.uint64(np.round(Random_Number_Magnitude[BOARD_SIZE -1]))
# Initialize cycle array
cycle = np.zeros(BOARD_SIZE, dtype=np.uint64)

# Seed the random number generator (optional for reproducibility)
np.random.seed(10)  # Replace 10 with any desired seed value

for _ in range(TRIALS):  
  random_walk = np.uint64(0) 
  
  # Simulate random walk with NumPy random integers
  for _ in range(Number_of_Bins):
        random_walk += np.random.randint(0, 2**32, dtype=np.uint32)

  index = np.uint32(np.mod(random_walk * Random_Number_Magnitude >> np.uint32(32), BOARD_SIZE))

  # Update cycle array
  cycle[index] += 1

plt.figure(figsize=(10, 6))
plt.plot(K,  cycle, label='cycle')
plt.xlabel('K')
plt.legend()
plt.grid(True)
plt.show()


#Board_SIZE 34
#Board_size 9
#rn_range 15