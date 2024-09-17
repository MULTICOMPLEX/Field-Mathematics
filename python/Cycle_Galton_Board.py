import numpy as np
import matplotlib.pyplot as plt
#https://github.com/lemire/fastrand
import fastrand   # Assuming you have this library for faster random generation

initial_n_bins = np.linspace(2, 300, 2000) 

n_bins  = np.log(initial_n_bins * 6) * np.sqrt(np.pi)

rn_mag = initial_n_bins / np.sqrt(np.log2( initial_n_bins))

plt.figure(figsize=(10, 6))
plt.plot(initial_n_bins,  n_bins, label='NBins')
plt.plot(initial_n_bins, rn_mag, label='RNMag')
plt.xlabel('Initial NBins')
plt.ylabel('RNMag, NBins')
plt.title('NBins  =  log( 6 x Initial NBins ) sqrt( π )\n RNMag = Initial NBins / sqrt( log2( Initial NBins ))')
plt.legend()
plt.grid(True)
plt.show()

############################

INITIAL_NBINS = 200
TRIALS = 5000000

def simulate_wave_galton_board(trials, initial_n_bins):
    """Simulates a Cycle Galton board with efficient random number generation and bin updating.

    Args:
        trials: Number of times to drop a ball.
        initial_n_bins: Number of bins used for the initial random walk simulation.

    Returns:
        cycle: An array representing the number of balls in each bin.
    """

    n_bins  = np.uint64(np.round(np.log(initial_n_bins * 6) * np.sqrt(np.pi))) # Scaled Number of bins
    rn_mag = np.uint64(np.round(initial_n_bins  / np.sqrt( np.log2(initial_n_bins)))) # Scaling factor

    # Initialize cycle array
    cycle = np.zeros(initial_n_bins, dtype=np.uint64)  # Use uint64 to prevent overflow

    # Seed the random number generator (optional for reproducibility)
    fastrand.pcg32_seed(10)#np.random.seed( 10 )  # Replace 10 with any desired seed value

    for _ in range(trials):
        random_walk = np.uint64(0)

        # Simulate random walk with fastrand
        for _ in range(n_bins):
            random_walk +=  np.uint32(fastrand.pcg32())#np.random.randint(0, 2**32, dtype=np.uint32)

        # Calculate bin index with improved modulo operation
        index = np.uint32(np.mod(random_walk * rn_mag >> np.uint32(32), initial_n_bins))

        # Update cycle array
        cycle[index] += 1

    return cycle

cycle = simulate_wave_galton_board(TRIALS, INITIAL_NBINS)

initial_n_bins = np.linspace(0, INITIAL_NBINS, INITIAL_NBINS) 

plt.figure(figsize=(10, 6))
plt.step(initial_n_bins,  cycle, label='Cycle')
plt.xlabel('Bin')
plt.ylabel('Frequency')
plt.title('Simulate a Cycle Galton Board')
plt.legend()
plt.grid(True)
plt.show()
