from typing import Tuple, List
from math import log, sqrt, pi
from random import randint
from functools import reduce

def Probability_Wave(Board_SIZE: int, cycle: List[int], TRIALS: int) -> Tuple[float, int]:
    Board_size = round(log(Board_SIZE * 6) * sqrt(pi))
    rn_range = round(Board_SIZE / sqrt(log(Board_SIZE, 2)))

    random_walk = 0

    for i in range(TRIALS):
        for j in range(Board_size):
            random_walk += randint(0, 1)

        cycle[random_walk * rn_range >> 32 % Board_SIZE] += 1

    print("Board_size", Board_size)
    print("rn_range", rn_range)

    return rn_range, Board_size

# Example usage:

N_cycles = 10
N_Bins = 2048

if (N_Bins < 3 * N_cycles):  #minimum 3 x N_cycles
	N_Bins = 3 * N_cycles

Board_SIZE = round(N_Bins / N_cycles)

cycle = [0] * Board_SIZE
TRIALS = 10  # Replace with your desired value

result = Probability_Wave(Board_SIZE, cycle, TRIALS)
print(result)
