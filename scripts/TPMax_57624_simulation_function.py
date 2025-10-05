import numba
import numpy as np
from mrg32k3a_numba import MRG32k3a_numba


@numba.jit(cache=True)
def simulation_process(argsSim, rng1, rng2, rng3):
    # Unpack parameters
    s1, s2, s3 = argsSim[1], argsSim[2], argsSim[3]
    b2, b3 = argsSim[4], argsSim[5]

    # Initialize service time and exit time arrays
    ST = np.zeros((2050, 3))  # Service times
    ET = np.zeros((2050, 3))  # Exit times (completion times)

    # Generate service times using exponential distribution
    for i in range(2050):
        # Generate exponential random variates with mean 1/rate
        ST[i, 0] = rng1.expovariate(s1)  # Stage 1 service time
        ST[i, 1] = rng2.expovariate(s2)  # Stage 2 service time
        ST[i, 2] = rng3.expovariate(s3)  # Stage 3 service time

    # Initialize first job's exit times
    ET[0, 0] = ST[0, 0]  # Completion time at stage 1
    ET[0, 1] = ET[0, 0] + ST[0, 1]  # Completion time at stage 2
    ET[0, 2] = ET[0, 1] + ST[0, 2]  # Completion time at stage 3

    # Process subsequent jobs
    for i in range(1, 2050):
        # Stage 1: Starts after previous job finishes stage 1
        ET[i, 0] = ET[i - 1, 0] + ST[i, 0]

        # Stage 2: Starts when both stage 1 completes and previous job finishes stage 2
        ET[i, 1] = max(ET[i - 1, 1], ET[i, 0]) + ST[i, 1]

        # Stage 3: Starts when both stage 2 completes and previous job finishes stage 3
        ET[i, 2] = max(ET[i - 1, 2], ET[i, 1]) + ST[i, 2]

        # Apply blocking constraints
        if i >= b2:
            # Blocking from stage 2: Stage 1 can't start until job i-b2 finishes stage 2
            ET[i, 0] = max(ET[i, 0], ET[int(i - b2), 1])

        if i >= b3:
            # Blocking from stage 3: Stage 2 can't start until job i-b3 finishes stage 3
            ET[i, 1] = max(ET[i, 1], ET[int(i - b3), 2])

    # Calculate throughput: number of jobs processed between job 2000 and 2049
    # divided by the time difference between their completion at stage 3
    return (2050 - 2000) / (ET[-1, 2] - ET[2000 - 1, 2])


def simulation_function(argsSim, seedSim):
    rng1 = MRG32k3a_numba(np.array([seedSim[0], seedSim[1] + 1, seedSim[2]]))
    rng2 = MRG32k3a_numba(np.array([seedSim[0], seedSim[1] + 2, seedSim[2]]))
    rng3 = MRG32k3a_numba(np.array([seedSim[0], seedSim[1] + 3, seedSim[2]]))

    result = simulation_process(argsSim, rng1, rng2, rng3)
    return result
