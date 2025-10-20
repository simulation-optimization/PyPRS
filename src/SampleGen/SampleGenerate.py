# Copyright (c) 2025 Song Huang, Guangxin Jiang, Ying Zhong.
# Licensed under the MIT license.

import random
import time


class SampleGenerate:
    """
    A class representing a single alternative (or system) in a simulation study.

    This class holds all the necessary information for an alternative, such as
    its parameters, performance statistics (mean, variance), and manages its
    simulation runs.
    """

    def __init__(self, simulation_function):
        """
        Initializes the SampleGenerate object.

        Args:
            simulation_function (callable): The function that will be called to
                generate a single simulation sample for this alternative.
        """
        self.mean = 0               # Current sample mean.
        self.num = 0                # Number of simulation samples collected.
        self.s2 = 0                 # Sample variance.
        self.simulation_time = 0    # Total time spent in simulation runs.
        self.simulation_function = simulation_function # The external simulation function.
        self.args = []              # List of parameters for this alternative.
        # Default random replication_seed, can be overwritten by set_seed.
        self.seed = [random.randint(0, 2**50-1), random.randint(0, 2**47-1), random.randint(0, 2**47-1)]


    def get_args(self):
        """Returns the list of parameters for this alternative."""
        return self.args

    def set_args(self, args):
        """Sets the list of parameters for this alternative."""
        self.args = args

    def get_mean(self):
        """Returns the current sample mean of the alternative."""
        return self.mean

    def set_mean(self, mean):
        """Sets the sample mean of the alternative."""
        self.mean = mean

    def get_num(self):
        """Returns the number of observations (samples) collected."""
        return self.num

    def set_num(self, num):
        """Sets the number of observations (samples) collected."""
        self.num = num

    def get_s2(self):
        """Returns the sample variance of the alternative."""
        return self.s2

    def set_s2(self, s2):
        """Sets the sample variance of the alternative."""
        self.s2 = s2

    def set_simulation_time(self, simulation_time):
        """Sets the total simulation time."""
        self.simulation_time = simulation_time

    def get_simulation_time(self):
        """Returns the total simulation time."""
        return self.simulation_time

    def get_seed(self):
        """Returns the current random number replication_seed."""
        return self.seed

    def set_seed(self, seed):
        """Sets the random number replication_seed."""
        self.seed = seed

    def run_simulation(self):
        """
        Runs a single simulation replication for this alternative.

        This method calls the external simulation function, updates the sample
        mean and other statistics, and advances the random number replication_seed for the
        next call.

        Returns:
            float: The output value from the single simulation run.
        """
        start_time = time.time()
        # Call the provided simulation function with the alternative's arguments and replication_seed.
        simulation_output = self.simulation_function(self.args, self.seed)
        end_time = time.time()

        # Update the running mean.
        self.mean = (simulation_output + self.mean * self.num)/(self.num+1)
        self.num +=1
        self.simulation_time += (end_time - start_time)

        # Advance the third component of the replication_seed to ensure the next run is different.
        # This provides a simple subsubstream mechanism.
        self.seed = [self.seed[0], self.seed[1], self.seed[2]+1]

        return simulation_output
