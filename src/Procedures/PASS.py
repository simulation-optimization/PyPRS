# Copyright (c) 2025 Song Huang, Guangxin Jiang, Ying Zhong.
# Licensed under the MIT license.

import math
import time
from collections import deque
import ray


def split_by_modulo(elements, num_group):
    """
    Splits a list of elements into a specified number of groups using a modulo operator.

    This method provides a simple way to distribute work by assigning elements
    to groups in a round-robin fashion.

    Args:
        elements (list): The list of items to be distributed.
        num_group (int): The number of groups to create.

    Returns:
        list: A list of lists, where each inner list is a group containing
              elements distributed via `index % num_group`.
    """
    groups = [[] for _ in range(num_group)]
    for index, element in enumerate(elements):
        group_id = index % num_group
        groups[group_id].append(element)
    return groups


def flatten(lst):
    """
    Flattens a list of lists into a single, one-dimensional list.

    Args:
        lst (list): The list of lists to flatten (e.g., `[[1, 2], [3, 4]]`).

    Returns:
        list: A single flattened list containing all elements from the sublist
              (e.g., `[1, 2, 3, 4]`).
    """
    return [item for sublist in lst for item in sublist]


@ray.remote
def data_collection_phase(alternatives_list_core, n0):
    """
    Performs the initial data collection phase for a batch of alternatives.

    This remote function runs `n0` simulations for each alternative in its assigned
    list to compute initial estimates of their mean and variance.

    Args:
        alternatives_list_core (list): A list of alternative objects to process.
        n0 (int): The initial number of samples to collect for each alternative.

    Returns:
        tuple: A tuple containing:
            - (list): The updated list of alternatives with variance estimates.
            - (float): The sum of the sample means for the processed alternatives.
            - (float): The total simulation time consumed by this worker.
    """
    new_alternatives_list_core = []
    sum_mean = 0
    simulation_time = 0

    for alt in alternatives_list_core:
        mean = 0
        sum_sample2 = 0

        # Collect initial n0 samples to get a variance estimate.
        for _ in range(n0):
            start_time = time.time()
            sample = alt.run_simulation()
            simulation_time += (time.time() - start_time)
            mean += sample / n0
            sum_sample2 += sample * sample

        # Calculate the sample variance (s^2).
        s2 = (sum_sample2 - n0 * mean * mean) / (n0 - 1)
        alt.set_s2(s2)

        new_alternatives_list_core.append(alt)
        sum_mean += mean

    return new_alternatives_list_core, sum_mean, simulation_time


@ray.remote
def worker(alternative, Delta, elimination, mu_bar_star, c):
    """
    A remote worker process for simulating an alternative and checking elimination conditions.

    This function runs `Delta` additional simulations for a given alternative.
    If `elimination` is enabled, it checks after each simulation whether the
    alternative should be eliminated based on its performance relative to the
    reference mean `mu_bar_star`.

    Args:
        alternative: The alternative object to simulate.
        Delta (int): The number of additional simulations to perform in this batch.
        elimination (bool): If True, enables early elimination checks within the batch.
        mu_bar_star (float): The current reference mean value for comparison.
        c (float): A tuning parameter for the elimination condition calculation.

    Returns:
        dict: A dictionary of results containing:
              - "alternative": The updated alternative object.
              - "simulation time": Time spent on simulations in this worker.
              - "alternative elimination": A boolean indicating if the alternative was eliminated.
              - "old mean": The mean of the alternative before this batch of simulations.
              - "budget": The actual number of simulations performed before stopping.
    """
    old_mean = alternative.get_mean()
    alt_elimination = False
    simulation_time = 0
    budget_used = Delta  # Assume full budget is used unless early termination occurs.

    if elimination:
        # Run simulations one-by-one with an elimination check after each.
        for i in range(Delta):
            start_time = time.time()
            alternative.run_simulation()
            simulation_time += (time.time() - start_time)

            # Check the elimination condition.
            term = alternative.get_num() / alternative.get_s2()
            g_PASS = math.sqrt((c + math.log(term + 1)) * (term + 1))

            if (alternative.get_mean() - mu_bar_star) * term < -g_PASS:
                alt_elimination = True
                budget_used = i + 1  # Record the exact budget used.
                break  # Stop simulating for this alternative.
    else:
        # Run all Delta simulations without intermediate checks.
        for i in range(Delta):
            start_time = time.time()
            alternative.run_simulation()
            simulation_time += (time.time() - start_time)

    return {
        "alternative": alternative,
        "simulation time": simulation_time,
        "alternative elimination": alt_elimination,
        "old mean": old_mean,
        "cumulative sample size": budget_used
    }


def elimination_check(alternative, mu_bar_star, c):
    """
    Checks if a single alternative should be eliminated based on current statistics.

    This function is typically called before dispatching an alternative to a worker.

    Args:
        alternative: The alternative object to check.
        mu_bar_star (float): The current reference mean value.
        c (float): The algorithm's tuning parameter for the elimination calculation.

    Returns:
        bool: True if the alternative should be eliminated, False otherwise.
    """
    term = alternative.get_num() / alternative.get_s2()
    g_PASS = math.sqrt((c + math.log(term + 1)) * (term + 1))
    # The core elimination rule of the PASS algorithm.
    return (alternative.get_mean() - mu_bar_star) * term <= -g_PASS


def stopping_condition(survival_alternatives, max_number_sample=None):
    """
    Determines if the main algorithm loop should terminate.

    Args:
        survival_alternatives (list): The list of currently surviving alternatives.
        max_number_sample (int, optional): An optional maximum sample size per
            alternative. If provided, the algorithm stops when all survivors
            reach this size. Defaults to None.

    Returns:
        bool: True if the stopping conditions are met, False otherwise.
    """
    # Stop if only one alternative remains, as it is the best.
    if len(survival_alternatives) <= 1:
        return True

    # Stop if a maximum sample size is specified and all survivors have reached it.
    elif max_number_sample is not None:
        # Check if any alternative has taken fewer than the max samples.
        for item in survival_alternatives:
            if item.get_num() < max_number_sample:
                return False  # If any has not, continue running.
        return True  # All have reached the max sample size.

    # Otherwise, continue the procedure.
    else:
        return False


def PASS(alternatives, configs, replication):
    """
    Executes the Parallel Adaptive Subset Selection (PASS) algorithm.

    This is the main driver function that orchestrates the initial data collection
    and the subsequent asynchronous elimination phase.

    Args:
        alternatives (list): A list of all alternative objects to be evaluated.
        configs (dict): A dictionary of configuration parameters.
            Expected keys:
            - "n0" (int): Initial sample size for the data collection phase.
            - "Delta" (int): The batch size of simulations for each worker task.
            - "c" (float): A tuning parameter for the elimination rule.
            - "Worker Elimination" (bool): If True, workers can eliminate alternatives mid-batch.
            - "Termination Sample Size" (int, optional): Max samples before stopping.
            - "Number of Processors" (int): The number of parallel cores to use.
            - "Reference Seed" (list): Base replication_seed for the random number generator.
        replication (int): The current replication number of the experiment.

    Returns:
        list: A list of dictionaries containing results from each phase of the algorithm.
    """
    # ----- 1. Initialization and Configuration -----
    n0 = configs["n0"]
    Delta = configs["Delta"]
    c = configs["c"]
    worker_elimination = configs["Worker Elimination"]
    termination_sample_size = configs.get("Termination Sample Size")
    num_processors = configs["Number of Processors"]
    seed = configs["Reference Seed"]

    total_results = []
    k = len(alternatives)
    simulation_time = 0
    total_budget = 0

    # Set unique seeds for each alternative for this replication to ensure independence.
    for i in range(k):
        alternatives[i].set_seed([
            seed[0] + (replication - 1) * k + i,
            seed[1],
            seed[2]
        ])

    # ----- 2. Initial Data Collection Phase (Synchronous) -----
    start_time = time.time()

    # Distribute alternatives across processors for initial simulation.
    alternatives_list_core = split_by_modulo(alternatives, num_processors)
    data_collection_phase_core = [
        data_collection_phase.remote(group, n0)
        for group in alternatives_list_core
    ]
    data_collection_phase_list = ray.get(data_collection_phase_core)

    # Process results from the initial phase.
    data_collection_phase_alternatives = []
    mu_bar_star = 0  # Initialize the reference mean.
    data_collection_phase_simulation_time = 0

    for output in data_collection_phase_list:
        data_collection_phase_alternatives.append(output[0])
        mu_bar_star += output[1] / k
        data_collection_phase_simulation_time += output[2]

    total_budget += k * n0
    alternatives = flatten(data_collection_phase_alternatives)

    # Track statistics for reporting.
    means = {}
    nums = {}
    for alt in alternatives:
        alt_id = alt.get_args()[0]
        means[alt_id] = alt.get_mean()
        nums[alt_id] = alt.get_num()

    # Record initial phase results.
    data_collection_phase_result = {
        "Current Max Mean": max(means.values()),
        "Cumulative Sample Size": total_budget,
        "Wall-Clock Time (s)": time.time() - start_time,
        "Standard Mu": mu_bar_star,
        "Number of Surviving Alternatives": k,
        "Current Minimum Sample Size": min(nums.values())
    }
    total_results.append(data_collection_phase_result)

    # ----- 3. Main Elimination Phase (Asynchronous) -----
    survival_alternatives = alternatives
    alternatives_queue = deque(alternatives)  # Queue of alternatives waiting for simulation.
    work_alter = {}  # Dictionary of alternatives currently being processed by workers.
    active_tasks = []  # List of future objects for active Ray tasks.
    free_cores = num_processors
    cycle_count = 0

    while not stopping_condition(survival_alternatives, termination_sample_size):
        cycle_count += 1
        # Dispatch new tasks to available cores.
        while alternatives_queue and free_cores > 0:
            current_alt = alternatives_queue.popleft()
            alt_id = current_alt.get_args()[0]

            # Check for immediate elimination before dispatching.
            if elimination_check(current_alt, mu_bar_star, c):
                # Update reference mean upon elimination.
                mu_bar_star = (mu_bar_star * k - current_alt.get_mean()) / (k - 1)
                k -= 1
                del means[alt_id]
                del nums[alt_id]
            else:
                # Schedule a worker task for this alternative.
                work_alter[alt_id] = current_alt
                task = worker.remote(
                    current_alt,
                    Delta,
                    worker_elimination,
                    mu_bar_star,
                    c
                )
                active_tasks.append(task)
                free_cores -= 1

        # Process any completed tasks without blocking.
        if active_tasks:
            # Wait for at least one task to complete.
            done_ids, active_tasks = ray.wait(active_tasks, num_returns=1)
            num_done = len(done_ids)

            if num_done > 0:
                result = ray.get(done_ids)[0]
                # Unpack results from the completed worker task.
                updated_alt = result["alternative"]
                simulation_time += result["simulation time"]
                alt_elimination = result["alternative elimination"]
                old_mean = result["old mean"]
                used_budget = result["cumulative sample size"]
                updated_alt_id = updated_alt.get_args()[0]

                # Remove from the "in-progress" dictionary.
                if updated_alt_id in work_alter:
                    del work_alter[updated_alt_id]

                # Handle the result: either eliminate or update and re-queue.
                if alt_elimination:
                    mu_bar_star = (mu_bar_star * k - old_mean) / (k - 1)
                    k -= 1
                    total_budget += used_budget
                    del means[updated_alt_id]
                    del nums[updated_alt_id]
                else:
                    # Update the reference mean with the new information.
                    mu_bar_star = (mu_bar_star * k - old_mean + updated_alt.get_mean()) / k
                    nums[updated_alt_id] = updated_alt.get_num()
                    means[updated_alt_id] = updated_alt.get_mean()
                    # Add the updated alternative back to the queue for future simulation.
                    alternatives_queue.append(updated_alt)
                    total_budget += used_budget

                # Free up the core(s) that just finished.
                free_cores += num_done

        # Update the list of surviving alternatives for the stopping condition check.
        survival_alternatives = list(alternatives_queue) + list(work_alter.values())

        # Record results for this cycle.
        cycle_result = {
            "Current Max Mean": float(max(means.values())),
            "Cumulative Sample Size": total_budget,
            "Wall-Clock Time (s)": time.time() - start_time,
            "Standard Mu": mu_bar_star,
            "Number of Surviving Alternatives": k,
            "Current Minimum Sample Size": min(nums.values())
        }
        total_results.append(cycle_result)

    # ----- 4. Final Results -----
    # Handles cases with >1 survivor (due to max sample size) or 0 survivors.
    if len(survival_alternatives) == 1:
        best_alternative = survival_alternatives[0]
        final_result = {
            "Best Alternative": best_alternative.get_args(),
            "Total Simulation Time (s)": simulation_time,
            "Total Sample Size": total_budget,
            "Wall-Clock Time (s)": time.time() - start_time,
            "Utilization": simulation_time / ((time.time() - start_time) * num_processors),
        }
    else:
        final_result = {
            "Surviving Alternatives": [alt.get_args() for alt in survival_alternatives],
            "Total Simulation Time (s)": simulation_time,
            "Total Sample Size": total_budget,
            "Wall-Clock Time (s)": time.time() - start_time,
            "Utilization": simulation_time / ((time.time() - start_time) * num_processors),
        }

    total_results.append(final_result)
    return total_results