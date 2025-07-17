import math
import time
import numpy as np
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


def aggregate_core_results(outer_list):
    """
    Aggregates experimental results from multiple parallel runs (cores).

    This function takes a list of result lists (one for each core) and
    combines the metrics for each round of the experiment into a single
    summary list.

    Args:
        outer_list (list): A list of lists, where each inner list contains
            round-by-round result dictionaries from a single core.

    Returns:
        list: A list of aggregated result dictionaries, one for each round.
    """
    if not outer_list:
        return []

    # Find the maximum number of rounds completed by any core.
    max_rounds = max(len(inner_list) for inner_list in outer_list)
    aggregated_results = []

    # Iterate through each round and aggregate data from all cores.
    for round_idx in range(max_rounds):
        current_round_results = []
        for core_results in outer_list:
            if round_idx < len(core_results):
                current_round_results.append(core_results[round_idx])

        if not current_round_results:
            continue

        # Sum or find the max of metrics across all cores for the current round.
        max_means = [result["Current Max Mean"] for result in current_round_results]
        used_budget = sum(result["Cumulative Sample Size"] for result in current_round_results)
        end_times = [result["Wall-Clock Time"] for result in current_round_results]
        simulation_times = sum(result["Cumulative Simulation Time"] for result in current_round_results)

        aggregated_results.append({
            "Current Max Mean": float(max(max_means)),
            "Cumulative Sample Size": used_budget,
            "Wall-Clock Time": max(end_times),
            "Cumulative Simulation Time": simulation_times
        })

    return aggregated_results


@ray.remote
def FBKT_core(alternatives_list_core, total_budget, n0, phi, use_crns, replication_seed, start_time):
    """
    Core processing logic for the FBKT algorithm on a subset of alternatives.

    This remote function performs the initial seeding and iterative elimination
    rounds of the FBKT procedure for a list of alternatives assigned to a single core.

    Args:
        alternatives_list_core (list): The list of alternative objects to evaluate.
        total_budget (float): The simulation budget allocated to this core (excluding initial seeding).
        n0 (int): The initial number of samples to take for each alternative.
        phi (float): A parameter controlling budget allocation across rounds.
        use_crns (bool): Flag indicating whether to use Common Random Numbers.
        replication_seed (list): The replication_seed for the random number generator.
        start_time (float): The global start time of the experiment, for wall-clock timing.

    Returns:
        list: A list containing:
            - The single best alternative from this core.
            - The calculated budget for the final evaluation round.
            - The total number of samples used in the elimination rounds.
            - The total simulation time consumed on this core.
            - The total budget consumed on this core.
            - A list of dictionaries with results from each round.
    """
    k = len(alternatives_list_core)
    round_results = []
    core_simulation_time = 0
    core_budget = 0

    # Set initial seeds for all alternatives on this core if using CRNs.
    if use_crns:
        for alt in alternatives_list_core:
            alt.set_seed(replication_seed)

    # Initial sampling phase (seeding)
    seed_start_time = time.time()
    means = []
    for alt in alternatives_list_core:
        for _ in range(n0):
            alt.run_simulation()
        means.append(alt.get_mean())

    core_simulation_time += time.time() - seed_start_time
    core_budget += n0 * k
    round_results.append({
        "Current Max Mean": float(max(means)),
        "Cumulative Simulation Time": core_simulation_time,
        "Wall-Clock Time": time.time() - start_time,
        "Cumulative Sample Size": core_budget
    })

    # Calculate the number of elimination rounds needed (log base 2 of k).
    R = math.ceil(math.log(k) / math.log(2))
    current_alternatives = alternatives_list_core
    total_samples_used = 0

    for round_idx in range(R):
        surviving_alternatives = []

        # Sort alternatives by their current estimated mean to pair the best with the worst.
        means = [alt.get_mean() for alt in current_alternatives]
        sorted_indices = np.argsort(means)

        # Calculate the simulation budget for the current round based on the phi parameter.
        r = round_idx + 1
        round_budget = int(
            np.floor(r / ((phi - 1) * phi) * ((phi - 1) / phi) ** r * total_budget
                     ))
        samples_per_alt = int(np.floor(round_budget / len(current_alternatives)))
        total_samples_used += samples_per_alt

        # Update the replication_seed for CRNs to ensure this round has a unique random number stream.
        crn_seed = [replication_seed[0], replication_seed[1], replication_seed[2] + total_samples_used]

        # Process alternatives in pairs.
        if len(current_alternatives) % 2 == 0:  # Even number of alternatives.
            for pair_idx in range(len(current_alternatives) // 2):
                # Pair the i-th worst with the i-th best.
                alt1 = current_alternatives[sorted_indices[pair_idx]]
                alt2 = current_alternatives[sorted_indices[-1 - pair_idx]]
                select_result = select_between_two(alt1, alt2, samples_per_alt)
                core_budget += samples_per_alt * 2
                surviving_alternatives.append(select_result[1])
                core_simulation_time += select_result[0]


        else:  # Odd number of alternatives.
            # The current best alternative automatically survives this round.
            best_alt = current_alternatives[sorted_indices[-1]]
            if use_crns:
                best_alt.set_seed(crn_seed)

            surviving_alternatives.append(best_alt)

            # Process the remaining alternatives in pairs.
            for pair_idx in range((len(current_alternatives) - 1) // 2):
                alt1 = current_alternatives[sorted_indices[pair_idx]]
                alt2 = current_alternatives[sorted_indices[-2 - pair_idx]]
                select_result = select_between_two(alt1, alt2, samples_per_alt)
                core_budget += samples_per_alt * 2
                surviving_alternatives.append(select_result[1])
                core_simulation_time += select_result[0]

        # Update the set of alternatives for the next round and record results.
        current_alternatives = surviving_alternatives
        means = [alt.get_mean() for alt in current_alternatives]

        round_results.append({
            "Current Max Mean": max(means),
            "Cumulative Simulation Time": core_simulation_time,
            "Wall-Clock Time": time.time() - start_time,
            "Cumulative Sample Size": core_budget
        })

        # Exit if only one alternative remains.
        if len(current_alternatives) == 1:
            break


    return [
        current_alternatives[0],
        total_samples_used,
        core_simulation_time,
        core_budget,
        round_results
    ]


def select_between_two(alt1, alt2, n_samples):
    """
    Compares two alternatives by running simulations and selects the one with the higher mean.

    Args:
        alt1: The first alternative object.
        alt2: The second alternative object.
        n_samples (int): The number of simulations to run for each alternative.

    Returns:
        list: A list containing:
            - ((float): The total simulation time for the comparison.
            - (object): The alternative object that performed better.
    """
    start_time = time.time()

    # Run simulations and accumulate the sum of outputs for each alternative.
    sum1 = sum(alt1.run_simulation() for _ in range(n_samples))
    sum2 = sum(alt2.run_simulation() for _ in range(n_samples))

    simulation_time = time.time() - start_time

    # Select the alternative with the higher sum of simulation outputs.
    return [simulation_time, alt1 if sum1 >= sum2 else alt2]


@ray.remote
def final_round(alternative, final_round_budget, use_crns, replication_seed, offset):
    """
    Performs the final evaluation round for a single surviving alternative.

    Args:
        alternative: The alternative object to evaluate.
        final_round_budget (int): The number of additional simulations to run.
        use_crns (bool): Flag indicating whether to use Common Random Numbers.
        replication_seed (list): The base replication_seed for the random number generator.
        offset (int): An offset to add to the replication_seed to ensure a unique random stream.

    Returns:
        list: A list containing:
            - The evaluated alternative object.
            - The estimated mean from this final round.
            - The budget used in this round.
            - The simulation time consumed.
    """
    # Set a unique replication_seed for this final round if using CRNs.
    if use_crns:
        alternative.set_seed([replication_seed[0], replication_seed[1], replication_seed[2] + offset])

    # Run the final batch of simulations.
    start_time_final = time.time()
    mean = 0
    for _ in range(final_round_budget):
        mean += alternative.run_simulation() / final_round_budget
    simulation_time = time.time() - start_time_final

    return [
        alternative,
        mean,
        final_round_budget,
        simulation_time
    ]


def FBKT(alternatives, configs, replication):
    """
    Executes the Fixed Budget Knapsack-Triangle (FBKT) algorithm in parallel.

    This is the main driver function that orchestrates the entire FBKT procedure,
    from initial seeding to parallel elimination and final evaluation.

    Args:
        alternatives (list): A list of all alternative objects to be evaluated.
        configs (dict): A dictionary of configuration parameters.
            Expected keys:
            - "N" (int): The total simulation budget for the entire experiment.
            - "n0" (int): The initial sample size for seeding.
            - "phi" (float): A parameter for budget allocation across rounds.
            - "Number of Processors" (int): The number of parallel cores to use.
            - "Reference Seed" (list): Base replication_seed for the random number generator.
            - "CRNs" (bool): True to use Common Random Numbers.
        replication (int): The current replication number of the experiment.

    Returns:
        list: A list of dictionaries containing results from each phase of the algorithm.
    """
    # Extract configuration parameters.
    N = configs["N"]
    n0 = configs["n0"]
    phi = configs["phi"]
    num_processors = configs["Number of Processors"]
    seed = configs["Reference Seed"]
    use_crns = configs["CRNs"]

    total_results = []
    start_time = time.time()
    k = len(alternatives)

    # Set seeds for this replication.
    if use_crns:
        replication_seed = [seed[0] + replication, seed[1], seed[2] + 1]
    else:
        for alt in alternatives:
            alt.set_seed([seed[0] + replication * k + alt.get_args()[0], seed[1], seed[2]  + 1])
        replication_seed = [seed[0] + replication, seed[1], seed[2] + 1]

    # Calculate the remaining budget per processor after the initial seeding phase.
    remaining_budget = (N - n0 * k) / num_processors
    R = math.ceil(math.log(2,k/num_processors))
    final_round_budget = int(
        np.floor((R + 1) / ((phi - 1) * phi) * ((phi - 1) / phi) ** (R + 1) * remaining_budget)
    )

    # Split alternatives across the available processors.
    alternatives_groups = split_by_modulo(alternatives, num_processors)

    # Execute the core FBKT elimination rounds in parallel.
    futures = [
        FBKT_core.remote(
            group,
            remaining_budget,
            n0,
            phi,
            use_crns,
            replication_seed,
            start_time
        )
        for group in alternatives_groups
    ]
    core_results = ray.get(futures)

    # Process the results from each parallel core.
    surviving_alternatives = []
    samples_used = []
    simulation_time = 0
    used_budget = 0
    round_results = []

    for result in core_results:
        surviving_alternatives.append(result[0])
        samples_used.append(result[1])
        simulation_time += result[2]
        used_budget += result[3]
        round_results.append(result[4])

    # Aggregate and record the round-by-round results.
    aggregated_rounds = aggregate_core_results(round_results)
    for round_result in aggregated_rounds:
        total_results.append({
            "Current Max Mean": round_result['Current Max Mean'],
            "Cumulative Simulation Time (s)": round_result['Cumulative Simulation Time'],
            "Cumulative Sample Size": round_result['Cumulative Sample Size'],
            "Wall-Clock Time (s)": round_result['Wall-Clock Time']
        })

    # Execute the final evaluation round in parallel on the survivors.
    max_offset = max(samples_used)
    final_futures = [
        final_round.remote(
            alt,
            final_round_budget,
            use_crns,
            replication_seed,
            max_offset
        )
        for alt in surviving_alternatives
    ]
    final_results = ray.get(final_futures)

    # Process the final results.
    final_alternatives = []
    final_means = []

    for result in final_results:
        final_alternatives.append(result[0])
        final_means.append(result[1])
        used_budget += result[2]
        simulation_time += result[3]

    # Select the best alternative based on the highest final mean.
    best_idx = np.argmax(final_means)
    best_alternative = final_alternatives[best_idx]
    total_results.append({
        "Current Max Mean": float(final_means[best_idx]),
        "Cumulative Simulation Time (s)": simulation_time,
        "Cumulative Sample Size": used_budget,
        "Wall-Clock Time (s)": time.time() - start_time
    })

    # Append a final summary object with all key metrics.
    total_results.append({
        "Best Alternative": best_alternative.get_args(),
        "Total Simulation Time (s)": simulation_time,
        "Total Sample Size": used_budget,
        "Wall-Clock Time (s)": time.time() - start_time,
        "Utilization": simulation_time / (num_processors * (time.time() - start_time))
    })

    return total_results