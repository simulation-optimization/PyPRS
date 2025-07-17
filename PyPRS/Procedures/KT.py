import math
import time
import ray
import numpy as np
from ..Utilities.rinott import rinott


def split_by_modulo(elements, num_groups):
    """
    Splits a list of elements into a specified number of groups using a modulo operator.

    This method provides a simple way to distribute work by assigning elements
    to groups in a round-robin fashion.

    Args:
        elements (list): The list of items to split.
        num_groups (int): The number of groups to create.

    Returns:
        list: A list of lists, where each inner list is a group.
    """
    groups = [[] for _ in range(num_groups)]
    for index, element in enumerate(elements):
        group_id = index % num_groups
        groups[group_id].append(element)
    return groups


def group_list(lst, group_size):
    """
    Splits a list into smaller chunks of a specified maximum size.

    Args:
        lst (list): The list to be split.
        group_size (int): The maximum size of each chunk.

    Returns:
        list: A list of lists, where each inner list is a chunk of the original list.
    """

    return [lst[i:i + group_size] for i in range(0, len(lst), group_size)]


def aggregate_round_results(outer_list):
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
        total_budget = sum(result["Cumulative Sample Size"] for result in current_round_results)
        end_times = [result["Wall-Clock Time"] for result in current_round_results]
        simulation_times = sum(result["Cumulative Simulation Time"] for result in current_round_results)

        aggregated_results.append({
            "Current Max Mean": float(max(max_means)),
            "Cumulative Sample Size": total_budget,
            "Wall-Clock Time": max(end_times),
            "Cumulative Simulation Time": simulation_times
        })

    return aggregated_results


def KN(alternatives_group, alpha_r, delta, n0, use_crns, kn_seed):
    """
    Implements the Kim-Nelson (KN) procedure to select the best alternative from a group.

    The KN procedure is a fully sequential elimination algorithm. It starts with
    `n0` samples and then takes one additional sample at a time for all
    surviving alternatives, eliminating any that are statistically inferior.

    Args:
        alternatives_group (list): The list of alternatives in the current group.
        alpha_r (float): The confidence level for the current round.
        delta (float): The indifference-zone parameter.
        n0 (int): The initial number of samples to take for each alternative.
        use_crns (bool): Flag indicating whether to use Common Random Numbers.
        kn_seed (list): The random replication_seed configuration.

    Returns:
        list: A list containing:
              - The best alternative identified in the group.
              - The total simulation time consumed.
              - The total simulation budget (samples) used.
              - The final mean of the best alternative.
              - The total number of samples taken per survivor.
    """
    simulation_time = 0.0
    group_budget = 0
    k = len(alternatives_group)

    # Set random seeds for reproducibility.
    if use_crns:
        for alt in alternatives_group:
            alt.set_seed(kn_seed)

    # If only one alternative is in the group, it's the winner by default.
    if k == 1:
        return [
            alternatives_group[0],
            simulation_time,
            group_budget,
            0,
            0
        ]

    # Calculate the KN procedure constant h^2.
    h2 = (n0 - 1) * ((2 * alpha_r / (k - 1)) ** (-2 / (n0 - 1)) - 1)
    alt_indices = list(range(k))

    # Run initial n0 simulations for each alternative.
    start_time = time.time()
    samples = [[alt.run_simulation() for _ in range(n0)] for alt in alternatives_group]
    simulation_time += time.time() - start_time

    sum_samples = [sum(alt_samples) for alt_samples in samples]
    variance_matrix = np.zeros((k, k))

    # Compute pairwise sample variances of the differences.
    for i in range(1, k):
        for j in range(i):
            diffs = [samples[i][l] - samples[j][l] for l in range(n0)]
            mean_diff = sum(diffs) / n0
            var = sum((d - mean_diff) ** 2 for d in diffs) / (n0 - 1)
            variance_matrix[i, j] = var
            variance_matrix[j, i] = var

    t = n0  # Initialize sample count.
    group_budget += k * t

    # Main elimination loop.
    while len(alt_indices) > 1:
        elimination_counts = [0] * len(alt_indices)

        # Compare all surviving pairs of alternatives.
        for i_idx in range(1, len(alt_indices)):
            for j_idx in range(i_idx):
                i = alt_indices[i_idx]
                j = alt_indices[j_idx]
                var_ij = variance_matrix[i, j]

                # Calculate the required sample size for a decision.
                N = math.floor(var_ij * h2 / (delta ** 2))

                # Elimination logic based on the number of samples taken.
                if t > N:
                    # If enough samples are taken, eliminate the one with the lower sum.
                    if sum_samples[i] > sum_samples[j]:
                        elimination_counts[j_idx] += 1
                    else:
                        elimination_counts[i_idx] += 1
                else:
                    # Otherwise, use the continuation region formula.
                    diff_ij = sum_samples[i] - sum_samples[j]
                    threshold = max(h2 * var_ij / (2 * delta) - delta * t / 2, 0)
                    if diff_ij < -threshold:  # "i" is significantly worse than j
                        elimination_counts[i_idx] += 1
                    elif -diff_ij < -threshold:  # j is significantly worse than i
                        elimination_counts[j_idx] += 1

        # Remove the eliminated alternatives from the active set.
        # Iterate backwards to avoid index shifting issues.
        for idx in reversed(range(len(elimination_counts))):
            if elimination_counts[idx] > 0:
                del alt_indices[idx]

        # If only one survivor remains, exit the loop.
        if len(alt_indices) == 1:
            break

        # If more than one survivor, take one additional sample for each.
        start_time = time.time()
        for idx in alt_indices:
            sample = alternatives_group[idx].run_simulation()
            sum_samples[idx] += sample
            group_budget += 1
        simulation_time += time.time() - start_time
        t += 1

    # Prepare and return results for the winning alternative.
    best_alt = alternatives_group[alt_indices[0]]
    final_mean = sum_samples[alt_indices[0]] / t

    return [
        best_alt,
        simulation_time,
        group_budget,
        final_mean,
        t
    ]


@ray.remote
def KT_core(alternatives_core, alpha, delta, n0, group_size, use_crns, replication_seed, start_time):
    """
    Executes the core KT procedure on a single processor (a remote Ray actor).

    This function manages the rounds of elimination. In each round, it splits
    its assigned alternatives into groups, runs the KN procedure on each group,
    and collects the survivors for the next round.

    Args:
        alternatives_core (list): The subset of alternatives assigned to this core.
        alpha (float): The overall confidence level (e.g., 0.05).
        delta (float): The indifference-zone parameter.
        n0 (int): The initial sample size for the KN procedure.
        group_size (int): The maximum size of groups for the KN procedure.
        use_crns (bool): Flag for using Common Random Numbers.
        replication_seed (list): The base random replication_seed for this replication.
        start_time (float): The global start time of the experiment, for wall-clock timing.

    Returns:
        list: A list containing:
              - The single best alternative from this core.
              - The alpha value to be used in the final Rinott selection stage.
              - The total number of samples taken (for replication_seed management).
              - The total simulation time on this core.
              - The total budget consumed on this core.
              - A list of dictionaries with results from each round.
    """
    k = len(alternatives_core)
    # Calculate the maximum number of rounds required.
    max_rounds = math.ceil(math.log(k) / math.log(group_size))
    current_alternatives = alternatives_core
    core_sim_time = 0
    core_budget = 0
    round_results = []
    seed_offset = 0  # Offset to ensure unique random streams in each round.

    # Loop through the elimination rounds.
    for round_num in range(1, max_rounds + 1):
        survivors = []
        means = []
        sample_counts = []

        # Adjust alpha for the current round to distribute the error probability.
        alpha_r = alpha / (2 ** round_num)
        round_seed = [replication_seed[0], replication_seed[1], replication_seed[2] + seed_offset]

        # Split the current set of alternatives into smaller groups.
        num_groups = math.ceil(len(current_alternatives) / group_size)
        groups = split_by_modulo(current_alternatives, num_groups)

        # Run the KN procedure on each group.
        for group in groups:
            kn_result = KN(group, alpha_r, delta, n0, use_crns, round_seed)
            survivors.append(kn_result[0])
            core_sim_time += kn_result[1]
            core_budget += kn_result[2]
            means.append(kn_result[3])
            sample_counts.append(kn_result[4])

        # Update the replication_seed offset based on the max samples used in this round.
        seed_offset += max(sample_counts)

        # Record performance metrics for the completed round.
        round_results.append({
            "Current Max Mean": max(means),
            "Cumulative Sample Size": core_budget,
            "Wall-Clock Time": time.time() - start_time,
            "Cumulative Simulation Time": core_sim_time
        })

        # The survivors of this round become the candidates for the next.
        current_alternatives = survivors
        if len(current_alternatives) == 1:
            break

    # The final survivor from this core.
    return [
        current_alternatives[0],
        seed_offset,
        core_sim_time,
        core_budget,
        round_results
    ]


@ray.remote
def Rinott_final(alternative, alpha, delta, n0, num_processors, use_crns, seed, max_samples):
    """
    Performs the final selection stage using Rinott's procedure.

    This function takes one of the winning alternatives from a core and runs
    additional simulations to meet the (`alpha`, `delta`) indifference-zone guarantee.

    Args:
        alternative: The alternative to evaluate.
        alpha (float): The confidence level for this final stage.
        delta (float): The indifference-zone parameter.
        n0 (int): The initial sample size.
        num_processors (int): The total number of processors (contenders).
        use_crns (bool): Flag for using Common Random Numbers.
        seed (list): The base random replication_seed configuration.
        max_samples (int): The maximum number of samples used in prior stages,
                           used as an offset for the random number stream.

    Returns:
        dict: A dictionary containing the final evaluation results.
    """
    # Set a unique random replication_seed stream to avoid overlap with previous stages.
    if use_crns:
        alternative.set_seed([seed[0], seed[1], seed[2] + max_samples])

    # Calculate Rinott's h, a constant needed for the selection guarantee.
    h = rinott(num_processors, 1 - alpha, n0 - 1)
    mean = 0
    sum_sq = 0
    sim_time = 0
    budget = 0

    # Run initial n0 samples to estimate variance.
    for _ in range(n0):
        start_time = time.time()
        sample = alternative.run_simulation()
        sim_time += time.time() - start_time
        budget += 1
        sum_sq += sample * sample
        mean += sample

    mean_initial = mean / n0
    # Calculate sample variance.
    variance = (sum_sq - n0 * mean_initial ** 2) / (n0 - 1)

    # Determine the number of additional samples required.
    required_samples = max(math.ceil((h / delta) ** 2 * variance) - n0, 0)

    # Run the additional samples.
    for _ in range(required_samples):
        start_time = time.time()
        sample = alternative.run_simulation()
        sim_time += time.time() - start_time
        mean += sample
        budget += 1

    # Calculate the final mean over all samples taken.
    total_samples = n0 + required_samples
    final_mean = mean / total_samples if total_samples > 0 else 0

    return {
        "alternative": alternative,
        "mean": final_mean,
        "simulation_time": sim_time,
        "cumulative sample size": budget
    }


def KT(alternatives, configs, replication):
    """
    Executes the full KT Selection Procedure.

    This is the main driver function that orchestrates the entire multi-stage,
    parallel selection process.

    Args:
        alternatives (list): A list of all alternative objects to be evaluated.
        configs (dict): A dictionary of configuration parameters.
            Expected keys:
            - "alpha" (float): Overall confidence level (e.g., 0.05).
            - "delta" (float): Indifference-zone parameter.
            - "n0" (int): Initial sample size for KN and Rinott procedures.
            - "g" (int): Group size for the KN procedure.
            - "Number of Processors" (int): Number of parallel cores to use.
            - "Reference Seed" (list): Base replication_seed for the random number generator.
            - "CRNs" (bool): True to use Common Random Numbers.
        replication (int): The current replication number of the experiment.

    Returns:
        list: A list of dictionaries containing results from each stage and a
              final summary.
    """
    # Extract configuration parameters.
    alpha = configs["alpha"]
    delta = configs["delta"]
    n0 = configs["n0"]
    group_size = configs["g"]
    num_processors = configs["Number of Processors"]
    seed = configs["Reference Seed"]
    use_crns = configs["CRNs"]

    total_results = []
    simulation_time = 0
    total_budget = 0
    global_start = time.time()
    k = len(alternatives)

    # Configure the replication_seed for this specific replication.
    if use_crns:
        # For CRNs, each replication gets a unique primary replication_seed.
        replication_seed = [seed[0] + replication, seed[1], seed[2] + 1]
    else:
        # For independent streams, each alternative gets a unique replication_seed.
        for alt in alternatives:
            # A complex replication_seed ensures streams don't overlap across replications.
            alt.set_seed([seed[0] + (replication - 1) * k + alt.get_args()[0], seed[1], seed[2] + 1])
        replication_seed = [seed[0] + replication, seed[1], seed[2] + 1]

    # 1. Distribute alternatives across processors.
    alt_groups = split_by_modulo(alternatives, num_processors)

    # 2. Execute the parallel elimination stage (KT_core).
    futures = [
        KT_core.remote(
            group, alpha, delta, n0, group_size,
            use_crns, replication_seed, global_start
        )
        for group in alt_groups
    ]
    core_results = ray.get(futures)

    # Process and consolidate results from the parallel elimination stage.
    final_alts = []
    total_samples = []
    round_data = []

    for result in core_results:
        final_alts.append(result[0])
        total_samples.append(result[1])
        simulation_time += result[2]
        total_budget += result[3]
        round_data.append(result[4])

    # Aggregate and store round-by-round results.
    aggregated_rounds = aggregate_round_results(round_data)
    for round_result in aggregated_rounds:
        total_results.append({
            "Current Max Mean": round_result["Current Max Mean"],
            "Cumulative Simulation Time (s)": round_result["Cumulative Simulation Time"],
            "Cumulative Sample Size": round_result["Cumulative Sample Size"],
            "Wall-Clock Time (s)": round_result["Wall-Clock Time"]
        })

    # 3. Final selection using Rinott's procedure on the winners from each core.
    max_sample_offset = max(total_samples)
    final_r = math.ceil(math.log(k/num_processors) / math.log(group_size))
    final_alpha = alpha / (2 ** final_r)
    rinott_futures = [
        Rinott_final.remote(
            alt, final_alpha, delta, n0,
            num_processors, use_crns, replication_seed, max_sample_offset
        )
        for alt in final_alts
    ]
    rinott_results = ray.get(rinott_futures)

    # Process and consolidate the final selection results.
    final_means = []
    final_alts_list = []

    for result in rinott_results:
        final_means.append(result["mean"])
        final_alts_list.append(result["alternative"])
        simulation_time += result["simulation_time"]
        total_budget += result["cumulative sample size"]

    best_idx = np.argmax(final_means)
    best_alt = final_alts_list[best_idx]

    # Record the result of the final selection stage.
    total_results.append({
        "Current Max Mean": float(final_means[best_idx]),
        "Cumulative Simulation Time (s)": simulation_time,
        "Cumulative Sample Size": total_budget,
        "Wall-Clock Time (s)": time.time() - global_start
    })

    # Append a final summary object with all key metrics.
    total_results.append({
        "Best Alternative": best_alt.get_args(),
        "Total Simulation Time (s)": simulation_time,
        "Total Sample Size": total_budget,
        "Wall-Clock Time (s)": time.time() - global_start,
        "Utilization": simulation_time / (num_processors * (time.time() - global_start))
    })

    return total_results