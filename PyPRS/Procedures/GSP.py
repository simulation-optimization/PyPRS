import math
import time
import numpy as np
import ray
from ..Utilities.etafunc import EtaFunc
from ..Utilities.rinott import rinott


@ray.remote
def stage0_simulation(alternatives_list_core, n0):
    """
    Performs the initial simulation stage (Stage 0) to estimate computation time.

    This stage runs a small number of simulations (`n0`) for each alternative
    to get a rough estimate of how long a single simulation takes. This
    information is used later for load balancing.

    Args:
        alternatives_list_core (list): A list of alternative objects assigned to
            this processing core.
        n0 (int): The number of initial simulation replications to run for each
            alternative.

    Returns:
        list: A list where each element is a sublist containing the alternative
              object and its estimated average simulation time.
              Example: `[[alternative1, time1], [alternative2, time2], ...]`.
    """
    new_alternatives_list_core = []
    for alt in alternatives_list_core:
        alternative_information = []
        # Run n0 simulations for this alternative to warm up and get a time estimate.
        for _ in range(n0):
            alt.run_simulation()

        # Calculate the average simulation time and reset statistics.
        # This prevents the initial n0 runs from biasing future results.
        estimate_time = alt.get_simulation_time() / n0
        alt.set_mean(0)
        alt.set_num(0)
        alt.set_simulation_time(0)

        # Store the alternative along with its estimated time.
        alternative_information.append(alt)
        alternative_information.append(estimate_time)
        new_alternatives_list_core.append(alternative_information)

    return new_alternatives_list_core


@ray.remote
def stage1_simulation(alternatives_list_core, n1):
    """
    Performs the first main simulation stage (Stage 1) to collect statistics.

    This stage runs `n1` simulations for each alternative to gather initial
    estimates of their performance mean and variance.

    Args:
        alternatives_list_core (list): A list of alternatives, each paired with
            its estimated simulation time from Stage 0.
        n1 (int): The number of simulation replications to run in this stage.

    Returns:
        list: A list containing:
              - A list of updated alternatives with their Stage 1 statistics.
              - The total simulation time consumed in this stage.
              - The sum of `sqrt(s^2 / estimated_time)` for load balancing.
              - A list of the estimated simulation times for each alternative.
    """
    result_core = []
    new_alternatives_list_core = []
    estimated_times = []
    sum_average_s = 0
    stage1_simulation_time = 0

    for alt_info in alternatives_list_core:
        alternative = alt_info[0]
        est_time = alt_info[1]
        mean = 0
        sum_sample2 = 0

        # Run n1 simulations and collect performance data.
        for _ in range(n1):
            start_time = time.time()
            sample = alternative.run_simulation()
            stage1_simulation_time += (time.time() - start_time)
            mean += sample / n1
            sum_sample2 += sample * sample

        # Calculate the sample variance (s^2).
        s2 = (sum_sample2 - n1 * mean * mean) / (n1 - 1)
        # This term is used for allocating additional simulation runs in Stage 2.
        sum_average_s += math.sqrt(s2 / est_time)

        # Update the alternative's statistics.
        alternative.set_s2(s2)
        new_entry = [alternative, est_time]
        new_alternatives_list_core.append(new_entry)
        estimated_times.append(est_time)

    # Package results for this core.
    result_core.append(new_alternatives_list_core)
    result_core.append(stage1_simulation_time)
    result_core.append(sum_average_s)
    result_core.append(estimated_times)

    return result_core


@ray.remote
def stage1_screen(alternatives_list_core, n1, beta, rMax, eta, average_s):
    """
    Screens alternatives after Stage 1 using pairwise comparisons.

    This function first calculates the number of additional simulations (batch size)
    for Stage 2, and then applies a screening procedure to eliminate non-competitive
    alternatives.

    Args:
        alternatives_list_core (list): List of alternatives with their statistics.
        n1 (int): The number of simulations performed in Stage 1.
        beta (float): An allocation parameter for distributing simulation budget.
        rMax (int): The maximum number of additional replications allowed in Stage 2.
        eta (float): The screening parameter derived from confidence levels.
        average_s (float): The average of `sqrt(s^2 / estimated_time)` across all alternatives.

    Returns:
        dict: A dictionary containing the list of surviving alternatives and the
              maximum mean observed among them.
    """
    alternatives_information = []
    for alt_info in alternatives_list_core:
        alternative = alt_info[0]
        est_time = alt_info[1]

        # Calculate batch size for additional simulations in Stage 2.
        # This proportionally allocates more runs to alternatives with higher variance.
        batch = math.ceil(beta * math.sqrt(alternative.get_s2() / est_time) / average_s)
        new_entry = [
            alternative,
            est_time,
            batch,
            n1 + rMax * batch  # Total planned simulations for this alternative.
        ]
        alternatives_information.append(new_entry)

    # Perform the screening procedure.
    stage1_screen_result = screen(alternatives_information, n1, eta)
    return stage1_screen_result


def screen(alternatives_information, n1, eta):
    """
    Performs pairwise screening to eliminate non-competitive alternatives.

    This procedure compares every pair of alternatives (i, j) and eliminates an
    alternative if it is statistically worse than another, based on the `eta`
    parameter.

    Args:
        alternatives_information (list): A list containing alternatives and their
            simulation metadata.
        n1 (int): The number of initial simulations.
        eta (float): The screening parameter.

    Returns:
        dict: A dictionary containing:
              - "surviving alternatives": A list of alternatives that were not eliminated.
              - "max mean": The maximum mean observed across all alternatives before screening.
              - "the estimated times of survival alternatives": A list of estimated times for survivors.
    """
    survival_alternatives_information = []
    estimated_times = []
    k = len(alternatives_information)
    if k == 1:
        survival_alternatives_information.append(alternatives_information[0])
        max_mean = alternatives_information[0][0].get_mean()
        estimated_times.append(alternatives_information[0][1])
        screen_result = {"Surviving Alternatives": survival_alternatives_information, "Current Max Mean": max_mean,
                             "The Estimated Times of Surviving Alternatives": estimated_times}
        return screen_result

    elimination = np.zeros(k, dtype=int)
    new_mean = []
    mean = []
    c1 = []
    c2 = []
    # Perform pairwise comparisons to mark alternatives for elimination.
    for i in range(1, k):
        mean.append(alternatives_information[i][0].get_mean())
        # Variance of the sample mean after n1 runs.
        c1.append(alternatives_information[i][0].get_s2() / alternatives_information[i][0].get_num())
        # Projected variance of the sample mean after all planned runs.
        c2.append(alternatives_information[i][0].get_s2() / alternatives_information[i][3])
        for j in range(i):
            # Standardized difference of means.
            y = (mean[i] - mean[j]) / (c1[i] + c1[j])
            # Critical value for comparison.
            a = eta * np.sqrt((n1 - 1) / (c2[i] + c2[j]))
            # If alternative i is significantly worse than j, mark i for elimination.
            if y < -a:
                elimination[i] += 1
            # If alternative j is significantly worse than i, mark j for elimination.
            if -y < -a:
                elimination[j] += 1
    # Collect all alternatives that were not marked for elimination.
    for l in range(k - 1, -1, -1):
        if elimination[l] < 1:
            survival_alternatives_information.append(alternatives_information[l])
            new_mean.append(mean[l])
            estimated_times.append(alternatives_information[l][1])
    max_mean = max(mean)
    screen_result = {"Surviving Alternatives": survival_alternatives_information, "Current Max Mean": max_mean,
                     "The Estimated Times of Surviving Alternatives": estimated_times}
    return screen_result


@ray.remote
def stage2(alternatives_list_core, n1, eta):
    """
    Performs a single iteration of Stage 2 simulation and screening.

    This function runs an additional batch of simulations for each surviving
    alternative and then re-applies the screening procedure.

    Args:
        alternatives_list_core (list): The list of surviving alternatives from the
            previous stage.
        n1 (int): The number of Stage 1 simulations.
        eta (float): The screening parameter.

    Returns:
        dict: A dictionary containing:
              - "surviving alternatives": Survivors after this round of screening.
              - "max mean": The maximum observed mean.
              - "simulation time": The simulation time consumed in this stage.
              - "budget": The number of simulation runs consumed.
              - "the estimated times of surviving alternatives": Estimated times for survivors.
    """
    stage2_budget = 0
    stage2_simulation_time = 0


    # Run an additional batch of simulations for each alternative.
    for alt_info in alternatives_list_core:
        alternative = alt_info[0]
        batch_size = alt_info[2]  # Pre-calculated batch size from stage1_screen.

        for _ in range(batch_size):
            start_time = time.time()
            alternative.run_simulation()
            stage2_simulation_time += (time.time() - start_time)

        stage2_budget += batch_size

    # Screen alternatives again with the newly updated statistics.
    screen_result = screen(alternatives_list_core, n1, eta)

    return {
        "Surviving Alternatives": screen_result["Surviving Alternatives"],
        "Current Max Mean": screen_result["Current Max Mean"],
        "Cumulative Simulation Time": stage2_simulation_time,
        "Cumulative Sample Size": stage2_budget,
        "The Estimated Times of Surviving Alternatives": screen_result["The Estimated Times of Surviving Alternatives"]
    }


@ray.remote
def stage3(alternatives_list_core, h, delta):
    """
    Performs the final simulation stage (Stage 3) to select the best alternative.

    This stage ensures that the best alternative is selected with a pre-specified
    probability by running enough simulations to satisfy the indifference-zone
    selection criterion based on Rinott's constant.

    Args:
        alternatives_list_core (list): The final list of surviving alternatives from Stage 2.
        h (float): Rinott's constant, used for the final selection guarantee.
        delta (float): The indifference-zone parameter.

    Returns:
        dict: A dictionary containing:
              - "surviving alternatives": The single best alternative identified.
              - "max mean": The final mean of the best alternative.
              - "simulation time": The simulation time consumed in this stage.
              - "budget": The number of simulation runs consumed.
    """
    stage3_simulation_time = 0
    stage3_budget = 0
    means = []

    for alt_info in alternatives_list_core:
        alternative = alt_info[0]
        current_sims = alternative.get_num()
        s2 = alternative.get_s2()

        # Calculate the total number of simulations required to meet the statistical guarantee.
        required_sims = max(math.ceil((h / delta) ** 2 * s2) - current_sims, 0)

        # Run the necessary additional simulations.
        for _ in range(required_sims):
            start_time = time.time()
            alternative.run_simulation()
            stage3_simulation_time += (time.time() - start_time)

        means.append(alternative.get_mean())
        stage3_budget += required_sims

    # Identify the alternative with the highest sample mean as the best.
    best_alternative_index = np.argmax(means)

    return {
        "Surviving Alternatives": alternatives_list_core[best_alternative_index][0],
        "Current Max Mean": max(means),
        "Cumulative Simulation Time": stage3_simulation_time,
        "Cumulative Sample Size": stage3_budget
    }


def split_by_modulo(elements, num_groups):
    """
    Splits a list of elements into a specified number of groups using a modulo operator.

    This is a simple way to distribute work but may not be balanced if the `elements`
    have varying computational costs.

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


def split_into_groups(elements, values, num_groups):
    """
    Splits elements into groups, balancing the total sum of associated values in each group.

    This function implements a greedy algorithm (longest processing time first) to achieve
    load balancing. It assigns the " heaviest" items first to the least loaded groups.

    Args:
        elements (list): The list of elements to distribute.
        values (list): A list of numerical values corresponding to each element (e.g., estimated run time).
        num_groups (int): The number of groups to create.

    Returns:
        list: A list of groups, where each group is a list of elements, balanced by their values.
    """
    # Validate inputs
    if len(elements) != len(values):
        raise ValueError("Elements and values lists must have the same length")
    if num_groups <= 0:
        raise ValueError("Number of groups must be ≥1")
    if num_groups > len(elements):
        raise ValueError("Number of groups cannot exceed number of elements")

    # Create indexed elements sorted by value descending
    indexed_elements = list(zip(elements, values, range(len(elements))))
    indexed_elements.sort(key=lambda x: -x[1])

    # Initialize groups
    group_sums = [0.0] * num_groups
    grouped_elements = [[] for _ in range(num_groups)]

    # Assign elements to the least loaded groups
    for elem, val, idx in indexed_elements:
        target_group = np.argmin(group_sums)
        grouped_elements[target_group].append((elem, idx))
        group_sums[target_group] += val

    # Reconstruct groups in original order
    result = []
    for group in grouped_elements:
        sorted_group = sorted(group, key=lambda x: x[1])  # Sort by original index
        result.append([elem for elem, _ in sorted_group])

    return result


def flatten(nested_list):
    """
    Flattens a list of lists into a single list.

    Args:
        nested_list (list): A list containing other lists (e.g., `[[1, 2], [3, 4]]`).

    Returns:
        list: A single, flattened list (e.g., `[1, 2, 3, 4]`).
    """
    return [item for sublist in nested_list for item in sublist]


def GSP(alternatives, configs, replication):
    """
    Executes the Good Selection Procedure (GSP) for ranking and selection.

    This is the main driver function that orchestrates the entire multi-stage
    selection process, from initial time estimation to final selection, using
    parallel processing.

    Args:
        alternatives (list): A list of alternative objects to be evaluated.
        configs (dict): A dictionary of configuration parameters for the procedure.
            Expected keys:
            - "n0" (int): Initial sample size for time estimation. Set to 0 to skip.
            - "n1" (int): Sample size for Stage 1.
            - "r_bar" (int): Maximum number of screening iterations in Stage 2.
            - "beta" (float): Budget allocation parameter.
            - "alpha1" (float): Confidence level for Stage 1/2 screening.
            - "alpha2" (float): Confidence level for Stage 3 final selection.
            - "delta" (float): Indifference-zone parameter.
            - "Number of Processors" (int): The number of parallel cores to use.
            - "Reference Seed" (list): A list of seeds for the random number generator.
            - "CRNs" (bool): True to use Common Random Numbers, False for independent streams.
        replication (int): The current replication number of the experiment.

    Returns:
        list: A list of dictionaries, where each dictionary contains the results
              and performance metrics from each stage of the procedure.
    """
    # Extract configuration parameters from the dictionary.
    n0 = configs.get("n0", 0)
    n1 = configs["n1"]
    r_bar = configs["r_bar"]
    beta = configs["beta"]
    alpha1 = configs["alpha1"]
    alpha2 = configs["alpha2"]
    delta = configs["delta"]
    num_processors = configs["Number of Processors"]
    seed = configs["Reference Seed"]
    CRNs = configs["CRNs"]

    # Initialize results and statistics.
    total_results = []
    simulation_time = 0
    k = len(alternatives)

    # Set random number seeds for alternatives.
    if CRNs:  # Common Random Numbers ensure pairs are compared under similar conditions.
        for alt in alternatives:
            alt.set_seed([seed[0] + replication, seed[1], seed[2] + 1])
    else:  # Independent streams for each alternative.
        for alt in alternatives:
            alt.set_seed([seed[0] + replication * k + alt.get_args()[0], seed[1], seed[2]  + 1])

    # Calculate statistical parameters needed for screening and selection.
    eta = EtaFunc.find_eta(n1, alpha1, k)
    h = rinott(k, 1 - alpha2, n1 - 1)

    # STAGE 0: Initial simulation for time estimation (optional).
    if n0 != 0:
        stage0_groups = split_by_modulo(alternatives, num_processors)
        stage0_futures = [stage0_simulation.remote(group, n0) for group in stage0_groups]
        stage0_results = ray.get(stage0_futures)

        alternative_stage1_list = []
        estimate_time = []
        for group_result in stage0_results:
            for alt_info in group_result:
                alternative_stage1_list.append(alt_info)
                estimate_time.append(alt_info[1])
    else:
        # If Stage 0 is skipped, assume all alternatives take equal time.
        alternative_stage1_list = [[alt, 1] for alt in alternatives]
        estimate_time = [1] * k

    # STAGE 1: Main simulation and screening.
    stage1_start_time = time.time()
    average_s = 0

    # Split alternatives into balanced groups based on estimated computation time.
    stage1_groups = split_into_groups(alternative_stage1_list, estimate_time, num_processors)
    stage1_futures = [stage1_simulation.remote(group, n1) for group in stage1_groups]
    stage1_results = ray.get(stage1_futures)

    # Process Stage 1 results from all cores.
    stage1_alternatives_info = []
    stage1_estimated_times_list = []

    for result in stage1_results:
        stage1_alternatives_info.append(result[0])
        simulation_time += result[1]
        average_s += result[2] / k  # Calculate the average of sqrt(s^2/time).
        stage1_estimated_times_list.append(result[3])

    total_budget = k * n1
    flat_stage1_alternatives = flatten(stage1_alternatives_info)
    flat_stage1_times = flatten(stage1_estimated_times_list)

    # Distribute alternatives for parallel screening.
    screen_groups = split_into_groups(flat_stage1_alternatives, flat_stage1_times, num_processors)
    screen_futures = [stage1_screen.remote(group, n1, beta, r_bar, eta, average_s)
                      for group in screen_groups]
    screen_results = ray.get(screen_futures)

    # Consolidate screening results.
    surviving_alternatives = []
    surviving_estimated_times = []
    stage1_means = []

    for result in screen_results:
        surviving_alternatives.append(result["Surviving Alternatives"])
        stage1_means.append(result["Current Max Mean"])
        surviving_estimated_times.append(result["The Estimated Times of Surviving Alternatives"])

    stage1_alternatives_end = flatten(surviving_alternatives)
    stage2_estimated_times = flatten(surviving_estimated_times)

    # Record Stage 1 performance metrics.
    stage1_result = {
        "Current Max Mean": float(max(stage1_means)),
        "Cumulative Simulation Time (s)": simulation_time,
        "Cumulative Sample Size": total_budget,
        "Wall-Clock Time (s)": time.time() - stage1_start_time,
        "Utilization": simulation_time / ((time.time() - stage1_start_time) * num_processors),
        "Number of Surviving Alternatives": len(stage1_alternatives_end)
    }
    total_results.append(stage1_result)

    # STAGE 2: Iterative screening.
    stage2_alternatives = stage1_alternatives_end

    for i in range(r_bar):
        # Terminate if only one alternative remains.
        if len(stage2_alternatives) == 1:
            stage2_result = {
                "Current Max Mean": float(stage2_alternatives[0][0].get_mean()),
                "Cumulative Simulation Time (s)": simulation_time,
                "Cumulative Sample Size": total_budget,
                "Wall-Clock Time (s)": time.time() - stage1_start_time,
                "Utilization": simulation_time / ((time.time() - stage1_start_time) * num_processors),
                "Number of Surviving Alternatives": 1
            }
            total_results.append(stage2_result)
            break

        # Distribute surviving alternatives for the next round of simulation and screening.
        stage2_groups = split_into_groups(stage2_alternatives, stage2_estimated_times, num_processors)
        stage2_futures = [stage2.remote(group, n1, eta) for group in stage2_groups]
        stage2_results = ray.get(stage2_futures)

        # Process results from the current Stage 2 iteration.
        new_stage2_alternatives = []
        stage2_means = []
        new_stage2_times = []

        for result in stage2_results:
            new_stage2_alternatives.append(result["Surviving Alternatives"])
            stage2_means.append(result["Current Max Mean"])
            simulation_time += result["Cumulative Simulation Time"]
            total_budget += result["Cumulative Sample Size"]
            new_stage2_times.append(result["The Estimated Times of Surviving Alternatives"])

        stage2_alternatives = flatten(new_stage2_alternatives)
        stage2_estimated_times = flatten(new_stage2_times)

        # Record performance metrics for this iteration.
        stage2_result = {
            "Current Max Mean": float(max(stage2_means)),
            "Cumulative Simulation Time (s)": simulation_time,
            "Cumulative Sample Size": total_budget,
            "Wall-Clock Time (s)": time.time() - stage1_start_time,
            "Utilization": simulation_time / ((time.time() - stage1_start_time) * num_processors),
            "Number of Surviving Alternatives": len(stage2_alternatives)
        }
        total_results.append(stage2_result)

    # STAGE 3: Final selection.

    # Distribute the final candidates for the final simulation stage.
    stage3_groups = split_into_groups(stage2_alternatives, stage2_estimated_times, num_processors)
    stage3_futures = [stage3.remote(group, h, delta) for group in stage3_groups]
    stage3_results = ray.get(stage3_futures)

    # Process final results.
    stage3_means = []
    stage3_alternatives = []

    for result in stage3_results:
        stage3_alternatives.append(result["Surviving Alternatives"])
        stage3_means.append(result["Current Max Mean"])
        simulation_time += result["Cumulative Simulation Time"]
        total_budget += result["Cumulative Sample Size"]
    best_alternative_id = np.argmax(stage3_means)
    best_alternative = stage3_alternatives[best_alternative_id]

    # Record Stage 3 performance metrics.
    stage3_result = {
        "Current Max Mean": float(max(stage3_means)),
        "Cumulative Simulation Time (s)": simulation_time,
        "Cumulative Sample Size": total_budget,
        "Wall-Clock Time (s)": time.time() - stage1_start_time,
        "Number of Surviving Alternatives": 1,
        "Utilization": simulation_time / ((time.time() - stage1_start_time) * num_processors)
    }
    total_results.append(stage3_result)

    # Final consolidated results.
    final_result = {
        "Best Alternative": best_alternative.get_args(),
        "Total Simulation Time (s)": simulation_time,
        "Total Sample Size": total_budget,
        "Wall-Clock Time (s)": time.time() - stage1_start_time,
        "Utilization": simulation_time / ((time.time() - stage1_start_time) * num_processors)
    }
    total_results.append(final_result)

    return total_results