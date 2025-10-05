import math
import time

import numpy as np
import ray


@ray.remote
def simulation_batch(alternatives, batch):
    simulation_time = 0
    means = []
    core_budget = 0
    for alt in alternatives:
        mean = 0
        for i in range(batch):
            start_time = time.time()
            mean += alt.run_simulation()/batch
            simulation_time += (time.time()-start_time)
        means.append(mean)
        core_budget += batch
    best_alt_id = np.argmax(means)
    best_alternative = alternatives[best_alt_id]
    return best_alternative, means[best_alt_id], simulation_time, core_budget

def custom_procedure(alternatives, configs, replication):
    N = configs["N"]
    num_processors = configs["Number of Processors"]
    seed = configs["Reference Seed"]
    k = len(alternatives)
    start_time = time.time()
    simulation_time = 0
    cumulative_sample_size = 0
    results = []

    for alt in alternatives:
        alt.set_seed([seed[0] + (replication - 1) * k + alt.get_args()[0], seed[1], seed[2] + 1])

    groups = [[] for _ in range(num_processors)]
    for index, element in enumerate(alternatives):
        group_id = index % num_processors
        groups[group_id].append(element)

    batch = math.floor(N/k)
    futures = [simulation_batch.remote(group, batch) for group in groups]
    core_results = ray.get(futures)
    means = []
    survival_alternatives = []
    for result in core_results:
        survival_alternatives.append(result[0])
        means.append(result[1])
        simulation_time += result[2]
        cumulative_sample_size += result[3]

    best_alt_id = np.argmax(means)
    best_alternative = survival_alternatives[best_alt_id]

    results.append({
        "Best Alternative": best_alternative.get_args(),
        "Total Simulation Time (s)": simulation_time,
        "Total Sample Size": cumulative_sample_size,
        "Wall-Clock Time (s)": time.time() - start_time,
        "Utilization": simulation_time / (num_processors * (time.time() - start_time))
    })
    return results

