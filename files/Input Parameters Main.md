| [**Main Page**](../README.md) | [**How to Use PyPRS**](How%20to%20Use%20PyPRS.md) | [**Output**](Output.md) | [**A Demo Application**](A%20Demo%20Application.md) |

# Input Parameters for Built-in Procedures
## 📋 Input Parameters for GSP

| Parameter                  | Explanation                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `n0`                       | Sample size for Stage 0.                                                   |
| `n1`                       | Sample size for Stage 1.                                                   |
| `r_bar`                    | Maximum number of rounds in Stage 2.                                       |
| `beta`                     | Average number of simulation samples per alternative in Stage 2.           |
| `alpha1, alpha2`           | Splits of the tolerable PICS alpha (i.e., `alpha = alpha1 + alpha2`).      |
| `delta`                    | Indifference-zone (IZ) parameter.                                          |
| `Number of Processors`     | Number of processors used to run the procedure.                            |
| `Repeat`                   | Number of times the problem is repeatedly solved.                          |
| `Reference Seed (Optional)` | Seed used to initialize random number generators.                          |


## 📋 Input Parameters for KT Procedure

| Parameter                  | Explanation                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `delta`                    | Indifference-zone (IZ) parameter.                                          |
| `alpha`                    | Tolerable Probability of Incorrect Selection (PICS).                       |
| `g`                        | Number of alternatives in a group.                                         |
| `n0`                       | First-stage sample size for the KN procedure.                              |
| `Number of Processors`     | Number of processors used to run the procedure.                            |
| `Repeat`                   | Number of times the problem is repeatedly solved.                          |
| `Reference Seed (Optional)` | Seed used to initialize random number generators.                          |

## 📋 Input Parameters for PASS Procedure

| Parameter                  | Explanation                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `n0`                       | Initial sample size.                                                       |
| `Delta`                    | Number of simulation samples taken per alternative each time.              |
| `c`                        | Constant that determines the boundary function.                            |
| `Termination Sample Size`  | Forces the procedure to stop when sample size exceeds this value.           |
| `Worker Elimination`       | Whether workers are used to eliminate alternatives.                        |
| `Number of Processors`     | Number of processors used to run the procedure.                            |
| `Repeat`                   | Number of times the problem is repeatedly solved.                          |

## 📋 Input Parameters for FBKT Procedure

| Parameter                  | Explanation                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `N`                        | Total sampling budget.                                                     |
| `n0`                       | Sample size needed at the initial stage for seeding.                       |
| `phi`                      | Positive integer that determines budget allocation.                        |
| `Number of Processors`     | Number of processors used to run the procedure.                            |
| `Repeat`                   | Number of times the problem is repeatedly solved.                          |
| `Reference Seed (Optional)` | Seed used to initialize random number generators.                          |

<a href="../README.md">Back to Main Page</a>

