| [**Main Page**](../README.md) | [**How to Use PyPRS**](How%20to%20Use%20PyPRS.md) | [**Output**](Output.md) | [**A Demo Application**](A%20Demo%20Application.md) |
# Uploading Files

## 1 The Alternatives Information File (`.txt` File)
In this file, users need to define parameter information for each alternative.
- **Structure**:
  - Each line represents one alternative.
  - The first entry is the index of the alternative (starting from 1, incrementing by 1).
  - The following entries are the parameter values (e.g., `x1`, `x2`, `x3`).
- **Example Format**:
```markdown
1 0.5 0.8 1.2
2 1.3 0.6 0.9
3 0.7 1.1 1.0
```
In this example:
- Alternative 1: `x1 = 0.5`, `x2 = 0.8`, `x3 = 1.2`
- Alternative 2: `x1 = 1.3`, `x2 = 0.6`, `x3 = 0.9`
- Alternative 3: `x1 = 0.7`, `x2 = 1.1`, `x3 = 1.0`

*Notes*: All values must be numerical. Non-numeric values will cause errors.

## 2 The Simulation Function File (`.py` File)
In this file, users need to define a Python function named `simulation_function`, which is responsible for generating a **simulation sample** from a given alternative using specific random number sequences.

**Arguments**
- **`argsSim`** (List of Floats):
  - Contains the parameter information of the alternative.
  - Format matches the `.txt` file:
    - `argsSim[0]`: Index of the alternative.
    - `argsSim[1]`, `argsSim[2]`, etc: Parameter values (e.g., `x1`, `x2`, `x3`).
- **`seedSim`**  (List of Integers):
  - A list of three positive integers used to seed the random number generators.

**Random Number Generation**

PyPRS uses the `MRG32k3a_numba` package for random number generation. (For more details about the `MRG32k3a_numba` package, please go to <a href="./MRG32k3a_numba.md" target="_blank">MRG32k3a_numba</a>.) If the simulation process requires multiple sources of randomness (e.g., interarrival times and service times in a queueing model), users can initialize multiple instances of `MRG32k3a_numba`. A source of randomness refers to distinct needs for random numbers in a simulation model. For example, a simple single-server queueing model might designate two such sources of randomness: one for interarrival times and one for service times.

In the `simulation_function`, whenever instances of `MRG32k3a_numba` are initialized, proper seeding is required. For example, when two independent sources of randomness are needed within the `simulation_function`, the `seedSim` parameter can be used to seed the instances as follows:

 - The first instance is seeded using `[seedSim[0], seedSim[1]+1, seedSim[2]]`.
 - The second instance is seeded using `[seedSim[0], seedSim[1]+2, seedSim[2]]`.

**Numba Acceleration**

PyPRS allows users to apply the `numba` library to accelerate computationally intensive parts of the `simulation_function`. By using Numba's Just-In-Time compilation, users can optimize specific code segments for faster execution. For details on how to implement and optimize with Numba, refer to the official  <a href="https://numba.pydata.org/">Numba site</a>


**Return**

The output of the function should be a float that records the output for one run of the simulation model.

**Function Template**
```python
from mrg32k3a_numba import MRG32k3a_numba
import numba
import numpy as np

# Apply the @numba.njit decorator to accelerate computations
@numba.njit(cache=True)
def simulation_logic(argsSim, rng1, rng2) -> float:
    # Users can modify this function to implement custom simulation logic
    # Extract alternative's parameter information from argsSim
    idx= argsSim[0]  # Index
    x1 = argsSim[1]  # Parameter 1
    x2 = argsSim[2]  # Parameter 2
    x3 = argsSim[3]  # Parameter 3
    # Extract more as needed...

    random_val_1 = rng1.random() # Get a random value from rng1
    random_val_2 = rng2.random() # Get a random value from rng2
    result = (x1 * random_val_1 + x2 * random_val_2) / x3 # Example calculation
    return float(result)

def simulation_function(argsSim, seedSim) -> float:
    # Initialize random number generators based on seedSim
    rng1 = MRG32k3a_numba(np.array[seedSim[0], seedSim[1] + 1, seedSim[2]]) # First random number generator
    rng2 = MRG32k3a_numba(np.array[seedSim[0], seedSim[1] + 2, seedSim[2]]) # Second random number generator
    # Add more random number generators as needed...

    # Call the accelerated simulation logic
    sample = simulation_logic(argsSim, rng1, rng2)

    # Return the result as a float
    return float(sample)
```
<a href="../README.md">Back to Main Page</a>
