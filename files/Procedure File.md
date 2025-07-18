| [**Main Page**](../README.md) | [**How to Use PyPRS**](How%20to%20Use%20PyPRS.md) | [**Output**](Output.md) | [**A Demo Application**](A%20Demo%20Application.md) |

# The Procedure (`.py`) File

In this file, users need to define a function named `custom_procedure`, which implements the computational steps for one run of the custom procedure. The function takes specific input arguments and returns a dictionary with required output keys. 

**Prerequisites**

Before creating the `custom_procedure` function, ensure the following:
- The **alternatives information file** (**`.txt`** file) is created.
- The **simulation function file** (**`.py`** file) is created.
- **Input parameters** are specified in the GUI, including their names and values.

**Arguments**
- **`alternatives`**: A list of `SampleGenerate` class instances.
  - Each instance corresponds to an alternative specified in the **alternatives information file** (**`.txt`** file).
  - The `SampleGenerate` class provides methods to manage simulations and alternative details, including:
    - `alternatives[i].get_args()`: Retrieves the alternative's parameter information as a **list** of floats, formatted consistently with the structure defined in the **alternatives information file**.
    - `alternatives[i].set_seed(seed)`: Sets the random number seed for the alternative's next simulation run where `seed` is a **list** containing 3 non-negative integers.
    - `alternatives[i].get_seed()`: Retrieves the random number seed for the alternative's next simulation run as a **list** of 3 non-negative integers.
    - `alternatives[i].run_simulation()`: Executes the `simulation_function`, passing the values of `alternatives[i].get_args()` to `argsSim` and `alternatives[i].get_seed()` to `seedSim`, and returns the simulation output as a float.
    - `alternatives[i].get_mean()`: Retrieves the alternative's sample mean as a float.
    - `alternatives[i].set_mean(mean)`: Sets the alternative's sample mean where `mean` is a float.
    - `alternatives[i].get_num()`: Retrieves the alternative's sample size as an integer.
    - `alternatives[i].set_num(num)`: Sets the alternative's sample size where `num` is an integer.
    - `alternatives[i].get_s2()`: Retrieves the alternative's sample variance as a float.
    - `alternatives[i].set_s2(s2)`: Sets the alternative's sample variance where `s2` is a float.
    - `alternatives[i].get_simulation_time()`: Retrieves the total time spent in generating simulation samples for the alternative as a float.
    - `alternatives[i].set_simulation_time(simulation_time)`: Sets the alternative's simulation time where `simulation_time` is a float.
- **`configs`**: A dictionary containing input parameters.
  - Keys are parameter names, and values are the corresponding values specified in the GUI.
- **`replication`**: An integer that records the number of times the procedure has been applied to solve the problem.
  - Useful for managing random number seeds or implementing Common Random Numbers.
 
**Selection Process**

Based on the descriptions of **Arguments**, with `alternatives`, users can generate simulation samples from each alternative and get or set various details for each alternative. The `configs` dictionary provides procedure-specific input parameters, while `replication` helps manage random number seeds or implement Common Random Numbers. With these arguments, users can implement a custom selection process that includes defining how the best alternative is selected and the procedure leverages Ray for parallelization. For guidance on programming and using Ray, refer to the official <a href="https://docs.ray.io/en/latest/index.html">Ray site</a>.


**Return**

The function must return a dictionary with the following five keys:
- **Best Alternative**: Parameter information for the selected best alternative.
- **Total Simulation Time (s)**: Total simulation time used (in appropriate units, e.g., seconds).
- **Total Sample Size**: Total number of samples generated during the procedure.
- **Wall-Clock Time (s)**: Real-world time elapsed during the procedure (in seconds).
- **Utilization**: A measure of processor efficiency in parallel computing, calculated as:  utilization = total simulation time / (number of processors * wall-clock time).

**Function Template**
```python
def custom_procedure(alternatives, configs, replication):
    # Extract input parameter information from configs
    param1 = configs.get("Repeat")
    param2 = configs.get("Number of Processors")
    # Extract more as needed...

    # Custom designed selection process
    # ...

    return {
        "Best Alternative": ...,
        "Total Simulation Time (s)": ...,
        "Total Sample Size": ...,
        "Wall-Clock Time (s)": ...,
        "Utilization": ...
    }
```




<a href="How to Use PyPRS.md#PF">Back to How to Use PyPRS</a>
