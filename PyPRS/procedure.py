import importlib.util
import os
import pathlib
import random

from scipy import stats

import numpy as np
import ray

from .Procedures.FBKT import FBKT
from .Procedures.GSP import GSP
from .Procedures.PASS import PASS
from .Procedures.KT import KT
from .SampleGen.SampleGenerate import SampleGenerate

class Procedure:
    """
    A class to manage and run different ranking and selection procedures.

    This class acts as a dispatcher. It can use built-in procedures like GSP, KT,
    PASS, and FBKT, or it can dynamically load a custom procedure from a user-provided Python file.
    """
    procedures_map = {
        "GSP": GSP,
        "KT": KT,
        "PASS": PASS,
        "FBKT": FBKT,
    }

    def __init__(self, select_procedure: str = None):
        """
        Initializes the Procedure class and sets the desired selection procedure.

        The procedure can be identified by a predefined string key (for built-in
        procedures) or by a file path to a Python module. The module is expected
        to contain a function named 'custom_procedure'.

        Args:
            select_procedure (str, optional): The name of the built-in procedure
                                             or a path to a .py file.
                                             Defaults to None.
        """
        self.results = []
        self.CRNs = False
        self.procedure_function = None

        if not select_procedure:
            print("No procedure selected.")
            return

        # Case 1: The input is a key for a built-in procedure.
        if select_procedure in self.procedures_map:
            self.procedure_function = self.procedures_map[select_procedure]
            print(f"Select the built-in procedure: {select_procedure}")

        # Case 2: The input is a file path, so try to load the procedure from it.
        elif os.path.exists(select_procedure):
            self.procedure_function = load_procedure_from_file(select_procedure)

        # Case 3: The input is neither a valid key nor a valid path.
        else:
            print(
                f"Input '{select_procedure}' is not a built-in procedure or a valid file path."
            )

    def get_result(self):
        """Returns the collected results from all replications."""
        return self.results

    def set_CRNs(self):
        """Enables the use of Common Random Numbers (CRNs) for the procedures."""
        self.CRNs = True

    def run_procedure(self, args):
        """
        Executes the selected procedure for a specified number of replications.

        This method initializes Ray for parallel processing, runs the procedure
        multiple times, collects the results, and handles saving the output.

        Args:
            args (dict): A dictionary containing all configuration parameters for
                         the run, such as number of processors, file paths, and
                         procedure-specific settings.

        Returns:
            list: A list of dictionaries, where each dictionary contains the
                  summary results from the final stage of one replication.
        """
        # Set a random replication_seed if one is not provided in the arguments.
        if args.get("Reference Seed") is None:
            args["Reference Seed"] = [random.randint(0, 2 ** 50 - 1), random.randint(0, 2 ** 47 - 1),
                                      random.randint(0, 2 ** 47 - 1)]
        args["CRNs"] = self.CRNs
        num_processors = args["Number of Processors"]
        repeat = args.get("Repeat", 1)
        alt_param_file = args["Alternatives Information File"]
        sim_func_file = args["Simulation Function File"]
        process_results = []

        progress_callback = args.get("progress_callback")

        # Send an initial "0%" progress update before the loop starts.
        if progress_callback:
            initial_message = f"Macroreplication: 0/{repeat}\n"
            print(f"\r{initial_message}", end="", flush=True)
            progress_callback({'text': initial_message, 'value': 0})

        # Initialize Ray for parallel computation.
        ray.init(num_cpus=num_processors)
        for replication in range(1, repeat + 1):
            # The main work of the replication happens here.
            simulation_func = get_sim_func(sim_func_file)
            alternatives = read_alternatives_para(alt_param_file, simulation_func)
            result = self.procedure_function(alternatives, args, replication)
            self.results.append(result[-1])  # Collect summary result.
            process_results.append(result[:-1])  # Collect detailed process results.

            # --- Progress is updated *after* a replication is complete. ---
            percent = (replication / repeat) * 100
            progress_message = f"Macroreplication: {replication}/{repeat}"

            print(f"\r{progress_message}{' ' * 10}", end="", flush=True)  # Add spaces to clear previous line.

            if progress_callback:
                # Send a dictionary with both text and percentage value for the UI.
                progress_callback({'text': progress_message, 'value': percent})

        # Shutdown Ray after all replications are complete.
        ray.shutdown()

        completion_message = f"Completed all {repeat} Macroreplications."
        print(f"\n{completion_message}")

        # Save the detailed and summary results to files.
        if self.procedure_function.__name__ in ("FBKT", "KT", "GSP", "PASS"):
            save_detailed_results(process_results, self.procedure_function.__name__)
        save_summary_results(self.results, self.procedure_function.__name__)

        return self.results


def read_alternatives_para(file_path, simulation_func):
    """
    Reads and parses an alternatives configuration file to create SampleGenerate objects.

    The file should contain space-separated numerical values per line, where the first
    value is the alternative ID and subsequent values are its parameters.

    Args:
        file_path (str): The path to the text file with alternative configurations.
        simulation_func (callable): A reference to the simulation function for
                                    initializing SampleGenerate objects.

    Returns:
        list[SampleGenerate]: A list of initialized SampleGenerate instances.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If the file contains empty lines, missing IDs, or non-numeric parameters.
    """

    def to_number(s):
        """
        Converts a string to the appropriate numeric type (int or float).

        Args:
            s (str): The input string to convert.

        Returns:
            int or float: The converted numeric value.

        Raises:
            ValueError: If the string cannot be converted to a number.
        """
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                raise ValueError(f"Cannot convert to number: '{s}'")

    alternatives = []

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Clean and split the line content into parts.
                parts = [p for p in line.strip().split(" ") if p]

                # Skip empty lines.
                if not parts:
                    continue

                # Convert all parts to numeric values.
                try:
                    numeric_parts = [to_number(p) for p in parts]
                except ValueError as e:
                    raise ValueError(f"Line {line_num}: {str(e)}")

                # Validate that at least one parameter (the ID) exists.
                if len(numeric_parts) < 1:
                    raise ValueError(f"Line {line_num}: Missing alternative ID")

                # Initialize and configure the SampleGenerate instance for this alternative.
                alt = SampleGenerate(simulation_function=simulation_func)
                alt.set_args(numeric_parts)
                alternatives.append(alt)
    except FileNotFoundError:
        raise FileNotFoundError(f"Alternatives file not found: {file_path}")

    return alternatives


def get_sim_func(file_path):
    """
    Dynamically loads a 'simulation_function' from a specified Python file.

    Args:
        file_path (str): The path to the Python file containing the simulation function.

    Returns:
        callable: The loaded simulation function.
    """
    spec = importlib.util.spec_from_file_location("simulation_function", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simulation_function


def save_detailed_results(data_list: list, procedure_name: str):
    """
    Saves the detailed, step-by-step results of each replication to a file.

    The output file is named 'DetailedResults.txt' and is saved in a structured
    output directory. The formatting depends on the `process_type`.

    Args:
        data_list (list): A list where each element represents a replication's
                          detailed data (which is a list of dictionaries).
        procedure_name (str): The name of the procedure ("GSP", "KT", etc.),
                            which determines the formatting logic.
    """
    try:
        # --- File Path Setup ---
        # Get the current working directory (where the user ran the command)
        base_dir = os.getcwd()
        output_dir = os.path.join(base_dir, 'Output', 'Detailed Results')
        os.makedirs(output_dir, exist_ok=True)

        output_file_path = os.path.join(output_dir, 'DetailedResults.txt')

        # --- Data Processing and File Writing ---
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Write a dynamic header to the top of the file.
            f.write(f"{procedure_name} Detailed Results\n")
            f.write("=" * 30 + "\n\n")

            for i, replication_data in enumerate(data_list, start=1):
                f.write(f"Macroreplication {i} Results:\n")
                f.write("-" * 20 + "\n")

                # --- GSP Logic ---
                if procedure_name == "GSP":
                    num_records = len(replication_data)
                    for j, record in enumerate(replication_data):
                        if j == 0:
                            f.write("Stage 1:\n")
                        elif j == num_records - 1:
                            f.write("Stage 3:\n")
                        else:
                            f.write(f"Stage 2 Round {j}:\n")
                        for key, value in record.items():
                            f.write(f"  {key}: {value}\n")
                        f.write("\n")

                # --- KT Logic ---
                elif procedure_name == "KT":
                    for j, record in enumerate(replication_data, start=1):
                        f.write(f"Round {j}:\n")
                        for key, value in record.items():
                            f.write(f"  {key}: {value}\n")
                        f.write("\n")

                # --- PASS Logic ---
                elif procedure_name == "PASS":
                    for j, record in enumerate(replication_data):
                        if j == 0:
                            f.write("Data Collection Phase:\n")
                        else:
                            f.write(f"Interceptor {j}:\n")
                        for key, value in record.items():
                            f.write(f"  {key}: {value}\n")
                        f.write("\n")

                # --- FBKT Logic ---
                elif procedure_name == "FBKT":
                    for j, record in enumerate(replication_data):
                        if j == 0:
                            f.write("Seeding Stage:\n")
                        else:
                            f.write(f"Round {j}:\n")
                        for key, value in record.items():
                            f.write(f"  {key}: {value}\n")
                        f.write("\n")

                else:
                    f.write(f"Error: Unknown process type '{procedure_name}'.\n")
                    return

                f.write("=" * 20 + "\n\n")

        print(f"Detailed Results saved to: {os.path.abspath(output_file_path)}")

    except Exception as e:
        print(f"An error occurred: {e}")


def save_summary_results(data_list: list[dict], process_type: str):
    """
    Analyzes summary results from all replications, calculates statistics
    (mean, confidence interval), and saves them to a summary report file.

    Args:
        data_list (list[dict]): A list of dictionaries, where each dictionary
                                is the final summary result from one replication.
        process_type (str): The name of the procedure, used for the report title.
    """
    try:
        # --- 1. Set up the output file path ---
        # Get the current working directory
        base_dir = pathlib.Path.cwd()
        output_dir = base_dir / 'Output' / 'Summary Results'

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / 'SummaryResults.txt'

        # --- 2. Process the input data ---
        if not data_list:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(f"{process_type} Summary Results\n")
                f.write("Input data list is empty. No analysis performed.\n")
            print(f"Summary Results saved to: {output_file_path.resolve()}")
            return

        all_keys = list(data_list[0].keys())
        id_key = all_keys[0]  # The first key is assumed to be the identifier.
        numeric_keys = all_keys[1:]
        num_replications = len(data_list)

        # --- 3. Write data to the file ---
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Write a dynamic header for the report.
            f.write(f"{process_type} Summary Results\n")
            f.write("=" * 50 + "\n\n")

            for i, item in enumerate(data_list):
                f.write(f"Macroreplication {i + 1} Results:\n")
                f.write(f"  {id_key}: {item[id_key]}\n")
                for key in numeric_keys:
                    f.write(f"  {key}: {item[key]}\n")
                f.write("\n")

            f.write("=" * 50 + "\n")
            f.write("Summary Statistics:\n")
            f.write("-" * 50 + "\n")

            # Calculate and write statistics for each numeric key.
            for key in numeric_keys:
                values = [d[key] for d in data_list]
                mean_val = np.mean(values)

                f.write(f"{key}:\n")
                f.write(f"  Mean: {mean_val:.4f}\n")

                # Calculate confidence interval only if there are enough replications.
                if num_replications >= 10:
                    degrees_freedom = num_replications - 1
                    sample_std = np.std(values, ddof=1)
                    std_error = sample_std / np.sqrt(num_replications)

                    if std_error > 0:
                        ci = stats.t.interval(confidence=0.95, df=degrees_freedom, loc=mean_val, scale=std_error)
                        f.write(f"  95% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})\n")
                    else:
                        f.write(
                            f"  95% Confidence Interval: ({mean_val:.4f}, {mean_val:.4f}) (all values are identical)\n")
                f.write("\n")

        print(f"Summary Results saved to: {output_file_path.resolve()}")

    except Exception as e:
        print(f"An error occurred: {e}")


def load_procedure_from_file(file_path: str):
    """
    Dynamically loads a function named 'custom_procedure' from a given Python file path.

    Args:
        file_path (str): The absolute or relative path to the .py file.

    Returns:
        callable or None: The 'custom_procedure' function if found, otherwise None.
    """
    try:
        # Create a unique module name from the file path to avoid import conflicts.
        module_name = os.path.basename(file_path).replace('.py', '')

        # Create a module specification from the file's location.
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"Could not create module spec from path: {file_path}")
            return None

        custom_module = importlib.util.module_from_spec(spec)

        # Execute the module to make its contents available.
        spec.loader.exec_module(custom_module)

        # Attempt to get the specific function from the loaded module.
        procedure_func = getattr(custom_module, "custom_procedure")
        print(f"Successfully loaded 'custom_procedure' from file: {file_path}")
        return procedure_func

    except AttributeError:
        print(
            f"Error: The file '{file_path}' does not contain a function named 'custom_procedure'."
        )
        return None
    except Exception as e:
        # Catch other potential import errors, such as syntax errors in the user's file.
        print(f"Failed to load procedure from '{file_path}'. Reason: {e}")
        return None