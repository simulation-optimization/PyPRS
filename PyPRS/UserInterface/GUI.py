import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import importlib.util
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import pprint
from tkinter.ttk import Entry

from ..procedure import Procedure


# ==============================================================================
# START: Plotting functions
# ==============================================================================

def parse_results_file(file_path):
    """
    Parses a 'DetailedResults.txt' file and structures the data into a nested dictionary.

    The structure is: {replication_id: [{stage_name: {metric: value, ...}}, ...]}

    Args:
        file_path (str): The path to the detailed results file.

    Returns:
        dict: A dictionary containing the parsed data from all replications.
    """
    replications = {}
    current_replication = None
    stage_data = []
    current_stage_name = None
    record = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the start of the actual data, skipping any header.
    start_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Macroreplication"):
            start_index = i
            break

    # Process lines from the start of the data.
    for line in lines[start_index:]:
        line = line.strip()
        if not line or line.startswith("="):
            continue

        # Detect the start of a new replication block.
        replication_match = re.match(r"Macroreplication (\d+)", line)
        if replication_match:
            # Save the completed data for the previous replication before starting a new one.
            if current_replication is not None:
                if record and current_stage_name:
                    stage_data.append({current_stage_name: record})
                replications[current_replication] = stage_data

            # Reset state for the new replication.
            current_replication = int(replication_match.group(1))
            stage_data = []
            record = {}
            current_stage_name = None
            continue

        # Detect the start of a new stage/round within a replication.
        if line.endswith(':') and not line.startswith(" "):
            if record and current_stage_name:
                stage_data.append({current_stage_name: record})
            current_stage_name = line[:-1].strip()
            record = {}
            continue

        # Parse key-value pairs for the current stage.
        kv_match = re.match(r"(.+?):\s+(.+)", line)
        if kv_match and current_stage_name:
            key = kv_match.group(1).strip()
            value_str = kv_match.group(2).strip()
            # Try to convert value to float, otherwise keep as string.
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
            record[key] = value

    # Save the very last record after the loop finishes.
    if current_replication is not None:
        if record and current_stage_name:
            stage_data.append({current_stage_name: record})
        replications[current_replication] = stage_data

    return replications


def plot_gsp(data, process_type, output_dir, results_callback=None):
    """
    Generates and saves plots specifically for the GSP procedure results.
    This function creates a plot for each metric, with a hierarchical x-axis
    to distinguish Stage 1, Stage 2 rounds, and Stage 3.

    Args:
        data (dict): The parsed data from `parse_results_file`.
        process_type (str): The name of the procedure (e.g., "GSP").
        output_dir (str): The directory to save the generated plot images.
        results_callback (callable, optional): A function to send status messages to the GUI.
    """
    num_replications = len(data)
    if num_replications == 0:
        if results_callback:
            results_callback("No replication data found to plot.")
        return

    # Use the first replication to determine the structure (stage names, metrics).
    first_replication_data = data[1]
    stage_names = [list(d.keys())[0] for d in first_replication_data]

    if not first_replication_data or not list(first_replication_data[0].values()):
        if results_callback:
            results_callback(f"No data records found for {process_type} plotting.")
        return

    # Dynamically find all numeric keys in the first record to create plots for.
    metrics_to_plot = []
    for key, value in list(first_replication_data[0].values())[0].items():
        if isinstance(value, (int, float)):
            metrics_to_plot.append(key)

    # Aggregate data from all replications.
    aggregated_data = {metric: {stage: [] for stage in stage_names} for metric in metrics_to_plot}
    for rep_id, stages in data.items():
        for i, stage_data in enumerate(stages):
            stage_name = list(stage_data.keys())[0]
            for metric, value in stage_data[stage_name].items():
                if metric in metrics_to_plot and isinstance(value, (int, float)):
                    if metric in aggregated_data and stage_name in aggregated_data[metric]:
                        aggregated_data[metric][stage_name].append(value)

    plot_files = []
    for metric, stage_values in aggregated_data.items():
        if not any(stage_values.values()):
            continue

        # Calculate average values for plotting.
        avg_values = [np.mean(stage_values[stage]) if stage_values[stage] else 0 for stage in stage_names]

        fig, ax = plt.subplots(figsize=(12, 8))
        x_coords = np.arange(len(stage_names))
        y_label_text = f'Average {metric}'
        ax.plot(x_coords, avg_values, marker='o', linestyle='-', label=y_label_text)

        # Add 95% confidence intervals if there are enough replications.
        if num_replications >= 10:
            std_devs = [np.std(stage_values[stage], ddof=1) if len(stage_values[stage]) > 1 else 0 for stage in
                        stage_names]
            confidence_intervals = [
                t.ppf(0.975, len(stage_values[stage]) - 1) * sd / np.sqrt(len(stage_values[stage])) if len(
                    stage_values[stage]) > 1 else 0
                for sd, stage in zip(std_devs, stage_names)
            ]
            lower_bound = np.array(avg_values) - np.array(confidence_intervals)
            upper_bound = np.array(avg_values) + np.array(confidence_intervals)
            ax.fill_between(x_coords, lower_bound, upper_bound, color='b', alpha=0.2, label='95% Confidence Interval')

        # Multi-line X-axis Labeling
        final_labels = []
        for name in stage_names:
            if "Stage 2" in name:
                round_text = name.replace("Stage 2 ", "")
                final_labels.append(f"Stage 2\n{round_text}")
            else:
                final_labels.append(name)

        ax.set_xticks(x_coords)
        ax.set_xticklabels(final_labels, rotation=0, ha='center', fontsize=10)

        ax.set_xlabel("Stage", fontsize=12, fontweight='bold', labelpad=15)

        ax.set_title(f'{process_type}: {y_label_text} per Stage')

        ax.set_ylabel(y_label_text)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        clean_y_label = y_label_text.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        plot_filename = os.path.join(output_dir, f'{process_type}_{clean_y_label}_by_Stage.png')
        plt.savefig(plot_filename)
        plt.close(fig)
        plot_files.append(plot_filename)

    if plot_files and results_callback:
        results_callback(f"Figures saved in: {os.path.abspath(output_dir)}")


def plot_kt_fbkt(data, process_type, output_dir, results_callback=None):
    """
    Generates and saves plots for the KT and FBKT procedures. These have a simple
    x-axis representing rounds.
    """
    num_replications = len(data)
    if num_replications == 0:
        if results_callback:
            results_callback("No replication data found to plot.")
        return

    first_replication_data = data[1]
    stage_names = [list(d.keys())[0] for d in first_replication_data]

    if not first_replication_data or not list(first_replication_data[0].values()):
        if results_callback:
            results_callback(f"No data records found for {process_type} plotting.")
        return

    metrics_to_plot = []
    for key, value in list(first_replication_data[0].values())[0].items():
        if isinstance(value, (int, float)):
            metrics_to_plot.append(key)

    aggregated_data = {metric: {stage: [] for stage in stage_names} for metric in metrics_to_plot}

    for rep_id, stages in data.items():
        for i, stage_data in enumerate(stages):
            stage_name = list(stage_data.keys())[0]
            for metric, value in stage_data[stage_name].items():
                if metric in metrics_to_plot and isinstance(value, (int, float)):
                    if metric in aggregated_data and stage_name in aggregated_data[metric]:
                        aggregated_data[metric][stage_name].append(value)

    plot_files = []
    for metric, stage_values in aggregated_data.items():
        if not any(stage_values.values()):
            continue

        avg_values = [np.mean(stage_values[stage]) if stage_values[stage] else 0 for stage in stage_names]

        plt.figure(figsize=(12, 7))
        y_label_text = f'Average {metric}'
        plt.plot(stage_names, avg_values, marker='o', linestyle='-', label=y_label_text)

        if num_replications >= 10:
            std_devs = [np.std(stage_values[stage], ddof=1) if len(stage_values[stage]) > 1 else 0 for stage in
                        stage_names]
            confidence_intervals = [
                t.ppf(0.975, len(stage_values[stage]) - 1) * sd / np.sqrt(len(stage_values[stage])) if len(
                    stage_values[stage]) > 1 else 0
                for sd, stage in zip(std_devs, stage_names)
            ]
            lower_bound = np.array(avg_values) - np.array(confidence_intervals)
            upper_bound = np.array(avg_values) + np.array(confidence_intervals)
            plt.fill_between(stage_names, lower_bound, upper_bound, color='b', alpha=0.2,
                             label='95% Confidence Interval')

        x_label_text = 'Round'
        plt.title(f'{process_type}: {y_label_text} per {x_label_text}')
        plt.xlabel(x_label_text, fontweight='bold')
        plt.ylabel(y_label_text)
        plt.xticks(rotation=0, ha='center')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()

        clean_y_label = y_label_text.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        clean_x_label = x_label_text.replace(' ', '_').replace('/', '_')
        plot_filename = os.path.join(output_dir, f'{process_type}_{clean_y_label}_{clean_x_label}.png')

        plt.savefig(plot_filename)
        plt.close()
        plot_files.append(plot_filename)

    if plot_files and results_callback:
        results_callback(f"Figures saved in: {os.path.abspath(output_dir)}")


def plot_pass(data, output_dir, results_callback=None):
    """
    Interpolates, aggregates, and plots data for the PASS procedure.
    PASS results are plotted against 'Cumulative Sample Size' on the x-axis, requiring
    interpolation to create a continuous view across all replications.
    """
    num_replications = len(data)
    if num_replications == 0:
        if results_callback:
            results_callback("No replication data found to plot.")
        return

    metrics_to_interpolate = {
        'Current Max Mean': 'previous',
        'Standard Mu': 'previous',
        'Number of Surviving Alternatives': 'previous',
        'Current Minimum Sample Size': 'previous',
        'Wall-Clock Time (s)': 'linear'
    }
    metrics_to_plot = list(metrics_to_interpolate.keys())

    all_budgets = set()
    for rep_id, stages in data.items():
        interceptor_stages = [s for s in stages if list(s.keys())[0] != "Data Collection Phase"]
        for stage_data in interceptor_stages:
            record = list(stage_data.values())[0]
            if 'Cumulative Sample Size' in record:
                all_budgets.add(int(record['Cumulative Sample Size']))

    if not all_budgets:
        if results_callback:
            results_callback("No 'Cumulative Sample Size' found in PASS data for plotting.")
        return

    unified_budget_axis = sorted(list(all_budgets))
    interpolated_data = {metric: [] for metric in metrics_to_plot}

    for rep_id in range(1, num_replications + 1):
        stages = data.get(rep_id, [])
        interceptor_stages = [s for s in stages if list(s.keys())[0] != "Data Collection Phase"]
        if not interceptor_stages:
            continue

        temp_stages = []
        for stage_data in interceptor_stages:
            record = list(stage_data.values())[0]
            if 'Cumulative Sample Size' in record:
                budget = int(record['Cumulative Sample Size'])
                temp_stages.append({'budget': budget, 'record': record})
        if not temp_stages:
            continue

        sorted_interceptor_stages = sorted(temp_stages, key=lambda x: x['budget'])
        rep_budgets = [s['budget'] for s in sorted_interceptor_stages]

        for metric, method in metrics_to_interpolate.items():
            metric_values_raw = [s['record'].get(metric, np.nan) for s in sorted_interceptor_stages]
            clean_budgets = [b for i, b in enumerate(rep_budgets) if not np.isnan(metric_values_raw[i])]
            clean_values = [v for v in metric_values_raw if not np.isnan(v)]
            if not clean_budgets:
                continue

            rep_max_budget = clean_budgets[-1]
            last_value = clean_values[-1]
            final_rep_values = {}

            if method == 'previous':
                value_idx = 0
                for budget_point in unified_budget_axis:
                    if budget_point > rep_max_budget:
                        final_rep_values[budget_point] = last_value
                        continue
                    while value_idx + 1 < len(clean_budgets) and clean_budgets[value_idx + 1] <= budget_point:
                        value_idx += 1
                    final_rep_values[budget_point] = clean_values[value_idx]
            elif method == 'linear':
                interpolated_vals = np.interp(unified_budget_axis, clean_budgets, clean_values, right=last_value)
                final_rep_values = dict(zip(unified_budget_axis, interpolated_vals))

            interpolated_data[metric].append(final_rep_values)

    plot_files = []
    for metric in metrics_to_plot:
        if not interpolated_data.get(metric):
            continue

        metric_agg = {budget: [] for budget in unified_budget_axis}
        for rep_dict in interpolated_data[metric]:
            for budget, value in rep_dict.items():
                metric_agg[budget].append(value)

        plot_budgets = sorted([b for b, v in metric_agg.items() if v])
        if not plot_budgets:
            continue

        avg_values = [np.mean(metric_agg[b]) for b in plot_budgets]
        y_label_text = f'Average {metric}'

        plt.figure(figsize=(12, 7))
        plt.plot(plot_budgets, avg_values, label=y_label_text)

        if num_replications >= 10:
            valid_reps_per_budget = [len(metric_agg[b]) for b in plot_budgets]
            std_devs = [np.std(metric_agg[b], ddof=1) if len(metric_agg[b]) > 1 else 0 for b in plot_budgets]
            confidence_intervals = [
                t.ppf(0.975, n_reps - 1) * sd / np.sqrt(n_reps) if n_reps > 1 else 0
                for n_reps, sd in zip(valid_reps_per_budget, std_devs)
            ]
            lower_bound = np.array(avg_values) - np.array(confidence_intervals)
            upper_bound = np.array(avg_values) + np.array(confidence_intervals)
            plt.fill_between(plot_budgets, lower_bound, upper_bound, color='b', alpha=0.2,
                             label='95% Confidence Interval')

        x_label_text = 'Cumulative Sample Size'
        plt.title(f'PASS: {y_label_text} by {x_label_text}')
        plt.xlabel(x_label_text, fontweight='bold')
        plt.ylabel(y_label_text)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()

        clean_y_label = y_label_text.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        clean_x_label = x_label_text.replace(' ', '_').replace('/', '_')
        plot_filename = os.path.join(output_dir, f'PASS_{clean_y_label}_{clean_x_label}.png')
        plt.savefig(plot_filename)
        plt.close()
        plot_files.append(plot_filename)

    if plot_files and results_callback:
        results_callback(f"Figures saved in: {os.path.abspath(output_dir)}")


def plot_process_results(file_path: str, process_type: str, results_callback=None):
    """
    Main dispatcher for plotting. It parses the results file and calls the
    appropriate plotting function based on the procedure type.
    """
    try:
        if not os.path.exists(file_path):
            if results_callback:
                results_callback(f"Plotting Skipped: Could not find file for plotting:\n{file_path}")
            return

        # This correctly saves figures in the same directory as the results file.
        output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)

        all_replications_data = parse_results_file(file_path)

        if not all_replications_data:
            if results_callback:
                results_callback("Plotting Info: No replication data found in the file to plot.")
            return

        if process_type == "GSP":
            plot_gsp(all_replications_data, process_type, output_dir, results_callback)
        elif process_type in ["KT", "FBKT"]:
            plot_kt_fbkt(all_replications_data, process_type, output_dir, results_callback)
        elif process_type == "PASS":
            plot_pass(all_replications_data, output_dir, results_callback)
        else:
            if results_callback:
                results_callback(f"Plotting not supported for process type: '{process_type}'")
            return

    except Exception as e:
        if results_callback:
            results_callback(f"Plotting Error: {e}")
        else:
            messagebox.showerror("Plotting Error", f"An error occurred during plotting: {e}")


# ==============================================================================
# END: Plotting functions
# ==============================================================================


def load_python_function(filepath, expected_function):
    """
    Dynamically loads a specific function from a Python file with enhanced error handling.
    """
    try:
        module_name = f"dynamic_module_{abs(hash(filepath))}"
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None:
            raise ImportError(f"Could not create module spec from path: {filepath}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, expected_function):
            raise AttributeError(f"Module missing required function: {expected_function}")
        func = getattr(module, expected_function)
        if not callable(func):
            raise TypeError(f"Loaded {expected_function} is not a callable function")
        return func
    except Exception as e:
        messagebox.showerror(
            "Load Failure",
            f"Failed loading {expected_function} from:\n{filepath}\n\n"
            "Common issues:\n"
            "1. Incorrect file path\n"
            "2. Function name mismatch\n"
            "3. Function not properly defined\n"
            f"System message: {str(e)}"
        )
        return None


def select_file(entry_widget, file_types):
    """
    Opens a file dialog for the user to select a file and updates the
    corresponding entry widget with the selected path.
    """
    file_path = filedialog.askopenfilename(filetypes=file_types)
    if file_path:
        entry_widget.config(state="normal")
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)
        entry_widget.config(state="readonly")


def create_file_upload(parent, row, label_text, file_types):
    """
    Creates a standard file upload widget group (label, entry, browse button).
    """
    frame = ttk.Frame(parent)
    frame.grid(row=row, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=2)
    ttk.Label(frame, text=label_text).grid(row=0, column=0, padx=5, sticky=tk.W)
    entry = ttk.Entry(frame, state="readonly", width=40)
    entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
    frame.columnconfigure(1, weight=1)
    ttk.Button(frame, text="Browse",
               command=lambda: select_file(entry, file_types)).grid(row=0, column=2, padx=5)
    return entry


def get_algorithm_params(algorithm):
    """
    Returns the default parameter configuration for a given algorithm.
    """
    params_config = {
        "GSP": {
            "n0": {"type": "int", "default": 0},
            "n1": {"type": "int", "default": 50},
            "r_bar": {"type": "int", "default": 10},
            "beta": {"type": "int", "default": 100},
            "alpha1": {"type": "float", "default": 0.025},
            "alpha2": {"type": "float", "default": 0.025},
            "delta": {"type": "float", "default": 0.1},
            "Number of Processors": {"type": "int", "default": 16},
            "Repeat": {"type": "int", "default": 1}
        },
        "KT": {
            "delta": {"type": "float", "default": 0.1},
            "alpha": {"type": "float", "default": 0.05},
            "g": {"type": "int", "default": 20},
            "n0": {"type": "int", "default": 20},
            "Number of Processors": {"type": "int", "default": 16},
            "Repeat": {"type": "int", "default": 1}
        },
        "PASS": {
            "n0": {"type": "int", "default": 10},
            "Delta": {"type": "int", "default": 100},
            "c": {"type": "float", "default": 8.6},
            "Termination Sample Size": {"type": "int", "default": 2000},
            "Worker Elimination": {"type": "str", "default": "NO"},
            "Number of Processors": {"type": "int", "default": 16},
            "Repeat": {"type": "int", "default": 1}
        },
        "FBKT": {
            "N": {"type": "int", "default": 1000},
            "n0": {"type": "int", "default": 15},
            "phi": {"type": "int", "default": 3},
            "Number of Processors": {"type": "int", "default": 16},
            "Repeat": {"type": "int", "default": 1}
        },
        "Custom": {}
    }
    return params_config.get(algorithm, {})


def convert_custom_param(value_str, value_type):
    """
    Converts a string input from the custom parameter table into the specified Python type.
    This version is robust against empty inputs and trailing commas for list types.
    """
    try:
        # Handle non-list types first
        if not value_type.startswith("List["):
            if value_type == "Bool":
                return value_str.lower() in ["true", "yes", "1", "t"]
            elif value_type == "String":
                return value_str
            elif value_type == "Float":
                return float(value_str)
            elif value_type == "Integer":
                return int(value_str)
            else:
                return value_str

        # Handle all list types robustly
        # If the input string is empty or just whitespace, return an empty list.
        if not value_str.strip():
            return []

        # Split by comma, strip whitespace from each part, and filter out empty strings
        # that result from trailing or consecutive commas.
        values = [v.strip() for v in value_str.split(",")]
        non_empty_values = [v for v in values if v]

        if value_type == "List[Bool]":
            return [v.lower() in ["true", "yes", "1", "t"] for v in non_empty_values]
        elif value_type == "List[String]":
            return non_empty_values
        elif value_type == "List[Float]":
            return [float(v) for v in non_empty_values]
        elif value_type == "List[Integer]":
            return [int(v) for v in non_empty_values]
        else:
            return value_str  # Fallback for unknown list types

    except Exception as e:
        messagebox.showerror("Conversion Error",
                             f"Could not convert '{value_str}' to {value_type}: {str(e)}")
        raise


def save_results_to_file(result):
    """
    Saves the raw algorithm results to a log file for debugging purposes.
    """
    try:
        os.makedirs("results", exist_ok=True)
        filename = "results/output_gui_log.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(pprint.pformat(result))
        return filename
    except Exception as e:
        messagebox.showerror("Save Error", f"Failed to save results log: {str(e)}")
        return None


class AlgorithmGUI:
    """The main class for the PyPRS GUI application."""
    alternatives_entry: Entry
    simulation_function_entry: Entry
    custom_procedure_entry: Entry

    def __init__(self, master):
        """
        Initializes the GUI window and all its widgets.
        """
        self.master = master
        master.title("PyPRS GUI")
        master.geometry("700x700")

        main_frame = ttk.Frame(master, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Select Procedure:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.algorithm_var = tk.StringVar(value="GSP")
        self.algorithm_combobox = ttk.Combobox(
            main_frame, textvariable=self.algorithm_var,
            values=["GSP", "KT", "PASS", "FBKT", "Custom"], state="readonly"
        )
        self.algorithm_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.algorithm_combobox.bind("<<ComboboxSelected>>", self.update_interface)

        self.alternatives_entry = create_file_upload(
            main_frame, row=1, label_text="Alternatives Information File (.txt):",
            file_types=[("Text files", "*.txt")]
        )
        self.simulation_function_entry = create_file_upload(
            main_frame, row=2, label_text="Simulation Function File (.py):",
            file_types=[("Python files", "*.py")]
        )

        self.custom_procedure_frame = ttk.Frame(main_frame)
        self.custom_procedure_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=2)
        self.custom_procedure_label = ttk.Label(self.custom_procedure_frame, text="Procedure File (.py):")
        self.custom_procedure_label.grid(row=0, column=0, padx=5, sticky=tk.W)
        self.custom_procedure_entry = ttk.Entry(self.custom_procedure_frame, state="readonly")
        self.custom_procedure_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        self.custom_procedure_frame.columnconfigure(1, weight=1)
        ttk.Button(self.custom_procedure_frame, text="Browse",
                   command=lambda: select_file(self.custom_procedure_entry, [("Python files", "*.py")])).grid(row=0,
                                                                                                              column=2,
                                                                                                              padx=5)
        self.note_label = ttk.Label(self.custom_procedure_frame,
                                    text="Note: The python file must contain a function named 'custom_procedure'",
                                    font=("Arial", 8))
        self.note_label.grid(row=1, column=0, columnspan=3, padx=5, pady=2, sticky=tk.W)
        self.custom_procedure_frame.grid_remove()
        self.note_label.grid_remove()

        self.crn_frame = ttk.Frame(main_frame)
        self.crn_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)
        self.crn_frame.columnconfigure(1, weight=1)
        ttk.Label(self.crn_frame, text="Enable CRNs:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.crns_var = tk.StringVar(value="NO")
        self.crns_combobox = ttk.Combobox(
            self.crn_frame, textvariable=self.crns_var,
            values=["NO", "YES"], state="readonly"
        )
        self.crns_combobox.grid(row=0, column=1, padx=5, sticky=tk.EW)

        self.param_frame = ttk.LabelFrame(main_frame, text="Input Parameters")
        self.param_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky=tk.NSEW)

        self.ref_seed_frame = ttk.Frame(self.param_frame)
        self.ref_seed_frame.grid(row=100, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W)
        self.ref_seed_frame.columnconfigure(1, weight=1)
        ttk.Label(self.ref_seed_frame, text="Set Reference Seed:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.ref_seed_var = tk.StringVar(value="NO")
        self.ref_seed_combobox = ttk.Combobox(
            self.ref_seed_frame, textvariable=self.ref_seed_var,
            values=["NO", "YES"], state="readonly"
        )
        self.ref_seed_combobox.grid(row=0, column=1, padx=5, sticky=tk.EW)
        self.ref_seed_combobox.bind("<<ComboboxSelected>>", self.toggle_ref_seed_input)

        self.ref_seed_input_frame = ttk.Frame(self.param_frame)
        self.ref_seed_input_frame.grid(row=101, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W)
        self.ref_seed_input_frame.grid_remove()
        ttk.Label(self.ref_seed_input_frame, text="List[int]:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.seed_entry1 = ttk.Entry(self.ref_seed_input_frame, width=10)
        self.seed_entry1.grid(row=0, column=1, padx=5, sticky=tk.W)
        self.seed_entry1.insert(0, "0")
        self.seed_entry2 = ttk.Entry(self.ref_seed_input_frame, width=10)
        self.seed_entry2.grid(row=0, column=2, padx=5, sticky=tk.W)
        self.seed_entry2.insert(0, "0")
        self.seed_entry3 = ttk.Entry(self.ref_seed_input_frame, width=10)
        self.seed_entry3.grid(row=0, column=3, padx=5, sticky=tk.W)
        self.seed_entry3.insert(0, "0")

        self.manual_param_frame = ttk.Frame(self.param_frame)
        self.manual_param_frame.grid(row=102, column=0, columnspan=4, padx=5, pady=5, sticky=tk.NSEW)
        self.manual_param_frame.columnconfigure(0, weight=1)
        self.manual_param_frame.grid_remove()
        self.table_frame = ttk.Frame(self.manual_param_frame)
        self.table_frame.grid(row=0, column=0, sticky=tk.NSEW)
        self.table_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.table_frame, height=200)
        self.scrollbar = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.scrollable_frame.columnconfigure(0, weight=1)
        self.scrollable_frame.columnconfigure(1, weight=1)
        self.scrollable_frame.columnconfigure(2, weight=1)
        ttk.Label(self.scrollable_frame, text="Parameter Name", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=5,
                                                                                                pady=5, sticky=tk.W)
        ttk.Label(self.scrollable_frame, text="Parameter Value", font=("Arial", 9, "bold")).grid(row=0, column=1,
                                                                                                 padx=5, pady=5,
                                                                                                 sticky=tk.W)
        ttk.Label(self.scrollable_frame, text="Parameter Type", font=("Arial", 9, "bold")).grid(row=0, column=2, padx=5,
                                                                                                pady=5, sticky=tk.W)
        ttk.Separator(self.scrollable_frame, orient="horizontal").grid(row=1, column=0, columnspan=4, sticky=tk.EW,
                                                                       pady=2)
        self.param_rows = []
        self.add_param_btn = ttk.Button(self.manual_param_frame, text="+ Add Parameter", command=self.add_param_row)
        self.add_param_btn.grid(row=1, column=0, pady=5)

        self.progress_frame = ttk.LabelFrame(main_frame, text="Progress")
        self.progress_frame.grid(row=6, column=0, columnspan=3, padx=5, pady=(10, 5), sticky=tk.EW)

        self.progress_text_var = tk.StringVar(value="Not started")
        self.progress_text_label = ttk.Label(self.progress_frame, textvariable=self.progress_text_var,
                                             font=("Courier", 10))
        self.progress_text_label.grid(row=0, column=0, padx=(10, 5), pady=5, sticky=tk.W)

        self.progressbar = ttk.Progressbar(self.progress_frame, orient='horizontal', mode='determinate', length=300)
        self.progressbar.grid(row=0, column=1, padx=(5, 10), pady=5, sticky=tk.EW)

        self.progress_frame.columnconfigure(1, weight=1)

        self.results_frame = ttk.LabelFrame(main_frame, text="Results")
        self.results_frame.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky=tk.EW)
        self.results_text = tk.Text(self.results_frame, height=5, wrap=tk.WORD, state="disabled", font=("Courier", 11))
        self.results_text.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        results_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_text.yview)
        results_scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text.config(yscrollcommand=results_scrollbar.set)
        self.results_frame.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(main_frame, text="Run Procedure", command=self.run_algorithm)
        self.run_button.grid(row=8, column=0, columnspan=3, pady=10, ipady=5, sticky=tk.EW)

        self.update_interface()
        main_frame.columnconfigure(1, weight=1)

    def add_param_row(self, name="", value="", value_type="String", fixed=False):
        """
        Adds a new row of widgets to the dynamic parameter table for the "Custom" procedure.
        """
        row_index = 2 + len(self.param_rows)

        name_var = tk.StringVar(value=name)
        name_entry = ttk.Entry(self.scrollable_frame, textvariable=name_var)
        name_entry.grid(row=row_index, column=0, padx=5, pady=2, sticky="ew")
        if fixed:
            name_entry.config(state="readonly")

        value_var = tk.StringVar(value=value)
        value_entry = ttk.Entry(self.scrollable_frame, textvariable=value_var)
        value_entry.grid(row=row_index, column=1, padx=5, pady=2, sticky="ew")

        type_options = ["Bool", "String", "Float", "Integer", "List[Bool]", "List[String]", "List[Float]",
                        "List[Integer]"]
        type_var = tk.StringVar(value=value_type)
        type_combobox = ttk.Combobox(self.scrollable_frame, textvariable=type_var, values=type_options,
                                     state="readonly")
        type_combobox.grid(row=row_index, column=2, padx=5, pady=2, sticky="ew")
        if fixed:
            type_combobox.config(state="disabled")

        del_btn = None
        if not fixed:
            del_btn = ttk.Button(self.scrollable_frame, text="×", width=2)
            del_btn.grid(row=row_index, column=3, padx=(5, 2), pady=2)

        row_widgets = (name_entry, value_entry, type_combobox, del_btn)
        if not fixed:
            del_btn.config(command=lambda w=row_widgets: self.remove_param_row(w))

        self.param_rows.append((row_widgets, name_var, value_var, type_var, fixed))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def remove_param_row(self, widgets_to_remove):
        """
        Removes a specific row of widgets from the custom parameter table.
        """
        found_index = -1
        for i, (widgets, _, _, _, _) in enumerate(self.param_rows):
            if widgets == widgets_to_remove:
                found_index = i
                break

        if found_index != -1:
            for widget in widgets_to_remove:
                if widget:
                    widget.destroy()
            self.param_rows.pop(found_index)
            # Re-grid the subsequent rows to fill the gap.
            for i in range(found_index, len(self.param_rows)):
                current_widgets, _, _, _, _ = self.param_rows[i]
                new_row_index = 2 + i
                for col_index, widget in enumerate(current_widgets):
                    if widget:
                        widget.grid(row=new_row_index, column=col_index, padx=5, pady=2, sticky="ew")

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def toggle_ref_seed_input(self, event=None):
        """Shows or hides the reference replication_seed input fields based on the user's selection."""
        if self.ref_seed_var.get() == "YES":
            self.ref_seed_input_frame.grid()
        else:
            self.ref_seed_input_frame.grid_remove()

    def reset_common_fields(self):
        """Resets all common input fields to their default state."""
        self.alternatives_entry.config(state='normal')
        self.alternatives_entry.delete(0, tk.END)
        self.alternatives_entry.config(state='readonly')
        self.simulation_function_entry.config(state='normal')
        self.simulation_function_entry.delete(0, tk.END)
        self.simulation_function_entry.config(state='readonly')
        self.custom_procedure_entry.config(state='normal')
        self.custom_procedure_entry.delete(0, tk.END)
        self.custom_procedure_entry.config(state='readonly')
        self.crns_var.set("NO")
        self.ref_seed_var.set("NO")
        self.ref_seed_input_frame.grid_remove()
        self.manual_param_frame.grid_remove()

        for widgets, _, _, _, _ in self.param_rows[:]:
            for widget in widgets:
                if widget:
                    widget.destroy()
        self.param_rows.clear()

    def update_interface(self, event=None):
        """
        Updates the entire GUI to reflect the currently selected algorithm.
        """
        algorithm = self.algorithm_var.get()
        self.reset_common_fields()

        self.progress_text_var.set("Not started")
        self.progressbar['value'] = 0
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.config(state="disabled")

        if algorithm == "Custom":
            self.custom_procedure_frame.grid()
            self.note_label.grid()
        else:
            self.custom_procedure_frame.grid_remove()
            self.note_label.grid_remove()

        if algorithm in ["PASS", "Custom"]:
            self.crn_frame.grid_remove()
        else:
            self.crn_frame.grid()

        for widget in self.param_frame.winfo_children():
            if widget not in [self.ref_seed_frame, self.ref_seed_input_frame, self.manual_param_frame]:
                widget.destroy()

        self.param_frame.columnconfigure(1, weight=1, uniform="group1")
        self.param_frame.columnconfigure(3, weight=1, uniform="group1")

        params = get_algorithm_params(algorithm)
        for idx, (param, config) in enumerate(params.items()):
            row = idx // 2
            col = (idx % 2) * 2
            label_text = f"{param}:" if config["type"] == "str" else f"{param}: ({config['type']})"
            ttk.Label(self.param_frame, text=label_text).grid(row=row, column=col, padx=5, pady=2, sticky=tk.E)
            if config["type"] == "str":
                var = tk.StringVar(value=config["default"])
                combobox = ttk.Combobox(self.param_frame, textvariable=var, values=["NO", "YES"], state="readonly")
                combobox.grid(row=row, column=col + 1, padx=5, pady=2, sticky=tk.EW)
                setattr(self, f"param_{param}", var)
            else:
                entry = ttk.Entry(self.param_frame)
                entry.insert(0, str(config["default"]))
                entry.grid(row=row, column=col + 1, padx=5, pady=2, sticky=tk.EW)
                setattr(self, f"param_{param}", entry)

        param_rows = (len(params) + 1) // 2
        ref_seed_row = param_rows

        if algorithm == "Custom":
            self.ref_seed_frame.grid_remove()
            self.ref_seed_input_frame.grid_remove()
            self.add_param_row(name="Repeat", value="1", value_type="Integer", fixed=True)
            self.add_param_row(name="Number of Processors", value="1", value_type="Integer", fixed=True)
            self.manual_param_frame.grid(row=ref_seed_row, column=0, columnspan=4, padx=5, pady=5, sticky=tk.NSEW)
        else:
            if algorithm != "PASS":
                self.ref_seed_frame.grid(row=ref_seed_row, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W)
                self.ref_seed_input_frame.grid(row=ref_seed_row + 1, column=0, columnspan=4, padx=5, pady=5,
                                               sticky=tk.W)
                self.toggle_ref_seed_input()
            else:
                self.ref_seed_frame.grid_remove()
                self.ref_seed_input_frame.grid_remove()
            self.manual_param_frame.grid_remove()

    def validate_parameters(self, params, algorithm):
        """
        Validates the user-provided parameters before running the algorithm.
        """
        if algorithm == "Custom":
            if not self.custom_procedure_entry.get():
                messagebox.showerror("Error", "Please select a custom procedure file.")
                return False
            repeat_count = sum(1 for _, name_var, _, _, _ in self.param_rows if name_var.get() == "Repeat")
            if repeat_count > 1:
                messagebox.showerror("Parameter Error", "Only one 'Repeat' parameter is allowed for custom procedures.")
                return False
            return True

        for param, config in get_algorithm_params(algorithm).items():
            value = params.get(param)
            if config["type"] == "int" and not isinstance(value, int):
                messagebox.showerror("Parameter Error", f"Parameter '{param}' must be an integer.")
                return False
            elif config["type"] == "float" and not isinstance(value, float):
                messagebox.showerror("Parameter Error", f"Parameter '{param}' must be a float.")
                return False
            elif config["type"] == "str" and value not in ["YES", "NO"]:
                messagebox.showerror("Parameter Error", f"Parameter '{param}' must be either 'YES' or 'NO'.")
                return False
        return True

    def run_algorithm(self):
        """
        The main execution function, triggered by the 'Run Procedure' button.
        """

        def display_result_message(message):
            """Helper function to append messages to the results text box."""
            self.results_text.config(state="normal")
            self.results_text.insert(tk.END, message + "\n")
            self.results_text.config(state="disabled")
            self.results_text.see(tk.END)
            self.results_text.update()

        def gui_progress_update(data):
            """Callback function to update the progress bar and text."""
            self.progress_text_var.set(data['text'])
            self.progressbar['value'] = data['value']
            self.progress_frame.update()

        gui_progress_update({'text': "Not started", 'value': 0})
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        display_result_message("Solving, please wait...")
        self.results_text.config(state="disabled")

        algorithm = self.algorithm_var.get()
        sim_func_path = self.simulation_function_entry.get()
        if not sim_func_path:
            messagebox.showerror("Error", "Please select a simulation function file.")
            return

        alt_param_file = self.alternatives_entry.get()
        if not alt_param_file:
            messagebox.showerror("Error", "Please select an alternatives information file.")
            return

        gui_progress_update({'text': "Preparing to run...", 'value': 0})

        use_crns = False
        if algorithm not in ["PASS", "Custom"]:
            use_crns = self.crns_var.get() == "YES"

        seed = None
        if algorithm not in ["Custom", "PASS"] and self.ref_seed_var.get() == "YES":
            try:
                seed = [int(self.seed_entry1.get()), int(self.seed_entry2.get()), int(self.seed_entry3.get())]
            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid replication_seed values: {str(e)}\nPlease enter integers only.")
                return

        if algorithm == "Custom":
            try:
                algo_path = self.custom_procedure_entry.get()
                if not algo_path:
                    messagebox.showerror("Error", "Please select a custom procedure file.")
                    return
                # Validate the custom procedure function exists before running
                if not load_python_function(algo_path, "custom_procedure"):
                    return

                params = {}
                repeat = 1
                for _, name_var, value_var, type_var, _ in self.param_rows:
                    name = name_var.get().strip()
                    value_str = value_var.get().strip()
                    value_type = type_var.get()
                    if not name:
                        continue
                    try:
                        value = convert_custom_param(value_str, value_type)
                        params[name] = value
                        if name == "Repeat":
                            repeat = value
                    except Exception:
                        # Error is shown by convert_custom_param, so just exit
                        return

                params.update({
                    "Simulation Function File": sim_func_path,
                    "Alternatives Information File": alt_param_file,
                    "Repeat": repeat,
                    "progress_callback": gui_progress_update
                })

                proc = Procedure(select_procedure=algo_path)
                if use_crns:
                    proc.set_CRNs()
                proc.run_procedure(params)

                display_result_message("\nOutput file paths are as follows:")
                base_dir = os.getcwd()
                summary_dir = os.path.join(base_dir, 'Output', 'Summary Results')
                summary_result_path = os.path.join(summary_dir, 'SummaryResults.txt')
                display_result_message(f"Summary Results: {os.path.abspath(summary_result_path)}")

                display_result_message("\nSolved")

            except Exception as e:
                messagebox.showerror("Custom Algorithm Error", f"Failed to run custom algorithm:\n{str(e)}")

            return

        # The rest of the function for built-in procedures
        params = {}
        for param, config in get_algorithm_params(algorithm).items():
            widget = getattr(self, f"param_{param}")
            value = widget.get()
            try:
                if config["type"] == "int":
                    params[param] = int(value)
                elif config["type"] == "float":
                    params[param] = float(value)
                else:
                    params[param] = value
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid value for {param}: {value}")
                return

        params.update({
            "Simulation Function File": sim_func_path,
            "Alternatives Information File": alt_param_file,
            "Reference Seed": seed,
            "progress_callback": gui_progress_update
        })

        if not self.validate_parameters(params, algorithm):
            return

        try:
            proc = Procedure(select_procedure=algorithm)
            if use_crns:
                proc.set_CRNs()
            proc.run_procedure(params)

            display_result_message("\nOutput file paths are as follows:")

            base_dir = os.getcwd()
            detailed_dir = os.path.join(base_dir, 'Output', 'Detailed Results')
            summary_dir = os.path.join(base_dir, 'Output', 'Summary Results')

            detail_result_path = os.path.join(detailed_dir, 'DetailedResults.txt')
            summary_result_path = os.path.join(summary_dir, 'SummaryResults.txt')

            display_result_message(f"Detailed Results: {os.path.abspath(detail_result_path)}")
            display_result_message(f"Summary Results: {os.path.abspath(summary_result_path)}")

            plot_process_results(detail_result_path, algorithm, results_callback=display_result_message)

            display_result_message("\nSolved")

        except Exception as e:
            gui_progress_update({'text': "An error occurred.", 'value': self.progressbar['value']})
            messagebox.showerror("Execution Error", f"Algorithm execution failed: {str(e)}")


def main():
    """Main function to launch the GUI application."""
    root = tk.Tk()
    app = AlgorithmGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()