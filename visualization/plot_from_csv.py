import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import traceback
import csv

def load_dataset_files(file_paths, include_group=False):
    """
    Loads a list of dataset files and returns a list of lists.

    Args:
        file_paths (list of str): Paths to CSV or data files.
        include_group (bool): Whether to expect a 'group' column.

    Returns:
        data_list (list of lists): Each inner list corresponds to one file's data as a list of lists.
    """
    data_list = []

    for path in file_paths:
        df = pd.read_csv(path)

        if include_group:
            if 'group' not in df.columns:
                raise ValueError(f"Expected 'group' column in {path}, but it was not found.")


        # Convert DataFrame to list of lists and append
        data_list.append(df.values.tolist())

    return data_list


measure_mapping = {
    'loss': 0,
    'accuracy': 1,
    'precision': 2,
    'recall': 3,
    'eod': 4,
    'spd': 5,
    'mad': 6,
    'f1': -1
}



def plot_measure_evolution(scores1, scores2, label1, label2, measure: str, title=None):
    """
    Plots the evolution of a specific measure (mean across runs) with
    mean +/- standard deviation shading.
    Handles 'f1' by calculating it from precision and recall.

    Args:
        scores1 (list): List of runs for group 1. Each run is a list of lists,
                        where inner lists contain [loss, accuracy, precision, recall, eod, spd]
                        at each iteration. Shape: (num_runs1, num_iterations, num_metrics).
        scores2 (list): List of runs for group 2. Shape: (num_runs2, num_iterations, num_metrics).
        label1 (str): Label for group 1 plot.
        label2 (str): Label for group 2 plot.
        measure (str): Which measure to plot ('loss', 'accuracy', 'precision', 'recall', 'eod', 'spd', 'f1').
        title (str, optional): Title for the plot. Defaults to None.
    """

    # --- Input Validation ---

    # Add balanced_accuracy if it's in your data, e.g.:
    # 'balanced_accuracy': 1, # If replacing standard accuracy

    if measure not in measure_mapping:
        raise ValueError(f"Invalid measure. Expected one of {list(measure_mapping.keys())}")
    measure_index = measure_mapping[measure]

    if not scores1 or not isinstance(scores1, list) or not isinstance(scores1[0], list) or not scores1[0]:
        print("Warning: scores1 is empty or not in the expected format (list of runs with iterations). Skipping plot.")
        return
    if not scores2 or not isinstance(scores2, list) or not isinstance(scores2[0], list) or not scores2[0]:
        print("Warning: scores2 is empty or not in the expected format (list of runs with iterations). Skipping plot.")
        return

    # --- Data Processing ---
    try:
        # Add checks for consistent iteration lengths within each group
        num_iterations1 = len(scores1[0])
        if not all(len(run) == num_iterations1 for run in scores1):
            raise ValueError(
                f"Runs in scores1 have inconsistent lengths (number of iterations). Expected {num_iterations1}.")
        scores1_np = np.array(scores1)  # Shape: (num_runs1, num_iterations, num_metrics)

        num_iterations2 = len(scores2[0])
        if not all(len(run) == num_iterations2 for run in scores2):
            raise ValueError(
                f"Runs in scores2 have inconsistent lengths (number of iterations). Expected {num_iterations2}.")
        scores2_np = np.array(scores2)  # Shape: (num_runs2, num_iterations, num_metrics)

        # Check if number of iterations match between groups
        if num_iterations1 != num_iterations2:
            # Optional: Handle by truncating to minimum length? Or raise error.
            min_iterations = min(num_iterations1, num_iterations2)
            print(f"Warning: Iteration lengths differ ({num_iterations1} vs {num_iterations2}). Truncating plots to {min_iterations} iterations.")
            scores1_np = scores1_np[:, :min_iterations, :]
            scores2_np = scores2_np[:, :min_iterations, :]
            num_iterations = min_iterations
            # Or raise error:
            # raise ValueError(
            #     f"Number of iterations differs between scores1 ({num_iterations1}) and scores2 ({num_iterations2}). Cannot plot together.")
        else:
             num_iterations = num_iterations1


        if measure == 'f1':
            # Calculate F1 score from precision and recall
            precision_data1 = scores1_np[..., measure_mapping['precision']]
            recall_data1 = scores1_np[..., measure_mapping['recall']]
            # Add small epsilon for numerical stability
            f1_data1 = 2 * (precision_data1 * recall_data1) / np.maximum(precision_data1 + recall_data1, 1e-8)

            precision_data2 = scores2_np[..., measure_mapping['precision']]
            recall_data2 = scores2_np[..., measure_mapping['recall']]
            f1_data2 = 2 * (precision_data2 * recall_data2) / np.maximum(precision_data2 + recall_data2, 1e-8)

            # Calculate mean and std dev for F1
            mean1 = np.mean(f1_data1, axis=0)
            std1 = np.std(f1_data1, axis=0) # CHANGED from min

            mean2 = np.mean(f1_data2, axis=0)
            std2 = np.std(f1_data2, axis=0) # CHANGED from min

            # max1 and max2 are no longer needed for std dev plot

        else:
            # Extract the specific measure for each group
            measure_data1 = scores1_np[..., measure_index]  # Shape: (num_runs1, num_iterations)
            measure_data2 = scores2_np[..., measure_index]  # Shape: (num_runs2, num_iterations)

            # Calculate mean and std dev across runs (axis=0)
            mean1 = np.mean(measure_data1, axis=0)
            std1 = np.std(measure_data1, axis=0) # CHANGED from min

            mean2 = np.mean(measure_data2, axis=0)
            std2 = np.std(measure_data2, axis=0) # CHANGED from min

            # max1 and max2 are no longer needed for std dev plot

    except Exception as e:
        print(f"Error processing scores data: {e}")
        print("Please ensure scores1 and scores2 are lists of runs, where each run is a list of metric lists per iteration.")
        traceback.print_exc()
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))
    iterations = np.arange(num_iterations)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Group 1
    ax.plot(iterations, mean1, label=label1, color=colors[0], linewidth=2)
    # CHANGED fill_between bounds to mean +/- std dev
    ax.fill_between(iterations, mean1 - std1, mean1 + std1, color=colors[0], alpha=0.2,
                    label=f'{label1} (Mean ± Std Dev)') # Updated label (though maybe unused in legend)

    # Group 2
    ax.plot(iterations, mean2, label=label2, color=colors[1], linewidth=2)
    # CHANGED fill_between bounds to mean +/- std dev
    ax.fill_between(iterations, mean2 - std2, mean2 + std2, color=colors[1], alpha=0.2,
                    label=f'{label2} (Mean ± Std Dev)') # Updated label (though maybe unused in legend)


    # Customization
    ax.set_xlabel('Iterations / Rounds', fontsize=14)
    # Use measure name directly if it's well-formatted, otherwise capitalize
    ylabel = measure.upper() if measure in ['eod', 'spd'] else measure.replace('_', ' ').capitalize()
    ax.set_ylabel(ylabel, fontsize=14)

    # Improve legend - keep only labels for the mean lines
    handles, labels = ax.get_legend_handles_labels()
    # Assuming the plot lines are handles[0] and handles[2]
    # Adjust if you add/remove plots
    try:
        ax.legend([handles[0], handles[2]], [labels[0], labels[2]], fontsize=20)
    except IndexError:
        print("Warning: Could not generate legend correctly. Check plot elements.")
        ax.legend(fontsize=12) # Fallback legend

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.show()



def read_pfl_values(csv_path):
    floats = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:
                try:
                    value = float(row[1])
                    floats.append(value)
                except ValueError:
                    # Skip non-float values (e.g., headers or bad data)
                    continue
    return floats


def plot_utility_and_fairness_subplots(
    scores_list,
    labels,
    title=None,
    savepath=None,
    utility_metrics=None,
    fairness_metrics=None,
):
    """
    Plots two rows of subplots:
    - Row 1 (Utility): two user-selected utility metrics
    - Row 2 (Fairness): two user-selected fairness metrics
    
    Each subplot shows mean +/- std dev for all groups.

    Args:
        scores_list (list): List of score arrays, one per group. Each: (num_runs, num_iterations, num_metrics).
        labels (list): List of labels for each group.
        title (str, optional): Suptitle for the figure.
        savepath (str, optional): Path to save the figure. If None, shows interactively.
        utility_metrics (list[str], optional): Exactly two metrics for top row.
            Default: ['loss', 'accuracy']
        fairness_metrics (list[str], optional): Exactly two metrics for bottom row.
            Default: ['spd', 'mad']
    """
    # Metrics to plot: first row (utility), second row (fairness)
    if utility_metrics is None:
        utility_metrics = ['loss', 'accuracy']
    if fairness_metrics is None:
        fairness_metrics = ['spd', 'mad']

    if len(utility_metrics) != 2:
        raise ValueError(f"utility_metrics must contain exactly 2 metrics, got {len(utility_metrics)}")
    if len(fairness_metrics) != 2:
        raise ValueError(f"fairness_metrics must contain exactly 2 metrics, got {len(fairness_metrics)}")

    for metric in utility_metrics + fairness_metrics:
        if metric not in measure_mapping:
            raise ValueError(f"Invalid metric '{metric}'. Expected one of {list(measure_mapping.keys())}")
    
    # Input validation
    if not scores_list or not all(scores_list):
        print("Warning: Empty scores. Skipping plot.")
        return
    
    try:
        # Convert all score arrays to numpy
        scores_np_list = [np.array(scores, dtype=float) for scores in scores_list]
        
        # Handle iteration length mismatch - find minimum
        num_iterations_list = [s.shape[1] for s in scores_np_list]
        if len(set(num_iterations_list)) > 1:
            min_iterations = min(num_iterations_list)
            print(f"Warning: Truncating to {min_iterations} iterations.")
            scores_np_list = [s[:, :min_iterations, :] for s in scores_np_list]
            num_iterations = min_iterations
        else:
            num_iterations = num_iterations_list[0]
            
    except Exception as e:
        print(f"Error processing scores: {e}")
        traceback.print_exc()
        return
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    iterations = np.arange(num_iterations)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    def compute_metric_stats(scores_np, measure):
        """Compute mean and std for a given measure."""
        if measure == 'f1':
            precision = scores_np[..., measure_mapping['precision']]
            recall = scores_np[..., measure_mapping['recall']]
            data = 2 * (precision * recall) / np.maximum(precision + recall, 1e-8)
        else:
            data = scores_np[..., measure_mapping[measure]]
        return np.mean(data, axis=0), np.std(data, axis=0)
    
    def plot_single_metric(ax, measure, ylabel):
        """Plot a single metric on the given axes for all groups."""
        for i, (scores_np, label) in enumerate(zip(scores_np_list, labels)):
            mean, std = compute_metric_stats(scores_np, measure)
            ax.plot(iterations, mean, label=label, color=colors[i], linewidth=2, markersize=4)
            ax.fill_between(iterations, mean - std, mean + std, color=colors[i], alpha=0.2)
        
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Plot utility metrics (first row)
    ylabel_map = {
        'loss': 'CE Loss', 
        'accuracy': 'Bal. Accuracy', 
        'f1': 'F1 Score',
        'spd': 'SPD', 
        'eod': 'EOD', 
        'mad': 'MAD'
    }
    
    for i, measure in enumerate(utility_metrics):
        plot_single_metric(axes[0, i], measure, ylabel_map[measure])
    
    # Plot fairness metrics (second row)
    for i, measure in enumerate(fairness_metrics):
        plot_single_metric(axes[1, i], measure, ylabel_map[measure])
    
    # Add shared x-axis label to bottom row
    for ax in axes[1, :]:
        ax.set_xlabel('Rounds', fontsize=14)
    
    # Add legend outside figures, at the top center
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=len(labels), fontsize=11, 
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for legend
    
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {savepath}")
    else:
        plt.show()

# Modified Function
def plot_measure_evolution_with_individual_runs(
    scores_group1,        # Data for group plotted with mean/stddev (e.g., Optimal)
    scores_group2,        # Data for group plotted as individual runs (e.g., Random)
    label_group1,         # Label for group 1 (e.g., 'Optimal')
    label_group2_base,    # Base label for group 2 runs (e.g., 'Random')
    group2_pfl_scores,    # List/array of PFL scores for each run in scores_group2
    measure: str,
    title=None,
    group2_state_names=None,# Optional: Pass state names instead of PFL for labels
    colormap_group2='tab10', # Parameter to choose colormap ('viridis','plasma','tab10','tab20')
    linewidth_group1=2.5,  # Linewidth for group 1 (Optimal mean)
    linewidth_group2=1.8   # INCREASED default linewidth for group 2 (Random runs)
    ):
    """
    Plots group1 as mean +/- stddev and group2 as individual runs labeled
    by PFL score or state names. Legend INSIDE plot. Allows setting linewidths.
    Handles 'f1' calculation.

    Args:
        linewidth_group1 (float): Linewidth for the group 1 mean plot.
        linewidth_group2 (float): Linewidth for the group 2 individual run plots.
    """
    # --- Input Validation ---
    if measure not in measure_mapping: raise ValueError(f"Invalid measure '{measure}'. Expected one of {list(measure_mapping.keys())}")
    measure_index = measure_mapping[measure]
    if not scores_group1 or not isinstance(scores_group1, (list, np.ndarray)) or not scores_group1[0] or not isinstance(scores_group1[0], (list, np.ndarray)) or not scores_group1[0]: print(f"Warning: {label_group1} scores empty or invalid format. Skipping plot."); return
    if not scores_group2 or not isinstance(scores_group2, (list, np.ndarray)) or not scores_group2[0] or not isinstance(scores_group2[0], (list, np.ndarray)) or not scores_group2[0]: print(f"Warning: {label_group2_base} scores empty or invalid format. Skipping plot."); return
    if group2_pfl_scores is None and group2_state_names is None: print(f"Warning: Must provide PFL scores or state names for group 2 labels. Skipping plot."); return
    if group2_pfl_scores is not None and len(scores_group2) != len(group2_pfl_scores): print(f"Warning: Length mismatch: scores_group2 ({len(scores_group2)}) vs PFLs ({len(group2_pfl_scores)}). Skipping plot."); return
    if group2_state_names is not None and len(scores_group2) != len(group2_state_names): print(f"Warning: Length mismatch: scores_group2 ({len(scores_group2)}) vs State names ({len(group2_state_names)}). Skipping plot."); return

    # --- Data Processing ---
    try:
        scores1_np = np.array(scores_group1, dtype=float)
        scores2_np = np.array(scores_group2, dtype=float)
        # Consistency checks and truncation
        num_iterations1 = scores1_np.shape[1]; num_iterations2 = scores2_np.shape[1]
        if scores1_np.ndim != 3: raise ValueError(f"scores_group1 has wrong dimensions: {scores1_np.ndim}")
        if scores2_np.ndim != 3: raise ValueError(f"scores_group2 has wrong dimensions: {scores2_np.ndim}")
        if num_iterations1 != num_iterations2:
            min_iterations = min(num_iterations1, num_iterations2)
            print(f"Warning: Iteration lengths differ. Truncating plots to {min_iterations} iterations.")
            scores1_np = scores1_np[:, :min_iterations, :]; scores2_np = scores2_np[:, :min_iterations, :]
            num_iterations = min_iterations
        else: num_iterations = num_iterations1

        # Calculate data for Group 1 (Mean/StdDev)
        if measure == 'f1':
             prec1 = scores1_np[..., measure_mapping['precision']]; rec1 = scores1_np[..., measure_mapping['recall']]
             f1_data1 = 2 * (prec1 * rec1) / np.maximum(prec1 + rec1, 1e-9)
             mean1 = np.mean(f1_data1, axis=0); std1 = np.std(f1_data1, axis=0)
        else:
            if measure_index >= scores1_np.shape[2]: raise IndexError(f"Index {measure_index} ('{measure}') OOB for scores1 shape {scores1_np.shape}")
            measure_data1 = scores1_np[..., measure_index]; mean1 = np.mean(measure_data1, axis=0); std1 = np.std(measure_data1, axis=0)

        # Get data for Group 2 (Individual Runs)
        if measure == 'f1':
            prec2 = scores2_np[..., measure_mapping['precision']]; rec2 = scores2_np[..., measure_mapping['recall']]
            group2_runs_data = 2 * (prec2 * rec2) / np.maximum(prec2 + rec2, 1e-9)
        else:
            if measure_index >= scores2_np.shape[2]: raise IndexError(f"Index {measure_index} ('{measure}') OOB for scores2 shape {scores2_np.shape}")
            group2_runs_data = scores2_np[..., measure_index]

    except Exception as e:
        print(f"Error processing scores data: {e}"); traceback.print_exc(); return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted figsize slightly
    iterations = np.arange(num_iterations)
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color1 = base_colors[0]

    # --- Plot Group 1 (Mean +/- Std Dev) ---
    ax.plot(iterations, mean1, label=label_group1, color=color1,
            markersize=5, linewidth=linewidth_group1, zorder=10) # Use linewidth parameter
    ax.fill_between(iterations, mean1 - std1, mean1 + std1, color=color1, alpha=0.15, zorder=3)

    # --- Plot Group 2 (Individual Runs with Different Colors & Increased Linewidth) ---
    num_runs_group2 = scores2_np.shape[0]
    try: # Get colormap
        try: cmap = plt.colormaps[colormap_group2]
        except AttributeError: cmap = cm.get_cmap(colormap_group2) # Fallback
        if num_runs_group2 > cmap.N and colormap_group2 in ['tab10', 'tab20', 'Paired', 'Set1', 'Set2', 'Set3']: print(f"Warning: Runs ({num_runs_group2}) > colors ({cmap.N}). Colors repeat.")
        run_colors = cmap(np.linspace(0, 1, num_runs_group2))
    except KeyError: # Fallback colormap
        print(f"Warning: Colormap '{colormap_group2}' not found. Using default cycle.")
        run_colors = [base_colors[(i+1) % len(base_colors)] for i in range(num_runs_group2)]

    for i in range(num_runs_group2):
        run_data = group2_runs_data[i, :]
        # Create label
        run_label = f'{label_group2_base} Run {i+1}' # Default
        if group2_state_names:
             state_str = group2_state_names[i]; len_limit=20
             if isinstance(state_str, str) and len(state_str) > len_limit: state_str = state_str[:len_limit-2] + '..'
             run_label = f'{label_group2_base} ({state_str})'
        elif group2_pfl_scores is not None:
             try:
                 pfl_score = float(group2_pfl_scores[i])
                 if not np.isnan(pfl_score): run_label = f'{label_group2_base} (PFL: {pfl_score:.3f})'
             except (ValueError, TypeError, IndexError): pass

        # Plot individual run with specified linewidth (linewidth_group2)
        ax.plot(iterations, run_data, label=run_label, color=run_colors[i],
                linewidth=linewidth_group2, # *** USE linewidth_group2 HERE ***
                linestyle='-', alpha=0.7, zorder=5) # Use linewidth parameter

    # --- Customization ---
    ax.set_xlabel('Iterations / Rounds', fontsize=12)
    ylabel_map = {'eod': 'EOD', 'spd': 'SPD', 'f1':'F1 Score'}
    ylabel = ylabel_map.get(measure, measure.replace('_', ' ').capitalize())
    ax.set_ylabel(ylabel, fontsize=12)

    # --- Legend Inside Plot ---
    # Change legend positioning: remove bbox_to_anchor, use loc='best' or specific corner
    legend_fontsize = max(20, 9 - num_runs_group2 // 4)
    ax.legend(loc='best', fontsize=legend_fontsize)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.show()


task = 'ACSTravelTime' 
exp = 15
n_seeds = 3

# Load data from four directories for client selection comparison
file_paths_optimal = [f"Convergence2/{task}/fixed_size_{exp}/Optimal_FedAvg/seed_{seed}.csv" for seed in range(0, 2)]
file_paths_random_ucb = [f"Convergence2/{task}/fixed_size_{exp}/Random_FedAvg_UCB/seed_{seed}.csv" for seed in range(n_seeds)]
file_paths_random_threshold = [f"Convergence2/{task}/fixed_size_{exp}/Random_FedAvg_Threshold/seed_{seed}.csv" for seed in range(n_seeds)]
file_paths_random_fedsampling = [f"Convergence2/{task}/fixed_size_{exp}/Random_FedAvg_FedSamp/seed_{seed}.csv" for seed in range(n_seeds)]


file_paths_random_fedavg = [f"Convergence2/{task}/fixed_size_{exp}/Random_FedAvg/seed_{seed}.csv" for seed in range(n_seeds)]
file_paths_random_scaffold = [f"Convergence2/{task}/fixed_size_{exp}/Random_SCAFFOLD/seed_{seed}.csv" for seed in range(n_seeds)]
file_paths_random_fedprox = [f"Convergence2/{task}/fixed_size_{exp}/Random_FedProx/seed_{seed}.csv" for seed in range(n_seeds)]

all_data_optimal = load_dataset_files(file_paths_optimal)
all_data_random_ucb = load_dataset_files(file_paths_random_ucb)
all_data_random_threshold = load_dataset_files(file_paths_random_threshold)
all_data_random_fedsampling = load_dataset_files(file_paths_random_fedsampling)

all_data_random_fedavg = load_dataset_files(file_paths_random_fedavg)
all_data_random_fedprox = load_dataset_files(file_paths_random_fedprox)
all_data_random_scaffold = load_dataset_files(file_paths_random_scaffold)

# Plot utility and fairness subplots (2x2 grid) with 4 curves


plot_utility_and_fairness_subplots(
    [all_data_optimal, all_data_random_fedavg, all_data_random_scaffold, all_data_random_fedprox],
    ["Optimal (FedAvg)", "Random (FedAvg)", "Random (SCAFFOLD)", "Random (FedProx)"],
    title=f"{task} (k={exp}): Optimal vs Client Selection Methods",
    utility_metrics=['loss', 'accuracy'],
    fairness_metrics=['spd', 'eod']
)



plot_utility_and_fairness_subplots(
    [all_data_optimal, all_data_random_ucb, all_data_random_threshold, all_data_random_fedsampling],
    ["Optimal (FedAvg)", "Random (UCB-CS)", "Random (Threshold)", "Random (FedSampling)"],
    title=f"{task} (k={exp}): Optimal vs Client Selection Methods",
    utility_metrics=['loss', 'accuracy'],
    fairness_metrics=['spd', 'eod']
)


