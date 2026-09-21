from mi_utils import *
from optimization import *
from reporting import *
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
import traceback
#local import
from pfl_from_dataframe import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Main Experiment ---
print("Starting Folktables Optimal Federation Experiments...")

# Define binning strategy
SPEC_1 = {
    'AGEP': {'bins': AGE_BINS, 'labels': AGE_LABELS}
}

SPEC_2 = {
    'AGEP': {'bins': AGE_BINS,  'labels': AGE_LABELS},
    'OCCP': {'bins': OCCP_BINS, 'labels': OCCP_LABELS}
}

SPEC_3 = {
    'AGEP': {'bins': AGE_BINS,  'labels': AGE_LABELS},
    'SCHL': {'bins': SCHL_BINS, 'labels': SCHL_LABELS},
    'WKHP': {'bins': WKHP_BINS, 'labels': WKHP_LABELS},
    'OCCP': {'bins': OCCP_BINS, 'labels': OCCP_LABELS}
}

SPEC_4 = {
    'AGEP': {'bins': AGE_BINS,  'labels': AGE_LABELS},
    'SCHL': {'bins': SCHL_BINS, 'labels': SCHL_LABELS},
    'WKHP': {'bins': WKHP_BINS, 'labels': WKHP_LABELS},
    'OCCP': {'bins': OCCP_BINS, 'labels': OCCP_LABELS},
    'POBP' : {'bins': POBP_BINS, 'labels': POBP_LABELS}
}

BIN_SPEC = SPEC_4
# Define columns to use *after* binning
#binned_feature_name = 'AGEP_BINNED'
#MI_VARS_TO_USE = ['SEX', 'RAC1P', binned_feature_name]

# --- Load Data, Discretize, and Compute Local Components ---
ORIGINAL_MI_FEATURES = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P', 'label']

#ORIGINAL_MI_FEATURES = MI_VARS_TO_USE
# --- Loop to Load Data, Discretize, and Compute Local Components ---
all_dataframes = {}       # Stores the processed (binned) dataframes per state
all_mi_components = {}    # Stores the computed components per state

print(f"Starting processing for {len(STATES_TO_SIMULATE)} states...")
for state in STATES_TO_SIMULATE:
    print(f"\nProcessing State: {state}...")
    try:
        # 1. Load raw data for the state
        acs_data = DATASOURCE.get_data(states=[state], download=False) # Set download=False if already downloaded

        # 2. Extract features and labels (assuming ACSIncome task here)
        #    Adapt this part if using a different task object
        features_np, labels_np, _ = ACSIncome.df_to_numpy(acs_data)
        # Create initial DataFrame
        df_state = pd.DataFrame(features_np, columns=ACSIncome.features)
        df_state['label'] = labels_np # Add the label column

        # 3. Select only the original columns needed for MI/PFL analysis BEFORE binning
        #    This ensures we only process relevant data
        cols_to_process = [col for col in ORIGINAL_MI_FEATURES if col in df_state.columns]
        if 'label' not in cols_to_process and 'label' in df_state.columns:
             cols_to_process.append('label') # Ensure label is included
        # Check if essential columns are present (e.g., sensitive, label)
        if 'SEX' not in cols_to_process or 'label' not in cols_to_process:
             print(f"  Warning: Skipping state {state} - missing required SEX or label column.")
             continue
        if len(cols_to_process) < 2:
            print(f"  Skipping state {state} - less than 2 relevant columns found.")
            continue

        df_state_subset = df_state[list(set(cols_to_process))].copy() # Unique columns

        # 4. Apply discretization using the FINE_BIN_SPECS dictionary
        #    This function should return a DataFrame where specified columns
        #    are replaced or augmented by new columns named like 'COL_BINNED'
        df_binned = discretize_features(df_state_subset, BIN_SPEC)

        # Store the fully processed (binned) dataframe for this state
        all_dataframes[state] = df_binned

        # 5. Construct the FINAL list of variable names for MI calculation
        #    This list includes original categorical names + NEW binned names
        vars_for_mi_calc = []
        for original_var in ORIGINAL_MI_FEATURES:
            binned_var_name = f"{original_var}_BINNED"
            if original_var in BIN_SPEC:
                # If this var was specified for binning, use the binned name
                if binned_var_name in df_binned.columns:
                    vars_for_mi_calc.append(binned_var_name)
                # else: # Optional Debug: Binning might have failed for this column
                    # print(f"  Debug: Binned column {binned_var_name} not found in df_binned for {state}.")
            elif original_var in df_binned.columns:
                # If it wasn't specified for binning but still exists, use original name
                # (This covers 'SEX', 'RAC1P', 'label' etc.)
                vars_for_mi_calc.append(original_var)

        # Ensure label is definitely in the list if it exists in the binned df
        if 'label' not in vars_for_mi_calc and 'label' in df_binned.columns:
             vars_for_mi_calc.append('label')

        # Final list of variables present in the binned df to compute MI over
        vars_present_final = sorted([var for var in list(set(vars_for_mi_calc)) if var in df_binned.columns])

        print(f"  Variables used for MI calculation: {vars_present_final}")

        # 6. Compute MI components using the correctly identified final variable list
        if len(vars_present_final) >= 2:
            # Use your chosen component function (noisy or exact)
            #local_components = compute_mi_components(df_binned, vars_present_final)
            # Or, for the LDP version:
            epsilon_value = 0.05  #float('inf')              # Or float('inf') for exact contingecy tab;es
            #local_components = compute_noisy_mi_components_gaussian(df_binned, vars_present_final, epsilon=epsilon_value, delta=1e-5)
            local_components = compute_noisy_mi_components_gaussian(df_binned, vars_present_final, epsilon=epsilon_value, delta=1e-5)
            # Check if component calculation was successful
            if local_components and local_components.get('contingency_tables'):
                 all_mi_components[state] = local_components
                 print(f"  MI Components computed successfully for {state}.")
                 size = 0
                 for feature_couple in local_components['contingency_tables'].keys() :
                     size+= local_components['contingency_tables'][feature_couple].size
                 print(f"  {len(local_components['contingency_tables'])} Local contingency tables released (Total {size} queries) ({local_components['N']} samples)")
            else:
                 print(f"  Warning: MI Component calculation returned empty/invalid for {state}.")

        else:
            print(f"  Skipping MI components for {state} due to < 2 available variables after processing.")

    except Exception as e:
        print(f"ERROR processing state {state}: {e}")
        traceback.print_exc() # Print full traceback for debugging

# Check if any components were successfully computed
if not all_mi_components:
    print("\nNo MI components were computed. Exiting.")
    exit()

# --- Aggregation ---
print("\n--- Aggregating MI Components ---")
components_to_aggregate = list(all_mi_components.values())
aggregated_mi_components = aggregate_mi_components(components_to_aggregate)
print(f"Aggregation complete. Aggregated N = {aggregated_mi_components.get('N', 0)}")

# --- Global Calculation ---
print("\n--- Calculating Global Mutual Information ---")
global_mi_values = {}
aggregated_pairs = list(aggregated_mi_components.get('contingency_tables', {}).keys())
if not aggregated_pairs:
    print("No aggregated contingency tables found.")
else:
    for pair in aggregated_pairs:
        mi_value = calculate_global_mi(aggregated_mi_components, pair)
        global_mi_values[pair] = mi_value
        print(f"Global MI({pair[0]}, {pair[1]}): {mi_value if not np.isnan(mi_value) else 'NaN':.6f}")

# --- Verification ---
print("\n--- Verification: Calculating MI Directly on Concatenated Data ---")
if all_dataframes:
    valid_dfs = [df for state, df in all_dataframes.items() if state in all_mi_components]
    if not valid_dfs:
        print("No valid dataframes available for concatenation.")
        exit()

    combined_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"Total size of combined dataframe: {len(combined_df)}")

    if combined_df.empty:
        print("Combined dataframe is empty, skipping direct calculation.")
    else:
        print("\nCalculating Direct MI values:")
        direct_mi_values = {}
        vars_in_agg = set()
        for pair in aggregated_pairs:
            vars_in_agg.update(pair)
        vars_to_verify = sorted(list(vars_in_agg))

        combined_tables = {}
        for i, var1 in enumerate(vars_to_verify):
            if var1 not in combined_df.columns:
                print(f"Warning: Direct calc - var1 '{var1}' not in combined_df")
                continue
            for j in range(i + 1, len(vars_to_verify)):
                var2 = vars_to_verify[j]
                if var2 not in combined_df.columns:
                    print(f"Warning: Direct calc - var2 '{var2}' not in combined_df for pair ({var1}, {var2})")
                    continue
                pair = tuple(sorted((var1, var2)))
                try:
                    series1_direct = combined_df[var1].fillna('NaN_STR').astype(str)
                    series2_direct = combined_df[var2].fillna('NaN_STR').astype(str)
                    crosstab_direct = pd.crosstab(series1_direct, series2_direct)
                    if not crosstab_direct.empty:
                        combined_tables[pair] = crosstab_direct
                    else:
                        print(f"Skipping empty direct contingency table for {pair}")

                except Exception as e:
                    print(f"Warning: Direct crosstab failed for {pair}: {e}")

        temp_agg_components_direct = {'N': len(combined_df), 'contingency_tables': combined_tables}
        for pair in aggregated_pairs:
            if pair in temp_agg_components_direct.get('contingency_tables', {}):
                mi_value_direct = calculate_global_mi(temp_agg_components_direct, pair)
                direct_mi_values[pair] = mi_value_direct
                print(f"Direct MI({pair[0]}, {pair[1]}): {mi_value_direct if not np.isnan(mi_value_direct) else 'NaN':.6f}")
            else:
                print(f"Direct MI({pair[0]}, {pair[1]}): Could not compute contingency table directly.")
                direct_mi_values[pair] = np.nan

        # --- Comparison ---
        print("\n--- Comparison Summary (MI) ---")
        print(f"{'Variable Pair':<25} | {'From Agg. Components':<25} | {'Direct Calculation':<25} | {'Error':<25}")
        print("-" * 100)
        all_pairs_compared = sorted(list(set(global_mi_values.keys()) | set(direct_mi_values.keys())))
        errors = []
        for pair in all_pairs_compared:
            pair_str = f"{pair[0]}-{pair[1]}"
            agg_val = global_mi_values.get(pair, np.nan)
            dir_val = direct_mi_values.get(pair, np.nan)
            agg_str_fmt = f"{agg_val:<25.6f}" if not np.isnan(agg_val) else f"{'nan':<25}"
            dir_str_fmt = f"{dir_val:<25.6f}" if not np.isnan(dir_val) else f"{'nan':<25}"
            error_fmt = f"{agg_val - dir_val}"
            errors.append(agg_val - dir_val)
            print(f"{pair_str:<25} | {agg_str_fmt} | {dir_str_fmt} | {error_fmt}")
        print(f'Average error : {np.mean(errors)} StdDev {np.std(errors)}')
        plt.hist(errors, bins=50)
        plt.show()

else:
    print("No binned dataframes were available, skipping direct verification.")

# --- Main Experiment Execution ---
print("Starting Folktables MI Experiment with Greedy Selection and Clustering...")

# --- Define Optimization Parameters ---
print("\n--- Setting up Optimization ---")

NON_SENSITIVE_VARIABLE_NAMES = [
    BINNED_AGEP_COL if var == 'AGEP' else var for var in MI_FEATURES if var != SENSITIVE_VARIABLE_NAME
]
common_vars = set()
for comp_dict in all_mi_components.values():
    for pair in comp_dict.get('contingency_tables', {}).keys():
        common_vars.update(pair)
NON_SENSITIVE_VARIABLE_NAMES = [var for var in NON_SENSITIVE_VARIABLE_NAMES if var in common_vars]
print('non sensitve : ', NON_SENSITIVE_VARIABLE_NAMES)
print('sensitive : ', SENSITIVE_VARIABLE_NAME)
if SENSITIVE_VARIABLE_NAME not in common_vars:
    print(f"Error: Sensitive variable '{SENSITIVE_VARIABLE_NAME}' not found. Cannot proceed.")
    exit()

IDEAL_TARGETS = {
    'target_ISN': 0.0,   #Mutual info between Sensitive (S) and Non sensitive (N)
    'target_INN': 0.0    #Mutual info between two non-sensitive attributes (N-N)
}

# Optimized PFL weights (via differential evolution)
WEIGHTS = {
    'alpha_SN': 0.8889,  # Penalize Sensitive-NonSensitive correlation
    'alpha_ST': 2.0,     # Penalize Sensitive-Target correlation
    'beta_NN': 0.1111,   # Penalize NonSensitive-NonSensitive redundancy
    'delta_NT': 1.3333   # Reward NonSensitive-Target utility
}

K_MAX = 10
N_MIN = 100000

print(f"Targeting K_MAX = {K_MAX} clients from {len(all_mi_components)} available.")
print(f"Sensitive Variable: {SENSITIVE_VARIABLE_NAME}")
print(f"Non-Sensitive Variables: {NON_SENSITIVE_VARIABLE_NAMES}")
print(f"Ideal Targets: {IDEAL_TARGETS}")
print(f"Loss Weights: {WEIGHTS}")
if N_MIN:
    print(f"Minimum Total N: {N_MIN}")

# --- Clustering ---
print("\n--- Clustering Clients ---")

if all_mi_components:
    clusters = {}  # Initialize clusters to an empty dict
    try:
        clusters = cluster_clients_by_similarity(
            all_mi_components,
            SENSITIVE_VARIABLE_NAME,
            NON_SENSITIVE_VARIABLE_NAMES,
            n_clusters=3,
            distance_metric='jensenshannon',
            clustering_algorithm='agglomerative',
            linkage='average'
        )

        print("\n--- Cluster Assignments ---")
        for cluster_id, client_ids in clusters.items():
            print(f"Cluster {cluster_id}: {client_ids}")

        # Example Selection Strategy
        selected_ids_from_clusters = []
        for cluster_id, client_ids in clusters.items():
            if client_ids:
                selected_ids_from_clusters.append(client_ids[0])
        print("\nSelected clients from clusters:", selected_ids_from_clusters)

    except Exception as e:
        print(f"Clustering failed: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without clustering...")
        # If clustering fails, proceed with a default empty clustering, or handle as appropriate
        clusters = {}
else:
    print("Skipping clustering and optimization: No components available.")

# --- Greedy Selection ---
print("\n--- Hill Climbing with Simulated Annealing Selection ---")
actual_k_max = min(K_MAX, len(all_mi_components))
if actual_k_max < K_MAX:
    print(f"Warning: Requested k_max={K_MAX}, but only {len(all_mi_components)} clients available.")

solutions = []
for i in range(10) :
    selected_ids, final_agg_components, final_loss = simulated_annealing_selection(
        all_mi_components,
        IDEAL_TARGETS,
        WEIGHTS,
        SENSITIVE_VARIABLE_NAME,
        NON_SENSITIVE_VARIABLE_NAMES,
        actual_k_max,
        N_MIN
    )

    report_selection_results(selected_ids, final_agg_components, final_loss, 'Hill Climbing with Simulated Annealing')
    solutions.append(selected_ids)

my_divergence_weights = create_divergence_weights(
    sensitive_var_name=SENSITIVE_VARIABLE_NAME,
    non_sensitive_var_names=NON_SENSITIVE_VARIABLE_NAMES,
    outcome_var_name='label',
    default_fairness_weight=15.0, # You can make sensitive divergence more important
    default_utility_weight=2.0,
    include_cross_non_sensitive_weights=True,
    default_cross_non_sensitive_weight=0.5
)

optimal_states, final_agg_components, finla_loss = simulated_annealing_selection_variable_size_snapshot(
    all_mi_components,
    SENSITIVE_VARIABLE_NAME,
    NON_SENSITIVE_VARIABLE_NAMES,
    my_divergence_weights,
)

report_selection_results(selected_ids, final_agg_components, final_loss, 'Smallest representative via Simulated Annealing')



print(solutions)
exit()
print("\n--- Greedy Additive Selection ---")

selected_ids, final_agg_components, final_loss = greedy_additive_selection_parallel(
    all_mi_components,
    IDEAL_TARGETS,
    WEIGHTS,
    SENSITIVE_VARIABLE_NAME,
    NON_SENSITIVE_VARIABLE_NAMES,
    actual_k_max,
    N_MIN
)

report_selection_results(selected_ids, final_agg_components, final_loss, 'greedy_additive_selection')




selected_mi_components = {key: all_mi_components[key] for key in selected_ids}

obtained_loss = calculate_loss(selected_mi_components, IDEAL_TARGETS, WEIGHTS, SENSITIVE_VARIABLE_NAME, NON_SENSITIVE_VARIABLE_NAMES)

print(f'obtained federated has PFL : {obtained_loss}')
#print(f'obtained federated has PFL : {pfl_of_federation(selected_ids)}')


# --- Subtractive Greedy ---
run_subtractive = True
if run_subtractive:
    LOSS_THRESHOLD = 0.005
    N_MIN_SUBTRACTIVE = 150000
    print("\n" + "=" * 30 + " SUBTRACTIVE GREEDY SELECTION " + "=" * 30)
    print(f"Parameters: loss_threshold<={LOSS_THRESHOLD}, n_min>={N_MIN_SUBTRACTIVE}")

    if not all_mi_components:
        print("Skipping Subtractive Greedy: No components available.")
    else:
        selected_ids_sub, final_agg_components_sub, final_loss_sub = subtractive_greedy_selection(
            all_mi_components,
            IDEAL_TARGETS,
            WEIGHTS,
            SENSITIVE_VARIABLE_NAME,
            NON_SENSITIVE_VARIABLE_NAMES,
            LOSS_THRESHOLD,
            N_MIN_SUBTRACTIVE
        )

    report_selection_results(selected_ids_sub, final_agg_components_sub, final_loss_sub,
                             'subtractive_greedy_selection')

print("\nExperiment Finished.")
