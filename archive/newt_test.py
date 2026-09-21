import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from folktables import ACSDataSource, ACSIncome # Use ACSIncome task




all_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']


# --- Constants for Folktables ---
DATASOURCE = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
# Using a smaller list for quicker demonstration, use your full list if needed
STATES_TO_SIMULATE = ['CA', 'TX', 'NY', 'FL', 'IL']
SENSITIVE_COL = 'SEX' # 1: Male, 2: Female (as per ACS codes)
TARGET_COL_RAW = 'PINCP' # Total person's income - will be added back from labels_np
TARGET_COL_PROCESSED = 'label' # Our binary target
TARGET_POSITIVE_THRESHOLD = 50000 # Threshold for high income
TARGET_POSITIVE_VALUE = 1 # Value representing high income
NUMERIC_FEATURE_COLS = ['AGEP', 'WKHP', 'POVPIP'] # Example numeric features for utility avg

# --- Re-use the functions (ASSUMED TO BE DEFINED HERE) ---
# compute_local_components(df, sensitive_col, target_col, target_positive_value, feature_cols=None)
# aggregate_components(local_components_list)
# calculate_global_metrics(aggregated_components)
# --- START FUNCTION DEFINITIONS ---
def compute_local_components(df, sensitive_col, target_col, target_positive_value, feature_cols=None):
    """
    Computes the additive components needed for global metric calculation from a local dataframe.
    (Code from previous response - assuming it's defined here)
    """
    if df.empty:
         return {
            'N': 0,
            'counts_S': Counter(),
            'counts_Y1_S': Counter(),
            'sums_X': defaultdict(float)
        }

    n_total = len(df)
    # Ensure sensitive column exists and handle potential missing values if necessary
    if sensitive_col not in df.columns:
        print(f"Warning: Sensitive column '{sensitive_col}' not found in dataframe.")
        return { 'N': n_total, 'counts_S': Counter(), 'counts_Y1_S': Counter(), 'sums_X': defaultdict(float) }
    # Drop rows where sensitive attribute is NA before counting groups
    counts_s = Counter(df[sensitive_col].dropna())

    # Ensure target column exists
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found in dataframe.")
        return { 'N': n_total, 'counts_S': counts_s, 'counts_Y1_S': Counter(), 'sums_X': defaultdict(float) }

    # Count positive target outcomes within each sensitive group
    # Ensure we only process rows where sensitive attribute is not NA
    df_filtered_sens = df.dropna(subset=[sensitive_col])
    positive_mask = (df_filtered_sens[target_col] == target_positive_value)
    df_positive = df_filtered_sens[positive_mask]

    # Check if sensitive column exists in the filtered positive dataframe
    if sensitive_col in df_positive.columns:
        counts_y1_s = Counter(df_positive[sensitive_col]) # Already dropped NA above
    else:
         counts_y1_s = Counter() # Should not happen if check above passed, but safe


    # Ensure all groups from counts_s are present in counts_y1_s, even if with zero count
    for group in counts_s:
        if group not in counts_y1_s:
            counts_y1_s[group] = 0

    components = {
        'N': n_total,
        'counts_S': counts_s,
        'counts_Y1_S': counts_y1_s,
        'sums_X': defaultdict(float) # Initialize even if feature_cols is None
    }

    # Compute sums for specified numerical features
    if feature_cols:
        sums_x = defaultdict(float)
        for col in feature_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                 # Convert to numeric, coercing errors, then fill NA with 0 before summing
                sums_x[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
            # else: # Removing warning spam for default features not used
                # print(f"Warning: Feature column '{col}' not found or not numeric. Skipping sum.")
                pass
        components['sums_X'] = sums_x

    return components

def aggregate_components(local_components_list):
    """
    Aggregates (sums) the components from a list of local component dictionaries.
    (Code from previous response - assuming it's defined here)
    """
    if not local_components_list:
        return {
            'N': 0,
            'counts_S': Counter(),
            'counts_Y1_S': Counter(),
            'sums_X': defaultdict(float)
        }

    aggregated = {
        'N': 0,
        'counts_S': Counter(),
        'counts_Y1_S': Counter(),
        'sums_X': defaultdict(float)
    }

    for components in local_components_list:
        aggregated['N'] += components.get('N', 0)
        # Safely update Counters, handle if key doesn't exist in components
        if 'counts_S' in components:
            aggregated['counts_S'].update(components['counts_S'])
        if 'counts_Y1_S' in components:
            aggregated['counts_Y1_S'].update(components['counts_Y1_S'])

        # Sum feature sums
        local_sums_x = components.get('sums_X', defaultdict(float))
        for feature, local_sum in local_sums_x.items():
             aggregated['sums_X'][feature] += local_sum

    return aggregated

def calculate_global_metrics(aggregated_components):
    """
    Calculates global metrics (proportions, rates, averages) from aggregated components.
    (Code from previous response - slightly adjusted group names for clarity)
    """
    metrics = {}
    N_total = aggregated_components.get('N', 0)
    counts_S = aggregated_components.get('counts_S', Counter())
    counts_Y1_S = aggregated_components.get('counts_Y1_S', Counter())
    sums_X = aggregated_components.get('sums_X', defaultdict(float))

    metrics['N_total'] = N_total

    # Calculate Group Proportions
    group_proportions = {}
    if N_total > 0:
        for group, count in counts_S.items():
             # Calculate proportion based on sum of group counts, not N_total,
             # if we want proportions *within the population that has a non-NA sensitive attribute*
             # Sticking to N_total for overall proportion as initially designed.
            group_proportions[group] = count / N_total
    metrics['group_proportions'] = group_proportions

    # Calculate Outcome Rates per Group
    outcome_rates = {}
    for group, count_s in counts_S.items():
        if count_s > 0:
            outcome_rates[group] = counts_Y1_S.get(group, 0) / count_s
        else:
            outcome_rates[group] = None # Or 0, or NaN, depending on desired handling
    metrics['outcome_rates_per_group'] = outcome_rates

    # Calculate Average Feature Values
    avg_features = {}
    if N_total > 0:
       for feature, total_sum in sums_X.items():
           # Avoid division by zero if N_total somehow became 0 after check
           if N_total > 0:
                avg_features[feature] = total_sum / N_total
           else:
                avg_features[feature] = None

    metrics['average_features'] = avg_features


    # --- Calculate specific fairness metrics (example for SEX=1 vs SEX=2) ---
    # Use actual group values (1 and 2 for SEX)
    groupA_val = 1 # e.g., Male
    groupB_val = 2 # e.g., Female

    rateA = outcome_rates.get(groupA_val)
    rateB = outcome_rates.get(groupB_val)

    metrics['spd'] = None
    metrics['di'] = None

    # Check if rates are valid numbers before calculating SPD/DI
    if rateA is not None and rateB is not None:
         metrics['spd'] = rateA - rateB
         if rateB > 1e-7 : # Avoid division by zero or near-zero
             metrics['di'] = rateA / rateB
         else:
             metrics['di'] = float('inf') if rateA > 1e-7 else None # Handle DI edge case
    # Add case where one rate might be None but the other exists
    elif rateA is not None: # Only A exists
        metrics['spd'] = rateA
        metrics['di'] = float('inf')
    elif rateB is not None: # Only B exists
        metrics['spd'] = -rateB
        metrics['di'] = 0.0


    return metrics
# --- END FUNCTION DEFINITIONS ---


print("Starting Folktables Experiment (Corrected v2)...")

# 3. Load Data and Compute Local Components for Each State
all_dataframes = {}
all_local_components = {}

for state in STATES_TO_SIMULATE:
    print(f"\nProcessing State: {state}...")
    try:
        # Load raw data
        acs_data = DATASOURCE.get_data(states=[state], download=True)

        # CORRECTED: Use df_to_numpy and construct DataFrame manually
        features_np, labels_np, _ = ACSIncome.df_to_numpy(acs_data)
        df_state = pd.DataFrame(features_np, columns=ACSIncome.features)

        # Add the raw label back (it corresponds to TARGET_COL_RAW for ACSIncome)
        df_state[TARGET_COL_RAW] = labels_np

        # --- Preprocessing ---
        # Create binary target label by thresholding the raw target
        df_state[TARGET_COL_PROCESSED] = (df_state[TARGET_COL_RAW] > TARGET_POSITIVE_THRESHOLD).astype(int)

        # Optional: Handle potential NaN values introduced if df_to_numpy behaves unexpectedly
        # or if certain features have missingness not handled by folktables' defaults.
        # For simplicity, we assume columns used (SEX, label, AGEP, WKHP, POVPIP) are reasonable.
        # May need df_state.dropna(subset=[SENSITIVE_COL, TARGET_COL_PROCESSED]) depending on data quality.

        # Store dataframe for later verification
        all_dataframes[state] = df_state

        # Compute local components
        local_components = compute_local_components(
            df_state,
            SENSITIVE_COL,
            TARGET_COL_PROCESSED,
            TARGET_POSITIVE_VALUE,
            NUMERIC_FEATURE_COLS # Pass the list of numeric features for sums
        )
        all_local_components[state] = local_components
        print(f"Local Components for {state}:")
        print(f"  N: {local_components['N']}")
        print(f"  Counts_S (SEX): {dict(local_components['counts_S'])}")
        print(f"  Counts_Y1_S (High Income by SEX): {dict(local_components['counts_Y1_S'])}")
        print(f"  Sums_X: {dict(local_components['sums_X'])}")

    except Exception as e:
        print(f"Error processing state {state}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # Optionally, decide if you want to continue with other states
        # continue

# Check if any components were successfully computed
if not all_local_components:
    print("\nNo local components were computed. Exiting.")
    exit()

# 4. Simulate Aggregation for the Combined Dataset (All Successfully Processed States)
print("\n--- Aggregating Components for All Successfully Processed States ---")
components_to_aggregate = list(all_local_components.values())
aggregated = aggregate_components(components_to_aggregate)
print("Aggregated Components (Combined):")
# Add checks for empty aggregated components
print(f"  N: {aggregated.get('N', 0)}")
print(f"  Counts_S (SEX): {dict(aggregated.get('counts_S', Counter()))}")
print(f"  Counts_Y1_S (High Income by SEX): {dict(aggregated.get('counts_Y1_S', Counter()))}")
print(f"  Sums_X: {dict(aggregated.get('sums_X', defaultdict(float)))}")


# 5. Calculate Global Metrics from Aggregated Components
print("\n--- Calculating Global Metrics from Aggregated Components ---")
global_metrics_from_agg = calculate_global_metrics(aggregated)
print(f"Total Size: {global_metrics_from_agg.get('N_total', 'N/A')}")
print(f"Group Proportions: {global_metrics_from_agg.get('group_proportions', {})}")
print(f"Outcome Rates per Group: {global_metrics_from_agg.get('outcome_rates_per_group', {})}")
print(f"Average Features: {global_metrics_from_agg.get('average_features', {})}")
print(f"Statistical Parity Difference (SPD) [SEX=1 vs SEX=2]: {global_metrics_from_agg.get('spd', 'N/A')}")
print(f"Disparate Impact (DI) [SEX=1 / SEX=2]: {global_metrics_from_agg.get('di', 'N/A')}")

# 6. Verification: Calculate Metrics Directly on Concatenated Data
print("\n--- Verification: Calculating Metrics Directly on Concatenated Data ---")
if all_dataframes:
    # Ensure only successfully processed states are included
    valid_dfs = [df for state, df in all_dataframes.items() if state in all_local_components]
    if not valid_dfs:
         print("No valid dataframes available for concatenation.")
         # Use exit() correctly
         exit()

    combined_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"Total size of combined dataframe: {len(combined_df)}")

    if combined_df.empty:
        print("Combined dataframe is empty, skipping direct calculation.")
    else:
        # Direct Calculation: Group Proportions
        # Check if sensitive column exists and is not empty after dropping NA
        if SENSITIVE_COL in combined_df.columns and not combined_df[SENSITIVE_COL].dropna().empty:
             direct_proportions = combined_df[SENSITIVE_COL].dropna().value_counts(normalize=True).to_dict()
             print(f"Direct Group Proportions: {direct_proportions}")
        else:
             direct_proportions = {}
             print(f"Direct Group Proportions: N/A (Column '{SENSITIVE_COL}' missing or empty/all NA)")


        # Direct Calculation: Outcome Rates per Group
        # Check required columns exist
        if SENSITIVE_COL in combined_df.columns and TARGET_COL_PROCESSED in combined_df.columns:
            # Drop rows where sensitive group is NA before grouping
            df_filtered_sens_direct = combined_df.dropna(subset=[SENSITIVE_COL])
            if not df_filtered_sens_direct.empty: # Check if df is empty after dropping NA
                grouped_data = df_filtered_sens_direct.groupby(SENSITIVE_COL)
                if not grouped_data.groups: # Check if groupby result is empty
                    direct_rates = {}
                    print(f"Direct Outcome Rates per Group: N/A (No groups found after dropping NA in '{SENSITIVE_COL}')")
                else:
                    direct_rates = grouped_data[TARGET_COL_PROCESSED].mean().to_dict()
                    print(f"Direct Outcome Rates per Group: {direct_rates}")
            else:
                 direct_rates = {}
                 print(f"Direct Outcome Rates per Group: N/A (DataFrame empty after dropping NA in '{SENSITIVE_COL}')")
        else:
            direct_rates = {}
            print(f"Direct Outcome Rates per Group: N/A (Columns '{SENSITIVE_COL}' or '{TARGET_COL_PROCESSED}' missing)")



        # Direct Calculation: Average Features
        direct_avg_features = {}
        for col in NUMERIC_FEATURE_COLS:
            if col in combined_df.columns:
                 # Ensure numeric conversion and handle NAs before calculating mean
                 numeric_col = pd.to_numeric(combined_df[col], errors='coerce')
                 if not numeric_col.isna().all(): # Check if column is not all NA
                    direct_avg_features[col] = numeric_col.mean(skipna=True)
                 else:
                     direct_avg_features[col] = None # Column was all NA or non-numeric
            else:
                direct_avg_features[col] = None # Column not found

        print(f"Direct Average Features: {direct_avg_features}")


        # Direct Calculation: SPD and DI
        rateA_direct = direct_rates.get(1) # SEX=1 (Male)
        rateB_direct = direct_rates.get(2) # SEX=2 (Female)
        spd_direct = None
        di_direct = None
        if rateA_direct is not None and rateB_direct is not None:
            spd_direct = rateA_direct - rateB_direct
            if rateB_direct > 1e-7:
                di_direct = rateA_direct / rateB_direct
            else:
                 di_direct = float('inf') if rateA_direct > 1e-7 else None
        # Add cases where one rate might be None
        elif rateA_direct is not None:
            spd_direct = rateA_direct
            di_direct = float('inf')
        elif rateB_direct is not None:
             spd_direct = -rateB_direct
             di_direct = 0.0


        print(f"Direct SPD [SEX=1 vs SEX=2]: {spd_direct}")
        print(f"Direct DI [SEX=1 / SEX=2]: {di_direct}")

        # --- Comparison ---
        print("\n--- Comparison Summary ---")
        print(f"{'Metric':<25} | {'From Agg. Components':<25} | {'Direct Calculation':<25}")
        print("-" * 77)
        # Use .get for safety in case metrics are missing
        print(f"{'Total N':<25} | {global_metrics_from_agg.get('N_total', 'N/A'):<25} | {len(combined_df):<25}")

        # Compare proportions
        agg_props = global_metrics_from_agg.get('group_proportions', {})
        all_prop_groups = set(agg_props.keys()) | set(direct_proportions.keys())
        # Ensure printing format handles None
        for group in sorted(list(all_prop_groups)):
            agg_val = agg_props.get(group, None)
            dir_val = direct_proportions.get(group, None)
            agg_str = f"{agg_val:<25.6f}" if agg_val is not None else f"{str(None):<25}"
            dir_str = f"{dir_val:<25.6f}" if dir_val is not None else f"{str(None):<25}"
            print(f"{f'Prop(SEX={group})':<25} | {agg_str} | {dir_str}")


        # Compare rates
        agg_rates = global_metrics_from_agg.get('outcome_rates_per_group', {})
        all_rate_groups = set(agg_rates.keys()) | set(direct_rates.keys())
        # Ensure printing format handles None
        for group in sorted(list(all_rate_groups)):
            agg_val = agg_rates.get(group, None)
            dir_val = direct_rates.get(group, None)
            agg_str = f"{agg_val:<25.6f}" if agg_val is not None else f"{str(None):<25}"
            dir_str = f"{dir_val:<25.6f}" if dir_val is not None else f"{str(None):<25}"
            print(f"{f'Rate(Y=1|SEX={group})':<25} | {agg_str} | {dir_str}")

        # Compare features
        agg_feats = global_metrics_from_agg.get('average_features', {})
        all_feat_names = set(agg_feats.keys()) | set(direct_avg_features.keys())
         # Ensure printing format handles None
        for feature in sorted(list(all_feat_names)):
            agg_val = agg_feats.get(feature, None)
            dir_val = direct_avg_features.get(feature, None)
            agg_str = f"{agg_val:<25.2f}" if agg_val is not None else f"{str(None):<25}"
            dir_str = f"{dir_val:<25.2f}" if dir_val is not None else f"{str(None):<25}"
            print(f"{f'Avg({feature})':<25} | {agg_str} | {dir_str}")

        # Ensure printing format handles None for SPD/DI
        spd_agg = global_metrics_from_agg.get('spd', None)
        spd_agg_str = f"{spd_agg:<25.6f}" if spd_agg is not None else f"{str(None):<25}"
        spd_dir_str = f"{spd_direct:<25.6f}" if spd_direct is not None else f"{str(None):<25}"
        print(f"{'SPD (SEX 1 vs 2)':<25} | {spd_agg_str} | {spd_dir_str}")

        # Handle potential 'inf', None in DI for comparison printing
        di_agg = global_metrics_from_agg.get('di', None)
        di_direct_print = di_direct
        try:
            di_agg_str = f"{di_agg:<25.6f}" if di_agg is not None and di_agg != float('inf') else f"{str(di_agg):<25}"
        except TypeError:
            di_agg_str = f"{str(di_agg):<25}"
        try:
             di_direct_str = f"{di_direct_print:<25.6f}" if di_direct_print is not None and di_direct_print != float('inf') else f"{str(di_direct_print):<25}"
        except TypeError:
             di_direct_str = f"{str(di_direct_print):<25}"
        print(f"{'DI (SEX 1 / 2)':<25} | {di_agg_str} | {di_direct_str}")


else:
    print("No dataframes were loaded successfully, skipping direct verification.")

print("\nExperiment Finished.")


# --- Constants for Folktables ---
DATASOURCE = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
# Using a smaller list for quicker demonstration, use your full list if needed
STATES_TO_SIMULATE = all_states
SENSITIVE_COL = 'SEX' # 1: Male, 2: Female (as per ACS codes)
TARGET_COL_RAW = 'PINCP' # Total person's income - will be added back from labels_np
TARGET_COL_PROCESSED = 'label' # Our binary target
TARGET_POSITIVE_THRESHOLD = 50000 # Threshold for high income
TARGET_POSITIVE_VALUE = 1 # Value representing high income
# Define the columns needed specifically for the correlation score
# Make sure these columns are appropriate for correlation (numeric)
# Using AGEP and WKHP as example non-sensitive numerics.
NON_SENSITIVE_COLS_FOR_SCORE = ['AGEP', 'WKHP']

# *** CORRECT DEFINITION LOCATION FOR COLS_FOR_SCORE ***
COLS_FOR_SCORE = [SENSITIVE_COL, TARGET_COL_PROCESSED] + NON_SENSITIVE_COLS_FOR_SCORE


def prepare_df_for_corr(df, col_names):
    """Selects columns, converts to numeric, fills NA with 0."""
    # Select only the columns needed
    # Ensure columns exist before selecting
    cols_present = [col for col in col_names if col in df.columns]
    if not cols_present:
        return pd.DataFrame() # Return empty if no columns found
    df_subset = df[cols_present].copy()

    # Convert all columns to numeric, coercing errors.
    for col in cols_present:
        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
    # Fill any NaNs resulted from coercion or originally present with 0
    # WARNING: Filling NA with 0 can distort correlations.
    df_subset = df_subset.fillna(0)
    return df_subset

# --- Local Component Computation ---
def compute_correlation_components(df, var_names):
    """
    Computes the additive components needed for global correlation calculation.
    (Code from previous response)
    """
    if df.empty or not var_names:
        return {'N': 0, 'sums': Counter(), 'sums_sq': Counter(), 'sums_xy': defaultdict(Counter)}

    # Ensure var_names only contains columns present in df
    var_names_present = [var for var in var_names if var in df.columns]
    if not var_names_present:
         return {'N': 0, 'sums': Counter(), 'sums_sq': Counter(), 'sums_xy': defaultdict(Counter)}


    df_numeric = prepare_df_for_corr(df, var_names_present)
    n_total = len(df_numeric)

    if n_total == 0:
        return {'N': 0, 'sums': Counter(), 'sums_sq': Counter(), 'sums_xy': defaultdict(Counter)}

    # Calculate sums and sums of squares
    sums = df_numeric.sum().to_dict()
    sums_sq = (df_numeric**2).sum().to_dict()

    # Calculate sums of products (cross-products)
    sums_xy = defaultdict(Counter)
    for i, var1 in enumerate(var_names_present):
        for j in range(i, len(var_names_present)):
            var2 = var_names_present[j]
            # Ensure correct order for storage if needed, though Counter dict handles it
            key1, key2 = sorted([var1, var2])
            sums_xy[key1][key2] = (df_numeric[var1] * df_numeric[var2]).sum()

    return {
        'N': n_total,
        'sums': Counter(sums),
        'sums_sq': Counter(sums_sq),
        'sums_xy': sums_xy,
    }

# --- Aggregation Function ---
def aggregate_correlation_components(local_components_list):
    """
    Aggregates (sums) the correlation components from multiple clients.
    (Code from previous response)
    """
    if not local_components_list:
        return {'N': 0, 'sums': Counter(), 'sums_sq': Counter(), 'sums_xy': defaultdict(Counter)}

    aggregated = {
        'N': 0,
        'sums': Counter(),
        'sums_sq': Counter(),
        'sums_xy': defaultdict(Counter) # Use Counter for the inner dict too for easy update
    }

    all_vars = set() # Keep track of all variables encountered

    for components in local_components_list:
        aggregated['N'] += components.get('N', 0)
        local_sums = components.get('sums', Counter())
        aggregated['sums'].update(local_sums)
        aggregated['sums_sq'].update(components.get('sums_sq', Counter()))

        local_sums_xy = components.get('sums_xy', defaultdict(Counter))
        for var1, inner_dict in local_sums_xy.items():
             all_vars.add(var1)
             # Ensure inner dict is Counter before updating
             if not isinstance(aggregated['sums_xy'][var1], Counter):
                  aggregated['sums_xy'][var1] = Counter(aggregated['sums_xy'][var1])

             for var2, value in inner_dict.items():
                 all_vars.add(var2)
                 aggregated['sums_xy'][var1].update({var2: value}) # Use update with dict

        # Add keys from sums/sums_sq as well
        all_vars.update(local_sums.keys())


    # Ensure sums_xy structure is complete for all pairs (useful later)
    # Use only variables that actually appear in sums (have non-zero counts somewhere)
    valid_vars = list(aggregated['sums'].keys())
    sorted_vars = sorted(valid_vars)

    final_sums_xy = defaultdict(Counter)
    for i, var1 in enumerate(sorted_vars):
         for j in range(i, len(sorted_vars)):
              var2 = sorted_vars[j]
              key1, key2 = sorted([var1, var2]) # Ensure consistent ordering
              # Get value, defaulting to 0 if pair didn't exist in aggregated sums_xy
              value = aggregated['sums_xy'].get(key1, Counter()).get(key2, 0)
              final_sums_xy[key1][key2] = value

    aggregated['sums_xy'] = final_sums_xy

    return aggregated

# --- Global Calculation Function (CORRECTED) ---
def calculate_global_correlation_score(aggregated_components, var_names, sensitive_name, target_name, non_sensitive_names):
    """
    Calculates the global correlation matrix and the specific correlation_score
    from aggregated components. Returns NaN for correlations involving zero-variance variables.

    Args:
        aggregated_components (dict): Aggregated sums and counts.
        var_names (list): List of all variable names actually present in aggregated components.
        sensitive_name (str): Name of the sensitive variable column.
        target_name (str): Name of the target variable column.
        non_sensitive_names (list): List of non-sensitive variable column names present for the score.

    Returns:
        tuple: (correlation_score, global_correlation_matrix_df)
               Returns (np.nan, matrix_with_nans) if calculation involves NaN.
               Returns (None, None) if N <= 1.
    """
    N = aggregated_components.get('N', 0)
    if N <= 1: # Need more than 1 data point for variance/correlation
        print("Warning: Cannot calculate correlation with N <= 1.")
        return None, None

    sums = aggregated_components.get('sums', Counter())
    sums_sq = aggregated_components.get('sums_sq', Counter())
    sums_xy = aggregated_components.get('sums_xy', defaultdict(Counter))

    # Calculate Means
    means = {var: sums.get(var, 0) / N for var in var_names}

    # Calculate Variances and Std Devs
    variances = {}
    std_devs = {}
    for var in var_names:
        mean_val = means.get(var, 0) # Use get for safety
        # Var(X) = E[X^2] - (E[X])^2 = Sum(X^2)/N - (Sum(X)/N)^2
        variance = (sums_sq.get(var, 0) / N) - (mean_val ** 2)
        # Handle potential floating point inaccuracies leading to tiny negative variance
        variances[var] = max(0, variance)
        std_devs[var] = np.sqrt(variances[var])

    # Calculate Covariances and Correlations
    num_vars = len(var_names)
    global_cov_matrix = pd.DataFrame(np.nan, index=var_names, columns=var_names) # Initialize with NaN
    global_corr_matrix = pd.DataFrame(np.nan, index=var_names, columns=var_names) # Initialize with NaN

    for i, var1 in enumerate(var_names):
        for j in range(i, len(var_names)):
            var2 = var_names[j]
            key1, key2 = sorted([var1, var2]) # Use consistent key order

            # Cov(X, Y) = E[XY] - E[X]E[Y] = Sum(XY)/N - (Sum(X)/N)(Sum(Y)/N)
            sum_xy_val = sums_xy.get(key1, Counter()).get(key2, 0)
            covariance = (sum_xy_val / N) - (means[var1] * means[var2])

            global_cov_matrix.loc[var1, var2] = covariance
            global_cov_matrix.loc[var2, var1] = covariance # Symmetric

            # Correlation - CORRECTED HANDLING OF ZERO VARIANCE
            std_dev1 = std_devs.get(var1, 0) # Use get with default
            std_dev2 = std_devs.get(var2, 0) # Use get with default
            correlation = np.nan # Default to NaN

            if std_dev1 > 1e-10 and std_dev2 > 1e-10: # Avoid division by zero/near-zero
                 correlation = covariance / (std_dev1 * std_dev2)
                 # Clamp correlation to [-1, 1] due to potential floating point issues
                 correlation = max(-1.0, min(1.0, correlation))
            # If std dev is zero for either, correlation remains NaN

            global_corr_matrix.loc[var1, var2] = correlation
            global_corr_matrix.loc[var2, var1] = correlation

    # Set diagonal to 1 for variables with non-zero variance
    for var in var_names:
        if std_devs.get(var, 0) > 1e-10:
             global_corr_matrix.loc[var, var] = 1.0
        # Diagonal remains NaN if variance is zero


    # --- Calculate the user's specific correlation_score ---
    correlation_score = np.nan # Default to NaN
    try:
        # Ensure all needed columns are in the calculated matrix
        required_cols = [sensitive_name, target_name] + non_sensitive_names
        if not all(col in global_corr_matrix.index for col in required_cols):
             print("Warning: Not all required columns for score found in correlation matrix.")
             # Score remains NaN, return matrix
             return correlation_score, global_corr_matrix

        # Step 1: Direct correlation
        direct_corr = global_corr_matrix.loc[sensitive_name, target_name]

        # Step 2: Indirect correlations
        indirect_corr = 0
        all_indirect_valid = True
        for non_sensitive_name in non_sensitive_names:
            corr_sens_non_sens = global_corr_matrix.loc[sensitive_name, non_sensitive_name]
            corr_non_sens_target = global_corr_matrix.loc[non_sensitive_name, target_name]
            # Check if any component correlation is NaN
            if np.isnan(corr_sens_non_sens) or np.isnan(corr_non_sens_target):
                 all_indirect_valid = False
                 break # One NaN makes the indirect sum calculation invalid (NaN)
            indirect_corr += corr_sens_non_sens * corr_non_sens_target

        # Step 3: Total score computation - will be NaN if direct_corr is NaN or any indirect part was NaN
        if all_indirect_valid and not np.isnan(direct_corr):
            correlation_score = direct_corr + indirect_corr
        else:
            correlation_score = np.nan # Ensure score is NaN if any component is NaN


    except KeyError as e:
        print(f"Error calculating correlation_score: Missing key {e}")
        correlation_score = np.nan
    except Exception as e:
        print(f"Error calculating correlation_score: {e}")
        correlation_score = np.nan


    return correlation_score, global_corr_matrix


# --- Main Experiment ---

print("Starting Folktables Experiment (Exact Correlation - Corrected v3)...")

# --- Load Data and Compute Local Components ---
all_dataframes = {}
all_corr_components = {} # Store correlation components

for state in STATES_TO_SIMULATE:
    print(f"\nProcessing State: {state}...")
    try:
        # Load raw data
        acs_data = DATASOURCE.get_data(states=[state], download=True)

        # Use df_to_numpy and construct DataFrame manually
        features_np, labels_np, _ = ACSIncome.df_to_numpy(acs_data)
        df_state = pd.DataFrame(features_np, columns=ACSIncome.features)

        # Add the raw label back (it corresponds to TARGET_COL_RAW for ACSIncome)
        df_state[TARGET_COL_RAW] = labels_np

        # Preprocessing: Create binary target label
        df_state[TARGET_COL_PROCESSED] = (df_state[TARGET_COL_RAW] > TARGET_POSITIVE_THRESHOLD).astype(int)

        # Store dataframe for later verification
        all_dataframes[state] = df_state

        # --- Compute correlation components ---
        # Ensure only relevant columns are passed, handle potential missing ones
        available_cols_for_score = [col for col in COLS_FOR_SCORE if col in df_state.columns]
        if len(available_cols_for_score) < len(COLS_FOR_SCORE):
             print(f"Warning: Not all columns for score ({COLS_FOR_SCORE}) found in {state}. Using: {available_cols_for_score}")

        if len(available_cols_for_score) > 1: # Need at least 2 vars for correlation components
            local_components = compute_correlation_components(
                df_state,
                available_cols_for_score # Use only available columns
            )
            all_corr_components[state] = local_components
            print(f"Correlation Components computed for {state} using {available_cols_for_score}.")
        else:
            print(f"Skipping correlation components for {state} due to missing columns or only 1 column available.")

    except Exception as e:
        print(f"Error processing state {state}: {e}")
        import traceback
        traceback.print_exc()

# Check if any components were successfully computed
if not all_corr_components:
    print("\nNo correlation components were computed. Exiting.")
    exit()

# --- Aggregation ---
print("\n--- Aggregating Correlation Components ---")
components_to_aggregate = list(all_corr_components.values())
aggregated_corr_components = aggregate_correlation_components(components_to_aggregate)
print("Aggregation complete.")

# --- Global Calculation ---
print("\n--- Calculating Global Correlation Score from Aggregated Components ---")
# Determine the final list of variables present in aggregated components
final_var_names = sorted(list(aggregated_corr_components.get('sums', Counter()).keys()))

# Filter non_sensitive_names to only those present in the final list
final_non_sensitive_names = [name for name in NON_SENSITIVE_COLS_FOR_SCORE if name in final_var_names]

# Check if sensitive and target names are present
if SENSITIVE_COL not in final_var_names or TARGET_COL_PROCESSED not in final_var_names:
     print("Error: Sensitive or Target column missing from aggregated data. Cannot compute score.")
     global_score_agg = None
     global_corr_matrix_agg = None
else:
    global_score_agg, global_corr_matrix_agg = calculate_global_correlation_score(
        aggregated_corr_components,
        final_var_names, # Use variables actually present
        SENSITIVE_COL,
        TARGET_COL_PROCESSED,
        final_non_sensitive_names # Use variables actually present
    )

print(f"Global Correlation Score (from Aggregation): {global_score_agg}")
# print("\nGlobal Correlation Matrix (from Aggregation):")
# print(global_corr_matrix_agg) # Can be large

# --- Verification ---
print("\n--- Verification: Calculating Correlation Score Directly on Concatenated Data ---")
if all_dataframes:
    valid_dfs = [df for state, df in all_dataframes.items() if state in all_corr_components] # Use states where components were computed
    if not valid_dfs:
         print("No valid dataframes available for concatenation.")
         exit()

    combined_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"Total size of combined dataframe: {len(combined_df)}")

    if combined_df.empty:
        print("Combined dataframe is empty, skipping direct calculation.")
    else:
        # Use only columns present in the final aggregated analysis for fair comparison
        available_cols_direct = [col for col in COLS_FOR_SCORE if col in combined_df.columns]

        # Check if enough columns available for correlation
        if len(available_cols_direct) < 2:
             print("Direct calculation skipped: Less than 2 columns available.")
             global_score_direct = None
             global_corr_matrix_direct = None
        # Check if sensitive and target columns are available for score calculation
        elif SENSITIVE_COL not in available_cols_direct or TARGET_COL_PROCESSED not in available_cols_direct:
             print("Direct calculation skipped: Sensitive or Target column missing in combined DF for score calculation.")
             global_score_direct = None
             global_corr_matrix_direct = None # Matrix might be calculable but score is not
             # Optionally calculate matrix anyway:
             df_direct_numeric = prepare_df_for_corr(combined_df, available_cols_direct)
             global_corr_matrix_direct = df_direct_numeric.corr()
             print("\nGlobal Correlation Matrix (Direct Calculation - Score N/A):")
             print(global_corr_matrix_direct)

        else:
            # Proceed with direct calculation
            df_direct_numeric = prepare_df_for_corr(combined_df, available_cols_direct)
            global_corr_matrix_direct = df_direct_numeric.corr()

            print("\nGlobal Correlation Matrix (Direct Calculation):")
            print(global_corr_matrix_direct)

            # Calculate score directly from this matrix
            direct_score = np.nan # Default to NaN
            try:
                 # Filter non-sensitive names to those available directly
                 direct_non_sensitive_names = [name for name in NON_SENSITIVE_COLS_FOR_SCORE if name in available_cols_direct]

                 direct_corr = global_corr_matrix_direct.loc[SENSITIVE_COL, TARGET_COL_PROCESSED]
                 indirect_corr = 0
                 all_indirect_valid_direct = True

                 if np.isnan(direct_corr): # If direct is NaN, score is NaN
                      all_indirect_valid_direct = False
                 else:
                     for non_sensitive_name in direct_non_sensitive_names:
                         corr_sens_non_sens = global_corr_matrix_direct.loc[SENSITIVE_COL, non_sensitive_name]
                         corr_non_sens_target = global_corr_matrix_direct.loc[non_sensitive_name, TARGET_COL_PROCESSED]
                         # Check if any component correlation is NaN
                         if np.isnan(corr_sens_non_sens) or np.isnan(corr_non_sens_target):
                              all_indirect_valid_direct = False
                              break # One NaN makes the indirect sum calculation invalid (NaN)
                         indirect_corr += corr_sens_non_sens * corr_non_sens_target

                 # Step 3: Total score computation - will be NaN if direct_corr is NaN or any indirect part was NaN
                 if all_indirect_valid_direct:
                     direct_score = direct_corr + indirect_corr
                 # else: direct_score remains np.nan

            except KeyError as e:
                 print(f"Error calculating direct score: Missing key {e}")
                 direct_score = np.nan
            except Exception as e:
                 print(f"Error calculating direct score: {e}")
                 direct_score = np.nan

            global_score_direct = direct_score
            print(f"\nGlobal Correlation Score (Direct Calculation): {global_score_direct}")

            # --- Comparison ---
            print("\n--- Comparison Summary ---")
            print(f"{'Metric':<25} | {'From Agg. Components':<25} | {'Direct Calculation':<25}")
            print("-" * 77)
            # Use format specifiers that handle NaN
            print(f"{'Correlation Score':<25} | {global_score_agg:<25.8f} | {global_score_direct:<25.8f}")

else:
    print("No dataframes were loaded successfully, skipping direct verification.")

print("\nExperiment Finished.")
