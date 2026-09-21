import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from itertools import combinations, product
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage
import time # To estimate calculation time
from mi_utils import MI_FEATURES
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


all_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']



SENSITIVE_VARIABLE_NAME = 'SEX'
NON_SENSITIVE_VARIABLE_NAMES = [var for var in MI_FEATURES if var not in SENSITIVE_VARIABLE_NAME and var != 'labels']

WEIGHTS = {
    'alpha_ST': 2.0,      # Direct bias MI(S, T)
    'alpha_SN': 0.8889,   # Proxy bias MI(S, N_k)
    'beta_NN': 0.1111,    # Feature redundancy MI(N_i, N_j) - penalize correlated proxies
    'delta_NT': 1.3333    # Utility reward MI(N_k, T)
}

# --- Assume calculate_mi function is defined ---
# Using the robust version from the previous example
def calculate_mi(s1, s2):
    """
    Calculates Mutual Information between two discrete pandas Series.
    Returns MI >= 0 or np.nan if calculation fails.
    """
    if s1.name == s2.name: # MI(X,X) is Entropy H(X)
         # Returning NaN because PFL typically uses MI between *different* variables.
         # If H(X) was needed, it would require a separate calculation.
         return np.nan
    # Check for constant Series (MI is 0)
    if s1.nunique(dropna=False) <= 1 or s2.nunique(dropna=False) <= 1:
        return 0.0
    try:
        contingency_matrix = pd.crosstab(s1, s2)
        # Handle cases where crosstab might be empty or have zero sum
        if contingency_matrix.empty or contingency_matrix.sum().sum() < 1e-9:
            return 0.0
        mi = mutual_info_score(labels_true=None, labels_pred=None, contingency=contingency_matrix)
        # Ensure non-negative due to potential float precision issues
        return max(0.0, mi)
    except ValueError as ve:
        # Catches errors often related to incompatible data types or empty results
        print(f"Warning: ValueError during MI calc between '{s1.name}' & '{s2.name}'. Ret 0. Err: {ve}")
        return 0.0 # Treat as 0 MI
    except Exception as e:
        print(f"Warning: Could not calculate MI between '{s1.name}' & '{s2.name}'. Ret NaN. Err: {e}")
        return np.nan

# --- The Requested Function --- #
def compute_PFL_of_dataframe(df, sensitive_var=SENSITIVE_VARIABLE_NAME, non_sensitive_vars=NON_SENSITIVE_VARIABLE_NAMES, target_var='label', weights=WEIGHTS):
    """
    Computes the PFL score for a given DataFrame by first calculating
    all required pairwise Mutual Information values from the DataFrame,
    and then combining them using the PFL formula (with linear penalties).

    Args:
        df (pd.DataFrame): DataFrame containing combined, discrete data for a federation.
        sensitive_var (str): Name of the sensitive column.
        non_sensitive_vars (list): List of names of non-sensitive columns.
        target_var (str): Name of the target/label column.
        weights (dict): Dictionary with PFL weights (alpha_ST, alpha_SN, beta_NN, delta_NT).

    Returns:
        float: The calculated PFL score, or NaN/Inf if errors occur.
    """
    #print(f"Calculating PFL for DataFrame with shape {df.shape}...")
    start_time = time.time()

    pfl_score = 0.0
    calculation_successful = True # Flag to track MI calculation success

    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Error: Input DataFrame is empty or invalid.")
        return float('inf')

    # Define all potentially relevant columns for MI calculation
    all_potential_vars = [sensitive_var] + non_sensitive_vars + [target_var]
    all_potential_vars = sorted(list(set(all_potential_vars))) # Unique and sorted
    #print(all_potential_vars)
    #print(df.columns)
    # Check if columns actually exist in the DataFrame
    vars_present = [col for col in all_potential_vars if col in df.columns]
    missing_cols = [col for col in all_potential_vars if col not in df.columns]
    if missing_cols:
        print(f"Warning: DataFrame missing required columns: {missing_cols}. PFL might be inaccurate.")
        # Decide how to handle: return inf? Or proceed with available columns?
        # Let's proceed, but essential S and T must be present.
    if sensitive_var not in vars_present or target_var not in vars_present:
         print(f"Error: Sensitive ('{sensitive_var}') or Target ('{target_var}') column not found in DataFrame.")
         return float('inf')

    # Use only the valid non-sensitive variables present in the df
    valid_non_sensitive = [ns for ns in non_sensitive_vars if ns in vars_present]

    # --- Step 1: Calculate and store all required pairwise MI values ---
    # Store MI values in a dictionary using sorted tuples as keys
    #print(" Calculating required MI pairs from DataFrame...")
    mi_values = {}
    required_pairs = set()

    # Define pairs needed for PFL calculation
    required_pairs.add(tuple(sorted((sensitive_var, target_var)))) # S-T
    for ns_var in valid_non_sensitive:
        required_pairs.add(tuple(sorted((sensitive_var, ns_var)))) # S-N
        required_pairs.add(tuple(sorted((ns_var, target_var))))    # N-T
    for ns_var1, ns_var2 in combinations(valid_non_sensitive, 2):
        required_pairs.add(tuple(sorted((ns_var1, ns_var2))))       # N-N

    # Calculate MI only for the required pairs
    for pair in required_pairs:
        var1, var2 = pair
        if var1 is not var2 :  #Although this is not supposed to happen
            # print(f" Calculating MI for {pair}") # Very verbose debugging
            mi = calculate_mi(df[var1], df[var2])
            mi_values[pair] = mi
            if pd.isna(mi):
                calculation_successful = False
                print(f" PFL Error: MI calculation failed (NaN) for pair {pair}. Cannot compute accurate PFL.")
                # Decide: Stop now or return NaN at the end? Let's stop now.
                return float('nan')
        else :
            continue

    mi_calc_time = time.time()
    #print(f" Calculated {len(required_pairs)} MI pairs in {mi_calc_time - start_time:.2f} seconds.")

    # --- Step 2: Calculate PFL using pre-calculated MI values ---
    # Get weights with defaults
    w_st = weights.get('alpha_ST', 1.0)
    w_sn = weights.get('alpha_SN', 1.0)
    w_nn = weights.get('beta_NN', 0.1)
    w_nt = weights.get('delta_NT', 1.0)

    # Term 1: Direct Bias MI(S, T)
    pair_st = tuple(sorted((sensitive_var, target_var)))
    mi_st = mi_values.get(pair_st, 0.0) # Use 0.0 if somehow missing after checks
    pfl_score += w_st * mi_st

    # Term 2: Proxy Bias sum(MI(S, Nk))
    sum_sn_mi = 0.0
    for ns_var in valid_non_sensitive:
        pair_sn = tuple(sorted((sensitive_var, ns_var)))
        sum_sn_mi += mi_values.get(pair_sn, 0.0)
    pfl_score += w_sn * sum_sn_mi

    # Term 3: Redundancy sum(MI(Nk, Nj))
    sum_nn_mi = 0.0
    for ns_var1, ns_var2 in combinations(valid_non_sensitive, 2):
        pair_nn = tuple(sorted((ns_var1, ns_var2)))
        sum_nn_mi += mi_values.get(pair_nn, 0.0)
    pfl_score += w_nn * sum_nn_mi

    # Term 4: Utility sum(MI(Nk, T))
    sum_nt_mi = 0.0
    for ns_var in valid_non_sensitive:
        pair_nt = tuple(sorted((ns_var, target_var)))
        sum_nt_mi += mi_values.get(pair_nt, 0.0)
    pfl_score -= w_nt * sum_nt_mi # Subtract utility reward

    pfl_calc_time = time.time()
    #print(f" Aggregated PFL score in {pfl_calc_time - mi_calc_time:.4f} seconds.")
    total_time = pfl_calc_time - start_time
    #print(f" Total time for PFL calculation from DataFrame: {total_time:.2f} seconds.")

    return pfl_score

def pfl_of_federation(states, task='ACSIncome') :
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=states, download=True)
    if task == 'ACSEmployment' :
        df = pd.concat([ACSEmployment.df_to_pandas(acs_data)[0], ACSEmployment.df_to_pandas(acs_data)[1]], axis=1)
        #df.rename(columns={'P1NCP': 'label'}, inplace=True)
    if task == 'ACSPublicCoverage' :
        df = pd.concat([ACSPublicCoverage.df_to_pandas(acs_data)[0], ACSPublicCoverage.df_to_pandas(acs_data)[1]], axis=1)
        #df.rename(columns={'P1NCP': 'label'}, inplace=True)
    if task == 'ACSIncome' :
        df = pd.concat([ACSIncome.df_to_pandas(acs_data)[0], ACSIncome.df_to_pandas(acs_data)[1]], axis=1)
        df.rename(columns={'PINCP': 'label'}, inplace=True)
    return compute_PFL_of_dataframe(df)
