import pandas as pd
import numpy as np
from collections import defaultdict
from folktables import ACSDataSource, ACSIncome # Use ACSIncome task
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import math

# If using mutual_info_score for comparison, import it
# from sklearn.metrics import mutual_info_score
import warnings

warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
    message="^The behavior of DataFrame.sum with axis=None is deprecated"
)

# --- Constants ---
DATASOURCE = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
# Using a smaller list for quicker demonstration, use your full list if needed

# Select features - focus on categorical or easily binnable ones for MI demo
# Example: SEX (categorical), RAC1P (categorical), AGEP (needs binning)
MI_FEATURES = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P', 'label']

all_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

STATES_TO_SIMULATE = all_states
# SENSITIVE_COL = 'SEX' # Define if needed for specific analysis later

# --- Binning Strategy (Example for AGEP) ---
# IMPORTANT: This strategy MUST be identical across all clients
AGE_BINS = [0, 18, 30, 45, 65, 100] # Example age bins
AGE_LABELS = ['0-18', '19-30', '31-45', '46-65', '65+']
SCHL_BINS = [0, 16, 17, 21, 22, 23, 25] # Using approx Folktables codes
SCHL_LABELS = ['<HS', 'HS Grad', 'Some College/Assoc', 'Bachelor', 'Master', 'Doctorate/Prof']

# WKHP: Increased from ~3 bins to 8 bins for hours worked per week
# Covers not working, part-time ranges, full-time, and overtime ranges
WKHP_BINS = [-1, 0, 10, 20, 30, 39, 41, 50, 100] # 9 edges -> 8 bins
WKHP_LABELS = ['0', '1-9', '10-19', '20-29', '30-38', '39-40', '41-49', '50+'] # Adjusted labels


#OCCP Occupation
OCCP_MIN = 0            # min OCCP code
OCCP_MAX = 9999         # max OCCP code
NUM_OCCP_BINS = 10      # Creates 10 finer bins
OCCP_BINS = np.linspace(OCCP_MIN, OCCP_MAX + 1, NUM_OCCP_BINS + 1).tolist() # +1 to include max
OCCP_LABELS = [f'Occ_Grp_{i+1}' for i in range(NUM_OCCP_BINS)]



# POBP: Place of Birth — binned using WAOB-based world regions
# Fixed bin edges — 9 edges = 8 bins
POBP_BINS = [
    0,     # Start before any valid POBP
    60,    # 001–059: US States
    100,   # 061–099: PR/US Islands
    200,   # 100–199: Europe (approx, merged)
    300,   # 200–299: Asia (approx)
    310,   # 300–309: Northern America
    400,   # 310–399: Latin America
    500,   # 400–499: Africa
    555    # 500–554: Oceania/At Sea
]

POBP_LABELS = [
    'US States',
    'PR/US Islands',
    'Europe',
    'Asia',
    'Northern America',
    'Latin America',
    'Africa',
    'Oceania/At Sea'
]

BINNED_AGEP_COL = 'AGEP_BINNED'
BINNED_SCHL_COL = 'SCHL_BINNED'
BINNED_WKHP_COL = 'WKHP_BINNED'
BINNED_OCCP_COL = 'OCCP_BINNED'
BINNED_POPB_COL = 'POBP_BINNED'

SENSITIVE_VARIABLE_NAME = 'SEX' # 1: Male, 2: Female





# --- Function Definitions ---

def discretize_features(df, bin_specs):
    """Applies binning to specified columns."""
    df_binned = df.copy()
    for col, spec in bin_specs.items():
        if col in df.columns:
            bin_col_name = f"{col}_BINNED" # Standardize naming
            # Ensure labels match number of bins - 1
            if len(spec['labels']) != len(spec['bins']) - 1:
                 print(f"Warning: Mismatch between number of bins ({len(spec['bins'])}) and labels ({len(spec['labels'])}) for {col}. Skipping.")
                 continue

            # Use pd.Series method for potentially better NA handling if column is Series
            if isinstance(df[col], pd.Series):
                df_binned[bin_col_name] = pd.cut(df[col],
                                                 bins=spec['bins'],
                                                 labels=spec['labels'],
                                                 right=False, # Bins are [a, b)
                                                 include_lowest=True)
            else: # Fallback for non-Series (less likely with DataFrame input)
                 df_binned[bin_col_name] = pd.cut(df[col],
                                                 bins=spec['bins'],
                                                 labels=spec['labels'],
                                                 right=False,
                                                 include_lowest=True)

            # Convert the new column to string to handle potential NA from cut as a category
            if bin_col_name in df_binned.columns:
                df_binned[bin_col_name] = df_binned[bin_col_name].astype(str) # Treat resulting bins as strings/categories

        else:
            print(f"Warning: Column '{col}' not found for binning.")
    return df_binned

def compute_mi_components(df, var_names):
    """
    Computes local contingency tables for all pairs of specified variables.
    Handles potential NaNs by dropping rows pair-wise or using explicit category.
    """
    if df.empty or not var_names:
        return {'N': 0, 'contingency_tables': {}}

    n_total = len(df) # Use original length before potential pairwise drops
    contingency_tables = {}

    # Ensure all specified columns exist
    vars_present = [var for var in var_names if var in df.columns]
    if len(vars_present) < 2:
        return {'N': n_total, 'contingency_tables': {}} # Not enough vars for pairs

    for i, var1 in enumerate(vars_present):
        for j in range(i + 1, len(vars_present)):
            var2 = vars_present[j]
            pair_key = tuple(sorted((var1, var2)))
            try:
                # Calculate contingency table, explicitly handle NAs if not done in binning
                # Convert to string first to treat everything as discrete categories robustly
                # Use fillna('NaN_STR') to create an explicit category for missing values (treates as 0)
                series1 = df[var1].fillna('NaN_STR').astype(str)
                series2 = df[var2].fillna('NaN_STR').astype(str)
                #pd.crosstab provides the joint frequency counts for the two attributes (series1, series2)
                contingency_table = pd.crosstab(series1, series2)

                # Check if table is valid (might be empty if a column was all NaN etc)
                if not contingency_table.empty:
                     contingency_tables[pair_key] = contingency_table
                # else: print(f"Skipping empty contingency table for {pair_key}")

            except Exception as e:
                 print(f"Warning: Could not compute contingency table for ({var1}, {var2}): {e}")

    return {
        'N': n_total,
        'contingency_tables': contingency_tables
    }


def compute_sigma_rdp(epsilon, delta, num_queries):
    """
    Compute the Gaussian noise scale sigma using Rényi DP composition.
    
    For M independent Gaussian mechanism queries with L2 sensitivity = 1,
    the closed-form solution for sigma to achieve (epsilon, delta)-DP is:
    
    σ = (√(2M ln(1/δ)) + √(2M(ln(1/δ) + ε))) / (2ε)
    
    This is derived from optimizing over the RDP order α in the conversion
    from RDP to (ε,δ)-DP after linear composition of M queries.
    
    Args:
        epsilon: Target privacy parameter (total budget for all queries)
        delta: Target delta parameter
        num_queries: Number of queries M (contingency tables)
        
    Returns:
        float: Required noise standard deviation sigma
        
    Reference:
        Mironov, I. (2017). Rényi Differential Privacy. CSF.
    """
    M = num_queries
    ln_inv_delta = np.log(1.0 / delta)
    
    # Closed-form solution from RDP analysis
    # σ = (√(2M ln(1/δ)) + √(2M(ln(1/δ) + ε))) / (2ε)
    term1 = np.sqrt(2 * M * ln_inv_delta)
    term2 = np.sqrt(2 * M * (ln_inv_delta + epsilon))
    sigma = (term1 + term2) / (2 * epsilon)
    
    return sigma


def compute_noisy_mi_components_gaussian(df, var_names, epsilon, delta):
    """
    Computes local contingency tables for all pairs of specified variables
    and adds Gaussian noise calibrated using RDP composition to satisfy
    (epsilon, delta)-DP for the entire release.

    Uses Rényi Differential Privacy (RDP) for tighter composition bounds
    when releasing M = K(K+1)/2 contingency tables, where K is the number
    of variables (features + target).
    
    The L2 sensitivity of each contingency table is 1 (adding/removing a 
    record changes exactly one cell by ±1).

    Args:
        df (pd.DataFrame): Local data DataFrame. Assumes relevant columns contain
                           discrete/categorical data.
        var_names (list): List of variable names to compute pairwise tables for.
        epsilon (float): Total privacy budget epsilon > 0. Use float('inf') for no privacy.
        delta (float): Privacy budget delta (0 < delta < 1). Ignored if epsilon=inf.

    Returns:
        dict: {'N': int, 'contingency_tables': {pair_tuple: noisy_table_df}}
              Returns {'N': 0, 'contingency_tables': {}} if input is invalid.
              Tables contain floating point noisy counts. Returns 'N' exactly.
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame) or df.empty or not var_names:
        print("Warning: Empty DataFrame or no var_names provided.")
        return {'N': 0, 'contingency_tables': {}}
    if not isinstance(epsilon, (int, float)): raise TypeError("Epsilon must be numeric.")
    if not isinstance(delta, (int, float)): raise TypeError("Delta must be numeric.")
    if epsilon < 0: raise ValueError("Epsilon must be non-negative.")
    if not np.isinf(epsilon) and not (0 < delta < 1):
         raise ValueError("Delta must be strictly between 0 and 1 for finite epsilon > 0.")

    n_total = len(df)
    exact_contingency_tables = {}

    vars_present = [var for var in var_names if var in df.columns]
    if len(vars_present) < 2:
        print("Warning: Fewer than 2 specified variables present in DataFrame.")
        return {'N': n_total, 'contingency_tables': {}}

    # --- 1. Compute exact contingency tables ---
    print(" Computing exact contingency tables...")
    for i, var1 in enumerate(vars_present):
        for j in range(i + 1, len(vars_present)):
            var2 = vars_present[j]
            pair_key = tuple(sorted((var1, var2)))
            try:
                series1 = df[var1].fillna('NaN_STR').astype(str)
                series2 = df[var2].fillna('NaN_STR').astype(str)
                contingency_table = pd.crosstab(series1, series2)
                if not contingency_table.empty:
                    exact_contingency_tables[pair_key] = contingency_table
            except Exception as e:
                print(f"Warning: Could not compute exact contingency table for ({var1}, {var2}): {e}")

    if not exact_contingency_tables:
         print("Warning: No valid contingency tables computed.")
         return {'N': n_total, 'contingency_tables': {}}

    # --- Handle Epsilon = inf (No DP) ---
    if np.isinf(epsilon):
        print("Info: Epsilon is infinite, returning exact counts.")
        float_tables = {p: t.astype(float) for p, t in exact_contingency_tables.items()}
        return {'N': n_total, 'contingency_tables': float_tables}

    # --- Handle Epsilon <= 0 ---
    if epsilon <= 0:
         print("Warning: Epsilon is non-positive. Returning empty tables.")
         return {'N': n_total, 'contingency_tables': {}}

    # --- 2. Calculate Gaussian Noise Scale using RDP Composition ---
    # Number of queries M = number of variable pairs = K(K-1)/2
    K = len(vars_present)
    M = len(exact_contingency_tables)  # Should equal K*(K-1)/2
    
    # L2 sensitivity per contingency table = 1 (one person changes one cell by ±1)
    sensitivity_l2 = 1.0
    
    # Check delta is valid
    if delta <= 0 or delta >= 1:
        print(f"Error: Invalid delta ({delta}) for noise calculation.")
        return {'N': n_total, 'contingency_tables': {}}

    # Compute sigma using RDP closed-form formula
    try:
        sigma = compute_sigma_rdp(epsilon, delta, M)
        if np.isnan(sigma) or np.isinf(sigma) or sigma < 0:
            raise ValueError("Calculated sigma is invalid.")
    except (ValueError, FloatingPointError) as ve:
        print(f"Error calculating sigma (check inputs delta={delta}, eps={epsilon}, M={M}): {ve}")
        return {'N': n_total, 'contingency_tables': {}}

    print(f"Info: RDP Composition for {K} variables, {M} contingency tables")
    print(f"      Target (ε={epsilon}, δ={delta:.2e})-DP")
    print(f"      Required Gaussian noise σ = {sigma:.2f} per cell (L2 sensitivity = {sensitivity_l2})")

    # --- 3. Add Gaussian noise to each table ---
    noisy_contingency_tables = {}
    for pair, table in exact_contingency_tables.items():
        if table.size == 0: continue
        try:
            noise = np.random.normal(loc=0.0, scale=sigma, size=table.shape)
            noisy_table = table.astype(float) + noise
            noisy_table = noisy_table.clip(lower=0)  # Ensure non-negative counts
            noisy_table = james_stein_shrinkage(noisy_table)  # Post-processing denoising
            noisy_contingency_tables[pair] = pd.DataFrame(noisy_table, index=table.index, columns=table.columns)
        except Exception as e:
             print(f"Error adding Gaussian noise to table for pair {pair}: {e}")
             pass

    # Return the original N and the dictionary of noisy tables
    return {
        'N': n_total,
        'contingency_tables': noisy_contingency_tables
    }



def james_stein_shrinkage(noisy_table):
    """
    Apply the James-Stein estimator on a noisy contingency table.

    Args:
        noisy_table (np.ndarray): Noisy contingency table of counts.

    Returns:
        np.ndarray: Shrunk contingency table using James-Stein estimator.
    """
    # Convert to numpy array if not already
    noisy_table = np.array(noisy_table)

    # Calculate the mean of the noisy table
    table_mean = np.mean(noisy_table)

    # Calculate the squared deviations from the mean
    squared_deviations = (noisy_table - table_mean) ** 2

    # Compute the shrinkage factor (for p-dimensional tables)
    p = noisy_table.size  # The number of elements in the table
    shrinkage_factor = 1 - ((p - 2) * np.var(noisy_table)) / np.sum(squared_deviations)

    # Apply shrinkage factor (only shrink if shrinkage factor is positive)
    shrinkage_factor = max(shrinkage_factor, 0)  # Avoid negative shrinkage
    shrunk_table = noisy_table * shrinkage_factor

    return shrunk_table


def aggregate_mi_components(local_components_list):
    """
    Aggregates (sums) the contingency tables from multiple clients.
    """
    if not local_components_list:
        return {'N': 0, 'contingency_tables': {}}

    aggregated_n = sum(comp.get('N', 0) for comp in local_components_list)
    # Initialize with float type to handle potential non-integer results if needed later
    aggregated_tables = defaultdict(lambda: pd.DataFrame().astype(float))

    for components in local_components_list:
        local_tables = components.get('contingency_tables', {})
        for pair, table in local_tables.items():
             # Use add with fill_value=0 to handle non-overlapping indices/columns
            if not table.empty:
                 current_agg = aggregated_tables[pair]
                 # Ensure both are numeric (counts) before adding
                 table_numeric = table.apply(pd.to_numeric, errors='coerce').fillna(0)
                 # Ensure current aggregate is also numeric
                 current_agg_numeric = current_agg.apply(pd.to_numeric, errors='coerce').fillna(0)

                 # Perform addition aligning on both index and columns
                 aggregated_tables[pair] = current_agg_numeric.add(table_numeric, fill_value=0)

    # Ensure all aggregated tables have numeric type (int or float)
    for pair in aggregated_tables:
        aggregated_tables[pair] = aggregated_tables[pair].fillna(0).astype(float) # Use float

    return {
        'N': aggregated_n,
        'contingency_tables': dict(aggregated_tables) # Convert back to plain dict
    }

# --- Global Calculation Function (CORRECTED for Broadcasting) ---
def calculate_global_mi(aggregated_components, var_pair):
    """
    Calculates global Mutual Information for a specific variable pair
    from aggregated contingency tables. Uses log base 2 for marginals and joint probabilities.
    Handles potential numerical issues.
    """
    N = aggregated_components.get('N', 0)
    tables = aggregated_components.get('contingency_tables', {})
    pair_key = tuple(sorted(var_pair)) # Ensure correct key ordering

    # Check if N is valid and table exists/is valid
    if N <= 0 or pair_key not in tables:
        print("pair key not in tables")
        return np.nan
    counts_xy = tables[pair_key]
    if not isinstance(counts_xy, pd.DataFrame) or counts_xy.empty:
        print("not dataframe or is empty")
        return np.nan
    # Ensure table contains numeric counts
    counts_xy = counts_xy.apply(pd.to_numeric, errors='coerce').fillna(0)
    if counts_xy.sum().sum() < 1e-9: # Check if total count is effectively zero
         return 0.0 # No counts means no information shared

    # Calculate probabilities using the actual sum of the table
    N_table_sum = counts_xy.sum().sum()
    if N_table_sum <= 0:
        return 0.0
    p_xy = counts_xy / N_table_sum # Normalize by actual counts in table

    if p_xy.empty:
         return 0.0

    p_x = p_xy.sum(axis=1) # Marginal probabilities for rows (var1)
    p_y = p_xy.sum(axis=0) # Marginal probabilities for columns (var2)

    # Remove zero probability marginals to avoid issues
    p_x = p_x[p_x > 1e-12]
    p_y = p_y[p_y > 1e-12]
    if p_x.empty or p_y.empty:
        return 0.0 # If either marginal is all zero, MI is 0 -> Avoid log(0) issues

    # Align p_xy to only include rows/columns with non-zero marginals
    p_xy_filtered = p_xy.loc[p_x.index, p_y.index]

    # Calculate outer product p(x)p(y) using valid marginals
    p_x_p_y = pd.DataFrame(np.outer(p_x.values, p_y.values),
                           index=p_x.index, columns=p_y.index)

    # --- CORRECTED MI CALCULATION ---
    # Create mask for where p(x,y) and p(x)p(y) are both positive
    mask_xy_pos = (p_xy_filtered > 1e-12)
    mask_outer_pos = (p_x_p_y > 1e-12)
    valid_mask = mask_xy_pos & mask_outer_pos # Boolean DataFrame

    # Initialize DataFrame for the log term, filled with zeros
    log_ratio_terms = pd.DataFrame(0.0, index=p_xy_filtered.index, columns=p_xy_filtered.columns)

    # Calculate log2( p(x,y) / (p(x)p(y)) ) only where the mask is True
    # Get the values where the mask is true (these will be 1D arrays or scalar if only one True)
    p_xy_valid = p_xy_filtered[valid_mask]
    p_x_p_y_valid = p_x_p_y[valid_mask]

    # Calculate the log of the ratio for valid elements
    # Check if there are any valid elements before calculation
    if p_xy_valid.size > 0:
        # Ensure denominator is non-zero before division within log
        safe_p_x_p_y_valid = np.where(p_x_p_y_valid > 1e-12, p_x_p_y_valid, 1) # Avoid division by zero warning
        ratio = p_xy_valid / safe_p_x_p_y_valid
        # Take log only where ratio is positive (p_xy was > 0)
        log_values = np.log2(ratio[ratio > 1e-12]) # Apply log only on positive ratios
        # Place the results back into the log_ratio DataFrame using the original valid_mask
        # Make sure to place results only where ratio was positive
        log_ratio_terms[valid_mask & (ratio > 1e-12)] = log_values

    # Calculate MI: Sum of p(x,y) * log(...)
    # Use the mask again on p_xy to ensure we only sum terms where p(x,y) was positive
    mi = np.sum(np.sum(p_xy_filtered[valid_mask] * log_ratio_terms[valid_mask]))
    # --- END CORRECTION ---

    # Handle potential -0.0 results, return 0.0 instead
    if np.isclose(mi, 0.0):
         return 0.0
    # Return NaN if calculation somehow resulted in NaN, otherwise the value
    return mi if not np.isnan(mi) else np.nan



#To ensure that all sampled state datasets have the same shape after pre-procesing,
#-->we take the CA state dataset because it is large and contains all Values
#-->its column : values serve as the base of our one-hot-encoding

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=["CA"], download=True)
acs_data = ACSIncome.df_to_numpy(ca_data)
df = pd.DataFrame(acs_data[0], columns=ACSIncome.features)

def get_column_values_dict():
    return {col: sorted(df[col].unique().tolist()) for col in df.columns}


def preprocess_acs_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses ACS Folktables data for MLP input.
    Handles missing values, encodes categoricals (one-hot using global cats),
    scales numericals.
    """
    global_categories = get_column_values_dict()
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Error: Input is not a valid or non-empty pandas DataFrame.")
        return pd.DataFrame()

    df_processed = df.copy()
    print(f"Input shape: {df_processed.shape}")

    # --- Define Column Types ---
    numerical_cols = ['AGEP', 'WKHP']
    low_card_categorical_cols = ['COW', 'SCHL', 'MAR', 'RELP', 'SEX', 'RAC1P']
    high_card_categorical_cols = ['OCCP', 'POBP'] # To be dropped

    # Filter lists based on columns actually present in df_processed
    numerical_cols = [col for col in numerical_cols if col in df_processed.columns]
    low_card_categorical_cols = [col for col in low_card_categorical_cols if col in df_processed.columns]
    high_card_categorical_cols = [col for col in high_card_categorical_cols if col in df_processed.columns]

    # --- 1. Handle High Cardinality Columns ---
    if high_card_categorical_cols:
        print(f"Dropping high-cardinality columns: {high_card_categorical_cols}")
        df_processed.drop(columns=high_card_categorical_cols, inplace=True, errors='ignore')

    # --- 2. Impute Missing Values ---
    print("Imputing missing values...")
    impute_cat_placeholder = 'MISSING' # Make sure this is in GLOBAL_CATEGORIES lists

    for col in numerical_cols:
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            # print(f"  Filled NaNs in '{col}' with median ({median_val})")

    for col in low_card_categorical_cols:
         if df_processed[col].isnull().any():
            df_processed[col].fillna(impute_cat_placeholder, inplace=True)
            # print(f"  Filled NaNs in '{col}' with placeholder '{impute_cat_placeholder}'")

    # --- 3. Convert to Categorical and One-Hot Encode ---
    if low_card_categorical_cols:
        print(f"Applying pd.Categorical and One-hot encoding for: {low_card_categorical_cols}")
        cols_to_encode = []
        for col in low_card_categorical_cols:
            if col in global_categories:
                # Ensure the placeholder value used in imputation is in the categories
                all_cats = global_categories[col]
                if impute_cat_placeholder not in all_cats and df_processed[col].astype(str).str.contains(impute_cat_placeholder).any():
                     # Dynamically add if imputation happened and placeholder wasn't pre-defined
                     all_cats = all_cats + [impute_cat_placeholder]
                     print(f"  Added '{impute_cat_placeholder}' to categories for {col}")

                # Convert to categorical dtype with ALL possible categories defined
                df_processed[col] = pd.Categorical(df_processed[col], categories=all_cats)
                cols_to_encode.append(col) # Mark for get_dummies
            else:
                print(f"Warning: Global categories not defined for '{col}'. Skipping strict categorical conversion.")
                # Option: Convert to string/category anyway? Might still cause shape issues later.
                # df_processed[col] = df_processed[col].astype('category')
                # cols_to_encode.append(col)

        # Apply get_dummies ONLY to columns converted to pd.Categorical with global categories
        if cols_to_encode:
             df_processed = pd.get_dummies(df_processed, columns=cols_to_encode, drop_first=False, dummy_na=False) # dummy_na=False as we handle imputation
             print(f"Shape after one-hot encoding: {df_processed.shape}")

    # --- 4. Scale Numerical Features ---
    if numerical_cols:
        print(f"Scaling numerical columns: {numerical_cols}")
        # Ensure numerical columns exist after potential drops/encoding name changes (unlikely here)
        num_cols_to_scale = [col for col in numerical_cols if col in df_processed.columns]
        if num_cols_to_scale:
             scaler = StandardScaler()
             df_processed[num_cols_to_scale] = scaler.fit_transform(df_processed[num_cols_to_scale])
             print("  Numerical columns scaled using StandardScaler.")

    print(f"Preprocessing complete. Final shape: {df_processed.shape}")
    return df_processed






def calculate_mi(s1, s2):
    """
    Calculates Mutual Information between two discrete pandas Series.

    Args:
        s1 (pd.Series): First series (column).
        s2 (pd.Series): Second series (column).

    Returns:
        float: The calculated Mutual Information score. Returns NaN if calculation fails.
    """
    try:
        # Calculate the contingency matrix (joint frequency table)
        contingency_matrix = pd.crosstab(s1, s2)

        # Use sklearn's mutual_info_score with the contingency table
        # It calculates MI based on the observed joint frequencies
        mi = mutual_info_score(labels_true=None, labels_pred=None, contingency=contingency_matrix)
        return mi
    except Exception as e:
        # Handle potential errors (e.g., if a series is constant)
        print(f"Warning: Could not calculate MI between columns '{s1.name}' and '{s2.name}'. Error: {e}")
        return np.nan
