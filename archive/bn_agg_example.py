import pandas as pd
import numpy as np
import random
from folktables import ACSDataSource, ACSIncome
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import entropy # Needed for KL divergence
from sklearn.metrics import mutual_info_score # Needed for compute_mi_from_bn helper
from itertools import combinations
import time
import warnings

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning) # pgmpy might issue UserWarnings


# --- Constants ---
DATASOURCE = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
# Make sure these names EXACTLY match the columns after load_and_preprocess AND BN nodes
TARGET_VAR = 'INCOME'
SENSITIVE_VAR = 'SEX'
# Non-sensitive variables used in the BN structure
NON_SENSITIVE_VARS = ['AGEP', 'SCHL', 'OCCP'] # Make sure OCCP is handled in preprocessing and structure

# --- PFL Weights ---
# Optimized weights (via differential evolution on validation data)
WEIGHTS = {
    'alpha_ST': 2.0,      # Direct Bias: MI(S, T)
    'alpha_SN': 0.8889,   # Proxy Bias: sum(MI(S, Nk))
    'delta_NT': 1.3333,   # Utility: sum(MI(Nk, T)) (Reward factor)
    'beta_NN': 0.1111     # Redundancy: sum(MI(Nk, Nj))
}

# --- Function Definitions ---

def load_and_preprocess(states):
    """Loads and preprocesses data for specified states, ensuring discrete outputs."""
    print(f"  Loading data for {states}...")
    # Assuming DATASOURCE is defined globally or passed as argument
    data = DATASOURCE.get_data(states=states, download=True) # Consider download=False if cached
    features, labels, _ = ACSIncome.df_to_numpy(data)
    df = pd.DataFrame(features, columns=ACSIncome.features)
    df[TARGET_VAR] = labels # Use constant TARGET_VAR

    relevant_cols = list(set([SENSITIVE_VAR] + NON_SENSITIVE_VARS + [TARGET_VAR]))
    df = df[[col for col in relevant_cols if col in df.columns]].copy() # Use copy
    print(f"  Initial columns kept: {df.columns.tolist()}")

    print("  Discretizing features...")
    # Discretization logic (same as before)
    if 'AGEP' in df.columns:
        df['AGEP'] = pd.cut(df['AGEP'], bins=[0, 25, 40, 60, 100], labels=["young", "adult", "middle", "senior"], right=False, include_lowest=True).astype(str)
    if 'SCHL' in df.columns:
        df['SCHL'] = pd.cut(df['SCHL'], bins=[0, 10, 13, 16, 20], labels=["low", "med", "high", "very_high"], right=False, include_lowest=True).astype(str)
    if 'OCCP' in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df['OCCP']):
                 df['OCCP'] = pd.qcut(df['OCCP'], q=5, labels=[f"occ_q{i+1}" for i in range(5)], duplicates='drop')
            else:
                 top_cats = df['OCCP'].astype(str).value_counts().nlargest(4).index
                 df['OCCP'] = df['OCCP'].apply(lambda x: str(x) if x in top_cats else 'OCCP_Other')
            df['OCCP'] = df['OCCP'].astype(str)
        except Exception as e:
            print(f"  Warning: Failed to discretize OCCP, using raw values as strings. Error: {e}")
            df['OCCP'] = df['OCCP'].astype(str) # Fallback
    if SENSITIVE_VAR in df.columns: df[SENSITIVE_VAR] = df[SENSITIVE_VAR].astype(str)
    if TARGET_VAR in df.columns: df[TARGET_VAR] = df[TARGET_VAR].astype(str)

    # Handle NaNs introduced by processing
    final_cols = df.columns.tolist()
    for col in final_cols:
         # Check if column exists before trying to fillna
         if col in df:
            # Check if it's already categorical
            if pd.api.types.is_categorical_dtype(df[col]):
                 # Check if 'MISSING' is already a category
                 if 'MISSING' not in df[col].cat.categories:
                      df[col] = df[col].cat.add_categories('MISSING')
                 df[col] = df[col].fillna('MISSING')
            else:
                 # Convert to string before filling NaN if not categorical
                 df[col] = df[col].astype(str).fillna('MISSING')


    df = df.reset_index(drop=True)
    print(f"  Preprocessing complete. Shape: {df.shape}")
    return df


# Define BN structure (ensure nodes match preprocessed column names)
structure = BayesianNetwork([
    ('AGEP', 'INCOME'), ('SCHL', 'INCOME'), ('SEX', 'INCOME'), ('OCCP', 'INCOME'), # To Target
    ('SEX', 'SCHL'), ('AGEP', 'SCHL'), ('AGEP','OCCP') # Example dependencies between features
])
# Ensure OCCP is added if needed based on NON_SENSITIVE_VARS
if 'OCCP' in NON_SENSITIVE_VARS and 'OCCP' not in structure.nodes():
     structure.add_node('OCCP')
     # Add edges if OCCP should be connected, e.g., structure.add_edge('OCCP', 'INCOME')
     if ('OCCP', 'INCOME') not in structure.edges(): structure.add_edge('OCCP', 'INCOME')

# Learn BN
def learn_bn(data, structure):
    """Learns BN parameters for a given structure and data."""
    print(f"  Learning BN from data shape {data.shape}...")
    nodes_in_structure = list(structure.nodes())
    data_filtered = data[[node for node in nodes_in_structure if node in data.columns]].copy()
    if not all(node in data_filtered.columns for node in nodes_in_structure):
         missing = [node for node in nodes_in_structure if node not in data_filtered.columns]
         print(f"  Error: Data is missing columns required by BN structure: {missing}. Cannot fit.")
         return None
    for col in data_filtered.columns:
        # Ensure discrete type (string or category)
        if not pd.api.types.is_string_dtype(data_filtered[col]) and \
           not pd.api.types.is_object_dtype(data_filtered[col]) and \
           not pd.api.types.is_categorical_dtype(data_filtered[col]):
              data_filtered[col] = data_filtered[col].astype(str)

    model = structure.copy()
    try:
        model.fit(data_filtered, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)
        # model.check_model() # Optional verification
        print("  BN fitting complete.")
        return model
    except Exception as e:
        print(f"  ERROR during BN fitting: {e}")
        return None


# === --- User's Original aggregate_cpds Function (with minor validation) --- ===
def aggregate_cpds(models, sizes):
    """ Aggregates CPDs - Reverted to User's Original Version from Prompt #161.
        WARNING: Potential issues noted in comments might exist.
    """
    print("Aggregating CPDs (using user's original logic)...")
    if not models or not sizes or len(models) != len(sizes): print("Error: Invalid input to aggregate_cpds."); return None
    valid_models = [(m, s) for m, s in zip(models, sizes) if m is not None];
    if not valid_models: print("Error: No valid models to aggregate."); return None
    models, sizes = zip(*valid_models)
    total = sum(sizes)
    if total <= 0: print("Warning: Total size is zero or negative."); return models[0].copy() if models else None

    base_model = models[0].copy()
    for var in base_model.nodes():
        cpd_sum = None; cpd_info_model = None
        for model, size in zip(models, sizes):
            cpd = model.get_cpds(var);
            if cpd is None: continue
            if cpd_info_model is None: cpd_info_model = model
            try:
                 weighted = np.array(cpd.values, dtype=float) * size
                 cpd_sum = weighted if cpd_sum is None else cpd_sum + weighted
            except ValueError as e: print(f"Warning: ValueError summing CPD for {var}. {e}"); continue
        if cpd_sum is None or cpd_info_model is None: print(f"Warning: Could not aggregate CPDs for variable {var}. Skipping."); continue

        avg_values = cpd_sum / total
        ref_cpd = cpd_info_model.get_cpds(var);
        if ref_cpd is None: print(f"ERROR: Could not get ref CPD for {var}."); continue
        avg_values = np.atleast_2d(avg_values);
        if avg_values.shape[0] == 1: avg_values = avg_values.T
        evidence = ref_cpd.get_evidence(); cardinality = ref_cpd.get_cardinality(evidence)
        evidence_card = [cardinality[evi] for evi in evidence] if evidence else []
        variable_card = ref_cpd.variable_card; shape = (variable_card, int(np.round(np.prod(evidence_card))) if evidence_card else 1)
        if avg_values.size != np.prod(shape): print(f"Error: Size mismatch reshaping CPD {var}. Exp {np.prod(shape)}, got {avg_values.size}. Skip."); continue
        try: reshaped_avg_values = avg_values.reshape(shape, order='F')
        except ValueError as e: print(f"Error: Reshaping failed for CPD {var}. Shape {shape}, Vals {avg_values.shape}. {e}. Skip."); continue

        try:
            new_cpd = TabularCPD(var, variable_card, reshaped_avg_values.tolist(), evidence=evidence, evidence_card=evidence_card)
            new_cpd.normalize() # Try to normalize
            base_model.add_cpds(new_cpd)
        except Exception as e: print(f"Error creating/adding TabularCPD for {var}: {e}")
    try: base_model.check_model(); print("CPD Aggregation complete (User's Original Logic)."); return base_model
    except Exception as e: print(f"Error: Aggregated model check failed: {e}"); return None
# === --- END Reverted aggregate_cpds --- ===


# --- Helper Function to Calculate MI from BN (using Inference - CORRECTED) ---
def compute_mi_from_bn(bn_model, var1, var2, inference_engine):
    """Calculates MI(var1; var2) using probabilities from the BN."""
    if var1 == var2: return 0.0
    try:
        if var1 not in bn_model.nodes() or var2 not in bn_model.nodes(): return np.nan
        joint_factor = inference_engine.query(variables=[var1, var2], joint=True, show_progress=False)
        joint_factor.normalize(inplace=True)
        # REMOVED .reorder_variables() - relies on query order matching [var1, var2]
        if joint_factor.variables != [var1, var2]:
             print(f"Warning: Factor order {joint_factor.variables} != query order ({[var1, var2]}). Axis assumption may fail.")
        p_xy = joint_factor.values
        if not isinstance(p_xy, np.ndarray) or p_xy.ndim != 2: return np.nan
        p_xy_sum = np.sum(p_xy)
        if not np.isclose(p_xy_sum, 1.0):
             print(f"Warning: P({var1},{var2}) sum is {p_xy_sum}. Renormalizing.")
             if p_xy_sum > 1e-9: p_xy = p_xy / p_xy_sum
             else: return 0.0
        p_x = np.sum(p_xy, axis=1); p_y = np.sum(p_xy, axis=0)
        mi = 0.0; eps = 1e-12
        for i in range(p_xy.shape[0]):
            for j in range(p_xy.shape[1]):
                if p_xy[i, j] > eps and p_x[i] > eps and p_y[j] > eps:
                     ratio = p_xy[i, j] / (p_x[i] * p_y[j])
                     if ratio > eps: mi += p_xy[i, j] * np.log2(ratio)
        return max(0.0, mi)
    except Exception as e: print(f"ERROR computing MI for ({var1}, {var2}): {e}"); return np.nan


# --- PFL FUNCTION using BN ---
def calculate_pfl_from_bn(bn_aggregated, sensitive_var, non_sensitive_vars, target_var, weights):
    """Calculates PFL directly from an aggregated pgmpy BayesianNetwork object."""
    # print(f"Calculating PFL from BN for S='{sensitive_var}', T='{target_var}'...") # Verbose
    start_time = time.time()
    pfl_score = 0.0; contains_nan_mi = False
    if not isinstance(bn_aggregated, BayesianNetwork): print("Error: Input is not BayesianNetwork."); return np.inf
    available_vars = set(bn_aggregated.nodes())
    if sensitive_var not in available_vars or target_var not in available_vars: print("Error: BN missing S or T node."); return np.inf
    valid_non_sensitive = [ns for ns in non_sensitive_vars if ns in available_vars]
    # if not valid_non_sensitive: print(f"Warning: No valid non-sensitive vars found in BN.") # Verbose

    try: inference_engine = VariableElimination(bn_aggregated)
    except Exception as e: print(f"Error initializing inference engine: {e}"); return np.inf

    w_st = weights.get('alpha_ST', 1.0); w_sn = weights.get('alpha_SN', 1.0)
    w_nn = weights.get('beta_NN', 0.1); w_nt = weights.get('delta_NT', 1.0)
    mi_results = {}
    required_pairs = set()
    required_pairs.add(tuple(sorted((sensitive_var, target_var))))
    for ns_var in valid_non_sensitive: required_pairs.add(tuple(sorted((sensitive_var, ns_var)))); required_pairs.add(tuple(sorted((ns_var, target_var))))
    for ns_var1, ns_var2 in combinations(valid_non_sensitive, 2): required_pairs.add(tuple(sorted((ns_var1, ns_var2))))

    for pair in required_pairs:
        v1, v2 = pair
        mi = compute_mi_from_bn(bn_aggregated, v1, v2, inference_engine)
        mi_results[pair] = mi
        if pd.isna(mi): contains_nan_mi = True; print(f"PFL Error: MI failed for pair {pair}"); break

    if contains_nan_mi: return np.nan

    pair_st = tuple(sorted((sensitive_var, target_var)))
    pfl_score += w_st * mi_results.get(pair_st, 0.0)
    pfl_score += w_sn * sum(mi_results.get(tuple(sorted((sensitive_var, ns_var))), 0.0) for ns_var in valid_non_sensitive)
    pfl_score += w_nn * sum(mi_results.get(tuple(sorted((ns_var1, ns_var2))), 0.0) for ns_var1, ns_var2 in combinations(valid_non_sensitive, 2))
    pfl_score -= w_nt * sum(mi_results.get(tuple(sorted((ns_var, target_var))), 0.0) for ns_var in valid_non_sensitive)

    end_time = time.time()
    print(f" PFL calculation time: {end_time - start_time:.2f} seconds.")
    return pfl_score


# --- Main execution section ---
all_states = [ # Example: Use a smaller subset for faster testing
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'MD', 'MA', 'NY', 'TX', 'VA'
]

n = 5 # Number of states to sample
sampled_states = random.sample(all_states, n)
# sampled_states = ['CA', 'MD', 'VA', 'NY', 'FL'] # Override for specific test
models = []
sizes = []
datasets = [] # **** Store datasets for union model ****

print(f"Preprocessing data and training local BNs for states: {sampled_states}")
for state in sampled_states:
    df = load_and_preprocess([state])
    if len(df) < 100: print(f"  Skipping state {state}, too few records: {len(df)}"); continue
    model = learn_bn(df, structure)
    models.append(model) # Store model even if None
    sizes.append(len(df))
    if model: datasets.append(df) # **** Store dataframe only if model learning succeeded ****


# Filter out None models *before* aggregation, keeping sizes aligned
valid_indices = [i for i, m in enumerate(models) if m is not None]
valid_models = [models[i] for i in valid_indices]
valid_sizes = [sizes[i] for i in valid_indices]


if valid_models and valid_sizes:
    # Aggregate using the original function provided by user
    agg_model = aggregate_cpds(valid_models, valid_sizes)

    # Calculate PFL for the aggregated model
    a_pfl = np.nan # Default if aggregation fails
    if agg_model:
        print("\nCalculating PFL score for the aggregated BN model...")
        a_pfl = calculate_pfl_from_bn(agg_model, SENSITIVE_VAR, NON_SENSITIVE_VARS, TARGET_VAR, WEIGHTS)
        print(f'\n>>> PFL Score from aggregated model: {a_pfl}')
    else:
        print("\nCould not calculate PFL for aggregated model (aggregation failed).")

    # --- **** ADDED: Union Model Training and Comparison **** ---
    print("\nConcatenating data and training Union BN model...")
    if datasets: # Check if we successfully stored any datasets
        df_union = pd.concat(datasets, ignore_index=True)
        print(f"Union dataset shape: {df_union.shape}")
        if not df_union.empty:
            union_model = learn_bn(df_union, structure)

            if union_model:
                print("\nCalculating PFL score for the union BN model...")
                u_pfl = calculate_pfl_from_bn(union_model, SENSITIVE_VAR, NON_SENSITIVE_VARS, TARGET_VAR, WEIGHTS)
                print(f'\n>>> PFL Score from union model: {u_pfl}')

                # --- Compare CPTs using KL divergence ---
                if agg_model: # Only compare if aggregation also succeeded
                    print("\nKL divergences between Union (P) and Aggregated (Q) BN CPTs: [KL(P || Q)]")
                    for var in structure.nodes():
                        try:
                             cpd_union = union_model.get_cpds(var)
                             cpd_agg = agg_model.get_cpds(var)

                             if cpd_union is None or cpd_agg is None:
                                 print(f"  Skipping KL for {var}: CPD missing in one of the models.")
                                 continue

                             # Flatten, add epsilon, normalize for entropy calculation
                             p_union = cpd_union.values.flatten() + 1e-12
                             q_agg = cpd_agg.values.flatten() + 1e-12

                             # Ensure shapes match before comparison (might differ slightly due to states)
                             if p_union.shape != q_agg.shape:
                                  print(f"  Skipping KL for {var}: CPT shapes mismatch - Union={p_union.shape}, Agg={q_agg.shape}")
                                  continue

                             p_union /= p_union.sum()
                             q_agg /= q_agg.sum()

                             kl = entropy(p_union, q_agg) # KL(P || Q) using scipy.stats.entropy
                             print(f"  KL({var} || Agg): {kl:.4f}")
                        except Exception as e:
                             print(f"  Error calculating KL for {var}: {e}")
                else:
                     print("\nSkipping CPT comparison because aggregated model failed.")
            else:
                 print("\nUnion model training failed, cannot calculate its PFL or compare CPTs.")
        else:
            print("\nUnion dataset is empty, cannot train union model.")
    else:
        print("\nNo valid local datasets were collected, cannot create union dataset.")
    # --- **** END ADDED SECTION **** ---

else:
    print("\nNot enough valid local models were trained. Skipping aggregation and PFL calculation.")
