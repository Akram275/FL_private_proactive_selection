import pandas as pd
import numpy as np
from folktables import ACSDataSource, ACSIncome
from sklearn.covariance import GraphicalLasso
import random
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed
import warnings
import traceback


warnings.filterwarnings(
    "ignore",
    message=".*is_sparse is deprecated.*",
    category=FutureWarning
)

# Step 1: Setup
def get_state_data(state):
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=[state], download=True)
    features, labels, _ = ACSIncome.df_to_numpy(acs_data)
    feature_names = ACSIncome.features
    df = pd.DataFrame(features, columns=feature_names, dtype=np.float64)
    df['Income'] = labels
    return df

# Step 2: State list
all_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

def test_lasso_all_features(n_states=3, random_seed=42):
    random.seed(random_seed)

    sampled_states = random.sample(all_states, n_states)
    print(f"Sampled States: {sampled_states}")

    dfs = []
    sizes = []
    models = []

    for state in sampled_states:
        df = get_state_data(state)
        X = df.dropna()

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, dtype=np.float64)

        dfs.append(X)
        sizes.append(len(X))

        # Train local Graphical Lasso model
        model = GraphicalLasso(alpha=0.01).fit(X.values)
        models.append(model)

    total_samples = sum(sizes)

    # Aggregate local precision matrices (weighted by number of samples)
    agg_precision = sum(n * m.precision_ for n, m in zip(sizes, models)) / total_samples

    # Train Graphical Lasso on the union data
    union_df = pd.concat(dfs, axis=0)
    model_union = GraphicalLasso(alpha=0.01).fit(union_df.values)

    # Compare
    print("\n=== Aggregated Precision Matrix ===")
    print(agg_precision)

    print("\n=== True Precision Matrix ===")
    print(model_union.precision_)

    print("\n=== Absolute Difference ===")
    print(np.abs(agg_precision - model_union.precision_))


all_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]


# --- Core loss function ---
def federation_sensitive_loss(selected_states, sensitive_features=['SEX'], target_feature='Income', alpha=0.01, random_seed=42):
    warnings.filterwarnings(
        "ignore",
        message=".*is_sparse is deprecated.*",
        category=FutureWarning
    )

    random.seed(random_seed)

    dfs = []
    sizes = []
    models = []

    for state in selected_states:
        df = get_state_data(state)
        X = df.dropna()

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, dtype=np.float64)


        dfs.append(X_scaled)
        sizes.append(len(X_scaled))

        model = GraphicalLasso(alpha=alpha).fit(X_scaled.values)
        models.append(model)

    total_samples = sum(sizes)

    # Aggregate local precision matrices (weighted average)
    agg_precision = sum(n * m.precision_ for n, m in zip(sizes, models)) / total_samples

    # --- Now compute the correlations ---
    d = agg_precision.shape[0]
    feature_list = dfs[0].columns.tolist()

    correlation_matrix = -agg_precision / np.sqrt(np.outer(np.diag(agg_precision), np.diag(agg_precision)))

    feature_to_idx = {feature: idx for idx, feature in enumerate(feature_list)}

    sensitive_idx = [feature_to_idx[f] for f in sensitive_features]
    target_idx = feature_to_idx[target_feature]
    non_sensitive_idx = [i for i in range(d) if i not in sensitive_idx and i != target_idx]

    # Extract correlations
    S_T_corrs = [correlation_matrix[s_idx, target_idx] for s_idx in sensitive_idx]
    S_N_corrs = [correlation_matrix[s_idx, n_idx] for s_idx in sensitive_idx for n_idx in non_sensitive_idx]
    N_N_corrs = [correlation_matrix[i, j] for i in non_sensitive_idx for j in non_sensitive_idx if i < j]
    N_T_corrs = [correlation_matrix[n_idx, target_idx] for n_idx in non_sensitive_idx]

    # --- Compute loss components ---
    penalty_S_T = np.sum(np.abs(S_T_corrs)) if S_T_corrs else 0
    penalty_S_N = np.sum(np.abs(S_N_corrs)) if S_N_corrs else 0
    penalty_N_N = np.sum(np.abs(N_N_corrs)) if N_N_corrs else 0
    reward_N_T = np.sum(np.abs(N_T_corrs)) if N_T_corrs else 0

    # --- Define final loss ---
    loss = (penalty_S_T + penalty_S_N + 0.1 * penalty_N_N) - reward_N_T

    return loss


def greedy_federation_search(all_states, sensitive_features=['SEX'], target_feature='Income',
                              federation_size=2, alpha=0.01, random_seed=42):
    random.seed(random_seed)

    # Start from an empty federation
    selected_states = []
    available_states = all_states.copy()

    while len(selected_states) < federation_size:
        best_loss = float('inf')
        best_state = None

        for candidate_state in available_states:
            # Try to add candidate state
            candidate_federation = selected_states + [candidate_state]

            # Evaluate federation loss
            loss = federation_sensitive_loss(candidate_federation, sensitive_features, target_feature, alpha, random_seed)
            print(f'Trying {candidate_state} resulting PFL {np.round(loss, 5)}')
            if loss < best_loss:
                print('PFL Improved')
                best_loss = loss
                best_state = candidate_state

        if best_state is None:
            print("No improvement found, stopping early.")
            break

        # Accept the best state
        selected_states.append(best_state)
        available_states.remove(best_state)

        print(f"Added {best_state}, current federation: {selected_states}, loss: {best_loss}")

    return selected_states





def greedy_federation_search_parallel(all_states, sensitive_features=['SEX'], target_feature='Income',
                                      federation_size=2, alpha=0.01, random_seed=42, n_jobs=-1):
    random.seed(random_seed)
    selected_states = []
    available_states = all_states.copy()

    while len(selected_states) < federation_size:
        best_loss = float('inf')
        best_state = None

        def evaluate_candidate(candidate_state):
            try:
                candidate_federation = selected_states + [candidate_state]
                loss = federation_sensitive_loss(candidate_federation, sensitive_features, target_feature, alpha, random_seed)
                print(f"Trying {candidate_state} resulting PFL {loss}")
                return candidate_state, loss
            except Exception as e:
                print(f"Error with state {candidate_state}: {e}")
                traceback.print_exc()
                return candidate_state, float('inf')  # Penalize failing candidates

        # Run parallel evaluations
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(evaluate_candidate)(state) for state in available_states
        )

        for candidate_state, loss in results:
            if loss < best_loss:
                print(f"PFL improved with {candidate_state}")
                best_loss = loss
                best_state = candidate_state

        if best_state is None:
            print("No improvement found or all candidates failed. Stopping early.")
            break

        selected_states.append(best_state)
        available_states.remove(best_state)
        print(f"Added {best_state}, current federation: {selected_states}, loss: {best_loss}")

    return selected_states





#test_lasso_all_features(n_states=3)
greedy_federation_search(all_states, federation_size=10)
