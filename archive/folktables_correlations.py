import folktables
import pandas as pd
import numpy as np
from folktables import ACSDataSource
from dppy.finite_dpps import FiniteDPP
from scipy.spatial.distance import pdist, squareform


all_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']


# Path to ACSIncome datasets
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

# Dictionary to store correlation matrices
correlation_matrices = {}
dataset_sizes = {}
# Process each state's ACSIncome dataset


def Compute_correlation_matrix(state) :
    acs_data = data_source.get_data(states=state, download=True)

    # Extract ACSIncome features and labels
    features, label, _ = folktables.ACSIncome.df_to_numpy(acs_data)

    # Convert to DataFrame for correlation computation
    df = pd.DataFrame(features, columns=folktables.ACSIncome.features)
    df['label'] = label  # Append label column

    # Compute correlation matrix
    corr_matrix = df.corr()
    return corr_matrix, features.size

for state in all_states:
    print(state)
    correlation_matrices[state], dataset_sizes[state] = Compute_correlation_matrix([state])



# Example: Print correlation matrix for California

corr_al_ar = (dataset_sizes['AL'] * correlation_matrices['AL'] + dataset_sizes['AR'] * correlation_matrices['AR'])/(dataset_sizes['AL'] + dataset_sizes['AR'])

print(corr_al_ar)
print(Compute_correlation_matrix(['AL', 'AR']))



def dpp_select_optimal_datasets(correlation_matrices, k):
    """
    Selects an optimal subset of datasets using Determinantal Point Processes (DPP).
    - correlation_matrices: List of correlation matrices.
    - k: Number of datasets to select.
    Returns: Indices of selected datasets.
    """
    # Compute similarity kernel (Gaussian RBF based on correlation distances)
    # Flatten each correlation matrix into a 1D vector
    flattened_matrices = np.array([correlation_matrices[mat].to_numpy().flatten() for mat in correlation_matrices])

    # Compute pairwise distances
    L = np.exp(-squareform(pdist(flattened_matrices, metric='euclidean')))


    # DPP Sampling
    dpp = FiniteDPP(kernel_type='likelihood', L=L)
    dpp.sample_exact_k_dpp(size=k)

    return dpp.list_of_samples[0]  # First sample

dpp_select_optimal_datasets(correlation_matrices, 10)



def correlation_score(correlation_matrix, sensitive_attr_idx, target_attr_idx, non_sensitive_attr_indices):
    """
    Compute the correlation score based on direct and indirect correlations.

    Parameters:
    - correlation_matrix: 2D numpy array of correlation coefficients between features.
    - sensitive_attr_idx: Index of the sensitive attribute in the matrix (e.g., 'SEX').
    - target_attr_idx: Index of the target attribute in the matrix.
    - non_sensitive_attr_indices: List of indices corresponding to non-sensitive attributes.

    Returns:
    - score: The computed correlation score.
    """

    # Step 1: Direct correlation between sensitive attribute and target
    direct_corr = correlation_matrix[sensitive_attr_idx][target_attr_idx]

    # Step 2: Indirect correlations
    indirect_corr = 0
    for non_sensitive_idx in non_sensitive_attr_indices:
        # Correlation of non-sensitive attribute with sensitive attribute
        corr_sens = correlation_matrix[sensitive_attr_idx][non_sensitive_idx]

        # Correlation of non-sensitive attribute with target
        corr_non_sens_target = correlation_matrix[non_sensitive_idx][target_attr_idx]

        # Combine the two correlations, weighted by the correlation with the target
        indirect_corr += corr_sens * corr_non_sens_target

    # Step 3: Total score computation
    # Combine direct and indirect correlations and account for the sign of each
    score = direct_corr + indirect_corr

    return score



columns_to_use = correlation_matrices['CA'].columns.difference(['label', 'SEX']).to_numpy()
corr_scores = {}
for mat in correlation_matrices :

    score = correlation_score(correlation_matrices[mat], 'SEX', 'label', ['COW'])
    corr_scores[mat] = score
    print(f"State {mat}, Correlation score: {score}")


for state1 in all_states :
    for state2 in all_states :
        if state1 != state2 :
            comb_score = correlation_score(Compute_correlation_matrix([state1, state2])[0], 'SEX', 'label', ['COW'])
            avg_score1 = (dataset_sizes[state1] * corr_scores[state1] + dataset_sizes[state2] * corr_scores[state2])/(dataset_sizes[state1] + dataset_sizes[state2])
            avg_score2 = (corr_scores[state1] + corr_scores[state2])/2
            print(f'CORR_SCORE({state1} U {state2}) : {comb_score}, AVG : { avg_score1 } diff : {comb_score - avg_score1}')
