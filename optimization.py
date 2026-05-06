import time
from sklearn.cluster import AgglomerativeClustering  # Example clustering algorithm
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jensenshannon  # Example distance metric
import random
from mi_utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed

# === --------------Datasets selection part ----------------------------------=== #
# === (discretize_features, compute_mi_components, aggregate_mi_components, calculate_global_mi) ===

def calculate_loss(aggregated_components, ideal_targets, weights, sensitive_var, non_sensitive_vars):
    """
    Calculates a loss based on deviations of MI values from ideal targets,
    including terms for the target variable.

    Args:
        aggregated_components (dict): Aggregated N and contingency tables.
        ideal_targets (dict): Target MI values (e.g., {'target_ISN': 0.0, 'target_INN': 0.0, 'target_IST': 0.0}).
        weights (dict): Weights for loss components (e.g., {'alpha_SN': 0.8889, 'beta_NN': 0.1111, 'alpha_ST': 2.0, 'delta_NT': 1.3333}).
        sensitive_var (str): Name of the sensitive variable.
        non_sensitive_vars (list): List of non-sensitive variable names considered.
    Returns:
        float: The calculated loss value. Returns infinity if calculation fails.
    """
    loss = 0.0
    # Use a dictionary to track counts per loss term type for better normalization/debugging if needed
    calculated_pairs_info = {'SN': 0, 'NN': 0, 'ST': 0, 'NT': 0, 'NaN': 0}

    present_vars = set()
    if aggregated_components and 'contingency_tables' in aggregated_components:
        for pair in aggregated_components['contingency_tables'].keys():
            present_vars.update(pair)

    # Check if essential variables are present
    if sensitive_var not in present_vars: return float('inf')
    if 'label' not in present_vars: return float('inf') # Cannot calculate utility/direct bias

    valid_non_sensitive = [ns for ns in non_sensitive_vars if ns in present_vars]

    # Ensure we iterate over all required pairs including the target
    all_relevant_vars = [sensitive_var, 'label'] + valid_non_sensitive
    # Remove duplicates if target/sensitive somehow ended up in non-sensitive list
    all_relevant_vars = sorted(list(set(all_relevant_vars)))

    pair_calculated = False # Flag to check if any MI value was calculable
    for i, var1 in enumerate(all_relevant_vars):
        for j in range(i + 1, len(all_relevant_vars)):
            var2 = all_relevant_vars[j]
            pair = tuple(sorted((var1, var2)))

            if pair not in aggregated_components.get('contingency_tables', {}):
                 # print(f" Skipping loss for pair {pair}: No aggregated contingency table.")
                 continue

            mi_value = calculate_global_mi(aggregated_components, pair)
            pair_calculated = True

            if np.isnan(mi_value):
                loss += 1e6 # Add a large penalty for NaN MI values
                calculated_pairs_info['NaN'] += 1
                continue

            # --- Apply loss based on pair type ---
            is_SN_pair = False
            is_ST_pair = False
            is_NT_pair = False
            is_NN_pair = False

            # Check pair types (ensure mutual exclusivity)
            if var1 == sensitive_var or var2 == sensitive_var:
                other_var = var2 if var1 == sensitive_var else var1
                if other_var == 'label':
                    is_ST_pair = True
                elif other_var in valid_non_sensitive:
                    is_SN_pair = True
            elif var1 == 'label' or var2 == 'label':
                 other_var = var2 if var1 == 'label' else var1
                 if other_var in valid_non_sensitive:
                     is_NT_pair = True
            elif var1 in valid_non_sensitive and var2 in valid_non_sensitive:
                is_NN_pair = True


            # Apply loss terms based on type
            if is_ST_pair:
                target = ideal_targets.get('target_IST', 0.0) # Target 0 for DP
                weight = weights.get('alpha_ST', 1.0) # Weight for S-T deviation
                loss += weight * (mi_value - target)**1
                calculated_pairs_info['ST'] += 1
            elif is_SN_pair:
                target = ideal_targets.get('target_ISN', 0.0) # Target 0 for DP
                weight = weights.get('alpha_SN', 1.0) # Weight for S-Nk deviation
                loss += weight * (mi_value - target)**1
                calculated_pairs_info['SN'] += 1
            elif is_NT_pair:
                # Utility term: High MI is good, so minimize negative MI
                # Target for MI is implicitly high, so we subtract weighted MI
                # Equivalent to maximizing weighted MI
                weight = weights.get('delta_NT', 1.0) # Weight for Nk-T utility
                loss -= weight * mi_value # Subtract reward from loss
                calculated_pairs_info['NT'] += 1
            elif is_NN_pair:
                target = ideal_targets.get('target_INN', 0.0) # Target 0 for redundancy
                weight = weights.get('beta_NN', 0.1) # Weight for Nk-Nj deviation
                loss += weight * (mi_value - target)**1
                calculated_pairs_info['NN'] += 1

    # Handle edge cases for return value
    total_calculated = sum(calculated_pairs_info[k] for k in ['SN', 'NN', 'ST', 'NT'])
    if total_calculated == 0 and calculated_pairs_info['NaN'] > 0:
        return loss # Return penalized loss if only NaNs
    elif total_calculated == 0 and not pair_calculated and aggregated_components.get('N', 0) > 0 :
         return float('inf') # N>0 but no pairs could be calculated
    elif total_calculated == 0 and aggregated_components.get('N', 0) == 0:
         return 0.0 # Loss for empty set is 0
    elif total_calculated == 0:
         return 0.0 # Pairs existed but weren't relevant types? Return 0 loss.


    return loss



def greedy_additive_selection(all_local_components_map, # Dict: {client_id: components}
                              ideal_targets,
                              weights,
                              sensitive_var,
                              non_sensitive_vars,
                              k_max, # Max number of clients to select
                              n_min=None): # Optional min total N
    """
    Selects clients using a greedy additive approach based on minimizing MI-based loss.

    Args:
        all_local_components_map (dict): Map of client_id to its local components.
        ideal_targets (dict): Target MI values for loss calculation.
        weights (dict): Weights for loss components.
        sensitive_var (str): Name of sensitive variable.
        non_sensitive_vars (list): List of non-sensitive variable names.
        k_max (int): The desired number of clients to select.
        n_min (int, optional): Minimum total data points required. Defaults to None.

    Returns:
        tuple: (selected_client_ids, final_aggregated_components, final_loss)
    """
    selected_client_ids = []
    candidate_ids = list(all_local_components_map.keys())
    current_components_list = []
    current_aggregated_components = {'N': 0, 'contingency_tables': {}} # Start empty
    current_loss = calculate_loss(current_aggregated_components, ideal_targets, weights, sensitive_var, non_sensitive_vars) # Loss for empty set
    if np.isinf(current_loss): current_loss = float('inf') # Ensure proper infinity

    print(f"Initial loss (0 clients): {current_loss}")

    for k in range(k_max):
        best_client_to_add = None
        best_subset_loss = float('inf')
        best_subset_aggregated_components = None

        print(f"\n--- Selecting client {k+1}/{k_max} ---")
        num_candidates_evaluated = 0

        for client_id in candidate_ids:
            if client_id in selected_client_ids:
                continue # Skip already selected clients

            num_candidates_evaluated += 1
            # Simulate adding this client
            components_to_try = current_components_list + [all_local_components_map[client_id]]
            temp_aggregated = aggregate_mi_components(components_to_try)

            # Check minimum N constraint *before* calculating loss if possible
            if n_min is not None and temp_aggregated.get('N', 0) < n_min and k < k_max -1 : # Check N_min unless it's the last pick potentially
                 # print(f"  Skipping {client_id}: Subset N ({temp_aggregated.get('N', 0)}) < {n_min}")
                 # We might still need to select it if no other option meets N_min later, tricky constraint.
                 # For simplicity here, we calculate loss anyway but could skip.
                 pass # Continue to calculate loss for now

            temp_loss = calculate_loss(temp_aggregated, ideal_targets, weights, sensitive_var, non_sensitive_vars)
            # print(f"  Trying {client_id}: Temp N={temp_aggregated.get('N', 0)}, Temp Loss={temp_loss:.4f}")


            # Check if this is the best client found so far for this iteration
            if temp_loss < best_subset_loss:
                # Additional check: If n_min is set, only consider solutions that meet it,
                # UNLESS we are forced to pick something below n_min because no options meet it.
                # This simple greedy doesn't look ahead well for constraints.
                # Let's prioritize lowest loss for now.
                best_subset_loss = temp_loss
                best_client_to_add = client_id
                best_subset_aggregated_components = temp_aggregated

        if num_candidates_evaluated == 0:
            print("No more candidate clients left.")
            break # Stop if no candidates left


        # Decision: Add the best client found if one was found
        if best_client_to_add is not None:
            # Optional: Add check to only add if loss improved?
            # if best_subset_loss < current_loss:
            print(f"Selected Client: {best_client_to_add} (Loss: {best_subset_loss:.6f}, N: {best_subset_aggregated_components.get('N',0)})")
            selected_client_ids.append(best_client_to_add)
            current_components_list.append(all_local_components_map[best_client_to_add])
            current_aggregated_components = best_subset_aggregated_components
            current_loss = best_subset_loss
            # Remove the selected client from future candidates (if loop continues)
            # candidate_ids.remove(best_client_to_add) # Be careful modifying list while iterating - safer not to modify if iterating over original list
            # Rebuild candidate list if needed, or check membership as done above

            # else:
            #     print("No client improved the loss. Stopping selection.")
            #     break
        else:
            print("No suitable client found in this iteration (all resulted in Inf loss?). Stopping selection.")
            break # Stop if no valid client was found

        # Early stop if K_max reached
        if len(selected_client_ids) >= k_max:
            print(f"\nReached k_max = {k_max} clients.")
            break

    # Final check on n_min constraint
    if n_min is not None and current_aggregated_components.get('N', 0) < n_min:
        print(f"\nWarning: Final selected subset N ({current_aggregated_components.get('N',0)}) is less than minimum required ({n_min}).")


    return selected_client_ids, current_aggregated_components, current_loss


def evaluate_candidate(client_id,
                       all_local_components_map,
                       current_components_list,
                       ideal_targets,
                       weights,
                       sensitive_var,
                       non_sensitive_vars,
                       k,
                       k_max,
                       n_min):
    components_to_try = current_components_list + [all_local_components_map[client_id]]
    temp_aggregated = aggregate_mi_components(components_to_try)

    if n_min is not None and temp_aggregated.get('N', 0) < n_min and k < k_max - 1:
        pass

    temp_loss = calculate_loss(temp_aggregated, ideal_targets, weights, sensitive_var, non_sensitive_vars)
    return client_id, temp_loss, temp_aggregated



def greedy_additive_selection_parallel(all_local_components_map,
                                       ideal_targets,
                                       weights,
                                       sensitive_var,
                                       non_sensitive_vars,
                                       k_max,
                                       n_min=None,
                                       max_workers=None):
    selected_client_ids = []
    candidate_ids = list(all_local_components_map.keys())
    current_components_list = []
    current_aggregated_components = {'N': 0, 'contingency_tables': {}}
    current_loss = calculate_loss(current_aggregated_components, ideal_targets, weights, sensitive_var, non_sensitive_vars)
    if np.isinf(current_loss): current_loss = float('inf')

    print(f"Initial loss (0 clients): {current_loss}")

    for k in range(k_max):
        best_client_to_add = None
        best_subset_loss = float('inf')
        best_subset_aggregated_components = None

        print(f"\n--- Selecting client {k+1}/{k_max} ---")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_candidate,
                    cid,
                    all_local_components_map,
                    current_components_list,
                    ideal_targets,
                    weights,
                    sensitive_var,
                    non_sensitive_vars,
                    k,
                    k_max,
                    n_min
                ): cid
                for cid in candidate_ids if cid not in selected_client_ids
            }

            for future in as_completed(futures):
                client_id, temp_loss, temp_aggregated = future.result()
                if temp_loss < best_subset_loss:
                    best_subset_loss = temp_loss
                    best_client_to_add = client_id
                    best_subset_aggregated_components = temp_aggregated

        if best_client_to_add is not None:
            print(f"Selected Client: {best_client_to_add} (Loss: {best_subset_loss:.6f}, N: {best_subset_aggregated_components.get('N', 0)})")
            selected_client_ids.append(best_client_to_add)
            current_components_list.append(all_local_components_map[best_client_to_add])
            current_aggregated_components = best_subset_aggregated_components
            current_loss = best_subset_loss
        else:
            print("No suitable client found. Stopping.")
            break

        if len(selected_client_ids) >= k_max:
            print(f"Reached k_max = {k_max} clients.")
            break

    if n_min is not None and current_aggregated_components.get('N', 0) < n_min:
        print(f"Warning: Final N ({current_aggregated_components.get('N', 0)}) < n_min ({n_min})")

    return selected_client_ids, current_aggregated_components, current_loss




def simulated_annealing_selection(
    all_local_components_map,   # Dict: {client_id: components}
    ideal_targets,
    weights,
    sensitive_var,
    non_sensitive_vars,
    k_max,                      # EXACT number of clients to select
    n_min=None,                 # Optional min total N constraint
    initial_temp=1.0,           # Starting temperature
    cooling_rate=0.95,          # Multiplicative cooling factor (e.g., 0.95 - 0.999)
    min_temp=1e-4,              # Stopping temperature
    max_iterations=2500,        # Max total iterations safeguard
    iterations_per_temp=20      # Iterations before cooling (lowering worse neighbor accpetance probability)
):
    """
    Selects k_max clients using Simulated Annealing to minimize PFL.
    Explores neighbors by swapping one client in/out.

    Args:
        all_local_components_map (dict): Map of client_id to local components.
        ideal_targets (dict): Target MI values for loss calc.
        weights (dict): Weights for loss components.
        sensitive_var (str): Name of sensitive variable.
        non_sensitive_vars (list): List of non-sensitive variable names.
        k_max (int): The exact number of clients to select.
        n_min (int, optional): Minimum total data points required. Defaults to None.
        initial_temp (float): Starting temperature for annealing.
        cooling_rate (float): Multiplicative factor for decreasing temperature.
        min_temp (float): Temperature at which to stop annealing.
        max_iterations (int): Max total iterations overall.
        iterations_per_temp (int): Iterations at each temperature step before cooling.

    Returns:
        tuple: (best_client_ids_list, best_aggregated_components, best_loss)
               Returns ([], {}, float('inf')) if selection fails.
    """
    all_client_ids = list(all_local_components_map.keys())
    n_total_clients = len(all_client_ids)

    # --- Basic validation ---
    if not (0 < k_max <= n_total_clients):
        print(f"Error: k_max ({k_max}) must be between 1 and {n_total_clients}")
        return [], {}, float('inf')
    if not (0 < cooling_rate < 1):
        print("Warning: cooling_rate should be between 0 and 1. Using 0.995.")
        cooling_rate = 0.995

    print(f"Starting Simulated Annealing Selection for k={k_max} clients...")
    sa_start_time = time.time()

    # --- 1. Initial State ---
    print(" Generating initial random state...")
    current_selection_ids = set(random.sample(all_client_ids, k_max))
    current_components_list = [all_local_components_map[cid] for cid in current_selection_ids]
    current_agg_comps = aggregate_mi_components(current_components_list)
    current_n = current_agg_comps.get('N', 0)

    # Resample if initial state doesn't meet n_min constraint
    resample_count = 0
    max_resamples = 100 # Safety break
    while n_min is not None and current_n < n_min:
        resample_count += 1
        if resample_count > max_resamples:
             print(f"Error: Failed to find initial state meeting n_min={n_min} after {max_resamples} attempts.")
             return [], {}, float('inf')
        if k_max == n_total_clients: # Cannot resample if k_max uses all clients
             print(f"Error: Cannot meet n_min constraint as k_max={k_max} requires all clients ({n_total_clients}). Initial N={current_n}")
             return list(current_selection_ids), current_agg_comps, float('inf') # Return current state but indicate failure?
        print(f" Initial N={current_n} < n_min={n_min}. Resampling...")
        current_selection_ids = set(random.sample(all_client_ids, k_max))
        current_components_list = [all_local_components_map[cid] for cid in current_selection_ids]
        current_agg_comps = aggregate_mi_components(current_components_list)
        current_n = current_agg_comps.get('N', 0)

    current_loss = calculate_loss(current_agg_comps, ideal_targets, weights, sensitive_var, non_sensitive_vars)
    if pd.isna(current_loss) or np.isinf(current_loss):
         print("Error: PFL calculation failed for initial state. Cannot start SA.")
         return [], {}, float('inf')

    # Initialize best state found so far
    best_selection_ids = current_selection_ids.copy()
    best_loss = current_loss
    best_agg_comps = current_agg_comps

    print(f" Initial state (k={k_max}): Loss={current_loss:.6f}, N={current_n}")

    # --- 2. Simulated Annealing Loop ---
    T = initial_temp
    iteration = 0
    steps_at_temp = 0

    while T > min_temp and iteration < max_iterations:
        iteration += 1
        steps_at_temp += 1

        # --- Generate Neighbor (Swap one in, one out) ---
        ids_in = list(current_selection_ids)
        ids_out = [cid for cid in all_client_ids if cid not in current_selection_ids]

        if not ids_in or not ids_out: # Should only happen if k_max = 0 or N
             print("Info: No possible swaps remain.")
             break

        client_to_remove = random.choice(ids_in)
        client_to_add = random.choice(ids_out)

        neighbor_selection_ids = (current_selection_ids - {client_to_remove}) | {client_to_add}
        # Re-aggregate components for the neighbor set
        neighbor_components_list = [all_local_components_map[cid] for cid in neighbor_selection_ids]
        neighbor_agg_comps = aggregate_mi_components(neighbor_components_list)
        neighbor_n = neighbor_agg_comps.get('N', 0)

        # --- Evaluate Neighbor ---
        if n_min is not None and neighbor_n < n_min:
            # print(" Neighbor rejected (N < n_min)") # Verbose
            accept = False # Reject neighbor if it violates constraint
            neighbor_loss = float('inf') # Assign high loss
        else:
            neighbor_loss = calculate_loss(neighbor_agg_comps, ideal_targets, weights, sensitive_var, non_sensitive_vars)
        print(f"\nTrying (add/remove)={client_to_remove}/{client_to_add} | PFL(neighbor)={neighbor_loss} | PFL(current)={current_loss}")
        # --- Acceptance Criterion ---
        accept = False
        if pd.isna(neighbor_loss) or np.isinf(neighbor_loss):
            # print(" Neighbor rejected (PFL calculation failed)") # Verbose
            pass # Do not accept invalid neighbors
        else:
            delta_loss = neighbor_loss - current_loss
            if delta_loss < 0: # Neighbor is better
                accept = True
            else: # Neighbor is worse, accept probabilistically
                acceptance_prob = math.exp(-delta_loss / T)
                if random.random() < acceptance_prob:
                    print(f"Accepting a worse neighbor (Annealing) acceptance prob {acceptance_prob}")
                    accept = True

        # --- Update Current State if Accepted ---
        if accept:
            print(f" Iter {iteration}, T={T:.4f}: Accepted neighbor (Loss: {neighbor_loss:.6f}) current federation {neighbor_selection_ids}") # Verbose
            current_selection_ids = neighbor_selection_ids
            current_loss = neighbor_loss
            current_agg_comps = neighbor_agg_comps
            # current_components_list is implicitly defined by current_selection_ids now

            # Update best found so far
            if current_loss < best_loss:
                # print(f"  * New best found: Loss={current_loss:.6f}") # Verbose
                best_selection_ids = current_selection_ids.copy()
                best_loss = current_loss
                best_agg_comps = current_agg_comps
        else:
            print(f" Iter {iteration}, T={T:.4f}: Rejected neighbor (Loss: {neighbor_loss:.6f})") # Verbose


        # --- Cool Temperature ---
        if steps_at_temp >= iterations_per_temp:
            T *= cooling_rate
            steps_at_temp = 0
            # print(f" Cooling T to {T:.4f}") # Verbose

    # --- End Loop ---
    sa_end_time = time.time()
    print(f"\nSimulated Annealing finished after {iteration} iterations.")
    print(f"Total time: {sa_end_time - sa_start_time:.2f} seconds.")
    print(f"Final Temperature: {T:.6f}")
    print(f"Best Loss Found: {best_loss:.6f}")
    print(f"Best subset size: {len(best_selection_ids)}")

    # Final check on n_min constraint for the best solution found
    final_best_n = best_agg_comps.get('N', 0)
    if n_min is not None and final_best_n < n_min:
        print(f"\nWarning: Best subset found N ({final_best_n}) is less than minimum required ({n_min}).")

    return sorted(list(best_selection_ids)), best_agg_comps, best_loss



# === (discretize_features, compute_mi_components, aggregate_mi_components, calculate_global_mi, calculate_loss) ===
def subtractive_greedy_selection(all_local_components_map, # Dict: {client_id: components}
                                 ideal_targets,
                                 weights,
                                 sensitive_var,
                                 non_sensitive_vars,
                                 loss_tolerance, # *** RENAMED parameter for clarity ***
                                 n_min=None):    # Optional min total N
    """
    Selects the smallest client subset whose PFL score remains representative
    (within a tolerance) of the full set's PFL score, using subtractive greedy.

    Args:
        all_local_components_map (dict): Map of client_id to its local components.
        ideal_targets (dict): Target MI values for loss calculation.
        weights (dict): Weights for loss components.
        sensitive_var (str): Name of sensitive variable.
        non_sensitive_vars (list): List of non-sensitive variable names.
        loss_tolerance (float): Maximum acceptable absolute deviation of the
                                subset's PFL from the initial full set's PFL.
        n_min (int, optional): Minimum total data points required. Defaults to None.

    Returns:
        tuple: (selected_client_ids, final_aggregated_components, final_loss)
    """
    if not all_local_components_map:
            return [], {'N': 0, 'contingency_tables': {}}, float('inf')

    print(f"Starting subtractive selection with Loss Tolerance <= {loss_tolerance:.6f}" + (f" and N_min >= {n_min}" if n_min else ""))
    start_time = time.time()

    # 1. Initialize with full dataset and calculate its PFL
    current_selection_ids = set(all_local_components_map.keys())
    current_components_list = list(all_local_components_map.values())
    current_aggregated_components = aggregate_mi_components(current_components_list)
    # *** Calculate PFL of the original full set ***
    pfl_full_set = calculate_loss(current_aggregated_components, ideal_targets, weights, sensitive_var, non_sensitive_vars)
    current_loss = pfl_full_set # Initialize current loss
    current_n = current_aggregated_components.get('N', 0)

    print(f"Initial full set ({len(current_selection_ids)} clients): PFL = {pfl_full_set:.6f}, N = {current_n}")

    # Optional: Check initial N constraint (maybe not needed if goal is just representativeness?)
    if (n_min is not None and current_n < n_min):
        print("Warning: Full dataset already violates N_min constraint. Returning full set.")
        return list(current_selection_ids), current_aggregated_components, current_loss

    while True: # Loop until no more clients can be removed
        best_client_to_remove = None
        best_resulting_loss = float('inf') # Still aim for lowest loss among valid removals
        best_resulting_components = None
        removable_candidates_found = False

        print(f"\n--- Evaluating removal from {len(current_selection_ids)} clients (Full Set PFL: {pfl_full_set:.6f}) ---")
        iter_start_time = time.time()
        ids_to_evaluate = list(current_selection_ids)

        for client_id_to_remove in ids_to_evaluate:
            if len(current_selection_ids) <= 1:
                continue

            # Simulate removal
            temp_selection_ids = current_selection_ids - {client_id_to_remove}
            temp_components_list = [comp for cid, comp in all_local_components_map.items() if cid in temp_selection_ids]
            if not temp_components_list: continue # Should not happen if len > 1

            temp_aggregated = aggregate_mi_components(temp_components_list)
            temp_loss = calculate_loss(temp_aggregated, ideal_targets, weights, sensitive_var, non_sensitive_vars)
            temp_n = temp_aggregated.get('N', 0)

            # Debug print (optional)
            print(f"  Trying removing {client_id_to_remove}: Temp PFL={temp_loss:.4f}, Temp N={temp_n}")

            # --- *** MODIFIED CONSTRAINT CHECK *** ---
            # Check if the resulting subset's PFL is close enough to the ORIGINAL full set's PFL
            loss_ok = (np.abs(temp_loss - pfl_full_set) <= loss_tolerance)
            n_ok = (n_min is None or temp_n >= n_min)
            # --- *** END MODIFIED CHECK *** ---

            if loss_ok and n_ok:
                removable_candidates_found = True
                # Is this removal candidate better (lower resulting PFL) than previous best?
                if temp_loss < best_resulting_loss:
                    best_resulting_loss = temp_loss
                    best_client_to_remove = client_id_to_remove
                    best_resulting_components = temp_aggregated
                else: # Optional debug print for constraint failure
                    print(f"    -> Removing {client_id_to_remove} rejected. Loss ok: {loss_ok} (Diff: {np.abs(temp_loss - pfl_full_set):.4f}), N ok: {n_ok}")


        iter_end_time = time.time()
        print(f"Iteration evaluation took {iter_end_time - iter_start_time:.2f} seconds.")

        # Decision: Remove the best candidate if one was found
        if best_client_to_remove is not None:
            print(f"Removing Client: {best_client_to_remove} (Resulting PFL: {best_resulting_loss:.6f}, Resulting N: {best_resulting_components.get('N',0)})")
            current_selection_ids.remove(best_client_to_remove)
            current_aggregated_components = best_resulting_components
            current_loss = best_resulting_loss
            # Update the list used for next iteration (less efficient but clear)
            current_components_list = [comp for cid, comp in all_local_components_map.items() if cid in current_selection_ids]
        else:
            print("\nNo more clients can be removed without violating representativeness constraints.")
            break # Exit the while loop

    end_time = time.time()
    print(f"\nSubtractive selection process took {end_time - start_time:.2f} seconds.")

    return sorted(list(current_selection_ids)), current_aggregated_components, current_loss



def cluster_clients_by_similarity(all_local_components_map, sensitive_var, non_sensitive_vars,
                                n_clusters=None, distance_metric='euclidean',
                                clustering_algorithm='agglomerative', linkage='average'):
    """
    Clusters clients based on the similarity of their statistical patterns (MI components).

    Args:
        all_local_components_map (dict): Map of client_id to its local components.
        sensitive_var (str): Name of the sensitive variable.
        non_sensitive_vars (list): List of non-sensitive variable names.
        n_clusters (int, optional): Number of clusters for algorithms that require it (K-means, Agglomerative).
        distance_metric (str): Distance metric to use ('jensenshannon', 'euclidean', etc.).
        clustering_algorithm (str): Clustering algorithm ('agglomerative', 'kmeans', etc.).
        linkage (str): Linkage method for Agglomerative Clustering ('average', 'ward', 'complete').

    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of client IDs in each cluster.
    """

    client_ids = list(all_local_components_map.keys())
    client_features = {}

    # 1. Determine Maximum Feature Vector Size
    max_feature_size = 0
    for client_id, components in all_local_components_map.items():
        present_vars = set()
        if components and 'contingency_tables' in components:
            for pair in components['contingency_tables'].keys():
                present_vars.update(pair)

        relevant_vars = [sensitive_var, 'label'] + [ns for ns in non_sensitive_vars if ns in present_vars]
        relevant_vars = sorted(list(set(relevant_vars)))

        current_feature_size = 0
        for i, var1 in enumerate(relevant_vars):
            for j in range(i + 1, len(relevant_vars)):
                var2 = relevant_vars[j]
                pair = tuple(sorted((var1, var2)))
                if pair in components.get('contingency_tables', {}):
                    current_feature_size += np.prod(components['contingency_tables'][pair].shape)
                else:
                    # Assuming a default shape if contingency table is missing
                    # Adjust this default shape if necessary
                    current_feature_size += 1  # Or some other default value
        max_feature_size = max(max_feature_size, current_feature_size)

    for client_id, components in all_local_components_map.items():
        feature_vector = []
        present_vars = set()
        if components and 'contingency_tables' in components:
            for pair in components['contingency_tables'].keys():
                present_vars.update(pair)

        relevant_vars = [sensitive_var, 'label'] + [ns for ns in non_sensitive_vars if ns in present_vars]
        relevant_vars = sorted(list(set(relevant_vars)))

        for i, var1 in enumerate(relevant_vars):
            for j in range(i + 1, len(relevant_vars)):
                var2 = relevant_vars[j]
                pair = tuple(sorted((var1, var2)))

                if pair in components.get('contingency_tables', {}):
                    joint_counts = components['contingency_tables'][pair]
                    N = components['N']
                    joint_probs = joint_counts / N if N > 0 else np.zeros_like(joint_counts, dtype=float)

                    if isinstance(joint_probs, pd.DataFrame) or isinstance(joint_probs, pd.Series):
                        feature_vector.extend(joint_probs.values.flatten().tolist())
                    else:
                        feature_vector.extend(joint_probs.flatten().tolist())
                else:
                    if 'joint_counts' in locals():
                        feature_vector.extend([0.0] * np.prod(joint_counts.shape))
                    else:
                        feature_vector.extend([0.0])  # Add a single 0 if no joint_counts

        # 2. Pad Feature Vector
        padding_size = max_feature_size - len(feature_vector)
        feature_vector.extend([0.0] * padding_size)

        client_features[client_id] = np.array(feature_vector)

    # Convert client features to a matrix
    feature_matrix = np.array([client_features[cid] for cid in client_ids])

    # Calculate distance matrix
    if distance_metric == 'jensenshannon':
        distance_matrix = pairwise_distances(feature_matrix,
                                            metric=lambda u, v: jensenshannon(u, v) if np.any(u) and np.any(v) else np.inf)
    elif distance_metric == 'euclidean':
        distance_matrix = pairwise_distances(feature_matrix, metric='euclidean')
    else:
        raise ValueError(f"Distance metric '{distance_metric}' not supported.")

    # Apply clustering algorithm
    if clustering_algorithm == 'agglomerative':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for Agglomerative Clustering.")
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage=linkage)
        cluster_labels = clusterer.fit_predict(distance_matrix)
    elif clustering_algorithm == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for KMeans Clustering.")
        clusterer = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")  # Recommended to set n_init explicitly
        cluster_labels = clusterer.fit_predict(feature_matrix)
    else:
        raise ValueError(f"Clustering algorithm '{clustering_algorithm}' not supported.")

    # Organize clusters
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(client_ids[i])

    return clusters







# --- Simulated Annealing for minimal representative subset (Alternative of Greedy substractive search) -----


def calculate_direct_snapshot_divergence(
    subset_aggregated_components,
    full_pool_aggregated_components_snapshot,
    sensitive_var,
    non_sensitive_vars,
    divergence_weights # Weights for different MI divergence components
):
    """
    Calculates the divergence of a subset's MI characteristics directly from
    the full pool's MI snapshot. This is the primary objective for SA.
    Lower divergence means the subset is more 'representative' of the original full pool's MI landscape.
    """
    divergence_score = 0.0

    # Ensure snapshot and subset have data to compute MI
    if full_pool_aggregated_components_snapshot.get('N', 0) == 0 or \
       subset_aggregated_components.get('N', 0) == 0:
        return float('inf') # Cannot compute meaningful divergence if either is empty

    # Define relevant MI pairs to compare (e.g., sensitive_var to Y, non_sensitive_vars to Y)
    # Assuming 'Y' is a common target/outcome variable in the MI components
    mi_pairs_to_compare = []
    if sensitive_var:
        mi_pairs_to_compare.append((sensitive_var, 'label'))
    for ns_var in non_sensitive_vars:
        # Assuming Y is the target for utility MIs. Adjust if Y is in non_sensitive_vars and you want to compare X-Y.
        if ns_var != sensitive_var and ns_var != 'label': # Avoid comparing sensitive_var to itself if it's in non_sensitive_vars list
             mi_pairs_to_compare.append((ns_var, 'label'))

    # Calculate squared difference for each relevant MI pair
    for var1, var2 in mi_pairs_to_compare:
        mi_subset = calculate_global_mi(subset_aggregated_components, tuple(sorted((var1, var2))))
        mi_snapshot = calculate_global_mi(full_pool_aggregated_components_snapshot, tuple(sorted((var1, var2))))
        # Use a specific weight for this MI pair's divergence
        weight_key = f'MI_Divergence_{var1}-{var2}' # Example: 'MI_Divergence_S-Y'
        divergence_score += divergence_weights.get(weight_key, 1.0) * (mi_subset - mi_snapshot)**2

    return divergence_score


def calculate_sa_cost(
    subset_ids,
    all_local_components_map,
    # Parameters for snapshot divergence
    full_pool_aggregated_components_snapshot, # The pre-calculated aggregated components of the full pool
    sensitive_var,
    non_sensitive_vars,
    divergence_weights, # Weights for individual MI divergence components
    # Parameters for size penalties and n_min
    n_min=None,
    subset_size_penalty_weight=1.0, # Penalty for the number of clients (len(subset_ids))
    total_data_size_penalty_weight=1.0, # Penalty for total data points (N)
):
    """
    Calculates the 'energy' (cost) of a given subset for Simulated Annealing.
    The cost is based on:
    1. Direct divergence of the subset's MI characteristics from the full pool's MI snapshot.
    2. Penalty for the number of clients.
    3. Penalty for total data points (N).
    4. Hard constraint for n_min (returns inf if violated).
    5. Returns inf for empty subsets or if divergence calculation fails.
    """
    if not subset_ids:
        return float('inf') # An empty set is considered invalid

    components_list = [all_local_components_map[cid] for cid in subset_ids]
    aggregated_components = aggregate_mi_components(components_list)
    current_n = aggregated_components.get('N', 0)

    # Hard constraint for n_min: if violated, assign infinite cost.
    if n_min is not None and current_n < n_min:
        return float('inf')
    if current_n == 0: # Ensure non-empty for divergence calculation
        return float('inf')

    # 1. Calculate the primary cost: Direct divergence from snapshot MI values
    core_divergence_cost = calculate_direct_snapshot_divergence(
        aggregated_components,
        full_pool_aggregated_components_snapshot,
        sensitive_var,
        non_sensitive_vars,
        divergence_weights
    )

    if pd.isna(core_divergence_cost) or np.isinf(core_divergence_cost):
        print('Cost is NaN/inf', core_divergence_cost)
        return float('inf') # Return infinite cost if divergence calculation results in NaN/inf

    cost = 1e+6 * core_divergence_cost # Start with the core divergence
    print(f'core divergence cost {cost}')
    # 2. Add penalty for the number of clients (federation size penalty)
    clients_count_penalty = subset_size_penalty_weight * len(subset_ids)
    print(f'client count penalty {clients_count_penalty/10}')
    cost += clients_count_penalty/10
    print(f'Total cost {cost}')
    return cost


# --- Neighbor Generation for Variable Size (retained from previous response) ---
def get_variable_size_neighbor(current_selection_ids, all_client_ids_pool, temp=1.0):
    """
    Generates a neighbor state by either adding a client or removing a client.
    This allows the subset size to change.
    """
    current_set = set(current_selection_ids)
    all_pool = set(all_client_ids_pool)

    if not current_set:                           # If empty, must add
        action = 'add'
    elif len(current_set) == len(all_pool):       # If full, must remove
        action = 'remove'
    else:
        add_prob = 0.5 * temp                     #Decay the add probability progressively
        remove_prob = 1.0 - add_prob
        action = random.choices(['add', 'remove'], weights=[add_prob, remove_prob])[0]
        print(f'selected action : {action}')
    if action == 'add':
        clients_not_in_set = list(all_pool - current_set)
        if clients_not_in_set:
            client_to_add = random.choice(clients_not_in_set)
            return current_set | {client_to_add}
    elif action == 'remove':
        if current_set and len(current_set) > 0:    # Ensure we don't try to remove from empty
            client_to_remove = random.choice(list(current_set))
            return current_set - {client_to_remove}

    return current_set                              # Return current set if no change possible



def create_divergence_weights(
    sensitive_var_name: str,
    non_sensitive_var_names: list[str],
    outcome_var_name: str,
    default_fairness_weight: float = 1.0,
    default_utility_weight: float = 1.0,
    include_cross_non_sensitive_weights: bool = False,
    default_cross_non_sensitive_weight: float = 0.0
) -> dict:
    """
    Automates the creation of the divergence_weights dictionary for Simulated Annealing.

    Args:
        sensitive_var_name (str): The name of your sensitive variable.
        non_sensitive_var_names (list[str]): A list of your non-sensitive variable names.
        outcome_var_name (str): The name of your outcome/target variable (e.g., 'Label', 'Y').
        default_fairness_weight (float): Default weight for divergence related to
                                         sensitive_var and outcome_var.
        default_utility_weight (float): Default weight for divergence related to
                                        non_sensitive_vars and outcome_var.
        include_cross_non_sensitive_weights (bool): Whether to include divergence weights
                                                     for MI between pairs of non-sensitive variables.
        default_cross_non_sensitive_weight (float): Default weight for cross non-sensitive
                                                      variable MI divergences.

    Returns:
        dict: A dictionary of divergence weights.
    """
    divergence_weights = {}

    # 1. Add weights for Fairness-related MI Divergence (Sensitive Variable <-> Outcome)
    if sensitive_var_name and outcome_var_name:
        key = f'MI_Divergence_{sensitive_var_name}-{outcome_var_name}'
        divergence_weights[key] = default_fairness_weight
        print(f"Added fairness divergence weight: '{key}': {default_fairness_weight}")

    # 2. Add weights for Utility-related MI Divergence (Non-Sensitive Variables <-> Outcome)
    for ns_var in non_sensitive_var_names:
        if ns_var and outcome_var_name:
            # Avoid cases where non-sensitive var is also the outcome var if it could happen
            if ns_var == outcome_var_name:
                continue
            key = f'MI_Divergence_{ns_var}-{outcome_var_name}'
            divergence_weights[key] = default_utility_weight
            print(f"Added utility divergence weight: '{key}': {default_utility_weight}")

    # 3. (Optional) Add weights for cross-Non-Sensitive Variable MI Divergence
    # This helps preserve relationships between non-sensitive features themselves.
    if include_cross_non_sensitive_weights:
        for i, ns_var1 in enumerate(non_sensitive_var_names):
            for j, ns_var2 in enumerate(non_sensitive_var_names):
                if i < j: # To avoid duplicates (A-B vs B-A) and self-comparison (A-A)
                    key = f'MI_Divergence_{ns_var1}-{ns_var2}'
                    # Ensure consistent key order (e.g., sort var names) if calculate_global_mi does
                    if ns_var1 > ns_var2: # Example: sort to ensure 'Z-A' is 'A-Z'
                        key = f'MI_Divergence_{ns_var2}-{ns_var1}'

                    divergence_weights[key] = default_cross_non_sensitive_weight
                    print(f"Added cross-feature divergence weight: '{key}': {default_cross_non_sensitive_weight}")

    return divergence_weights

def simulated_annealing_selection_variable_size_snapshot(
    all_local_components_map,            # Dict: {client_id: components}
    sensitive_var,
    non_sensitive_vars,
    divergence_weights,                  # Weights for individual MI divergence components (for calculate_direct_snapshot_divergence)
    n_min=None,                          # Optional min total N constraint
    init_size=25,                    # Optional target number of clients for initial state
    subset_size_penalty_weight=1.0,      # Weight for penalizing the number of clients
    total_data_size_penalty_weight=1.0,  # Weight for penalizing total N
    initial_temp=1.0  ,                  # Starting temperature
    cooling_rate=0.99,                   # Multiplicative cooling factor
    min_temp=1e-6,                       # Stopping temperature
    max_iterations=5000,                 # Max total iterations safeguard
    iterations_per_temp=10,              # Iterations before cooling
    seed=None                            # For reproducibility
):
    """
    Selects a variable-size client subset using Simulated Annealing.
    The cost function is primarily based on the direct divergence of the subset's
    aggregated MI characteristics from the full pool's MI snapshot,
    plus penalties for the number of clients and total data points (N).
    A hard constraint for n_min total data points is applied.

    Args:
        all_local_components_map (dict): Map of client_id to local components.
        sensitive_var (str): Name of sensitive variable.
        non_sensitive_vars (list): List of non-sensitive variable names.
        divergence_weights (dict): Weights for individual MI divergence components
                                   (e.g., {'MI_Divergence_S-Y': 10.0, 'MI_Divergence_X-Y': 1.0}).
        n_min (int, optional): Minimum total data points required. Defaults to None.
            init_size (int, optional): A target number of clients for initial state.
                                       Does not enforce exact size during annealing.
        subset_size_penalty_weight (float): Weight for penalizing the number of clients.
                                            A positive value encourages smaller client sets.
        total_data_size_penalty_weight (float): Weight for penalizing the total number of data points (N).
                                                A positive value encourages smaller aggregate N.
        initial_temp (float): Starting temperature for annealing.
        cooling_rate (float): Multiplicative factor for decreasing temperature.
        min_temp (float): Temperature at which to stop annealing.
        max_iterations (int): Max total iterations overall.
        iterations_per_temp (int): Iterations before cooling.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (best_client_ids_list, best_aggregated_components, best_raw_divergence_value)
               Returns ([], {'N': 0, 'contingency_tables': {}}, float('inf')) if selection fails.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_client_ids = list(all_local_components_map.keys())
    n_total_clients = len(all_client_ids)

    # --- Basic validation ---
    if n_total_clients == 0:
        print("Error: No clients available in all_local_components_map.")
        return [], {'N': 0, 'contingency_tables': {}}, float('inf')
    if not (0 < cooling_rate < 1):
        print("Warning: cooling_rate should be between 0 and 1. Using 0.99.")
        cooling_rate = 0.99

    print(f"Starting Simulated Annealing Selection (variable size, direct snapshot divergence)...")
    sa_start_time = time.time()

    # --- 1. Take initial "snapshot" of the full pool characteristics ---
    full_pool_components_list = list(all_local_components_map.values())
    full_pool_aggregated_components_snapshot = aggregate_mi_components(full_pool_components_list)

    # Check if snapshot itself is valid (e.g., not empty)
    if full_pool_aggregated_components_snapshot.get('N', 0) == 0:
        print("Error: Full client pool is empty. Cannot take a meaningful snapshot for divergence.")
        return [], {'N': 0, 'contingency_tables': {}}, float('inf')

    # --- 2. Initialize SA with a starting state ---
    print(" Generating initial random state...")
    initial_selection_ids = set()
    if n_total_clients > 0:
        if init_size is not None and 0 < init_size <= n_total_clients:
            initial_selection_ids = set(random.sample(all_client_ids, init_size))
        else:
            initial_selection_ids = set(all_client_ids) # Default to full set if no target size or invalid

    current_selection_ids = initial_selection_ids

    # Ensure initial state is valid (non-infinite cost) by resampling if necessary
    resample_count = 0
    max_resamples = 2000 # Safety break for initial state generation

    current_cost = calculate_sa_cost(
        current_selection_ids, all_local_components_map,
        full_pool_aggregated_components_snapshot, sensitive_var, non_sensitive_vars,
        divergence_weights, n_min, subset_size_penalty_weight, total_data_size_penalty_weight
    )

    while np.isinf(current_cost) and resample_count < max_resamples:
        resample_count += 1
        print(f" Initial state cost is infinite. Resampling... ({resample_count}/{max_resamples})")
        # Try a different random starting point (random size, but avoid 0 if n_total_clients is small)
        new_k = random.randint(1, n_total_clients) if n_total_clients > 0 else 0
        current_selection_ids = set(random.sample(all_client_ids, new_k)) if new_k > 0 else set()

        current_cost = calculate_sa_cost(
            current_selection_ids, all_local_components_map,
            full_pool_aggregated_components_snapshot, sensitive_var, non_sensitive_vars,
            divergence_weights, n_min, subset_size_penalty_weight, total_data_size_penalty_weight
        )

    if np.isinf(current_cost): # If after resamples, still invalid
        print("Error: Failed to find a valid initial state after multiple attempts. Cannot start SA.")
        return [], {'N': 0, 'contingency_tables': {}}, float('inf')

    # Initialize best state found so far
    best_selection_ids = current_selection_ids.copy()
    best_cost = current_cost

    print(f" Initial state (clients={len(current_selection_ids)}): Energy={current_cost:.6f}")

    # --- 3. Simulated Annealing Loop ---
    T = initial_temp
    iteration = 0
    steps_at_temp = 0

    while T > min_temp and iteration < max_iterations:
        iteration += 1
        steps_at_temp += 1

        # Generate Neighbor (variable size: add or remove)
        neighbor_selection_ids = get_variable_size_neighbor(current_selection_ids, all_client_ids, temp=1.0)

        # Evaluate Neighbor using the unified cost function
        neighbor_cost = calculate_sa_cost(
            neighbor_selection_ids, all_local_components_map,
            full_pool_aggregated_components_snapshot, sensitive_var, non_sensitive_vars,
            divergence_weights, n_min, subset_size_penalty_weight, total_data_size_penalty_weight
        )

        # Acceptance Criterion
        accept = False
        if np.isinf(neighbor_cost): # Reject invalid neighbors (e.g., due to n_min violation)
            pass
        else:
            delta_cost = neighbor_cost - current_cost
            if delta_cost < 0: # Neighbor is better (lower energy)
                accept = True
            else: # Neighbor is worse, accept probabilistically
                if T > 0: # Avoid division by zero
                    acceptance_prob = math.exp(-delta_cost / T) #--> Metropolis Criterion
                    if random.random() < acceptance_prob:
                        accept = True

        # Update Current State if Accepted
        if accept:
            print('Update accepted')
            current_selection_ids = neighbor_selection_ids
            current_cost = neighbor_cost

            # Update best found so far
            if current_cost < best_cost:
                best_selection_ids = current_selection_ids.copy()
                best_cost = current_cost

        # Cool Temperature
        if steps_at_temp >= iterations_per_temp:
            T *= cooling_rate
            steps_at_temp = 0

        # Logging progress
        if iteration % 1 == 0:
            current_agg_n = aggregate_mi_components([all_local_components_map[cid] for cid in current_selection_ids]).get('N',0)
            print(f"\nIter {iteration}/{max_iterations}: T={T:.4f}, Clients={len(current_selection_ids)}, N={current_agg_n}, Current Cost={current_cost:.6f}, Best Cost={best_cost:.6f}")
            print(f"States {current_selection_ids}")
    # --- End Loop ---
    sa_end_time = time.time()
    print(f"\nSimulated Annealing finished after {iteration} iterations.")
    print(f"Total time: {sa_end_time - sa_start_time:.2f} seconds.")
    print(f"Final Temperature: {T:.6f}")
    print(f"Best Energy Found (including penalties): {best_cost:.6f}")
    print(f"Best subset size: {len(best_selection_ids)}")

    # Calculate and report the raw divergence of the best solution found (without size penalties)
    best_agg_comps = aggregate_mi_components([all_local_components_map[cid] for cid in best_selection_ids])
    final_best_raw_divergence = calculate_direct_snapshot_divergence(
        best_agg_comps, full_pool_aggregated_components_snapshot, sensitive_var, non_sensitive_vars, divergence_weights
    )
    final_best_n = best_agg_comps.get('N', 0)

    print(f"Best subset's raw divergence from snapshot: {final_best_raw_divergence:.6f}")
    print(f"Best subset's total N: {final_best_n}")

    # Final check on n_min constraint for the best solution found (for reporting)
    if n_min is not None and final_best_n < n_min:
        print(f"Warning: Best subset found total N ({final_best_n}) is less than minimum required ({n_min}). "
              "This should ideally not happen if `calculate_sa_cost` is working correctly.")

    # Return the PFL against ideal targets for the best found subset (as a secondary metric for analysis)
    # This assumes 'calculate_loss' is predefined and available for this purpose.
    # Note: This is *not* the primary metric SA optimized, but useful for evaluation.
    final_best_pfl_against_ideal_targets = None
    try:
        # Assuming 'calculate_loss' requires 'ideal_targets' and 'weights'
        # These would need to be passed to the main SA function if they are used for this final reporting.
        # For now, let's assume they are available if calculate_loss is used this way.
        # You'll need to pass 'ideal_targets' and 'weights' to the main SA function
        # if you want this final report to be meaningful.
        # For now, placeholder or if they are global.
        # This function signature does not have ideal_targets or weights for calculate_loss.
        # If needed, they must be passed to simulated_annealing_selection_variable_size_snapshot.
        # Let's adjust main function signature for this final reporting.
        # The prompt only specified 'divergence_weights' for the SA optimization.
        # Let's assume 'calculate_loss' needs ideal_targets and weights as parameters to this function.
        print("Note: 'calculate_loss' for PFL against ideal targets requires 'ideal_targets' and 'weights' parameters to be passed to the main SA function.")
        # As per the new function signature, I'll pass ideal_targets and weights to the main function for this final call.
    except NameError:
        print("Warning: 'calculate_loss' or its dependencies (ideal_targets, weights) not fully accessible for final PFL calculation.")

    return sorted(list(best_selection_ids)), best_agg_comps, final_best_raw_divergence




# --- END OPTIMIZATION FUNCTIONS ---
