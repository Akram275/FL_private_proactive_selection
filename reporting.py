# reporting.py
# Function to report selection results

import numpy as np
# Import necessary function from the other module
from mi_utils import calculate_global_mi

def report_selection_results(selected_ids, final_agg_components, final_loss, algorithm_name):
    """Prints a summary report for a selection algorithm's results."""

    print(f"\n--- Results for {algorithm_name} ---")
    print(f"Selected Client IDs ({len(selected_ids)}): {selected_ids}")
    print(f"Final Aggregated N: {final_agg_components.get('N', 0)}")
    print(f"Final Loss Score: {final_loss:.6f}")

    print("\nCalculating MI matrix for the final selected subset...")
    final_mi_values = {}
    final_pairs = list(final_agg_components.get('contingency_tables', {}).keys())
    if not final_pairs:
         print("No contingency tables in final aggregated components.")
    else:
        final_vars_present = set()
        for pair in final_pairs:
            final_vars_present.update(pair)
        print(f"Variables present in final subset: {sorted(list(final_vars_present))}")

        # Recalculate MI values for the final selected subset
        # Iterate through unique pairs present in the keys
        unique_vars = sorted(list(final_vars_present))
        calculated_pairs_report = set()
        for i, var1 in enumerate(unique_vars):
             for j in range(i + 1, len(unique_vars)):
                  var2 = unique_vars[j]
                  pair = tuple(sorted((var1, var2)))
                  # Check if this pair's contingency table exists before calculating MI
                  if pair in final_pairs:
                       mi_value = calculate_global_mi(final_agg_components, pair)
                       final_mi_values[pair] = mi_value
                       print(f"  Final MI({pair[0]}, {pair[1]}): {mi_value if not np.isnan(mi_value) else 'NaN':.6f}")
                       calculated_pairs_report.add(pair)
                  # else: Pair combination not present in aggregated tables

        # Check if any expected pairs were missing
        expected_pairs = {tuple(sorted((v1, v2))) for idx1, v1 in enumerate(unique_vars) for idx2, v2 in enumerate(unique_vars) if idx1 < idx2}
        missing_pairs = expected_pairs - calculated_pairs_report
        if missing_pairs:
             print(f"  Note: MI could not be calculated for some pairs: {missing_pairs}")


    print("-" * (len(algorithm_name) + 15))
