"""
Generalized Optimal Federation Selection for FolkTables Datasets

This module provides a unified interface for computing optimal federations
of size k for any FolkTables prediction task.

Usage:
    python run_optimal_federation.py --task ACSIncome --k 5 --epsilon 0.05
    python run_optimal_federation.py --task ACSEmployment --k 10 --states CA,TX,NY,FL
    
Or programmatically:
    from run_optimal_federation import OptimalFederationSelector
    
    selector = OptimalFederationSelector(task_name='ACSIncome', k=5)
    optimal_states, loss = selector.run()
"""

import argparse
import numpy as np
import pandas as pd
import traceback
import warnings
import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class SAParams:
    """Simulated Annealing parameters."""
    initial_temp: float = 1.0
    cooling_rate: float = 0.95
    min_temp: float = 1e-4
    max_iterations: int = 2500
    iterations_per_temp: int = 20


@dataclass 
class ExperimentResult:
    """Container for a single experiment run result."""
    # Experiment identification
    run_id: int
    timestamp: str
    
    # Task configuration
    task: str
    k: int
    epsilon: float
    delta: float
    survey_year: str
    
    # Algorithm configuration
    method: str
    n_runs: int  # For SA: which run number this is
    
    # SA Parameters
    sa_initial_temp: float
    sa_cooling_rate: float
    sa_min_temp: float
    sa_max_iterations: int
    sa_iterations_per_temp: int
    
    # Results
    selected_states: str  # Comma-separated
    final_loss: float
    n_total: int
    
    # Weights used
    weight_alpha_SN: float
    weight_alpha_ST: float
    weight_beta_NN: float
    weight_delta_NT: float

# Local imports
from task_config import (
    get_task_config, 
    list_available_tasks, 
    print_task_info,
    TaskConfig,
    ALL_STATES,
    get_data_source
)
from mi_utils import (
    discretize_features, 
    compute_noisy_mi_components_gaussian,
    aggregate_mi_components,
    calculate_global_mi
)
from optimization import (
    greedy_additive_selection,
    greedy_additive_selection_parallel,
    simulated_annealing_selection,
    subtractive_greedy_selection,
    calculate_loss
)
from reporting import report_selection_results


class OptimalFederationSelector:
    """
    A class for selecting optimal federations for any FolkTables task.
    
    Attributes:
        task_config: Configuration for the selected task
        k: Target federation size
        epsilon: Privacy parameter (use float('inf') for no privacy)
        delta: Privacy delta parameter
        states: List of states to consider
        survey_year: ACS survey year
    """
    
    def __init__(
        self,
        task_name: str,
        k: int,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        states: Optional[List[str]] = None,
        survey_year: str = '2018',
        n_min: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ):
        """
        Initialize the Optimal Federation Selector.
        
        Args:
            task_name: Name of the FolkTables task (e.g., 'ACSIncome')
            k: Target federation size (number of states to select)
            epsilon: Privacy parameter for differential privacy
            delta: Delta parameter for (epsilon, delta)-DP
            states: List of state codes to consider (default: all 50 states)
            survey_year: ACS survey year (default: '2018')
            n_min: Minimum total sample size (optional)
            weights: Custom optimization weights (optional)
            verbose: Whether to print progress information
        """
        self.task_config = get_task_config(task_name)
        self.k = k
        self.epsilon = epsilon
        self.delta = delta
        self.states = states or ALL_STATES
        self.survey_year = survey_year
        self.n_min = n_min
        self.weights = weights or self.task_config.default_weights
        self.verbose = verbose
        
        # SA Parameters
        self.sa_params = SAParams()
        
        # Storage for computed data
        self.all_dataframes: Dict[str, pd.DataFrame] = {}
        self.all_mi_components: Dict[str, Dict] = {}
        self.data_source = None
        
        # Results
        self.selected_states: List[str] = []
        self.final_loss: float = float('inf')
        self.aggregated_components: Dict = {}
        
        # Track all experiment runs
        self.all_results: List[ExperimentResult] = []
        
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_and_process_data(self) -> bool:
        """
        Load data for all specified states, discretize features, and compute MI components.
        
        Returns:
            True if at least some data was successfully processed
        """
        self._log(f"\n{'='*60}")
        self._log(f"Loading data for task: {self.task_config.name}")
        self._log(f"States: {len(self.states)}, Target K: {self.k}")
        self._log(f"Privacy: epsilon={self.epsilon}, delta={self.delta}")
        self._log(f"{'='*60}")
        
        self.data_source = get_data_source(survey_year=self.survey_year)
        task = self.task_config.task_object
        bin_specs = self.task_config.bin_specs
        mi_features = self.task_config.mi_features
        
        self._log(f"\nProcessing {len(self.states)} states...")
        
        for state in self.states:
            try:
                self._log(f"\n  Processing {state}...")
                
                # 1. Load raw data
                acs_data = self.data_source.get_data(states=[state], download=False)
                
                # 2. Extract features and labels using the task's df_to_numpy method
                features_np, labels_np, _ = task.df_to_numpy(acs_data)
                
                # 3. Create DataFrame with feature names from task definition
                df_state = pd.DataFrame(features_np, columns=task.features)
                df_state['label'] = labels_np
                
                # 4. Select columns needed for MI analysis
                cols_to_process = [col for col in mi_features if col in df_state.columns]
                if 'label' not in cols_to_process and 'label' in df_state.columns:
                    cols_to_process.append('label')
                
                # Check required columns
                if self.task_config.sensitive_var not in cols_to_process:
                    self._log(f"    Warning: Missing sensitive variable {self.task_config.sensitive_var}, skipping")
                    continue
                if 'label' not in cols_to_process:
                    self._log(f"    Warning: Missing label, skipping")
                    continue
                if len(cols_to_process) < 2:
                    self._log(f"    Skipping - less than 2 relevant columns")
                    continue
                
                df_state_subset = df_state[list(set(cols_to_process))].copy()
                
                # 5. Apply discretization
                df_binned = discretize_features(df_state_subset, bin_specs)
                self.all_dataframes[state] = df_binned
                
                # 6. Build final variable list for MI calculation
                vars_for_mi_calc = []
                for original_var in mi_features:
                    binned_var_name = f"{original_var}_BINNED"
                    if original_var in bin_specs:
                        if binned_var_name in df_binned.columns:
                            vars_for_mi_calc.append(binned_var_name)
                    elif original_var in df_binned.columns:
                        vars_for_mi_calc.append(original_var)
                
                if 'label' not in vars_for_mi_calc and 'label' in df_binned.columns:
                    vars_for_mi_calc.append('label')
                
                vars_present = sorted(list(set([v for v in vars_for_mi_calc if v in df_binned.columns])))
                
                # 7. Compute MI components
                if len(vars_present) >= 2:
                    local_components = compute_noisy_mi_components_gaussian(
                        df_binned, vars_present, 
                        epsilon=self.epsilon, delta=self.delta
                    )
                    
                    if local_components and local_components.get('contingency_tables'):
                        self.all_mi_components[state] = local_components
                        n_tables = len(local_components['contingency_tables'])
                        n_samples = local_components['N']
                        self._log(f"    ✓ {n_tables} tables, {n_samples} samples")
                    else:
                        self._log(f"    Warning: MI computation returned empty")
                else:
                    self._log(f"    Skipping - less than 2 variables after processing")
                    
            except FileNotFoundError:
                self._log(f"    Warning: Data file not found for {state}")
            except Exception as e:
                self._log(f"    ERROR: {e}")
                if self.verbose:
                    traceback.print_exc()
        
        self._log(f"\n✓ Successfully processed {len(self.all_mi_components)} states")
        return len(self.all_mi_components) > 0
    
    def _get_common_vars(self) -> set:
        """Get variables common across all computed MI components."""
        common_vars = set()
        for comp_dict in self.all_mi_components.values():
            for pair in comp_dict.get('contingency_tables', {}).keys():
                common_vars.update(pair)
        return common_vars
    
    def _get_non_sensitive_vars(self) -> List[str]:
        """Get non-sensitive variable names available in the data."""
        common_vars = self._get_common_vars()
        sensitive = self.task_config.sensitive_var
        
        non_sensitive = []
        for var in self.task_config.mi_features:
            var_name = self.task_config.get_binned_feature_name(var)
            if var_name in common_vars and var_name != sensitive and var_name != 'label':
                non_sensitive.append(var_name)
        
        return non_sensitive
    
    def run_greedy_selection(self) -> Tuple[List[str], float]:
        """
        Run greedy additive selection to find optimal federation.
        
        Returns:
            Tuple of (selected_state_ids, final_loss)
        """
        if not self.all_mi_components:
            raise RuntimeError("No MI components available. Call load_and_process_data() first.")
        
        common_vars = self._get_common_vars()
        sensitive_var = self.task_config.sensitive_var
        non_sensitive_vars = self._get_non_sensitive_vars()
        
        if sensitive_var not in common_vars:
            raise ValueError(f"Sensitive variable '{sensitive_var}' not found in data")
        
        ideal_targets = {
            'target_ISN': 0.0,
            'target_INN': 0.0,
            'target_IST': 0.0
        }
        
        actual_k = min(self.k, len(self.all_mi_components))
        if actual_k < self.k:
            self._log(f"Warning: Requested k={self.k}, but only {len(self.all_mi_components)} states available")
        
        self._log(f"\n{'='*60}")
        self._log(f"Running Greedy Selection (k={actual_k})")
        self._log(f"Sensitive: {sensitive_var}")
        self._log(f"Non-sensitive ({len(non_sensitive_vars)}): {non_sensitive_vars}")
        self._log(f"{'='*60}")
        
        selected_ids, final_agg_components, final_loss = greedy_additive_selection_parallel(
            self.all_mi_components,
            ideal_targets,
            self.weights,
            sensitive_var,
            non_sensitive_vars,
            actual_k,
            self.n_min
        )
        
        self.selected_states = selected_ids
        self.final_loss = final_loss
        self.aggregated_components = final_agg_components
        
        if self.verbose:
            report_selection_results(selected_ids, final_agg_components, final_loss, 
                                   f"Greedy Selection ({self.task_config.name})")
        
        return selected_ids, final_loss
    
    def set_sa_params(
        self,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.95,
        min_temp: float = 1e-4,
        max_iterations: int = 2500,
        iterations_per_temp: int = 20
    ):
        """Set simulated annealing parameters."""
        self.sa_params = SAParams(
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            min_temp=min_temp,
            max_iterations=max_iterations,
            iterations_per_temp=iterations_per_temp
        )
        self._log(f"SA Params: T0={initial_temp}, cooling={cooling_rate}, "
                  f"T_min={min_temp}, max_iter={max_iterations}, iter/T={iterations_per_temp}")
    
    def _create_result_record(
        self, 
        run_id: int, 
        method: str, 
        run_number: int,
        selected_ids: List[str], 
        loss: float, 
        n_total: int
    ) -> ExperimentResult:
        """Create an experiment result record."""
        return ExperimentResult(
            run_id=run_id,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            task=self.task_config.name,
            k=self.k,
            epsilon=self.epsilon,
            delta=self.delta,
            survey_year=self.survey_year,
            method=method,
            n_runs=run_number,
            sa_initial_temp=self.sa_params.initial_temp,
            sa_cooling_rate=self.sa_params.cooling_rate,
            sa_min_temp=self.sa_params.min_temp,
            sa_max_iterations=self.sa_params.max_iterations,
            sa_iterations_per_temp=self.sa_params.iterations_per_temp,
            selected_states=','.join(selected_ids) if selected_ids else '',
            final_loss=loss,
            n_total=n_total,
            weight_alpha_SN=self.weights.get('alpha_SN', 0.0),
            weight_alpha_ST=self.weights.get('alpha_ST', 0.0),
            weight_beta_NN=self.weights.get('beta_NN', 0.0),
            weight_delta_NT=self.weights.get('delta_NT', 0.0)
        )
    
    def run_simulated_annealing(self, n_runs: int = 5) -> Tuple[List[str], float]:
        """
        Run simulated annealing selection multiple times and return best result.
        
        Args:
            n_runs: Number of SA runs to perform
            
        Returns:
            Tuple of (best_selected_state_ids, best_loss)
        """
        if not self.all_mi_components:
            raise RuntimeError("No MI components available. Call load_and_process_data() first.")
        
        common_vars = self._get_common_vars()
        sensitive_var = self.task_config.sensitive_var
        non_sensitive_vars = self._get_non_sensitive_vars()
        
        ideal_targets = {'target_ISN': 0.0, 'target_INN': 0.0, 'target_IST': 0.0}
        actual_k = min(self.k, len(self.all_mi_components))
        
        self._log(f"\n{'='*60}")
        self._log(f"Running Simulated Annealing ({n_runs} runs, k={actual_k})")
        self._log(f"SA Params: T0={self.sa_params.initial_temp}, cooling={self.sa_params.cooling_rate}, "
                  f"T_min={self.sa_params.min_temp}, max_iter={self.sa_params.max_iterations}")
        self._log(f"{'='*60}")
        
        best_solution = None
        best_loss = float('inf')
        best_components = None
        
        base_run_id = len(self.all_results)
        
        for i in range(n_runs):
            selected_ids, agg_components, loss = simulated_annealing_selection(
                self.all_mi_components,
                ideal_targets,
                self.weights,
                sensitive_var,
                non_sensitive_vars,
                actual_k,
                self.n_min,
                initial_temp=self.sa_params.initial_temp,
                cooling_rate=self.sa_params.cooling_rate,
                min_temp=self.sa_params.min_temp,
                max_iterations=self.sa_params.max_iterations,
                iterations_per_temp=self.sa_params.iterations_per_temp
            )
            
            n_total = agg_components.get('N', 0) if agg_components else 0
            
            # Record this run
            result = self._create_result_record(
                run_id=base_run_id + i,
                method='simulated_annealing',
                run_number=i + 1,
                selected_ids=selected_ids,
                loss=loss,
                n_total=n_total
            )
            self.all_results.append(result)
            
            self._log(f"  Run {i+1}: loss={loss:.6f}, states={selected_ids}")
            
            if loss < best_loss:
                best_loss = loss
                best_solution = selected_ids
                best_components = agg_components
        
        self.selected_states = best_solution
        self.final_loss = best_loss
        self.aggregated_components = best_components
        
        if self.verbose:
            report_selection_results(best_solution, best_components, best_loss,
                                   f"SA Best ({self.task_config.name})")
        
        return best_solution, best_loss
    
    def run(self, method: str = 'greedy', **kwargs) -> Tuple[List[str], float]:
        """
        Run the complete optimal federation selection pipeline.
        
        Args:
            method: Selection method ('greedy', 'sa', or 'both')
            **kwargs: Additional arguments passed to the selection method
            
        Returns:
            Tuple of (selected_state_ids, final_loss)
        """
        # Load and process data
        if not self.all_mi_components:
            success = self.load_and_process_data()
            if not success:
                raise RuntimeError("Failed to load and process data")
        
        # Run selection
        if method == 'greedy':
            return self.run_greedy_selection()
        elif method == 'sa':
            n_runs = kwargs.get('n_runs', 5)
            return self.run_simulated_annealing(n_runs=n_runs)
        elif method == 'both':
            # Run both and compare
            greedy_result = self.run_greedy_selection()
            sa_result = self.run_simulated_annealing(**kwargs)
            
            if sa_result[1] < greedy_result[1]:
                self._log(f"\n✓ SA found better solution: {sa_result[1]:.6f} vs {greedy_result[1]:.6f}")
                return sa_result
            else:
                self._log(f"\n✓ Greedy found better solution: {greedy_result[1]:.6f} vs {sa_result[1]:.6f}")
                return greedy_result
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'greedy', 'sa', or 'both'")
    
    def save_results(self, output_dir: str = 'results', save_all_runs: bool = True):
        """Save selection results to files.
        
        Args:
            output_dir: Directory to save results
            save_all_runs: If True, save all SA runs to CSV; otherwise just best result
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{self.task_config.name}_k{self.k}_eps{self.epsilon}_{timestamp}"
        
        # Save all runs to CSV if requested
        if save_all_runs and self.all_results:
            csv_path = os.path.join(output_dir, f"{base_name}_all_runs.csv")
            self._save_results_to_csv(csv_path)
            self._log(f"\n✓ All runs saved to {csv_path}")
        
        # Save best result as JSON (backward compatible)
        result = {
            'task': self.task_config.name,
            'k': self.k,
            'epsilon': self.epsilon,
            'delta': self.delta,
            'selected_states': self.selected_states,
            'final_loss': self.final_loss,
            'n_total': self.aggregated_components.get('N', 0),
            'timestamp': timestamp,
            'sa_params': asdict(self.sa_params),
            'weights': self.weights,
            'total_runs': len(self.all_results)
        }
        
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        self._log(f"✓ Best result saved to {json_path}")
        return json_path
    
    def _save_results_to_csv(self, csv_path: str):
        """Save all experiment results to a CSV file."""
        if not self.all_results:
            return
        
        # Get field names from dataclass
        fieldnames = list(asdict(self.all_results[0]).keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.all_results:
                writer.writerow(asdict(result))
    
    def append_to_master_csv(self, csv_path: str = 'results/all_experiments.csv'):
        """
        Append all results to a master CSV file.
        Creates the file if it doesn't exist.
        
        Args:
            csv_path: Path to the master CSV file
        """
        if not self.all_results:
            self._log("No results to save.")
            return
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        fieldnames = list(asdict(self.all_results[0]).keys())
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for result in self.all_results:
                writer.writerow(asdict(result))
        
        self._log(f"✓ {len(self.all_results)} results appended to {csv_path}")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all results as a pandas DataFrame."""
        if not self.all_results:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self.all_results])
    
    def get_mi_matrix(self) -> pd.DataFrame:
        """Get the MI matrix for the selected federation."""
        if not self.aggregated_components:
            raise RuntimeError("No aggregated components. Run selection first.")
        
        pairs = list(self.aggregated_components.get('contingency_tables', {}).keys())
        vars_present = set()
        for pair in pairs:
            vars_present.update(pair)
        vars_list = sorted(list(vars_present))
        
        mi_matrix = pd.DataFrame(index=vars_list, columns=vars_list, dtype=float)
        mi_matrix.fillna(0.0, inplace=True)
        
        for pair in pairs:
            mi_val = calculate_global_mi(self.aggregated_components, pair)
            mi_matrix.loc[pair[0], pair[1]] = mi_val
            mi_matrix.loc[pair[1], pair[0]] = mi_val
        
        return mi_matrix


def find_optimal_federation(
    task_name: str,
    k: int,
    epsilon: float = 1.0,
    method: str = 'greedy',
    states: Optional[List[str]] = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[List[str], float]:
    """
    Convenience function to find optimal federation for a given task.
    
    Args:
        task_name: Name of the FolkTables task
        k: Federation size
        epsilon: Privacy parameter
        method: Selection method ('greedy', 'sa', or 'both')
        states: List of states to consider (default: all 50)
        verbose: Print progress information
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (selected_state_codes, final_loss)
        
    Example:
        >>> optimal_states, loss = find_optimal_federation('ACSIncome', k=5)
        >>> print(f"Optimal states: {optimal_states}, Loss: {loss:.4f}")
    """
    selector = OptimalFederationSelector(
        task_name=task_name,
        k=k,
        epsilon=epsilon,
        states=states,
        verbose=verbose,
        **kwargs
    )
    return selector.run(method=method)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Find optimal federations for FolkTables prediction tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimal_federation.py --task ACSIncome --k 5
  python run_optimal_federation.py --task ACSEmployment --k 10 --epsilon 0.1
  python run_optimal_federation.py --task ACSPublicCoverage --k 3 --method sa --runs 10
  python run_optimal_federation.py --list-tasks
        """
    )
    
    parser.add_argument('--task', '-t', type=str, 
                       help='FolkTables task name (e.g., ACSIncome, ACSEmployment)')
    parser.add_argument('--k', '-k', type=int, default=5,
                       help='Federation size (number of states to select)')
    parser.add_argument('--epsilon', '-e', type=float, default=0.05,
                       help='Privacy parameter epsilon (use "inf" for no privacy)')
    parser.add_argument('--delta', '-d', type=float, default=1e-5,
                       help='Privacy parameter delta')
    parser.add_argument('--method', '-m', type=str, default='greedy',
                       choices=['greedy', 'sa', 'both'],
                       help='Selection method')
    parser.add_argument('--runs', '-r', type=int, default=5,
                       help='Number of SA runs (for --method sa)')
    parser.add_argument('--states', '-s', type=str, default=None,
                       help='Comma-separated list of state codes to consider')
    parser.add_argument('--year', '-y', type=str, default='2018',
                       help='ACS survey year')
    parser.add_argument('--output', '-o', type=str, default='results',
                       help='Output directory for results')
    
    # SA Parameters
    parser.add_argument('--sa-temp', type=float, default=1.0,
                       help='SA initial temperature (default: 1.0)')
    parser.add_argument('--sa-cooling', type=float, default=0.95,
                       help='SA cooling rate (default: 0.95)')
    parser.add_argument('--sa-min-temp', type=float, default=1e-4,
                       help='SA minimum temperature (default: 1e-4)')
    parser.add_argument('--sa-max-iter', type=int, default=2500,
                       help='SA maximum iterations (default: 2500)')
    parser.add_argument('--sa-iter-per-temp', type=int, default=20,
                       help='SA iterations per temperature step (default: 20)')
    
    parser.add_argument('--master-csv', type=str, default=None,
                       help='Append results to this master CSV file')
    parser.add_argument('--list-tasks', action='store_true',
                       help='List available tasks and exit')
    parser.add_argument('--task-info', type=str, default=None,
                       help='Show detailed info for a specific task')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.list_tasks:
        print_task_info()
        return
    
    if args.task_info:
        print_task_info(args.task_info)
        return
    
    # Validate required arguments
    if not args.task:
        parser.error("--task is required. Use --list-tasks to see available tasks.")
    
    # Parse states if provided
    states = None
    if args.states:
        states = [s.strip().upper() for s in args.states.split(',')]
    
    # Parse epsilon
    epsilon = float('inf') if args.epsilon == float('inf') or str(args.epsilon).lower() == 'inf' else args.epsilon
    
    # Run selection
    print(f"\n{'='*60}")
    print(f"Optimal Federation Selection")
    print(f"Task: {args.task}")
    print(f"Target k: {args.k}")
    print(f"Privacy: epsilon={epsilon}, delta={args.delta}")
    print(f"Method: {args.method}")
    print(f"{'='*60}\n")
    
    selector = OptimalFederationSelector(
        task_name=args.task,
        k=args.k,
        epsilon=epsilon,
        delta=args.delta,
        states=states,
        survey_year=args.year,
        verbose=not args.quiet
    )
    
    # Set SA parameters
    selector.set_sa_params(
        initial_temp=args.sa_temp,
        cooling_rate=args.sa_cooling,
        min_temp=args.sa_min_temp,
        max_iterations=args.sa_max_iter,
        iterations_per_temp=args.sa_iter_per_temp
    )
    
    try:
        selected_states, loss = selector.run(method=args.method, n_runs=args.runs)
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Task: {args.task}")
        print(f"Method: {args.method}")
        print(f"Optimal {args.k} states: {selected_states}")
        print(f"Final loss: {loss:.6f}")
        print(f"Total samples: {selector.aggregated_components.get('N', 0)}")
        print(f"Total runs recorded: {len(selector.all_results)}")
        
        if args.method == 'sa':
            print(f"\nSA Parameters:")
            print(f"  Initial Temp: {args.sa_temp}")
            print(f"  Cooling Rate: {args.sa_cooling}")
            print(f"  Min Temp: {args.sa_min_temp}")
            print(f"  Max Iterations: {args.sa_max_iter}")
            print(f"  Iterations/Temp: {args.sa_iter_per_temp}")
        
        # Save results
        result_path = selector.save_results(args.output, save_all_runs=True)
        print(f"\nResults saved to: {result_path}")
        
        # Append to master CSV if specified
        if args.master_csv:
            selector.append_to_master_csv(args.master_csv)
        
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
