#!/usr/bin/env python3
"""
Example: Finding Optimal Federations for Different FolkTables Tasks

This script demonstrates how to use the generalized optimal federation
selection for any FolkTables prediction task.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # optimization/ (self, for future-proofing)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from run_optimal_federation import OptimalFederationSelector, find_optimal_federation
from task_config import list_available_tasks, print_task_info, get_task_config


def example_basic_usage():
    """Basic example: Find optimal 5-state federation for ACSIncome."""
    print("\n" + "="*70)
    print("Example 1: Basic Usage - ACSIncome with k=5")
    print("="*70)
    
    optimal_states, loss = find_optimal_federation(
        task_name='ACSIncome',
        k=5,
        epsilon=0.05,
        method='greedy'
    )
    
    print(f"\n✓ Optimal states: {optimal_states}")
    print(f"✓ Final loss: {loss:.6f}")
    
    return optimal_states, loss


def example_multiple_tasks():
    """Run optimal federation selection for multiple tasks."""
    print("\n" + "="*70)
    print("Example 2: Compare Optimal Federations Across Tasks (k=3)")
    print("="*70)
    
    tasks = ['ACSIncome', 'ACSEmployment', 'ACSPublicCoverage']
    k = 3
    
    results = {}
    for task_name in tasks:
        print(f"\n--- {task_name} ---")
        try:
            optimal_states, loss = find_optimal_federation(
                task_name=task_name,
                k=k,
                epsilon=0.1,  # Use higher epsilon for faster demo
                method='greedy',
                verbose=False  # Quiet mode for cleaner output
            )
            results[task_name] = {'states': optimal_states, 'loss': loss}
            print(f"  Optimal states: {optimal_states}")
            print(f"  Loss: {loss:.6f}")
        except Exception as e:
            print(f"  Error: {e}")
            results[task_name] = {'error': str(e)}
    
    return results


def example_varying_k():
    """Find optimal federations for different sizes k."""
    print("\n" + "="*70)
    print("Example 3: Optimal Federations for k = 2, 3, 5, 10 (ACSIncome)")
    print("="*70)
    
    task_name = 'ACSIncome'
    k_values = [2, 3, 5, 10]
    
    results = {}
    for k in k_values:
        print(f"\n--- k={k} ---")
        selector = OptimalFederationSelector(
            task_name=task_name,
            k=k,
            epsilon=0.05,
            verbose=False
        )
        
        try:
            optimal_states, loss = selector.run(method='greedy')
            results[k] = {
                'states': optimal_states,
                'loss': loss,
                'n_total': selector.aggregated_components.get('N', 0)
            }
            print(f"  States: {optimal_states}")
            print(f"  Loss: {loss:.6f}")
            print(f"  Total samples: {results[k]['n_total']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    return results


def example_privacy_comparison():
    """Compare optimal federations under different privacy levels."""
    print("\n" + "="*70)
    print("Example 4: Privacy Level Comparison (ACSIncome, k=5)")
    print("="*70)
    
    task_name = 'ACSIncome'
    k = 5
    epsilon_values = [0.01, 0.05, 0.1, 1.0, float('inf')]
    
    results = {}
    for epsilon in epsilon_values:
        eps_str = 'inf' if epsilon == float('inf') else f'{epsilon}'
        print(f"\n--- epsilon={eps_str} ---")
        
        try:
            optimal_states, loss = find_optimal_federation(
                task_name=task_name,
                k=k,
                epsilon=epsilon,
                method='greedy',
                verbose=False
            )
            results[eps_str] = {'states': optimal_states, 'loss': loss}
            print(f"  States: {optimal_states}")
            print(f"  Loss: {loss:.6f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    return results


def example_simulated_annealing():
    """Use simulated annealing for potentially better solutions."""
    print("\n" + "="*70)
    print("Example 5: Simulated Annealing vs Greedy (ACSIncome, k=5)")
    print("="*70)
    
    selector = OptimalFederationSelector(
        task_name='ACSIncome',
        k=5,
        epsilon=0.05,
        verbose=True
    )
    
    # Load data once
    selector.load_and_process_data()
    
    # Run greedy
    print("\n--- Greedy Selection ---")
    greedy_states, greedy_loss = selector.run_greedy_selection()
    
    # Run SA
    print("\n--- Simulated Annealing (5 runs) ---")
    sa_states, sa_loss = selector.run_simulated_annealing(n_runs=5)
    
    print("\n--- Comparison ---")
    print(f"Greedy: {greedy_states}, loss={greedy_loss:.6f}")
    print(f"SA:     {sa_states}, loss={sa_loss:.6f}")
    
    if sa_loss < greedy_loss:
        print("✓ Simulated Annealing found a better solution!")
    else:
        print("✓ Greedy found an equally good or better solution")
    
    return {'greedy': (greedy_states, greedy_loss), 'sa': (sa_states, sa_loss)}


def example_subset_states():
    """Run optimization on a subset of states."""
    print("\n" + "="*70)
    print("Example 6: Optimize over a Subset of States")
    print("="*70)
    
    # Only consider coastal states
    coastal_states = ['CA', 'OR', 'WA', 'FL', 'NY', 'NJ', 'MA', 'CT', 
                     'ME', 'NC', 'SC', 'GA', 'TX', 'LA', 'AL', 'MS']
    
    print(f"Considering only: {coastal_states}")
    
    optimal_states, loss = find_optimal_federation(
        task_name='ACSIncome',
        k=5,
        epsilon=0.05,
        states=coastal_states,
        method='greedy',
        verbose=False
    )
    
    print(f"\n✓ Optimal coastal states: {optimal_states}")
    print(f"✓ Loss: {loss:.6f}")
    
    return optimal_states, loss


def example_get_mi_matrix():
    """Get the MI matrix for the selected federation."""
    print("\n" + "="*70)
    print("Example 7: Analyze MI Matrix of Optimal Federation")
    print("="*70)
    
    selector = OptimalFederationSelector(
        task_name='ACSIncome',
        k=3,
        epsilon=0.1,
        verbose=False
    )
    
    optimal_states, loss = selector.run(method='greedy')
    
    print(f"Optimal states: {optimal_states}")
    print(f"Loss: {loss:.6f}")
    
    # Get MI matrix
    mi_matrix = selector.get_mi_matrix()
    print("\nMI Matrix (first 5x5):")
    print(mi_matrix.iloc[:5, :5].to_string())
    
    return mi_matrix


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("AVAILABLE FOLKTABLES TASKS")
    print("="*70)
    print_task_info()
    
    # Choose which examples to run
    print("\n\nRunning Examples...\n")
    
    # Example 1: Basic usage
    example_basic_usage()
    
    # Uncomment to run additional examples:
    # example_multiple_tasks()
    # example_varying_k()
    # example_privacy_comparison()
    # example_simulated_annealing()
    # example_subset_states()
    # example_get_mi_matrix()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
