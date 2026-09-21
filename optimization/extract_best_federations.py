#!/usr/bin/env python3
"""
Extract best federations from all_federations.csv files across multiple runs.

For each task and k value, finds the federation with the lowest final_loss.
"""

import pandas as pd
from pathlib import Path
import argparse


def find_all_federation_files(results_dir: Path) -> list[Path]:
    """Find all all_federations.csv files in results/full_*/ directories."""
    pattern = "full_*/all_federations.csv"
    return list(results_dir.glob(pattern))


def load_and_combine_federations(files: list[Path]) -> pd.DataFrame:
    """Load and combine all federation CSV files."""
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = str(f)
            df['source_dir'] = f.parent.name
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {f.parent.name}/all_federations.csv")
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not dfs:
        raise ValueError("No federation files found or loaded")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows combined: {len(combined)}")
    return combined


def find_best_federations(df: pd.DataFrame) -> pd.DataFrame:
    """Find the best federation (lowest final_loss) for each task and k."""
    # Group by task and k, find the row with minimum final_loss
    idx = df.groupby(['task', 'k'])['final_loss'].idxmin()
    best = df.loc[idx].copy()
    
    # Sort by task and k for better readability
    best = best.sort_values(['task', 'k']).reset_index(drop=True)
    
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Extract best federations from all experiment runs"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory containing full_*/ subdirectories (default: results)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file for best federations (default: results/best_federations.csv)"
    )
    parser.add_argument(
        "--show-columns",
        nargs="+",
        default=["task", "k", "final_loss", "selected_states", "n_total", "source_dir"],
        help="Columns to display in summary"
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.results_dir / "best_federations.csv"
    
    # Find all federation files
    files = find_all_federation_files(args.results_dir)
    if not files:
        print(f"No all_federations.csv files found in {args.results_dir}/full_*/")
        return
    
    print(f"Found {len(files)} federation files:\n")
    for f in sorted(files):
        print(f"  - {f}")
    print()
    
    # Load and combine
    combined_df = load_and_combine_federations(files)
    
    # Find best federations
    best_df = find_best_federations(combined_df)
    
    # Display summary
    print("\n" + "=" * 80)
    print("BEST FEDERATIONS (by task and k)")
    print("=" * 80 + "\n")
    
    # Filter display columns to those that exist
    display_cols = [c for c in args.show_columns if c in best_df.columns]
    
    for task in sorted(best_df['task'].unique()):
        print(f"\n{task}:")
        print("-" * 60)
        task_df = best_df[best_df['task'] == task][display_cols]
        print(task_df.to_string(index=False))
        print()
    
    # Save to CSV
    best_df.to_csv(args.output, index=False)
    print(f"\nBest federations saved to: {args.output}")
    
    # Also create a summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    summary = best_df.pivot_table(
        index='task',
        columns='k',
        values='final_loss',
        aggfunc='first'
    )
    print(summary.to_string())
    
    print("\n\nSelected states for each best federation:")
    for _, row in best_df.iterrows():
        print(f"  {row['task']} (k={row['k']}): {row['selected_states']}")


if __name__ == "__main__":
    main()
