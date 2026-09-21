"""
Subprocess worker: trains FL on one federation and writes its metrics to a
JSON file, then exits.

Why this exists: run_exp() (FolkTables_FL.py) leaks TensorFlow/Keras memory
across repeated calls (each federation builds fresh models without clearing
the Keras backend session), observed growing from ~3.3GB to ~14.75GB over 12
federations before the kernel OOM-killed the long-running orchestrator
process. run_exp() is also called by run_comparison_experiment(), the
production FL training pipeline behind the paper's main reported results, so
patching it directly risks that pipeline for a problem that's only actually
blocking this one-off validation script. Running each federation's training
in its own short-lived subprocess sidesteps the leak entirely -- the OS
reclaims all memory unconditionally on process exit, regardless of what
Keras did internally -- with zero changes to FolkTables_FL.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_training"))

import argparse
import json

from pfl_validation_experiment import run_federation_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--states', type=str, required=True, help='Comma-separated state codes')
    parser.add_argument('--n-seeds', type=int, required=True)
    parser.add_argument('--max-iterations', type=int, required=True)
    parser.add_argument('--output', type=str, required=True, help='Path to write result JSON')
    args = parser.parse_args()

    states = args.states.split(',')
    metrics = run_federation_experiment(
        states=states, task=args.task, n_seeds=args.n_seeds, max_iterations=args.max_iterations
    )
    # np.mean/np.std return numpy scalar types (float32/float64), not
    # JSON-serializable -- convert to native Python types before writing.
    metrics = {k: (v.item() if hasattr(v, 'item') else v) for k, v in metrics.items()}
    with open(args.output, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    main()
