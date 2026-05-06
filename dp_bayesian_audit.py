"""
dp_bayesian_audit.py
====================
Bayesian membership-inference audit for DP contingency-table releases.

Strategy
--------
Unlike the frequentist LRT in dp_contingency_audit.py, the server here uses
*population-level priors* (census proportions π_{xy}) to replace the unknown
baseline count.  The posterior log-odds for a target record r = (x, y, z) is:

    Score(r) = Σ_t  (c̃_t  −  μ_t  −  0.5) / v_t  +  log_prior_odds

where, for each feature-pair table t = (A, B):
    c̃_t            = noisy observed cell(a, b)
    μ_t  = n_i · π_{ab}    prior expected count  (from full ACS population)
    v_t  = n_i · π_{ab}·(1−π_{ab}) + σ²         sampling + DP variance
    σ    = Gaussian DP noise std

and log_prior_odds = log( n_i·π_{xyz} / (1 − n_i·π_{xyz}) ) ≈ log(n_i·π_{xyz})
for low-prevalence demographics.

Four attack variants are evaluated per (task, state, ε):
  1. bayes_single   – one table (first feature vs target)
  2. bayes_two      – two tables (first two features vs target)
  3. bayes_joint    – all K(K−1)/2 pairwise tables
  4. naive_lrt      – frequentist residual (no prior), for comparison

Theoretical AUC (closed form):
    AUC_theory = Φ( ½ · √( Σ_t 1/v_t ) )

Records are stratified by expected cell count n_i·π_{xyz} into
    sparse  (<5)  |  medium  (5–50)  |  dense  (≥50)
to identify where privacy risk concentrates.

Usage
-----
    python dp_bayesian_audit.py --tasks ACSIncome --states CA TX --epsilons 0.1 0.5 1.0 2.0 5.0 inf

Output
------
    results/dp_bayesian_audit/
        dp_bayesian_audit_detailed.csv
        dp_bayesian_audit_summary.csv
        {task}_bayes_panels.png          (AUC + TPR@1%FPR per task)
        {task}_sparsity_panels.png       (sparse / medium / dense split)
"""

import argparse
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage, ACSTravelTime
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_OBJECTS = {
    "ACSIncome": ACSIncome,
    "ACSEmployment": ACSEmployment,
    "ACSPublicCoverage": ACSPublicCoverage,
    "ACSTravelTime": ACSTravelTime,
}

ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

SPARSITY_BINS = {"sparse": (0, 5), "medium": (5, 50), "dense": (50, float("inf"))}

ATTACK_COLORS = {
    "naive_lrt":    "#7f7f7f",
    "bayes_single": "#1f77b4",
    "bayes_two":    "#ff7f0e",
    "bayes_joint":  "#d62728",
}

ATTACK_LABELS = {
    "naive_lrt":    "Naive LRT",
    "bayes_single": "Bayes single-table",
    "bayes_two":    "Bayes two-table",
    "bayes_joint":  "Bayes joint",
}


# ---------------------------------------------------------------------------
# DP helpers
# ---------------------------------------------------------------------------

def sigma_from_epsilon(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    if math.isinf(epsilon):
        return 0.0
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 or inf")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float = 0.01) -> float:
    negatives = y_score[y_true == 0]
    positives = y_score[y_true == 1]
    if negatives.size == 0 or positives.size == 0:
        return float("nan")
    threshold = float(np.quantile(negatives, 1.0 - fpr_target))
    return float(np.mean(positives >= threshold))


def theoretical_auc(v_list: List[float]) -> float:
    """AUC_theory = Φ( ½ · √(Σ_t 1/v_t) )  (Gaussian posterior predictive)."""
    if not v_list or any(v <= 0 for v in v_list):
        return float("nan")
    return float(norm.cdf(0.5 * math.sqrt(sum(1.0 / v for v in v_list))))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_state_task_dataframe(data_source: ACSDataSource, task_obj, state: str) -> pd.DataFrame:
    acs_data = data_source.get_data(states=[state], download=True)
    out = task_obj.df_to_pandas(acs_data)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError("task_obj.df_to_pandas must return at least (X, y)")

    x_df = out[0]
    y_s = out[1]
    y_name = str(getattr(task_obj, "target", "target"))

    y_arr = np.asarray(y_s)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr[:, 0]
    elif y_arr.ndim > 1:
        y_arr = y_arr.reshape(-1)

    y_series = pd.Series(y_arr, name=y_name).astype(int)
    df = x_df.copy()
    df[y_name] = y_series.values
    return df


# ---------------------------------------------------------------------------
# Prior estimation
# ---------------------------------------------------------------------------

def estimate_priors(
    prior_df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> pd.Series:
    """
    Compute empirical joint proportions π_{ab} = count(a,b) / N_total
    from prior_df (e.g. full ACS population across all states).
    Returns a Series indexed by (a, b) tuples.
    """
    counts = prior_df.groupby([col_a, col_b], dropna=False).size()
    total = len(prior_df)
    return counts / total


def marginal_prior(
    prior_df: pd.DataFrame,
    cols: List[str],
) -> pd.Series:
    """
    Compute empirical joint proportions π_{cols[0], cols[1], ...} over any
    combination of columns.
    """
    counts = prior_df.groupby(cols, dropna=False).size()
    total = len(prior_df)
    return counts / total


# ---------------------------------------------------------------------------
# Bayesian score computation
# ---------------------------------------------------------------------------

def _log_prior_odds(n_i: int, pi_xyz: float) -> float:
    """
    log( P(r ∈ D_i) / P(r ∉ D_i) )  assuming Bernoulli inclusion with
    probability 1 / n_pop (all-or-nothing approximation for a specific record).
    In practice we use n_i · π_{xyz} as the expected count of records of this
    demographic in client i.

    Returns 0.0 when n_i * pi_xyz is nearly 0 (effectively no prior).
    """
    expected = n_i * pi_xyz
    if expected <= 0.0:
        return 0.0
    # cap to avoid log(0) or log(negative)
    expected = min(expected, n_i - 1)
    return math.log(expected / max(n_i - expected, 1e-9))


def bayesian_scores_for_tables(
    sampled: pd.DataFrame,
    col_pairs: List[Tuple[str, str]],
    target_col: str,
    sigma: float,
    prior_df: pd.DataFrame,
    rng: np.random.Generator,
    include_log_prior_odds: bool = True,
) -> np.ndarray:
    """
    Compute per-record Bayesian log-posterior-odds scores for *members* and
    *non-members* using the supplied list of column pairs.

    Parameters
    ----------
    sampled        : client dataset (D_i), already sub-sampled to max_records
    col_pairs      : list of (feature_col, target_col) pairs defining tables
    target_col     : label column name
    sigma          : DP noise std (per table)
    prior_df       : full ACS population dataframe used to compute priors
    rng            : numpy Generator for reproducibility
    include_log_prior_odds : whether to add the log-prior-odds term

    Returns
    -------
    score_pos : (n,) scores for member records
    score_neg : (n,) scores for non-member records (D_i unchanged)
    v_list    : list of per-table variances v_t  (for theoretical AUC)
    """
    n_i = len(sampled)

    # --- per-table contributions -------------------------------------------
    score_pos = np.zeros(n_i)
    score_neg = np.zeros(n_i)
    v_list = []

    for col_a, col_b in col_pairs:
        # population prior proportions
        pi_ab = estimate_priors(prior_df, col_a, col_b)

        # per-record true counts in client dataset
        counts_client = sampled.groupby([col_a, col_b], dropna=False).size().to_dict()
        a_vals = sampled[col_a].tolist()
        b_vals = sampled[col_b].tolist()

        true_count = np.empty(n_i, dtype=float)
        prior_mu = np.empty(n_i, dtype=float)
        prior_v = np.empty(n_i, dtype=float)

        for i in range(n_i):
            key = (a_vals[i], b_vals[i])
            c_true = float(counts_client.get(key, 0))
            pi = float(pi_ab.get(key, 1e-9))  # fallback to very small value

            mu = n_i * pi
            v = n_i * pi * (1.0 - pi) + sigma ** 2

            true_count[i] = c_true
            prior_mu[i] = mu
            prior_v[i] = max(v, 1e-9)  # avoid division by zero

        # representative v for theoretical AUC (mean over records)
        v_list.append(float(np.mean(prior_v)))

        # noisy observations
        if sigma > 0:
            noise_pos = rng.normal(0.0, sigma, size=n_i)
            noise_neg = rng.normal(0.0, sigma, size=n_i)
        else:
            noise_pos = np.zeros(n_i)
            noise_neg = np.zeros(n_i)

        # member: true count *includes* the record  →  c̃ = true_count + noise
        c_tilde_pos = true_count + noise_pos
        # non-member: true count *excludes* the record  →  c̃ = (true_count - 1) + noise
        c_tilde_neg = (true_count - 1.0) + noise_neg

        # Bayesian log-likelihood ratio contribution: (c̃ - μ - 0.5) / v
        score_pos += (c_tilde_pos - prior_mu - 0.5) / prior_v
        score_neg += (c_tilde_neg - prior_mu - 0.5) / prior_v

    # --- log-prior-odds term -----------------------------------------------
    if include_log_prior_odds and col_pairs:
        # Build multi-column prior for (feature_cols ∪ {target_col})
        all_cols = list(dict.fromkeys([c for pair in col_pairs for c in pair]))
        if target_col not in all_cols:
            all_cols.append(target_col)
        pi_xyz_series = marginal_prior(prior_df, all_cols)

        a_vals_all = {col: sampled[col].tolist() for col in all_cols}

        for i in range(n_i):
            key = tuple(a_vals_all[col][i] for col in all_cols)
            pi_xyz = float(pi_xyz_series.get(key, 1e-9))
            lpo = _log_prior_odds(n_i, pi_xyz)
            score_pos[i] += lpo
            score_neg[i] += lpo

    return score_pos, score_neg, v_list


def naive_lrt_scores(
    sampled: pd.DataFrame,
    col_pairs: List[Tuple[str, str]],
    sigma: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frequentist LRT residual scores (no prior).  Mirrors the joint LRT from
    dp_contingency_audit.py but operates on arbitrary col_pairs.

    Score(r) = Σ_t  [ c̃_t(member) − (true_count_t − 1) ]
    """
    n_i = len(sampled)
    score_pos = np.zeros(n_i)
    score_neg = np.zeros(n_i)

    for col_a, col_b in col_pairs:
        counts = sampled.groupby([col_a, col_b], dropna=False).size().to_dict()
        a_vals = sampled[col_a].tolist()
        b_vals = sampled[col_b].tolist()

        baseline = np.array([float(counts.get((a_vals[i], b_vals[i]), 0)) - 1.0
                              for i in range(n_i)])

        if sigma > 0:
            noise_pos = rng.normal(0.0, sigma, size=n_i)
            noise_neg = rng.normal(0.0, sigma, size=n_i)
        else:
            noise_pos = np.zeros(n_i)
            noise_neg = np.zeros(n_i)

        score_pos += (baseline + 1.0 + noise_pos) - baseline
        score_neg += (baseline + noise_neg) - baseline

    return score_pos, score_neg


# ---------------------------------------------------------------------------
# Sparsity category
# ---------------------------------------------------------------------------

def sparsity_category(n_pi: float) -> str:
    """Return 'sparse', 'medium', or 'dense' based on expected cell count."""
    if n_pi < 5:
        return "sparse"
    if n_pi < 50:
        return "medium"
    return "dense"


def record_sparsity_labels(
    sampled: pd.DataFrame,
    prior_df: pd.DataFrame,
    col_pairs: List[Tuple[str, str]],
) -> np.ndarray:
    """
    Assign each record the sparsity category based on the *minimum* expected
    cell count across all col_pairs (most-sparse table dominates).
    """
    n_i = len(sampled)
    min_expected = np.full(n_i, float("inf"))

    for col_a, col_b in col_pairs:
        pi_ab = estimate_priors(prior_df, col_a, col_b)
        a_vals = sampled[col_a].tolist()
        b_vals = sampled[col_b].tolist()
        for i in range(n_i):
            key = (a_vals[i], b_vals[i])
            pi = float(pi_ab.get(key, 0.0))
            min_expected[i] = min(min_expected[i], n_i * pi)

    return np.array([sparsity_category(v) for v in min_expected])


# ---------------------------------------------------------------------------
# Single-state audit
# ---------------------------------------------------------------------------

def audit_state(
    df_client: pd.DataFrame,
    prior_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    sigma: float,
    max_records: int,
    rng: np.random.Generator,
    fpr_target: float = 0.01,
) -> List[Dict]:
    """
    Run all four attack variants on a single client dataset.
    Returns a list of row dicts (one per attack variant).
    """
    n = min(max_records, len(df_client))
    sampled = df_client.sample(n=n, random_state=int(rng.integers(0, 10_000_000))).reset_index(drop=True)
    n_i = len(sampled)

    # --- build column pairs for each attack variant -------------------------
    f1 = feature_cols[0] if len(feature_cols) >= 1 else None
    f2 = feature_cols[1] if len(feature_cols) >= 2 else None

    pairs_single = [(f1, target_col)] if f1 else []
    pairs_two = [(f1, target_col), (f2, target_col)] if f1 and f2 else pairs_single
    pairs_joint = [(col, target_col) for col in feature_cols]  # all K feature-label tables

    rows = []

    # --- sparsity labelling (based on joint pairs) -------------------------
    sparsity_labels = record_sparsity_labels(sampled, prior_df, pairs_joint)

    for attack_name, col_pairs, use_prior in [
        ("naive_lrt",    pairs_joint, False),
        ("bayes_single", pairs_single, True),
        ("bayes_two",    pairs_two,   True),
        ("bayes_joint",  pairs_joint, True),
    ]:
        if not col_pairs:
            continue

        if not use_prior:
            score_pos, score_neg = naive_lrt_scores(sampled, col_pairs, sigma, rng)
            v_list = []
        else:
            score_pos, score_neg, v_list = bayesian_scores_for_tables(
                sampled=sampled,
                col_pairs=col_pairs,
                target_col=target_col,
                sigma=sigma,
                prior_df=prior_df,
                rng=rng,
                include_log_prior_odds=True,
            )

        y_true = np.concatenate([np.ones(n_i), np.zeros(n_i)])
        y_score = np.concatenate([score_pos, score_neg])

        row_base = dict(
            attack=attack_name,
            n_records=n_i,
            n_tables=len(col_pairs),
            sigma=sigma,
            auc=safe_auc(y_true, y_score),
            tpr_at_1pct=tpr_at_fpr(y_true, y_score, fpr_target),
            theoretical_auc=theoretical_auc(v_list) if v_list else float("nan"),
        )

        # per-sparsity metrics
        for scat in ("sparse", "medium", "dense"):
            mask = sparsity_labels == scat
            if mask.sum() == 0:
                row_base[f"auc_{scat}"] = float("nan")
                row_base[f"tpr_{scat}"] = float("nan")
                continue
            y_true_s = np.concatenate([np.ones(mask.sum()), np.zeros(mask.sum())])
            y_score_s = np.concatenate([score_pos[mask], score_neg[mask]])
            row_base[f"auc_{scat}"] = safe_auc(y_true_s, y_score_s)
            row_base[f"tpr_{scat}"] = tpr_at_fpr(y_true_s, y_score_s, fpr_target)

        rows.append(row_base)

    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = {
        "auc_mean": ("auc", "mean"),
        "auc_std":  ("auc", "std"),
        "tpr_mean": ("tpr_at_1pct", "mean"),
        "tpr_std":  ("tpr_at_1pct", "std"),
        "theoretical_auc_mean": ("theoretical_auc", "mean"),
        "auc_sparse_mean": ("auc_sparse", "mean"),
        "auc_medium_mean": ("auc_medium", "mean"),
        "auc_dense_mean":  ("auc_dense",  "mean"),
        "tpr_sparse_mean": ("tpr_sparse", "mean"),
        "tpr_medium_mean": ("tpr_medium", "mean"),
        "tpr_dense_mean":  ("tpr_dense",  "mean"),
        "n_states": ("state", "nunique"),
    }
    return (
        df.groupby(["task", "attack", "epsilon", "epsilon_label"], as_index=False)
        .agg(**agg_cols)
        .sort_values(["task", "attack", "epsilon"], kind="stable")
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_x_axis(eps_labels: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    x_positions = np.arange(len(eps_labels))
    eps_map = {label: idx for idx, label in enumerate(eps_labels)}
    return x_positions, eps_map


def plot_main_panels(summary_df: pd.DataFrame, out_dir: Path, delta: float, fpr: float = 0.01) -> None:
    """AUC and TPR@1%FPR panels for each task, all four attack variants."""
    plt.style.use("seaborn-v0_8-whitegrid")
    tasks = summary_df["task"].unique().tolist()

    for task in tasks:
        task_df = summary_df[summary_df["task"] == task].copy()
        eps_labels = task_df["epsilon_label"].drop_duplicates().tolist()
        x_positions, eps_map = _make_x_axis(eps_labels)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)

        for attack_name in ["naive_lrt", "bayes_single", "bayes_two", "bayes_joint"]:
            sub = task_df[task_df["attack"] == attack_name].copy()
            if sub.empty:
                continue
            xs = np.array([eps_map[v] for v in sub["epsilon_label"]])
            color = ATTACK_COLORS[attack_name]
            label = ATTACK_LABELS[attack_name]

            axes[0].plot(xs, sub["auc_mean"], marker="o", linewidth=2.2, color=color, label=label)
            axes[0].fill_between(xs,
                                 sub["auc_mean"] - sub["auc_std"],
                                 sub["auc_mean"] + sub["auc_std"],
                                 color=color, alpha=0.15)

            # plot theoretical AUC for Bayesian attacks (dashed)
            if attack_name.startswith("bayes") and not sub["theoretical_auc_mean"].isna().all():
                axes[0].plot(xs, sub["theoretical_auc_mean"], linestyle="--", linewidth=1.4,
                             color=color, alpha=0.7)

            axes[1].plot(xs, sub["tpr_mean"], marker="o", linewidth=2.2, color=color, label=label)
            axes[1].fill_between(xs,
                                 sub["tpr_mean"] - sub["tpr_std"],
                                 sub["tpr_mean"] + sub["tpr_std"],
                                 color=color, alpha=0.15)

        for ax in axes:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(eps_labels)
            ax.set_xlabel("Epsilon", fontsize=14)
            ax.tick_params(axis="both", labelsize=14)

        axes[0].set_ylabel("AUC", fontsize=14)
        axes[0].set_ylim(0.45, 1.02)

        axes[1].set_ylabel("TPR @ 1% FPR", fontsize=14)
        axes[1].set_yscale("log")
        axes[1].set_ylim(1e-3, 1.05)

        handles, labels_ = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels_, fontsize=13, loc="upper right")

        out_path = out_dir / f"{task}_bayes_panels.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)


def plot_sparsity_panels(summary_df: pd.DataFrame, out_dir: Path, fpr: float = 0.01) -> None:
    """TPR@1%FPR split by sparsity bin for the strongest attack (bayes_joint)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    tasks = summary_df["task"].unique().tolist()

    sparsity_colors = {"sparse": "#d62728", "medium": "#ff7f0e", "dense": "#2ca02c"}

    for task in tasks:
        task_df = summary_df[(summary_df["task"] == task) &
                             (summary_df["attack"] == "bayes_joint")].copy()
        if task_df.empty:
            continue

        eps_labels = task_df["epsilon_label"].drop_duplicates().tolist()
        x_positions, eps_map = _make_x_axis(eps_labels)
        xs = np.array([eps_map[v] for v in task_df["epsilon_label"]])

        fig, ax = plt.subplots(1, 1, figsize=(7, 4.8), constrained_layout=True)

        for scat, color in sparsity_colors.items():
            col = f"tpr_{scat}_mean"
            if col not in task_df.columns:
                continue
            ax.plot(xs, task_df[col], marker="o", linewidth=2.2, color=color, label=scat.title())

        ax.set_xticks(x_positions)
        ax.set_xticklabels(eps_labels)
        ax.set_xlabel("Epsilon", fontsize=14)
        ax.set_ylabel("TPR @ 1% FPR  (Bayes joint)", fontsize=14)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.05)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=13)

        out_path = out_dir / f"{task}_sparsity_panels.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_audit(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_values: List[float] = []
    for raw in args.epsilons:
        if str(raw).lower() in {"inf", "infty", "infinity"}:
            eps_values.append(float("inf"))
        else:
            eps_values.append(float(raw))

    data_source = ACSDataSource(survey_year=args.survey_year, horizon=args.horizon, survey="person")
    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------------------------
    # Build population-level prior dataframe from *all* specified prior states
    # -----------------------------------------------------------------------
    prior_states = args.prior_states if args.prior_states else ALL_STATES
    print(f"Loading prior dataset from {len(prior_states)} states: {prior_states[:5]}{'...' if len(prior_states) > 5 else ''}")

    # We need to load priors task by task because features differ.
    # We cache them in a dict keyed by task name.
    prior_dfs: Dict[str, pd.DataFrame] = {}

    rows = []

    for task_name in args.tasks:
        task_obj = TASK_OBJECTS[task_name]
        print(f"\n=== Task: {task_name} ===")

        # Build prior (full population across prior_states)
        if task_name not in prior_dfs:
            prior_parts = []
            for ps in prior_states:
                try:
                    pdata = data_source.get_data(states=[ps], download=True)
                    out = task_obj.df_to_pandas(pdata)
                    xdf = out[0]
                    ys = out[1]
                    y_name = str(getattr(task_obj, "target", "target"))
                    y_arr = np.asarray(ys)
                    if y_arr.ndim == 2:
                        y_arr = y_arr[:, 0]
                    y_arr = y_arr.reshape(-1)
                    xdf = xdf.copy()
                    xdf[y_name] = y_arr.astype(int)
                    prior_parts.append(xdf)
                except Exception as e:
                    print(f"  Skipping prior state {ps} for {task_name}: {e}")

            if not prior_parts:
                print(f"  No prior data for {task_name}, skipping.")
                continue
            prior_dfs[task_name] = pd.concat(prior_parts, ignore_index=True)
            print(f"  Prior dataset: {len(prior_dfs[task_name])} records from {len(prior_parts)} states")

        prior_df = prior_dfs[task_name]

        for state in args.states:
            try:
                df_client = load_state_task_dataframe(data_source, task_obj, state)
            except Exception as e:
                print(f"  Skipping client {task_name}/{state}: {e}")
                continue

            if len(df_client) < 50:
                print(f"  Skipping {task_name}/{state}: too few rows ({len(df_client)})")
                continue

            target_col = str(getattr(task_obj, "target", "target"))
            feature_cols = [c for c in getattr(task_obj, "features", []) if c in df_client.columns]
            if not feature_cols:
                print(f"  Skipping {task_name}/{state}: no valid feature columns")
                continue

            for eps in eps_values:
                sigma = sigma_from_epsilon(eps, args.delta)
                eps_label = "inf" if math.isinf(eps) else str(eps)

                attack_rows = audit_state(
                    df_client=df_client,
                    prior_df=prior_df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    sigma=sigma,
                    max_records=args.max_records_per_state,
                    rng=rng,
                    fpr_target=args.fpr,
                )

                for ar in attack_rows:
                    ar["task"] = task_name
                    ar["state"] = state
                    ar["epsilon"] = eps
                    ar["epsilon_label"] = eps_label
                    rows.append(ar)

            print(f"  Done {task_name}/{state}")

    if not rows:
        raise RuntimeError("No audit rows produced. Check data availability and parameters.")

    detailed_df = pd.DataFrame(rows)
    summary_df = summarize(detailed_df)

    detailed_path = out_dir / "dp_bayesian_audit_detailed.csv"
    summary_path  = out_dir / "dp_bayesian_audit_summary.csv"
    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved detailed: {detailed_path}")
    print(f"Saved summary:  {summary_path}")

    plot_main_panels(summary_df, out_dir, delta=args.delta, fpr=args.fpr)
    plot_sparsity_panels(summary_df, out_dir, fpr=args.fpr)
    print(f"Saved plots in: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian membership-inference audit for DP contingency-table releases. "
            "Implements four attack variants: naive LRT, Bayes single-table, "
            "Bayes two-table, and Bayes joint (all K feature-label tables). "
            "Results are stratified by cell sparsity (sparse/medium/dense)."
        )
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["ACSIncome", "ACSEmployment", "ACSPublicCoverage", "ACSTravelTime"],
        choices=list(TASK_OBJECTS.keys()),
        help="ACS tasks to audit.",
    )
    parser.add_argument(
        "--states", nargs="+", default=["CA", "TX"],
        help="Client states to audit (default: CA TX).",
    )
    parser.add_argument(
        "--prior-states", nargs="+", default=None,
        help=(
            "States to include in the population-level prior. "
            "Defaults to all 50 US states. Use a subset for faster runs."
        ),
    )
    parser.add_argument(
        "--epsilons", nargs="+",
        default=["0.1", "0.5", "1.0", "2.0", "5.0", "inf"],
        help="Epsilon values. Use 'inf' for the no-noise baseline.",
    )
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Delta for the Gaussian mechanism.")
    parser.add_argument("--fpr", type=float, default=0.01,
                        help="FPR target for TPR@FPR metric.")
    parser.add_argument("--survey-year", type=str, default="2018",
                        help="ACS survey year.")
    parser.add_argument("--horizon", type=str, default="1-Year",
                        help="ACS horizon.")
    parser.add_argument("--max-records-per-state", type=int, default=4000,
                        help="Maximum records sampled per (state, task, epsilon).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed.")
    parser.add_argument("--output-dir", type=str, default="results/dp_bayesian_audit",
                        help="Output directory for CSV and PNG files.")
    return parser.parse_args()


if __name__ == "__main__":
    run_audit(parse_args())
