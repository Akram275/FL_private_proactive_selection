import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage, ACSTravelTime
from sklearn.metrics import roc_auc_score


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


def sigma_from_epsilon(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    if math.isinf(epsilon):
        return 0.0
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 or inf")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    unique = np.unique(y_true)
    if unique.size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float = 0.01) -> float:
    negatives = y_score[y_true == 0]
    positives = y_score[y_true == 1]
    if negatives.size == 0 or positives.size == 0:
        return float("nan")
    threshold = float(np.quantile(negatives, 1.0 - fpr_target))
    return float(np.mean(positives >= threshold))


def load_state_task_dataframe(
    data_source: ACSDataSource,
    task_obj,
    state: str,
) -> pd.DataFrame:
    acs_data = data_source.get_data(states=[state], download=True)
    out = task_obj.df_to_pandas(acs_data)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError("task_obj.df_to_pandas must return at least (X, y)")

    x_df = out[0]
    y_s = out[1]

    y_name = str(getattr(task_obj, "target", "target"))

    # Folktables may return labels as Series, DataFrame, or ndarray with shape (n, 1).
    y_arr = np.asarray(y_s)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr[:, 0]
    elif y_arr.ndim > 1:
        y_arr = y_arr.reshape(-1)

    y_series = pd.Series(y_arr).astype(int)
    y_series.name = y_name

    df = x_df.copy()
    if len(df) != len(y_series):
        raise ValueError(f"Length mismatch between features ({len(df)}) and labels ({len(y_series)})")
    df[y_name] = y_series.values
    return df


def make_single_table_scores(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    sigma: float,
    max_records: int,
    rng: np.random.Generator,
    use_noisy_total_estimate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_col not in df.columns:
        raise ValueError(f"Feature column '{feature_col}' not found")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    n = min(max_records, len(df))
    sampled = df.sample(n=n, random_state=int(rng.integers(0, 10_000_000))).reset_index(drop=True)

    counts = sampled.groupby([feature_col, target_col], dropna=False).size().to_dict()

    feature_vals = sampled[feature_col].tolist()
    target_vals = sampled[target_col].tolist()

    baseline = np.empty(n, dtype=float)
    for i in range(n):
        key = (feature_vals[i], target_vals[i])
        baseline[i] = float(counts.get(key, 0) - 1)

    noise_pos = rng.normal(0.0, sigma, size=n) if sigma > 0 else np.zeros(n)
    noise_neg = rng.normal(0.0, sigma, size=n) if sigma > 0 else np.zeros(n)

    score_pos = baseline + 1.0 + noise_pos
    score_neg = baseline + noise_neg

    if use_noisy_total_estimate and sigma > 0:
        n_x = max(1, sampled[feature_col].nunique(dropna=False))
        n_y = max(1, sampled[target_col].nunique(dropna=False))
        n_cells = n_x * n_y
        sigma_total = math.sqrt(n_cells) * sigma

        n_minus_r = max(1.0, float(n - 1))
        n_hat_pos = n_minus_r + 1.0 + rng.normal(0.0, sigma_total, size=n)
        n_hat_neg = n_minus_r + rng.normal(0.0, sigma_total, size=n)

        baseline_est_pos = baseline * (n_hat_pos / n_minus_r)
        baseline_est_neg = baseline * (n_hat_neg / n_minus_r)
    else:
        baseline_est_pos = baseline
        baseline_est_neg = baseline

    residual_pos = score_pos - baseline_est_pos
    residual_neg = score_neg - baseline_est_neg

    y_true = np.concatenate([np.ones(n), np.zeros(n)])
    y_score = np.concatenate([residual_pos, residual_neg])

    sparse_mask = baseline <= 1.0
    sparse_y_true = np.concatenate([np.ones(np.sum(sparse_mask)), np.zeros(np.sum(sparse_mask))])
    sparse_y_score = np.concatenate([residual_pos[sparse_mask], residual_neg[sparse_mask]])

    return y_true, y_score, (sparse_y_true, sparse_y_score)


def make_joint_table_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sigma_per_table: float,
    max_records: int,
    rng: np.random.Generator,
    use_noisy_total_estimate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(max_records, len(df))
    sampled = df.sample(n=n, random_state=int(rng.integers(0, 10_000_000))).reset_index(drop=True)

    per_feature_baselines = []
    for feature_col in feature_cols:
        counts = sampled.groupby([feature_col, target_col], dropna=False).size().to_dict()
        vals = sampled[feature_col].tolist()
        targets = sampled[target_col].tolist()
        baseline = np.empty(n, dtype=float)
        for i in range(n):
            key = (vals[i], targets[i])
            baseline[i] = float(counts.get(key, 0) - 1)
        per_feature_baselines.append(baseline)

    baseline_matrix = np.vstack(per_feature_baselines)
    sparse_mask = np.any(baseline_matrix <= 1.0, axis=0)

    m = len(feature_cols)
    if sigma_per_table > 0:
        noise_pos = rng.normal(0.0, sigma_per_table, size=(m, n))
        noise_neg = rng.normal(0.0, sigma_per_table, size=(m, n))
    else:
        noise_pos = np.zeros((m, n))
        noise_neg = np.zeros((m, n))

    score_pos_matrix = baseline_matrix + 1.0 + noise_pos
    score_neg_matrix = baseline_matrix + noise_neg

    if use_noisy_total_estimate and sigma_per_table > 0:
        n_minus_r = max(1.0, float(n - 1))
        baseline_est_pos_matrix = np.empty_like(baseline_matrix)
        baseline_est_neg_matrix = np.empty_like(baseline_matrix)

        for j, feature_col in enumerate(feature_cols):
            n_x = max(1, sampled[feature_col].nunique(dropna=False))
            n_y = max(1, sampled[target_col].nunique(dropna=False))
            n_cells_j = n_x * n_y
            sigma_total_j = math.sqrt(n_cells_j) * sigma_per_table

            n_hat_pos_j = n_minus_r + 1.0 + rng.normal(0.0, sigma_total_j, size=n)
            n_hat_neg_j = n_minus_r + rng.normal(0.0, sigma_total_j, size=n)

            baseline_est_pos_matrix[j, :] = baseline_matrix[j, :] * (n_hat_pos_j / n_minus_r)
            baseline_est_neg_matrix[j, :] = baseline_matrix[j, :] * (n_hat_neg_j / n_minus_r)
    else:
        baseline_est_pos_matrix = baseline_matrix
        baseline_est_neg_matrix = baseline_matrix

    residual_pos = (score_pos_matrix - baseline_est_pos_matrix).sum(axis=0)
    residual_neg = (score_neg_matrix - baseline_est_neg_matrix).sum(axis=0)

    y_true = np.concatenate([np.ones(n), np.zeros(n)])
    y_score = np.concatenate([residual_pos, residual_neg])

    sparse_y_true = np.concatenate([np.ones(np.sum(sparse_mask)), np.zeros(np.sum(sparse_mask))])
    sparse_y_score = np.concatenate([residual_pos[sparse_mask], residual_neg[sparse_mask]])

    return y_true, y_score, (sparse_y_true, sparse_y_score)


def summarize_by_task(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["task", "attack", "epsilon", "epsilon_label"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            tpr01_mean=("tpr_at_1pct", "mean"),
            tpr01_std=("tpr_at_1pct", "std"),
            tpr01_sparse_mean=("tpr_at_1pct_sparse", "mean"),
            tpr01_sparse_std=("tpr_at_1pct_sparse", "std"),
            n_states=("state", "nunique"),
        )
        .sort_values(["task", "attack", "epsilon"], kind="stable")
    )
    return grouped


def plot_task_panels(summary_df: pd.DataFrame, out_dir: Path, delta: float, fpr: float = 0.01) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    tasks = summary_df["task"].unique().tolist()

    for task in tasks:
        task_df = summary_df[summary_df["task"] == task].copy()
        eps_labels = task_df["epsilon_label"].drop_duplicates().tolist()
        eps_vals = task_df["epsilon"].drop_duplicates().tolist()

        x_positions = np.arange(len(eps_labels))
        eps_map = {label: idx for idx, label in enumerate(eps_labels)}

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        axis_label_fs = 24
        tick_label_fs = 22
        legend_fs = 22

        for attack_name, color in [("single", "#1f77b4"), ("joint", "#d62728")]:
            sub = task_df[task_df["attack"] == attack_name].copy()
            if sub.empty:
                continue
            xs = np.array([eps_map[v] for v in sub["epsilon_label"]])

            axes[0].plot(xs, sub["auc_mean"], marker="o", linewidth=2.2, color=color, label=f"{attack_name.title()} LRT")
            axes[0].fill_between(xs, sub["auc_mean"] - sub["auc_std"], sub["auc_mean"] + sub["auc_std"], color=color, alpha=0.15)

            axes[1].plot(xs, sub["tpr01_mean"], marker="o", linewidth=2.2, color=color, label=f"{attack_name.title()} LRT")
            axes[1].fill_between(xs, sub["tpr01_mean"] - sub["tpr01_std"], sub["tpr01_mean"] + sub["tpr01_std"], color=color, alpha=0.15)

        for ax in axes:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(eps_labels, fontsize=tick_label_fs)
            ax.set_xlabel("Epsilon", fontsize=axis_label_fs)
            ax.tick_params(axis="both", labelsize=tick_label_fs)

        axes[0].set_ylabel("AUC", fontsize=axis_label_fs)
        axes[0].set_ylim(0.45, 1.02)

        axes[1].set_ylabel("TPR", fontsize=axis_label_fs)
        axes[1].set_yscale("log")
        axes[1].set_ylim(1e-3, 1.05)

        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, labels, fontsize=legend_fs, loc="upper left")

        out_path = out_dir / f"{task}_audit_panels.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)


def run_audit(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_values = []
    for raw in args.epsilons:
        if str(raw).lower() in {"inf", "infty", "infinity"}:
            eps_values.append(float("inf"))
        else:
            eps_values.append(float(raw))

    data_source = ACSDataSource(survey_year=args.survey_year, horizon=args.horizon, survey="person")
    rng = np.random.default_rng(args.seed)

    rows = []

    for task_name in args.tasks:
        task_obj = TASK_OBJECTS[task_name]
        for state in args.states:
            try:
                df = load_state_task_dataframe(data_source, task_obj, state)
            except Exception as e:
                print(f"Skipping {task_name}/{state}: {e}")
                continue

            if len(df) < 50:
                print(f"Skipping {task_name}/{state}: too few rows ({len(df)})")
                continue

            target_col = str(getattr(task_obj, "target", "target"))
            feature_cols = [c for c in getattr(task_obj, "features", []) if c in df.columns]
            if not feature_cols:
                print(f"Skipping {task_name}/{state}: no valid feature columns found")
                continue

            single_feature = feature_cols[0]

            for eps in eps_values:
                sigma_single = sigma_from_epsilon(eps, args.delta)

                if args.joint_scale_with_m:
                    sigma_joint = sigma_single * math.sqrt(len(feature_cols))
                else:
                    sigma_joint = sigma_single

                y_true_s, y_score_s, sparse_s = make_single_table_scores(
                    df=df,
                    feature_col=single_feature,
                    target_col=target_col,
                    sigma=sigma_single,
                    max_records=args.max_records_per_state,
                    rng=rng,
                    use_noisy_total_estimate=args.use_noisy_total_estimate,
                )
                auc_s = safe_auc(y_true_s, y_score_s)
                tpr_s = tpr_at_fpr(y_true_s, y_score_s, fpr_target=args.fpr)
                auc_sparse_s = safe_auc(sparse_s[0], sparse_s[1]) if sparse_s[0].size > 0 else float("nan")
                tpr_sparse_s = tpr_at_fpr(sparse_s[0], sparse_s[1], fpr_target=args.fpr) if sparse_s[0].size > 0 else float("nan")

                rows.append({
                    "task": task_name,
                    "state": state,
                    "attack": "single",
                    "epsilon": eps,
                    "epsilon_label": "inf" if math.isinf(eps) else str(eps),
                    "sigma": sigma_single,
                    "auc": auc_s,
                    "tpr_at_1pct": tpr_s,
                    "auc_sparse": auc_sparse_s,
                    "tpr_at_1pct_sparse": tpr_sparse_s,
                    "n_records": min(args.max_records_per_state, len(df)),
                    "single_feature": single_feature,
                    "n_features_joint": len(feature_cols),
                })

                y_true_j, y_score_j, sparse_j = make_joint_table_scores(
                    df=df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    sigma_per_table=sigma_joint,
                    max_records=args.max_records_per_state,
                    rng=rng,
                    use_noisy_total_estimate=args.use_noisy_total_estimate,
                )
                auc_j = safe_auc(y_true_j, y_score_j)
                tpr_j = tpr_at_fpr(y_true_j, y_score_j, fpr_target=args.fpr)
                auc_sparse_j = safe_auc(sparse_j[0], sparse_j[1]) if sparse_j[0].size > 0 else float("nan")
                tpr_sparse_j = tpr_at_fpr(sparse_j[0], sparse_j[1], fpr_target=args.fpr) if sparse_j[0].size > 0 else float("nan")

                rows.append({
                    "task": task_name,
                    "state": state,
                    "attack": "joint",
                    "epsilon": eps,
                    "epsilon_label": "inf" if math.isinf(eps) else str(eps),
                    "sigma": sigma_joint,
                    "auc": auc_j,
                    "tpr_at_1pct": tpr_j,
                    "auc_sparse": auc_sparse_j,
                    "tpr_at_1pct_sparse": tpr_sparse_j,
                    "n_records": min(args.max_records_per_state, len(df)),
                    "single_feature": single_feature,
                    "n_features_joint": len(feature_cols),
                })

            print(f"Completed {task_name}/{state}")

    if not rows:
        raise RuntimeError("No audit rows were produced. Check data availability and parameters.")

    detailed_df = pd.DataFrame(rows)
    summary_df = summarize_by_task(detailed_df)

    detailed_path = out_dir / "dp_audit_detailed.csv"
    summary_path = out_dir / "dp_audit_summary.csv"

    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_task_panels(summary_df=summary_df, out_dir=out_dir, delta=args.delta, fpr=args.fpr)

    print(f"Saved detailed metrics: {detailed_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(f"Saved plots in: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DP contingency-table release with optimal LRT attacks (single and joint)."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["ACSIncome", "ACSEmployment", "ACSPublicCoverage", "ACSTravelTime"],
        choices=list(TASK_OBJECTS.keys()),
        help="Tasks to audit",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=["CA", "TX"],
        help="State codes to audit (default: CA TX for a fast audit)",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        default=["0.1", "0.5", "1.0", "2.0", "5.0", "inf"],
        help="Epsilon values. Use 'inf' for no-noise baseline.",
    )
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta for Gaussian mechanism")
    parser.add_argument("--fpr", type=float, default=0.01, help="FPR target for TPR@FPR")
    parser.add_argument("--survey-year", type=str, default="2018", help="ACS survey year")
    parser.add_argument("--horizon", type=str, default="1-Year", help="ACS horizon")
    parser.add_argument("--max-records-per-state", type=int, default=4000, help="Max records per state per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--joint-scale-with-m",
        action="store_true",
        help="Scale joint per-table sigma by sqrt(M) to reflect multi-table release calibration.",
    )
    parser.add_argument(
        "--exact-total-knowledge",
        action="store_true",
        help="Disable noisy total-count estimate and assume attacker knows exact total N.",
    )
    parser.add_argument("--output-dir", type=str, default="results/dp_audit", help="Output directory")
    args = parser.parse_args()
    args.use_noisy_total_estimate = not args.exact_total_knowledge
    return args


if __name__ == "__main__":
    run_audit(parse_args())
