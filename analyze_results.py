"""Aggregate saved query results and paired bootstrap uncertainty estimates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
NOISE = ("clean", "mild", "moderate", "severe")
CHUNKERS = ("fixed", "recursive", "semantic", "structure")
METRICS = ("hit_at_1", "hit_at_5", "reciprocal_rank")
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 20260913


def validate_query_results(query_results):
    expected = None
    for chunker in CHUNKERS:
        for noise in NOISE:
            group = query_results[(query_results.chunker == chunker)
                                  & (query_results.noise_level == noise)]
            ids = set(group.query_id)
            if len(group) != 8456 or len(ids) != 8456:
                raise ValueError(f"Incomplete or duplicate query results for {chunker}/{noise}")
            if expected is None:
                expected = ids
            elif ids != expected:
                raise ValueError("Configurations do not share exactly the same QA IDs")
    if len(query_results) != 16 * 8456:
        raise ValueError("Unexpected query result count")
    return sorted(expected)


def aggregate(query_results, subset=False):
    selected = query_results[query_results.context_verified == 1] if subset else query_results
    result = (selected.groupby(["chunker", "noise_level"], sort=False)
              .agg(num_queries=("query_id", "size"), hit_at_1=("hit_at_1", "mean"),
                   hit_at_5=("hit_at_5", "mean"), mrr_at_100=("reciprocal_rank", "mean"))
              .reset_index())
    result["evaluation_set"] = "context_verified" if subset else "primary"
    return result


def degradation(metrics):
    clean = metrics[metrics.noise_level == "clean"].set_index("chunker")
    rows = []
    for row in metrics.itertuples(index=False):
        if row.noise_level == "clean":
            continue
        for metric in ("hit_at_1", "hit_at_5", "mrr_at_100"):
            baseline = float(clean.loc[row.chunker, metric])
            current = float(getattr(row, metric))
            rows.append({"evaluation_set": row.evaluation_set, "chunker": row.chunker,
                         "noise_level": row.noise_level, "metric": metric,
                         "clean_value": baseline, "noisy_value": current,
                         "absolute_degradation": baseline - current,
                         "relative_degradation_percent":
                             100 * (baseline - current) / baseline if baseline > 0 else np.nan})
    return pd.DataFrame(rows)


def paired_bootstrap(query_results, subset=False, iterations=BOOTSTRAP_ITERATIONS,
                     seed=BOOTSTRAP_SEED):
    selected = query_results[query_results.context_verified == 1] if subset else query_results
    ids = sorted(selected.query_id.unique())
    expected_n = 5277 if subset else 8456
    if len(ids) != expected_n:
        raise ValueError("Unexpected paired QA count")
    labels = [(chunker, noise) for chunker in CHUNKERS for noise in NOISE]
    arrays = []
    for chunker, noise in labels:
        group = selected[(selected.chunker == chunker) & (selected.noise_level == noise)]
        group = group.set_index("query_id").loc[ids]
        arrays.append(group[list(METRICS)].to_numpy(dtype=np.float32))
    data = np.stack(arrays, axis=1)  # QA x configuration x metric
    rng = np.random.default_rng(seed)
    draws = np.empty((iterations, len(labels), len(METRICS)), dtype=np.float32)
    for start in range(0, iterations, 64):
        count = min(64, iterations - start)
        indices = rng.integers(0, len(ids), size=(count, len(ids)))
        draws[start:start + count] = data[indices].mean(axis=1)
    rows = []
    set_name = "context_verified" if subset else "primary"
    metric_names = ("hit_at_1", "hit_at_5", "mrr_at_100")
    for position, (chunker, noise) in enumerate(labels):
        for metric_index, metric in enumerate(metric_names):
            low, high = np.quantile(draws[:, position, metric_index], [0.025, 0.975])
            rows.append({"evaluation_set": set_name, "statistic": "mean",
                         "chunker": chunker, "noise_level": noise, "metric": metric,
                         "estimate": float(data[:, position, metric_index].mean()),
                         "ci_95_low": float(low), "ci_95_high": float(high),
                         "bootstrap_iterations": iterations, "seed": seed})
    for chunker_index, chunker in enumerate(CHUNKERS):
        clean_position = 4 * chunker_index
        for noise_offset, noise in enumerate(NOISE[1:], 1):
            noisy_position = clean_position + noise_offset
            for metric_index, metric in enumerate(metric_names):
                differences = draws[:, clean_position, metric_index] - draws[:, noisy_position, metric_index]
                low, high = np.quantile(differences, [0.025, 0.975])
                rows.append({"evaluation_set": set_name, "statistic": "absolute_degradation",
                             "chunker": chunker, "noise_level": noise, "metric": metric,
                             "estimate": float(data[:, clean_position, metric_index].mean()
                                               - data[:, noisy_position, metric_index].mean()),
                             "ci_95_low": float(low), "ci_95_high": float(high),
                             "bootstrap_iterations": iterations, "seed": seed})
    for chunker_index, chunker in enumerate(CHUNKERS[1:], 1):
        for noise_offset, noise in enumerate(NOISE[1:], 1):
            for metric_index, metric in enumerate(metric_names):
                candidate = draws[:, 4 * chunker_index, metric_index] - draws[:, 4 * chunker_index + noise_offset, metric_index]
                fixed = draws[:, 0, metric_index] - draws[:, noise_offset, metric_index]
                contrast = candidate - fixed
                low, high = np.quantile(contrast, [0.025, 0.975])
                rows.append({"evaluation_set": set_name, "statistic": "degradation_difference_vs_fixed",
                             "chunker": chunker, "noise_level": noise, "metric": metric,
                             "estimate": float((data[:, 4 * chunker_index, metric_index]
                                                - data[:, 4 * chunker_index + noise_offset, metric_index]).mean()
                                               - (data[:, 0, metric_index]
                                                  - data[:, noise_offset, metric_index]).mean()),
                             "ci_95_low": float(low), "ci_95_high": float(high),
                             "bootstrap_iterations": iterations, "seed": seed})
    return pd.DataFrame(rows)


def main():
    query_results = pd.read_csv(RESULTS / "query_results.csv")
    configurations = pd.read_csv(RESULTS / "configuration_results.csv")
    validate_query_results(query_results)
    if len(configurations) != 16 or not (configurations.status == "complete").all():
        raise ValueError("All 16 configurations must finish before final analysis")
    primary = aggregate(query_results)
    sensitivity = aggregate(query_results, subset=True)
    if not (primary.num_queries == 8456).all() or not (sensitivity.num_queries == 5277).all():
        raise ValueError("Aggregate QA counts changed")
    primary.to_csv(RESULTS / "final_metrics.csv", index=False)
    sensitivity.to_csv(RESULTS / "sensitivity_metrics.csv", index=False)
    pd.concat([degradation(primary), degradation(sensitivity)], ignore_index=True).to_csv(
        RESULTS / "robustness_degradation.csv", index=False)
    pd.concat([paired_bootstrap(query_results), paired_bootstrap(query_results, subset=True)],
              ignore_index=True).to_csv(RESULTS / "statistical_results.csv", index=False)
    configurations.to_csv(RESULTS / "efficiency_table.csv", index=False)
    print(primary.to_string(index=False))
    print("Saved primary, sensitivity, degradation, statistics, and efficiency tables")


if __name__ == "__main__":
    main()
