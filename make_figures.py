"""Generate IEEE-friendly figures only from saved final result tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
NOISE = ("clean", "mild", "moderate", "severe")
CHUNKERS = ("fixed", "recursive", "semantic", "structure")
STYLE = {
    "fixed": ("o", "-", "0.10"),
    "recursive": ("s", "--", "0.35"),
    "semantic": ("^", "-.", "0.55"),
    "structure": ("D", ":", "0.75"),
}


def draw_lines(table, value, ylabel, filename):
    fig, ax = plt.subplots(figsize=(6.7, 3.7))
    for chunker in CHUNKERS:
        subset = table[table.chunker == chunker].set_index("noise_level").loc[list(NOISE)]
        marker, linestyle, color = STYLE[chunker]
        ax.plot(range(4), subset[value], label=chunker.capitalize(), marker=marker,
                linestyle=linestyle, color=color, linewidth=1.6, markersize=5)
    ax.set_xticks(range(4), [item.capitalize() for item in NOISE])
    ax.set_xlabel("Formatting condition")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.87", linewidth=0.5)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / f"{filename}.png", dpi=600, bbox_inches="tight")
    fig.savefig(PLOTS / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    primary = pd.read_csv(RESULTS / "final_metrics.csv")
    sensitivity = pd.read_csv(RESULTS / "sensitivity_metrics.csv")
    configs = pd.read_csv(RESULTS / "configuration_results.csv")
    degradation = pd.read_csv(RESULTS / "robustness_degradation.csv")
    draw_lines(primary, "hit_at_1", "Hit@1", "hit_at_1_vs_noise")
    draw_lines(primary, "hit_at_5", "Hit@5", "hit_at_5_vs_noise")
    draw_lines(primary, "mrr_at_100", "MRR@100", "mrr_at_100_vs_noise")
    degradation = degradation[(degradation.evaluation_set == "primary")
                              & (degradation.metric == "hit_at_1")]
    baseline = pd.DataFrame([{"chunker": chunker, "noise_level": "clean",
                              "absolute_degradation": 0.0} for chunker in CHUNKERS])
    draw_lines(pd.concat([degradation, baseline]), "absolute_degradation",
               "Absolute Hit@1 degradation from clean", "hit_at_1_degradation")
    draw_lines(configs, "num_chunks", "Number of chunks", "chunk_count_vs_noise")
    draw_lines(configs, "mean_chunk_tokens", "Mean chunk length (tokens)",
               "mean_chunk_tokens_vs_noise")
    configs = configs.copy()
    configs["processing_time_sec"] = (configs.chunking_time_sec + configs.embedding_time_sec
                                       + configs.index_build_time_sec + configs.total_retrieval_time_sec)
    draw_lines(configs, "processing_time_sec", "Processing time (seconds)", "runtime_vs_noise")
    draw_lines(configs, "mean_query_latency_ms", "Mean query latency (ms)", "query_latency_vs_noise")
    draw_lines(sensitivity, "hit_at_1", "Hit@1, context-verified QAs", "sensitivity_hit_at_1")
    print("Saved 9 figures as PNG and PDF under plots/")


if __name__ == "__main__":
    main()
