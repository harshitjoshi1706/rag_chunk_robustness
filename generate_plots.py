from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATASET_PATH = Path("data/raw/OHR-Bench_v2.parquet")
PLOTS_DIR = Path("plots")

DOCUMENT_HEATMAP_PNG = (
    PLOTS_DIR / "tfidf_cosine_similarity_heatmap.png"
)

DOCUMENT_HEATMAP_PDF = (
    PLOTS_DIR / "tfidf_cosine_similarity_heatmap.pdf"
)

DOMAIN_HEATMAP_PNG = (
    PLOTS_DIR / "tfidf_domain_similarity_heatmap.png"
)

DOMAIN_HEATMAP_PDF = (
    PLOTS_DIR / "tfidf_domain_similarity_heatmap.pdf"
)


def load_clean_documents(
    dataset_path: Path,
) -> pd.DataFrame:
    """
    Load OHRBench and reconstruct complete clean documents.

    Each OHRBench row represents a page. Pages belonging to the
    same document are sorted by page_idx and concatenated using
    the clean ground-truth text (gt_text).

    Returns one row per document.
    """

    print("Loading OHRBench dataset...")

    df = pd.read_parquet(
        dataset_path,
        columns=[
            "domain",
            "doc_name",
            "page_idx",
            "gt_text",
        ],
    )

    print(
        f"Pages loaded: {len(df):,}"
    )

    # Remove rows lacking the information required for
    # document reconstruction.
    df = df.dropna(
        subset=[
            "domain",
            "doc_name",
            "page_idx",
            "gt_text",
        ]
    ).copy()

    df["gt_text"] = (
        df["gt_text"]
        .astype(str)
    )

    # Ignore completely empty page text.
    df = df[
        df["gt_text"].str.strip() != ""
    ].copy()

    # Preserve original page order.
    df = df.sort_values(
        [
            "domain",
            "doc_name",
            "page_idx",
        ]
    )

    documents = (
        df.groupby(
            [
                "domain",
                "doc_name",
            ],
            as_index=False,
        )
        .agg(
            clean_text=(
                "gt_text",
                lambda pages: "\n\n".join(
                    pages
                ),
            ),
            num_pages=(
                "page_idx",
                "count",
            ),
        )
    )

    # Sorting by domain makes domain-specific patterns
    # easier to inspect in the full similarity matrix.
    documents = (
        documents
        .sort_values(
            [
                "domain",
                "doc_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return documents


def compute_tfidf_similarity(
    documents: pd.DataFrame,
):
    """
    Create TF-IDF representations and compute pairwise
    document cosine similarity.

    Configuration:
    - bigrams only
    - English stop-word removal
    - retain bigrams appearing in at least two documents
    """

    print(
        "\nBuilding TF-IDF document representation..."
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(2, 2),
        stop_words="english",
        min_df=2,
        dtype=np.float32,
    )

    tfidf_matrix = (
        vectorizer.fit_transform(
            documents["clean_text"]
        )
    )

    print(
        "TF-IDF matrix shape: "
        f"{tfidf_matrix.shape[0]:,} documents × "
        f"{tfidf_matrix.shape[1]:,} bigrams"
    )

    print(
        "Computing pairwise cosine similarity..."
    )

    similarity_matrix = (
        cosine_similarity(
            tfidf_matrix,
            dense_output=True,
        )
        .astype(
            np.float32
        )
    )

    return (
        similarity_matrix,
        vectorizer,
    )


def get_domain_layout(
    documents: pd.DataFrame,
):
    """
    Calculate domain positions and boundaries for
    the document-level heatmap.
    """

    domain_counts = (
        documents.groupby(
            "domain",
            sort=False,
        )
        .size()
    )

    domain_labels = []
    domain_centers = []
    domain_boundaries = []

    start = 0

    for domain, count in domain_counts.items():
        end = (
            start + count
        )

        domain_labels.append(
            domain.capitalize()
        )

        domain_centers.append(
            (start + end) / 2
        )

        if end < len(documents):
            domain_boundaries.append(
                end
            )

        start = end

    return (
        domain_labels,
        domain_centers,
        domain_boundaries,
        domain_counts,
    )


def print_similarity_statistics(
    similarity_matrix: np.ndarray,
) -> None:
    """
    Print document-to-document corpus similarity statistics.

    Diagonal self-similarity values are excluded.
    """

    num_documents = (
        similarity_matrix.shape[0]
    )

    upper_triangle = (
        similarity_matrix[
            np.triu_indices(
                num_documents,
                k=1,
            )
        ]
    )

    print(
        "\nCorpus similarity statistics:"
    )

    print(
        "  Mean pairwise similarity: "
        f"{upper_triangle.mean():.4f}"
    )

    print(
        "  Median pairwise similarity: "
        f"{np.median(upper_triangle):.4f}"
    )

    print(
        "  Maximum non-self similarity: "
        f"{upper_triangle.max():.4f}"
    )

    print(
        "  95th percentile similarity: "
        f"{np.percentile(upper_triangle, 95):.4f}"
    )

    print(
        "  99th percentile similarity: "
        f"{np.percentile(upper_triangle, 99):.4f}"
    )


def plot_document_similarity_heatmap(
    similarity_matrix: np.ndarray,
    documents: pd.DataFrame,
) -> None:
    """
    Generate the complete 1,261-document cosine-similarity heatmap.

    The diagonal is masked because every document has cosine
    similarity 1.0 with itself.

    Color scaling uses the 99th percentile of non-self similarity
    values so that low-to-moderate similarities become visually
    distinguishable.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        domain_labels,
        domain_centers,
        domain_boundaries,
        domain_counts,
    ) = get_domain_layout(
        documents
    )

    print(
        "\nDocuments by domain:"
    )

    for domain, count in domain_counts.items():
        print(
            f"  {domain}: {count}"
        )

    diagonal_mask = np.eye(
        similarity_matrix.shape[0],
        dtype=bool,
    )

    non_self_values = (
        similarity_matrix[
            ~diagonal_mask
        ]
    )

    visualization_vmax = (
        np.percentile(
            non_self_values,
            99,
        )
    )

    print(
        "\nDocument heatmap color-scale maximum "
        "(99th percentile): "
        f"{visualization_vmax:.4f}"
    )

    sns.set_context(
        "paper"
    )

    sns.set_style(
        "white"
    )

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            6.2,
        )
    )

    sns.heatmap(
        similarity_matrix,
        mask=diagonal_mask,
        ax=ax,
        cmap="viridis",
        vmin=0.0,
        vmax=visualization_vmax,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={
            "label":
                "TF-IDF cosine similarity",
            "shrink":
                0.82,
        },
        rasterized=True,
    )

    # Domain boundaries.
    for boundary in domain_boundaries:
        ax.axhline(
            boundary,
            linewidth=0.7,
            color="white",
        )

        ax.axvline(
            boundary,
            linewidth=0.7,
            color="white",
        )

    ax.set_xticks(
        domain_centers
    )

    ax.set_xticklabels(
        domain_labels,
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_yticks(
        domain_centers
    )

    ax.set_yticklabels(
        domain_labels,
        rotation=0,
        fontsize=8,
    )

    ax.set_xlabel(
        "Documents grouped by domain"
    )

    ax.set_ylabel(
        "Documents grouped by domain"
    )

    # Intentionally no title inside the figure.
    # IEEE caption will describe the figure.

    fig.tight_layout()

    fig.savefig(
        DOCUMENT_HEATMAP_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        DOCUMENT_HEATMAP_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "\nSaved full document-level heatmap:"
    )

    print(
        f"  {DOCUMENT_HEATMAP_PNG}"
    )

    print(
        f"  {DOCUMENT_HEATMAP_PDF}"
    )


def compute_domain_similarity_matrix(
    similarity_matrix: np.ndarray,
    documents: pd.DataFrame,
):
    """
    Aggregate the complete document-level similarity matrix
    into mean domain-to-domain similarities.

    Within-domain values exclude each document's self-similarity.
    """

    domains = (
        documents[
            "domain"
        ]
        .unique()
        .tolist()
    )

    document_domains = (
        documents[
            "domain"
        ]
        .to_numpy()
    )

    domain_indices = {
        domain: np.where(
            document_domains
            == domain
        )[0]
        for domain in domains
    }

    domain_matrix = np.zeros(
        (
            len(domains),
            len(domains),
        ),
        dtype=np.float32,
    )

    for i, domain_a in enumerate(
        domains
    ):
        indices_a = (
            domain_indices[
                domain_a
            ]
        )

        for j, domain_b in enumerate(
            domains
        ):
            indices_b = (
                domain_indices[
                    domain_b
                ]
            )

            block = (
                similarity_matrix[
                    np.ix_(
                        indices_a,
                        indices_b,
                    )
                ]
            )

            if domain_a == domain_b:
                # Remove document self-similarity
                # from within-domain averages.
                if len(indices_a) > 1:
                    block_mask = (
                        ~np.eye(
                            len(indices_a),
                            dtype=bool,
                        )
                    )

                    values = (
                        block[
                            block_mask
                        ]
                    )

                else:
                    values = (
                        np.array(
                            [],
                            dtype=np.float32,
                        )
                    )

            else:
                values = (
                    block.ravel()
                )

            if values.size > 0:
                domain_matrix[
                    i,
                    j,
                ] = (
                    values.mean()
                )

    return (
        domain_matrix,
        domains,
    )


def print_domain_similarity_matrix(
    domain_matrix: np.ndarray,
    domains: list,
) -> None:
    """
    Print domain-level mean cosine similarities.
    """

    domain_dataframe = pd.DataFrame(
        domain_matrix,
        index=[
            domain.capitalize()
            for domain in domains
        ],
        columns=[
            domain.capitalize()
            for domain in domains
        ],
    )

    print(
        "\nMean domain-to-domain similarity:"
    )

    print(
        domain_dataframe.round(
            4
        )
    )


def plot_domain_similarity_heatmap(
    similarity_matrix: np.ndarray,
    documents: pd.DataFrame,
) -> None:
    """
    Generate an annotated domain-level similarity heatmap.

    This is derived from the full document-level matrix rather
    than from a sampled subset of documents.

    Only the lower triangle is displayed for readability.
    """

    (
        domain_matrix,
        domains,
    ) = compute_domain_similarity_matrix(
        similarity_matrix,
        documents,
    )

    print_domain_similarity_matrix(
        domain_matrix,
        domains,
    )

    domain_labels = [
        domain.capitalize()
        for domain in domains
    ]

    # Hide the upper triangle while keeping
    # the diagonal visible.
    upper_triangle_mask = (
        np.triu(
            np.ones_like(
                domain_matrix,
                dtype=bool,
            ),
            k=1,
        )
    )

    # Use the maximum domain-level value for the
    # visible color range rather than forcing 0–1.
    visible_values = (
        domain_matrix[
            ~upper_triangle_mask
        ]
    )

    domain_vmax = (
        visible_values.max()
    )

    print(
        "\nDomain heatmap color-scale maximum: "
        f"{domain_vmax:.4f}"
    )

    sns.set_context(
        "paper"
    )

    sns.set_style(
        "white"
    )

    fig, ax = plt.subplots(
        figsize=(
            6.5,
            5.5,
        )
    )

    sns.heatmap(
        domain_matrix,
        mask=upper_triangle_mask,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.0,
        vmax=domain_vmax,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=domain_labels,
        yticklabels=domain_labels,
        cbar_kws={
            "label":
                "Mean TF-IDF cosine similarity",
            "shrink":
                0.80,
        },
        ax=ax,
    )

    ax.set_xlabel(
        ""
    )

    ax.set_ylabel(
        ""
    )

    ax.tick_params(
        axis="x",
        labelrotation=45,
    )

    for label in (
        ax.get_xticklabels()
    ):
        label.set_horizontalalignment(
            "right"
        )

    ax.tick_params(
        axis="y",
        labelrotation=0,
    )

    # Intentionally no title inside the figure.

    fig.tight_layout()

    fig.savefig(
        DOMAIN_HEATMAP_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        DOMAIN_HEATMAP_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "\nSaved domain-level heatmap:"
    )

    print(
        f"  {DOMAIN_HEATMAP_PNG}"
    )

    print(
        f"  {DOMAIN_HEATMAP_PDF}"
    )


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "OHRBench dataset not found at: "
            f"{DATASET_PATH}"
        )

    documents = (
        load_clean_documents(
            DATASET_PATH
        )
    )

    print(
        "\nClean documents reconstructed: "
        f"{len(documents):,}"
    )

    print(
        "Domains represented: "
        f"{documents['domain'].nunique()}"
    )

    (
        similarity_matrix,
        vectorizer,
    ) = compute_tfidf_similarity(
        documents
    )

    print(
        "\nRetained bigram vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print_similarity_statistics(
        similarity_matrix
    )

    # Figure 1:
    # Complete document-to-document similarity matrix.
    plot_document_similarity_heatmap(
        similarity_matrix,
        documents,
    )

    # Figure 2:
    # Readable domain-level aggregation with numeric values.
    plot_domain_similarity_heatmap(
        similarity_matrix,
        documents,
    )

    print(
        "\nTF-IDF corpus characterization completed successfully."
    )


if __name__ == "__main__":
    main()