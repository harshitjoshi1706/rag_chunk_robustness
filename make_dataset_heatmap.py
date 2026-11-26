from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Domain-wise counts from your experiment audit
data = {
    "Domain": [
        "Academic",
        "Administration",
        "Finance",
        "Law",
        "Manual",
        "News",
        "Textbook",
    ],
    "Primary QAs": [1179, 1332, 1367, 1142, 1151, 1202, 1083],
    "Context-verified QAs": [643, 814, 753, 828, 805, 746, 688],
}

df = pd.DataFrame(data).set_index("Domain")

# Output folder
plots_dir = Path("plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# Style
sns.set_theme(style="white", font_scale=1.0)

plt.figure(figsize=(6.4, 4.8))
ax = sns.heatmap(
    df,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Number of questions"},
)

# No title inside figure (IEEE style)
ax.set_xlabel("Evaluation set")
ax.set_ylabel("Document domain")
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()

png_path = plots_dir / "dataset_distribution_heatmap.png"
pdf_path = plots_dir / "dataset_distribution_heatmap.pdf"

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")