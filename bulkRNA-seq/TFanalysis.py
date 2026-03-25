# =============================================================================
# TF Activity Analysis: Identifying Upstream Regulators of ST8SIA2 in pHGG
# =============================================================================
# Author: Fudhail Sayed
# Date: March 2026
#
# PURPOSE:
#   This script identifies transcription factors (TFs) whose activity is
#   correlated with ST8SIA2 expression in pediatric high-grade glioma (pHGG)
#   using bulk RNA-seq data from the Pediatric Brain Tumor Atlas (PBTA).
#
# APPROACH:
#   1. Load PBTA bulk RNA-seq TPM matrix (RSEM, stranded)
#   2. Filter to pHGG samples using clinical metadata
#   3. Infer TF activity using DoRothEA regulons + ULM (decoupler)
#   4. Correlate TF activity scores with ST8SIA2 expression
#   5. Visualize results
#
# INPUT FILES:
#   - pbta-gene-expression-rsem-tpm.stranded.tsv.gz  (expression matrix)
#   - pbta_all_clinical_data0.tsv                    (clinical metadata)
#   Both available from: https://pedcbioportal.kidsfirstdrc.org/
#
# OUTPUT FILES:
#   - pbta_tf_activities_ulm.csv     (TF activity matrix, samples x TFs)
#   - tf_st8sia2_correlations.csv    (ranked TF correlations with ST8SIA2)
#   - tf_st8sia2_results.pdf         (visualization)
#
# DEPENDENCIES:
#   pip install decoupler omnipath pandas numpy scipy statsmodels matplotlib
#   seaborn
# =============================================================================

import decoupler as dc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# =============================================================================
# SECTION 1: Load and Clean Expression Matrix
# =============================================================================
# The PBTA TPM matrix has Ensembl IDs with version numbers and gene symbols
# appended (e.g. ENSG00000000003.14_TSPAN6). We strip the Ensembl prefix
# and keep only the gene symbol for compatibility with DoRothEA.

print("Loading expression matrix...")
mat = pd.read_csv(
    "/Users/fudhailsayed/prololab/datasets/PBTA/pbta-gene-expression-rsem-tpm.stranded.tsv.gz",
    sep="\t",
    index_col=0,
)

# Strip Ensembl ID prefix: ENSG00000000003.14_TSPAN6 → TSPAN6
mat.index = mat.index.str.split("_", n=1).str[1]
print(f"Duplicate gene symbols before collapse: {mat.index.duplicated().sum()}")

# For duplicate gene symbols, keep the one with highest mean expression
mat = mat.groupby(mat.index).mean()

# Transpose so rows = samples, columns = genes (required by decoupler)
mat = mat.T
print(f"Expression matrix shape (samples x genes): {mat.shape}")

# =============================================================================
# SECTION 2: Log Transform
# =============================================================================
# DoRothEA/ULM performs better on log-transformed data.
# We use log2(TPM + 1) which is standard for bulk RNA-seq.

mat_log = np.log2(mat + 1)

# Confirm ST8SIA2 is present and expressed
print(f"\nST8SIA2 in matrix: {'ST8SIA2' in mat_log.columns}")
print(f"ST8SIA2 expression summary:")
print(mat_log["ST8SIA2"].describe())
print(f"Samples with ST8SIA2 > 0: {(mat_log['ST8SIA2'] > 0).sum()}")

# =============================================================================
# SECTION 3: Load Clinical Data and Filter to pHGG
# =============================================================================
# We filter to three pHGG subtypes:
#   - High-grade glioma (HGG)
#   - Diffuse midline glioma (DMG, includes DIPG)
#   - Diffuse hemispheric glioma (DHG)
#
# The expression matrix uses BS_ biospecimen IDs which map to the
# SPECIMEN_ID column in the clinical file (not "Sample ID").
# SPECIMEN_ID can contain semicolon-separated values, so we explode these.

print("\nLoading clinical data...")
clinical = pd.read_csv(
    "/Users/fudhailsayed/prololab/datasets/PBTA/pbta_all_clinical_data0.tsv",
    sep="\t",
    on_bad_lines="skip",
)

phgg_groups = [
    "High-grade glioma",
    "Diffuse midline glioma",
    "Diffuse hemispheric glioma",
]

phgg_clinical = clinical[clinical["CANCER_GROUP"].isin(phgg_groups)].copy()
print(f"pHGG patients in clinical file: {len(phgg_clinical)}")
print(phgg_clinical["CANCER_GROUP"].value_counts())

# Explode semicolon-separated SPECIMEN_IDs and extract BS_ IDs
phgg_clinical["SPECIMEN_ID"] = phgg_clinical["SPECIMEN_ID"].astype(str)
specimen_ids = phgg_clinical["SPECIMEN_ID"].str.split(";").explode().str.strip()
phgg_bs_ids = set(specimen_ids[specimen_ids.str.startswith("BS_")])
print(f"\npHGG BS_ IDs from clinical: {len(phgg_bs_ids)}")

# Subset expression matrix to pHGG samples
mat_phgg = mat_log[mat_log.index.isin(phgg_bs_ids)]
print(f"pHGG samples matched in expression matrix: {mat_phgg.shape[0]}")

# =============================================================================
# SECTION 4: Load DoRothEA Regulons
# =============================================================================
# DoRothEA is a curated TF regulon resource. We use confidence levels A, B, C:
#   A = TF-target interaction supported by multiple lines of evidence
#   B = supported by ChIP-seq + additional evidence
#   C = supported by ChIP-seq only
# Level D is lower confidence and excluded to reduce noise.
#
# ULM (Univariate Linear Model) estimates TF activity per sample by fitting
# a linear model between TF target gene expression and a null distribution.

print("\nLoading DoRothEA regulons...")
net = dc.get_dorothea(organism="human", levels=["A", "B", "C"])

# Clean network — force standard dtypes to avoid pandas NA conflicts
net["source"] = net["source"].astype(str)
net["target"] = net["target"].astype(str)
net["weight"] = net["weight"].astype(float)
net = net.dropna()
net = net[net["weight"] != 0]
print(f"Regulons loaded after cleaning: {net.shape[0]}")

# =============================================================================
# SECTION 5: Run ULM to Infer TF Activity
# =============================================================================

print("\nRunning ULM TF activity inference...")
tf_acts, tf_pvals = dc.run_ulm(
    mat=mat_phgg,
    net=net,
    source="source",
    target="target",
    weight="weight",
    verbose=True,
)

tf_acts.to_csv("/Users/fudhailsayed/prololab/datasets/PBTA/pbta_tf_activities_ulm.csv")
print(f"TF activity matrix shape: {tf_acts.shape}")

# =============================================================================
# SECTION 6: Correlate TF Activity with ST8SIA2 Expression
# =============================================================================
# For each TF, compute Spearman correlation between its regulon activity
# score and ST8SIA2 expression across pHGG samples.
# Multiple testing correction uses Benjamini-Hochberg FDR.

print("\nCorrelating TF activities with ST8SIA2 expression...")
st8sia2 = mat_phgg["ST8SIA2"]

correlations = {}
for tf in tf_acts.columns:
    shared = st8sia2.index.intersection(tf_acts.index)
    r, p = spearmanr(st8sia2.loc[shared], tf_acts.loc[shared, tf])
    correlations[tf] = {"r": r, "p": p}

corr_df = pd.DataFrame(correlations).T
corr_df.columns = ["r", "p"]
corr_df["p_adj"] = multipletests(corr_df["p"], method="fdr_bh")[1]
corr_df = corr_df.sort_values("r", ascending=False)
corr_df.to_csv("/Users/fudhailsayed/prololab/datasets/PBTA/tf_st8sia2_correlations.csv")

print("\nTop 20 TFs positively correlated with ST8SIA2:")
print(corr_df.head(20).to_string())
print("\nBottom 10 TFs (negatively correlated):")
print(corr_df.tail(10).to_string())

# HIF1A result (original hypothesis)
if "HIF1A" in corr_df.index:
    h = corr_df.loc["HIF1A"]
    print(f"\nHIF1A: r={h['r']:.3f}, p={h['p']:.3e}, p_adj={h['p_adj']:.3e}")
    print(f"HIF1A rank: {corr_df.index.get_loc('HIF1A') + 1} out of {len(corr_df)}")

# =============================================================================
# SECTION 7: Visualization
# =============================================================================

print("\nGenerating plots...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(
    "TF Activity Correlation with ST8SIA2 in pHGG (PBTA, n=123)",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)

# ── Plot 1: Ranked bar chart of top/bottom TFs ───────────────────────
ax = axes[0]
n_show = 15
top = corr_df.head(n_show)
bot = corr_df.tail(n_show)
plot_df = pd.concat([top, bot]).drop_duplicates()
plot_df = plot_df.sort_values("r")

colors = ["crimson" if r > 0 else "steelblue" for r in plot_df["r"]]
# Highlight significant hits
edge_colors = ["black" if p < 0.05 else "none" for p in plot_df["p_adj"]]

bars = ax.barh(
    plot_df.index, plot_df["r"], color=colors, edgecolor=edge_colors, linewidth=1.2
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Spearman r (TF activity vs ST8SIA2)", fontsize=11)
ax.set_title("Top & Bottom Correlated TFs", fontsize=12)

# Legend
sig_patch = mpatches.Patch(
    edgecolor="black", facecolor="white", linewidth=1.2, label="p_adj < 0.05"
)
pos_patch = mpatches.Patch(color="crimson", label="Positive")
neg_patch = mpatches.Patch(color="steelblue", label="Negative")
ax.legend(handles=[sig_patch, pos_patch, neg_patch], fontsize=8, loc="lower right")

# ── Plot 2: Volcano plot ─────────────────────────────────────────────
ax = axes[1]
sig_mask = corr_df["p_adj"] < 0.05
top_tfs = corr_df.head(10).index.tolist()
bot_tfs = corr_df.tail(10).index.tolist()
label_tfs = top_tfs + bot_tfs + ["HIF1A", "MYCN", "TWIST1", "SNAI1"]

ax.scatter(
    corr_df.loc[~sig_mask, "r"],
    -np.log10(corr_df.loc[~sig_mask, "p_adj"]),
    alpha=0.4,
    color="gray",
    s=20,
    label="ns",
)
ax.scatter(
    corr_df.loc[sig_mask, "r"],
    -np.log10(corr_df.loc[sig_mask, "p_adj"]),
    alpha=0.8,
    color="crimson",
    s=40,
    label="p_adj < 0.05",
)

# Label key TFs
for tf in set(label_tfs):
    if tf in corr_df.index:
        ax.annotate(
            tf,
            (corr_df.loc[tf, "r"], -np.log10(corr_df.loc[tf, "p_adj"])),
            fontsize=7,
            xytext=(4, 2),
            textcoords="offset points",
        )

ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
ax.axhline(
    -np.log10(0.05), color="blue", linewidth=0.5, linestyle="--", label="p_adj=0.05"
)
ax.set_xlabel("Spearman r", fontsize=11)
ax.set_ylabel("-log10(adjusted p-value)", fontsize=11)
ax.set_title("Volcano Plot: TF Activity vs ST8SIA2", fontsize=12)
ax.legend(fontsize=8)

# ── Plot 3: Scatter of MYCN activity vs ST8SIA2 ──────────────────────
ax = axes[2]
if "MYCN" in tf_acts.columns:
    shared = st8sia2.index.intersection(tf_acts.index)
    mycn_auc = tf_acts.loc[shared, "MYCN"]
    st8_vals = st8sia2.loc[shared]

    ax.scatter(
        mycn_auc,
        st8_vals,
        alpha=0.6,
        color="crimson",
        s=40,
        edgecolors="white",
        linewidth=0.5,
    )

    # Trend line
    z = np.polyfit(mycn_auc, st8_vals, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(mycn_auc.min(), mycn_auc.max(), 100)
    ax.plot(x_line, p_line(x_line), color="black", linewidth=1.5, linestyle="--")

    r_val = corr_df.loc["MYCN", "r"]
    p_val = corr_df.loc["MYCN", "p_adj"]
    ax.set_xlabel("MYCN Regulon Activity (ULM)", fontsize=11)
    ax.set_ylabel("ST8SIA2 Expression (log2 TPM+1)", fontsize=11)
    ax.set_title(
        f"MYCN Activity vs ST8SIA2\nr={r_val:.3f}, p_adj={p_val:.3f}", fontsize=12
    )

plt.tight_layout()
plt.savefig(
    "/Users/fudhailsayed/prololab/datasets/PBTA/tf_st8sia2_results.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "/Users/fudhailsayed/prololab/datasets/PBTA/tf_st8sia2_results.png",
    dpi=300,
    bbox_inches="tight",
)
print("\nPlots saved to PBTA folder.")
print("Analysis complete.")
