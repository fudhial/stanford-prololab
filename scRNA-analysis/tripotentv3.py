"""
Tri-IPC Cell Identification and Characterization for pHGG scRNA-seq (v3)
=========================================================================
Pipeline aligned with integrated_featureplot_v2.py:

    QC:
        - Per-sample QC filtering BEFORE concatenation (mirrors featureplot v2)
        - Same thresholds: min/max genes, MT%, 500-cell minimum per sample
        - Bad samples skipped gracefully with a warning

    Normalization:
        - normalize_total(1e4) + log1p() BEFORE HVG selection
        - lognorm layer stored for all downstream expression retrieval
        - adata.raw frozen AFTER QC, BEFORE normalization (matches featureplot)
        - HVG flavor = 'seurat' (appropriate for lognorm data)
        - regress_out() REMOVED — too slow/memory intensive at scale
          (was in v2; featureplot v2 removed it for the same reason)
        - scale() kept for PCA only — lognorm layer untouched

    Batch Correction:
        - Full Harmony batch correction via direct harmonypy call
          (same implementation as featureplot v2, not the scanpy wrapper)
        - Handles both old (50, cells) and new (cells, 50) harmonypy output shapes
        - Neighbors graph built on X_pca_harmony embedding

    Reproducibility:
        - Random seed fixed at 42 across PCA, neighbors, UMAP, Leiden
          (was missing entirely in v2)

    HVG count:
        - Raised from 2000 → 3000 to match featureplot v2

    Tri-IPC definition:
        - EGFR+ AND OLIG2+ using fixed lognorm thresholds (kept from v2)
        - Percentile-based thresholds deliberately NOT used here —
          fixed lognorm values are more reproducible across datasets

⚠️  sc.concat uses join='outer' — zero-fills missing genes across samples.
    Switch to join='inner' when gene set consistency is confirmed.

Author: Fudhail Sayed
Updated: 3/24/26
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor="white", figsize=(8, 6))

# ============================================================================
# CONFIGURATION — EDIT THIS SECTION
# ============================================================================

SAMPLES_TO_ANALYZE = [
    "SCPCS000001",
    "SCPCS000002",
    "SCPCS000003",
    "SCPCS000004",
    "SCPCS000005",
    "SCPCS000006",
    "SCPCS000007",
    "SCPCS000008",
    "SCPCS000009",
    "SCPCS000010",
    "SCPCS000011",
    "SCPCS000012",
    "SCPCS000013",
    "SCPCS000014",
    "SCPCS000015",
    "SCPCS000016",
    "SCPCS000017",
    "SCPCS000018",
    "SCPCS000019",
    "SCPCS000020",
    "SCPCS000021",
    "SCPCS000022",
    "SCPCS000023",
    # Add more samples here
]

# Genes of interest
GENES_OF_INTEREST = [
    "MAP4K4",
    "EGFR",
    "OLIG2",
    # Add more here
]

# Tri-IPC thresholds (fixed lognorm values — more reproducible than percentiles)
# 0.5 in lognorm space ≈ ~1.6x over background for a gene normalized to 1e4.
# Inspect expression_distributions.png to validate/adjust.
EGFR_LOGNORM_THRESHOLD = 1.0
OLIG2_LOGNORM_THRESHOLD = 1.0

# QC thresholds
MIN_GENES_PER_CELL = 500
MAX_GENES_PER_CELL = 8000
MAX_PCT_MT = 20
MIN_CELLS_PER_GENE = 10
MIN_CELLS_PER_SAMPLE = 500  # samples below this after QC are dropped

# Additional markers
SURFACE_MARKERS = ["F3", "CD38", "PDGFRA", "ITGA2"]
VALIDATION_NEGATIVE = ["RBFOX3", "SPARCL1", "DLX5"]  # Should be LOW in Tri-IPC

# Random seed — fixed for reproducibility across PCA, neighbors, UMAP, Leiden
RANDOM_SEED = 42

BASE_PATH = "/Users/fudhailsayed/prololab"
OUTPUT_DIR = Path(BASE_PATH) / "figures" / "tri_ipc_v3"

# ============================================================================


# ----------------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------------


def run_qc_single(adata, sample_name):
    """
    QC-filter a single sample BEFORE concatenation.
    Mirrors featureplot v2 run_qc_single() exactly.
    """
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    sc.pp.filter_cells(adata, max_genes=MAX_GENES_PER_CELL)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_PCT_MT].copy()
    after = adata.n_obs

    print(
        f"    QC: {before} -> {after} cells "
        f"(removed {before - after}, "
        f"median MT={adata.obs['pct_counts_mt'].median():.1f}%)"
    )
    return adata


def load_all_samples(base_path):
    """
    Load, QC-filter per sample, then concatenate.
    Samples with < MIN_CELLS_PER_SAMPLE cells after QC are dropped.
    Mirrors featureplot v2 load_all_samples().
    """
    print("\n" + "=" * 80)
    print(f"LOADING {len(SAMPLES_TO_ANALYZE)} SAMPLES")
    print("=" * 80)

    data_path = Path(base_path) / "datasets/pHGG_scRNA_anndata/scRNA"
    adatas = []
    sample_names = []

    for sample_name in SAMPLES_TO_ANALYZE:
        h5ad_file = (
            data_path
            / sample_name
            / f'{sample_name.replace("SCPCS", "SCPCL")}_filtered_rna_symbols.h5ad'
        )
        try:
            print(f"\n  Loading {sample_name}...")
            adata = sc.read_h5ad(h5ad_file)

            # Ensure sparse float32
            if hasattr(adata.X, "toarray"):
                adata.X = csr_matrix(adata.X.astype(np.float32))
            else:
                adata.X = csr_matrix(np.asarray(adata.X, dtype=np.float32))

            adata.obs["sample"] = sample_name
            adata.obs["sample_id"] = sample_name

            # QC filter BEFORE concatenation
            adata = run_qc_single(adata, sample_name)

            if adata.n_obs < MIN_CELLS_PER_SAMPLE:
                print(
                    f"    Skipping {sample_name}: only {adata.n_obs} cells after QC "
                    f"(min={MIN_CELLS_PER_SAMPLE})"
                )
                continue

            adatas.append(adata)
            sample_names.append(sample_name)
            print(f"    {adata.n_obs} cells retained")

        except FileNotFoundError:
            print(f"    Skipping {sample_name}: file not found")

    if len(adatas) == 0:
        raise ValueError("No samples loaded!")
    if len(adatas) == 1:
        print(f"\nSingle sample mode: {sample_names[0]}")
        return adatas[0]

    combined = sc.concat(adatas, join="outer", label="batch", keys=sample_names)

    if hasattr(combined.X, "toarray"):
        combined.X = csr_matrix(combined.X.astype(np.float32))
    else:
        combined.X = csr_matrix(np.asarray(combined.X, dtype=np.float32))

    print(f"\nCombined: {combined.n_obs} cells x {combined.n_vars} genes")
    return combined


# ----------------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------------


def preprocess(adata):
    """
    Normalization pipeline (ORDER MATTERS — mirrors featureplot v2):
        1. Filter rare genes
        2. Freeze raw counts in adata.raw (after QC, before normalization)
        3. normalize_total(1e4) + log1p()
        4. Store lognorm layer
        5. HVG selection (flavor='seurat', n=3000)
        6. scale() for PCA only — lognorm layer untouched

    NOTE: regress_out() removed (too slow at scale — same decision as featureplot v2).
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING")
    print("=" * 80)

    print(f"\n  Cells: {adata.n_obs}  Genes: {adata.n_vars}")

    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    print(f"  After gene filtering: {adata.n_vars} genes")

    adata.raw = adata
    print("  Raw counts frozen in adata.raw")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("  normalize_total(1e4) + log1p applied")

    if hasattr(adata.X, "toarray"):
        adata.layers["lognorm"] = csr_matrix(adata.X.astype(np.float32))
    else:
        adata.layers["lognorm"] = adata.X.copy().astype(np.float32)
    print("  lognorm layer stored")

    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
    print(f"  {adata.var['highly_variable'].sum()} HVGs selected (flavor='seurat')")

    sc.pp.scale(adata, max_value=10)
    print("  Scaled for PCA — expression plots use lognorm layer")

    return adata


# ----------------------------------------------------------------------------
# EMBEDDING & CLUSTERING
# ----------------------------------------------------------------------------


def embed_and_cluster(adata):
    """
    PCA -> Harmony batch correction -> neighbors -> UMAP -> Leiden.
    Direct harmonypy call (not scanpy wrapper) — mirrors featureplot v2.
    Handles both old (50, cells) and new (cells, 50) harmonypy output shapes.
    Random seed fixed at RANDOM_SEED throughout.
    """
    print("\n" + "=" * 80)
    print("EMBEDDING & CLUSTERING")
    print("=" * 80)

    sc.tl.pca(adata, svd_solver="arpack", n_comps=50, random_state=RANDOM_SEED)

    import harmonypy as hm

    ho = hm.run_harmony(
        adata.obsm["X_pca"],
        adata.obs,
        "sample",
        max_iter_harmony=20,
        random_state=RANDOM_SEED,
    )

    result = ho.Z_corr
    print(f"  Harmony raw output shape: {result.shape}")
    if result.shape[0] == 50:  # old format: (50, cells) — transpose
        adata.obsm["X_pca_harmony"] = result.T
    elif result.shape[1] == 50:  # new format: (cells, 50) — already correct
        adata.obsm["X_pca_harmony"] = result
    else:
        raise ValueError(f"Unexpected Harmony output shape: {result.shape}")

    print(
        f"  Harmony complete — corrected embedding shape: {adata.obsm['X_pca_harmony'].shape}"
    )

    sc.pp.neighbors(
        adata,
        n_neighbors=15,
        n_pcs=50,
        use_rep="X_pca_harmony",
        random_state=RANDOM_SEED,
    )

    sc.tl.umap(adata, random_state=RANDOM_SEED)
    sc.tl.leiden(adata, resolution=0.5, random_state=RANDOM_SEED)

    print(f"  {adata.obs['leiden'].nunique()} Leiden clusters")
    return adata


# ----------------------------------------------------------------------------
# EXPRESSION RETRIEVAL
# ----------------------------------------------------------------------------


def get_lognorm_expression(adata, gene):
    """
    Safely retrieve lognorm-normalized expression for a gene.
    Priority: lognorm layer → raw → scaled X (last resort, with warning).
    Identical to featureplot v2 get_lognorm_expression().
    """
    if "lognorm" in adata.layers and gene in adata.var_names:
        idx = list(adata.var_names).index(gene)
        expr = adata.layers["lognorm"][:, idx]
    elif (
        hasattr(adata, "raw") and adata.raw is not None and gene in adata.raw.var_names
    ):
        print(f"  WARNING: {gene} falling back to adata.raw (raw counts!)")
        idx = list(adata.raw.var_names).index(gene)
        expr = adata.raw.X[:, idx]
    elif gene in adata.var_names:
        print(f"  WARNING: {gene} using scaled adata.X (z-scores, not expression!)")
        idx = list(adata.var_names).index(gene)
        expr = adata.X[:, idx]
    else:
        return None

    if hasattr(expr, "toarray"):
        expr = expr.toarray().flatten()
    else:
        expr = np.asarray(expr).flatten()

    return expr.astype(np.float32)


# ----------------------------------------------------------------------------
# TRI-IPC IDENTIFICATION
# ----------------------------------------------------------------------------


def plot_expression_distributions(adata, genes, output_dir, thresholds=None):
    """
    Violin plots of lognorm expression — validate EGFR/OLIG2 thresholds before analysis.
    """
    print("\n  Plotting expression distributions for threshold validation...")
    fig, axes = plt.subplots(1, len(genes), figsize=(5 * len(genes), 5))
    if len(genes) == 1:
        axes = [axes]

    for ax, gene in zip(axes, genes):
        expr = get_lognorm_expression(adata, gene)
        if expr is None:
            ax.set_title(f"{gene}\n(not found)")
            continue

        expressing = expr[expr > 0]
        thresh = thresholds.get(gene, 0.5) if thresholds else 0.5
        ax.violinplot(expressing, showmedians=True)
        ax.axhline(y=thresh, color="red", linestyle="--", label=f"threshold ({thresh})")
        ax.set_title(f"{gene}\n(n expressing={len(expressing)})")
        ax.set_ylabel("lognorm expression")
        ax.legend(fontsize=7)

    plt.suptitle(
        "Expression distributions — inspect before finalising thresholds", fontsize=11
    )
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / "expression_distributions.png", dpi=200, bbox_inches="tight"
    )
    plt.close()
    print("  Saved expression_distributions.png")


def identify_tri_ipc_cells(adata):
    """
    Identify Tri-IPC cells: EGFR+ AND OLIG2+ using fixed lognorm thresholds.
    Fixed thresholds (not percentile-based) for cross-dataset reproducibility.
    """
    print("\n" + "=" * 80)
    print("IDENTIFYING TRI-IPC CELLS")
    print("=" * 80)
    print(
        f"  Thresholds: EGFR lognorm > {EGFR_LOGNORM_THRESHOLD}, "
        f"OLIG2 lognorm > {OLIG2_LOGNORM_THRESHOLD}"
    )

    for gene, threshold in [
        ("EGFR", EGFR_LOGNORM_THRESHOLD),
        ("OLIG2", OLIG2_LOGNORM_THRESHOLD),
    ]:
        expr = get_lognorm_expression(adata, gene)
        if expr is None:
            print(f"  {gene} not found!")
            return adata

        adata.obs[f"{gene}_expr"] = expr
        adata.obs[f"{gene}_high"] = expr > threshold

        pct_exp = (expr > 0).sum() / len(expr) * 100
        pct_high = adata.obs[f"{gene}_high"].sum() / adata.n_obs * 100
        print(f"\n  {gene}:")
        print(f"    % cells expressing (>0):  {pct_exp:.1f}%")
        print(f"    % cells above threshold:  {pct_high:.1f}%  (lognorm > {threshold})")
        print(f"    Mean (all cells):          {expr.mean():.3f}")
        print(f"    Mean (expressing cells):   {expr[expr > 0].mean():.3f}")

    adata.obs["Tri_IPC"] = adata.obs["EGFR_high"] & adata.obs["OLIG2_high"]
    n = adata.obs["Tri_IPC"].sum()
    pct = n / adata.n_obs * 100
    print(f"\n  Tri-IPC cells (EGFR+ AND OLIG2+): {n} ({pct:.1f}%)")

    print(f"\n  Surface marker enrichment (Tri-IPC vs Other):")
    for marker in SURFACE_MARKERS:
        expr = get_lognorm_expression(adata, marker)
        if expr is not None:
            adata.obs[f"{marker}_expr"] = expr
            t = expr[adata.obs["Tri_IPC"]]
            o = expr[~adata.obs["Tri_IPC"]]
            fc = t.mean() / (o.mean() + 1e-10)
            print(
                f"    {marker:10s}  Tri-IPC={t.mean():.3f}  Other={o.mean():.3f}  FC={fc:.2f}x"
            )
        else:
            print(f"    {marker}: not found")

    print(f"\n  Negative validation markers (should be LOW in Tri-IPC):")
    for marker in VALIDATION_NEGATIVE:
        expr = get_lognorm_expression(adata, marker)
        if expr is not None:
            adata.obs[f"{marker}_expr"] = expr
            t = expr[adata.obs["Tri_IPC"]]
            o = expr[~adata.obs["Tri_IPC"]]
            status = "low" if t.mean() < o.mean() else "ELEVATED"
            print(
                f"    {marker:10s}  Tri-IPC={t.mean():.3f}  Other={o.mean():.3f}  {status}"
            )

    return adata


def characterize_tri_ipc(adata):
    """Distribution of Tri-IPC cells by sample and cluster."""
    print("\n" + "=" * 80)
    print("CHARACTERIZING TRI-IPC CELLS")
    print("=" * 80)

    if "Tri_IPC" not in adata.obs.columns:
        print("  Tri-IPC cells not identified — run identify_tri_ipc_cells() first")
        return adata

    print("\n  By sample:")
    for sample in adata.obs["sample"].unique():
        mask = adata.obs["sample"] == sample
        n_tri = (mask & adata.obs["Tri_IPC"]).sum()
        total = mask.sum()
        print(f"    {sample}: {n_tri}/{total} ({n_tri/total*100:.1f}%)")

    print("\n  By Leiden cluster:")
    for cluster in sorted(adata.obs["leiden"].unique(), key=int):
        mask = adata.obs["leiden"] == cluster
        n_tri = (mask & adata.obs["Tri_IPC"]).sum()
        total = mask.sum()
        pct = n_tri / total * 100 if total > 0 else 0
        flag = "  enriched" if pct > 20 else ""
        print(f"    Cluster {cluster:>2s}: {n_tri:>4}/{total:<5} ({pct:4.1f}%){flag}")

    return adata


# ----------------------------------------------------------------------------
# GENE ANALYSIS
# ----------------------------------------------------------------------------


def analyze_genes(adata, gene_list):
    """
    Compare expression of genes of interest in Tri-IPC vs non-Tri-IPC cells.
    Uses lognorm values. Applies Benjamini-Hochberg FDR correction.
    """
    print("\n" + "=" * 80)
    print("GENE ANALYSIS: Tri-IPC vs Non-Tri-IPC")
    print("=" * 80)

    results = []
    n_tri = adata.obs["Tri_IPC"].sum()

    for gene in gene_list:
        expr = get_lognorm_expression(adata, gene)
        if expr is None:
            print(f"\n  {gene}: not found — skipping")
            continue

        adata.obs[f"{gene}_expr"] = expr

        tri_expr = expr[adata.obs["Tri_IPC"]]
        non_tri_expr = expr[~adata.obs["Tri_IPC"]]

        stat, pval = mannwhitneyu(tri_expr, non_tri_expr, alternative="two-sided")
        fc = tri_expr.mean() / (non_tri_expr.mean() + 1e-10)

        threshold = np.percentile(expr[expr > 0], 75) if (expr > 0).any() else 0
        adata.obs[f"{gene}_high"] = expr > threshold
        overlap = (adata.obs["Tri_IPC"] & adata.obs[f"{gene}_high"]).sum()
        pct_over = overlap / n_tri * 100 if n_tri > 0 else 0

        egfr_corr = olig2_corr = np.nan
        if "EGFR_expr" in adata.obs.columns:
            egfr_corr, _ = spearmanr(expr, adata.obs["EGFR_expr"])
        if "OLIG2_expr" in adata.obs.columns:
            olig2_corr, _ = spearmanr(expr, adata.obs["OLIG2_expr"])

        print(f"\n  {gene}:")
        print(
            f"    Tri-IPC mean ± SD:     {tri_expr.mean():.3f} ± {tri_expr.std():.3f}"
        )
        print(
            f"    Non-Tri-IPC mean ± SD: {non_tri_expr.mean():.3f} ± {non_tri_expr.std():.3f}"
        )
        print(f"    Fold change:           {fc:.2f}x")
        print(f"    p-value (raw):         {pval:.2e}")
        print(f"    Overlap w/ Tri-IPC:    {overlap}/{n_tri} ({pct_over:.1f}%)")
        print(f"    Spearman r (EGFR):     {egfr_corr:.3f}")
        print(f"    Spearman r (OLIG2):    {olig2_corr:.3f}")

        results.append(
            {
                "gene": gene,
                "tri_ipc_mean": tri_expr.mean(),
                "non_tri_mean": non_tri_expr.mean(),
                "fold_change": fc,
                "pvalue_raw": pval,
                "pct_overlap": pct_over,
                "corr_egfr": egfr_corr,
                "corr_olig2": olig2_corr,
            }
        )

    if not results:
        return adata, pd.DataFrame()

    results_df = pd.DataFrame(results)
    _, pvals_adj, _, _ = multipletests(results_df["pvalue_raw"], method="fdr_bh")
    results_df["pvalue_adj_BH"] = pvals_adj
    results_df["significant"] = pvals_adj < 0.05

    print("\n" + "-" * 60)
    print("  FDR-adjusted p-values (Benjamini-Hochberg):")
    for _, row in results_df.iterrows():
        sig = (
            "***"
            if row["pvalue_adj_BH"] < 0.001
            else (
                "**"
                if row["pvalue_adj_BH"] < 0.01
                else "*" if row["pvalue_adj_BH"] < 0.05 else "ns"
            )
        )
        print(
            f"    {row['gene']:12s}  FC={row['fold_change']:.2f}x  "
            f"p_adj={row['pvalue_adj_BH']:.2e} {sig}"
        )

    return adata, results_df


# ----------------------------------------------------------------------------
# VISUALIZATIONS
# ----------------------------------------------------------------------------


def create_visualizations(adata, gene_list, output_dir):
    """Generate all UMAP plots, overlays, and summary heatmap."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    sc.pl.umap(
        adata, color="sample", ax=axes[0], show=False, title="Samples", palette="tab20"
    )
    sc.pl.umap(
        adata,
        color="leiden",
        ax=axes[1],
        show=False,
        title="Leiden Clusters",
        legend_loc="on data",
    )
    sc.pl.umap(
        adata,
        color="EGFR_expr",
        ax=axes[2],
        show=False,
        title="EGFR (lognorm)",
        cmap="viridis",
        vmax="p99",
    )
    sc.pl.umap(
        adata,
        color="OLIG2_expr",
        ax=axes[3],
        show=False,
        title="OLIG2 (lognorm)",
        cmap="viridis",
        vmax="p99",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  overview.png")

    # 2. QC metrics
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sc.pl.umap(
        adata,
        color="n_genes_by_counts",
        ax=axes[0],
        show=False,
        title="Genes per cell",
        cmap="viridis",
    )
    sc.pl.umap(
        adata,
        color="total_counts",
        ax=axes[1],
        show=False,
        title="Total counts",
        cmap="viridis",
    )
    sc.pl.umap(
        adata,
        color="pct_counts_mt",
        ax=axes[2],
        show=False,
        title="% MT",
        cmap="viridis",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "qc_umap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  qc_umap.png")

    # 3. Tri-IPC identification
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sc.pl.umap(
        adata,
        color="EGFR_high",
        ax=axes[0],
        show=False,
        title="EGFR-high",
        palette=["lightgray", "green"],
    )
    sc.pl.umap(
        adata,
        color="OLIG2_high",
        ax=axes[1],
        show=False,
        title="OLIG2-high",
        palette=["lightgray", "blue"],
    )
    sc.pl.umap(
        adata,
        color="Tri_IPC",
        ax=axes[2],
        show=False,
        title="Tri-IPC (EGFR+ OLIG2+)",
        palette=["lightgray", "red"],
    )
    plt.tight_layout()
    plt.savefig(output_dir / "tri_ipc_identification.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  tri_ipc_identification.png")

    # 4. Gene overlays (lognorm)
    valid_genes = [g for g in gene_list if f"{g}_expr" in adata.obs.columns]
    if valid_genes:
        n_cols = 3
        n_rows = (len(valid_genes) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = np.array(axes).flatten()
        for i, gene in enumerate(valid_genes):
            sc.pl.umap(
                adata,
                color=f"{gene}_expr",
                cmap="viridis",
                vmax="p99",
                ax=axes[i],
                show=False,
                title=f"{gene} (lognorm)",
            )
        for j in range(len(valid_genes), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "genes_of_interest.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  genes_of_interest.png")

    # 5. Per-gene 3-panel overlay: expression + Tri-IPC + co-localization
    for gene in valid_genes:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle(
            f"Tri-IPC  x  {gene} Expression", fontsize=14, fontweight="bold", y=1.01
        )

        sc.pl.umap(
            adata,
            color=f"{gene}_expr",
            cmap="viridis",
            vmax="p99",
            ax=axes[0],
            show=False,
            title=f"{gene} (lognorm)",
        )
        sc.pl.umap(
            adata,
            color="Tri_IPC",
            palette=["lightgray", "red"],
            ax=axes[1],
            show=False,
            title="Tri-IPC cells",
        )

        umap_coords = adata.obsm["X_umap"]
        tri = adata.obs["Tri_IPC"].values
        high = adata.obs.get(
            f"{gene}_high", pd.Series(False, index=adata.obs_names)
        ).values
        overlap = tri & high

        # Layer ordering matches featureplot v2 co-localization panels
        axes[2].scatter(
            umap_coords[~tri & ~high, 0],
            umap_coords[~tri & ~high, 1],
            c="lightgray",
            s=4,
            alpha=0.3,
            label="Other",
            rasterized=True,
        )
        axes[2].scatter(
            umap_coords[high & ~tri, 0],
            umap_coords[high & ~tri, 1],
            c="#F4D03F",
            s=10,
            alpha=0.5,
            label=f"{gene}-high only",
            rasterized=True,
        )
        axes[2].scatter(
            umap_coords[tri, 0],
            umap_coords[tri, 1],
            c="red",
            s=12,
            alpha=0.5,
            label="Tri-IPC",
            rasterized=True,
        )
        if overlap.sum() > 0:
            axes[2].scatter(
                umap_coords[overlap, 0],
                umap_coords[overlap, 1],
                c="purple",
                s=25,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.5,
                label=f"Tri-IPC + {gene}-high",
                zorder=10,
                rasterized=True,
            )

        axes[2].set_title(f"Tri-IPC & {gene} Co-localization (4-group)")
        axes[2].set_xlabel("")
        axes[2].set_ylabel("")
        axes[2].legend(loc="upper left", fontsize=8)

        # Stats annotation (bottom-right) — same style as featureplot v2
        tri_expr_vals = adata.obs.loc[adata.obs["Tri_IPC"], f"{gene}_expr"].values
        non_tri_vals = adata.obs.loc[~adata.obs["Tri_IPC"], f"{gene}_expr"].values
        _, pval = mannwhitneyu(tri_expr_vals, non_tri_vals, alternative="two-sided")
        sig = (
            "***"
            if pval < 0.001
            else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        )
        stats_text = f"Mann-Whitney U\np = {pval:.2e} {sig}"
        axes[2].text(
            0.98,
            0.02,
            stats_text,
            transform=axes[2].transAxes,
            fontsize=18,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{gene}_tri_ipc_overlay.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print(f"  {gene}_tri_ipc_overlay.png")

    # 6. Cluster summary heatmap
    summary_rows = []
    for cluster in sorted(adata.obs["leiden"].unique(), key=int):
        mask = adata.obs["leiden"] == cluster
        tri_mask = mask & adata.obs["Tri_IPC"]
        row = {
            "cluster": f"C{cluster}",
            "n_cells": int(mask.sum()),
            "pct_Tri_IPC": tri_mask.sum() / mask.sum() * 100 if mask.sum() > 0 else 0,
        }
        for gene in valid_genes:
            col = f"{gene}_expr"
            if col in adata.obs.columns:
                row[f"{gene}_triIPC_mean"] = (
                    adata.obs.loc[tri_mask, col].mean() if tri_mask.sum() > 0 else 0
                )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "cluster_summary.csv", index=False)

    plot_cols = [c for c in summary_df.columns if c not in ("cluster", "n_cells")]
    if plot_cols:
        heat_data = summary_df[plot_cols].set_index(summary_df["cluster"]).T
        fig, ax = plt.subplots(
            figsize=(max(8, len(summary_df) * 0.7), max(6, len(plot_cols) * 0.8))
        )
        sns.heatmap(
            heat_data,
            annot=True,
            fmt=".1f",
            cmap="RdYlBu_r",
            linewidths=0.3,
            ax=ax,
            cbar_kws={"label": "Value"},
        )
        ax.set_title("Tri-IPC Characterization by Cluster (lognorm expression)")
        plt.tight_layout()
        plt.savefig(
            output_dir / "cluster_summary_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("  cluster_summary_heatmap.png")

    print(f"\n  All figures saved to: {output_dir}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------


def main():
    print("\n" + "=" * 80)
    print("TRI-IPC ANALYSIS v3")
    print(f"Samples: {', '.join(SAMPLES_TO_ANALYZE)}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load (QC per sample, then concat)
    adata = load_all_samples(BASE_PATH)

    # Preprocess (normalize, lognorm layer, HVG, scale)
    adata = preprocess(adata)

    # Embed & cluster (PCA → Harmony → neighbors → UMAP → Leiden)
    adata = embed_and_cluster(adata)

    # Inspect threshold distributions before committing
    plot_expression_distributions(
        adata,
        ["EGFR", "OLIG2"],
        OUTPUT_DIR,
        thresholds={"EGFR": EGFR_LOGNORM_THRESHOLD, "OLIG2": OLIG2_LOGNORM_THRESHOLD},
    )

    # Identify Tri-IPC
    adata = identify_tri_ipc_cells(adata)

    # Characterize
    adata = characterize_tri_ipc(adata)

    # Gene analysis + FDR
    adata, results_df = analyze_genes(adata, GENES_OF_INTEREST)
    if not results_df.empty:
        results_df.to_csv(OUTPUT_DIR / "gene_analysis_summary.csv", index=False)
        print(f"  Saved gene_analysis_summary.csv")

    # Visualize
    create_visualizations(adata, GENES_OF_INTEREST, OUTPUT_DIR)

    # Save processed AnnData
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)
    out_h5ad = OUTPUT_DIR / "tri_ipc_v3_processed.h5ad"
    adata.write(out_h5ad)
    print(f"\n  Saved: {out_h5ad}")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    n_tri = int(adata.obs["Tri_IPC"].sum())
    print(f"  Cells analysed:  {adata.n_obs}")
    print(f"  Tri-IPC cells:   {n_tri} ({n_tri/adata.n_obs*100:.1f}%)")

    if not results_df.empty:
        print("\n  Gene results (FDR-corrected):")
        for _, row in results_df.iterrows():
            sig = (
                "***"
                if row["pvalue_adj_BH"] < 0.001
                else (
                    "**"
                    if row["pvalue_adj_BH"] < 0.01
                    else "*" if row["pvalue_adj_BH"] < 0.05 else "ns"
                )
            )
            print(
                f"    {row['gene']:12s}  FC={row['fold_change']:.2f}x  "
                f"p_adj={row['pvalue_adj_BH']:.2e} {sig}  "
                f"overlap={row['pct_overlap']:.1f}%"
            )

    print(f"\n  Output directory: {OUTPUT_DIR}")
    return adata


if __name__ == "__main__":
    adata = main()
