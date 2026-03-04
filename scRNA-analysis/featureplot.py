"""
Integrated Multi-Sample pHGG Analysis (v2)
==========================================
Proper preprocessing pipeline:
    1. QC filtering per sample (MT%, min/max genes)
    2. normalize_total() + log1p() BEFORE HVG selection
    3. lognorm layer stored for safe gene expression retrieval
    4. adata.raw set correctly (after QC, before normalization)
    5. HVG flavor = 'seurat' (appropriate for lognorm data)
    6. Gene plots pull from lognorm layer — NOT use_raw=True

Visualizations:
    - For each cell state (MES, OPC, NPC, AC): one dedicated 3-panel figure
        Panel 1: Cell state score on UMAP
        Panel 2: MAP4K4 expression (lognorm) on UMAP
        Panel 3: Three-group co-localization (same style as Tri-IPC figure)
                 Gray = other, State color = state-high, Purple = overlap
                 Spearman r and p-value in bottom-right corner

⚠️  TO DO LATER — Harmony batch correction:
    Currently removed due to memory/compute constraints.
    When ready, go to embed_and_cluster() and uncomment the Harmony block.
    Also install: pip install harmonypy

⚠️  regress_out() removed — too slow/memory intensive on large datasets.
    Add back later when running on a better machine.

⚠️  Seeds not yet added — to be done later alongside tri_ipc_analysis_v2.py

Author: Fudhail Sayed
Updated: 2/26/26
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor="white", figsize=(8, 6))  # Ignore

# ============================================================================
# CONFIGURATION — EDIT THIS SECTION
# ============================================================================

# Samples to analyze — add or remove as needed
SAMPLES_TO_ANALYZE = [
    "SCPCS000001",
    "SCPCS000002",
    "SCPCS000003",
]

# Primary gene of interest for overlap analysis
FOCUS_GENE = "MAP4K4"

# Additional genes to plot individually on UMAP
GENES_OF_INTEREST = [
    "MAP4K4",
    "CD44",
    "OLIG2",
    # Add more genes here
]

# Top % threshold to define "high" expressors
# 75 = top 25% of expressing cells are called "high"
FOCUS_GENE_PERCENTILE = 75
STATE_SCORE_PERCENTILE = 75

# QC thresholds
MIN_GENES_PER_CELL = 500
MAX_GENES_PER_CELL = 8000
MAX_PCT_MT = 20
MIN_CELLS_PER_GENE = 10

# Colors for each cell state (used in co-localization panel)
STATE_COLORS = {
    "MES": "#E74C3C",  # red
    "OPC": "#3498DB",  # blue
    "NPC": "#2ECC71",  # green
    "AC": "#F39C12",  # orange
}

# ============================================================================


def get_neftel_gene_programs():
    """Gene programs from Neftel et al., 2019. AC signature added."""
    gene_programs = {
        "MES1": [
            "CHI3L1",
            "ANXA2",
            "ANXA1",
            "CD44",
            "VIM",
            "MT2A",
            "C1S",
            "NAMPT",
            "EFEMP1",
            "C1R",
            "SOD2",
            "IFITM3",
            "TIMP1",
            "SPP1",
            "A2M",
            "S100A11",
            "MT1X",
            "S100A10",
            "FN1",
            "LGALS1",
            "S100A16",
            "CLIC1",
            "MCST1",
            "RCAN1",
            "TAGLN2",
            "NPC2",
            "SERPINF1",
            "C8orf4",
            "EMP1",
            "APOE",
            "CTSB",
            "C3",
            "LGALS3",
            "MT1E",
            "EMP3",
            "SERPINA3",
            "ACTN1",
            "PRDX6",
            "IGFBP7",
            "SERPINE1",
            "PLP2",
            "MGP",
            "CLIC4",
            "GFPT2",
            "GSN",
            "NNMT",
            "TUBA1C",
            "GJA1",
            "TNFRSF1A",
            "WWTR1",
        ],
        "MES2": [
            "HILPDA",
            "ADM",
            "DDIT3",
            "NDRG1",
            "HERPUD1",
            "DNAJB9",
            "TRIB3",
            "ENO2",
            "AKAP12",
            "SQSTM1",
            "MT1X",
            "ATF3",
            "NAMPT",
            "NRN1",
            "SLC2A1",
            "BNIP3",
            "LGALS3",
            "INSIG2",
            "IGFBP3",
            "PFKFB3A",
            "VIM",
            "PLOD2",
            "GBE1",
            "SLC2A3",
            "FTL",
            "WARS",
            "ERO1L",
            "XPOT",
            "HSPA5",
            "GDF15",
            "ANXA2",
            "EPAS1",
            "LDHA",
            "P4HA1",
            "SERTAD1",
            "PFKP",
            "PGK1",
            "EGLN3",
            "SLC6A6",
            "CA9",
            "BNIP3L",
            "RPL21",
            "TRAM1",
            "UFM1",
            "ASNS",
            "GOLT1B",
            "ANGPTL4",
            "SLC39A14",
            "CDKN1A",
            "HSPA9",
        ],
        "OPC": [
            "BCAN",
            "PLP1",
            "GPR17",
            "FIBIN",
            "LHFPL3",
            "OLIG1",
            "PSAT1",
            "SCRG1",
            "OMG",
            "APOD",
            "SIRT2",
            "TNR",
            "TNY1",
            "PHYHIPL",
            "SOX2-OT",
            "NKAN4",
            "LPPR1",
            "PTPRZ1",
            "VCAN",
            "DBI",
            "PMP2",
            "CNP",
            "TNS3",
            "LIMA1",
            "CA10",
            "PCDHGC3",
            "CNTN1",
            "SCD5",
            "P2RX7",
            "CADM2",
            "TTY1H",
            "FGF12",
            "TMEM206",
            "NEU4",
            "FXY06",
            "RNF13",
            "PRKN",
            "GPM6B",
            "LMF1",
            "PGRMC1",
            "SERINC5",
            "GPR37L1",
        ],
        "NPC1": [
            "DLL3",
            "DLL1",
            "SOX4",
            "TUBB3",
            "HES6",
            "TAGLN3",
            "NEU4",
            "MARCKSL1",
            "CD24",
            "STMN1",
            "TCF12",
            "BEX1",
            "OLIG1",
            "MAP2",
            "FXY06",
            "PTPRS",
            "MLLT11",
            "NPPA",
            "BCAN",
            "MEST",
            "ASCL1",
            "BTG2",
            "DCX",
            "NXPH1",
            "HN1",
            "PFN2",
            "SCG3",
            "MYT1",
            "CHD7",
            "GPR56",
            "TUBA1A",
            "PCBP4",
            "ETV1",
            "SHD",
            "TNR",
            "AMOTL2",
            "DBN1",
            "HIP1",
            "ABAT",
            "ELAVL4",
            "LMF1",
            "GRIK2",
            "SERINC5",
            "TSPAN13",
            "ELMO1",
            "GLCCI1",
            "SEZ6L",
            "LRRN1",
            "SEZ6",
            "SOX11",
        ],
        "NPC2": [
            "STMN2",
            "CD24",
            "RND3",
            "HMP19",
            "TUBB3",
            "MIAT",
            "DCX",
            "NSG1",
            "ELAVL4",
            "MLLT11",
            "DLX6-AS1",
            "SOX11",
            "NREP",
            "FNBP1L",
            "TAGLN3",
            "STMN4",
            "DLX5",
            "SOX4",
            "MAP1B",
            "SERPINX2",
            "IGFBPL1",
            "STMN1",
            "HN1",
            "TMEM161B-AS1",
            "DPYSL3",
            "SEPT3",
            "PKIA",
            "ATP1B1",
            "DYNC1I1",
            "CD200",
            "SNAP25",
            "PAK3",
            "NDRG4",
            "KIF5A",
            "UCHL1",
            "ENO2",
            "KIF5C",
            "DDAH2",
            "TUBB2A",
            "LBH",
            "LOC150568",
            "TCF4",
            "GNG3",
            "NFIB",
            "DPYSL5",
            "CRABP1",
            "DBN1",
            "NFIX",
            "CEP170",
            "BLCAP",
        ],
        "AC": [
            "APOE",
            "ALDOC",
            "CLU",
            "AGT",
            "ATP1A2",
            "SPARCL1",
            "GLUL",
            "S100B",
            "NDRG2",
            "EDNRB",
            "SLC1A3",
            "GJA1",
            "GFAP",
            "AQP4",
            "FGFR3",
            "SLC1A2",
            "ACSL6",
            "ETNPPL",
            "DTNA",
            "LCAT",
            "PRODH",
            "BBOX1",
            "SLC6A11",
            "PAPLN",
            "PHGDH",
            "GATM",
            "PON2",
            "ATP13A4",
            "BCAN",
            "HEPACAM",
            "MFGE8",
            "ITPKB",
            "PLCB4",
            "F3",
            "DIO2",
            "SPON1",
            "ETNK2",
            "PTGDS",
            "SLC4A4",
            "ABLIM2",
            "COL5A3",
            "ANXA3",
            "TNC",
            "PMP2",
            "IGFBP2",
            "VEGFA",
            "OSMR",
            "TGFB2",
            "CD44",
            "VIM",
        ],
    }
    return gene_programs


def run_qc_single(adata, sample_name):
    """QC filter a single sample before concatenation."""
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


def load_all_samples(data_dir):
    """Load, QC-filter, and concatenate specified samples."""
    print("\n" + "=" * 80)
    print(f"LOADING {len(SAMPLES_TO_ANALYZE)} SAMPLES")
    print("=" * 80)

    data_path = Path(data_dir)
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

            if hasattr(adata.X, "toarray"):
                adata.X = csr_matrix(adata.X.astype(np.float32))
            else:
                adata.X = csr_matrix(np.asarray(adata.X, dtype=np.float32))

            adata.obs["sample"] = sample_name
            adata.obs["sample_id"] = sample_name
            adata = run_qc_single(adata, sample_name)

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


def preprocess_integrated(adata):
    """
    Normalization pipeline (ORDER MATTERS):
        1. Filter rare genes
        2. Freeze raw counts in adata.raw
        3. normalize_total() + log1p()
        4. Store lognorm layer
        5. HVG selection (flavor='seurat')
        6. scale() for PCA only — lognorm layer untouched
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


def get_lognorm_expression(adata, gene):
    """Safely retrieve lognorm-normalized expression for a gene."""
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


def embed_and_cluster(adata):
    """
    PCA -> neighbors -> UMAP -> Leiden clustering.

    TO DO LATER: Harmony batch correction — awaiting compute.
    When ready, uncomment the block below and install harmonypy:

    # import harmonypy as hm
    # ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'sample',
    #                     max_iter_harmony=20)
    # adata.obsm['X_pca_harmony'] = ho.Z_corr.T
    # sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, use_rep='X_pca_harmony')
    # (comment out the sc.pp.neighbors line below when using Harmony)
    """
    print("\n" + "=" * 80)
    print("EMBEDDING & CLUSTERING")
    print("=" * 80)

    sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)

    print(f"  {adata.obs['leiden'].nunique()} Leiden clusters")
    return adata


def score_cell_states(adata):
    """
    Score Neftel gene programs (MES, OPC, NPC, AC) using lognorm values.
    Temporarily swaps lognorm into adata.X so sc.tl.score_genes() uses
    meaningful expression values rather than z-scores.
    """
    print("\n" + "=" * 80)
    print("SCORING CELL STATES (Neftel et al. 2019)")
    print("=" * 80)

    gene_programs = get_neftel_gene_programs()

    original_X = adata.X.copy()
    adata.X = adata.layers["lognorm"]

    for program_name, genes in gene_programs.items():
        genes_present = [g for g in genes if g in adata.var_names]
        if genes_present:
            print(f"  {program_name}: {len(genes_present)}/{len(genes)} genes present")
            sc.tl.score_genes(adata, genes_present, score_name=f"{program_name}_score")

    adata.X = original_X

    if "MES1_score" in adata.obs and "MES2_score" in adata.obs:
        adata.obs["MES_score"] = (adata.obs["MES1_score"] + adata.obs["MES2_score"]) / 2
        print("  MES1 + MES2 -> MES_score")

    if "NPC1_score" in adata.obs and "NPC2_score" in adata.obs:
        adata.obs["NPC_score"] = (adata.obs["NPC1_score"] + adata.obs["NPC2_score"]) / 2
        print("  NPC1 + NPC2 -> NPC_score")

    if "AC_score" in adata.obs:
        print("  AC_score computed")

    return adata


def plot_state_map4k4_overlap(adata, state, output_dir):
    """
    One dedicated 3-panel figure per cell state:
        Panel 1: Cell state score on UMAP (viridis)
        Panel 2: MAP4K4 expression (lognorm) on UMAP (viridis)
        Panel 3: Three-group co-localization — same style as Tri-IPC figure
                 Gray  = all other cells
                 Color = state-high cells (top 25% by score)
                 Purple = state-high AND MAP4K4-high (overlap)
                 Spearman r and p-value in bottom-right corner
    """
    score_col = f"{state}_score"
    if score_col not in adata.obs.columns:
        print(f"  {score_col} not found — skipping {state}")
        return

    map4k4_expr = get_lognorm_expression(adata, FOCUS_GENE)
    if map4k4_expr is None:
        print(f"  {FOCUS_GENE} not found — skipping overlap figures")
        return

    adata.obs[f"{FOCUS_GENE}_lognorm"] = map4k4_expr

    # Define MAP4K4-high
    expressing = map4k4_expr[map4k4_expr > 0]
    m4k4_thresh = (
        np.percentile(expressing, FOCUS_GENE_PERCENTILE) if len(expressing) > 0 else 0
    )
    map4k4_high = map4k4_expr > m4k4_thresh

    # Define state-high (top 25% by score)
    state_scores = adata.obs[score_col].values
    state_thresh = np.percentile(state_scores, STATE_SCORE_PERCENTILE)
    state_high = state_scores > state_thresh

    # Overlap: state-high AND MAP4K4-high
    overlap = state_high & map4k4_high

    # Spearman correlation
    corr, pval = spearmanr(map4k4_expr, state_scores)
    sig = (
        "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    )

    print(
        f"  {state}: Spearman r={corr:.3f}, p={pval:.2e} {sig}  "
        f"| state-high={state_high.sum()}  "
        f"| MAP4K4-high={map4k4_high.sum()}  "
        f"| overlap={overlap.sum()}"
    )

    # ---- Figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(
        f"{state} Cell State  x  {FOCUS_GENE} Expression",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # Panel 1: State score
    sc.pl.umap(
        adata,
        color=score_col,
        cmap="viridis",
        vmax="p99",
        ax=axes[0],
        show=False,
        title=f"{state} Score",
    )

    # Panel 2: MAP4K4 expression
    sc.pl.umap(
        adata,
        color=f"{FOCUS_GENE}_lognorm",
        cmap="viridis",
        vmax="p99",
        ax=axes[1],
        show=False,
        title=f"{FOCUS_GENE} Expression (lognorm)",
    )

    # Panel 3: Three-group co-localization
    umap_coords = adata.obsm["X_umap"]
    state_color = STATE_COLORS.get(state, "steelblue")

    # Layer 1: all other cells — gray background
    other_mask = ~state_high & ~map4k4_high
    axes[2].scatter(
        umap_coords[other_mask, 0],
        umap_coords[other_mask, 1],
        c="lightgray",
        s=5,
        alpha=0.3,
        label="Other",
        rasterized=True,
    )

    # Layer 2: state-high cells
    axes[2].scatter(
        umap_coords[state_high, 0],
        umap_coords[state_high, 1],
        c=state_color,
        s=10,
        alpha=0.5,
        label=f"{state}-high (top {100 - STATE_SCORE_PERCENTILE}%)",
        rasterized=True,
    )

    # Layer 3: overlap — state-high AND MAP4K4-high
    if overlap.sum() > 0:
        axes[2].scatter(
            umap_coords[overlap, 0],
            umap_coords[overlap, 1],
            c="purple",
            s=25,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
            label=f"{state}-high + {FOCUS_GENE}-high",
            zorder=10,
            rasterized=True,
        )

    axes[2].set_title(f"{state}-high & {FOCUS_GENE}-high Co-localization")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")
    axes[2].legend(loc="upper left", fontsize=8)

    # Stats annotation bottom-right
    stats_text = f"Spearman r = {corr:.3f}\np = {pval:.2e} {sig}"
    axes[2].text(
        0.98,
        0.02,
        stats_text,
        transform=axes[2].transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    out_path = output_dir / f"{state}_MAP4K4_overlap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def create_visualizations(adata, output_dir, genes_of_interest=None):
    """
    All visualizations:
        1. Sample distribution
        2. Leiden clusters
        3. QC metrics
        4. Per-state MAP4K4 overlap (MES, OPC, NPC, AC) — one figure each
        5. Individual gene expression plots
        6. Dominant cell state
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample distribution
    print("\n  1. Sample distribution...")
    sc.pl.umap(
        adata,
        color="sample",
        show=False,
        legend_loc="on data",
        title="Sample Distribution",
        palette="tab20",
    )
    plt.savefig(output_dir / "integrated_samples.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Clusters
    print("  2. Clusters...")
    sc.pl.umap(
        adata, color="leiden", show=False, legend_loc="on data", title="Leiden Clusters"
    )
    plt.savefig(output_dir / "integrated_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. QC metrics
    print("  3. QC metrics...")
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
    plt.savefig(output_dir / "integrated_qc_umap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Per-state MAP4K4 overlap — one clean figure per state
    print(f"\n  4. {FOCUS_GENE} vs cell state overlaps...")
    for state in ["MES", "OPC", "NPC", "AC"]:
        plot_state_map4k4_overlap(adata, state, output_dir)

    # 5. Individual gene expression plots
    if genes_of_interest:
        print(f"\n  5. Gene expression plots...")
        genes_found = [g for g in genes_of_interest if g in adata.var_names]
        genes_missing = [g for g in genes_of_interest if g not in adata.var_names]

        if genes_missing:
            print(f"     Not found: {', '.join(genes_missing)}")

        for gene in genes_found:
            expr = get_lognorm_expression(adata, gene)
            if expr is None:
                continue
            adata.obs[f"{gene}_lognorm"] = expr

            fig, ax = plt.subplots(figsize=(9, 7))
            sc.pl.umap(
                adata,
                color=f"{gene}_lognorm",
                cmap="viridis",
                vmax="p99",
                title=f"{gene} (lognorm)",
                ax=ax,
                show=False,
            )
            plt.savefig(
                output_dir / f"integrated_{gene}_expression.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"     {gene} saved")

    # 6. Dominant cell state
    print("\n  6. Dominant cell state...")
    state_list = ["MES", "OPC", "NPC", "AC"]
    score_cols = [f"{s}_score" for s in state_list if f"{s}_score" in adata.obs.columns]

    if len(score_cols) >= 2:
        score_matrix = adata.obs[score_cols].values
        dominant_idx = np.argmax(score_matrix, axis=1)
        adata.obs["dominant_state"] = [
            score_cols[i].replace("_score", "") for i in dominant_idx
        ]
        sorted_scores = np.sort(score_matrix, axis=1)
        adata.obs["state_dominance"] = sorted_scores[:, -1] - sorted_scores[:, -2]
        adata.obs.loc[adata.obs["state_dominance"] < 0.5, "dominant_state"] = "Mixed"

        palette = {
            "MES": "#E74C3C",
            "OPC": "#3498DB",
            "NPC": "#2ECC71",
            "AC": "#F39C12",
            "Mixed": "lightgray",
        }
        sc.pl.umap(
            adata,
            color="dominant_state",
            show=False,
            title="Dominant Cell State",
            palette=palette,
            legend_loc="right margin",
        )
        plt.savefig(
            output_dir / "integrated_dominant_state.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"\n  All figures saved to: {output_dir}")
    return adata


def create_summary_stats(adata, output_dir):
    """Summary statistics CSV."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    output_dir = Path(output_dir)

    print(f"\n  Total cells:  {adata.n_obs}")
    print(f"  Total genes:  {adata.n_vars}")
    print(f"  Samples:      {adata.obs['sample'].nunique()}")
    print(f"  Clusters:     {adata.obs['leiden'].nunique()}")

    print("\n  Per-sample cell counts:")
    for sample, count in adata.obs["sample"].value_counts().sort_index().items():
        print(f"    {sample}: {count} cells")

    if "dominant_state" in adata.obs.columns:
        print("\n  Cell state distribution:")
        for state, count in adata.obs["dominant_state"].value_counts().items():
            print(f"    {state}: {count} ({count/adata.n_obs*100:.1f}%)")

    stats_rows = []
    for sample in adata.obs["sample"].unique():
        mask = adata.obs["sample"] == sample
        row = {"sample": sample, "n_cells": int(mask.sum())}
        if "dominant_state" in adata.obs.columns:
            for state in ["MES", "OPC", "NPC", "AC", "Mixed"]:
                n = int(
                    (
                        (adata.obs["sample"] == sample)
                        & (adata.obs["dominant_state"] == state)
                    ).sum()
                )
                row[f"{state}_cells"] = n
                row[f"{state}_pct"] = round(n / mask.sum() * 100, 1)
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_dir / "integrated_summary.csv", index=False)
    print(f"\n  Saved integrated_summary.csv")

    return stats_df


def main():
    print("\n" + "=" * 80)
    print("INTEGRATED pHGG ANALYSIS v2")
    print(f"Samples: {', '.join(SAMPLES_TO_ANALYZE)}")
    print("=" * 80)

    BASE_PATH = "/Users/fudhailsayed/prololab"
    DATA_DIR = f"{BASE_PATH}/datasets/pHGG_scRNA_anndata/scRNA"
    OUTPUT_DIR = f"{BASE_PATH}/figures/integrated_analysis_v2"

    adata = load_all_samples(DATA_DIR)
    adata = preprocess_integrated(adata)
    adata = embed_and_cluster(adata)
    adata = score_cell_states(adata)
    adata = create_visualizations(
        adata, OUTPUT_DIR, genes_of_interest=GENES_OF_INTEREST
    )
    create_summary_stats(adata, OUTPUT_DIR)

    out_h5ad = Path(OUTPUT_DIR) / "integrated_all_samples_v2.h5ad"
    # Fix mixed-type obs columns before saving
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)
    adata.write(out_h5ad)
    print(f"\n  Saved: {out_h5ad}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print(f"Results: {OUTPUT_DIR}")
    print("=" * 80)

    return adata


if __name__ == "__main__":
    adata = main()
