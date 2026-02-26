"""
Tri-IPC Cell Identification and Characterization for pHGG scRNA-seq (v2)
=========================================================================
Improved version with:
    1. Proper QC filtering (MT%, min genes/counts)
    2. normalize_total() + log1p() BEFORE HVG selection
    3. LogNorm layer stored for downstream expression retrieval
    4. gene expression retrieved from lognorm layer (not scaled X)
    5. adata.raw set from truly unprocessed counts (before any filtering)
    6. HVG flavor changed to 'seurat' (appropriate for lognorm data)
    7. Regression of technical covariates (total_counts, pct_counts_mt)
    8. FDR correction (Benjamini-Hochberg) across gene list comparisons
    9. Single-sample mode for easier troubleshooting

⚠️  Harmony batch correction is stubbed out — enable when compute allows.

Author: Fudhail Sayed
Updated: 2/26/26
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

sc.settings.set_figure_params(dpi=80, facecolor='white')

# ============================================================================
# CONFIGURATION — EDIT THIS SECTION
# ============================================================================

# Single sample for now (add more to the list later when ready)
SAMPLES_TO_ANALYZE = [
    'SCPCS000001',
    'SCPCS000002',  
    'SCPCS000003',
]

# Genes of interest
GENES_OF_INTEREST = [
    'MAP4K4',
    'EGFR',
    'OLIG2',
    # Add more here
]

# Tri-IPC thresholds
# Using lognorm value thresholds rather than percentile-based for reproducibility.
# Adjust after inspecting expression distributions with plot_expression_distributions().
EGFR_LOGNORM_THRESHOLD  = 0.5
OLIG2_LOGNORM_THRESHOLD = 0.5

# QC thresholds — inspect violin plots first, then adjust
MIN_GENES_PER_CELL   = 500     # Remove low-complexity cells
MAX_GENES_PER_CELL   = 8000    # Remove likely doublets
MAX_PCT_MT           = 20      # Remove dying/low-quality cells
MIN_CELLS_PER_GENE   = 10      # Remove very rare genes

# Additional markers
SURFACE_MARKERS      = ['F3', 'CD38', 'PDGFRA', 'ITGA2']
VALIDATION_NEGATIVE  = ['RBFOX3', 'SPARCL1', 'DLX5']  # Should be LOW in Tri-IPC

# ============================================================================


def load_samples(base_path, sample_list):
    """Load and concatenate multiple samples."""
    print("\n" + "="*80)
    print(f"LOADING {len(sample_list)} SAMPLES")
    print("="*80)

    adatas = []
    for sample_name in sample_list:
        try:
            adata = load_sample(base_path, sample_name)
            adatas.append(adata)
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipping {sample_name}: {e}")

    if len(adatas) == 0:
        raise ValueError("No samples loaded!")
    if len(adatas) == 1:
        return adatas[0]

    combined = sc.concat(adatas, join='outer', label='sample',
                         keys=[a.obs['sample'].iloc[0] for a in adatas])

    if hasattr(combined.X, 'toarray'):
        combined.X = csr_matrix(combined.X.astype(np.float32))
    else:
        combined.X = csr_matrix(np.asarray(combined.X, dtype=np.float32))

    print(f"\n  ✓ Combined: {combined.n_obs} cells × {combined.n_vars} genes")
    return combined


def load_sample(base_path, sample_name):
    """
    Load a single sample and store truly raw counts before any modification.
    adata.raw is set HERE — before filtering — so it reflects unprocessed counts.
    """
    print("\n" + "="*80)
    print(f"LOADING SAMPLE: {sample_name}")
    print("="*80)

    data_dir  = Path(base_path) / 'datasets/pHGG_scRNA_anndata/scRNA'
    sample_dir = data_dir / sample_name
    h5ad_file  = sample_dir / f'{sample_name.replace("SCPCS", "SCPCL")}_filtered_rna_symbols.h5ad'

    if not h5ad_file.exists():
        raise FileNotFoundError(f"File not found: {h5ad_file}")

    adata = sc.read_h5ad(h5ad_file)

    # Ensure sparse float32
    if hasattr(adata.X, 'toarray'):
        adata.X = csr_matrix(adata.X.astype(np.float32))
    else:
        adata.X = csr_matrix(np.asarray(adata.X, dtype=np.float32))

    adata.obs['sample'] = sample_name
    print(f"  ✓ Loaded {adata.n_obs} cells × {adata.n_vars} genes")

    return adata


def run_qc(adata):
    """
    QC metrics, violin plots, and filtering.
    Inspect the saved violin plot before finalising thresholds.
    """
    print("\n" + "="*80)
    print("QC FILTERING")
    print("="*80)

    # Mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None,
                               log1p=False, inplace=True)

    print(f"  Before filtering: {adata.n_obs} cells")
    print(f"    Median genes/cell:      {adata.obs['n_genes_by_counts'].median():.0f}")
    print(f"    Median counts/cell:     {adata.obs['total_counts'].median():.0f}")
    print(f"    Median %MT:             {adata.obs['pct_counts_mt'].median():.1f}%")

    # Filter cells
    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    sc.pp.filter_cells(adata, max_genes=MAX_GENES_PER_CELL)
    adata = adata[adata.obs['pct_counts_mt'] < MAX_PCT_MT].copy()

    # Filter genes (applied after cell filter so counts are accurate)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)

    print(f"\n  After filtering:  {adata.n_obs} cells × {adata.n_vars} genes")

    return adata


def normalize_and_store_lognorm(adata):
    """
    Normalization pipeline (ORDER MATTERS):
        1. Store raw counts in adata.raw (genes already QC-filtered)
        2. normalize_total → depth-normalize each cell to 10,000 counts
        3. log1p          → log-transform (stabilizes variance)
        4. Store lognorm values as a named layer for safe retrieval later
        5. HVG selection on lognorm values (flavor='seurat')
        6. scale()        → zero-mean, unit-variance for PCA only
    """
    print("\n" + "="*80)
    print("NORMALIZATION")
    print("="*80)

    # --- Step 1: freeze raw counts ----------------------------------------
    adata.raw = adata          # adata.X still holds raw integer counts here
    print("  ✓ Raw counts frozen in adata.raw")

    # --- Step 2 & 3: normalize + log1p ------------------------------------
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("  ✓ normalize_total(1e4) + log1p applied")

    # --- Step 4: store lognorm layer --------------------------------------
    if hasattr(adata.X, 'toarray'):
        adata.layers['lognorm'] = csr_matrix(adata.X.astype(np.float32))
    else:
        adata.layers['lognorm'] = adata.X.copy().astype(np.float32)
    print("  ✓ lognorm layer stored in adata.layers['lognorm']")

    # --- Step 5: HVG on lognorm values ------------------------------------
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
    n_hvg = adata.var['highly_variable'].sum()
    print(f"  ✓ {n_hvg} highly variable genes selected (flavor='seurat')")

    # --- Step 6: regress out technical variation --------------------------
    print("  Regressing out total_counts and pct_counts_mt...")
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    print("  ✓ Regressed out technical covariates")

    # --- Step 7: scale for PCA only (does NOT affect lognorm layer) -------
    sc.pp.scale(adata, max_value=10)
    print("  ✓ Scaled (z-score, max_value=10) — PCA only; expression retrieved from lognorm layer")

    return adata


def embed_and_cluster(adata):
    """PCA → neighbors → UMAP → Leiden clustering."""
    print("\n" + "="*80)
    print("EMBEDDING & CLUSTERING")
    print("="*80)

    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)

    n_clusters = adata.obs['leiden'].nunique()
    print(f"  ✓ {n_clusters} Leiden clusters (resolution=0.5)")

    # NOTE: Harmony batch correction goes here when multiple samples are used.
    # Uncomment below and install harmonypy when compute allows:
    #
    #   import harmonypy as hm
    #   ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'sample')
    #   adata.obsm['X_pca_harmony'] = ho.Z_corr.T
    #   sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=15, n_pcs=40)
    #   sc.tl.umap(adata)
    #   sc.tl.leiden(adata, resolution=0.5)

    return adata


def get_lognorm_expression(adata, gene):
    """
    Safely retrieve lognorm-normalized expression for a gene.
    Priority: lognorm layer → raw → scaled X (last resort, with warning).
    """
    if 'lognorm' in adata.layers and gene in adata.var_names:
        idx  = list(adata.var_names).index(gene)
        expr = adata.layers['lognorm'][:, idx]

    elif hasattr(adata, 'raw') and adata.raw is not None and gene in adata.raw.var_names:
        print(f"  ⚠️  {gene}: lognorm layer missing — falling back to adata.raw (raw counts!)")
        idx  = list(adata.raw.var_names).index(gene)
        expr = adata.raw.X[:, idx]

    elif gene in adata.var_names:
        print(f"  ⚠️  {gene}: using scaled adata.X — values are z-scores, not expression!")
        idx  = list(adata.var_names).index(gene)
        expr = adata.X[:, idx]

    else:
        return None

    if hasattr(expr, 'toarray'):
        expr = expr.toarray().flatten()
    else:
        expr = np.asarray(expr).flatten()

    return expr.astype(np.float32)


def plot_expression_distributions(adata, genes, output_dir, thresholds=None):
    """
    Violin plots of lognorm expression for key genes.
    Use these to validate and adjust EGFR/OLIG2 lognorm thresholds.
    """
    print("\n  Plotting expression distributions for threshold validation...")
    fig, axes = plt.subplots(1, len(genes), figsize=(5*len(genes), 5))
    if len(genes) == 1:
        axes = [axes]

    for ax, gene in zip(axes, genes):
        expr = get_lognorm_expression(adata, gene)
        if expr is None:
            ax.set_title(f'{gene}\n(not found)')
            continue

        expressing = expr[expr > 0]
        thresh = thresholds.get(gene, 0.5) if thresholds else 0.5
        ax.violinplot(expressing, showmedians=True)
        ax.axhline(y=thresh, color='red', linestyle='--', label=f'threshold ({thresh})')
        ax.set_title(f'{gene}\n(n expressing={len(expressing)})')
        ax.set_ylabel('lognorm expression')
        ax.legend(fontsize=7)

    plt.suptitle('Expression distributions — use to calibrate thresholds', fontsize=11)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'expression_distributions.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved expression_distributions.png — inspect before finalising thresholds")


def identify_tri_ipc_cells(adata, egfr_threshold=0.5, olig2_threshold=0.5):
    """
    Identify Tri-IPC cells: EGFR+ AND OLIG2+ using lognorm thresholds.
    Thresholds are fixed values in lognorm space (not percentile-based),
    making results reproducible regardless of dataset composition.
    """
    print("\n" + "="*80)
    print("IDENTIFYING TRI-IPC CELLS")
    print("="*80)
    print(f"  Thresholds: EGFR lognorm > {egfr_threshold}, OLIG2 lognorm > {olig2_threshold}")

    for gene, threshold, label in [
        ('EGFR',  egfr_threshold,  'EGFR'),
        ('OLIG2', olig2_threshold, 'OLIG2'),
    ]:
        expr = get_lognorm_expression(adata, gene)
        if expr is None:
            print(f"  ✗ {gene} not found!")
            return adata

        adata.obs[f'{gene}_expr'] = expr
        adata.obs[f'{gene}_high'] = expr > threshold

        pct_exp  = (expr > 0).sum() / len(expr) * 100
        pct_high = adata.obs[f'{gene}_high'].sum() / adata.n_obs * 100
        print(f"\n  {gene}:")
        print(f"    % cells expressing (>0):   {pct_exp:.1f}%")
        print(f"    % cells above threshold:   {pct_high:.1f}%  (lognorm > {threshold})")
        print(f"    Mean (all cells):           {expr.mean():.3f}")
        print(f"    Mean (expressing cells):    {expr[expr>0].mean():.3f}")

    adata.obs['Tri_IPC'] = adata.obs['EGFR_high'] & adata.obs['OLIG2_high']
    n   = adata.obs['Tri_IPC'].sum()
    pct = n / adata.n_obs * 100
    print(f"\n  ✓ Tri-IPC cells (EGFR+ AND OLIG2+): {n} ({pct:.1f}%)")

    # Surface markers
    print(f"\n  Surface marker enrichment (Tri-IPC vs Other):")
    for marker in SURFACE_MARKERS:
        expr = get_lognorm_expression(adata, marker)
        if expr is not None:
            adata.obs[f'{marker}_expr'] = expr
            t = expr[adata.obs['Tri_IPC']]
            o = expr[~adata.obs['Tri_IPC']]
            fc = t.mean() / (o.mean() + 1e-10)
            print(f"    {marker:10s}  Tri-IPC={t.mean():.3f}  Other={o.mean():.3f}  FC={fc:.2f}x")
        else:
            print(f"    {marker}: not found")

    # Negative validation markers
    print(f"\n  Negative validation markers (should be LOW in Tri-IPC):")
    for marker in VALIDATION_NEGATIVE:
        expr = get_lognorm_expression(adata, marker)
        if expr is not None:
            adata.obs[f'{marker}_expr'] = expr
            t = expr[adata.obs['Tri_IPC']]
            o = expr[~adata.obs['Tri_IPC']]
            status = "✓ low" if t.mean() < o.mean() else "⚠️  ELEVATED"
            print(f"    {marker:10s}  Tri-IPC={t.mean():.3f}  Other={o.mean():.3f}  {status}")

    return adata


def characterize_tri_ipc(adata):
    """Distribution of Tri-IPC cells by sample and cluster."""
    print("\n" + "="*80)
    print("CHARACTERIZING TRI-IPC CELLS")
    print("="*80)

    if 'Tri_IPC' not in adata.obs.columns:
        print("  ✗ Tri-IPC cells not identified — run identify_tri_ipc_cells() first")
        return adata

    print("\n  By sample:")
    for sample in adata.obs['sample'].unique():
        mask  = adata.obs['sample'] == sample
        n_tri = (mask & adata.obs['Tri_IPC']).sum()
        total = mask.sum()
        print(f"    {sample}: {n_tri}/{total} ({n_tri/total*100:.1f}%)")

    print("\n  By Leiden cluster:")
    for cluster in sorted(adata.obs['leiden'].unique(), key=int):
        mask  = adata.obs['leiden'] == cluster
        n_tri = (mask & adata.obs['Tri_IPC']).sum()
        total = mask.sum()
        pct   = n_tri / total * 100 if total > 0 else 0
        flag  = "  ⭐ enriched" if pct > 20 else ""
        print(f"    Cluster {cluster:>2s}: {n_tri:>4}/{total:<5} ({pct:4.1f}%){flag}")

    return adata


def analyze_genes(adata, gene_list):
    """
    Compare expression of genes of interest in Tri-IPC vs non-Tri-IPC cells.
    Uses lognorm values. Applies Benjamini-Hochberg FDR correction.
    """
    print("\n" + "="*80)
    print("GENE ANALYSIS: Tri-IPC vs Non-Tri-IPC")
    print("="*80)

    results = []
    n_tri   = adata.obs['Tri_IPC'].sum()

    for gene in gene_list:
        expr = get_lognorm_expression(adata, gene)
        if expr is None:
            print(f"\n  {gene}: not found — skipping")
            continue

        adata.obs[f'{gene}_expr'] = expr

        tri_expr     = expr[adata.obs['Tri_IPC']]
        non_tri_expr = expr[~adata.obs['Tri_IPC']]

        stat, pval = mannwhitneyu(tri_expr, non_tri_expr, alternative='two-sided')
        fc         = tri_expr.mean() / (non_tri_expr.mean() + 1e-10)

        # High = top 25% of expressing cells (for overlap calculation)
        threshold = np.percentile(expr[expr > 0], 75) if (expr > 0).any() else 0
        adata.obs[f'{gene}_high'] = expr > threshold
        overlap    = (adata.obs['Tri_IPC'] & adata.obs[f'{gene}_high']).sum()
        pct_over   = overlap / n_tri * 100 if n_tri > 0 else 0

        # Spearman correlations
        egfr_corr = olig2_corr = np.nan
        if 'EGFR_expr' in adata.obs.columns:
            egfr_corr, _ = spearmanr(expr, adata.obs['EGFR_expr'])
        if 'OLIG2_expr' in adata.obs.columns:
            olig2_corr, _ = spearmanr(expr, adata.obs['OLIG2_expr'])

        print(f"\n  {gene}:")
        print(f"    Tri-IPC mean ± SD:    {tri_expr.mean():.3f} ± {tri_expr.std():.3f}")
        print(f"    Non-Tri-IPC mean ± SD:{non_tri_expr.mean():.3f} ± {non_tri_expr.std():.3f}")
        print(f"    Fold change:          {fc:.2f}x")
        print(f"    p-value (raw):        {pval:.2e}")
        print(f"    Overlap w/ Tri-IPC:   {overlap}/{n_tri} ({pct_over:.1f}%)")
        print(f"    Spearman r (EGFR):    {egfr_corr:.3f}")
        print(f"    Spearman r (OLIG2):   {olig2_corr:.3f}")

        results.append({
            'gene':           gene,
            'tri_ipc_mean':   tri_expr.mean(),
            'non_tri_mean':   non_tri_expr.mean(),
            'fold_change':    fc,
            'pvalue_raw':     pval,
            'pct_overlap':    pct_over,
            'corr_egfr':      egfr_corr,
            'corr_olig2':     olig2_corr,
        })

    if not results:
        return adata, pd.DataFrame()

    results_df = pd.DataFrame(results)

    # FDR correction
    _, pvals_adj, _, _ = multipletests(results_df['pvalue_raw'], method='fdr_bh')
    results_df['pvalue_adj_BH'] = pvals_adj
    results_df['significant']   = pvals_adj < 0.05

    print("\n" + "-"*60)
    print("  FDR-adjusted p-values (Benjamini-Hochberg):")
    for _, row in results_df.iterrows():
        sig = "***" if row['pvalue_adj_BH'] < 0.001 else \
              "**"  if row['pvalue_adj_BH'] < 0.01  else \
              "*"   if row['pvalue_adj_BH'] < 0.05  else "ns"
        print(f"    {row['gene']:12s}  FC={row['fold_change']:.2f}x  "
              f"p_adj={row['pvalue_adj_BH']:.2e} {sig}")

    return adata, results_df


def create_visualizations(adata, gene_list, output_dir):
    """Generate all UMAP plots, overlays, and summary heatmap."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    sc.pl.umap(adata, color='sample',    ax=axes[0], show=False, title='Samples')
    sc.pl.umap(adata, color='leiden',    ax=axes[1], show=False, title='Leiden Clusters',
               legend_loc='on data')
    sc.pl.umap(adata, color='EGFR_expr', ax=axes[2], show=False, title='EGFR (lognorm)',
               cmap='viridis', vmax='p99')
    sc.pl.umap(adata, color='OLIG2_expr',ax=axes[3], show=False, title='OLIG2 (lognorm)',
               cmap='viridis', vmax='p99')
    plt.tight_layout()
    plt.savefig(output_dir / 'overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ overview.png")

    # 2. QC metrics on UMAP
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[0], show=False,
               title='Genes per cell', cmap='viridis')
    sc.pl.umap(adata, color='total_counts',      ax=axes[1], show=False,
               title='Total counts', cmap='viridis')
    sc.pl.umap(adata, color='pct_counts_mt',     ax=axes[2], show=False,
               title='% MT', cmap='viridis')
    plt.tight_layout()
    plt.savefig(output_dir / 'qc_umap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ qc_umap.png")

    # 3. Tri-IPC identification
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sc.pl.umap(adata, color='EGFR_high',  ax=axes[0], show=False,
               title='EGFR-high', palette=['lightgray', 'green'])
    sc.pl.umap(adata, color='OLIG2_high', ax=axes[1], show=False,
               title='OLIG2-high', palette=['lightgray', 'blue'])
    sc.pl.umap(adata, color='Tri_IPC',    ax=axes[2], show=False,
               title='Tri-IPC (EGFR+ OLIG2+)', palette=['lightgray', 'red'])
    plt.tight_layout()
    plt.savefig(output_dir / 'tri_ipc_identification.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ tri_ipc_identification.png")

    # 4. Gene overlays (lognorm)
    valid_genes = [g for g in gene_list if f'{g}_expr' in adata.obs.columns]
    if valid_genes:
        n_cols = 3
        n_rows = (len(valid_genes) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = np.array(axes).flatten()

        for i, gene in enumerate(valid_genes):
            sc.pl.umap(adata, color=f'{gene}_expr', cmap='viridis', vmax='p99',
                       ax=axes[i], show=False, title=f'{gene} (lognorm)')
        for j in range(len(valid_genes), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'genes_of_interest.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ genes_of_interest.png")

    # 5. Per-gene overlay: expression + Tri-IPC + co-localization
    for gene in valid_genes:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        sc.pl.umap(adata, color=f'{gene}_expr', cmap='viridis', vmax='p99',
                   ax=axes[0], show=False, title=f'{gene} (lognorm)')
        sc.pl.umap(adata, color='Tri_IPC', palette=['lightgray', 'red'],
                   ax=axes[1], show=False, title='Tri-IPC cells')

        umap = adata.obsm['X_umap']
        tri  = adata.obs['Tri_IPC'].values
        high = adata.obs.get(f'{gene}_high', pd.Series(False, index=adata.obs_names)).values
        over = tri & high

        axes[2].scatter(umap[:,0], umap[:,1], c='lightgray', s=4, alpha=0.3, label='Other',      rasterized=True)
        axes[2].scatter(umap[tri, 0], umap[tri, 1], c='red',    s=12, alpha=0.5, label='Tri-IPC', rasterized=True)
        if over.sum() > 0:
            axes[2].scatter(umap[over,0], umap[over,1], c='purple', s=25, alpha=0.9,
                            edgecolors='black', linewidths=0.5,
                            label=f'Tri-IPC + {gene}-high', zorder=10, rasterized=True)
        axes[2].set_xlabel('UMAP1'); axes[2].set_ylabel('UMAP2')
        axes[2].set_title(f'Tri-IPC & {gene} Co-localization')
        axes[2].legend(loc='upper left', fontsize=8)

        # Stats annotation in bottom-right corner
        tri_expr_vals = adata.obs.loc[adata.obs['Tri_IPC'],  f'{gene}_expr'].values
        non_tri_vals  = adata.obs.loc[~adata.obs['Tri_IPC'], f'{gene}_expr'].values
        _, pval = mannwhitneyu(tri_expr_vals, non_tri_vals, alternative='two-sided')
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        stats_text = f"Mann-Whitney U\np = {pval:.2e} {sig}"
        axes[2].text(0.98, 0.02, stats_text,
                     transform=axes[2].transAxes,
                     fontsize=8, verticalalignment='bottom',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_dir / f'{gene}_tri_ipc_overlay.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {gene}_tri_ipc_overlay.png")

    # 6. Cluster summary heatmap
    summary_rows = []
    for cluster in sorted(adata.obs['leiden'].unique(), key=int):
        mask     = adata.obs['leiden'] == cluster
        tri_mask = mask & adata.obs['Tri_IPC']
        row = {
            'cluster':    f'C{cluster}',
            'n_cells':    int(mask.sum()),
            'pct_Tri_IPC': tri_mask.sum() / mask.sum() * 100 if mask.sum() > 0 else 0,
        }
        for gene in valid_genes:
            col = f'{gene}_expr'
            if col in adata.obs.columns:
                row[f'{gene}_triIPC_mean'] = (
                    adata.obs.loc[tri_mask, col].mean() if tri_mask.sum() > 0 else 0
                )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'cluster_summary.csv', index=False)

    plot_cols = [c for c in summary_df.columns if c not in ('cluster', 'n_cells')]
    if plot_cols:
        heat_data = summary_df[plot_cols].set_index(summary_df['cluster']).T
        fig, ax = plt.subplots(figsize=(max(8, len(summary_df)*0.7), max(6, len(plot_cols)*0.8)))
        sns.heatmap(heat_data, annot=True, fmt='.1f', cmap='RdYlBu_r',
                    linewidths=0.3, ax=ax, cbar_kws={'label': 'Value'})
        ax.set_title('Tri-IPC Characterization by Cluster (lognorm expression)')
        plt.tight_layout()
        plt.savefig(output_dir / 'cluster_summary_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ cluster_summary_heatmap.png")

    print(f"\n  All figures saved to: {output_dir}")


def main():
    print("="*80)
    print("TRI-IPC ANALYSIS v2")
    print(f"Samples: {', '.join(SAMPLES_TO_ANALYZE)}")
    print("="*80)

    BASE_PATH  = '/Users/fudhailsayed/prololab'
    OUTPUT_DIR = Path(BASE_PATH) / 'figures' / 'tri_ipc_v2'

    # ---- Load ----------------------------------------------------------------
    adata = load_samples(BASE_PATH, SAMPLES_TO_ANALYZE)

    # ---- QC ------------------------------------------------------------------
    adata = run_qc(adata)

    # ---- Normalize (sets raw, lognorm layer, HVG, regression, scale) ---------
    adata = normalize_and_store_lognorm(adata)

    # ---- Embed & cluster -----------------------------------------------------
    adata = embed_and_cluster(adata)

    # ---- Inspect threshold distributions FIRST (save plot, then adjust if needed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_expression_distributions(adata, ['EGFR', 'OLIG2'], OUTPUT_DIR,
                                  thresholds={'EGFR':  EGFR_LOGNORM_THRESHOLD,
                                              'OLIG2': OLIG2_LOGNORM_THRESHOLD})

    # ---- Identify Tri-IPC ----------------------------------------------------
    adata = identify_tri_ipc_cells(adata,
                                    egfr_threshold=EGFR_LOGNORM_THRESHOLD,
                                    olig2_threshold=OLIG2_LOGNORM_THRESHOLD)

    # ---- Characterize --------------------------------------------------------
    adata = characterize_tri_ipc(adata)

    # ---- Gene analysis + FDR -------------------------------------------------
    adata, results_df = analyze_genes(adata, GENES_OF_INTEREST)

    if not results_df.empty:
        results_df.to_csv(OUTPUT_DIR / 'gene_analysis_summary.csv', index=False)
        print(f"\n  ✓ Saved gene_analysis_summary.csv")

    # ---- Visualize -----------------------------------------------------------
    create_visualizations(adata, GENES_OF_INTEREST, OUTPUT_DIR)

    # ---- Save processed AnnData ----------------------------------------------
    out_h5ad = OUTPUT_DIR / 'tri_ipc_v2_processed.h5ad'
    adata.write(out_h5ad)
    print(f"\n  ✓ Saved processed AnnData: {out_h5ad}")

    # ---- Summary -------------------------------------------------------------
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    n_tri = int(adata.obs['Tri_IPC'].sum())
    print(f"  Cells analysed:  {adata.n_obs}")
    print(f"  Tri-IPC cells:   {n_tri} ({n_tri/adata.n_obs*100:.1f}%)")

    if not results_df.empty:
        print("\n  Gene results (FDR-corrected):")
        for _, row in results_df.iterrows():
            sig = ("***" if row['pvalue_adj_BH'] < 0.001 else
                   "**"  if row['pvalue_adj_BH'] < 0.01  else
                   "*"   if row['pvalue_adj_BH'] < 0.05  else "ns")
            print(f"    {row['gene']:12s}  FC={row['fold_change']:.2f}x  "
                  f"p_adj={row['pvalue_adj_BH']:.2e} {sig}  "
                  f"overlap={row['pct_overlap']:.1f}%")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    return adata


if __name__ == "__main__":
    adata = main()
