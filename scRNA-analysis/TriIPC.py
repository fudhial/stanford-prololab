"""
Tri-IPC (EGFR+ OLIG2+) Analysis for PHGG
Focused analysis using developmental progenitor markers
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu, spearmanr

sc.settings.set_figure_params(dpi=80, facecolor='white')

# ============================================================================
# ⭐⭐⭐ CUSTOMIZABLE SECTION ⭐⭐⭐
# ============================================================================

# Samples to analyze
SAMPLES_TO_ANALYZE = [
    'SCPCS000001',
    'SCPCS000002',
    # Add more as needed
]

# Genes of interest to analyze
GENES_OF_INTEREST = [
    'ST8SIA2',
    'EGFR',
    'OLIG2',
    # Add more genes here
]

# Tri-IPC marker thresholds
EGFR_PERCENTILE = 75  # Top 25% = "high"
OLIG2_PERCENTILE = 75

# Additional markers to check
SURFACE_MARKERS = ['F3', 'CD38', 'PDGFRA', 'ITGA2']
VALIDATION_NEGATIVE = ['RBFOX3', 'SPARCL1', 'DLX5']  # Should be low/negative

# ============================================================================

def load_samples(base_path, sample_list):
    """Load and combine samples"""
    print("\n" + "="*80)
    print("LOADING SAMPLES")
    print("="*80)
    
    data_dir = Path(base_path) / 'data/datasets/pHGG_scRNA_anndata/scRNA'
    adatas = []
    
    for sample_name in sample_list:
        sample_dir = data_dir / sample_name
        h5ad_file = sample_dir / f'{sample_name.replace("SCPCS", "SCPCL")}_filtered_rna_symbols.h5ad'
        
        if h5ad_file.exists():
            print(f"  Loading {sample_name}...")
            adata = sc.read_h5ad(h5ad_file)
            
            if hasattr(adata.X, 'toarray'):
                adata.X = csr_matrix(adata.X.astype(np.float32))
            else:
                adata.X = np.asarray(adata.X, dtype=np.float32)
            
            adata.obs['sample'] = sample_name
            adatas.append(adata)
            print(f"    ✓ {adata.n_obs} cells")
    
    if len(adatas) == 0:
        raise ValueError("No samples loaded!")
    
    print(f"\n✓ Loaded {len(adatas)} samples")
    
    # Concatenate
    adata_combined = sc.concat(adatas, join='outer')
    
    if hasattr(adata_combined.X, 'toarray'):
        adata_combined.X = csr_matrix(adata_combined.X.astype(np.float32))
    else:
        adata_combined.X = np.asarray(adata_combined.X, dtype=np.float32)
    
    print(f"✓ Combined: {adata_combined.n_obs} cells, {adata_combined.n_vars} genes")
    
    return adata_combined

def process_data(adata):
    """Preprocess and cluster"""
    print("\n" + "="*80)
    print("PREPROCESSING")
    print("="*80)
    
    adata.raw = adata
    
    sc.pp.filter_genes(adata, min_cells=10)
    print(f"  Genes: {adata.n_vars}")
    
    print("  Finding highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger')
    
    print("  Scaling...")
    sc.pp.scale(adata, max_value=10)
    
    print("  PCA, neighbors, UMAP...")
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata)
    
    print("  Clustering...")
    sc.tl.leiden(adata, resolution=0.5)
    print(f"    {len(adata.obs['leiden'].unique())} clusters")
    
    return adata

def get_gene_expression(adata, gene):
    """Get gene expression"""
    if hasattr(adata, 'raw') and adata.raw is not None and gene in adata.raw.var_names:
        gene_idx = list(adata.raw.var_names).index(gene)
        expr = adata.raw.X[:, gene_idx]
    elif gene in adata.var_names:
        gene_idx = list(adata.var_names).index(gene)
        expr = adata.X[:, gene_idx]
    else:
        return None
    
    if hasattr(expr, 'toarray'):
        expr = expr.toarray().flatten()
    else:
        expr = np.asarray(expr).flatten()
    
    return expr

def identify_tri_ipc_cells(adata, egfr_percentile=75, olig2_percentile=75):
    """
    Identify Tri-IPC cells using EGFR+ OLIG2+ criteria
    """
    print("\n" + "="*80)
    print("IDENTIFYING TRI-IPC CELLS (EGFR+ OLIG2+)")
    print("="*80)
    
    # EGFR
    egfr_expr = get_gene_expression(adata, 'EGFR')
    if egfr_expr is None:
        print("  ✗ EGFR not found!")
        return adata
    
    adata.obs['EGFR_expr'] = egfr_expr
    egfr_threshold = np.percentile(egfr_expr[egfr_expr > 0], egfr_percentile) if (egfr_expr > 0).any() else 0
    adata.obs['EGFR_high'] = egfr_expr > egfr_threshold
    
    print(f"\n  EGFR:")
    print(f"    Mean expression: {egfr_expr.mean():.3f}")
    print(f"    % expressing: {(egfr_expr > 0).sum()/len(egfr_expr)*100:.1f}%")
    print(f"    Threshold (p{egfr_percentile}): {egfr_threshold:.3f}")
    print(f"    High cells: {adata.obs['EGFR_high'].sum()} ({adata.obs['EGFR_high'].sum()/adata.n_obs*100:.1f}%)")
    
    # OLIG2
    olig2_expr = get_gene_expression(adata, 'OLIG2')
    if olig2_expr is None:
        print("  ✗ OLIG2 not found!")
        return adata
    
    adata.obs['OLIG2_expr'] = olig2_expr
    olig2_threshold = np.percentile(olig2_expr[olig2_expr > 0], olig2_percentile) if (olig2_expr > 0).any() else 0
    adata.obs['OLIG2_high'] = olig2_expr > olig2_threshold
    
    print(f"\n  OLIG2:")
    print(f"    Mean expression: {olig2_expr.mean():.3f}")
    print(f"    % expressing: {(olig2_expr > 0).sum()/len(olig2_expr)*100:.1f}%")
    print(f"    Threshold (p{olig2_percentile}): {olig2_threshold:.3f}")
    print(f"    High cells: {adata.obs['OLIG2_high'].sum()} ({adata.obs['OLIG2_high'].sum()/adata.n_obs*100:.1f}%)")
    
    # Define Tri-IPC: EGFR+ AND OLIG2+
    adata.obs['Tri_IPC'] = adata.obs['EGFR_high'] & adata.obs['OLIG2_high']
    
    n_tri_ipc = adata.obs['Tri_IPC'].sum()
    print(f"\n  ✓ Tri-IPC cells (EGFR+ OLIG2+): {n_tri_ipc} ({n_tri_ipc/adata.n_obs*100:.1f}%)")
    
    # Check surface markers
    print(f"\n  Additional surface markers:")
    for marker in SURFACE_MARKERS:
        expr = get_gene_expression(adata, marker)
        if expr is not None:
            adata.obs[f'{marker}_expr'] = expr
            # Check expression in Tri-IPC vs non-Tri-IPC
            tri_ipc_expr = expr[adata.obs['Tri_IPC']]
            other_expr = expr[~adata.obs['Tri_IPC']]
            print(f"    {marker}: Tri-IPC mean={tri_ipc_expr.mean():.3f}, Other mean={other_expr.mean():.3f}, FC={tri_ipc_expr.mean()/(other_expr.mean()+1e-10):.2f}x")
        else:
            print(f"    {marker}: not found")
    
    # Check negative markers (should be low in Tri-IPC)
    print(f"\n  Validation markers (should be low in Tri-IPC):")
    for marker in VALIDATION_NEGATIVE:
        expr = get_gene_expression(adata, marker)
        if expr is not None:
            adata.obs[f'{marker}_expr'] = expr
            tri_ipc_expr = expr[adata.obs['Tri_IPC']]
            other_expr = expr[~adata.obs['Tri_IPC']]
            print(f"    {marker}: Tri-IPC mean={tri_ipc_expr.mean():.3f}, Other mean={other_expr.mean():.3f}")
    
    return adata

def characterize_tri_ipc(adata):
    """
    Characterize Tri-IPC cells vs non-Tri-IPC
    """
    print("\n" + "="*80)
    print("CHARACTERIZING TRI-IPC CELLS")
    print("="*80)
    
    if 'Tri_IPC' not in adata.obs.columns:
        print("  ✗ Tri-IPC cells not identified")
        return adata
    
    # Distribution by sample
    print("\n  Distribution by sample:")
    for sample in adata.obs['sample'].unique():
        sample_mask = adata.obs['sample'] == sample
        tri_ipc_in_sample = (sample_mask & adata.obs['Tri_IPC']).sum()
        total_in_sample = sample_mask.sum()
        print(f"    {sample}: {tri_ipc_in_sample}/{total_in_sample} ({tri_ipc_in_sample/total_in_sample*100:.1f}%)")
    
    # Distribution by cluster
    print("\n  Distribution by cluster:")
    for cluster in sorted(adata.obs['leiden'].unique(), key=lambda x: int(x)):
        cluster_mask = adata.obs['leiden'] == cluster
        tri_ipc_in_cluster = (cluster_mask & adata.obs['Tri_IPC']).sum()
        total_in_cluster = cluster_mask.sum()
        pct_tri_ipc = tri_ipc_in_cluster / total_in_cluster * 100 if total_in_cluster > 0 else 0
        print(f"    Cluster {cluster}: {tri_ipc_in_cluster}/{total_in_cluster} ({pct_tri_ipc:.1f}%)")
        if pct_tri_ipc > 20:
            print(f"      ⭐ Enriched!")
    
    return adata

def analyze_genes_in_tri_ipc(adata, gene_list):
    """
    Analyze expression of genes of interest in Tri-IPC vs non-Tri-IPC cells
    """
    print("\n" + "="*80)
    print("ANALYZING GENES OF INTEREST IN TRI-IPC CELLS")
    print("="*80)
    
    results = []
    
    for gene in gene_list:
        print(f"\n  {gene}:")
        expr = get_gene_expression(adata, gene)
        
        if expr is None:
            print(f"    ✗ Not found")
            continue
        
        adata.obs[f'{gene}_expr'] = expr
        
        # Define high expressors
        threshold = np.percentile(expr[expr > 0], 75) if (expr > 0).any() else 0
        adata.obs[f'{gene}_high'] = expr > threshold
        
        # Compare Tri-IPC vs non-Tri-IPC
        tri_ipc_expr = expr[adata.obs['Tri_IPC']]
        non_tri_ipc_expr = expr[~adata.obs['Tri_IPC']]
        
        # Statistics
        stat, pval = mannwhitneyu(tri_ipc_expr, non_tri_ipc_expr, alternative='two-sided')
        fold_change = tri_ipc_expr.mean() / (non_tri_ipc_expr.mean() + 1e-10)
        
        print(f"    Mean expression:")
        print(f"      Tri-IPC: {tri_ipc_expr.mean():.3f} ± {tri_ipc_expr.std():.3f}")
        print(f"      Non-Tri-IPC: {non_tri_ipc_expr.mean():.3f} ± {non_tri_ipc_expr.std():.3f}")
        print(f"    Fold change: {fold_change:.2f}x")
        print(f"    p-value: {pval:.2e} {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")
        
        # Overlap with Tri-IPC
        overlap = (adata.obs['Tri_IPC'] & adata.obs[f'{gene}_high']).sum()
        n_tri_ipc = adata.obs['Tri_IPC'].sum()
        print(f"    Overlap: {overlap}/{n_tri_ipc} Tri-IPC cells are {gene}-high ({overlap/n_tri_ipc*100:.1f}%)")
        
        # Correlation with EGFR and OLIG2
        if 'EGFR_expr' in adata.obs.columns:
            corr, _ = spearmanr(expr, adata.obs['EGFR_expr'])
            print(f"    Correlation with EGFR: r={corr:.3f}")
        
        if 'OLIG2_expr' in adata.obs.columns:
            corr, _ = spearmanr(expr, adata.obs['OLIG2_expr'])
            print(f"    Correlation with OLIG2: r={corr:.3f}")
        
        results.append({
            'gene': gene,
            'tri_ipc_mean': tri_ipc_expr.mean(),
            'non_tri_ipc_mean': non_tri_ipc_expr.mean(),
            'fold_change': fold_change,
            'pvalue': pval,
            'pct_overlap': overlap/n_tri_ipc*100 if n_tri_ipc > 0 else 0
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    
    return adata, results_df

def create_visualizations(adata, gene_list, output_dir):
    """Create comprehensive visualizations"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    sc.pl.umap(adata, color='sample', ax=axes[0], show=False, title='Samples', legend_loc='right margin')
    sc.pl.umap(adata, color='leiden', ax=axes[1], show=False, title='Clusters', legend_loc='on data')
    sc.pl.umap(adata, color='EGFR_expr', cmap='viridis', vmax='p99',
              ax=axes[2], show=False, title='EGFR Expression')
    sc.pl.umap(adata, color='OLIG2_expr', cmap='viridis', vmax='p99',
              ax=axes[3], show=False, title='OLIG2 Expression')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved overview")
    
    # 2. Tri-IPC identification
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    sc.pl.umap(adata, color='EGFR_high', ax=axes[0], show=False,
              title='EGFR-high cells', palette=['lightgray', 'green'])
    sc.pl.umap(adata, color='OLIG2_high', ax=axes[1], show=False,
              title='OLIG2-high cells', palette=['lightgray', 'blue'])
    sc.pl.umap(adata, color='Tri_IPC', ax=axes[2], show=False,
              title='Tri-IPC cells (EGFR+ OLIG2+)', palette=['lightgray', 'red'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tri_ipc_identification.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved Tri-IPC identification")
    
    # 3. Genes of interest
    n_genes = len([g for g in gene_list if f'{g}_expr' in adata.obs.columns])
    if n_genes > 0:
        n_cols = 3
        n_rows = (n_genes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        idx = 0
        for gene in gene_list:
            if f'{gene}_expr' in adata.obs.columns:
                sc.pl.umap(adata, color=f'{gene}_expr', cmap='viridis', vmax='p99',
                          ax=axes[idx], show=False, title=f'{gene} Expression')
                idx += 1
        
        # Hide empty subplots
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'genes_of_interest.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved genes of interest")
    
    # 4. Detailed overlay for each gene
    for gene in gene_list:
        if f'{gene}_expr' not in adata.obs.columns:
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        
        # Gene expression
        sc.pl.umap(adata, color=f'{gene}_expr', cmap='viridis', vmax='p99',
                  ax=axes[0], show=False, title=f'{gene} Expression')
        
        # Tri-IPC cells
        sc.pl.umap(adata, color='Tri_IPC', ax=axes[1], show=False,
                  title='Tri-IPC cells', palette=['lightgray', 'red'])
        
        # Overlay
        umap_coords = adata.obsm['X_umap']
        axes[2].scatter(umap_coords[:, 0], umap_coords[:, 1],
                       c='lightgray', s=5, alpha=0.3, label='Other')
        
        tri_ipc = adata.obs['Tri_IPC']
        axes[2].scatter(umap_coords[tri_ipc, 0], umap_coords[tri_ipc, 1],
                       c='red', s=10, alpha=0.5, label='Tri-IPC')
        
        gene_high = adata.obs[f'{gene}_high']
        overlap = tri_ipc & gene_high
        if overlap.sum() > 0:
            axes[2].scatter(umap_coords[overlap, 0], umap_coords[overlap, 1],
                           c='purple', s=20, alpha=0.9, edgecolors='black',
                           linewidths=0.5, label=f'Tri-IPC + {gene}-high', zorder=10)
        
        axes[2].set_xlabel('UMAP1')
        axes[2].set_ylabel('UMAP2')
        axes[2].set_title(f'Tri-IPC & {gene} Overlap')
        axes[2].legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{gene}_tri_ipc_overlay.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {gene} overlay")
    
    # 5. Summary heatmap
    if len(gene_list) > 0:
        # Create summary matrix
        summary_data = []
        
        for cluster in sorted(adata.obs['leiden'].unique(), key=lambda x: int(x)):
            cluster_mask = adata.obs['leiden'] == cluster
            tri_ipc_mask = cluster_mask & adata.obs['Tri_IPC']
            
            row = {
                'cluster': f'C{cluster}',
                'n_cells': cluster_mask.sum(),
                'pct_tri_ipc': (tri_ipc_mask.sum() / cluster_mask.sum() * 100) if cluster_mask.sum() > 0 else 0
            }
            
            # Add gene expression
            for gene in gene_list:
                if f'{gene}_expr' in adata.obs.columns:
                    # Mean in Tri-IPC cells of this cluster
                    if tri_ipc_mask.sum() > 0:
                        row[f'{gene}_tri_ipc'] = adata.obs.loc[tri_ipc_mask, f'{gene}_expr'].mean()
                    else:
                        row[f'{gene}_tri_ipc'] = 0
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Plot heatmap
        plot_cols = [col for col in summary_df.columns if col not in ['cluster', 'n_cells']]
        if len(plot_cols) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 10))
            
            plot_data = summary_df[plot_cols].set_index(summary_df['cluster'])
            sns.heatmap(plot_data.T, annot=True, fmt='.1f', cmap='RdYlBu_r',
                       cbar_kws={'label': 'Value'}, ax=ax)
            ax.set_xlabel('Cluster')
            ax.set_title('Tri-IPC Characterization by Cluster')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'cluster_summary_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved cluster heatmap")
        
        # Save summary table
        summary_df.to_csv(output_dir / 'cluster_summary.csv', index=False)
    
    # Save data
    adata.write(output_dir / 'tri_ipc_analyzed.h5ad')
    print("  ✓ Saved processed data")

def main():
    """Main workflow"""
    print("="*80)
    print("TRI-IPC (EGFR+ OLIG2+) ANALYSIS")
    print(f"Samples: {', '.join(SAMPLES_TO_ANALYZE)}")
    print("="*80)
    
    BASE_PATH = '/Users/fudhailsayed/prololab'
    OUTPUT_DIR = Path(BASE_PATH) / 'figures' / 'tri_ipc_focused'
    
    # Load and process
    adata = load_samples(BASE_PATH, SAMPLES_TO_ANALYZE)
    adata = process_data(adata)
    
    # Identify Tri-IPC cells
    adata = identify_tri_ipc_cells(adata, EGFR_PERCENTILE, OLIG2_PERCENTILE)
    
    # Characterize Tri-IPC
    adata = characterize_tri_ipc(adata)
    
    # Analyze genes of interest
    adata, results_df = analyze_genes_in_tri_ipc(adata, GENES_OF_INTEREST)
    
    # Save gene analysis results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / 'gene_analysis_summary.csv', index=False)
    print(f"\n✓ Saved gene analysis summary")
    
    # Visualize
    create_visualizations(adata, GENES_OF_INTEREST, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results: {OUTPUT_DIR}")
    print("="*80)
    
    print("\n📊 KEY FINDINGS:")
    print(f"  - Total Tri-IPC cells: {adata.obs['Tri_IPC'].sum()} ({adata.obs['Tri_IPC'].sum()/adata.n_obs*100:.1f}%)")
    
    if not results_df.empty:
        print(f"\n  Gene enrichment in Tri-IPC:")
        for _, row in results_df.iterrows():
            sig = "***" if row['pvalue'] < 0.001 else "**" if row['pvalue'] < 0.01 else "*" if row['pvalue'] < 0.05 else "ns"
            print(f"    {row['gene']}: {row['fold_change']:.2f}x fold change, {row['pct_overlap']:.1f}% overlap {sig}")
    
    return adata

if __name__ == "__main__":
    adata = main()
