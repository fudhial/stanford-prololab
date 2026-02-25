#!/usr/bin/env python3
"""
TF Activity Inference and Gene Correlation Analysis for pHGG scRNA-seq
=======================================================================
This script infers transcription factor (TF) activity at single-cell resolution
for pediatric high-grade glioma (pHGG) scRNA-seq data and identifies which TFs
are most correlated with a gene of interest.

Workflow:
    1. Loads a single pHGG scRNA-seq sample (.h5ad format)
    2. Maps Ensembl gene IDs to gene symbols using mygene.info
    3. Normalizes and log-transforms raw counts if needed
    4. Retrieves DoRothEA TF-gene regulons (confidence levels A/B)
    5. Runs Univariate Linear Model (ULM) to estimate per-cell TF activity
    6. Correlates each TF's activity with the gene of interest across all cells
    7. Outputs a ranked list and publication-quality lollipop plot of the top TFs

Inputs:
    - pHGG scRNA-seq .h5ad files (SCPCS* sample folders)
    - Gene of interest — set in the EDIT THIS SECTION block

Outputs:
    - TF activity matrix (.tsv.gz)
    - Top TF correlations table (.tsv)
    - Lollipop figure (.pdf and .png)

Dependencies:
    scanpy, decoupler, mygene, numpy, pandas, matplotlib

Author: Fudhail Sayed 

"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


# =============================
# *** EDIT THIS SECTION ***
# =============================

# Set to None to use SAMPLE_INDEX instead, or provide a name like "SCPCS000001"
SAMPLE_NAME: str | None = "SCPCS000001"
# If SAMPLE_NAME is None, this 0-based index picks the sample from the sorted list
SAMPLE_INDEX: int = 0

# Gene of interest for TF correlation analysis
# Change this to any gene symbol you care about each run
GENE_OF_INTEREST: str = "MAP4K4"

# How many top correlated TFs to report (and plot)
TOP_N_TFS: int = 20

# =============================
# CONFIG
# =============================
DATASET_ROOT = Path("/Users/fudhailsayed/prololab/data/datasets/pHGG_scRNA_anndata")
SCRNA_DIR = DATASET_ROOT / "scRNA"
H5AD_SUFFIX = "_processed_rna.h5ad"

# Output root — each run gets its own timestamped subfolder
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT = Path("/Users/fudhailsayed/prololab/data/_tf_activity")

# Mapping cache lives at root level (shared across runs)
MAP_CACHE = OUT_ROOT / "ensembl_to_symbol_cache.tsv"

# Memory safety knobs
N_GENES_FOR_ULM = 5000
CHUNK_SIZE = 1500
DENSE_DTYPE = np.float32

# DoRothEA
DOROTHEA_CONF = ["A", "B"]
MIN_N_TRY = [5, 3, 1]


# =============================
# HELPERS
# =============================

def make_run_dir(out_root: Path, sample_id: str, gene: str) -> Path:
    """Create a well-organised output folder for this run."""
    run_name = f"{_TIMESTAMP}__{sample_id}__{gene}"
    run_dir = out_root / run_name
    (run_dir / "tf_activity").mkdir(parents=True, exist_ok=True)
    (run_dir / "correlations").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def load_single_sample_h5ad(scrna_dir: Path, suffix: str,
                              sample_name: str | None, sample_index: int) -> tuple[sc.AnnData, str]:
    """Load exactly one sample, either by name or by sorted index."""
    sample_dirs = sorted([p for p in scrna_dir.glob("SCPCS*") if p.is_dir()])
    if not sample_dirs:
        raise FileNotFoundError(f"No SCPCS* folders found in {scrna_dir}")

    if sample_name is not None:
        matches = [p for p in sample_dirs if p.name == sample_name]
        if not matches:
            available = [p.name for p in sample_dirs]
            raise FileNotFoundError(
                f"Sample '{sample_name}' not found. Available:\n  " + "\n  ".join(available)
            )
        sd = matches[0]
    else:
        if sample_index >= len(sample_dirs):
            raise IndexError(
                f"SAMPLE_INDEX={sample_index} out of range; only {len(sample_dirs)} samples found."
            )
        sd = sample_dirs[sample_index]

    h5ads = sorted(sd.glob(f"*{suffix}"))
    if not h5ads:
        raise FileNotFoundError(f"No *{suffix} found in {sd}")

    h5ad_path = h5ads[0]
    a = sc.read_h5ad(h5ad_path)
    a.obs["sample_id"] = sd.name
    a.obs_names = [f"{sd.name}__{x}" for x in a.obs_names]
    a.var_names = a.var_names.astype(str)
    a.var_names_make_unique()

    print(f"Loaded sample: {sd.name}  shape={a.shape}  file={h5ad_path.name}")
    return a, sd.name


def strip_ensembl_version(ids: pd.Index) -> pd.Index:
    s = pd.Index(ids.astype(str))
    s = s.str.replace(r"\.\d+$", "", regex=True)
    return s


def load_mapping_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    df = pd.read_csv(cache_path, sep="\t")
    if not {"ensembl", "symbol"}.issubset(df.columns):
        return {}
    return dict(zip(df["ensembl"].astype(str), df["symbol"].astype(str)))


def save_mapping_cache(cache_path: Path, mapping: dict[str, str]) -> None:
    df = pd.DataFrame({"ensembl": list(mapping.keys()), "symbol": list(mapping.values())})
    df = df.sort_values("ensembl")
    df.to_csv(cache_path, sep="\t", index=False)


def map_ensembl_to_symbol_mygene(ensembl_ids: list[str], cache_path: Path) -> dict[str, str]:
    cached = load_mapping_cache(cache_path)
    needed = [eid for eid in ensembl_ids if eid not in cached]
    if needed:
        print(f"Mapping {len(needed)} Ensembl IDs via mygene.info (cached: {len(cached)})...")
        try:
            import mygene
        except ImportError:
            raise RuntimeError("Missing dependency: pip install mygene")

        mg = mygene.MyGeneInfo()
        newmap = {}
        batch_size = 1000
        for i in range(0, len(needed), batch_size):
            batch = needed[i:i + batch_size]
            res = mg.querymany(
                batch, scopes="ensembl.gene", fields="symbol",
                species="human", as_dataframe=False, returnall=False, verbose=False,
            )
            for r in res:
                q = str(r.get("query", ""))
                sym = r.get("symbol", None)
                if sym is None or str(sym).lower() in {"nan", "none", ""}:
                    continue
                newmap[q] = str(sym)

        cached.update(newmap)
        save_mapping_cache(cache_path, cached)
        print(f"Mapped {len(newmap)} new IDs. Cache now has {len(cached)} total.")

    return cached


def enforce_gene_symbols_from_ensembl(adata: sc.AnnData, cache_path: Path) -> sc.AnnData:
    print("\n--- Gene ID check ---")
    print("Example var_names:", list(adata.var_names[:5]))

    ensembl = strip_ensembl_version(adata.var_names)
    adata.var["ensembl"] = ensembl.to_numpy()

    unique_ids = pd.Index(adata.var["ensembl"].astype(str)).unique().tolist()
    mapping = map_ensembl_to_symbol_mygene(unique_ids, cache_path)

    sym = adata.var["ensembl"].map(mapping)
    good = sym.notna() & (sym.astype(str) != "")
    print(f"Mapped symbols for {int(good.sum())} / {adata.n_vars} genes")

    if int(good.sum()) < 500:
        raise RuntimeError("Too few genes mapped to symbols. Check internet access / mygene install.")

    adata2 = adata[:, good.to_numpy()].copy()
    adata2.var_names = sym[good].astype(str).to_numpy()
    adata2.var_names_make_unique()
    adata2.var["ensembl"] = adata.var.loc[good, "ensembl"].to_numpy()

    print("Example new var_names (symbols):", list(adata2.var_names[:8]))
    return adata2


def is_probably_logged(adata: sc.AnnData, n_sample: int = 5000) -> bool:
    X = adata.X
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            data = X.data
            if data.size == 0:
                return True
            s = data[np.random.choice(data.size, size=min(n_sample, data.size), replace=False)]
        else:
            flat = np.ravel(X)
            s = flat[np.random.choice(flat.size, size=min(n_sample, flat.size), replace=False)]
    except Exception:
        return True

    s = np.asarray(s)
    if np.nanmax(s) > 50:
        return False
    frac = np.mean(np.abs(s - np.round(s)) > 1e-6)
    return frac > 0.2


def normalize_if_needed(adata: sc.AnnData) -> None:
    if is_probably_logged(adata):
        print("Data looks already normalized/logged -> skipping normalize/log1p.")
        return
    print("Data looks like counts -> running normalize_total + log1p.")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def build_ulm_safe_net(net: pd.DataFrame) -> pd.DataFrame:
    net = net.copy()
    net["source"] = net["source"].astype(str)
    net["target"] = net["target"].astype(str)

    if "weight" in net.columns:
        w = pd.to_numeric(net["weight"], errors="coerce")
    elif ("mor" in net.columns) and ("likelihood" in net.columns):
        w = pd.to_numeric(net["mor"], errors="coerce") * pd.to_numeric(net["likelihood"], errors="coerce")
    elif "mor" in net.columns:
        w = pd.to_numeric(net["mor"], errors="coerce")
    elif "likelihood" in net.columns:
        w = pd.to_numeric(net["likelihood"], errors="coerce")
    else:
        w = pd.Series(1.0, index=net.index)

    net["weight"] = w.astype("float64")
    net = net.dropna(subset=["source", "target", "weight"]).copy()
    net = net[np.isfinite(net["weight"].to_numpy())]
    net = net[net["weight"] != 0]
    return net


def select_top_variable_genes(adata: sc.AnnData, n_top: int) -> sc.AnnData:
    n_top = min(n_top, adata.n_vars)
    print(f"Selecting top {n_top} highly variable genes")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top, flavor="cell_ranger",
        subset=True, inplace=True,
    )
    return adata


def extract_activity_df(ulm_res, cell_index) -> pd.DataFrame:
    ulm_est = ulm_res[0] if isinstance(ulm_res, tuple) else ulm_res
    if hasattr(ulm_est, "to_df"):
        df = ulm_est.to_df()
    elif isinstance(ulm_est, pd.DataFrame):
        df = ulm_est
    else:
        df = pd.DataFrame(ulm_est)

    if df.shape[0] != len(cell_index):
        if df.shape[1] == len(cell_index):
            df = df.T
        else:
            raise RuntimeError(f"Activity shape mismatch: got {df.shape}, expected rows={len(cell_index)}")

    df.index = cell_index
    return df


def run_ulm_chunked(adata: sc.AnnData, net: pd.DataFrame, chunk_size: int, dtype=np.float32) -> pd.DataFrame:
    import scipy.sparse as sp

    n_cells = adata.n_obs
    genes = adata.var_names.astype(str).tolist()
    out = []

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        obs_chunk = adata.obs_names[start:end]
        X = adata.X[start:end]

        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=dtype)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        mat = pd.DataFrame(X, index=obs_chunk, columns=genes)

        ulm_res = None
        last_err = None
        for min_n in MIN_N_TRY:
            try:
                ulm_res = dc.run_ulm(
                    mat=mat, net=net, source="source",
                    target="target", weight="weight", min_n=min_n,
                )
                break
            except Exception as e:
                last_err = e

        if ulm_res is None:
            raise RuntimeError(f"ULM failed for chunk {start}:{end}. Last error: {repr(last_err)}")

        act = extract_activity_df(ulm_res, cell_index=obs_chunk)
        out.append(act)
        print(f"  ULM done for cells {start}:{end} -> {act.shape}")

    return pd.concat(out, axis=0)


def get_gene_expression(adata: sc.AnnData, gene: str) -> pd.Series | None:
    """Extract per-cell expression for a gene from adata (after HVG filtering may have dropped it)."""
    if gene not in adata.var_names:
        return None
    import scipy.sparse as sp
    idx = list(adata.var_names).index(gene)
    X = adata.X[:, idx]
    if sp.issparse(X):
        X = X.toarray().ravel()
    return pd.Series(np.asarray(X, dtype=float), index=adata.obs_names, name=gene)


def compute_tf_gene_correlations(
    tf_activity: pd.DataFrame,
    gene_expr: pd.Series,
    top_n: int,
) -> pd.DataFrame:
    """
    Pearson correlation between each TF's activity and the gene's expression
    across cells. Returns a sorted DataFrame with columns:
        TF | pearson_r | abs_r
    """
    shared_cells = tf_activity.index.intersection(gene_expr.index)
    tf_sub = tf_activity.loc[shared_cells]
    g_sub = gene_expr.loc[shared_cells].values

    corrs = {}
    for tf in tf_sub.columns:
        tf_vals = tf_sub[tf].values
        # skip constant columns
        if np.std(tf_vals) < 1e-10 or np.std(g_sub) < 1e-10:
            continue
        r = np.corrcoef(tf_vals, g_sub)[0, 1]
        corrs[tf] = r

    df = (
        pd.DataFrame.from_dict(corrs, orient="index", columns=["pearson_r"])
        .assign(abs_r=lambda x: x["pearson_r"].abs())
        .sort_values("abs_r", ascending=False)
        .reset_index()
        .rename(columns={"index": "TF"})
    )
    return df.head(top_n)


def plot_tf_correlations(
    top_tfs: pd.DataFrame,
    gene: str,
    sample_id: str,
    out_path: Path,
    top_n: int = 20,
) -> None:
    """
    Publication-quality lollipop chart of TF–gene Pearson correlations.

    Visual encoding:
      • Bar length  = signed Pearson r  (negative = left, positive = right)
      • Dot size    = |r|  (larger dot = stronger absolute correlation)
      • Dot colour  = signed r  (diverging RdBu_r palette; blue=positive, red=negative)
      • Thin stem   = direction guide

    Saved as 300 dpi PDF + PNG (vector + raster) for journal submission.
    """
    df = top_tfs.head(top_n).copy()
    # Sort by signed r so positive and negative groups cluster naturally
    df = df.sort_values("pearson_r", ascending=True).reset_index(drop=True)

    n = len(df)
    fig_h = max(4.5, 0.38 * n + 1.5)
    fig, ax = plt.subplots(figsize=(5.5, fig_h))

    # ── Colour map: diverging, centred at 0 ──────────────────────────────────
    cmap = plt.get_cmap("RdBu_r")
    r_max = max(0.01, df["abs_r"].max())
    norm = mcolors.TwoSlopeNorm(vmin=-r_max, vcenter=0, vmax=r_max)

    # ── Grid & zero line ─────────────────────────────────────────────────────
    ax.set_facecolor("#F7F7F7")
    fig.patch.set_facecolor("white")
    ax.axvline(0, color="#333333", linewidth=0.9, zorder=2)
    ax.xaxis.grid(True, color="white", linewidth=1.2, zorder=1)
    ax.set_axisbelow(True)

    # ── Lollipops ────────────────────────────────────────────────────────────
    dot_scale = 320   # base area for abs_r = 1.0
    for i, row in df.iterrows():
        r   = row["pearson_r"]
        absr = row["abs_r"]
        col = cmap(norm(r))

        # Stem
        ax.plot([0, r], [i, i],
                color=col, linewidth=1.4, solid_capstyle="round", zorder=3, alpha=0.75)

        # Dot — area ∝ abs_r
        ax.scatter(r, i,
                   s=dot_scale * absr,
                   color=col,
                   edgecolors="#333333",
                   linewidths=0.5,
                   zorder=4)

        # Inline r-value label
        x_offset = 0.012 if r >= 0 else -0.012
        ha = "left" if r >= 0 else "right"
        ax.text(r + x_offset, i, f"{r:+.3f}",
                va="center", ha=ha,
                fontsize=7.5, color="#222222",
                fontfamily="DejaVu Sans")

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["TF"], fontsize=9, fontfamily="DejaVu Sans", style="italic")
    ax.set_xlabel("Pearson  r  (TF activity vs. gene expression)",
                  fontsize=9, labelpad=6, fontfamily="DejaVu Sans")

    # x-axis: symmetric, a little breathing room
    xlim = r_max * 1.30
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-0.7, n - 0.3)

    # Tick formatting
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AAAAAA")

    # ── Title block ──────────────────────────────────────────────────────────
    ax.set_title(
        f"Top {n} TFs correlated with $\\it{{{gene}}}$ expression\n"
        f"Sample: {sample_id}  ·  DoRothEA (A/B)  ·  ULM",
        fontsize=10, fontweight="bold", pad=10,
        fontfamily="DejaVu Sans", loc="left",
    )

    # ── Colorbar ─────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.025, pad=0.02, shrink=0.55)
    cbar.set_label("Pearson  r", fontsize=8, fontfamily="DejaVu Sans")
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)

    # ── Dot-size legend ──────────────────────────────────────────────────────
    legend_vals = [0.2, 0.5, 1.0]
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#888888", markeredgecolor="#333333",
               markeredgewidth=0.5,
               markersize=np.sqrt(dot_scale * v) / np.sqrt(plt.rcParams["figure.dpi"]) * 2.8,
               label=f"|r| = {v:.1f}")
        for v in legend_vals
        if v <= r_max + 0.05
    ]
    if legend_handles:
        leg = ax.legend(handles=legend_handles, title="|r| (dot size)",
                        title_fontsize=7.5, fontsize=7.5,
                        loc="lower right", frameon=True,
                        framealpha=0.85, edgecolor="#CCCCCC",
                        handletextpad=0.4, labelspacing=0.5)
        leg._legend_box.align = "left"

    fig.tight_layout(rect=[0, 0, 1, 1])

    # ── Save ─────────────────────────────────────────────────────────────────
    pdf_path = out_path.with_suffix(".pdf")
    png_path = out_path.with_suffix(".png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")


# =============================
# MAIN
# =============================

def main():
    print("=== pHGG scRNA ULM TF activity (single sample) ===")

    # Load single sample
    adata, sample_id = load_single_sample_h5ad(
        SCRNA_DIR, H5AD_SUFFIX, SAMPLE_NAME, SAMPLE_INDEX
    )
    print(f"AnnData shape: {adata.shape}")

    # Build output directory for this run
    run_dir = make_run_dir(OUT_ROOT, sample_id, GENE_OF_INTEREST)
    print(f"Output directory: {run_dir}\n")

    # Write a simple run manifest
    manifest_lines = [
        f"sample_id       : {sample_id}",
        f"gene_of_interest: {GENE_OF_INTEREST}",
        f"top_n_tfs       : {TOP_N_TFS}",
        f"dorothea_conf   : {DOROTHEA_CONF}",
        f"n_hvg           : {N_GENES_FOR_ULM}",
        f"timestamp       : {_TIMESTAMP}",
    ]
    (run_dir / "logs" / "run_manifest.txt").write_text("\n".join(manifest_lines) + "\n")

    # Convert Ensembl -> Symbol
    adata = enforce_gene_symbols_from_ensembl(adata, MAP_CACHE)

    # Normalize if needed
    normalize_if_needed(adata)

    # Grab gene expression BEFORE HVG filtering drops it
    gene_expr = get_gene_expression(adata, GENE_OF_INTEREST)
    if gene_expr is None:
        print(f"WARNING: {GENE_OF_INTEREST} not found in var_names after symbol mapping.")
    else:
        print(f"{GENE_OF_INTEREST} expression range: "
              f"[{gene_expr.min():.3f}, {gene_expr.max():.3f}]")

    # DoRothEA
    print("\nLoading DoRothEA regulons (human)...")
    net = dc.get_dorothea(organism="human")
    if "confidence" in net.columns:
        net = net[net["confidence"].isin(DOROTHEA_CONF)].copy()
    net = build_ulm_safe_net(net)

    dor_targets = pd.Index(net["target"].unique().astype(str))
    shared = adata.var_names.astype(str).intersection(dor_targets)
    print(f"Shared genes with DoRothEA: {len(shared)} / {adata.n_vars}")

    if len(shared) < 200:
        raise RuntimeError("Target overlap too small. Check gene mapping.")

    adata = adata[:, shared].copy()
    adata = select_top_variable_genes(adata, n_top=N_GENES_FOR_ULM)
    print(f"After HVG selection: {adata.shape}")

    # Run ULM
    print(f"\nRunning ULM (CHUNK_SIZE={CHUNK_SIZE})...")
    tf_activity = run_ulm_chunked(adata, net=net, chunk_size=CHUNK_SIZE, dtype=DENSE_DTYPE)
    print(f"TF activity matrix: {tf_activity.shape}")

    # ---- Save TF activity ----
    act_path = run_dir / "tf_activity" / f"{sample_id}_tf_activity_ulm.tsv.gz"
    tf_activity.to_csv(act_path, sep="\t", compression="gzip")
    print(f"Saved TF activity: {act_path}")

    # ---- TF correlation with gene of interest ----
    print(f"\n--- Top {TOP_N_TFS} TFs correlated with {GENE_OF_INTEREST} ---")

    if gene_expr is None:
        print(f"Skipping correlation: {GENE_OF_INTEREST} was not found in data.")
    else:
        # Re-align gene_expr to cells that made it through HVG (same obs_names)
        common_cells = tf_activity.index.intersection(gene_expr.index)
        gene_expr_aligned = gene_expr.loc[common_cells]

        top_tfs = compute_tf_gene_correlations(tf_activity, gene_expr_aligned, top_n=TOP_N_TFS)

        # Pretty print
        print(top_tfs.to_string(index=False))

        # Save TSV
        corr_path = run_dir / "correlations" / f"top{TOP_N_TFS}_TFs_corr_{GENE_OF_INTEREST}.tsv"
        top_tfs.to_csv(corr_path, sep="\t", index=False)
        print(f"\nSaved TF correlations: {corr_path}")

        # Publication figure
        fig_path = run_dir / "figures" / f"top{TOP_N_TFS}_TFs_corr_{GENE_OF_INTEREST}"
        plot_tf_correlations(
            top_tfs=top_tfs,
            gene=GENE_OF_INTEREST,
            sample_id=sample_id,
            out_path=fig_path,
            top_n=TOP_N_TFS,
        )

    # ---- Sanity check ----
    for g in ["ST8SIA2", "ST8SIA4"]:
        print(f"{g} present in adata.var_names? -> {g in set(adata.var_names)}")

    print(f"\nDone. All outputs in: {run_dir}")


if __name__ == "__main__":
    main()
