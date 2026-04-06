#!/usr/bin/env python3
"""
Neftal Cellular State Scoring and Visualization for pHGG scRNA-seq
===================================================================
This script scores and visualizes tumor cell states in pediatric high-grade
glioma (pHGG) scRNA-seq data using the neftal et al. 2019 framework —
classifying cells into four meta-module states: MES, AC, OPC, and NPC.

Workflow:
    1. Loads all pHGG scRNA-seq samples (SCPCS* folders) and pools them
    2. Scores each cell for six neftal gene programs (MES1/MES2, AC, OPC, NPC1/NPC2)
       plus optional cell cycle programs (G1S, G2M)
    3. Builds relative meta-module axes:
           x-axis: NPC vs OPC (progenitor subtype)
           y-axis: (OPC+NPC) vs (AC+MES) (stemness vs differentiation)
    4. Assigns each cell to a quadrant (OPC-like, NPC-like, AC-like, MES-like)
    5. Optionally filters to malignant cells only using CNV scores
    6. Bins gene expression and plots quadrant composition per bin
    7. Generates publication-quality 2D state plots with quadrant summaries

Inputs:
    - pHGG scRNA-seq .h5ad files (SCPCS* sample folders)
    - Optional: CNV column in adata.obs for malignant cell filtering
    - Gene of interest for expression binning (set via --color_gene flag)

Outputs:
    - Neftal 2D scatter plots with stacked quadrant bar (.png, 300 dpi)
    - Quadrant summary CSVs (all cells and malignant)
    - Expression bin stacked bar plots per gene
    - Run report CSV (gene coverage per program)

Usage:
    python cellular_state.py --color_gene ST8SIA4 --do_malignant_filter

Dependencies:
    scanpy, numpy, pandas, matplotlib, scipy

Author: Fudhail Sayed
"""

import argparse
import glob
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# =========================
#  DEFAULT SETTINGS
# =========================

DEFAULT_BASE_DIR = "/Users/fudhailsayed/prololab/datasets/pHGG_scRNA_anndata/scRNA"
DEFAULT_OUT_DIR = (
    "/Users/fudhailsayed/prololab/results/pHGG_scRNA/UMAPv2/cellular_state"
)

# Which file to load inside each SCPCS folder:
DEFAULT_FILE_GLOB = "SCPCL*_processed_rna.h5ad"

# Colormap like Single Cell Portal: low light → high dark red
DEFAULT_CMAP = "YlOrRd"

# Clip expression values for coloring (percentiles)
DEFAULT_COLOR_CLIP_PCT = (1, 99)

# Default gene to color by (None disables gene-coloring)
DEFAULT_COLOR_GENE = "MAP4K4"

# Malignant calling method (expects infercnv_total_cnv in adata.obs)
DEFAULT_CNV_COL = "infercnv_total_cnv"

# Center/scale axes? (strongly recommended when pooling multiple samples)
DEFAULT_STANDARDIZE_AXES = True

# Counts layer detection priority (we will try these keys)
COUNTS_LAYER_CANDIDATES = [
    "counts",
    "raw_counts",
    "rna_counts",
    "count",
    "Count",
    "Counts",
]

# If adata.var has gene symbols here, we will use it as var_names for scoring
GENE_SYMBOL_COL_CANDIDATES = ["gene_symbol", "symbol", "gene", "feature_name"]


# =========================
#  GENE SETS (FULL LISTS)
# =========================

GENESETS = {
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
        "MGST1",
        "RCAN1",
        "TAGLN2",
        "NPC2",
        "HEY1",
        "SERPING1",
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
        "PPP1R15A",
        "VIM",
        "PLOD2",
        "GBE1",
        "SLC2A3",
        "FTL",
        "WARS",
        "ERO1L",
        "XBP1",
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
    "AC": [
        "CST3",
        "S100B",
        "SLC1A3",
        "HEPN1",
        "HOPX",
        "MT3",
        "SPARCL1",
        "MLC1",
        "GFAP",
        "FABP7",
        "BCAN",
        "PON2",
        "METTL7B",
        "SPARC",
        "GATM",
        "RAMP1",
        "PMP2",
        "AQP4",
        "DBI",
        "EDNRB",
        "CLU",
        "PMP22",
        "ATP1A2",
        "S100A16",
        "CA10",
        "PCDHGC3",
        "TTYH1",
        "NDRG2",
        "PRCP",
        "ATP1B2",
        "AGT",
        "PLTP",
        "GPM6B",
        "F3",
        "RAB31",
        "PPAP2B",
        "ANXA5",
        "TSPAN7",
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
        "THY1",
        "PHYHIPL",
        "SOX2-OT",
        "NKAIN4",
        "LPPR1",
        "PTPRZ1",
        "VCAN",
        "DBI",
        "PMP2",
        "CNP",
        "TNS3",
        "LIMA1",
        "PCDHGC3",
        "CNTN1",
        "SCD5",
        "P2RX7",
        "CADM2",
        "TTYH1",
        "FGF12",
        "TMEM206",
        "NEU4",
        "FXYD6",
        "RNF13",
        "RTKN",
        "GPM6B",
        "LMF1",
        "ALCAM",
        "PGRMC1",
        "HRASLS",
        "BCAS1",
        "RAB31",
        "PLLP",
        "FABP5",
        "NLGN3",
        "SERINC5",
        "EPB41L2",
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
        "FXYD6",
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
        "DCX",
        "SOX11",
        "ELAVL4",
        "DLX6-AS1",
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
        "RBFOX2",
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
    "G1S": [
        "RRM2",
        "PCNA",
        "KIAA0101",
        "HIST1H4C",
        "MLF1IP",
        "GMNN",
        "RNASEH2A",
        "MELK",
        "CENPK",
        "TK1",
        "TMEM106C",
        "CDCA5",
        "CKS1B",
        "CDC45",
        "MCM3",
        "CENPM",
        "AURKB",
        "PKMYT1",
        "MCM4",
        "ASF1B",
        "GINS2",
        "MCM2",
        "FEN1",
        "RRM1",
        "DUT",
        "RAD51AP1",
        "MCM7",
        "CCNE2",
        "ZWINT",
    ],
    "G2M": [
        "CCNB1",
        "CDC20",
        "CCNB2",
        "PLK1",
        "CCNA2",
        "CKAP2",
        "KNSTRN",
        "RACGAP1",
        "CDCA3",
        "TROAP",
        "KIF2C",
        "AURKA",
        "CENPF",
        "KPNA2",
        "KIF20A",
        "ECT2",
        "BUB1",
        "CDCA8",
        "BUB1B",
        "TACC3",
        "TTK",
        "TUBA1C",
        "NCAPD2",
        "ARL6IP1",
        "KIF4A",
        "CKAP2L",
        "MZT1",
        "KIFC1",
        "SPAG5",
        "ANP32E",
        "KIF11",
        "PSRC1",
        "TUBB4B",
        "SMC4",
        "MXD3",
        "CDC25B",
        "OPIP5",
        "REEP4",
        "FOXM1",
        "TMPO",
        "GPSM2",
        "HMGB3",
        "ARHGAP11A",
        "RANGAP1",
        "H2AFZ",
    ],
}


# =========================
# Helpers
# =========================


def mkdirp(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def to_dense_1d(x) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x)
    return x.reshape(-1)


def guess_gene_symbol_col(adata):
    for c in GENE_SYMBOL_COL_CANDIDATES:
        if c in adata.var.columns:
            return c
    return None


def ensure_var_names_gene_symbol(adata):
    col = guess_gene_symbol_col(adata)
    if col is None:
        return adata, None
    adata.var_names = adata.var[col].astype(str).values
    adata.var_names_make_unique()
    return adata, col


def find_counts_layer(adata):
    for k in COUNTS_LAYER_CANDIDATES:
        if k in adata.layers:
            return k
    return None


def ensure_neftal_log1p_layer(adata, prefer_raw: bool = False):
    if "neftal_log1p" in adata.layers:
        return "neftal_log1p"

    def _minmax(X):
        if sp.issparse(X):
            return float(X.min()), float(X.max())
        Xd = np.asarray(X)
        return float(np.min(Xd)), float(np.max(Xd))

    if prefer_raw and adata.raw is not None and adata.raw.X is not None:
        Xr = adata.raw.X
        mn, mx = _minmax(Xr)
        if mn >= 0:
            if mx > 50:
                if sp.issparse(Xr):
                    Xtmp = Xr.copy()
                    Xtmp.data = np.log1p(Xtmp.data)
                    adata.layers["neftal_log1p"] = Xtmp
                else:
                    adata.layers["neftal_log1p"] = np.log1p(Xr)
                print(
                    "[INFO] prefer_raw: using adata.raw.X (counts-like) → created neftal_log1p"
                )
            else:
                adata.layers["neftal_log1p"] = Xr
                print(
                    "[INFO] prefer_raw: using adata.raw.X as neftal_log1p (already log-like)"
                )
            return "neftal_log1p"

    counts_key = find_counts_layer(adata)
    if counts_key is not None:
        Xc = adata.layers[counts_key]
        if sp.issparse(Xc):
            Xc2 = Xc.copy()
            Xc2.data = np.log1p(Xc2.data)
            adata.layers["neftal_log1p"] = Xc2
        else:
            adata.layers["neftal_log1p"] = np.log1p(Xc)
        print(f"[INFO] Using counts layer '{counts_key}' → created neftal_log1p")
        return "neftal_log1p"

    if adata.raw is not None and adata.raw.X is not None:
        Xr = adata.raw.X
        mn, mx = _minmax(Xr)
        if mn >= 0 and mx > 50:
            if sp.issparse(Xr):
                Xtmp = Xr.copy()
                Xtmp.data = np.log1p(Xtmp.data)
                adata.layers["neftal_log1p"] = Xtmp
            else:
                adata.layers["neftal_log1p"] = np.log1p(Xr)
            print("[INFO] Using adata.raw.X (counts-like) → created neftal_log1p")
        else:
            adata.layers["neftal_log1p"] = Xr
            print("[INFO] Using adata.raw.X as neftal_log1p (already log-like)")
        return "neftal_log1p"

    warnings.warn("No counts/raw found; using adata.X as neftal_log1p (may be scaled)")
    adata.layers["neftal_log1p"] = adata.X
    return "neftal_log1p"


def score_geneset(adata, genes, layer_key):
    present = [g for g in genes if g in adata.var_names]
    n_present = len(present)
    n_total = len(genes)

    if n_present == 0:
        return np.zeros(adata.n_obs, dtype=float), 0, n_total

    X = adata.layers[layer_key] if layer_key is not None else adata.X
    idx = np.flatnonzero(np.isin(adata.var_names, present))
    Xsub = X[:, idx]

    if sp.issparse(Xsub):
        scores = np.asarray(Xsub.mean(axis=1)).reshape(-1)
    else:
        scores = np.mean(np.asarray(Xsub), axis=1)

    return scores.astype(float), n_present, n_total


def signed_log2_diff(a, b):
    d = a - b
    return np.sign(d) * np.log2(np.abs(d) + 1.0)


def standardize(v):
    v = np.asarray(v, dtype=float)
    sd = np.nanstd(v)
    if sd == 0 or np.isnan(sd):
        return v
    return (v - np.nanmean(v)) / sd


def clip_for_color(vals, pct=(1, 99)):
    lo, hi = np.percentile(vals, list(pct))
    return np.clip(vals, lo, hi), lo, hi


def malignant_mask_from_cnv(
    adata, cnv_col: str, cnv_thr: Optional[float], cnv_quantile: float
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if cnv_col not in adata.obs.columns:
        return None, None
    cnv = pd.to_numeric(adata.obs[cnv_col], errors="coerce").values
    thr = (
        float(cnv_thr)
        if cnv_thr is not None
        else float(np.nanpercentile(cnv, cnv_quantile))
    )
    mask = cnv >= thr
    return mask, thr


def find_samples(base_dir):
    folders = sorted(
        [p for p in glob.glob(os.path.join(base_dir, "SCPCS*")) if os.path.isdir(p)]
    )
    if not folders:
        raise RuntimeError(f"No SCPCS* folders found under {base_dir}")
    return folders


def load_one_sample(
    scpcs_folder,
    file_glob,
    prefer_raw,
    standardize_axes,
    cnv_col,
    cnv_thr,
    cnv_quantile,
):
    sample_id = os.path.basename(scpcs_folder)
    candidates = sorted(glob.glob(os.path.join(scpcs_folder, file_glob)))
    if not candidates:
        raise RuntimeError(f"No file matching {file_glob} in {scpcs_folder}")
    path = candidates[0]
    print(f"[INFO] Loading {sample_id}: {os.path.basename(path)}")
    ad = sc.read_h5ad(path)

    col = guess_gene_symbol_col(ad)
    if col is not None:
        ad, used = ensure_var_names_gene_symbol(ad)
        print(f"[INFO] {sample_id}: var_names from: {used}")
    else:
        print(
            f"[WARN] {sample_id}: No gene symbol column found in ad.var; using existing var_names"
        )

    ad.obs_names_make_unique()
    ad.obs["sample_id"] = sample_id

    layer_key = ensure_neftal_log1p_layer(ad, prefer_raw=prefer_raw)

    report_rows = []
    for key in ["MES1", "MES2", "AC", "OPC", "NPC1", "NPC2", "G1S", "G2M"]:
        scores, n_present, n_total = score_geneset(ad, GENESETS[key], layer_key)
        ad.obs[f"score_{key}"] = scores
        if n_present == 0:
            print(
                f"[WARN] {sample_id}: score_{key}: 0/{n_total} genes found (score set to zeros)"
            )
        report_rows.append(
            {
                "sample_id": sample_id,
                "program": key,
                "n_present": n_present,
                "n_total": n_total,
            }
        )

    ad.obs["score_MES"] = 0.5 * (
        ad.obs["score_MES1"].values + ad.obs["score_MES2"].values
    )
    ad.obs["score_NPC"] = 0.5 * (
        ad.obs["score_NPC1"].values + ad.obs["score_NPC2"].values
    )

    x_rel = signed_log2_diff(ad.obs["score_NPC"].values, ad.obs["score_OPC"].values)
    prog_like = ad.obs["score_OPC"].values + ad.obs["score_NPC"].values
    diff_like = ad.obs["score_AC"].values + ad.obs["score_MES"].values
    y_rel = signed_log2_diff(prog_like, diff_like)

    if standardize_axes:
        x_rel = standardize(x_rel)
        y_rel = standardize(y_rel)

    ad.obs["neftal_x"] = x_rel
    ad.obs["neftal_y"] = y_rel

    mask, thr = malignant_mask_from_cnv(
        ad, cnv_col=cnv_col, cnv_thr=cnv_thr, cnv_quantile=cnv_quantile
    )
    if mask is not None:
        kept = int(np.nansum(mask))
        tot = ad.n_obs
        print(
            f"[INFO] {sample_id}: malignant(CNV) using {cnv_col}, thr={thr:.2f}, kept {kept}/{tot}"
        )
        ad.obs["is_malignant_cnv"] = mask
        ad.uns["cnv_thr"] = thr
    else:
        ad.obs["is_malignant_cnv"] = False
        print(
            f"[WARN] {sample_id}: CNV column '{cnv_col}' not found → malignant filtering unavailable"
        )

    return ad, pd.DataFrame(report_rows)


def get_gene_vector(adata, gene, layer_key):
    if gene is None:
        return None
    if gene not in adata.var_names:
        print(f"[WARN] Gene '{gene}' not found in var_names.")
        return None
    X = (
        adata.layers[layer_key]
        if (layer_key is not None and layer_key in adata.layers)
        else adata.X
    )
    j = int(np.where(adata.var_names == gene)[0][0])
    v = X[:, j]
    return to_dense_1d(v).astype(float)


# =========================
# Quadrant Quantification
# =========================


def assign_quadrants(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    x<0, y>=0  -> OPC-like (top-left)
    x>=0,y>=0  -> NPC-like (top-right)
    x<0, y<0   -> AC-like  (bottom-left)
    x>=0,y<0   -> MES-like (bottom-right)
    Edge: x==0 => right, y==0 => top
    """
    q = np.empty(len(x), dtype=object)
    left = x < 0
    top = y >= 0
    q[np.where(left & top)] = "OPC-like"
    q[np.where((~left) & top)] = "NPC-like"
    q[np.where(left & (~top))] = "AC-like"
    q[np.where((~left) & (~top))] = "MES-like"
    return q


def quadrant_summary_df(adata, label: str) -> pd.DataFrame:
    x = adata.obs["neftal_x"].values
    y = adata.obs["neftal_y"].values
    quads = assign_quadrants(x, y)
    adata.obs["neftal_quadrant"] = quads

    order = ["OPC-like", "MES-like", "NPC-like", "AC-like"]
    counts = pd.Series(quads).value_counts().reindex(order).fillna(0).astype(int)
    total = int(counts.sum())
    pct = (counts / total * 100.0) if total > 0 else counts.astype(float)

    return pd.DataFrame(
        {
            "group": label,
            "quadrant": counts.index,
            "n_cells": counts.values,
            "pct_cells": pct.values,
        }
    )


def plot_left_stacked_bar(ax, summary: pd.DataFrame, title: str):
    order = ["OPC-like", "MES-like", "NPC-like", "AC-like"]
    sub = summary.set_index("quadrant").reindex(order)
    fracs = (sub["pct_cells"].values / 100.0).astype(float)

    bottom = 0.0
    for quad, frac in zip(order, fracs):
        ax.bar(0, frac, bottom=bottom, width=0.8, edgecolor="black", linewidth=0.8)
        if frac >= 0.06:
            ax.text(
                0,
                bottom + frac / 2.0,
                f"{quad}\n{frac*100:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
            )
        bottom += frac

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.9, 0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)


# =========================
# Old feature: "high expressing" summary
# =========================


def high_expr_summary_df(
    adata,
    gene: str,
    layer_key: Optional[str],
    mode: str = "pct",
    pct: float = 90.0,
    thr: float = 1.0,
    label: str = "ALL",
) -> pd.DataFrame:
    v = get_gene_vector(adata, gene, layer_key)
    if v is None:
        return pd.DataFrame(
            [
                {
                    "group": label,
                    "gene": gene,
                    "mode": mode,
                    "threshold_log1p": np.nan,
                    "n_cells": int(adata.n_obs),
                    "n_high": 0,
                    "pct_high": 0.0,
                }
            ]
        )

    v = np.log1p(v.astype(float))

    if mode == "nonzero":
        thr_used = 0.0
        high = v > 0
    elif mode == "abs":
        thr_used = float(thr)
        high = v >= thr_used
    else:  # pct
        thr_used = float(np.nanpercentile(v, pct))
        high = v >= thr_used

    n = int(adata.n_obs)
    n_high = int(np.nansum(high))
    pct_high = (n_high / n * 100.0) if n > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "group": label,
                "gene": gene,
                "mode": mode,
                "threshold_log1p": thr_used,
                "n_cells": n,
                "n_high": n_high,
                "pct_high": pct_high,
            }
        ]
    )


# =========================
# NEW FEATURE: expression bins + stacked bars by quadrant
# =========================


def _parse_edges(edges_str: str) -> List[float]:
    parts = [p.strip() for p in edges_str.split(",") if p.strip() != ""]
    edges = [float(p) for p in parts]
    if len(edges) < 2:
        raise ValueError("Need at least 2 edges for binning.")
    if any(np.diff(edges) <= 0):
        raise ValueError("Edges must be strictly increasing.")
    return edges


def compute_expr_bins(
    v_log1p: np.ndarray,
    n_bins: int,
    mode: str,
    clip_pct: Tuple[float, float],
    edges_str: Optional[str] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    Returns:
      bin_ids: int array in [0..n_bins-1], or -1 if NA
      edges: list of bin edges length n_bins+1
    """
    v = np.asarray(v_log1p, dtype=float)
    ok = np.isfinite(v)

    if np.sum(ok) == 0:
        return np.full(len(v), -1, dtype=int), []

    if mode == "edges":
        if edges_str is None:
            raise ValueError("mode=edges requires --bin_edges like '0,0.4,0.8,1.2,1.6'")
        edges = _parse_edges(edges_str)
        n_bins_eff = len(edges) - 1
        # allow user edges to override n_bins
        n_bins = n_bins_eff

    elif mode == "quantile":
        qs = np.linspace(0, 100, n_bins + 1)
        edges = [float(np.nanpercentile(v[ok], q)) for q in qs]
        # enforce monotonicity (rare tie issue)
        edges = list(np.maximum.accumulate(edges))

    elif mode == "equalwidth":
        lo = float(np.nanmin(v[ok]))
        hi = float(np.nanmax(v[ok]))
        if hi == lo:
            hi = lo + 1e-6
        edges = list(np.linspace(lo, hi, n_bins + 1))

    else:  # equalwidth_clip (DEFAULT)
        lo = float(np.nanpercentile(v[ok], clip_pct[0]))
        hi = float(np.nanpercentile(v[ok], clip_pct[1]))
        if hi == lo:
            hi = lo + 1e-6
        edges = list(np.linspace(lo, hi, n_bins + 1))

    # digitize: bins are [edge_i, edge_{i+1}) except last includes right edge
    bin_ids = np.full(len(v), -1, dtype=int)
    x = v.copy()
    x[~ok] = np.nan

    # np.digitize returns 1..len(edges)-1
    b = np.digitize(x[ok], edges, right=False) - 1
    # handle values < edges[0] or >= edges[-1]
    b = np.clip(b, 0, len(edges) - 2)

    bin_ids[ok] = b.astype(int)
    return bin_ids, edges


def exprbin_quadrant_table(
    adata,
    gene: str,
    layer_key: Optional[str],
    group_label: str,
    bin_mode: str,
    n_bins: int,
    clip_pct: Tuple[float, float],
    bin_edges: Optional[str],
    y_mode: str = "counts",
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Builds a long table with counts (and percent-of-bin) for each (bin, quadrant).
    """
    if "neftal_quadrant" not in adata.obs.columns:
        adata.obs["neftal_quadrant"] = assign_quadrants(
            adata.obs["neftal_x"].values, adata.obs["neftal_y"].values
        )

    v = get_gene_vector(adata, gene, layer_key)
    if v is None:
        return pd.DataFrame(), []

    v_log1p = np.log1p(v.astype(float))
    bin_ids, edges = compute_expr_bins(
        v_log1p=v_log1p,
        n_bins=n_bins,
        mode=bin_mode,
        clip_pct=clip_pct,
        edges_str=bin_edges,
    )
    if len(edges) < 2:
        return pd.DataFrame(), []

    # label bins like "0.00–0.40"
    bin_labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]

    quad_order = [
        "MES-like",
        "OPC-like",
        "NPC-like",
        "AC-like",
    ]  # bottom-to-top-ish; stable order
    # We'll use the same quad names as elsewhere:
    quad_order = ["OPC-like", "NPC-like", "AC-like", "MES-like"]

    df = pd.DataFrame(
        {"bin_id": bin_ids, "quadrant": adata.obs["neftal_quadrant"].values}
    )
    df = df[df["bin_id"] >= 0].copy()
    if df.empty:
        return pd.DataFrame(), edges

    # counts per (bin, quadrant)
    ct = df.groupby(["bin_id", "quadrant"]).size().reset_index(name="n_cells")

    # complete all combos
    all_rows = []
    for b in range(len(edges) - 1):
        for q in quad_order:
            all_rows.append((b, q))
    full = pd.DataFrame(all_rows, columns=["bin_id", "quadrant"])
    ct = full.merge(ct, on=["bin_id", "quadrant"], how="left").fillna({"n_cells": 0})
    ct["n_cells"] = ct["n_cells"].astype(int)

    # bin totals + pct within bin
    bin_totals = ct.groupby("bin_id")["n_cells"].sum().rename("bin_total")
    ct = ct.merge(bin_totals, on="bin_id", how="left")
    ct["pct_within_bin"] = np.where(
        ct["bin_total"] > 0, ct["n_cells"] / ct["bin_total"] * 100.0, 0.0
    )

    ct["group"] = group_label
    ct["gene"] = gene
    ct["bin_label"] = ct["bin_id"].map(
        {i: bin_labels[i] for i in range(len(bin_labels))}
    )
    ct["bin_left"] = ct["bin_id"].map({i: edges[i] for i in range(len(bin_labels))})
    ct["bin_right"] = ct["bin_id"].map(
        {i: edges[i + 1] for i in range(len(bin_labels))}
    )
    ct["bin_mode"] = bin_mode
    ct["n_bins"] = len(bin_labels)
    ct["y_mode"] = y_mode

    # add overall counts too
    ct["n_cells_total_group"] = int(df.shape[0])
    return ct, edges


def plot_exprbin_quadrant_stackedbars(
    table: pd.DataFrame, out_png: str, title: str, y_mode: str = "counts"
):
    """
    Plot: x = expression bins, y = counts or percent-within-bin
    Bars are stacked by quadrant. When y_mode='pct', each bar is normalized
    to 100% and annotated with total N per bin.
    """
    if table is None or table.empty:
        print("[WARN] Empty expr-bin table; skipping stacked bar plot.")
        return

    quad_order = ["OPC-like", "NPC-like", "AC-like", "MES-like"]
    bin_order = table[["bin_id", "bin_label"]].drop_duplicates().sort_values("bin_id")
    bins = bin_order["bin_label"].tolist()

    # Build wide matrix: rows=bins, cols=quadrants
    mat = []
    for b in bin_order["bin_id"].tolist():
        row = []
        for q in quad_order:
            sub = table[(table["bin_id"] == b) & (table["quadrant"] == q)]
            if sub.empty:
                row.append(0.0)
            else:
                if y_mode == "pct":
                    row.append(float(sub["pct_within_bin"].iloc[0]))
                else:
                    row.append(float(sub["n_cells"].iloc[0]))
        mat.append(row)
    mat = np.asarray(mat, dtype=float)

    # ── NEW: normalize each bin row to exactly 100% ──────────────────────────
    if y_mode == "pct":
        bin_totals_pct = mat.sum(axis=1, keepdims=True)
        bin_totals_pct[bin_totals_pct == 0] = 1  # avoid divide-by-zero
        mat = mat / bin_totals_pct * 100.0
    # ─────────────────────────────────────────────────────────────────────────

    x = np.arange(len(bins))
    bottom = np.zeros(len(bins), dtype=float)

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111)

    for j, q in enumerate(quad_order):
        ax.bar(x, mat[:, j], bottom=bottom, edgecolor="black", linewidth=0.6)
        # Annotate each segment with its percentage
        for i in range(len(bins)):
            val = mat[i, j]
            if val >= 4.0:  # only label if segment is tall enough to fit text
                ax.text(
                    x[i],
                    bottom[i] + val / 2.0,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottom += mat[:, j]

    # ── NEW: annotate total N per bin above each bar ──────────────────────────
    if y_mode == "pct":
        bin_ns = table.groupby("bin_id")["n_cells"].sum()
        for i, b in enumerate(bin_order["bin_id"].tolist()):
            n = int(bin_ns.get(b, 0))
            ax.text(
                x[i],
                mat[i].sum() + 1,  # just above the top of the bar
                f"n={n:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    # ─────────────────────────────────────────────────────────────────────────

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=0, ha="center")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Gene expression bins (log1p scale)", fontsize=12)

    if y_mode == "pct":
        ax.set_ylabel("% of cells within bin", fontsize=12)
        ax.set_ylim(0, 108)  # extra headroom for the n= labels
    else:
        ax.set_ylabel("# of cells", fontsize=12)

    ax.legend(
        quad_order,
        title="Quadrant",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    mkdirp(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved expr-bin stacked bar plot: {out_png}")


# =========================
# neftal plot
# =========================


def plot_neftal_with_quadrants(
    adata,
    out_png,
    title,
    layer_key,
    color_gene,
    cmap,
    color_clip_pct,
    quadrant_summary: pd.DataFrame,
    leftbar_title: str,
    high_text: Optional[str] = None,
):
    x = adata.obs["neftal_x"].values
    y = adata.obs["neftal_y"].values

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 7.0], wspace=0.05)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    plot_left_stacked_bar(ax_bar, quadrant_summary, leftbar_title)

    if color_gene is not None:
        expr = get_gene_vector(adata, color_gene, layer_key)
        if expr is not None:
            expr = np.log1p(expr)
            expr, _, _ = clip_for_color(expr, color_clip_pct)
            sca = ax.scatter(x, y, c=expr, s=6, alpha=0.8, cmap=cmap, linewidths=0)
            cb = plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(
                f"log1p({color_gene}) (clipped {color_clip_pct[0]}–{color_clip_pct[1]}%)"
            )
        else:
            ax.scatter(x, y, s=6, alpha=0.8)
    else:
        ax.scatter(x, y, s=6, alpha=0.8)

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)

    ax.text(
        0.02, 0.95, "OPC-like", transform=ax.transAxes, ha="left", va="top", fontsize=13
    )
    ax.text(
        0.98,
        0.95,
        "NPC-like",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
    )
    ax.text(
        0.02,
        0.05,
        "AC-like",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
    )
    ax.text(
        0.98,
        0.05,
        "MES-like",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
    )

    if high_text:
        ax.text(
            0.02,
            0.02,
            high_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(
        "Relative meta-module score  [signed log2(|NPC − OPC| + 1)]", fontsize=12
    )
    ax.set_ylabel(
        "Relative meta-module score  [signed log2(|(OPC+NPC) − (AC+MES)| + 1)]",
        fontsize=12,
    )

    mkdirp(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved plot: {out_png}")


# =========================
# CLI
# =========================


def build_argparser():
    p = argparse.ArgumentParser(prog="5pgbm_column_scrna")

    p.add_argument(
        "--base_dir",
        default=DEFAULT_BASE_DIR,
        help="Base directory containing SCPCS* folders",
    )
    p.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for plots and report",
    )

    p.add_argument(
        "--color_gene",
        default=DEFAULT_COLOR_GENE,
        help="Gene to color points by (set to 'None' to disable)",
    )
    p.add_argument("--cmap", default=DEFAULT_CMAP, help="Matplotlib colormap name")

    p.add_argument(
        "--prefer_raw",
        action="store_true",
        help="Prefer adata.raw.X for scoring if present",
    )
    p.add_argument(
        "--do_malignant_filter",
        action="store_true",
        help="Also write malignant-only plot (if CNV col exists)",
    )

    p.add_argument(
        "--cnv_col",
        default=DEFAULT_CNV_COL,
        help="CNV column in adata.obs used for malignant filter",
    )
    p.add_argument(
        "--cnv_quantile",
        type=float,
        default=90.0,
        help="Percentile for CNV threshold if cnv_thr not set",
    )
    p.add_argument(
        "--cnv_thr",
        type=float,
        default=None,
        help="Manual CNV threshold (overrides cnv_quantile)",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used only for stable ordering)",
    )
    p.add_argument(
        "--file_glob",
        default=DEFAULT_FILE_GLOB,
        help="Pattern for .h5ad file inside each SCPCS folder",
    )
    p.add_argument(
        "--no_standardize",
        action="store_true",
        help="Disable centering/scaling of axes",
    )

    # Old: High expressing
    p.add_argument(
        "--high_gene",
        default=None,
        help="Gene for % high expressing calculation (default: color_gene)",
    )
    p.add_argument(
        "--high_mode",
        default="pct",
        choices=["pct", "abs", "nonzero"],
        help="How to define high expressers: pct=percentile, abs=absolute log1p threshold, nonzero",
    )
    p.add_argument(
        "--high_pct",
        type=float,
        default=90.0,
        help="Percentile for high (used if high_mode=pct)",
    )
    p.add_argument(
        "--high_thr",
        type=float,
        default=1.0,
        help="Absolute threshold on log1p(expr) (used if high_mode=abs)",
    )

    # NEW: Expression bins stacked-bars
    p.add_argument(
        "--bin_gene", default=None, help="Gene for binning (default: color_gene)"
    )
    p.add_argument(
        "--bin_mode",
        default="equalwidth_clip",
        choices=["equalwidth_clip", "equalwidth", "quantile", "edges"],
        help="How to create bins for log1p(expr)",
    )
    p.add_argument(
        "--bin_n",
        type=int,
        default=4,
        help="Number of bins (ignored if bin_mode=edges)",
    )
    p.add_argument(
        "--bin_edges",
        default=None,
        help="Comma-separated edges for bin_mode=edges, e.g. '0,0.4,0.8,1.2,1.6'",
    )
    p.add_argument(
        "--bin_y",
        default="pct",
        choices=["counts", "pct"],
        help="Y-axis for bin stacked bars: counts (#cells) or pct (% within bin)",
    )
    p.add_argument(
        "--bin_clip_pct_low",
        type=float,
        default=DEFAULT_COLOR_CLIP_PCT[0],
        help="Lower percentile for equalwidth_clip range",
    )
    p.add_argument(
        "--bin_clip_pct_high",
        type=float,
        default=DEFAULT_COLOR_CLIP_PCT[1],
        help="Upper percentile for equalwidth_clip range",
    )

    return p


def main():
    args = build_argparser().parse_args()
    np.random.seed(args.seed)

    base_dir = args.base_dir
    out_dir = args.out_dir
    file_glob = args.file_glob
    standardize_axes = not args.no_standardize

    color_gene = args.color_gene
    if isinstance(color_gene, str) and color_gene.strip().lower() == "none":
        color_gene = None

    mkdirp(out_dir)

    folders = find_samples(base_dir)

    all_ads = []
    reports = []

    for f in folders:
        ad, rep = load_one_sample(
            f,
            file_glob=file_glob,
            prefer_raw=args.prefer_raw,
            standardize_axes=standardize_axes,
            cnv_col=args.cnv_col,
            cnv_thr=args.cnv_thr,
            cnv_quantile=args.cnv_quantile,
        )
        all_ads.append(ad)
        reports.append(rep)

    report_df = pd.concat(reports, ignore_index=True)
    report_csv = os.path.join(out_dir, "neftal_run_report.csv")
    report_df.to_csv(report_csv, index=False)
    print(f"\n[DONE] Saved report: {report_csv}")

    print(f"\n[INFO] Concatenating ALL cells across {len(all_ads)} samples...")
    ad_all = sc.concat(
        all_ads,
        join="outer",
        label="sample_id",
        keys=[ad.obs["sample_id"].iloc[0] for ad in all_ads],
    )
    ad_all.obs_names_make_unique()

    layer_for_color = "neftal_log1p" if "neftal_log1p" in ad_all.layers else None
    if layer_for_color is None:
        print(
            "[WARN] pooled object has no neftal_log1p layer; gene extraction uses ad_all.X"
        )

    # Quadrant summary (ALL cells)
    q_all = quadrant_summary_df(ad_all, label="ALL")
    q_all_csv = os.path.join(out_dir, "neftal_quadrant_summary_ALL.csv")
    q_all.to_csv(q_all_csv, index=False)
    print(f"[DONE] Saved quadrant summary: {q_all_csv}")

    # Old feature: High-expressing summary (ALL)
    high_gene = args.high_gene if args.high_gene is not None else color_gene
    high_text_all = None
    if high_gene is not None:
        hi_all = high_expr_summary_df(
            ad_all,
            gene=high_gene,
            layer_key=layer_for_color,
            mode=args.high_mode,
            pct=args.high_pct,
            thr=args.high_thr,
            label="ALL",
        )
        hi_all_csv = os.path.join(out_dir, f"high_expr_summary_ALL_{high_gene}.csv")
        hi_all.to_csv(hi_all_csv, index=False)
        print(f"[DONE] Saved high-expressing summary: {hi_all_csv}")

        thr_used = (
            float(hi_all["threshold_log1p"].iloc[0])
            if not np.isnan(hi_all["threshold_log1p"].iloc[0])
            else np.nan
        )
        pct_high = float(hi_all["pct_high"].iloc[0])
        if args.high_mode == "pct":
            high_text_all = f"{high_gene}: top {args.high_pct:.0f}% (thr log1p={thr_used:.3f}) → {pct_high:.2f}% high"
        elif args.high_mode == "abs":
            high_text_all = (
                f"{high_gene}: log1p >= {args.high_thr:.3f} → {pct_high:.2f}% high"
            )
        else:
            high_text_all = f"{high_gene}: expr > 0 → {pct_high:.2f}% high"

    # neftal plot (ALL)
    out_all = os.path.join(
        out_dir, f"neftal_ALLSAMPLES_ALL_{color_gene if color_gene else 'noclr'}.png"
    )
    plot_neftal_with_quadrants(
        ad_all,
        out_all,
        title="Two-dimensional representation of cellular states\nALL samples • ALL cells"
        + (f"\nColored by {color_gene}" if color_gene else ""),
        layer_key=layer_for_color,
        color_gene=color_gene,
        cmap=args.cmap,
        color_clip_pct=DEFAULT_COLOR_CLIP_PCT,
        quadrant_summary=q_all,
        leftbar_title="% of cells",
        high_text=high_text_all,
    )

    # NEW: Expression-bin stacked bars (ALL)
    bin_gene = args.bin_gene if args.bin_gene is not None else color_gene
    if bin_gene is not None:
        clip_pct = (args.bin_clip_pct_low, args.bin_clip_pct_high)
        tab_all, edges_all = exprbin_quadrant_table(
            ad_all,
            gene=bin_gene,
            layer_key=layer_for_color,
            group_label="ALL",
            bin_mode=args.bin_mode,
            n_bins=args.bin_n,
            clip_pct=clip_pct,
            bin_edges=args.bin_edges,
            y_mode=args.bin_y,
        )
        if not tab_all.empty:
            tab_all_csv = os.path.join(
                out_dir, f"exprbin_quadrant_counts_ALL_{bin_gene}.csv"
            )
            tab_all.to_csv(tab_all_csv, index=False)
            print(f"[DONE] Saved expr-bin quadrant table: {tab_all_csv}")

            out_bar_all = os.path.join(
                out_dir, f"exprbin_quadrant_stackedbar_ALL_{bin_gene}.png"
            )
            plot_exprbin_quadrant_stackedbars(
                tab_all,
                out_png=out_bar_all,
                title=f"{bin_gene} expression bins vs neftal quadrant (ALL cells)\nmode={args.bin_mode}, bins={int(tab_all['n_bins'].iloc[0])}",
                y_mode=("pct" if args.bin_y == "pct" else "counts"),
            )
        else:
            print("[WARN] Could not compute expr-bin table for ALL (empty).")

    # MALIGNANT pooled plot + summary
    if args.do_malignant_filter and "is_malignant_cnv" in ad_all.obs.columns:
        mal = ad_all[ad_all.obs["is_malignant_cnv"].values].copy()
        if mal.n_obs == 0:
            print(
                "[WARN] No malignant cells detected after filtering. Skipping malignant plot."
            )
        else:
            q_mal = quadrant_summary_df(mal, label="MALIGNANT")
            q_mal_csv = os.path.join(out_dir, "neftal_quadrant_summary_MALIGNANT.csv")
            q_mal.to_csv(q_mal_csv, index=False)
            print(f"[DONE] Saved quadrant summary: {q_mal_csv}")

            high_text_mal = None
            if high_gene is not None:
                hi_mal = high_expr_summary_df(
                    mal,
                    gene=high_gene,
                    layer_key=layer_for_color,
                    mode=args.high_mode,
                    pct=args.high_pct,
                    thr=args.high_thr,
                    label="MALIGNANT",
                )
                hi_mal_csv = os.path.join(
                    out_dir, f"high_expr_summary_MALIGNANT_{high_gene}.csv"
                )
                hi_mal.to_csv(hi_mal_csv, index=False)
                print(f"[DONE] Saved high-expressing summary: {hi_mal_csv}")

                thr_used = (
                    float(hi_mal["threshold_log1p"].iloc[0])
                    if not np.isnan(hi_mal["threshold_log1p"].iloc[0])
                    else np.nan
                )
                pct_high = float(hi_mal["pct_high"].iloc[0])
                if args.high_mode == "pct":
                    high_text_mal = f"{high_gene}: top {args.high_pct:.0f}% (thr log1p={thr_used:.3f}) → {pct_high:.2f}% high"
                elif args.high_mode == "abs":
                    high_text_mal = f"{high_gene}: log1p >= {args.high_thr:.3f} → {pct_high:.2f}% high"
                else:
                    high_text_mal = f"{high_gene}: expr > 0 → {pct_high:.2f}% high"

            out_mal = os.path.join(
                out_dir,
                f"neftal_ALLSAMPLES_MALIGNANT_{color_gene if color_gene else 'noclr'}.png",
            )
            plot_neftal_with_quadrants(
                mal,
                out_mal,
                title="Two-dimensional representation of cellular states\nALL samples • MALIGNANT (CNV)"
                + (f"\nColored by {color_gene}" if color_gene else ""),
                layer_key=layer_for_color,
                color_gene=color_gene,
                cmap=args.cmap,
                color_clip_pct=DEFAULT_COLOR_CLIP_PCT,
                quadrant_summary=q_mal,
                leftbar_title="% of malignant",
                high_text=high_text_mal,
            )

            # NEW: Expression-bin stacked bars (MALIGNANT)
            if bin_gene is not None:
                clip_pct = (args.bin_clip_pct_low, args.bin_clip_pct_high)
                tab_mal, edges_mal = exprbin_quadrant_table(
                    mal,
                    gene=bin_gene,
                    layer_key=layer_for_color,
                    group_label="MALIGNANT",
                    bin_mode=args.bin_mode,
                    n_bins=args.bin_n,
                    clip_pct=clip_pct,
                    bin_edges=args.bin_edges,
                    y_mode=args.bin_y,
                )
                if not tab_mal.empty:
                    tab_mal_csv = os.path.join(
                        out_dir, f"exprbin_quadrant_counts_MALIGNANT_{bin_gene}.csv"
                    )
                    tab_mal.to_csv(tab_mal_csv, index=False)
                    print(f"[DONE] Saved expr-bin quadrant table: {tab_mal_csv}")

                    out_bar_mal = os.path.join(
                        out_dir, f"exprbin_quadrant_stackedbar_MALIGNANT_{bin_gene}.png"
                    )
                    plot_exprbin_quadrant_stackedbars(
                        tab_mal,
                        out_png=out_bar_mal,
                        title=f"{bin_gene} expression bins vs neftal quadrant (MALIGNANT)\nmode={args.bin_mode}, bins={int(tab_mal['n_bins'].iloc[0])}",
                        y_mode=("pct" if args.bin_y == "pct" else "counts"),
                    )
                else:
                    print(
                        "[WARN] Could not compute expr-bin table for MALIGNANT (empty)."
                    )

    print(f"\n[INFO] Output folder: {out_dir}")
    print("[DONE] Finished.")


if __name__ == "__main__":
    main()
