import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, spearmanr, linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  Ribosomal gene set
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RIBO_NAMES = [
    "Rpl26", "Rpl15", "Rpl4", "Rpl32", "Rpl29", "Rpl31", "Rpl13", "Rpl12",
    "Rpl7l1", "Rpl34", "Rpl28", "Rpl5l1", "Mrpl33", "Rpl6", "Rpl9-ps31",
    "Rpl13a-ps4", "Rpl35a-ps8", "Rpl35a-ps6", "Rpl37a", "Rpl37l4", "Rpl26-ps2",
    "Rpl30", "Rps10", "Rps12l2", "Rps15a", "Rps17l2", "Rps18l1", "Rps20",
    "Rps20l1", "Rps23", "Rps25l2", "Rps27a", "Rps27a-ps12", "Rps4x",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  Canonical-pathway gene sets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWN_REGS = {
    "mTOR pathway": ["Mtor", "Rptor", "Rictor", "Rps6kb1", "Rps6kb2", "Eif4ebp1",
                     "Eif4ebp2", "Rraga", "Rragb", "Rragc", "Rragd", "Rheb",
                     "Lamtor1", "Lamtor2", "Lamtor4"],
    "PolI / rDNA": ["Ubtf", "Polr1a", "Polr1b", "Polr1c", "Polr1d", "Polr1e",
                    "Rrn3", "Taf1d", "Taf1c", "Tcof1", "Nom1"],
    "MYC": ["Myc", "Mycn", "Mycl", "Max", "Mxd1", "Mxd4", "Mnt", "Mlx"],
    "p53/stress": ["Tp53", "Trp53", "Mdm2", "Mdm4", "Cdkn1a", "Rb1", "E2f1",
                   "Rbl1", "Rbl2"],
    "Ribo biogenesis": ["Bop1", "Pes1", "Wdr12", "Ddx56", "Nop2", "Rcl1", "Fbl",
                        "Nop58", "Nhp2", "Gar1", "Dkc1", "Naf1", "Nop10",
                        "Lsg1", "Nmd3", "Efl1", "Sbds", "Surf6", "Wdr3",
                        "Utp4", "Utp5", "Utp6", "Utp10", "Utp14a", "Utp15",
                        "Utp18", "Utp23", "Fcf1"],
    "eIF / transl.": ["Eif4e", "Eif4g1", "Eif4g2", "Eif4a1", "Eif4a2", "Eif2s1",
                      "Eif2s2", "Eif2s3", "Eif2b1", "Eif3a", "Eif3b", "Eif3c",
                      "Eif3d", "Eif3e", "Eif3f", "Eif3g", "Eif3h", "Eif3i",
                      "Eif3j", "Eif3k", "Eif3l", "Eif3m", "Eif5", "Eif5b",
                      "Eif6", "Eif1", "Eif1a", "Eif1ax"],
    "LARP / 5'TOP": ["Larp1", "Larp1b", "Larp4", "Larp4b", "Larp6", "Ybx1",
                     "Ybx2", "Ybx3", "Eif4e2"],
    "Notch/Wnt": ["Ctnnb1", "Axin1", "Axin2", "Apc", "Gsk3b", "Tcf7l2",
                  "Hey1", "Hey2", "Hes1", "Hes5", "Notch1", "Notch2",
                  "Ptch1", "Gli1", "Gli2", "Smo"],
}
ALL_KNOWN = {g: cat for cat, genes in KNOWN_REGS.items() for g in genes}

PATHWAY_COL = {
    "LARP / 5'TOP": "#E41A1C",
    "Ribo biogenesis": "#FF7F00",
    "Notch/Wnt": "#4DAF4A",
    "mTOR pathway": "#984EA3",
    "eIF / transl.": "#A65628",
    "PolI / rDNA": "#F781BF",
    "p53/stress": "#377EB8",
    "MYC": "#E41A1C",
    "Sig. DE+corr.": "#999999",
}

# Spotlight genes for annotation (gene, dot-colour, pathway label)
SPOTLIGHT = [
    ("Ybx1", "#E41A1C", "LARP / 5'TOP"),
    ("Bop1", "#FF7F00", "Ribo biogenesis"),
    ("Hey1", "#4DAF4A", "Notch/Wnt"),
    ("Hes1", "#4DAF4A", "Notch/Wnt"),
    ("Notch2", "#4DAF4A", "Notch/Wnt"),
    ("Eif4ebp2", "#984EA3", "mTOR pathway"),
    ("Eif4ebp1", "#984EA3", "mTOR pathway"),
    ("Eif4e", "#A65628", "eIF / transl."),
    ("Eif3k", "#A65628", "eIF / transl."),
    ("Tcof1", "#F781BF", "PolI / rDNA"),
    ("Utp23", "#FF7F00", "Ribo biogenesis"),
    ("Tmem212", "#999999", "Sig. DE+corr."),
    ("Cdkn1a", "#377EB8", "p53/stress"),
    ("Rb1", "#377EB8", "p53/stress"),
    ("Ctnnb1", "#4DAF4A", "Notch/Wnt"),
]


def transform_data(data, metadata, run_type='', skip=False, save_dir=None, skip_norm=False):
    """Transforms, imputes, and normalizes the data."""

    # Replace all exactly 0 values with NaN for imputation
    data = data.replace(0, np.nan)

    # Impute missing values
    data = impute_zeros(data, metadata, condition='Treatment', run_type=run_type, skip_if_exist=skip, save_dir=save_dir)

    # Log2 transform with a small pseudocount to prevent -inf and extreme negative compression
    # We use a standard min-value pseudocount if values < 1 exist
    pseudocount = 1.0 if data.min().min() < 1 else 0.0
    data = np.log2(data + pseudocount)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        data.to_csv(os.path.join(save_dir, f"imputed_log{run_type}.csv"))

    if not skip_norm:
        data = zscore_all_by_pbs(data, metadata)
        if save_dir:
            data.to_csv(os.path.join(save_dir, f"imputed_log_zscore{run_type}.csv"))

    return data, metadata


def impute_zeros(to_impute, meta_data, condition, run_type='', skip_if_exist=False, mean=False, save_dir=None):
    """
    Replaces NaNs with the mean/min of other biological replicates in the same treatment/antibiotic group.
    """
    save_path = os.path.join(save_dir or '.', f'imputed_all_zeros_removed{run_type}.csv')

    if skip_if_exist and os.path.exists(save_path):
        print(f"Loading existing imputed data from {save_path}")
        return pd.read_csv(save_path, index_col=0)

    # Get coordinates of NaNs
    row_indices, col_indices = np.where(to_impute.isnull())
    total = len(row_indices)

    print(f"Starting imputation for {total} missing values...")

    for counter, (i, j) in enumerate(zip(row_indices, col_indices), 1):
        gene_name = to_impute.index[i]
        sample_id = to_impute.columns[j]

        # Extract metadata for the current sample
        sample_meta = meta_data[meta_data['ID'] == sample_id]
        if sample_meta.empty:
            continue

        antibiotic = sample_meta['Drug'].values[0]
        treatment = sample_meta[condition].values[0]

        # Find other samples (mice) in the exact same condition, excluding the current one
        replicate_ids = meta_data[
            (meta_data['Drug'] == antibiotic) &
            (meta_data[condition] == treatment) &
            (meta_data['ID'] != sample_id)
            ]['ID'].values

        # Ensure the replicates actually exist in the dataframe columns
        valid_replicates = [m for m in replicate_ids if m in to_impute.columns]

        if not valid_replicates:
            # If no replicates exist, fill with overall gene min to avoid leaving NaNs
            to_impute.loc[gene_name, sample_id] = to_impute.loc[gene_name].min()
            continue

        # Get the expression values of the replicates for this specific gene
        replicate_values = to_impute.loc[gene_name, valid_replicates]
        imputed_val = np.nanmin(replicate_values)

        # Fallback if all replicates are also NaN
        if np.isnan(imputed_val):
            imputed_val = 0  # Absolute fallback if gene is NaN everywhere
        to_impute.loc[gene_name, sample_id] = imputed_val
        if counter % 5000 == 0:
            print(f"{counter}/{total} missing values imputed")
    if save_dir:
        to_impute.to_csv(save_path)
    return to_impute


def zscore_all_by_pbs(data, metadata):
    """
    Normalizes data by calculating Z-scores relative to the PBS control group.
    Incorporates an epsilon to prevent division-by-zero on low-variance genes.
    """
    treatments = metadata['Treatment'].unique()
    antibiotics = [abx for abx in metadata['Drug'].unique() if abx != "Control"]

    for treat in treatments:
        # Get PBS samples for this treatment
        pbs_meta = metadata[(metadata['Drug'] == "Control") & (metadata["Treatment"] == treat)]
        pbs_ids = [pid for pid in pbs_meta['ID'].values if pid in data.columns]

        if not pbs_ids:
            continue

        pbs_data = data[pbs_ids]

        # Calculate row-wise mean and std for the PBS group
        pbs_mean = pbs_data.mean(axis=1)
        pbs_std = pbs_data.std(axis=1)
        pbs_std[pbs_std == 0] = pbs_std[pbs_std != 0].min()

        # Normalize the PBS samples themselves
        data[pbs_ids] = data[pbs_ids].sub(pbs_mean, axis=0).div(pbs_std, axis=0)

        # Normalize all other antibiotic groups within this treatment against the PBS stats
        for anti in antibiotics:
            abx_meta = metadata[(metadata['Drug'] == anti) & (metadata["Treatment"] == treat)]
            abx_ids = [aid for aid in abx_meta['ID'].values if aid in data.columns]

            if abx_ids:
                data[abx_ids] = data[abx_ids].sub(pbs_mean, axis=0).div(pbs_std, axis=0)

    return data


# def run_robust_rf(df_norm, labels, n_runs=1000):
#     """
#     Runs RF 1000 times, averages feature importance and confusion matrices.
#
#     Args:
#         df_norm (pd.DataFrame): Genes (index) x Samples (cols). Normalized data.
#         labels (pd.Series/list): Binary labels aligned with df_norm columns.
#         n_runs (int): Number of iterations.
#
#     Returns:
#         avg_importance (pd.Series): Genes sorted by importance.
#         avg_cm (np.array): Averaged confusion matrix.
#     """
#     # Transpose for sklearn: (Samples x Features)
#     X = df_norm.T
#     y = np.array(labels)
#
#     # Initialize accumulators
#     n_features = X.shape[1]
#     importances_sum = np.zeros(n_features)
#     cm_sum = np.zeros((2, 2))
#
#     # Stratified Split ensures class balance in every train/test split
#     # Using 80/20 split for each iteration
#     sss = StratifiedShuffleSplit(n_splits=n_runs, test_size=0.2, random_state=42)
#
#     print(f"Starting {n_runs} RF runs...")
#
#     # Since we need to accumulate, we iterate through the splitter
#     for train_idx, test_idx in sss.split(X, y):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]
#
#         # Initialize and fit RF
#         # n_estimators=100 is standard; reduced max_depth prevents overfitting
#         rf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
#         rf.fit(X_train, y_train)
#
#         # Accumulate Importance
#         importances_sum += rf.feature_importances_
#
#         # Accumulate Confusion Matrix
#         y_pred = rf.predict(X_test)
#         cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
#         cm_sum += cm
#
#     # Calculate Averages
#     avg_importance = pd.Series(importances_sum / n_runs, index=X.columns)
#     avg_importance = avg_importance.sort_values(ascending=False)
#
#     avg_cm = cm_sum / n_runs
#
#     return avg_importance, avg_cm


# --- Configuration ---
BASE_PATH = "/Users/yonchlevin/Desktop/ErezLab/MouseAbxBel/Git/Data/rat-neurons"
RAW_PATH = os.path.join(BASE_PATH, "genes_abundance_named.tsv")
NORM_PATH = os.path.join(BASE_PATH, "genes_norm_named-20260211_NeuronInvitroRNAseq.tsv")
OUTPUT_DIR = os.path.join(BASE_PATH, "Analysis_Results")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_sanitize_data(raw_path, norm_path, to_drop):
    """Loads raw and normalized data, ensuring consistent indices."""
    print(f"Loading data from {BASE_PATH}...")

    # Load Raw Counts (for DESeq2)
    # Assuming the file has gene_id/gene_name columns. We need a unique index.
    raw_df = pd.read_csv(raw_path, sep='\t')
    if 'gene_name' in raw_df.columns:
        # Deduplicate by taking sum or mean? For raw counts, usually sum if duplicate IDs exist.
        # Ideally, use gene_id as index to be safe, but map names later.
        raw_df = raw_df.set_index('gene_id')

    # Load Normalized Data (for GSEA/RF)
    norm_df = pd.read_csv(norm_path, sep='\t')
    if 'gene_id' in norm_df.columns:
        norm_df = norm_df.set_index('gene_id')

    # Identify Gene Names map for later
    # Create a dictionary: ID -> Name
    id_to_name = {}
    if 'gene_name' in raw_df.columns:
        id_to_name.update(raw_df['gene_name'].to_dict())
    if 'gene_name' in norm_df.columns:
        id_to_name.update(norm_df['gene_name'].to_dict())

    raw_df = raw_df.fillna(0)
    norm_df = norm_df.fillna(0)

    # --- Remove Mitochondrial Genes (Mt-) ---
    # Find IDs where the associated name starts with 'mt-' (case insensitive)
    mito_ids = [gid for gid, name in id_to_name.items() if str(name).lower().startswith('mt-')]

    if mito_ids:
        print(f"Removing {len(mito_ids)} mitochondrial genes (starting with 'Mt-')...")
        # Drop from dataframe rows
        raw_df = raw_df.drop(index=mito_ids, errors='ignore')
        norm_df = norm_df.drop(index=mito_ids, errors='ignore')

    # Drop non-numeric columns for analysis
    raw_counts = raw_df.select_dtypes(include=[np.number])
    norm_counts = norm_df.select_dtypes(include=[np.number])

    # Remove the sample column if present. Use errors='ignore' so this is safe when the column
    # doesn't exist, and assign back to ensure the DataFrame is updated.
    raw_counts = raw_counts.drop(columns=to_drop, errors='ignore')
    norm_counts = norm_counts.drop(columns=to_drop, errors='ignore')
    # 5. Filter Sparse Genes (> 50% zeros across samples)
    # Replaces the previous "expressed in 0 or 1 sample" logic with the reference logic
    raw_zeros = (raw_counts == 0).sum(axis=1)
    raw_sparse = raw_zeros[raw_zeros > 0.5 * raw_counts.shape[1]]

    norm_zeros = (norm_counts == 0).sum(axis=1)
    norm_sparse = norm_zeros[norm_zeros > 0.5 * norm_counts.shape[1]]

    # Drop from both
    raw_counts = raw_counts.drop(index=raw_sparse.index, errors='ignore')
    norm_counts = norm_counts.drop(index=norm_sparse.index, errors='ignore')
    print(f"Dropped {len(raw_sparse.index)} from raw and {len(norm_sparse.index)} from norm")

    # CPM Normalization (Applied ONLY to norm_counts)
    # (df * 1000000).divide(df.sum(axis=0), axis=1)
    norm_counts = (norm_counts * 1000000).divide(norm_counts.sum(axis=0), axis=1)

    # Ensure integer counts for DESeq2
    raw_counts = raw_counts.round().astype(int)

    return raw_counts, norm_counts, id_to_name


def run_pca_analysis(df, id_to_name):
    """
    Runs PCA on normalized data and plots PC1 vs PC2 with sample annotations.
    """
    print("Running PCA Analysis...")

    # Transpose: Samples as rows, Genes as columns
    X = df.T

    # Standardize features (genes) - Important for PCA
    X_scaled = StandardScaler().fit_transform(X)

    # Run PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    # Create DataFrame for plotting
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'], index=X.index)

    # Assign Groups
    pca_df['Group'] = ['Van' if 'v' in name.lower() else 'Control' for name in pca_df.index]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='Group',
        palette={'Control': '#95a5a6', 'Van': '#e74c3c'},
        s=100
    )

    # Annotate every sample
    for sample_name in pca_df.index:
        plt.text(
            pca_df.loc[sample_name, 'PC1'] + 0.2,
            pca_df.loc[sample_name, 'PC2'] + 0.2,
            sample_name,
            fontsize=9
        )

    plt.title(f"PCA Analysis (Explained Var: {pca.explained_variance_ratio_[:2].sum():.2%})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "PCA_Plot.png"))
    plt.close()


def check_specific_outliers(df, id_to_name, z_threshold=3.0, min_fold_change=2.0, min_expression=5.0):
    """
    Checks specific samples (C21, C16, V21) for extreme gene expression
    relative to the REST of their group (Leave-One-Out approach).
    """
    print("Checking for specific outliers (C16, V21)...")

    # Define targets
    targets = {
        'Control': ['c_16'],
        'Van': ['v_21']
    }

    # Identify all columns for each group
    control_cols = [c for c in df.columns if 'c_' in c.lower()]
    van_cols = [c for c in df.columns if 'v_' in c.lower()]

    results = []

    def analyze_group(all_group_cols, group_name, target_substrings):
        if len(all_group_cols) < 3:
            print(
                f"Warning: Not enough samples in {group_name} to perform Leave-One-Out outlier detection (Need N>=3).")
            return

        # Find the specific sample names we want to test
        target_samples = []
        for t in target_substrings:
            found = [c for c in all_group_cols if t in c]
            target_samples.extend(found)

        # Iterate over the specific samples we are interested in (e.g., C21)
        for sample in target_samples:

            # --- CRITICAL: Leave-One-Out Calculation ---
            # Define the "reference" group as everyone EXCEPT the current sample
            ref_cols = [c for c in all_group_cols if c != sample]

            ref_df = df[ref_cols]

            # Calculate Mean and Std on the REFERENCE set only
            mu = ref_df.mean(axis=1)
            sigma = ref_df.std(axis=1)

            # Avoid division by zero
            # sigma = sigma.replace(0, 1e-9)
            sigma = sigma + 0.01

            # Calculate Z-score for the target sample against the reference stats
            # (Target - Ref_Mean) / Ref_Std
            z_scores = (df[sample] - mu) / sigma
            fc = np.log2((df[sample] + 1) / (mu + 1))

            # # Filter for extreme outliers
            # outliers = z_scores[abs(z_scores) > z_threshold]
            # A: Statistical Significance (Z-score)
            mask_stat = z_scores.abs() > z_threshold

            # B: Biological Magnitude (Fold Change)
            # abs(Log2FC) > 1 means magnitude change > 2x
            mask_bio = fc.abs() > np.log2(min_fold_change)

            # C: Minimum Expression (Ignore noise)
            # Require either the sample OR the group mean to be substantial
            mask_expr = (df[sample] > min_expression) | (mu > min_expression)

            # Combine Filters
            final_mask = mask_stat & mask_bio & mask_expr

            outliers = z_scores[final_mask]

            for gene_id, z in outliers.items():
                gene_name = str(id_to_name.get(gene_id, str(gene_id)))
                is_mito = 'mt-' in gene_name.lower() or 'mitochondri' in gene_name.lower()

                results.append({
                    'Sample': sample,
                    'Group': group_name,
                    'Gene ID': gene_id,
                    'Gene Name': gene_name,
                    'Z-Score': round(z, 2),
                    'Is_Mitochondrial': is_mito,
                    'Expression_Val': df.loc[gene_id, sample],
                    'Group_Mean_LOO': round(mu[gene_id], 2)  # LOO = Leave One Out
                })

    # Run checks
    analyze_group(control_cols, 'Control', control_cols)
    # analyze_group(control_cols, 'Control', targets['Control'])
    analyze_group(van_cols, 'Van', van_cols)
    # analyze_group(van_cols, 'Van', targets['Van'])

    # Save results
    if results:
        res_df = pd.DataFrame(results)
        # Sort by Z-score magnitude
        res_df['abs_z'] = res_df['Z-Score'].abs()
        res_df = res_df.sort_values('abs_z', ascending=False).drop(columns=['abs_z'])

        save_path = os.path.join(OUTPUT_DIR, "Specific_Outlier_Check.csv")
        res_df.to_csv(save_path, index=False)
        print(f"Found {len(res_df)} outliers. Saved to {save_path}")

        # Print summary of Mitochondrial hits
        mito_hits = res_df[res_df['Is_Mitochondrial']]
        if not mito_hits.empty:
            print("\n!!! Mitochondrial Outliers Detected !!!")
            print(mito_hits[['Sample', 'Gene Name', 'Z-Score', 'Expression_Val', 'Group_Mean_LOO']])

            # --- Plot Heatmap for Extreme Outliers (Z > 100) ---
            for group in res_df["Group"].unique():
                extreme_df = res_df[(res_df['Z-Score'].abs() > 100) & (res_df['Group'] == group)]

                if not extreme_df.empty:
                    print(f"\nPlotting heatmap for {len(extreme_df)} extreme outliers (|Z| > 100)...")

                    # Get unique Gene IDs for the plot
                    gene_ids_to_plot = extreme_df['Gene ID'].unique()

                    # Subset the ORIGINAL dataframe to show these genes across ALL samples
                    cols = [col for col in df.columns if group[0].lower() in col]
                    plot_data = df.loc[gene_ids_to_plot, cols].copy()
                    # plot_data = df.loc[gene_ids_to_plot].copy()

                    # Map IDs to Names
                    plot_data.index = plot_data.index.map(lambda x: id_to_name.get(x, x))

                    # Generate Colors: Van=Red, Control=Grey
                    col_colors = ['#e74c3c' if 'v_' in c.lower() else '#95a5a6' for c in plot_data.columns]

                    # Create Clustermap
                    # z_score=0 standardizes rows to visualize relative expression
                    plt.figure()
                    g = sns.clustermap(
                        plot_data,
                        z_score=0,
                        cmap="vlag",
                        col_colors=col_colors,
                        col_cluster=True,  # Cluster samples to see if the outlier stands apart
                        figsize=(12, max(6, len(gene_ids_to_plot) * 0.3)),
                        dendrogram_ratio=(0.1, 0.2),
                        cbar_pos=(0, .2, .03, .4)
                    )

                    g.ax_heatmap.set_title(f"Extreme Outliers (|Z| > 100), {group}", y=1.2)

                    # Add Legend
                    from matplotlib.patches import Patch
                    handles = [Patch(facecolor='#95a5a6', label='Control'), Patch(facecolor='#e74c3c', label='Van')]
                    g.ax_col_dendrogram.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 1.25), ncol=2)

                    plt.savefig(os.path.join(OUTPUT_DIR, f"Extreme_Outliers_Z100_Heatmap_{group}.png"),
                                bbox_inches='tight')
                    plt.close()
                else:
                    print("No outliers with |Z| > 100 found. Skipping heatmap.")
    else:
        print("No extreme outliers found for specified samples.")


def run_deseq2_analysis(counts_df, id_to_name):
    """Runs PyDESeq2 to get proper statistical rankings."""
    print("Running DESeq2 Analysis...")

    # 1. Create Metadata based on sample names
    # Assuming columns like '24h_c_13_S1' (Control) vs '24h_v_19_S7' (Van)
    metadata = pd.DataFrame(index=counts_df.columns)
    metadata['Condition'] = ['Control' if '_c_' in col else 'Van' for col in metadata.index]

    # 2. Transpose for PyDESeq2 (Expects Samples as Rows)
    counts_T = counts_df.T

    # 3. Run DESeq2
    dds = DeseqDataSet(
        counts=counts_T,
        metadata=metadata,
        design_factors="Condition"
    )
    dds.deseq2()

    # 4. Statistical Test (Van vs Control)
    stat_res = DeseqStats(dds, contrast=["Condition", "Van", "Control"])
    stat_res.summary()

    # 5. Extract Results
    res_df = stat_res.results_df

    # Create Rank Metric: -log10(pvalue) * sign(log2FoldChange)
    # Handle p=0 or NA
    res_df['log2FoldChange'] = res_df['log2FoldChange'].fillna(0)
    res_df['pvalue'] = res_df['pvalue'].fillna(1.0)
    res_df['rank_metric'] = -np.log10(res_df['pvalue'] + 1e-300) * np.sign(res_df['log2FoldChange'])

    res_df['gene name'] = res_df.index.map(id_to_name)

    # Save Full Results
    res_df.to_csv(os.path.join(OUTPUT_DIR, "DESeq2_results.csv"))

    return res_df


def run_random_forest(norm_df, metadata_labels):
    """Runs Random Forest on normalized data to find feature importance."""
    print("Running Random Forest...")

    # Filter low expression
    mask = norm_df.sum(axis=1) > 10
    filt_df = norm_df[mask]

    # Log Transform
    X = np.log2(filt_df.T + 1)
    y = metadata_labels

    # Train
    rf = RandomForestClassifier(n_estimators=1_000, random_state=42)
    rf.fit(X, y)

    # Extract Importances
    importances = pd.DataFrame({
        'Feature': filt_df.index,  # Gene IDs
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Save
    importances.to_csv(os.path.join(OUTPUT_DIR, "RandomForest_Importance.csv"), index=False)

    # Plot Top 20
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances.head(20), x='Importance', y='Feature')
    plt.title("Top 20 Genes by Random Forest Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "RF_Top20_Features.png"))
    plt.close()


def generate_gsea_enrichr_files(norm_df, deseq_res, id_to_name):
    """Generates standard GSEA (.txt, .cls, .rnk) and Enrichr lists."""
    print("Generating GSEA and Enrichr files...")

    # Map IDs to Names for human-readable files
    # Create a column "NAME" in deseq_res
    # Convert to Series -> Map -> Fill using original Series
    deseq_res['NAME'] = deseq_res.index.to_series().map(id_to_name).fillna(deseq_res.index.to_series())

    # --- 1. .rnk File (Ranked List from DESeq2) ---
    # Average ranks if gene names are duplicated
    rnk_df = deseq_res[['NAME', 'rank_metric']].groupby('NAME').mean()
    rnk_df = rnk_df.sort_values('rank_metric', ascending=False)
    rnk_df.to_csv(os.path.join(OUTPUT_DIR, "Rat_Neurons_DESeq2.rnk"), sep='\t', header=False)

    # --- 2. Enrichr Lists (Up/Down) ---
    padj_cutoff = 0.05
    lfc_cutoff = 0.5  # log2 fold change

    up_genes = deseq_res[(deseq_res['padj'] < padj_cutoff) & (deseq_res['log2FoldChange'] > lfc_cutoff)]['NAME']
    down_genes = deseq_res[(deseq_res['padj'] < padj_cutoff) & (deseq_res['log2FoldChange'] < -lfc_cutoff)]['NAME']

    up_genes.to_csv(os.path.join(OUTPUT_DIR, "UP_genes.txt"), index=False, header=False)
    down_genes.to_csv(os.path.join(OUTPUT_DIR, "DOWN_genes.txt"), index=False, header=False)

    # --- 3. .txt File (Expression Data for GSEA) ---
    # Use Normalized Data
    gsea_txt = norm_df.copy()
    gsea_txt['NAME'] = gsea_txt.index.to_series().map(id_to_name).fillna(gsea_txt.index.to_series())

    # Group by Name (averaging duplicates) and set as index
    gsea_txt = gsea_txt.groupby('NAME').mean()

    # Reset index to make NAME a column, insert DESCRIPTION
    gsea_txt = gsea_txt.reset_index()
    gsea_txt.insert(1, 'DESCRIPTION', 'na')
    gsea_txt.to_csv(os.path.join(OUTPUT_DIR, "Rat_Neurons_Expression.txt"), sep='\t', index=False)

    # --- 4. .cls File (Phenotypes) ---
    # Determine classes from columns
    # We need the columns from the norm_df (excluding gene info)
    samples = norm_df.columns
    classes = ['Control' if '_c_' in s else 'Van' for s in samples]

    with open(os.path.join(OUTPUT_DIR, "Rat_Neurons_Phenotypes.cls"), "w") as f:
        f.write(f"{len(samples)} {2} 1\n")
        f.write("# Control Van\n")
        f.write(" ".join(classes))


def plot_top_genes(df, labels, importance_scores, id_to_name, n_top, color_dict):
    # 1. Select Top N Genes
    top_genes = importance_scores.head(n_top).index

    # 2. Sort Data by Condition (for visual blocking)
    # This aligns columns: all 'Without Van' first, then 'With Van'
    sorted_indices = np.argsort(labels)
    sorted_cols = df.columns[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]

    data_subset = df.loc[top_genes, sorted_cols]
    data_subset.index = data_subset.index.map(id_to_name)

    # 3. Create Column Colors
    col_colors = pd.Series(sorted_labels).map(color_dict).to_numpy()

    # 4. Plot
    # z_score=0 standardizes rows (genes) so you see relative expression patterns
    # rather than absolute TPM magnitude (which hides the variation).
    g = sns.clustermap(data_subset,
                       col_cluster=False,  # Keep our sorted order (Control vs Van)
                       row_cluster=True,  # Cluster genes by similarity
                       z_score=0,  # Standardize rows (Critical for visual!)
                       cmap="vlag",  # Blue-White-Red Diverging palette
                       col_colors=col_colors,
                       figsize=(10, n_top * 0.25 + 2),  # Auto-adjust height
                       dendrogram_ratio=(.1, .2),
                       cbar_pos=(0, .2, .03, .4))

    # save the heatmap data as a CSV in the same order as the clustered heatmap
    # If seaborn returned a dendrogram ordering, use it to reorder rows to match the plot
    if hasattr(g, 'dendrogram_row') and getattr(g, 'dendrogram_row') is not None:
        try:
            row_order = g.dendrogram_row.reordered_ind
            clustered_df = data_subset.iloc[row_order, :]
        except Exception:
            print("no dendrogram")
            clustered_df = data_subset
    else:
        clustered_df = data_subset

    csv_path = os.path.join(OUTPUT_DIR, f"top_{n_top}_heatmap_data_clustered.csv")
    clustered_df.to_csv(csv_path)

    g.ax_heatmap.set_title(f"Top {n_top} Important Genes (RF)", y=1.2)
    # Add legend manually since clustermap removes it
    for label, color in color_dict.items():
        g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
    g.ax_col_dendrogram.legend(loc="center", ncol=2, bbox_to_anchor=(0.5, 1.25))
    plt.savefig(BASE_PATH + f"/Analysis_Results/top_{n_top}.png")
    # plt.show()
    plt.close()


def run_robust_rf(df_norm, labels, n_runs=1000):
    X = df_norm.T  # Transpose: Samples are rows for sklearn
    y = np.array(labels)

    # Storage for results
    feature_importances = np.zeros(df_norm.shape[0])  # Array of length n_genes
    cm_sum = np.zeros((2, 2))

    # 1000 Iterations
    sss = StratifiedShuffleSplit(n_splits=n_runs, test_size=0.2, random_state=42)

    print(f"Running {n_runs} iterations...")

    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train
        rf = RandomForestClassifier(n_estimators=50, max_depth=None, n_jobs=-1)
        rf.fit(X_train, y_train)

        # Accumulate Importance
        feature_importances += rf.feature_importances_

        # Accumulate Confusion Matrix
        y_pred = rf.predict(X_test)
        # Ensure labels are sorted so matrix indices are consistent
        unique_labels = sorted(list(set(labels)))
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        cm_sum += cm

        if (i + 1) % 100 == 0: print(f"Completed {i + 1} runs...")

    # Average results
    avg_importance = pd.Series(feature_importances / n_runs, index=df_norm.index)
    avg_importance = avg_importance.sort_values(ascending=False)

    avg_cm = cm_sum / n_runs

    return avg_importance, avg_cm, sorted(list(set(labels)))


def create_metadata_from_data(data):
    """Creates a metadata DataFrame based on column name characters."""

    # Initialize DataFrame with sample IDs from the columns
    metadata = pd.DataFrame({'ID': data.columns})

    # Convert IDs to lowercase for case-insensitive matching
    lower_ids = metadata['ID'].str.lower()

    # Apply conditions: 'c' -> Control, 'v' -> Van, else -> Unknown
    # Note: If a column contains both 'c' and 'v', 'Control' takes precedence.
    metadata['Drug'] = np.where(lower_ids.str.contains('c'), 'Control',
                                np.where(lower_ids.str.contains('v'), 'Van', 'Unknown'))

    # The transform_data and impute_zeros functions require a 'Treatment' column to group replicates.
    # Assigning a single default value so the grouping logic does not crash.
    metadata['Treatment'] = 'Standard'

    return metadata


def run_ribo_driver_analysis(norm_counts, deseq_res, id_to_name, output_dir):
    """
    Identifies driver genes for a 34-gene Ribosomal Gene Programme.
    Calculates a PC1 RiboScore, runs global/partial Spearman correlations,
    integrates DESeq2 DE stats, and outputs a 6-panel summary figure.
    Uses Rank Aggregation (Geometric Mean of Ranks) for final driver scoring.
    """
    print("Running Ribosomal Driver Analysis...")

    # ─── Constants & Gene Sets ───────────────────────────────────────────
    FDR_THRESH = 0.05
    C_CTRL, C_TREAT = "#95a5a6", "#e74c3c"  # Matching your pipeline's palette

    # ─── Group Parsing ───────────────────────────────────────────────────
    ctrl_cols = [c for c in norm_counts.columns if "_c_" in c.lower()]
    treat_cols = [c for c in norm_counts.columns if "_v_" in c.lower()]
    all_cols = ctrl_cols + treat_cols
    group_arr = np.array([0] * len(ctrl_cols) + [1] * len(treat_cols))
    group_col = [C_CTRL] * len(ctrl_cols) + [C_TREAT] * len(treat_cols)

    # ─── Map Genes to IDs ────────────────────────────────────────────────
    name_to_id = {str(v).lower(): k for k, v in id_to_name.items()}
    ribo_ids = [name_to_id[name.lower()] for name in RIBO_NAMES if name.lower() in name_to_id]
    ribo_ids = [rid for rid in ribo_ids if rid in norm_counts.index]

    if len(ribo_ids) < 5:
        print(f"Error: Only {len(ribo_ids)} ribosomal genes found. Skipping driver analysis.")
        return

    # ─── Ribosomal Program Score (PC1) ───────────────────────────────────
    ribo_df = norm_counts.loc[ribo_ids, all_cols]
    ribo_z = ribo_df.values.T
    # ribo_z = StandardScaler().fit_transform(ribo_df.values.T)
    pca = PCA(n_components=1)
    ribo_score = pca.fit_transform(ribo_z)[:, 0]

    # Orient PC1 so high expression = positive score
    if pca.components_[0].mean() < 0:
        ribo_score = -ribo_score

    ribo_score_s = pd.Series(ribo_score, index=all_cols, name="RiboScore")

    ctrl_sc = ribo_score_s[ctrl_cols].values
    treat_sc = ribo_score_s[treat_cols].values
    _, pu = mannwhitneyu(treat_sc, ctrl_sc, alternative="two-sided")

    # ─── Candidate Matrix (Non-Ribo) ─────────────────────────────────────
    cand_ids = [gid for gid in norm_counts.index if gid not in ribo_ids]
    expr_mat = norm_counts.loc[cand_ids, all_cols].values
    n_cand = len(cand_ids)

    # ─── Global & Partial Correlations ───────────────────────────────────
    def group_centre(vec, group):
        out = vec.copy().astype(float)
        for g in np.unique(group):
            mask = group == g
            out[mask] -= out[mask].mean()
        return out

    ribo_resid = group_centre(ribo_score_s.values, group_arr)

    rho_all, pval_all = np.empty(n_cand), np.empty(n_cand)
    rho_part, pval_part = np.empty(n_cand), np.empty(n_cand)

    for i in range(n_cand):
        rho_all[i], pval_all[i] = spearmanr(expr_mat[i], ribo_score_s.values)
        e_resid = group_centre(expr_mat[i], group_arr)
        rho_part[i], pval_part[i] = spearmanr(e_resid, ribo_resid)

    _, qval_all, _, _ = multipletests(np.nan_to_num(pval_all, nan=1.0), method="fdr_bh")
    _, qval_part, _, _ = multipletests(np.nan_to_num(pval_part, nan=1.0), method="fdr_bh")

    # ─── Integrate DESeq2 Results ────────────────────────────────────────
    deseq_aligned = deseq_res.reindex(cand_ids)
    lfc_arr = deseq_aligned['log2FoldChange'].fillna(0).values
    de_qval = deseq_aligned['padj'].fillna(1).values

    # ─── Rank Aggregation (Geometric Mean of Ranks) ──────────────────────
    # Rank from 1 (best) to N (worst).
    # Effect sizes (|rho|): Higher magnitude is better -> rank negative values.
    # Significance (q-value): Lower value is better -> rank positive values.

    # rank_rho_all = rankdata(-np.abs(rho_all))
    rank_q_all = rankdata(qval_all)
    # rank_rho_part = rankdata(-np.abs(rho_part))
    rank_q_part = rankdata(qval_part)
    rank_de_q = rankdata(de_qval)

    # Calculate Geometric Mean of Ranks (Rank Product)
    rank_product = (rank_q_all * rank_q_part * rank_de_q) ** (1.0 / 3.0)
    # rank_product = (rank_rho_all * rank_q_all * rank_rho_part * rank_q_part * rank_de_q) ** (1.0 / 5.0)

    # Invert rank product so a HIGHER composite score = a BETTER driver (for descending sort)
    composite = 1.0 / rank_product

    # ─── Construct Results DataFrame ─────────────────────────────────────
    gnames = [id_to_name.get(gid, gid) for gid in cand_ids]
    pathway_col = [ALL_KNOWN.get(str(g).lower(), "Other") for g in gnames]
    result = pd.DataFrame({
        "ensembl_id": cand_ids,
        "gene_name": gnames,
        "pathway": pathway_col,
        "rho_all": rho_all,
        "q_all": qval_all,
        "rho_partial": rho_part,
        "q_partial": qval_part,
        "log2FC": lfc_arr,
        "de_qval": de_qval,
        "rank_product": rank_product,
        "composite": composite,
    }).sort_values("composite", ascending=False)

    out_tsv = os.path.join(output_dir, "driver_results.tsv")
    result.to_csv(out_tsv, sep="\t", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.right": False, "axes.spines.top": False})

    # =====================================================================
    # PLOT 1: Ribosomal Heatmap
    # =====================================================================
    fig_h, ax_h = plt.subplots(figsize=(10, 8))
    im = ax_h.imshow(ribo_df.values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3))
    ax_h.set_xticks(range(len(all_cols)))
    ax_h.set_xticklabels(all_cols, rotation=45, ha="right", fontsize=9)
    ax_h.set_yticks(range(len(ribo_ids)))
    ax_h.set_yticklabels([id_to_name.get(x, x) for x in ribo_ids], fontsize=8)
    ax_h.set_title("Target Ribosomal Gene Expression (Z-Scored)", fontweight="bold", fontsize=14, pad=15)
    plt.colorbar(im, ax=ax_h, shrink=0.7, label="z-score")

    # Add group color bars at the bottom
    for j, col in enumerate(group_col):
        ax_h.add_patch(plt.Rectangle((j - 0.5, len(ribo_ids) - 0.5), 1, 0.8, color=col, clip_on=False))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot1_ribosomal_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig_h)

    # =====================================================================
    # PLOT 2: RiboScore Barplot
    # =====================================================================
    fig_b, ax_b = plt.subplots(figsize=(9, 6))
    ax_b.bar(np.arange(len(all_cols)), ribo_score_s.values, color=group_col, edgecolor="white", lw=0.5)
    ax_b.axhline(ctrl_sc.mean(), color=C_CTRL, lw=2, ls="--", alpha=0.8, label="Control Mean")
    ax_b.axhline(treat_sc.mean(), color=C_TREAT, lw=2, ls="--", alpha=0.8, label="Treated Mean")
    ax_b.axhline(0, color="black", lw=1)
    ax_b.set_xticks(np.arange(len(all_cols)))
    ax_b.set_xticklabels(all_cols, rotation=45, ha="right", fontsize=10)
    ax_b.set_ylabel("PC1 Score", fontsize=12)
    ax_b.set_title(f"Ribosomal Program Score (MWU p={pu:.4f})", fontweight="bold", fontsize=14)
    ax_b.legend(frameon=False)

    # Inset Boxplot
    ax_bx = ax_b.inset_axes([0.75, 0.65, 0.22, 0.3])
    ax_bx.boxplot([ctrl_sc, treat_sc], patch_artist=True, medianprops=dict(color="black", lw=2),
                  flierprops=dict(marker="o", ms=4))
    for i2, (data, col2) in enumerate(zip([ctrl_sc, treat_sc], [C_CTRL, C_TREAT]), 1):
        ax_bx.scatter(np.random.normal(i2, 0.05, len(data)), data, color=col2, s=30, zorder=3, edgecolors="white",
                      lw=0.5)
    ax_bx.set_xticks([1, 2])
    ax_bx.set_xticklabels(["Control", "Treated"], fontsize=9)
    ax_bx.spines["right"].set_visible(False)
    ax_bx.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot2_ribo_score.pdf"), bbox_inches="tight")
    plt.close(fig_b)

    # =====================================================================
    # PLOT 3: 15-Panel Spotlight Scatters (5 rows x 3 columns)
    # =====================================================================
    fig_s, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()

    for idx, (gname, col, label) in enumerate(SPOTLIGHT):
        ax_s = axes[idx]
        hit = result[result["gene_name"].str.lower() == gname.lower()]

        if not len(hit):
            ax_s.set_title(f"{gname} (Not Found)")
            ax_s.axis("off")
            continue

        row = hit.iloc[0]
        gene_expr = norm_counts.loc[row["ensembl_id"], all_cols].values

        ax_s.scatter(ribo_score_s.values, gene_expr, c=group_col, s=60, edgecolors="white", lw=0.8, zorder=3)
        m, b, *_ = linregress(ribo_score_s.values, gene_expr)
        xs = np.linspace(ribo_score_s.min(), ribo_score_s.max(), 50)
        ax_s.plot(xs, m * xs + b, "k--", lw=1.5, alpha=0.7)

        de_lbl = f"q={row['de_qval']:.3e}" if row['de_qval'] < FDR_THRESH else "ns"
        title_text = f"{row['gene_name']} [{label}]\nρ_all={row['rho_all']:.2f} | ρ_part={row['rho_partial']:.2f}\nLFC={row['log2FC']:.2f} | DE:{de_lbl}"
        ax_s.set_title(title_text, fontsize=10, fontweight="bold", color=col)
        ax_s.set_xlabel("RiboScore", fontsize=9)
        ax_s.set_ylabel("Normalized Expression", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot3_spotlight_scatters.pdf"), bbox_inches="tight")
    plt.close(fig_s)

    # =====================================================================
    # PLOT 4: Global Driver Bubble Plot
    # =====================================================================
    fig_f, ax_f = plt.subplots(figsize=(14, 10))

    result = result[result["de_qval"] < FDR_THRESH]

    # Process colors and sizes for ALL genes
    bubble_colors = result["pathway"].map(lambda p: PATHWAY_COL.get(p, "#D3D3D3")).values
    bubble_sizes = np.clip(np.abs(result["log2FC"]) * 150 + 50, 50, 500)
    edge_colors = ["none" if q < FDR_THRESH else "black" for q in result["de_qval"]]
    z_orders = [5 if p != "Other" else 2 for p in result["pathway"]]

    # Plot all genes
    ax_f.scatter(result["rho_all"], result["rho_partial"],
                 s=bubble_sizes, c=bubble_colors, edgecolors=edge_colors,
                 lw=1.0, alpha=0.7, zorder=3)

    # Annotate only the top 20 drivers by composite score
    top_drivers = result.head(5)
    for _, row in top_drivers.iterrows():
        ax_f.annotate(row["gene_name"], (row["rho_all"], row["rho_partial"]),
                      fontsize=10, fontweight="bold",
                      xytext=(5, 5), textcoords="offset points", zorder=10)

    ax_f.axhline(0, color="black", lw=0.8)
    ax_f.axvline(0, color="black", lw=0.8)
    ax_f.axhline(0.6, color="grey", lw=1, ls="--", alpha=0.5)
    ax_f.axvline(0.6, color="grey", lw=1, ls="--", alpha=0.5)

    ax_f.set_xlabel(f"Spearman ρ — global (q<{FDR_THRESH} threshold ~0.6)", fontsize=12)
    ax_f.set_ylabel("Spearman ρ — within-group", fontsize=12)
    ax_f.set_title("Global Driver Gene with sig DE Summary\n(Size ∝ |log₂FC|)",
                   fontweight="bold", fontsize=14, pad=15)

    # Custom Legend
    handles_pw = [Line2D([0], [0], marker="o", color="w", markerfacecolor=v, markersize=10, label=k)
                  for k, v in PATHWAY_COL.items() if k != "Sig. DE+corr."]
    handles_pw.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#D3D3D3", markersize=10, label="Other"))
    # ax_f.legend(handles=handles_pw, fontsize=10, frameon=True, loc="lower right", title="Pathways")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot4_driver_bubble.pdf"), bbox_inches="tight")
    plt.close(fig_f)

    print(f"Driver analysis complete. Separate plots saved to {output_dir}")

    # # ─── Extract Spotlight Genes for Plot ────────────────────────────────
    # spot_rows = []
    # for gname, col, label in SPOTLIGHT:
    #     hit = result[result["gene_name"].str.lower() == gname.lower()]
    #     if len(hit):
    #         r = hit.iloc[0].to_dict()
    #         r["colour"] = col
    #         r["pathway"] = label
    #         r["display_name"] = gname
    #         spot_rows.append(r)
    # spot_df = pd.DataFrame(spot_rows)

    # # ─── 6-Panel Figure Generation ───────────────────────────────────────
    # plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.right": False, "axes.spines.top": False})
    # fig = plt.figure(figsize=(22, 26))
    # gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.42)
    #
    # # Panel A: Ribo Heatmap
    # ax_h = fig.add_subplot(gs[0, :2])
    # im = ax_h.imshow(ribo_z.T, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3))
    # ax_h.set_xticks(range(len(all_cols)))
    # ax_h.set_xticklabels(all_cols, rotation=45, ha="right", fontsize=7.5)
    # ax_h.set_yticks(range(len(ribo_ids)))
    # ax_h.set_yticklabels([id_to_name.get(x, x) for x in ribo_ids], fontsize=7)
    # ax_h.set_title("A  Target ribosomal gene expression", fontweight="bold", fontsize=12, loc="left")
    # plt.colorbar(im, ax=ax_h, shrink=0.55, label="z-score")
    # for j, col in enumerate(group_col):
    #     ax_h.add_patch(
    #         plt.Rectangle((j - 0.5, -1.85), 1, 0.75, color=col, transform=ax_h.get_xaxis_transform(), clip_on=False))
    #
    # # Panel B: PC1 Barplot
    # ax_b = fig.add_subplot(gs[0, 2])
    # ax_b.bar(np.arange(len(all_cols)), ribo_score_s.values, color=group_col, edgecolor="white", lw=0.4)
    # ax_b.axhline(ctrl_sc.mean(), color=C_CTRL, lw=1.5, ls="--", alpha=0.8)
    # ax_b.axhline(treat_sc.mean(), color=C_TREAT, lw=1.5, ls="--", alpha=0.8)
    # ax_b.axhline(0, color="black", lw=0.8)
    # ax_b.set_xticks(np.arange(len(all_cols)))
    # ax_b.set_xticklabels(all_cols, rotation=45, ha="right", fontsize=7)
    # ax_b.set_title(f"B  Ribosomal Program Score\nMWU p={pu:.3f}", fontweight="bold", fontsize=12, loc="left")
    #
    # ax_bx = ax_b.inset_axes([0.66, 0.60, 0.31, 0.37])
    # ax_bx.boxplot([ctrl_sc, treat_sc], patch_artist=True, medianprops=dict(color="black", lw=2),
    #               flierprops=dict(marker="o", ms=4))
    # for i2, (data, col2) in enumerate(zip([ctrl_sc, treat_sc], [C_CTRL, C_TREAT]), 1):
    #     ax_bx.scatter(np.random.normal(i2, 0.05, len(data)), data, color=col2, s=20, zorder=3, edgecolors="white",
    #                   lw=0.3)
    # ax_bx.set_xticks([1, 2])
    # ax_bx.set_xticklabels(["Ctrl", "Trt"], fontsize=6)
    # ax_bx.spines["right"].set_visible(False)
    # ax_bx.spines["top"].set_visible(False)
    #
    # # Panel C: Global vs Partial Scatter
    # ax_c = fig.add_subplot(gs[1, :2])
    # sig_any = (result["q_all"] < FDR_THRESH) | (result["q_partial"] < FDR_THRESH)
    # ax_c.scatter(result.loc[~sig_any, "rho_all"], result.loc[~sig_any, "rho_partial"], s=3, color="lightgrey",
    #              alpha=0.4)
    # sig_bg = sig_any & ~result["gene_name"].isin([r.get("display_name") for r in spot_rows])
    # ax_c.scatter(result.loc[sig_bg, "rho_all"], result.loc[sig_bg, "rho_partial"], s=6, color="#CCCCCC", alpha=0.6)
    #
    # for _, row in spot_df.iterrows():
    #     ax_c.scatter(row["rho_all"], row["rho_partial"], s=200, color=row["colour"], edgecolors="white", lw=0.8,
    #                  zorder=5, alpha=0.9)
    #     ax_c.annotate(row["display_name"], (row["rho_all"], row["rho_partial"]), fontsize=8, fontweight="bold",
    #                   xytext=(5, 3), textcoords="offset points")
    #
    # ax_c.axhline(0, color="black", lw=0.5);
    # ax_c.axvline(0, color="black", lw=0.5)
    # ax_c.set_xlabel(f"Spearman ρ — global  (q<{FDR_THRESH})")
    # ax_c.set_ylabel("Spearman ρ — within-group")
    # ax_c.set_title("C  Global vs within-group correlation with RiboScore", fontweight="bold", fontsize=12, loc="left")
    #
    # # Panel D: Pathway Bar Chart
    # ax_d = fig.add_subplot(gs[1, 2])
    # pw_stats = []
    # for pw, genes in KNOWN_REGS.items():
    #     sub = result[result["gene_name"].str.lower().isin([g.lower() for g in genes])]
    #     if len(sub):
    #         pw_stats.append({
    #             "pathway": pw, "mean_rho": sub["rho_all"].mean(), "max_rho": sub["rho_all"].max(),
    #             "n_genes": len(sub), "n_sig": ((sub["q_all"] < FDR_THRESH) | (sub["q_partial"] < FDR_THRESH)).sum(),
    #             "col": PATHWAY_COL.get(pw, "grey")
    #         })
    # if pw_stats:
    #     pw_df = pd.DataFrame(pw_stats).sort_values("mean_rho", ascending=True)
    #     ypos = np.arange(len(pw_df))
    #     ax_d.barh(ypos, pw_df["mean_rho"], color=pw_df["col"].values, height=0.55, alpha=0.85)
    #     ax_d.scatter(pw_df["max_rho"], ypos, color=pw_df["col"].values, s=50, marker="|", lw=2, zorder=5)
    #     ax_d.set_yticks(ypos)
    #     ax_d.set_yticklabels([f"{r['pathway']} (n={r['n_genes']})" for _, r in pw_df.iterrows()], fontsize=8)
    #     ax_d.axvline(0, color="black", lw=0.5)
    #     ax_d.set_title("D  Pathway-level correlation summary", fontweight="bold", fontsize=12, loc="left")
    #
    # # Panel E: Spotlight Scatters
    # SHOWCASE = ["Ybx1", "Bop1", "Hey1", "Eif4ebp2", "Tcof1", "Tmem212"]
    # gs_e = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[2, :], hspace=0.65, wspace=0.40)
    # fig.add_subplot(gs[2, :]).axis("off")
    # plt.text(0, 1.05, "E  Individual expression vs RiboScore", fontweight="bold", fontsize=12,
    #          transform=fig.add_subplot(gs[2, :]).transAxes)
    #
    # for idx, gname in enumerate(SHOWCASE):
    #     ax_s = fig.add_subplot(gs_e[idx // 3, idx % 3])
    #     hit = result[result["gene_name"].str.lower() == gname.lower()]
    #     if not len(hit): continue
    #     row = hit.iloc[0]
    #     gene_expr = norm_counts.loc[row["ensembl_id"], all_cols].values
    #
    #     ax_s.scatter(ribo_score_s.values, gene_expr, c=group_col, s=55, edgecolors="white", lw=0.6, zorder=3)
    #     m, b, *_ = linregress(ribo_score_s.values, gene_expr)
    #     xs = np.linspace(ribo_score_s.min(), ribo_score_s.max(), 50)
    #     ax_s.plot(xs, m * xs + b, "k--", lw=1.2, alpha=0.75)
    #
    #     de_lbl = f"q={row['de_qval']:.3f}" if row['de_qval'] < 0.1 else "ns"
    #     ax_s.set_title(
    #         f"{row['gene_name']}\nρ_all={row['rho_all']:.2f}  ρ_within={row['rho_partial']:.2f}\nLFC={row['log2FC']:.2f}  DE:{de_lbl}",
    #         fontsize=7.5, fontweight="bold")
    #
    # # Panel F: Bubble Summary
    # ax_f = fig.add_subplot(gs[3, :])
    # for _, row in spot_df.iterrows():
    #     lfc_size = min(np.abs(row["log2FC"]) * 200 + 80, 600)
    #     de_edge = "black" if row["de_qval"] < FDR_THRESH else "white"
    #     ax_f.scatter(row["rho_all"], row["rho_partial"], s=lfc_size, color=row["colour"], edgecolors=de_edge, lw=1.5,
    #                  zorder=5, alpha=0.88)
    #     ax_f.annotate(row["display_name"], (row["rho_all"], row["rho_partial"]), fontsize=9, fontweight="bold",
    #                   xytext=(6, 4), textcoords="offset points")
    #
    # ax_f.axhline(0, color="black", lw=0.5)
    # ax_f.axvline(0, color="black", lw=0.5)
    # ax_f.set_title("F  Driver gene summary (Bubble size ∝ |log₂FC|, Black border = sig DE)", fontweight="bold",
    #                fontsize=12, loc="left")
    #
    # out_pdf = os.path.join(output_dir, "driver_analysis.pdf")
    # plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    # plt.close()
    # print(f"Driver analysis complete. Results saved to {out_pdf}")


def main():
    # 1. Load Data
    raw_counts, norm_counts, id_to_name = load_and_sanitize_data(RAW_PATH, NORM_PATH,
                                                                 to_drop=["24h_v_21_S9", "24h_c_16_S4"])
    metadata = create_metadata_from_data(norm_counts)
    # transform and z-score by controls
    norm_counts, processed_metadata = transform_data(norm_counts, metadata, run_type='_rat_neurons')

    # # 2. Run PCA Analysis (NEW)
    # run_pca_analysis(norm_counts, id_to_name)
    #
    # # # 3. Check Specific Outliers (NEW)
    # # # check_specific_outliers(norm_counts, id_to_name, z_threshold=100)
    # # check_specific_outliers(norm_counts, id_to_name, z_threshold=3)
    # # return
    #
    # run_rf_full(id_to_name, norm_counts)

    # 2. Run DESeq2 (The statistical engine)
    deseq_res = run_deseq2_analysis(raw_counts, id_to_name)

    # # 3. Generate GSEA/Enrichr Files (Using DESeq2 stats + Norm expression)
    # generate_gsea_enrichr_files(norm_counts, deseq_res, id_to_name)

    # # 4. Run Random Forest (The ML engine)
    # # Prepare labels for RF
    # labels = [0 if '_c_' in col else 1 for col in norm_counts.columns]
    # run_random_forest(norm_counts, labels)

    run_ribo_driver_analysis(norm_counts, deseq_res, id_to_name, OUTPUT_DIR)

    print(f"Pipeline Complete. Check results in: {OUTPUT_DIR}")


def run_rf_full(id_to_name, norm_counts):
    # For this example, let's assume 'norm_counts' is already loaded:
    df = norm_counts.copy()
    # B. MAKE 'CONDITIONS' (LABELS)
    # OPTION 1: Automatic (if sample names contain the group name)
    # Example: "Control_1", "Control_2", "Van_1", "Van_2"
    conditions = ['Van' if 'v' in col.lower() else 'Control' for col in df.columns]
    # OPTION 2: Manual (if names are obscure like "Sample1", "Sample2")
    # Ensure the order matches df.columns exactly!
    # conditions = ['Without Van', 'Without Van', 'Without Van', 'With Van', 'With Van', 'With Van']
    # Create a color palette for plotting later
    condition_colors = dict(zip(['Control', 'Van'], ['#95a5a6', '#e74c3c']))  # Grey vs Red
    print(f"Data Shape: {df.shape}")
    print(f"Conditions: {conditions}")
    # ==========================================
    # 2. RUN ROBUST RANDOM FOREST (1000 Runs)
    # ==========================================
    # RUN IT
    # avg_imp, avg_cm, class_names = run_robust_rf(df, conditions, n_runs=1)
    avg_imp, avg_cm, class_names = run_robust_rf(df, conditions, n_runs=1_000)
    df_imp = pd.DataFrame(avg_imp)
    df_imp['gene name'] = df_imp.index.map(id_to_name)
    df_imp = df_imp.reset_index().set_index("gene name")
    df_imp.columns = ["gene id", "importance"]
    df_imp = df_imp[["importance", "gene id"]]
    df_imp.to_csv(BASE_PATH + "/Analysis_Results/importance.csv")
    pd.DataFrame(avg_cm).to_csv(BASE_PATH + "/Analysis_Results/confusion.csv")
    # ==========================================
    # 3. PLOT 1: AVERAGE CONFUSION MATRIX
    # ==========================================
    plt.figure(figsize=(5, 4))
    sns.heatmap(avg_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Avg Confusion Matrix (1000 Runs)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(BASE_PATH + "/Analysis_Results/confusion_matrix.png")
    plt.show()
    # ==========================================
    # 4. PLOT 2: HEATMAPS (Top 20, 50, 100)
    # ==========================================
    # Run the plotting for 20, 50, and 100
    for n in [20, 50, 100, 200, 400, 800]:
        plot_top_genes(df, conditions, avg_imp, id_to_name, n_top=n, color_dict=condition_colors)
    # return


if __name__ == "__main__":
    main()
