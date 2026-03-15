import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.special import logit
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os


# =============================================================================
# Helper Functions for Plotting
# =============================================================================

def get_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Create an ellipse patch representing the covariance of x and y."""
    if len(x) < 2: return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_pcoa_results(distance_matrix, metadata, analysis_title, color_group='antibiotic'):
    """A streamlined function to perform PCoA and plot the results."""
    pca = PCA(n_components=2)
    pcoa_result = pca.fit_transform(distance_matrix.values)

    pcoa_df = pd.DataFrame(pcoa_result, index=distance_matrix.index, columns=['PCoA1', 'PCoA2'])
    merged_pcoa_df = pd.merge(metadata, pcoa_df, left_on='#SampleID', right_index=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Use seaborn for easier plotting with automatic color handling
    sns.scatterplot(data=merged_pcoa_df, x='PCoA1', y='PCoA2', hue=color_group, style=color_group, s=100, ax=ax)

    # Add ellipses for each group
    for name, group in merged_pcoa_df.groupby(color_group):
        get_ellipse(group['PCoA1'], group['PCoA2'], ax, alpha=0.2)

    ax.set_title(f'PCoA Plot ({analysis_title})', fontsize=16)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(title=color_group, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


# =============================================================================
# Transformation and Distance Functions
# =============================================================================

def calculate_aitchison_distance(counts_df):
    """Calculates Aitchison distance. Expects samples as COLUMNS."""
    data = counts_df.T  # Transpose to have samples as rows for processing
    pseudocount = 1e-6
    data_plus_pseudo = data + pseudocount
    geom_mean = np.exp(np.mean(np.log(data_plus_pseudo), axis=1))
    clr_data = np.log(data_plus_pseudo.div(geom_mean, axis=0))
    dist_array = pdist(clr_data, metric='euclidean')
    return pd.DataFrame(squareform(dist_array), index=data.index, columns=data.index)


def calculate_arcsin_sqrt_distance(counts_df):
    """Calculates Euclidean distance on arcsin-sqrt transformed proportions."""
    data = counts_df.T
    pseudocount = 1e-6
    data_plus_pseudo = data + pseudocount
    proportions = data_plus_pseudo.div(data_plus_pseudo.sum(axis=1), axis=0)
    transformed_data = np.arcsin(np.sqrt(proportions))
    dist_array = pdist(transformed_data, metric='euclidean')
    return pd.DataFrame(squareform(dist_array), index=data.index, columns=data.index)


def calculate_log_transform_distance(counts_df):
    """Calculates Euclidean distance on log-transformed counts."""
    data = counts_df.T
    pseudocount = 1
    data_plus_pseudo = data + pseudocount
    transformed_data = np.log(data_plus_pseudo)
    dist_array = pdist(transformed_data, metric='euclidean')
    return pd.DataFrame(squareform(dist_array), index=data.index, columns=data.index)


def calculate_logit_transform_distance(counts_df):
    """Calculates Euclidean distance on logit-transformed proportions."""
    data = counts_df.T
    pseudocount = 1e-6
    data_plus_pseudo = data + pseudocount
    proportions = data_plus_pseudo.div(data_plus_pseudo.sum(axis=1), axis=0)
    transformed_data = logit(proportions)
    dist_array = pdist(transformed_data, metric='euclidean')
    return pd.DataFrame(squareform(dist_array), index=data.index, columns=data.index)


# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == '__main__':
    # --- Define which treatment groups to analyze ---
    # Your script prepared files for each of these groups.
    groups_to_analyze = ["IV", "IP", "PO", "amp", "van", "met", "neo", "mix"]

    # --- Load the main metadata file once ---
    try:
        main_metadata = pd.read_csv('../Data/QIIME/qiime_metadata.tsv', sep='\t')
    except FileNotFoundError:
        print("ERROR: Could not find '../Data/QIIME/qiime_metadata.tsv'.")
        print("Please ensure the metadata file is in the correct subfolder.")
        exit()

    # --- Loop through each group, load its data, and run all analyses ---
    for group_name in groups_to_analyze:
        print(f"\n{'=' * 50}")
        print(f"Processing group: {group_name}")
        print(f"{'=' * 50}")

        file_path = f'prepared_feature_tables/{group_name}.tsv'

        try:
            # Load the pre-processed data for the current group
            # The .T transposes the data so that samples are rows and features are columns
            group_data = pd.read_csv(file_path, sep='\t', index_col="#OTU ID").T
            print(f"Successfully loaded {file_path}. Found {group_data.shape[0]} samples.")
        except FileNotFoundError:
            print(f"WARNING: Could not find file {file_path}. Skipping this group.")
            continue  # Skip to the next group

        # Filter the main metadata to only the samples in this group's data file
        filtered_metadata = main_metadata[main_metadata['#SampleID'].isin(group_data.index)]

        # The distance functions expect samples as COLUMNS.
        # Our loaded `group_data` has samples as ROWS, so we transpose it back.
        counts_for_distance = group_data.T

        # --- Calculate all distance matrices ---
        aitchison_dist = calculate_aitchison_distance(counts_for_distance)
        arcsin_dist = calculate_arcsin_sqrt_distance(counts_for_distance)
        log_dist = calculate_log_transform_distance(counts_for_distance)
        logit_dist = calculate_logit_transform_distance(counts_for_distance)

        # --- Generate all PCoA plots for the current group ---
        print("Generating PCoA plots for this group...")
        plot_pcoa_results(aitchison_dist, filtered_metadata, f"{group_name} - Aitchison")
        plot_pcoa_results(arcsin_dist, filtered_metadata, f"{group_name} - Arcsin-Sqrt")
        plot_pcoa_results(log_dist, filtered_metadata, f"{group_name} - Log Transform")
        plot_pcoa_results(logit_dist, filtered_metadata, f"{group_name} - Logit Transform")

    print("\nAll analyses complete.")