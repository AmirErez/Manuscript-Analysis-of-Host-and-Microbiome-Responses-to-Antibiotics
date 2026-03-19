import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ClusteringGO import build_tree, calculate_correlation, transform_data
from clusters_plot import plot_categories, plot_correlation_gsea
# Try importing pydeseq2, else warn user
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm  # For progress bar, optional but recommended


# Define mitochondrial genes list (or import it from your config)
# This is required for the filtering step.
# Example placeholder list based on common mouse mt-genes:
MITOCHONDRIAL_GENES = {
    'mt-nd1', 'mt-nd2', 'mt-co1', 'mt-co2', 'mt-atp8',
    'mt-atp6', 'mt-co3', 'mt-nd3', 'mt-nd4l', 'mt-nd4',
    'mt-nd5', 'mt-nd6', 'mt-cytb'
}
# Settings for publication-quality figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['svg.fonttype'] = 'none'


def plot_interference_noise_dashboard(pair_name, interference_path, noise_dist_path, noisy_genes_path):
    # --- Set Publishable Global Font Sizes ---
    plt.rcParams.update({
        'font.size': 14,  # Global font size
        'axes.titlesize': 16,  # Title size
        'axes.labelsize': 14,  # X/Y label size
        'xtick.labelsize': 12,  # X tick size
        'ytick.labelsize': 12,  # Y tick size
        'legend.fontsize': 12,  # Legend text size
        'figure.titlesize': 18  # Suptitle size
    })

    # --- Load Data ---
    # 1. Interference
    try:
        int_df = pd.read_csv(interference_path, index_col=0)
    except FileNotFoundError:
        print(f"No interference file for {pair_name}")
        int_df = pd.DataFrame()

    # 2. Global Noise
    try:
        dist_df = pd.read_csv(noise_dist_path, index_col=0)
    except FileNotFoundError:
        print(f"No global noise file for {pair_name}")
        dist_df = pd.DataFrame()

    # 3. Gene Noise
    try:
        noise_df = pd.read_csv(noisy_genes_path, index_col=0)
    except FileNotFoundError:
        print(f"No gene noise file for {pair_name}")
        noise_df = pd.DataFrame()

    # Create output directory if it doesn't exist
    out_dir = "./Private/Noise"
    os.makedirs(out_dir, exist_ok=True)

    safe_name = pair_name.replace('+', '_')

    # ==========================================
    # --- Plot 1: Interference Volcano ---
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    if not int_df.empty:
        # Create colors based on significance and direction
        conditions = [
            (int_df['padj'] < 0.05) & (int_df['log2FoldChange'] > 0),
            (int_df['padj'] < 0.05) & (int_df['log2FoldChange'] < 0)
        ]
        choices = ['Synergy', 'Antagonism']
        int_df['Type'] = np.select(conditions, choices, default='NS')

        # Plot NS
        sns.scatterplot(data=int_df[int_df['Type'] == 'NS'], x='log2FoldChange', y='padj',
                        ax=ax1, color='lightgrey', alpha=0.5, s=30, edgecolor=None)

        # Plot Sig
        sns.scatterplot(data=int_df[int_df['Type'] != 'NS'], x='log2FoldChange', y='padj',
                        ax=ax1, hue='Type', palette={'Synergy': '#d62728', 'Antagonism': '#1f77b4'},
                        alpha=0.8, s=60)

        # Log scale y-axis for p-values
        ax1.set_yscale('log')
        ax1.invert_yaxis()  # Small p-values at top
        ax1.set_title(f"Interference: {pair_name}\n(Deviation from Additivity)", fontweight='bold')
        ax1.set_xlabel("Interaction LFC\n(>0 = Pair is stronger than sum)")
        ax1.set_ylabel("Adjusted P-value")
        ax1.axvline(0, linestyle='--', color='black', linewidth=1)

        # Annotate top genes (Increased fontsize to 12)
        # top_genes = int_df.sort_values('padj').head(5)
        # for idx, row in top_genes.iterrows():
        #     ax1.text(row['log2FoldChange'], row['padj'], row['gene_name'], fontsize=12)
        from adjustText import adjust_text

        # Annotate top genes
        top_genes = int_df.sort_values('padj').head(5)
        texts = []
        for idx, row in top_genes.iterrows():
            # Append the text object to a list, but do not rely on it staying exactly here
            texts.append(ax1.text(row['log2FoldChange'], row['padj'], row['gene_name'], fontsize=12))

        # Automatically adjust text positions to prevent overlap
        adjust_text(texts, ax=ax1,
                    # arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
                    expand_points=(1.5, 1.5))  # Adds a bit of padding around the original points
    else:
        ax1.text(0.5, 0.5, "No Significant Interference", ha='center', va='center', fontsize=16)

    plt.tight_layout()
    # Save as PDF (Vectorized)
    out_file1 = f"{out_dir}/Interference_Volcano_{safe_name}.pdf"
    fig1.savefig(out_file1, format='pdf', bbox_inches='tight')
    out_file1_svg = f"{out_dir}/Interference_Volcano_{safe_name}.svg"
    fig1.savefig(out_file1_svg, format='svg', bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {out_file1}")

    # ==========================================
    # --- Plot 2: Global Noise (Distance to Centroid) ---
    # ==========================================
    if not dist_df.empty:
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        groups = dist_df['Group'].unique()
        groups = sorted(list(groups), key=lambda x: (x != 'PBS', x == pair_name))

        sns.boxplot(data=dist_df, x='Group', y='DistToCentroid', order=groups, ax=ax2, palette="Set2")
        sns.swarmplot(data=dist_df, x='Group', y='DistToCentroid', order=groups, ax=ax2, color=".25", size=6)

        ax2.set_title(f"Global Transcriptional Stability\n(Inter-replicate Heterogeneity)", fontweight='bold')
        ax2.set_ylabel("Euclidean Distance to Group Centroid")
        ax2.set_xlabel("")
        plt.xticks(rotation=45)  # Rotate just in case labels overlap with bigger fonts

        plt.tight_layout()
        out_file2 = f"{out_dir}/Global_Noise_{safe_name}.pdf"
        fig2.savefig(out_file2, format='pdf', bbox_inches='tight')
        out_file2_svg = f"{out_dir}/Global_Noise_{safe_name}.svg"
        fig2.savefig(out_file2_svg, format='svg', bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {out_file2}")

    # ==========================================
    # --- Plot 3: Gene-Specific Noise Landscape ---
    # ==========================================
    if not noise_df.empty:
        fig3, ax3 = plt.subplots(figsize=(8, 6))

        top_noisy = noise_df.head(10)

        sns.scatterplot(data=noise_df, x='Met', y='NoiseRatio', ax=ax3,
                        color='grey', alpha=0.4, s=40, label='All Genes')

        # Highlight
        sns.scatterplot(data=top_noisy, x='Met', y='NoiseRatio', ax=ax3,
                        color='red', s=80, label='Top Variable')

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_title(f"Gene-Specific Noise Induction\n({pair_name} vs Singles)", fontweight='bold')
        ax3.set_xlabel("Baseline Variance (Proxy for Expression)")
        ax3.set_ylabel("Noise Ratio (Pair Var / Single Var)")
        ax3.axhline(1, linestyle='--', color='black', linewidth=1)

        # Annotate (Increased fontsize to 11 to fit potentially crowded spaces)
        for idx, row in top_noisy.iterrows():
            ax3.text(row['Met'], row['NoiseRatio'], row['gene_name'], fontsize=11)

        plt.tight_layout()
        out_file3 = f"{out_dir}/Gene_Noise_{safe_name}.pdf"
        fig3.savefig(out_file3, format='pdf', bbox_inches='tight')
        plt.close(fig3)
        print(f"Saved: {out_file3}")
# def plot_interference_noise_dashboard(pair_name, interference_path, noise_dist_path, noisy_genes_path):
#     # --- Load Data ---
#     # 1. Interference
#     try:
#         int_df = pd.read_csv(interference_path, index_col=0)
#     except FileNotFoundError:
#         print(f"No interference file for {pair_name}")
#         int_df = pd.DataFrame()
#
#     # 2. Global Noise
#     dist_df = pd.read_csv(noise_dist_path, index_col=0)
#
#     # 3. Gene Noise
#     noise_df = pd.read_csv(noisy_genes_path, index_col=0)
#
#     # --- Setup Canvas ---
#     fig = plt.figure(figsize=(18, 6), constrained_layout=True)
#     gs = fig.add_gridspec(1, 3)
#     ax1 = fig.add_subplot(gs[0, 0])  # Volcano
#     ax2 = fig.add_subplot(gs[0, 1])  # Global Dist
#     ax3 = fig.add_subplot(gs[0, 2])  # Noise Scatter
#
#     # --- Plot 1: Interference Volcano ---
#     if not int_df.empty:
#         # Create colors based on significance and direction
#         conditions = [
#             # (int_df['padj_Interaction'] < 0.05) & (int_df['LFC_Interaction'] > 0),
#             (int_df['padj'] < 0.05) & (int_df['log2FoldChange'] > 0),
#             # (int_df['padj_Interaction'] < 0.05) & (int_df['LFC_Interaction'] < 0)
#             (int_df['padj'] < 0.05) & (int_df['log2FoldChange'] < 0)
#         ]
#         choices = ['Synergy/Amplified', 'Buffering/Antagonism']
#         int_df['Type'] = np.select(conditions, choices, default='NS')
#
#         # Plot NS
#         sns.scatterplot(data=int_df[int_df['Type'] == 'NS'], x='log2FoldChange', y='padj',
#                         # sns.scatterplot(data=int_df[int_df['Type'] == 'NS'], x='LFC_Interaction', y='padj_Interaction',
#                         ax=ax1, color='lightgrey', alpha=0.5, s=15, edgecolor=None)
#
#         # Plot Sig
#         sns.scatterplot(data=int_df[int_df['Type'] != 'NS'], x='log2FoldChange', y='padj',
#                         # sns.scatterplot(data=int_df[int_df['Type'] != 'NS'], x='LFC_Interaction', y='padj_Interaction',
#                         ax=ax1, hue='Type', palette={'Synergy/Amplified': '#d62728', 'Buffering/Antagonism': '#1f77b4'},
#                         alpha=0.8, s=40)
#
#         # Log scale y-axis for p-values
#         ax1.set_yscale('log')
#         ax1.invert_yaxis()  # Small p-values at top
#         ax1.set_title(f"Interference: {pair_name}\n(Deviation from Additivity)", fontsize=12, fontweight='bold')
#         ax1.set_xlabel("Interaction LFC\n(>0 = Pair is stronger than sum)")
#         ax1.set_ylabel("Adjusted P-value")
#         ax1.axvline(0, linestyle='--', color='black', linewidth=0.8)
#
#         # Annotate top genes
#         top_genes = int_df.sort_values('padj').head(5)
#         # top_genes = int_df.sort_values('padj_Interaction').head(5)
#         for idx, row in top_genes.iterrows():
#             ax1.text(row['log2FoldChange'], row['padj'], row['gene_name'], fontsize=9)
#
#     else:
#         ax1.text(0.5, 0.5, "No Significant Interference", ha='center', va='center')
#
#     # --- Plot 2: Global Noise (Distance to Centroid) ---
#     # Order: PBS, Single A, Single B, Pair
#     # We infer order from data present
#     groups = dist_df['Group'].unique()
#     # Simple sort to try to keep Pair last or PBS first
#     groups = sorted(list(groups), key=lambda x: (x != 'PBS', x == pair_name))
#
#     sns.boxplot(data=dist_df, x='Group', y='DistToCentroid', order=groups, ax=ax2, palette="Set2")
#     sns.swarmplot(data=dist_df, x='Group', y='DistToCentroid', order=groups, ax=ax2, color=".25", size=4)
#
#     ax2.set_title(f"Global Transcriptional Stability\n(Inter-replicate Heterogeneity)", fontsize=12, fontweight='bold')
#     ax2.set_ylabel("Euclidean Distance to Group Centroid")
#     ax2.set_xlabel("")
#
#     # --- Plot 3: Gene-Specific Noise Landscape ---
#     # We want to plot NoiseRatio vs Mean Expression
#     # (Assuming we have Mean Expression in the Noise DF or need to merge it back.
#     # The previous script didn't save Mean in Noisy_Genes, but let's assume we use what we have or 'var_Met' as proxy for abundance)
#
#     # If 'Mean' column missing, we can infer abundance roughly from variance (since Mean ~ Var in Poisson),
#     # but ideally you'd add 'mean_expr' to the save step in the previous script.
#     # For now, let's plot Ratio vs Variance (as proxy for expression magnitude)
#
#     # Highlight top noisy genes
#     top_noisy = noise_df.head(10)
#
#     sns.scatterplot(data=noise_df, x='Met', y='NoiseRatio', ax=ax3,
#                     color='grey', alpha=0.4, s=20, label='All Genes')
#
#     # Highlight
#     sns.scatterplot(data=top_noisy, x='Met', y='NoiseRatio', ax=ax3,
#                     color='red', s=50, label='Top Variable')
#
#     ax3.set_xscale('log')
#     ax3.set_yscale('log')
#     ax3.set_title(f"Gene-Specific Noise Induction\n({pair_name} vs Singles)", fontsize=12, fontweight='bold')
#     ax3.set_xlabel("Baseline Variance (Proxy for Expression)")
#     ax3.set_ylabel("Noise Ratio (Pair Var / Single Var)")
#     ax3.axhline(1, linestyle='--', color='black')
#
#     # Annotate
#     for idx, row in top_noisy.iterrows():
#         ax3.text(row['Met'], row['NoiseRatio'], row['gene_name'], fontsize=8)
#
#     plt.suptitle(f"Analysis Dashboard: {pair_name}", fontsize=16)
#     output_file = f"./Private/Noise/Dashboard_{pair_name.replace('+', '_')}.png"
#     plt.savefig(output_file, dpi=300)
#     print(f"Dashboard saved to {output_file}")
    # plt.show()


# def read_data_metadata():
#     base_path = '/Users/yonchlevin/Desktop/ErezLab/MouseAbxBel/Git/Data/Pairs/'
#     metadata_path = base_path + 'metadata.tsv'
#     metadata_df = pd.read_csv(metadata_path, sep='\t')
#     metadata_df["Sample"] = metadata_df["Sample"].str.replace("-", "").str.replace(" #", "_")
#     metadata_df["ID"] = metadata_df["Sample"]
#     metadata_df["Treatment"] = "?"
#     # metadata_df["Antibiotic"] = metadata_df["Drug"]
#     data_path = base_path + 'genes_norm_named-newRNAseq.tsv'
#     data_df = pd.read_csv(data_path, sep='\t')
#     id_to_name = dict(zip(data_df['gene_id'], data_df['gene_name']))
#     data_df = data_df.set_index('gene_id').drop(columns=['gene_name'])
#     data_df.columns = [f"{col.split('_')[0]}_{col.split('_')[1]}" for col in data_df.columns]
#     # ensure metadata samples are in data columns
#     assert all(sample in data_df.columns for sample in
#                metadata_df['Sample']), "Some samples in metadata are not in data columns"
#     return data_df, metadata_df, id_to_name

def significant(row, abx_data, pbs_data):
    from scipy.stats import ttest_ind
    # get treat-test score for the gene
    abx = (row[abx_data['ID']])
    pbs = (row[pbs_data['ID']])
    t_pbs, t_p_pbs = ttest_ind(pbs, abx)
    return t_p_pbs


def compute_all_genes_statistics():
    data, metadata, id_to_name = read_data_metadata(remove_mitochondrial=False)
    antibiotics = metadata['Drug'].unique().tolist()
    antibiotics.remove('PBS')
    treatments = metadata['Treatment'].unique().tolist()
    run_type = "_pairs"
    # transcriptome, metadata = transform_data(transcriptome, metadata, "RASflow")
    # ensmus_to_gene = get_ensmus_dict()
    # # change index using this dictionary
    # transcriptome.index = [ensmus_to_gene[gene] for gene in transcriptome.index]
    metadata["Category"] = metadata['Drug'] + "_" + metadata['Treatment']
    for treat in treatments:
        for abx in antibiotics:
            print(f"prepare .res and .cls for {abx} {treat}")
            samples_meta = metadata[
                ((metadata["Drug"] == abx) | (metadata["Drug"] == "PBS")) & (metadata["Treatment"] == treat)]
            # sort samples, put all Drug=PBS first
            samples_meta = samples_meta.sort_values(by='Drug', key=lambda x: x != 'PBS')
            curr = data[samples_meta["ID"]]
            abx_data = metadata[(metadata['Drug'] == abx) & (metadata["Treatment"] == treat)]
            pbs_data = metadata[(metadata['Drug'] == 'PBS') & (metadata["Treatment"] == treat)]
            # # reorder curr: first pbs_data then abx_data
            # samples = pd.concat([pbs_data["ID"], abx_data["ID"]])
            # curr = transcriptome[samples]
            # make a column named DESCRIPTION and put na in it, order it to be second column
            curr.insert(0, "DESCRIPTION", "na")
            # rename index column name to NAME
            curr = curr.reset_index().rename(columns={"gene_name": "NAME"})
            curr.to_csv(f"./Private/GO_enrichment/{abx}_{treat}_GSEA.txt", sep="\t", index=False)
            # curr.to_csv(f"./Private/GO_enrichment/{abx}_{treat}_GSEA_noC9C10C18.txt", sep="\t", index=False)
            num_samples = len(samples_meta)
            num_classes = len(samples_meta["Drug"].unique())
            class_names = samples_meta["Category"].unique()
            # class_labels = samples_meta["Category"].values
            class_labels = samples_meta["Category"].apply(lambda x: class_names.tolist().index(x))
            # Ensure the number of class labels matches the number of samples
            if len(class_labels) != num_samples:
                raise ValueError("The number of class labels must match the number of samples.")
            # Create CLS file content
            cls_content = []
            # First line
            cls_content.append(f"{num_samples} {num_classes} 1")
            # Second line (class names)
            cls_content.append(f"# {' '.join(class_names)}")
            # Third line (class labels)
            cls_content.append(' '.join(map(str, class_labels)))

            # Write to CLS file
            # with open(f"./Private/GO_enrichment/{abx}_{treat}_GSEA_noC9C10C18.cls", "w") as cls_file:
            with open(f"./Private/GO_enrichment/{abx}_{treat}_GSEA.cls", "w") as cls_file:
                cls_file.write("\n".join(cls_content))

    for treat in treatments:
        for abx in antibiotics:
            print(f"ttest for {abx} {treat}")
            samples_meta = metadata[
                ((metadata["Drug"] == abx) | (metadata["Drug"] == "PBS")) & (metadata["Treatment"] == treat)]
            curr = data[samples_meta["ID"]]
            abx_data = metadata[(metadata['Drug'] == abx) & (metadata["Treatment"] == treat)]
            pbs_data = metadata[(metadata['Drug'] == 'PBS') & (metadata["Treatment"] == treat)]
            # Apply the function to each row and store the results in a new DataFrame
            results = curr.apply(lambda row: significant(row, abx_data, pbs_data), axis=1)
            # Create a new DataFrame with index and the -log(results)
            result_df = pd.DataFrame({'index': curr.index, 'log_p_value': -np.log2(results)})
            # If you want to set the original index back
            result_df.set_index('index', inplace=True)
            # sort by p_value, highest first
            result_df = result_df.sort_values(by="log_p_value", ascending=False)
            # drop rows with nan values
            result_df = result_df.dropna()
            # merge rows with the same index, keep the higher value on the column
            result_df = result_df.groupby(result_df.index).max()
            # delete second columns name
            result_df.columns = ['']
            result_df.to_csv(f"./Private/GO_enrichment/{abx}_{treat}_GSEA.rnk", sep="\t")

def compute_all_genes_statistics_pairs(controls):
    data, metadata, id_to_name = read_data_metadata(remove_mitochondrial=False)
    antibiotics = metadata['Drug'].unique().tolist()
    antibiotics.remove('PBS')
    for control in controls:
        antibiotics.remove(control)
    treatments = metadata['Treatment'].unique().tolist()
    run_type = "_pairs"
    # transcriptome, metadata = transform_data(transcriptome, metadata, "RASflow")
    # ensmus_to_gene = get_ensmus_dict()
    # # change index using this dictionary
    # transcriptome.index = [ensmus_to_gene[gene] for gene in transcriptome.index]
    metadata["Category"] = metadata['Drug'] + "_" + metadata['Treatment']
    treat = "PO"
    for control in controls:
        for abx in antibiotics:
            print(f"prepare .res and .cls for {abx} {treat}")
            samples_meta = metadata[
                ((metadata["Drug"] == abx) | (metadata["Drug"] == control)) & (metadata["Treatment"] == treat)]
            # sort samples, put all Drug=PBS first
            samples_meta = samples_meta.sort_values(by='Drug', key=lambda x: x != control)
            curr = data[samples_meta["ID"]]
            abx_data = metadata[(metadata['Drug'] == abx) & (metadata["Treatment"] == treat)]
            pbs_data = metadata[(metadata['Drug'] == control) & (metadata["Treatment"] == treat)]
            # # reorder curr: first pbs_data then abx_data
            # samples = pd.concat([pbs_data["ID"], abx_data["ID"]])
            # curr = transcriptome[samples]
            # make a column named DESCRIPTION and put na in it, order it to be second column
            curr.insert(0, "DESCRIPTION", "na")
            # rename index column name to NAME
            curr = curr.reset_index().rename(columns={"gene_name": "NAME"})
            curr.to_csv(f"./Private/GO_enrichment/{abx}_{control}_GSEA.txt", sep="\t", index=False)
            # curr.to_csv(f"./Private/GO_enrichment/{abx}_{treat}_GSEA_noC9C10C18.txt", sep="\t", index=False)
            num_samples = len(samples_meta)
            num_classes = len(samples_meta["Drug"].unique())
            class_names = samples_meta["Category"].unique()
            # class_labels = samples_meta["Category"].values
            class_labels = samples_meta["Category"].apply(lambda x: class_names.tolist().index(x))
            # Ensure the number of class labels matches the number of samples
            if len(class_labels) != num_samples:
                raise ValueError("The number of class labels must match the number of samples.")
            # Create CLS file content
            cls_content = []
            # First line
            cls_content.append(f"{num_samples} {num_classes} 1")
            # Second line (class names)
            cls_content.append(f"# {' '.join(class_names)}")
            # Third line (class labels)
            cls_content.append(' '.join(map(str, class_labels)))

            # Write to CLS file
            # with open(f"./Private/GO_enrichment/{abx}_{treat}_GSEA_noC9C10C18.cls", "w") as cls_file:
            with open(f"./Private/GO_enrichment/{abx}_{control}_GSEA.cls", "w") as cls_file:
                cls_file.write("\n".join(cls_content))

    for treat in treatments:
        for abx in antibiotics:
            print(f"ttest for {abx} {treat}")
            samples_meta = metadata[
                ((metadata["Drug"] == abx) | (metadata["Drug"] == control)) & (metadata["Treatment"] == treat)]
            curr = data[samples_meta["ID"]]
            abx_data = metadata[(metadata['Drug'] == abx) & (metadata["Treatment"] == treat)]
            pbs_data = metadata[(metadata['Drug'] == control) & (metadata["Treatment"] == treat)]
            # Apply the function to each row and store the results in a new DataFrame
            results = curr.apply(lambda row: significant(row, abx_data, pbs_data), axis=1)
            # Create a new DataFrame with index and the -log(results)
            result_df = pd.DataFrame({'index': curr.index, 'log_p_value': -np.log2(results)})
            # If you want to set the original index back
            result_df.set_index('index', inplace=True)
            # sort by p_value, highest first
            result_df = result_df.sort_values(by="log_p_value", ascending=False)
            # drop rows with nan values
            result_df = result_df.dropna()
            # merge rows with the same index, keep the higher value on the column
            result_df = result_df.groupby(result_df.index).max()
            # delete second columns name
            result_df.columns = ['']
            result_df.to_csv(f"./Private/GO_enrichment/{abx}_{control}_GSEA.rnk", sep="\t")


def run_corrEnrich(data, metadata, run_type, antibiotics, treatments, id_to_name):
    # compute_all_genes_statistics()

    # tree, tree_size = build_tree(True)
    # make any value smaller than log10(5) to be 0 todo
    # data[data < np.log10(1)] = 0
    # corr = calculate_correlation(tree, data, metadata, tree_size, antibiotics, treatments, "H2-Ab1",

    # corr = calculate_correlation(data, metadata, antibiotics, treatments, "H2-Ab1",
    #                              f"diff_abx{run_type}", 'Treatment', id_to_name)

    # convert results from /Private/PairsCorrEnrichResults to /Private/clusters_properties/diff_abx" + run_type
    convert = True
    # convert = False
    if convert:
        dir_path = "./Private/PairsCorrEnrichResults"
        # Non-recursive (efficient)
        with os.scandir(dir_path) as it:
            for file in it:
                if file.is_file() and file.name.endswith(".tsv"):
                    temp = pd.read_csv(os.path.join("./Private/PairsCorrEnrichResults", file.name), sep="\t")
                    temp["fdr GO significance"] = fdrcorrection(temp["GO_Significance"])[1]

                    temp["fdr correlation"] = np.nan
                    filtered_p_values = \
                        temp[(temp["fdr GO significance"] < 0.05) & temp["Correlation_PValue"].notna()][
                            "Correlation_PValue"]
                    # Apply FDR correction to the filtered p-values
                    fdr_corrected = fdrcorrection(filtered_p_values.to_list())[1]
                    # temp["fdr correlation"] = fdrcorrection(temp["p-value correlation"])[1]
                    temp.loc[(temp["fdr GO significance"] < 0.05) & temp[
                        "Correlation_PValue"].notna(), "fdr correlation"] = fdr_corrected
                    # temp["fdr t-test"] = fdrcorrection(temp["treat-test p-value"])[1]
                    # Filter the rows where p-value correlation is less than 0.05
                    filtered_p_values = temp[(temp["fdr correlation"] < 0.05) & temp["MWU_PValue"].notna()][
                        "MWU_PValue"]
                    # Apply FDR correction to the filtered p-values
                    fdr_corrected = fdrcorrection(filtered_p_values)[1]
                    # Create a new column with NaN values
                    temp["fdr median t-test"] = np.nan
                    # Assign the FDR corrected values back to the DataFrame
                    # temp.loc[temp["fdr correlation"] < 0.05, "fdr median t-test"] = fdr_corrected
                    temp.loc[(temp["fdr correlation"] < 0.05) & temp["MWU_PValue"].notna(), "fdr MWU"] = fdr_corrected
                    temp["enhanced?"] = temp.apply(lambda row: True if row["Trend"] == "enhanced" else False, axis=1)
                    # rename column GO_term to GO term
                    temp = temp.rename(columns={"GO_Term": "GO term", "N_Genes": "size"})
                    new_name = os.path.join("./Private/clusters_properties", "diff_abx" + run_type,
                                            file.name.replace("results", "top_correlated_GO_terms"))
                    if "all_go_term" in new_name:
                        new_name = new_name.replace("all_go_term_", "")
                    temp.to_csv(new_name, sep="\t")

    our = plot_categories(antibiotics, treatments, "/diff_abx" + run_type, False, regular=False, mix=False)
    gsea_abx = [abx.replace("+", "_") for abx in antibiotics]
    gsea = plot_categories(gsea_abx, [''], "/diff_abx" + run_type + "GSEA", False, regular=False,
                           gsea=True, mix=False)
    plot_correlation_gsea(gsea, our)


# --- 1. Data Loading (Your Function) ---
def read_data_metadata(t=False, remove_mitochondrial=True, normalize=True):
    base_path = '/Users/yonchlevin/Desktop/ErezLab/MouseAbxBel/Git/Data/Pairs/'
    metadata_path = base_path + 'metadata.tsv'
    metadata_df = pd.read_csv(metadata_path, sep='\t')

    # Cleaning sample names to match convention
    metadata_df["Sample"] = metadata_df["Sample"].str.replace("-", "").str.replace(" #", "_")
    metadata_df["ID"] = metadata_df["Sample"]

    # Temporary placeholder (update if needed)
    metadata_df["Treatment"] = "PO"
    # metadata_df["Treatment"] = "?"

    data_path = base_path + 'genes_norm_named-newRNAseq.tsv'
    data_df = pd.read_csv(data_path, sep='\t')
    id_to_name = dict(zip(data_df['gene_id'], data_df['gene_name']))
    data_df = data_df.set_index('gene_id').drop(columns=['gene_name'])
    data_df.columns = [f"{col.split('_')[0]}_{col.split('_')[1]}" for col in data_df.columns]

    # Ensure alignment
    assert all(sample in data_df.columns for sample in
               metadata_df['Sample']), "Some samples in metadata are not in data columns"
    # common_samples = [s for s in metadata_df['Sample'] if s in data_df.columns]
    # data_df = data_df[common_samples]
    # metadata_df = metadata_df[metadata_df['Sample'].isin(common_samples)].set_index('Sample')

    # 1. Remove sparse genes (more than 50% zeros in a row)
    zeros_count = data_df[data_df == 0].count(axis=1)
    sparse_genes = zeros_count[zeros_count > 0.5 * data_df.shape[1]]
    data_df = data_df.drop(sparse_genes.index)

    # 2. Remove Mitochondrial Genes
    if remove_mitochondrial:
        # We need to check the gene names, but the index is currently gene_id.
        # We use the id_to_name map created earlier.

        # Get gene IDs where the mapped name is in the mitochondrial list
        mito_ids = [
            gid for gid, gname in id_to_name.items()
            if str(gname).lower() in MITOCHONDRIAL_GENES
        ]

        # Filter only those that exist in the current DataFrame
        matching_indices = [gid for gid in mito_ids if gid in data_df.index]
        data_df = data_df.drop(matching_indices, errors='ignore')

    # 3. Normalization (CPM / RPKM equivalent)
    if normalize:
        # (Count * 1,000,000) / Sum of Column
        data_df = (data_df * 1000000).divide(data_df.sum(axis=0), axis=1)

    if t:
        # Transpose for PyDESeq2 (samples as rows)
        counts_df = data_df.T
    else:
        counts_df = data_df

    return counts_df, metadata_df, id_to_name


# --- 2. Interference Analysis (DESeq2 Interaction) ---
def run_interference_analysis(counts, meta, drug_A, drug_B):
    """
    Tests for non-additivity (Interference).
    Model: ~ Has_A + Has_B + Interaction
    """
    pair_name = f"{drug_A}+{drug_B}"  # e.g. Met_Van (check your naming convention in metadata)

    # We filter for the relevant 4 groups: PBS, A, B, Pair
    target_groups = ['PBS', drug_A, drug_B, pair_name]
    sub_meta = meta[meta['Drug'].isin(target_groups)].copy()
    sub_counts = counts.loc[sub_meta['Sample']]
    # convert sub_counts to integers (DESeq2 requirement)
    sub_counts = sub_counts.astype(int)
    sub_meta = sub_meta.set_index('Sample')

    # Create Binary Factors for the Design
    # Has_A: 1 if sample treated with A (Single A or Pair)
    # Has_B: 1 if sample treated with B (Single B or Pair)
    # Interaction: 1 ONLY if Pair (This captures the deviation from additivity)

    sub_meta['Has-A'] = sub_meta['Drug'].apply(lambda x: 'True' if drug_A in x else 'False')
    sub_meta['Has-B'] = sub_meta['Drug'].apply(lambda x: 'True' if drug_B in x else 'False')
    sub_meta['Is-Combo'] = sub_meta['Drug'].apply(lambda x: 'True' if x == pair_name else 'False')

    # Run DESeq2
    # We test the 'Is_Combo' coefficient.
    # If Is_Combo is significant, the effect is NOT just Sum(A)+Sum(B).
    try:
        dds = DeseqDataSet(
            counts=sub_counts,
            metadata=sub_meta,
            design_factors=['Has-A', 'Has-B', 'Is-Combo'],
            n_cpus=4
        )
        dds.deseq2()

        stat_res = DeseqStats(dds, contrast=["Is-Combo", 'True', 'False'])
        stat_res.summary()
        res = stat_res.results_df
        res = res.rename(columns={'log2FoldChange': 'LFC_Interaction', 'padj': 'padj_Interaction'})

        # 3. Extract Main Effect: Drug A vs PBS
        stat_A = DeseqStats(dds, contrast=["Has-A", 'True', 'False'])
        stat_A.summary()
        res_A = stat_A.results_df
        # We only need the LFC from here
        res[f'LFC_{drug_A}'] = res_A['log2FoldChange']

        # 4. Extract Main Effect: Drug B vs PBS
        stat_B = DeseqStats(dds, contrast=["Has-B", 'True', 'False'])
        stat_B.summary()
        res_B = stat_B.results_df
        # We only need the LFC from here
        res[f'LFC_{drug_B}'] = res_B['log2FoldChange']

        # 5. Clean up and Sort
        # Filter for significant interactions (optional, or return all)
        # res_int = res_int[res_int['padj_Interaction'] < 0.05]

        # Reorder columns for readability
        cols = ['baseMean', 'LFC_Interaction', 'padj_Interaction', f'LFC_{drug_A}', f'LFC_{drug_B}', 'gene_name']
        # Select only cols that exist (in case gene_name isn't there yet)
        final_cols = [c for c in cols if c in res.columns]
        res_final = res[final_cols].sort_values("padj_Interaction")

        # res = res.sort_values("pvalue")
        # return res
        return res_final
    except Exception as e:
        print(f"DESeq2 Failed for {pair_name}: {e}")
        return pd.DataFrame()


# --- 3. Suppression Analysis (Pair < A AND Pair < B) ---
def run_suppression_analysis(counts, meta, drug_A, drug_B):
    """
    Identifies genes where the Pair expression is significantly LOWER
    than Drug A *AND* significantly LOWER than Drug B.

    Model: ~ Drug (Simple One-Factor design for direct pairwise contrasts)
    """
    pair_name = f"{drug_A}+{drug_B}"

    # Filter for the relevant 4 groups
    target_groups = ['PBS', drug_A, drug_B, pair_name]
    sub_meta = meta[meta['Drug'].isin(target_groups)].copy()
    sub_counts = counts.loc[sub_meta['Sample']]
    sub_counts = sub_counts.astype(int)
    sub_meta = sub_meta.set_index('Sample')

    # Unlike Interference, we don't need 'Has-A'/'Has-B'.
    # We just need the 'Drug' column to act as our condition.
    # Ensure 'Drug' is treated as a factor if not already (though strings usually work).

    try:
        # Run DESeq2 with a simple Group design
        dds = DeseqDataSet(
            counts=sub_counts,
            metadata=sub_meta,
            design_factors=['Drug'],
            n_cpus=4
        )
        dds.deseq2()

        # --- Contrast 1: Pair vs Drug A ---
        # "Is the Pair lower than A?"
        res_A_stat = DeseqStats(dds, contrast=["Drug", pair_name, drug_A])
        res_A_stat.summary()
        res_A = res_A_stat.results_df

        # --- Contrast 2: Pair vs Drug B ---
        # "Is the Pair lower than B?"
        res_B_stat = DeseqStats(dds, contrast=["Drug", pair_name, drug_B])
        res_B_stat.summary()
        res_B = res_B_stat.results_df

        # --- Contrast 3: PBS vs Drug A ---
        # "Is the Pair lower than A?"
        res_C_stat = DeseqStats(dds, contrast=["Drug", "PBS", drug_A])
        res_C_stat.summary()
        res_C = res_C_stat.results_df

        # --- Contrast 4: PBS vs Drug B ---
        # "Is the Pair lower than B?"
        res_D_stat = DeseqStats(dds, contrast=["Drug", "PBS", drug_B])
        res_D_stat.summary()
        res_D = res_D_stat.results_df

        # --- Find Intersection ---
        # We want genes where log2FoldChange is NEGATIVE in BOTH, and significant in BOTH.

        # 1. Filter A: Pair is significantly lower than A
        suppressed_vs_A = res_A[(res_A['log2FoldChange'] < 0) & (res_A['padj'] < 0.05)].index

        # 2. Filter B: Pair is significantly lower than B
        suppressed_vs_B = res_B[(res_B['log2FoldChange'] < 0) & (res_B['padj'] < 0.05)].index

        # 1. Filter C: PBS is significant than A
        PBS_vs_A = res_C[res_C['padj'] < 0.05].index

        # 2. Filter B: Pair is significantly lower than B
        PBS_vs_B = res_D[res_D['padj'] < 0.05].index

        # 3. Intersection
        common_genes1 = list(set(suppressed_vs_A) & set(suppressed_vs_B))
        # common_genes2 = list(set(PBS_vs_A) & set(PBS_vs_B))
        # common_genes = list(set(common_genes1) & set(common_genes2))
        common_genes = common_genes1  # Relaxed condition without PBS check

        # Create a result DataFrame for these specific genes
        # We will store the LFC vs A and LFC vs B for reference
        final_df = pd.DataFrame(index=common_genes)
        final_df['LFC_vs_A'] = res_A.loc[common_genes, 'log2FoldChange']
        final_df['padj_vs_A'] = res_A.loc[common_genes, 'padj']
        final_df['LFC_vs_B'] = res_B.loc[common_genes, 'log2FoldChange']
        final_df['padj_vs_B'] = res_B.loc[common_genes, 'padj']
        # final_df['LFC_A_vs_PBS'] = res_C.loc[common_genes, 'log2FoldChange']
        # final_df['padj_A_vs_PBS'] = res_C.loc[common_genes, 'padj']
        # final_df['LFC_B_vs_PBS'] = res_D.loc[common_genes, 'log2FoldChange']
        # final_df['padj_B_vs_PBS'] = res_D.loc[common_genes, 'padj']

        # Sort by the "least suppressed" value (closest to 0) to find the most robust ones first?
        # Or sort by the average suppression. Let's sort by average LFC.
        final_df['mean_LFC'] = (final_df['LFC_vs_A'] + final_df['LFC_vs_B']) / 2
        final_df = final_df.sort_values('mean_LFC')  # Most negative first

        print(f"Found {len(final_df)} genes suppressed vs BOTH singles.")
        return final_df

    except Exception as e:
        print(f"Suppression Analysis Failed for {pair_name}: {e}")
        return pd.DataFrame()


# --- 3. Noise Analysis (PCA Dist & Variance) ---
def analyze_noise(counts, meta, pair_name, singles):
    """
    1. Global Noise: Distance to Centroid in PCA
    2. Gene Noise: Variance of Residuals / CV
    """
    # Filter Data
    groups = ['PBS', pair_name] + singles
    sub_meta = meta[meta['Drug'].isin(groups)].copy()
    sub_meta = sub_meta.set_index('Sample')
    sub_counts = counts.loc[sub_meta.index]

    # Normalize (Log CPM)
    # Simple normalization for PCA/Variance (DESeq2 does its own)
    lib_size = sub_counts.sum(axis=1)
    cpm = sub_counts.div(lib_size, axis=0) * 1e6
    log_cpm = np.log2(cpm + 1)

    # A. Global Noise: PCA Distance
    pca = PCA(n_components=2)
    coords = pca.fit_transform(log_cpm)
    pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'], index=sub_meta.index)
    pca_df['Group'] = sub_meta['Drug']

    # Calculate Centroids and Distances
    centroids = pca_df.groupby('Group')[['PC1', 'PC2']].mean()
    distances = []
    for sample, row in pca_df.iterrows():
        grp = row['Group']
        cent = centroids.loc[grp]
        dist = euclidean(row[['PC1', 'PC2']], cent)
        distances.append({'Sample': sample, 'Group': grp, 'DistToCentroid': dist})

    dist_df = pd.DataFrame(distances)

    # B. Gene Noise: Residual Variance
    # We fit a model Expr ~ Group to remove mean effects, then look at residuals
    # Using OLS for speed on log-cpm
    residuals = pd.DataFrame(index=log_cpm.index, columns=log_cpm.columns)

    # Vectorized residual calculation: Data - GroupMean
    group_means = log_cpm.groupby(sub_meta['Drug']).mean()
    for sample in log_cpm.index:
        grp = sub_meta.loc[sample, 'Drug']
        residuals.loc[sample] = log_cpm.loc[sample] - group_means.loc[grp]

    # Calculate Variance of Residuals PER GROUP
    # We want to know if Pair variance > Single variance
    res_var = {}
    for grp in groups:
        samples_in_grp = sub_meta[sub_meta['Drug'] == grp].index
        # Variance across samples for each gene
        res_var[grp] = residuals.loc[samples_in_grp].var(axis=0)

    var_df = pd.DataFrame(res_var)
    # Calculate Noise Ratio: Pair / Mean(Singles)
    # (Adding small epsilon to avoid div/0)
    avg_single_var = var_df[singles].mean(axis=1)
    var_df['NoiseRatio'] = var_df[pair_name] / (avg_single_var + 1e-8)

    return dist_df, var_df


# --- Updated Interference Analysis (Gets Single LFCs + Interaction) ---
def run_interference_analysis_detailed(counts, meta, drug_A, drug_B):
    pair_name = f"{drug_A}+{drug_B}"
    target_groups = ['PBS', drug_A, drug_B, pair_name]

    sub_meta = meta[meta['Drug'].isin(target_groups)].copy()
    sub_counts = counts.loc[sub_meta['Sample']].astype(int)
    sub_meta = sub_meta.set_index('Sample')

    # Manual Factorial Design
    sub_meta['Has-A'] = sub_meta['Drug'].apply(lambda x: 'True' if drug_A in x else 'False')
    sub_meta['Has-B'] = sub_meta['Drug'].apply(lambda x: 'True' if drug_B in x else 'False')
    sub_meta['Is-Combo'] = sub_meta['Drug'].apply(lambda x: 'True' if x == pair_name else 'False')

    try:
        dds = DeseqDataSet(counts=sub_counts, metadata=sub_meta, design_factors=['Has-A', 'Has-B', 'Is-Combo'],
                           n_cpus=4)
        dds.deseq2()

        # 1. Interaction (Interference)
        stat_int = DeseqStats(dds, contrast=["Is-Combo", 'True', 'False'])
        stat_int.summary()
        res = stat_int.results_df.rename(columns={'log2FoldChange': 'LFC_Interaction', 'padj': 'padj_Interaction'})

        # 2. Main Effect A (vs PBS)
        stat_A = DeseqStats(dds, contrast=["Has-A", 'True', 'False'])
        stat_A.summary()
        res[f'LFC_{drug_A}'] = stat_A.results_df['log2FoldChange']

        # 3. Main Effect B (vs PBS)
        stat_B = DeseqStats(dds, contrast=["Has-B", 'True', 'False'])
        stat_B.summary()
        res[f'LFC_{drug_B}'] = stat_B.results_df['log2FoldChange']

        return res
    except Exception as e:
        print(f"Detailed Analysis Failed for {pair_name}: {e}")
        return pd.DataFrame()


# --- The 3 Deep Dives (Signal Spread, Opposing Signs, Synergy) ---
def investigate_mechanisms(counts, meta, res_detailed, drug_A, drug_B, id_map):
    pair_name = f"{drug_A}+{drug_B}"

    # 1. Signal Loss: Shift vs. Spread
    # Filter for significant buffering (Negative Interaction)
    suppressed = res_detailed.index
    # suppressed = res_detailed[(res_detailed['padj_Interaction'] < 0.05) & (res_detailed['LFC_Interaction'] < 0)].index

    if len(suppressed) > 0:
        stats_data = []
        for g in [drug_A, drug_B, pair_name]:
            samples = meta[meta['Drug'] == g]['Sample']
            g_counts = counts.loc[samples, suppressed]
            stats_data.append(pd.DataFrame({'mean': g_counts.mean(axis=0), 'std': g_counts.std(axis=0), 'group': g}))

        stats_df = pd.concat(stats_data)
        pivot_std = stats_df.reset_index().pivot(index='gene_id', columns='group', values='std')
        pivot_mean = stats_df.reset_index().pivot(index='gene_id', columns='group', values='mean')

        # Calculate Ratios
        pivot_std['Pair_vs_AvgSingle_SD'] = pivot_std[pair_name] / ((pivot_std[drug_A] + pivot_std[drug_B]) / 2)
        pivot_mean['Pair_vs_AvgSingle_Mean'] = pivot_mean[pair_name] / ((pivot_mean[drug_A] + pivot_mean[drug_B]) / 2)

        plt.figure(figsize=(6, 6))
        plt.scatter(pivot_mean['Pair_vs_AvgSingle_Mean'], pivot_std['Pair_vs_AvgSingle_SD'], alpha=0.5, c='teal')
        plt.axhline(1, color='red', linestyle='--')
        plt.axvline(1, color='red', linestyle='--')
        plt.xlabel("Mean Ratio (<1 = True Repression)")
        plt.ylabel("Spread Ratio (>1 = Increased Noise)")
        plt.title(f"Mechanism of Signal Loss ({len(suppressed)} genes)")
        plt.savefig(f"./Private/Noise/Mech_SignalLoss_{pair_name}.png")
        plt.close()


def opposite_signs(res_detailed, drug_A, drug_B, id_map):
    # 2. Opposing Signs (Tug of War)
    # Calculate approx Pair LFC
    pair_name = f"{drug_A}+{drug_B}"
    res_detailed['LFC_Pair_Total'] = res_detailed[f'LFC_{drug_A}'] + res_detailed[f'LFC_{drug_B}'] + res_detailed[
        'LFC_Interaction']

    opposing = res_detailed[
        (np.sign(res_detailed[f'LFC_{drug_A}']) != np.sign(res_detailed[f'LFC_{drug_B}'])) &
        (abs(res_detailed[f'LFC_{drug_A}']) > 0.5) & (abs(res_detailed[f'LFC_{drug_B}']) > 0.5)
        ].copy()

    if not opposing.empty:
        opposing['Outcome'] = opposing.apply(
            lambda x: f"Dominance ({drug_A})" if np.sign(x['LFC_Pair_Total']) == np.sign(x[f'LFC_{drug_A}'])
            else (
                f"Dominance ({drug_B})" if np.sign(x['LFC_Pair_Total']) == np.sign(x[f'LFC_{drug_B}']) else "Complex"),
            axis=1)
        opposing['gene_name'] = opposing.index.map(id_map)
        opposing.to_csv(f"./Private/Noise/Opposing_Signs_{pair_name}.csv")

    # 3. Super-Enhancement (True Synergy)
    # Pair > A, Pair > B, Pair > Sum
    super_enhanced = res_detailed[
        (res_detailed['padj_Interaction'] < 0.05) & (res_detailed['LFC_Interaction'] > 0) &
        (res_detailed[f'LFC_{drug_A}'] > 0) & (res_detailed[f'LFC_{drug_B}'] > 0)
        # & (res_detailed['LFC_Pair_Total'] > res_detailed[f'LFC_{drug_A}']) &
        # (res_detailed['LFC_Pair_Total'] > res_detailed[f'LFC_{drug_B}'])
        ].sort_values("LFC_Interaction", ascending=False)

    if not super_enhanced.empty:
        super_enhanced['gene_name'] = super_enhanced.index.map(id_map)
        super_enhanced.to_csv(f"./Private/Noise/Super_Enhanced_{pair_name}.csv")
        print(f"   -> Found {len(super_enhanced)} Super-Enhanced (True Synergy) genes.")


def plot_top_genes_for_pair(pair_name, csv_path, data, metadata, id_map, output_base="./Private/Noise/Top_Plots",
                            top_n=10):
    """
    Reads a results CSV, picks the top N genes, and plots their expression
    across PBS, Drug A, Drug B, and Pair.
    """
    # 1. Setup Directories
    # Create a subfolder for the specific file analysis (e.g., "Met+Van/Interference")
    analysis_name = os.path.basename(csv_path).replace(".csv", "")
    save_dir = os.path.join(output_base, pair_name, analysis_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Plotting top {top_n} genes from {analysis_name}...")

    # 2. Load and Sort Data
    try:
        res_df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    # Intelligent Sorting based on columns present
    if 'padj_Interaction' in res_df.columns:
        # Detailed Interference File -> Sort by Interaction Significance
        res_df = res_df.sort_values('padj_Interaction', ascending=True)
        metric = "padj_Interaction"
    elif 'padj' in res_df.columns:
        # Standard DESeq2 Results -> Sort by Significance
        res_df = res_df.sort_values('padj', ascending=True)
        metric = "padj"
    elif 'NoiseRatio' in res_df.columns:
        # Noise Analysis -> Sort by Ratio (Descending)
        res_df = res_df.sort_values('NoiseRatio', ascending=False)
        metric = "NoiseRatio"
    elif 'LFC_vs_A' in res_df.columns:
        # Suppression Analysis -> Sort by average suppression (Most negative first)
        res_df['avg_suppression'] = (res_df['LFC_vs_A'] + res_df['LFC_vs_B']) / 2
        res_df = res_df.sort_values('avg_suppression', ascending=True)
        metric = "avg_suppression"
    else:
        # Fallback
        metric = "Index (Unsorted)"

    top_genes = res_df.head(top_n).index.tolist()

    # 3. Define Groups based on Pair Name (e.g., "Met+Van")
    if '+' in pair_name:
        drug_a, drug_b = pair_name.split('+')
    else:
        # Fallback if naming is different, though your prompt implies "A+B"
        print(f"Cannot parse drugs from name '{pair_name}'. using default pair logic.")
        return

    target_groups = ['PBS', drug_a, drug_b, pair_name]

    # 4. Loop and Plot
    for gene_id in top_genes:
        # Handle cases where gene_id might not be in data (filtering issues)
        if gene_id not in data.index:
            continue

        gene_name = id_map.get(gene_id, gene_id)

        # Prepare Data for Plotting
        plot_data = []
        for g in target_groups:
            # Find samples for this group
            samples = metadata[metadata['Drug'] == g]["Sample"].to_list()
            # Intersect with available columns in data
            valid_samples = [s for s in samples if s in data.columns]

            expr_values = data.loc[gene_id, valid_samples].values
            for val in expr_values:
                plot_data.append({'Drug': g, 'Expression': val})

        plot_df = pd.DataFrame(plot_data)

        if plot_df.empty:
            continue

        # Dynamic Y-Axis Limits (Visual buffer)
        max_val = plot_df['Expression'].max() * 1.1
        min_val = plot_df['Expression'].min() * 0.9
        # If min_val is near 0 or negative (log space), handle carefully
        if min_val > 0:
            min_val = min_val * 0.9
        else:
            min_val = min_val * 1.1

        # --- Plotting ---
        plt.figure(figsize=(6, 4))
        sns.barplot(data=plot_df, x='Drug', y='Expression', capsize=.1, errorbar='sd', palette='Set2')
        sns.swarmplot(data=plot_df, x='Drug', y='Expression', color='black', size=4, alpha=0.7)

        # Add Metric info to title
        metric_val = res_df.loc[gene_id][metric] if metric in res_df.columns else "N/A"
        try:
            metric_str = f"{metric_val:.2e}" if isinstance(metric_val, float) else str(metric_val)
        except:
            metric_str = str(metric_val)

        plt.title(f"{gene_name}\n({metric}: {metric_str})")
        plt.ylabel("Expression")
        # plt.ylabel("Normalized Expression (Log)")
        plt.xlabel("")
        plt.ylim(min_val, max_val)

        # Save
        safe_gene = gene_name.replace("/", "-")  # Handle weird gene names
        out_path = os.path.join(save_dir, f"{safe_gene}_{analysis_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(top_genes)} plots to {save_dir}")


def plot_genes_for_pair(pair_name, csv_path, data, metadata, id_map, output_base="./Private/Noise/Top_Plots",
                        top_n=10):
    """
    Reads a results CSV, picks the top N genes, and plots their expression
    across PBS, Drug A, Drug B, and Pair.
    """
    # 1. Setup Directories
    # Create a subfolder for the specific file analysis (e.g., "Met+Van/Interference")
    analysis_name = "All_Suppressed_Genes"
    save_dir = os.path.join(output_base, pair_name, analysis_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Plotting genes from {analysis_name}...")

    # 2. Load and Sort Data
    try:
        res_df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    # Suppression Analysis -> Sort by average suppression (Most negative first)
    res_df['avg_suppression'] = (res_df['LFC_vs_A'] + res_df['LFC_vs_B']) / 2
    res_df = res_df.sort_values('avg_suppression', ascending=True)
    metric = "avg_suppression"

    top_genes = res_df.index.tolist()

    # 3. Define Groups based on Pair Name (e.g., "Met+Van")
    drug_a, drug_b = pair_name.split('+')

    target_groups = ['PBS', drug_a, drug_b, pair_name]

    # 4. Loop and Plot
    for gene_id in top_genes:
        # Handle cases where gene_id might not be in data (filtering issues)
        if gene_id not in data.index:
            continue

        gene_name = id_map.get(gene_id, gene_id)

        # Prepare Data for Plotting
        plot_data = []
        for g in target_groups:
            # Find samples for this group
            samples = metadata[metadata['Drug'] == g]["Sample"].to_list()
            # Intersect with available columns in data
            valid_samples = [s for s in samples if s in data.columns]

            expr_values = data.loc[gene_id, valid_samples].values
            for val in expr_values:
                plot_data.append({'Drug': g, 'Expression': val})

        plot_df = pd.DataFrame(plot_data)

        if plot_df.empty:
            continue

        # Dynamic Y-Axis Limits (Visual buffer)
        max_val = plot_df['Expression'].max() * 1.1
        min_val = plot_df['Expression'].min() * 0.9
        # If min_val is near 0 or negative (log space), handle carefully
        if min_val > 0:
            min_val = min_val * 0.9
        else:
            min_val = min_val * 1.1

        # --- Plotting ---
        plt.figure(figsize=(6, 4))
        sns.barplot(data=plot_df, x='Drug', y='Expression', capsize=.1, errorbar='sd', palette='Set2')
        sns.swarmplot(data=plot_df, x='Drug', y='Expression', color='black', size=4, alpha=0.7)

        # Add Metric info to title
        metric_val = res_df.loc[gene_id][metric] if metric in res_df.columns else "N/A"
        try:
            metric_str = f"{metric_val:.2e}" if isinstance(metric_val, float) else str(metric_val)
        except:
            metric_str = str(metric_val)

        plt.title(f"{gene_name}\n({metric}: {metric_str})")
        plt.ylabel("Expression")
        # plt.ylabel("Normalized Expression (Log)")
        plt.xlabel("")
        plt.ylim(min_val, max_val)

        # Save
        safe_gene = gene_name.replace("/", "-")  # Handle weird gene names
        out_path = os.path.join(save_dir, f"{safe_gene}_{analysis_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(top_genes)} plots to {save_dir}")


def run_analysis(counts_df, metadata_df, id_map, pairs_config):
    for config in pairs_config:
        pair = config['pair']
        print(f"\n--- Analyzing {pair} ---")
        #
        # # 2. Interference
        # print("Running DESeq2 Interference Analysis...")
        # # int_res = run_interference_analysis(counts_df, metadata_df, config['A'], config['B'])
        # detailed_res = run_interference_analysis_detailed(counts_df, metadata_df, config['A'], config['B'])
        #
        # if not detailed_res.empty:
        #     # A. Save the Master Table (with single LFCs and Interaction)
        #     detailed_res['gene_name'] = detailed_res.index.map(id_map)
        #     detailed_res.to_csv(f"./Private/Noise/Detailed_Interference_{pair}.csv")
        #
        #     # B. Save the "Standard" Interference file for the Dashboard
        #     # The plotting function expects 'log2FoldChange' and 'padj'
        #     sig_int_plot = detailed_res.rename(columns={
        #         'LFC_Interaction': 'log2FoldChange',
        #         'padj_Interaction': 'padj'
        #     })
        #     # Filter for significant interactions only for the plot file
        #     sig_int_plot = sig_int_plot[sig_int_plot['padj'] < 0.05]
        #     sig_int_plot.to_csv(f"./Private/Noise/Interference_Genes_{pair}.csv")
        #     print(f"   -> Found {len(sig_int_plot)} significant interference genes.")
        #
        #     # --- 2b. The 3 New Mechanisms (Deep Dive) ---
        #     print("Running Mechanism Investigations (Spread, Opposing, Super-Enhancement)...")
        #     opposite_signs(detailed_res, config['A'], config['B'], id_map)
        # # if not int_res.empty:
        # #     # Save significant interference genes (padj < 0.05)
        # #     sig_int = int_res[int_res['padj_Interaction'] < 0.05].copy()
        # #     # sig_int = int_res[int_res['padj'] < 0.05].copy()
        # #     sig_int['gene_name'] = sig_int.index.map(id_map)
        # #     output_file = f"./Private/Noise/Interference_Genes_{pair}.csv"
        # #     sig_int.to_csv(output_file)
        # #     print(f"Found {len(sig_int)} interference genes. Saved to {output_file}")
        #
        # 2b. Suppression
        print("Running DESeq2 Interference (single) Analysis...")
        sup_res = run_suppression_analysis(counts_df, metadata_df, config['A'], config['B'])

        if not sup_res.empty:
            # # Save significant interference genes (padj < 0.05)
            # sig_int = sup_res[(sup_res['padj_vs_A'] < 0.05) | (sup_res['padj_vs_B'] < 0.05)].copy()
            # sig_int['gene_name'] = sig_int.index.map(id_map)
            # output_file = f"./Private/Noise/Suppressed_Genes_{pair}.csv"
            # sig_int.to_csv(output_file)
            # print(f"Found {len(sig_int)} suppressed genes. Saved to {output_file}")
            sup_res.index.name = "gene_id"
            investigate_mechanisms(counts_df, metadata_df, sup_res, config['A'], config['B'], id_map)
            sup_res['gene_name'] = sup_res.index.map(id_map)
            output_file = f"./Private/Noise/Suppressed_Genes_{pair}.csv"
            sup_res.to_csv(output_file)
            print(f"   -> Found {len(sup_res)} strictly suppressed genes.")

        # # 3. Noise
        # print("Running Noise/Variance Analysis...")
        # dist_df, var_df = analyze_noise(counts_df, metadata_df, pair, [config['A'], config['B']])
        #
        # # Save Global Noise stats
        # dist_df.to_csv(f"./Private/Noise/Global_Noise_Distances_{pair}.csv")
        #
        # # Check if Pair is globally noisier (T-test vs singles)
        # pair_dists = dist_df[dist_df['Group'] == pair]['DistToCentroid']
        # single_dists = dist_df[dist_df['Group'].isin([config['A'], config['B']])]['DistToCentroid']
        # from scipy.stats import ttest_ind
        #
        # t_stat, p_val = ttest_ind(pair_dists, single_dists, equal_var=False)
        # print(f"Global Noise P-value (Pair vs Singles): {p_val:.4f}")
        #
        # # Save Gene Noise candidates
        # # Genes where Pair variance is > 2x Single variance
        # noisy_genes = var_df[var_df['NoiseRatio'] > 2].sort_values('NoiseRatio', ascending=False)
        # noisy_genes['gene_name'] = noisy_genes.index.map(id_map)
        # noisy_genes.to_csv(f"./Private/Noise/Noisy_Genes_{pair}.csv")
        # print(f"Found {len(noisy_genes)} genes with high induced noise (>2x ratio).")

        # Adjust filename pattern to match your previous save output
        safe_p = pair  # or p.replace("+", "_") depending on your file naming

        int_file = f"./Private/Noise/Interference_Genes_{safe_p}.csv"
        dist_file = f"./Private/Noise/Global_Noise_Distances_{safe_p}.csv"
        noise_file = f"./Private/Noise/Noisy_Genes_{safe_p}.csv"

        print(f"Generating plot for {pair}...")
        plot_interference_noise_dashboard(pair, int_file, dist_file, noise_file)
    print("\nAnalysis Complete.")


def plot_partial_results(counts_df, metadata_df, id_map, pairs_config):
    # plot genes
    print("Generating individual gene plots...")
    for config in pairs_config:
        pair = config['pair']
        # Define which CSVs you want to plot from
        files_to_plot = [
            f"./Private/Noise/Interference_Genes_{pair}.csv",
            f"./Private/Noise/Opposing_signs_{pair}.csv",
            f"./Private/Noise/Noisy_Genes_{pair}.csv",
            f"./Private/Noise/Suppressed_Genes_{pair}.csv",
            f"./Private/Noise/Super_Enhanced_{pair}.csv"
        ]

        for csv_path in files_to_plot:
            if os.path.exists(csv_path):
                # IMPORTANT: Pass 'data' (the normalized/log transformed dataframe), NOT 'counts_df'
                plot_top_genes_for_pair(
                    pair_name=pair,
                    csv_path=csv_path,
                    data=counts_df.T,  # <--- Make sure this is your logCPM or VST data
                    # data=data,  # <--- Make sure this is your logCPM or VST data
                    metadata=metadata_df,
                    id_map=id_map,
                    top_n=10
                )
    # exit()


def plot_partial_neo_results(counts_df, metadata_df, id_map, pairs_config):
    for pair in ["Met+Van", "Met+Neo"]:
        csv_path = f"./Private/Noise/Suppressed_Genes_{pair}.csv"
        plot_genes_for_pair(
            pair_name=pair,
            csv_path=csv_path,
            data=counts_df.T,  # <--- Make sure this is your logCPM or VST data
            # data=data,  # <--- Make sure this is your logCPM or VST data
            metadata=metadata_df,
            id_map=id_map,
        )


def plot_suppression_heatmap(counts, meta, drug_A, drug_B, id_map, output_dir="./Private/Noise"):
    """
    Generates a Z-scored Clustermap for genes that are strictly suppressed
    (Pair < A AND Pair < B).
    """
    from scipy.stats import zscore

    pair_name = f"{drug_A}+{drug_B}"
    print(f"\n--- Generating Suppression Heatmap for {pair_name} ---")

    # 1. Identify Suppressed Genes (Re-running logic to be sure)
    # We need genes where LFC vs A < 0 AND LFC vs B < 0 (and significant)
    # NOTE: This requires the 'run_suppression_analysis' function we defined earlier.
    supp_df = pd.read_csv(f"./Private/Noise/Suppressed_Genes_{pair_name}.csv", index_col=0)

    if supp_df.empty:
        print("No suppressed genes found. Skipping heatmap.")
        return

    supp_genes = supp_df.index.tolist()
    print(f"Plotting {len(supp_genes)} suppressed genes...")

    # 2. Extract Data for Plotting
    # We want columns: PBS, A, B, Pair
    target_groups = ['PBS', drug_A, drug_B, pair_name]
    sub_meta = meta[meta['Drug'].isin(target_groups)].copy()

    # Sort samples by group for clean visualization (PBS -> A -> B -> Pair)
    # Create a categorical type to force order
    sub_meta['Drug'] = pd.Categorical(sub_meta['Drug'], categories=target_groups, ordered=True)
    sub_meta = sub_meta.sort_values('Drug')
    ordered_samples = sub_meta['Sample'].tolist()

    # # Get Expression Matrix (Log-Normalized)
    # # Assuming 'counts' is raw, we normalize and log.
    # # If 'counts' is already normalized, skip the size_factor step.
    # # Here using a simple logCPM approximation for visualization:
    # raw_sub = counts.loc[supp_genes, ordered_samples]
    # # Simple depth normalization (CPM)
    # norm_counts = raw_sub.div(raw_sub.sum(axis=0), axis=1) * 1e6
    # log_counts = np.log1p(norm_counts)
    #
    # # 3. Z-Score (Row-wise)
    # # Subtract Mean, Divide by SD for each gene.
    # # This highlights relative differences (High vs Low) rather than absolute counts.
    # z_data = log_counts.apply(zscore, axis=1)

    # z-score by PBS mean and SD to highlight suppression relative to baseline
    raw_sub = counts.loc[supp_genes, ordered_samples]
    # raw_sub.to_csv(f"{output_dir}/Heatmap_Suppression_Data_{pair_name}_raw.csv")
    # Calculate mean and std for PBS group
    pbs_samples = sub_meta[sub_meta['Drug'] == 'PBS']['Sample']
    pbs_mean = raw_sub[pbs_samples].mean(axis=1)
    pbs_std = raw_sub[pbs_samples].std(axis=1)
    min_nonzero_std = pbs_std[pbs_std > 0].min()
    pbs_std[pbs_std == 0] = min_nonzero_std  # Avoid div by zero
    # Z-score relative to PBS
    z_data = raw_sub.sub(pbs_mean, axis=0).div(pbs_std, axis=0)

    # Map Index to Gene Names for readability
    z_data.index = z_data.index.map(lambda x: id_map.get(x, x))

    # 4. Create Annotation Colors for Columns (Samples)
    group_map = dict(zip(sub_meta['Sample'], sub_meta['Drug']))
    # Define colors: PBS=Gray, A=Red, B=Blue, Pair=Purple (example palette)
    # Or let Seaborn pick. We'll verify distinctness.
    palette = sns.color_palette("Set2", 4)
    lut = dict(zip(target_groups, palette))
    col_colors = [lut[abx] for abx in sub_meta['Drug'].tolist()]
    # 5. Plot Clustermap
    # col_cluster=False -> Keeps samples ordered by group (PBS, A, B, Pair)
    # row_cluster=True  -> Groups genes with similar suppression patterns
    # z_data.to_csv(f"{output_dir}/Heatmap_Suppression_Data_{pair_name}_zscored.csv")
    lim = 7
    g = sns.clustermap(z_data,
                       # col_cluster=True,
                       col_cluster=False,
                       row_cluster=True,
                       z_score=None,  # Already Z-scored manually
                       cmap="vlag",  # Blue-White-Red (Red=High, Blue=Low)
                       center=0,
                       col_colors=col_colors,
                       figsize=(10, 12),
                       vmax=lim, vmin=-lim,
                       xticklabels=False,  # Hide sample names if too many
                       yticklabels=True)  # Show gene names

    # save z_data in the order of the heatmap
    ordered_z_data = z_data.iloc[g.dendrogram_row.reordered_ind]
    ordered_z_data.to_csv(f"{output_dir}/Heatmap_Suppression_Data_{pair_name}_zscored_ordered.csv")
    # Add Legend for Groups
    # for label, color in lut.items():
    #     g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
    # g.ax_col_dendrogram.legend(loc="best", ncol=4)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=color, label=label) for label, color in lut.items()]
    plt.legend(handles=handles, title="Treatment", bbox_to_anchor=(1.02, 0.8), loc='upper left', borderaxespad=0.)

    plt.title(f"Suppression Signature: {pair_name}\n(Z-scored Expression)")

    save_path = f"{output_dir}/Heatmap_Suppression_{pair_name}.png"
    # save_path = f"{output_dir}/Heatmap_Suppression_{pair_name}_clustered.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {save_path}")


def verify_suppression_significance(counts, meta, drug_A, drug_B, n_shuffles=10):
    """
    Permutation Test for Suppression Analysis.
    Shuffles the labels of antibiotic-treated samples (A, B, Pair) while keeping PBS fixed.
    Checks if the number of suppressed genes found is higher than random chance.
    """
    output = ''
    pair_name = f"{drug_A}+{drug_B}"
    print(f"\n--- Verifying Suppression for {pair_name} (Permutation Test) ---")

    # 1. Get Baseline (Real Result)
    print("Running on TRUE labels...")
    real_res = run_suppression_analysis(counts, meta, drug_A, drug_B)
    real_count = len(real_res)
    print(f"-> Real Suppressed Genes: {real_count}")

    if real_count == 0:
        print(f"No genes to verify for {pair_name}. Skipping.")
        output += f"No suppressed genes found for {pair_name}. Permutation test skipped.\n"
        # Save the output to a text file
        with open(f"./Private/Noise/Suppression_Validation_{pair_name}.txt", "w") as f:
            f.write(output)
        return

    # 2. Prepare for Shuffling
    # We only shuffle samples that belong to A, B, or Pair. PBS stays PBS.
    target_groups = [drug_A, drug_B, pair_name]

    # Identify the samples to shuffle
    shuff_meta = meta.copy()
    mask_abx = shuff_meta['Drug'].isin(target_groups)

    # Get the list of labels and samples
    abx_samples = shuff_meta.loc[mask_abx].index.tolist()  # Assuming index is meaningless, we rely on row order
    abx_labels = shuff_meta.loc[mask_abx, 'Drug'].values.copy()

    shuffle_counts = []

    print(f"Running {n_shuffles} shuffles (this may take time)...")

    # 3. Permutation Loop
    for i in tqdm(range(n_shuffles)):
        # Shuffle the labels array
        np.random.shuffle(abx_labels)

        # Assign back to the metadata
        temp_meta = meta.copy()
        # We need to ensure we assign to the correct rows
        temp_meta.loc[mask_abx, 'Drug'] = abx_labels

        # Run the Suppression Analysis with "Quiet" mode if possible to avoid spamming console
        # We wrap it in try/except to ensure one bad shuffle doesn't crash the loop
        try:
            # Note: We rely on the function defined previously.
            # Ideally, modify run_suppression_analysis to accept a 'verbose' arg to silence it.
            res = run_suppression_analysis(counts, temp_meta, drug_A, drug_B)
            shuffle_counts.append(len(res))
        except Exception as e:
            print(f"Shuffle {i} failed: {e}")
            shuffle_counts.append(0)

    # 4. Calculate Statistics
    mean_shuff = np.mean(shuffle_counts)
    std_shuff = np.std(shuffle_counts)

    # Z-Score: How many standard deviations is the real result away from the random mean?
    if std_shuff > 0:
        z_score = (real_count - mean_shuff) / std_shuff
    else:
        z_score = np.inf if real_count > mean_shuff else 0

    print(f"\n--- Permutation Results for {pair_name} ---")
    print(f"Real Count:      {real_count}")
    print(f"Shuffle Mean:    {mean_shuff:.2f} ± {std_shuff:.2f}")
    print(f"Z-Score:         {z_score:.2f}")
    output += f"Permutation Test for {pair_name}\n"
    output += f"Real Suppressed Genes: {real_count}\n"
    output += f"Shuffle Mean: {mean_shuff:.2f}\n"
    output += f"Shuffle Std Dev: {std_shuff:.2f}\n"
    output += f"Z-Score: {z_score:.2f}\n"

    # Save validation stats
    stats_df = pd.DataFrame({
        'Metric': ['Real_Count', 'Shuffle_Mean', 'Shuffle_Std', 'Z_Score', 'N_Shuffles'],
        'Value': [real_count, mean_shuff, std_shuff, z_score, n_shuffles]
    })
    stats_df.to_csv(f"./Private/Noise/Suppression_Validation_{pair_name}.csv", index=False)

    # Visual check
    if z_score > 3:
        print("CONCLUSION: VALID. The suppression is significantly distinct from random noise.")
    else:
        print("CONCLUSION: WARNING. The number of suppressed genes is close to what we expect by chance.")


if __name__ == "__main__":
    data, metadata, id_to_name = read_data_metadata(remove_mitochondrial=False)
    antibiotics = metadata['Drug'].unique().tolist()
    antibiotics.remove('PBS')
    treatments = metadata['Treatment'].unique().tolist()
    run_type = "_pairs"
    data, metadata = transform_data(data, metadata, run_type, skip=True)
    # compute_all_genes_statistics_pairs(["Met", "Van", "Neo"])
    # exit()
    # run_corrEnrich(data, metadata, run_type, antibiotics, treatments, id_to_name)
    # exit()
    # --- Main Execution ---
    # Define pairs to analyze
    # Assuming metadata['Treatment'] contains: 'Met', 'Van', 'Neo', 'Met_Van', 'Met_Neo'
    pairs = [
        {'pair': 'Met+Van', 'A': 'Met', 'B': 'Van'},
        {'pair': 'Met+Neo', 'A': 'Met', 'B': 'Neo'}
    ]
    counts_df, metadata_df, id_map = read_data_metadata(t=True, remove_mitochondrial=True, normalize=False)
    # # save background genes
    # bg_genes = pd.DataFrame({'gene_id': counts_df.columns, 'gene_name': counts_df.columns.map(id_map)})
    # bg_genes.to_csv(f"./Private/Noise/Background_Genes.csv", index=False)

    run_analysis(counts_df, metadata_df, id_map, pairs)
    exit()
    # plot_partial_results(counts_df, metadata_df, id_map, pairs)
    # plot_partial_neo_results(counts_df, metadata_df, id_map, pairs)
    # Generate Suppression Heatmaps
    for config in pairs:
        # Plot Heatmap
        plot_suppression_heatmap(data, metadata, config['A'], config['B'], id_map)
        verify_suppression_significance(counts_df, metadata_df, config['A'], config['B'], n_shuffles=10)
        # verify_suppression_significance(counts_df, metadata_df, config['A'], config['B'], n_shuffles=1_000)
