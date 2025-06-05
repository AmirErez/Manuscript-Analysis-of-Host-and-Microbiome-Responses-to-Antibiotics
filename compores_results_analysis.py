import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ClusteringGO import antibiotics, treatments, read_process_files, transform_data
from random_forests import get_ensmus_dict


def get_significant(current, abx_data, pbs_data, threshold=0.05):
    from scipy.stats import ttest_ind

    genes = []
    for gene in current.index:
        # get treat-test score for the gene
        abx = (current.loc[gene][abx_data['ID']])
        pbs = (current.loc[gene][pbs_data['ID']])
        t_pbs, t_p_pbs = ttest_ind(pbs, abx)
        # if ensmus_to_gene[gene].lower() == "hspb8":
        #     ensmus_to_gene = get_ensmus_dict()
        #     # Combine the data into a DataFrame for easier plotting
        #     data = pd.DataFrame({
        #         'Condition': ['PBS IP'] * len(pbs) + ['Van IP'] * len(abx),
        #         'CPM': list(pbs) + list(abx)
        #     })
        #
        #     # Plotting
        #     fig, ax = plt.subplots(figsize=(5, 5))
        #
        #     # Scatter plot
        #     colors = ['#1f77b4', '#ff7f0e']  # Colors for the groups
        #     for (condition, group), color in zip(data.groupby('Condition'), colors):
        #         ax.scatter([condition] * len(group), group['CPM'], color=color, label=condition, s=50)
        #
        #     # Aesthetic settings
        #     ax.set_title(gene, fontsize=14)
        #     ax.set_ylabel('Transformed Value', fontsize=12)
        #     ax.set_xlabel('', fontsize=12)
        #     # ax.set_ylim(0, max(data['CPM']) + 20)  # Adjust y-axis
        #     ax.spines['top'].set_visible(False)
        #     ax.spines['right'].set_visible(False)
        #     plt.tight_layout()
        #     plt.show()
        if abs(t_p_pbs) < threshold:
            genes.append(gene)
    return genes


def prepare_genes_to_compores(threshold=0.05, by_genes=False, folder=None):
    from ClusteringGO import transform_data
    genome, metadata, partek, transcriptome = read_process_files(new=False)
    transcriptome, metadata = transform_data(transcriptome, metadata, "RASflow", skip=True)
    metadata["Category"] = metadata['Drug'] + "_" + metadata['Treatment']
    # if f"./Private/CompoResGenes/{folder}" does not exist, create it
    curr_path = os.path.join("Private", "CompoResGenes", folder)
    if folder and os.path.exists(curr_path) is False:
        os.makedirs(curr_path)
        # create folder "response" and "metadata"
        os.makedirs(os.path.join(curr_path, "response"), exist_ok=True)
        os.makedirs(os.path.join(curr_path, "metadata"), exist_ok=True)
    for treat in treatments:
        for abx in antibiotics:
            samples = metadata[
                ((metadata["Drug"] == abx) | (metadata["Drug"] == "PBS")) & (metadata["Treatment"] == treat)]
            curr = transcriptome[samples["ID"]]
            if not by_genes:
                abx_data = metadata[(metadata['Drug'] == abx) & (metadata["Treatment"] == treat)]
                pbs_data = metadata[(metadata['Drug'] == 'PBS') & (metadata["Treatment"] == treat)]
                genes = get_significant(curr, abx_data, pbs_data, threshold=threshold)
            else:
                genes = [gene for gene in by_genes if gene in curr.index]
                print(f"{len(genes)} are available out of original {len(by_genes)}")
            curr_genes = curr.T[genes]
            # curr_genes.to_csv(f"./Private/feeding/{abx}-{treat}.tsv", sep="\t")
            addition = f"{folder}/response/" if folder else ""
            compo_path = os.path.join("Private", "CompoResGenes")
            curr_genes.to_csv(os.path.join(compo_path + f"{addition}{abx}-{treat}.tsv"), sep="\t")
            print(f"Number of significant genes for {abx}-{treat}: {len(genes)}")
            save_meta = samples.set_index("ID")['Category']
            addition = f"{folder}/metadata/" if folder else ""
            save_meta.to_csv(os.path.join(compo_path + f"{addition}{abx}-{treat}-metadata.tsv"), sep="\t")


def calc_multi_abx_statistics(ttest=True):
    path = os.path.join("Private", "analysis", "Diff_abxRASflow", "all_stats_multi_abx.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    _, meta, _, df = read_process_files(new=False)
    df, meta = transform_data(df, meta, "RASflow", skip=True)

    from scipy.stats import ttest_ind
    all_stats = pd.DataFrame()
    all_stats.index = df.index
    for abx in antibiotics:
        for treat in treatments:
            abx_data = meta[(meta['Drug'] == abx) & (meta["Treatment"] == treat)]['ID'].values
            pbs_data = meta[(meta['Drug'] == 'PBS') & (meta["Treatment"] == treat)]['ID'].values
            # temp = raw.loc[np.concatenate(abx_data, pbs_data)]

            p_values = df.apply(lambda row: ttest_ind(row[pbs_data], row[abx_data])[1], axis=1)
            # else:
            #     p_values = df.apply(lambda row: ttest_ind(row[abx_data], row[pbs_data])[1], axis=1)

            # Calculate the fold changes
            fold_changes = df.apply(lambda row: np.median(row[abx_data]) - np.median(row[pbs_data]), axis=1)
            enhanced = df.apply(lambda row: np.median(row[abx_data]) > np.median(row[pbs_data]), axis=1)
            # add the p-values and fold changes to the all_stats df
            all_stats = pd.concat([all_stats, pd.DataFrame(
                {f"p-value_{abx}_{treat}": p_values, f"fold_change_{abx}_{treat}": fold_changes,
                 f"enhanced_{abx}_{treat}": enhanced}, index=df.index)], axis=1)
    all_stats.to_csv(path)
    return all_stats


def read_and_print_pkl(path):
    import pickle
    try:
        # Open the pickle file and load the dictionary
        with open(path, 'rb') as file:
            data = pickle.load(file)

        # Ensure the loaded object is a dictionary
        if isinstance(data, dict):
            # Print the first 5 elements (key-value pairs)
            for idx, (key, value) in enumerate(data.items()):
                if idx < 5:
                    print(f"{idx + 1}. {key}: {value}")
                else:
                    break
        else:
            print("The loaded object is not a dictionary.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return data


def show_case_correlated_genes():
    """
    showcase a couple of genes that are the most correlated (lowest p-value) with the microbiome. Saves the tables with
    the p-value of expression, p-value of compoRes, and the fold change in gene expression, and gene name.
    """
    ensmus_to_gene = get_ensmus_dict()
    path = r"D:\Master heavy files\CompoResAllConditions"
    multi_abx = calc_multi_abx_statistics()
    all_results = pd.DataFrame()
    for j, treat in enumerate(treatments):
        for i, abx in enumerate(antibiotics):
            column_names = ['gene', '-log(p-value) correlation']
            data = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\mean_log_p_value.pkl')
            # path + fr'\{abx}-{treat}-feces\mean_log_p_value.pkl')
            data = data[f"{abx}-{treat}-feces"]
            index = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\response_index.pkl')
            compores_results_spf = pd.DataFrame({column_names[0]: index, column_names[1]: data})
            compores_results_spf[f'{column_names[1]}_p'] = np.exp(-compores_results_spf[column_names[1]])
            compores_results_spf["genes_name"] = compores_results_spf[column_names[0]].str.split('_').str[-1]
            compores_results_spf = compores_results_spf.set_index("genes_name")
            # drop gene column
            # compores_results_spf = compores_results_spf.drop(column_names[0], axis=1)
            compores_results_spf = compores_results_spf.rename(
                columns={"-log(p-value) correlation_p": "p-value correlation"})
            spf = multi_abx[[f"p-value_{abx}_{treat}", f"fold_change_{abx}_{treat}"]]
            # spf = multi_abx[[f"p-value_{abx}_{treat}", f"fold_change_{abx}_{treat}", f"enhanced_{abx}_{treat}"]]
            spf_significant = spf[spf[f"p-value_{abx}_{treat}"] < 0.01]
            # rename column f"p-value_{abx}_{treat}" to f"t-test p-value"
            spf_significant = spf_significant.rename(columns={f"p-value_{abx}_{treat}": "t-test p-value",
                                                              f"fold_change_{abx}_{treat}": "fold change"})
            # merge the two dataframes based on index
            merged = pd.merge(compores_results_spf, spf_significant, left_on="genes_name", right_index=True)
            merged.index = [ensmus_to_gene.get(gene, gene) for gene in merged.index]
            # print(spf_significant.loc[list(set(spf_significant.index) - set(compores_results_spf.index))])
            merged["abx"] = abx
            merged["treat"] = treat
            # sort by f'{column_names[1]}_p', smallest to largest
            merged = merged.sort_values(by="p-value correlation")
            # save df
            merged.to_csv(f"./Private/CompoResGenes/{abx}-{treat}-results.tsv", sep="\t")
            # add to all_results
            all_results = pd.concat([all_results, merged], ignore_index=False)
    all_results = all_results.sort_values(by="p-value correlation")
    all_results.to_csv(f"./Private/CompoResGenes/all-results.tsv", sep="\t")


def prepare_clock_genes_to_compores(genes):
    from ClusteringGO import transform_data
    from random_forests import get_ensmus_dict

    genome, metadata, partek, transcriptome = read_process_files(new=False)
    transcriptome, metadata = transform_data(transcriptome, metadata, "RASflow", skip=True)
    metadata["Category"] = metadata['Drug'] + "_" + metadata['Treatment']
    # abx = "Van"
    # treat = "IP"
    for abx in antibiotics:
        for treat in treatments:
            samples = metadata[
                ((metadata["Drug"] == abx) | (metadata["Drug"] == "PBS")) & (metadata["Treatment"] == treat)]
            curr = transcriptome[samples["ID"]]
            ensmus_to_gene = get_ensmus_dict()
            curr.index = [ensmus_to_gene.get(gene, gene) for gene in curr.index]
            curr_genes = curr.T[genes]
            # curr_genes.to_csv(f"./Private/feeding/{abx}-{treat}.tsv", sep="\t")
            curr_genes.to_csv(f"./Private/CompoResGenes/clock/{abx}-{treat}.tsv", sep="\t")
            save_meta = samples.set_index("ID")['Category']
            save_meta.to_csv(f"./Private/CompoResGenes/clock/{abx}-{treat}-metadata.tsv", sep="\t")


def calc_gf_statistics(ttest=True):
    path = "./Private/Lilach gf/new_gf_stats_CompoRes_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    df = pd.read_csv("../Data/Lilach GF/genes_norm_named.csv")
    meta = pd.read_csv("../Data/Lilach GF/metadata-LilachGF-2024-05-29.tsv", sep="\t")
    meta["sex"] = "female"
    # S1–S5 are males
    females_abx = [f"S{num}" for num in range(1, 6)]
    # S12-S16 are males
    females_pbs = [f"S{num}" for num in range(12, 17)]
    meta.loc[meta["sample"].isin(females_abx), "sex"] = "male"
    meta.loc[meta["sample"].isin(females_pbs), "sex"] = "male"

    # drop all females
    meta = meta[meta["sex"] == "male"]
    from ClusteringGO import mitochondrial_genes
    df = df.drop("gene_id", axis=1).set_index("gene_name")
    matching_indices = df.index[
        df.index.str.lower().isin(set(mitochondrial_genes))].tolist()
    # remove mitochondrial genes from the dataframes
    df = df.drop(matching_indices, errors='ignore')
    samples = meta["sample"].values
    # keep only the samples in the data
    df = df[samples]
    # sum all rows with the same gene_name
    df = df.groupby(df.index).sum()
    # normalize the data so column will sum to 1_000_000
    sum_reads = df.sum(axis=0)
    # print the sum of reads for each sample in scientific notation
    print(f"sum of reads for each sample: {sum_reads.mean():.1E}", f"+-{sum_reads.std():.1E}")
    df = df.div(df.sum(axis=0), axis=1) * 1_000_000
    # remove empty rows
    df = df.loc[~(df == 0).all(axis=1)]
    # rename "sample" to "ID"
    meta = meta.rename(columns={"sample": "ID", "group": "Drug"})
    meta["Treatment"] = "GF"
    df, meta = transform_data(df, meta, "_gf", skip=False, gf=True)
    # return "ID" to "sample"
    meta = meta.rename(columns={"ID": "sample", "Drug": "group"})
    # drop Treatment column
    meta = meta.drop("Treatment", axis=1)
    from scipy.stats import ttest_ind
    all_stats = pd.DataFrame()
    all_stats.index = df.index
    abx_data = meta[meta['group'] == "Van"]['sample'].values
    pbs_data = meta[meta['group'] == 'PBS']['sample'].values
    # temp = raw.loc[np.concatenate(abx_data, pbs_data)]

    # if ttest:
    #     p_values = df.apply(lambda row: ttest_ind(row[abx_data], row[pbs_data])[1], axis=1)
    # else:
    p_values = df.apply(lambda row: ttest_ind(row[abx_data], row[pbs_data])[1], axis=1)

    # Calculate the fold changes
    fold_changes = df.apply(lambda row: np.log2(np.median(row[abx_data]) / np.median(row[pbs_data])), axis=1)
    # calculate if enhanced?
    enhanced = df.apply(lambda row: np.median(row[abx_data]) > np.median(row[pbs_data]), axis=1)
    # add the p-values and fold changes to the all_stats df
    all_stats = pd.concat([all_stats, pd.DataFrame(
        {f"p-value_gf": p_values, f"fold_change_gf": fold_changes, "enhanced": enhanced}, index=df.index)], axis=1)
    all_stats.to_csv(path)
    return all_stats


def compare_correlation_all():
    # multiabx = calc_multi_abx_statistics()
    # genome, metadata, partek, transcriptome = read_process_files(new=False)
    # transcriptome, metadata = transform_data(transcriptome, metadata, "RASflow", skip=True)

    # folder_dir = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
    # temp = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
    # ensmus_to_gene = temp.set_index('gene_id')['gene_name'].to_dict()
    from clusters_plot import set_figure, get_to_axis
    axis = set_figure(treatments, antibiotics)
    uncorrelated = pd.DataFrame(columns=antibiotics, index=treatments)
    path = r"D:\Master heavy files\CompoResAllConditions"

    data_list = []
    from scipy.stats import ttest_ind
    for i, abx in enumerate(antibiotics):
        for j, treat in enumerate(treatments):
            column_names = ['gene', 'correlation', 'rmse']
            rms = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\mean_rmse.pkl')
            # path + fr'\{abx}-{treat}-feces\mean_log_p_value.pkl')
            rms = rms[f"{abx}-{treat}-feces"]
            data = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\mean_log_p_value.pkl')
            # path + fr'\{abx}-{treat}-feces\mean_log_p_value.pkl')
            data = data[f"{abx}-{treat}-feces"]
            index = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\response_index.pkl')
            compores_results_spf = pd.DataFrame({column_names[0]: index, column_names[1]: data, column_names[2]: rms})
            compores_results_spf[f'{column_names[1]}_p'] = np.exp(-compores_results_spf[column_names[1]])
            compores_results_spf['genes_name'] = compores_results_spf['gene'].str.split('_').str[-1]
            compores_results_spf['treatment'] = treat
            compores_results_spf['antibiotic'] = abx
            data_list.append(compores_results_spf)
    # Combine all data
    all_data = pd.concat(data_list, axis=0)
    # Create a single figure with subplots for each antibiotic
    n_abx = len(antibiotics)
    fig, axes = plt.subplots(nrows=1, ncols=n_abx, figsize=(10 * n_abx, 5), sharey=True)
    # Plot for each antibiotic
    for idx, abx in enumerate(antibiotics):
        subset = all_data[all_data['antibiotic'] == abx]
        # Create box plot
        # plt.figure(figsize=(8, 6))
        sns.boxplot(data=subset, x='treatment', y='correlation_p', palette='Set2', ax=axes[idx])
        plt.title(f"Correlation P-values by Treatment for {abx}")
        plt.ylabel("P-value")
        plt.xlabel("Treatment")
        # Perform t-tests
        treatments_unique = subset['treatment'].unique()
        y_max = subset['correlation_p'].max()
        # Define levels for annotations
        base_y = y_max + 0.05
        level_1 = base_y
        level_2 = base_y + 0.1  # Higher level for comparison 1-3
        for i in range(len(treatments_unique)):
            for j in range(i + 1, len(treatments_unique)):
                treat1 = treatments_unique[i]
                treat2 = treatments_unique[j]
                data1 = subset[subset['treatment'] == treat1]['correlation_p']
                data2 = subset[subset['treatment'] == treat2]['correlation_p']
                t_stat, p_val = ttest_ind(data1, data2)
                # Define annotation level
                if (i == 0 and j == 1) or (i == 1 and j == 2):  # Comparisons 1-2 and 2-3
                    y, h = level_1, 0.02
                elif i == 0 and j == 2:  # Comparison 1-3
                    y, h = level_2, 0.02
                # Plot lines and annotations
                x1, x2 = i, j
                axes[idx].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
                # Add significance level
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                axes[idx].text((x1 + x2) * 0.5, y + h, sig, ha='center', va='bottom', color='k')
        # set ylim to 0,1
        axes[idx].set_ylim(0, 1.2)
        # set subtitle for each subplot
        axes[idx].set_title(f"{abx}")
        # plt.savefig(f"./Private/compores_response_ranking/correlation_p_values_{abx}.png")
    plt.tight_layout()
    plt.savefig("./Private/compores_response_ranking/correlation_p_values_combined.png")
    # plt.show()
    plt.close()

    for j, treat in enumerate(treatments):
        for i, abx in enumerate(antibiotics):
            curr_axis = get_to_axis(axis, i, j, len(treatments), len(antibiotics))
            curr_axis.set_title(f"{abx}, {treat}")
            column_names = ['gene', 'correlation', 'rmse']
            rms = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\mean_rmse.pkl')
            # path + fr'\{abx}-{treat}-feces\mean_log_p_value.pkl')
            rms = rms[f"{abx}-{treat}-feces"]
            data = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\mean_log_p_value.pkl')
            # path + fr'\{abx}-{treat}-feces\mean_log_p_value.pkl')
            data = data[f"{abx}-{treat}-feces"]
            index = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\response_index.pkl')
            compores_results_spf = pd.DataFrame({column_names[0]: index, column_names[1]: data, column_names[2]: rms})
            # # keep only rows where the rmse is smaller than the quantile(0.25) rmse
            # compores_results_spf = compores_results_spf[compores_results_spf[column_names[2]] < compores_results_spf[
            #     column_names[2]].quantile(0.1)]
            # df_sorted = compores_results_spf.sort_values(by=column_names[1])
            compores_results_spf[f'{column_names[1]}_p'] = np.exp(-compores_results_spf[column_names[1]])
            compores_results_spf["genes_name"] = compores_results_spf[column_names[0]].str.split('_').str[-1]
            # plot a histogram of the p-values
            n, bins, patches = curr_axis.hist(compores_results_spf[f'{column_names[1]}_p'], bins=50, color='skyblue',
                                              edgecolor='black',
                                              density=True)
            # plot horizontal line for uniform distribution
            uniform_density = 1  # Density for a uniform distribution over the data range
            curr_axis.axhline(y=uniform_density, color='k', linestyle='--', label='Uniform Distribution')
            curr_axis.legend()
            curr_axis.set_xlabel('p-value')
            curr_axis.set_ylabel('Frequency')
            curr_axis.set_title(
                f'p-value distribution for {abx}, {treat} \n(total # of significant genes: {len(compores_results_spf)})')
            uncorrelated.loc[treat, abx] = len(
                compores_results_spf[compores_results_spf[f"{column_names[1]}_p"] >= 0.05]) / len(compores_results_spf)
    plt.savefig(f"./Private/compores_response_ranking/intersection_distribution.png")
    # plt.show()
    plt.close()
    # Plot the fraction of genes that are significant and that have p>=0.05
    uncorrelated_filled = uncorrelated.fillna(0).T
    # sns.heatmap(uncorrelated_filled, cmap='coolwarm', cbar_kws={'label': 'Fraction of genes with p>=0.05'})
    sns.heatmap(uncorrelated_filled, annot=True, fmt=".3f", cmap='coolwarm',
                cbar_kws={'label': 'Number of genes with p>=0.05'})
    plt.title("Fraction of genes with p>=0.05")
    plt.savefig(f"./Private/compores_response_ranking/uncorrelated_genes.png")
    plt.show()


def intersection_evaluation(compores_results_spf_intersect, random_compores_results_spf_not_intersect, threshold,
                            smaller, N, K):
    from scipy.stats import hypergeom
    if smaller:
        focus_spf_intersect = compores_results_spf_intersect[compores_results_spf_intersect < threshold]
        focus_spf_not_intersect = random_compores_results_spf_not_intersect[
            random_compores_results_spf_not_intersect < threshold]
    else:
        focus_spf_intersect = compores_results_spf_intersect[compores_results_spf_intersect > threshold]
        focus_spf_not_intersect = random_compores_results_spf_not_intersect[
            random_compores_results_spf_not_intersect > threshold]
    n = len(focus_spf_intersect) + len(focus_spf_not_intersect)
    k = len(focus_spf_intersect)
    # Calculate hypergeometric distribution mean and std
    mean = (K * n) / N
    std = ((K * n * (N - K) * (N - n)) / (N ** 2 * (N - 1))) ** 0.5

    # Print all parameters and the mean and standard deviation
    print(f"Threshold: {threshold}")
    print(f"Smaller: {smaller}")
    print(f"Total population size (N): {N}")
    print(f"Intersection size (K): {K}")
    print(f"Number of draws (n): {n}")
    print(f"Number of observed successes (k): {k}")
    print(f"Hypergeometric mean: {mean}")
    print(f"Hypergeometric standard deviation: {std}")

    # Calculate the hypergeometric probability of observing at least `k` successes in `n` draws
    p_value = hypergeom.sf(k - 1, N, K, n)  # sf gives P(X >= k)
    print(f"Hypergeometric p-value (P(X >= {k})): {p_value}")
    p_value = hypergeom.cdf(k, N, K, n)  # cdf gives P(X <= k)
    print(f"Hypergeometric p-value (P(X <= {k})): {p_value}")

    return {"mean": mean, "std": std, "p_value": p_value}


def simulate_intersections(N1, N2, n1, n2, compare, p=0.05, iterations=10_000):
    """
    Simulate the intersection of two groups of genes for a given number of iterations.

    Parameters:
    N1 (int): The first group (set).
    N2 (int): The second group (set).
    n1 (int): Number of genes chosen from the first group.
    n2 (int): Number of genes chosen from the second group.
    compare (int): The observed intersection size.
    iterations (int): Number of simulations to run (default 10,000).

    Returns:
    tuple: Mean and standard deviation of the intersection sizes.
    """
    # Track the intersection sizes
    intersection_sizes = np.zeros(iterations, dtype=int)

    for i in range(iterations):
        # Randomly pick n1 genes from N1 and n2 genes from N2
        chosen_from_N1 = set(np.random.choice(N1, n1, replace=False))
        chosen_from_N2 = set(np.random.choice(N2, n2, replace=False))

        # Calculate the size of the intersection
        intersection_sizes[i] = len(chosen_from_N1.intersection(chosen_from_N2))

    # Calculate mean and std of the intersection sizes
    mean_intersection = np.mean(intersection_sizes)
    std_intersection = np.std(intersection_sizes)
    # Calculate the p-value as the fraction of simulations with intersection >= compare
    p_value = np.sum(intersection_sizes >= compare) / iterations

    # plot a histogram of the intersection sizes
    plt.hist(intersection_sizes, bins=50, color='skyblue', edgecolor='black', density=True)
    plt.axvline(compare, color='red', linestyle='dashed', linewidth=1)
    plt.legend([f'Observed intersection size: {compare}'])
    plt.xlabel('Intersection Size')
    plt.ylabel('Frequency')
    plt.title(f'Intersection Size Distribution (p={p})')
    # plt.savefig(f"./Private/selected_clusters/intersection_distribution.png")
    # plt.show()
    plt.close()

    return mean_intersection, std_intersection, p_value


def compare_correlation_gf(abx, treat, threshold=0.05):
    genome, metadata, partek, transcriptome = read_process_files(new=False)
    transcriptome, metadata = transform_data(transcriptome, metadata, "RASflow", skip=True)

    folder_dir = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
    temp = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
    ensmus_to_gene = temp.set_index('gene_id')['gene_name'].to_dict()

    column_names = ['gene', 'correlation', 'rmse']
    path = fr"D:\Master heavy files\CompoResAllConditions\bootstrap\50"
    # if False:
    res_path = f"./Private/CompoResVerification/{abx}-{treat}-res{'0_05' if threshold == 0.05 else ''}.tsv"
    if False:
        # if os.path.exists(res_path):
        compores_results_spf = pd.read_csv(res_path, sep="\t")
    else:
        rms = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\pairs\mean_rmse.pkl')
        # path+fr'\{abx}-{treat}-feces\mean_log_p_value.pkl')
        rms = rms[f"{abx}-{treat}-feces"]
        data = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\pairs\mean_log_p_value.pkl')
        # path+fr'\{abx}-{treat}-feces\mean_log_p_value.pkl')
        data = data[f"{abx}-{treat}-feces"]
        index = read_and_print_pkl(path + fr'\{abx}-{treat}-feces\pairs\response_index.pkl')
        compores_results_spf = pd.DataFrame({column_names[0]: index, column_names[1]: data, column_names[2]: rms})
        # # keep only rows where the rmse is smaller than the quantile(0.25) rmse
        # compores_results_spf = compores_results_spf[compores_results_spf[column_names[2]] < compores_results_spf[
        #     column_names[2]].quantile(0.1)]
        # df_sorted = compores_results_spf.sort_values(by=column_names[1])
        compores_results_spf[f'{column_names[1]}_p'] = np.exp(-compores_results_spf[column_names[1]])
        compores_results_spf["genes_name"] = compores_results_spf[column_names[0]].str.split('_').str[-1]

        # # df_sorted = compores_results_spf.sort_values(by=f"{column_names[1]}_p")
        # correlation_values = get_corr_vals(abx, treat)
        # correlation_values = pd.DataFrame(correlation_values.items(), columns=['genes_name', 'microbiome_correlation'])
        # # merge the correlation values with the compores results by the gene name (and the key)
        # compores_results_spf = compores_results_spf.merge(correlation_values, on='genes_name')
        # add column genes_id using genes_name and ensmus_to_gene
        compores_results_spf["genes_id"] = compores_results_spf["genes_name"].apply(lambda x: ensmus_to_gene[x])
        # save this csv
        compores_results_spf.to_csv(res_path, sep="\t", index=False)

    gf = calc_gf_statistics()
    gf_significant_genes = gf[gf["p-value_gf"] < threshold].index.to_list()
    gf_significant = gf[gf["p-value_gf"] < threshold]
    multiabx = calc_multi_abx_statistics()
    van = multiabx[[f"p-value_{abx}_{treat}", f"fold_change_{abx}_{treat}", f"enhanced_{abx}_{treat}"]]
    van_significant = van[van[f"p-value_{abx}_{treat}"] < threshold]
    van["gene_name"] = van.apply(lambda row: ensmus_to_gene[row.name], axis=1)
    van_significant.index = [ensmus_to_gene[mus] for mus in van_significant.index]
    # set index to be gene_name
    van.set_index("gene_name", inplace=True)
    intersection_gf_spf = list(
        set(van_significant.index).intersection(gf_significant.index))
    van_significant_aligned, gf_significant_aligned = van_significant.align(gf_significant, join="inner", axis=0)
    intersection_gf_spf = van_significant_aligned.index[
        van_significant_aligned[f'enhanced_{abx}_{treat}'] == gf_significant_aligned['enhanced']]

    # save this csv
    intersection_gf_spf = pd.DataFrame(intersection_gf_spf, columns=["genes_id"])
    intersection_gf_spf.to_csv(
        f"./Private/CompoResVerification/{abx}-{treat}-intersection-GF{'0_05' if threshold == 0.05 else ''}.tsv",
        sep="\t",
        index=False)

    # choose len(intersection) genes from the compores_results_spf that are not in intersection_gf_spf
    compores_results_spf_not_intersect = compores_results_spf[
        ~compores_results_spf["genes_id"].isin(intersection_gf_spf["genes_id"])]
    # choose len(intersection) genes from the compores_results_spf_not_intersect
    # random_compores_results_spf_not_intersect = compores_results_spf_not_intersect["correlation_p"].sample(n=len(intersection_gf_spf))
    random_compores_results_spf_not_intersect = compores_results_spf_not_intersect["correlation_p"]
    compores_results_spf_intersect = \
        compores_results_spf[compores_results_spf["genes_id"].isin(intersection_gf_spf["genes_id"])][f"correlation_p"]
    # save compores_results_spf_intersect to a csv
    to_save = compores_results_spf[compores_results_spf["genes_id"].isin(intersection_gf_spf["genes_id"])]
    to_save[["genes_id", "genes_name", "correlation_p"]].to_csv(
        f"./Private/CompoResVerification/{abx}-{treat}-intersect-GF{'0_05' if threshold == 0.05 else ''}.tsv", sep="\t",
        index=False)
    return
    alpha = 0.3
    bins = 40
    # plt.hist(compores_results_spf["correlation_p"], bins=bins, color='black', edgecolor='black', density=True,
    plt.hist(compores_results_spf["correlation_p"], bins=bins, color='black', edgecolor='black',
             alpha=alpha, label="All")
    plt.hist(random_compores_results_spf_not_intersect, bins=bins, color='red', edgecolor='black',
             alpha=alpha, label="Not intersecting")
    # density=True, alpha=alpha, label="Not intersecting")
    plt.hist(compores_results_spf_intersect, bins=bins, color='blue', edgecolor='black',
             alpha=alpha, label="Intersecting")
    # density=True, alpha=alpha, label="Intersecting")
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title(f'p-value distribution for {abx}, {treat}')
    plt.legend()
    plt.show()
    # Do MWU test between random_compores_results_spf_not_intersect and compores_results_spf_intersect
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(random_compores_results_spf_not_intersect, compores_results_spf_intersect)
    print(f"GF MWU test between random_compores_results_spf_not_intersect and compores_results_spf_intersect: p={p}")
    # print which median is greater
    if random_compores_results_spf_not_intersect.median() > compores_results_spf_intersect.median():
        print("The median of random_compores_results_spf_not_intersect is greater")
    else:
        print("The median of compores_results_spf_intersect is greater")
    stat, p = mannwhitneyu(random_compores_results_spf_not_intersect, compores_results_spf_intersect,
                           alternative='less')
    print(f"MWU test between random_compores_results_spf_not_intersect and compores_results_spf_intersect: p={p}")

    intersection_evaluation(compores_results_spf_intersect, random_compores_results_spf_not_intersect,
                            threshold=0.05, smaller=True, N=len(compores_results_spf),
                            K=len(compores_results_spf_intersect))
    intersection_evaluation(compores_results_spf_intersect, random_compores_results_spf_not_intersect,
                            threshold=0.5, smaller=False, N=len(compores_results_spf),
                            K=len(compores_results_spf_intersect))
    intersection_evaluation(compores_results_spf_intersect, random_compores_results_spf_not_intersect,
                            threshold=compores_results_spf["correlation_p"].median(), smaller=False,
                            N=len(compores_results_spf),
                            K=len(compores_results_spf_intersect))

    focus_spf_intersect = compores_results_spf_intersect[compores_results_spf_intersect < 0.05]
    focus_spf_not_intersect = random_compores_results_spf_not_intersect[
        random_compores_results_spf_not_intersect < 0.05]
    # plot histograms of p-values for both groups
    plt.hist(focus_spf_intersect, bins=10, color='skyblue', edgecolor='black', density=True, label="intersecting")
    plt.hist(focus_spf_not_intersect.sample(n=len(focus_spf_intersect)), bins=10, color='red', edgecolor='black',
             density=True, alpha=0.5, label="Not intersecting")
    plt.axvline(0.05, color='k', linestyle='--', label='p=0.05')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title(f'p-value distribution for {abx}, {treat}')
    plt.legend()
    plt.show()

    # Plot -logp vs. correlation
    plt.scatter(compores_results_spf['microbiome_correlation'], compores_results_spf[f'{column_names[1]}_p'], s=5,
                label="unique")
    compores_genes_raw = [ensmus_to_gene[mus] for mus in compores_results_spf[column_names[0]].str.split('_').str[-1]]
    intersection_compo_gf = list(set(compores_genes_raw).intersection(gf_significant_genes))
    # scatter plot intersection genes in red
    mask = compores_results_spf["genes_id"].isin(intersection_compo_gf)
    plt.scatter(compores_results_spf[mask]['microbiome_correlation'],
                compores_results_spf[mask][f'{column_names[1]}_p'],
                c='red', s=5, label="intersect")
    # add horizontal line at 0.05
    plt.axhline(y=0.05, color='k', linestyle='--', label='p=0.05')
    plt.xlabel('Correlation (OTU level)')
    plt.ylabel('p-value')
    plt.legend()
    plt.title(f'Correlation vs. p-value for {abx}-{treat}')
    plt.show()
    # plot percentage of intersection genes out of all genes as a function of microbiome correlation
    correlation_cuttoff = np.linspace(0.9, 0.999, 20)
    percentage = np.zeros(len(correlation_cuttoff) - 1)
    for i in range(len(correlation_cuttoff) - 1):
        mask = (compores_results_spf["microbiome_correlation"] >= correlation_cuttoff[i]) & (compores_results_spf[
                                                                                                 "microbiome_correlation"] <
                                                                                             correlation_cuttoff[i + 1])
        intersection = compores_results_spf[mask]["genes_id"].isin(intersection_compo_gf)
        percentage[i] = intersection.sum() / mask.sum()
    plt.bar(correlation_cuttoff[:-1], percentage, width=0.1 / len(correlation_cuttoff), color='skyblue',
            edgecolor='black')
    plt.xlabel('Correlation (OTU level)')
    plt.ylabel('Percentage of intersection genes')
    plt.title(f'Percentage of intersection genes vs. correlation (@OTU) for {abx}-{treat}')
    plt.show()
    # do the same for p-values
    p_values = np.linspace(0, 1, 20)
    percentage = np.zeros(len(p_values) - 1)
    for i in range(len(p_values) - 1):
        mask = (compores_results_spf[f"{column_names[1]}_p"] >= p_values[i]) & (
                compores_results_spf[f"{column_names[1]}_p"] < p_values[i + 1])
        intersection = compores_results_spf[mask]["genes_id"].isin(intersection_compo_gf)
        percentage[i] = intersection.sum() / mask.sum()
    plt.bar(p_values[:-1], percentage, width=0.05, color='skyblue', edgecolor='black')
    plt.xlabel('p-value')
    plt.ylabel('Percentage of intersection genes')
    plt.title(f'Percentage of intersection genes vs. p-value for {abx}-{treat}')
    plt.show()

    times = 41
    # p_values= np.linspace(0, 1, times)
    relevant_values = compores_results_spf[f"{column_names[1]}_p"]
    values_0_1 = np.linspace(relevant_values.min(), relevant_values.quantile(0.1), 10,
                             endpoint=False)  # 10 values from 0 to 0.1
    values_1_9 = np.linspace(relevant_values.quantile(0.1), relevant_values.quantile(0.9), 20,
                             endpoint=False)  # 20 values from 0.1 to 0.9
    values_9_1 = np.linspace(relevant_values.quantile(0.9), relevant_values.max(), 11,
                             endpoint=True)  # 10 values from 0.9 to 1
    # Combine the values
    p_values = np.concatenate([values_0_1, values_1_9, values_9_1])
    std_values = np.zeros(len(p_values))
    intersection_sizes = np.zeros(len(p_values))
    intersection_percent = np.zeros(len(p_values))
    subset_sizes = np.zeros(len(p_values))
    random_intersection_sizes = np.zeros(len(p_values))
    random_intersection_sizes_std = np.zeros(len(p_values))
    random_intersection_percent = np.zeros(len(p_values))
    random_intersection_percent_std = np.zeros(len(p_values))
    for i, threshold in enumerate(p_values):
        print(threshold)
        uncorrelated_genes = [ensmus_to_gene[mus] for mus in
                              compores_results_spf[compores_results_spf[f"{column_names[1]}_p"] >= threshold][
                                  column_names[0]].str.split('_').str[-1]]
        correlated_genes = [ensmus_to_gene[mus] for mus in
                            compores_results_spf[compores_results_spf[f"{column_names[1]}_p"] < threshold][
                                column_names[0]].str.split('_').str[-1]]
        subset_sizes[i] = len(uncorrelated_genes)
        # all_genes = [ensmus_to_gene[mus] for mus in
        #              compores_results_spf[column_names[0]].str.split('_').str[-1]]

        # correlated_intersection = list(set(correlated_genes).intersection(set(gf_significant_genes)))
        uncorrelated_intersection = list(set(uncorrelated_genes).intersection(set(gf_significant_genes)))

        # df[df[f"{column_names[1]}_p"] < 0.05][column_names[0]].str.split('_').str[-1]]
        # intersection = list(set(uncorrelated_genes).intersection(set(gf_significant_genes)))
        intersection_sizes[i] = len(uncorrelated_intersection)
        intersection_percent[i] = len(uncorrelated_intersection) / len(uncorrelated_genes) if len(
            uncorrelated_genes) > 0 else 0
        # intersection_all = list(set(all_genes).intersection(set(gf_significant_genes)))
        # I = gf.index.intersection(set([ensmus_to_gene[gene] for gene in transcriptome.index]))
        print(len(uncorrelated_genes), len(gf_significant_genes), len(uncorrelated_intersection),
              uncorrelated_intersection)
        # print('\n'.join(uncorrelated_intersection))
        # save '\n'.join(intersection) to a file
        with open(f"./Private/CompoResVerification/intersection_genes_{abx}_{treat}.txt", "w") as file:
            file.write('\n'.join(uncorrelated_intersection))
        # save the list(set(all_genes).union(set(gf_genes))) to a file
        with open(f"./Private/CompoResVerification/all_genes_{abx}_{treat}.txt", "w") as file:
            unique_genes = set(correlated_genes).union(gf_significant_genes)  # Combine the two sets
            file.write('\n'.join(str(gene) for gene in unique_genes))  # Write unique genes to the file

        mean_random, std_random, p = simulate_intersections(
            N1=np.array([ensmus_to_gene[gene] for gene in transcriptome.index]), N2=gf.index,
            n1=len(uncorrelated_genes),
            # n2=len(gf_significant_genes), compare=len(uncorrelated_intersection), p=threshold, iterations=10_000)
            n2=len(gf_significant_genes), compare=len(uncorrelated_intersection), p=threshold, iterations=1_0)
        std_values[i] = std_random
        print(
            f"Random mean {mean_random}, std {std_random}: z-score = {abs((len(uncorrelated_intersection) - mean_random) / std_random)}, p={p}")

        size = np.zeros(1_000)
        percent = np.zeros_like(size)
        for j in range(len(size)):
            # choose randomly from compores_results_spf group of the same size as 'uncorrelated_genes'
            random_choice = np.random.choice(
                compores_results_spf[column_names[0]].str.split('_').str[-1], len(uncorrelated_genes), replace=False)
            random_choice = [ensmus_to_gene[mus] for mus in random_choice]
            random_intersection = list(set(random_choice).intersection(set(gf_significant_genes)))
            size[j] = len(random_intersection)
            percent[j] = len(random_intersection) / len(random_choice) if len(random_choice) > 0 else 0
        random_intersection_sizes[i] = np.mean(size)
        random_intersection_percent[i] = np.mean(size) / len(uncorrelated_genes) if len(
            uncorrelated_genes) > 0 else 0
        random_intersection_sizes_std[i] = np.std(size)
        random_intersection_percent_std[i] = np.std(percent)

    # plot the intersection sizes and the std values vs. p-value threshold
    plt.scatter(p_values, intersection_sizes, label="Intersection size", c='r')
    # plt.scatter(p_values, random_intersection_sizes, label="Random intersection size")
    plt.errorbar(p_values, random_intersection_sizes, yerr=random_intersection_sizes_std, fmt='o',
                 label="Random intersection size")
    plt.xlabel("p-value threshold")
    plt.ylabel("Intersection size")
    plt.title(f"Intersection size between {abx} {treat} and GF as p-value threshold increases \n(less genes are "
              f"included as p increases)")
    plt.legend()
    plt.savefig(f"./Private/CompoResVerification/intersection_sizes_{abx}_{treat}.png")
    plt.show()

    # plot the intersection sizes and the std values vs. p-value threshold
    plt.scatter(subset_sizes, intersection_sizes, label="Intersection size", c='r')
    # plt.scatter(p_values, random_intersection_sizes, label="Random intersection size")
    plt.errorbar(subset_sizes, random_intersection_sizes, yerr=random_intersection_sizes_std, fmt='o',
                 label="Random intersection size")
    plt.xlabel("subset sizes")
    plt.ylabel("Intersection size")
    plt.title(
        f"Intersection size vs. subset size between {abx} {treat} and GF as p-value threshold increases \n(less genes are "
        f"included as p increases)")
    plt.legend()
    plt.savefig(f"./Private/CompoResVerification/intersection_sizes_vs_subset_size_{abx}_{treat}.png")
    plt.show()

    plt.scatter(subset_sizes, intersection_sizes / random_intersection_sizes, label="Intersection size", c='r')
    plt.xlabel("subset sizes")
    plt.ylabel("Intersection size / random intersection size")
    plt.title(
        f"Intersection size / random intersection size between {abx} {treat} and GF as p-value threshold increases \n(less genes are "
        f"included as p increases)")
    plt.axhline(y=1, color='k', linestyle='--')
    plt.legend()
    plt.savefig(f"./Private/CompoResVerification/intersection_fold_sizes_{abx}_{treat}.png")
    plt.show()

    plt.scatter(p_values, intersection_percent, label="Intersection size")
    plt.xlabel("p-value threshold")
    plt.ylabel("Intersection percent out of uncorrelated genes")
    plt.title(f"Intersection percent between {abx} {treat} and GF as p-value threshold increases \n(less genes are "
              f"included as p increases)")
    plt.savefig(f"./Private/CompoResVerification/intersection_percent_{abx}_{treat}.png", dpi=300)
    plt.show()

    plt.scatter(p_values, std_values, label="Std values")
    plt.xlabel("p-value threshold")
    plt.ylabel("Std values")
    plt.title(f"Std values between {abx} {treat} and GF as p-value threshold increases \n(less genes are "
              f"included as p increases)")
    plt.savefig(f"./Private/CompoResVerification/std_values_{abx}_{treat}.png", dpi=300)
    plt.show()

    abx_data = metadata[(metadata['Drug'] == abx) & (metadata['Treatment'] == treat)]['ID'].values
    pbs_data = metadata[(metadata['Drug'] == 'PBS') & (metadata['Treatment'] == treat)]['ID'].values
    transcriptome["fold_change_spf"] = transcriptome.apply(
        lambda row: np.log2(np.median(row[abx_data]) / np.median(row[pbs_data])), axis=1)

    correlated_genes = [ensmus_to_gene[mus] for mus in
                        compores_results_spf[
                            compores_results_spf[f"{column_names[1]}_p"] < 0.05][
                            column_names[0]].str.split('_').str[-1]]
    uncorrelated_genes = [ensmus_to_gene[mus] for mus in
                          compores_results_spf[
                              compores_results_spf[f"{column_names[1]}_p"] >= 0.05][
                              column_names[0]].str.split('_').str[-1]]
    correlated_intersection = list(set(correlated_genes).intersection(set(gf_significant_genes)))
    uncorrelated_intersection = list(set(uncorrelated_genes).intersection(set(gf_significant_genes)))

    return intersection_gf_spf, gf, transcriptome, correlated_intersection, uncorrelated_intersection


def read_fmt(normalize=True):
    filename = f"./Private/YasminRandomForest/Yasmin_FMT_merged{'_normalized' if normalize else ''}_significance.tsv"
    if os.path.exists(filename):
        return pd.read_csv(filename, sep="\t")
    metadata = pd.read_csv("../Data/Yasmin_FMT/metadata-Yasmin_FMT.tsv", sep="\t")
    # create columns Drug and Treatment, based on group.split("_") accordingly ([0] is drug, [1] is treatment)
    metadata["Drug"] = metadata["group"].apply(lambda x: x.split("_")[0])
    metadata["Treatment"] = metadata["group"].apply(lambda x: x.split("_")[1])
    metadata = metadata.rename(columns={"sample": "ID"})
    # metadata["ID"] = metadata["sample"].str.split("_").str[-1]
    metadata = metadata[metadata["Treatment"] == "RECIPIENT"]
    abx_samples = metadata[metadata["Drug"] == "VANCO"]["ID"]
    pbs_samples = metadata[metadata["Drug"] == "PBS"]["ID"]

    def compute_p_value(row, group1_cols, group2_cols, ttest=True):
        from scipy.stats import ttest_ind, mannwhitneyu
        group1 = row[group1_cols].values
        group2 = row[group2_cols].values
        if ttest:
            t_stat, p_value = ttest_ind(group1, group2)
        else:
            u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        return p_value

    temp = pd.read_csv(f"./Private/YasminRandomForest/Yasmin_FMT_merged{'_normalized' if normalize else ''}.tsv",
                       sep="\t")
    temp = temp.set_index("gene_name")[metadata["ID"]]

    temp = temp.groupby(temp.index).sum()
    # normalize the data so column will sum to 1_000_000
    sum_reads = temp.sum(axis=0)
    # print the sum of reads for each sample in scientific notation
    print(f"sum of reads for each sample: {sum_reads.mean():.1E}", f"+-{sum_reads.std():.1E}")
    temp = temp.div(temp.sum(axis=0), axis=1) * 1_000_000
    # remove empty rows
    temp = temp.loc[~(temp == 0).all(axis=1)]
    # rename "sample" to "ID"
    temp, metadata = transform_data(temp, metadata, "_fmt", skip=False, gf=True)

    # merge temp with df based on index
    temp['p-value_fmt'] = temp.apply(lambda row: compute_p_value(row, abx_samples, pbs_samples), axis=1)
    # temp = temp.reset_index()
    # calculate if enhanced?
    temp['enhanced'] = temp.apply(lambda row: np.median(row[abx_samples].values) > np.median(row[pbs_samples].values),
                                  axis=1)
    temp.to_csv(filename, sep="\t", index=True)
    return temp


def compare_correlation_fmt(abx="Van", treat="PO", vs_all=True, threshold=0.05):
    path = fr"D:\Master heavy files\CompoResAllConditions\bootstrap\50"
    # path = fr"D:\Master heavy files\CompoResAllConditions{'\\0_05' if threshold == 0.05 else '\\0_01'}"
    genome, metadata, partek, transcriptome = read_process_files(new=False)
    transcriptome, metadata = transform_data(transcriptome, metadata, "RASflow", skip=True)

    folder_dir = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
    temp = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
    ensmus_to_gene = temp.set_index('gene_id')['gene_name'].to_dict()

    data = read_and_print_pkl(path + rf'\{abx}-{treat}-feces\pairs\mean_log_p_value.pkl')
    data = data[f"{abx}-{treat}-feces"]
    index = read_and_print_pkl(path + rf'\{abx}-{treat}-feces\pairs\response_index.pkl')
    column_names = ['gene', 'correlation']
    compores_results_spf = pd.DataFrame({column_names[0]: index, column_names[1]: data})
    compores_results_spf[f'{column_names[1]}_p'] = np.exp(-compores_results_spf[column_names[1]])
    ensmus_to_gene = temp.set_index('gene_id')['gene_name'].to_dict()
    compores_results_spf["genes_name"] = compores_results_spf[column_names[0]].str.split('_').str[-1]
    compores_results_spf["genes_id"] = compores_results_spf["genes_name"].apply(lambda x: ensmus_to_gene[x])

    fmt = read_fmt()
    fmt_genes = fmt[fmt["p-value_fmt"] < threshold].gene_name.to_list()
    fmt_significant = fmt[fmt["p-value_fmt"] < threshold].set_index("gene_name")
    multiabx = calc_multi_abx_statistics()
    van = multiabx[[f"p-value_{abx}_{treat}", f"fold_change_{abx}_{treat}", f'enhanced_{abx}_{treat}']]
    van_significant = van[van[f"p-value_{abx}_{treat}"] < threshold]
    van["gene_name"] = van.apply(lambda row: ensmus_to_gene[row.name], axis=1)
    van_significant.index = [ensmus_to_gene[mus] for mus in van_significant.index]
    # set index to be gene_name
    van.set_index("gene_name", inplace=True)
    # intersection_fmt_spf = list(
    #     set(van_significant.index).intersection(fmt_significant.index))
    van_significant_aligned, gf_significant_aligned = van_significant.align(fmt_significant, join="inner", axis=0)
    intersection_fmt_spf = van_significant_aligned.index[
        van_significant_aligned[f'enhanced_{abx}_{treat}'] == gf_significant_aligned['enhanced']]

    # save this csv
    intersection_fmt_spf = pd.DataFrame(intersection_fmt_spf, columns=["genes_id"])
    intersection_fmt_spf.to_csv(
        f"./Private/CompoResVerification/{abx}-{treat}-intersection-FMT{'0_05' if threshold == 0.05 else ''}.tsv",
        sep="\t", index=False)

    # choose len(intersection) genes from the compores_results_spf that are not in intersection_gf_spf
    compores_results_spf_not_intersect = compores_results_spf[
        ~compores_results_spf["genes_id"].isin(intersection_fmt_spf["genes_id"])]
    # choose len(intersection) genes from the compores_results_spf_not_intersect
    # random_compores_results_spf_not_intersect = compores_results_spf_not_intersect["correlation_p"].sample(n=len(intersection_gf_spf))
    random_compores_results_spf_not_intersect = compores_results_spf_not_intersect["correlation_p"]
    compores_results_spf_intersect = \
        compores_results_spf[compores_results_spf["genes_id"].isin(intersection_fmt_spf["genes_id"])][f"correlation_p"]
    # save compores_results_spf_intersect to a csv
    to_save = compores_results_spf[compores_results_spf["genes_id"].isin(intersection_fmt_spf["genes_id"])]
    to_save[["genes_id", "genes_name", "correlation_p"]].to_csv(
        f"./Private/CompoResVerification/{abx}-{treat}-intersect-FMT{'0_05' if threshold == 0.05 else ''}.tsv",
        sep="\t", index=False)
    return
    bins = 20
    alpha = 0.3
    plt.hist(compores_results_spf["correlation_p"], bins=bins, color='black', edgecolor='black',
             # plt.hist(compores_results_spf["correlation_p"], bins=bins, color='black', edgecolor='black', density=True,
             alpha=alpha, label="All")
    plt.hist(random_compores_results_spf_not_intersect, bins=bins, color='red', edgecolor='black',
             alpha=alpha, label="Not intersecting")
    # density=True, alpha=alpha, label="Not intersecting")
    plt.hist(compores_results_spf_intersect, bins=bins, color='blue', edgecolor='black',
             alpha=alpha, label="Intersecting")
    # density=True, alpha=alpha, label="Intersecting")
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title(f'p-value distribution for {abx}, {treat}')
    plt.legend()
    plt.show()
    # Do MWU test between random_compores_results_spf_not_intersect and compores_results_spf_intersect
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(random_compores_results_spf_not_intersect, compores_results_spf_intersect)
    print(f"FMT MWU test between random_compores_results_spf_not_intersect and compores_results_spf_intersect: p={p}")

    if random_compores_results_spf_not_intersect.median() > compores_results_spf_intersect.median():
        print("The median of random_compores_results_spf_not_intersect is greater")
    else:
        print("The median of compores_results_spf_intersect is greater")
    stat, p = mannwhitneyu(random_compores_results_spf_not_intersect, compores_results_spf_intersect,
                           alternative='greater')
    print(f"MWU test between random_compores_results_spf_not_intersect and compores_results_spf_intersect: p={p}")

    return
    intersection_evaluation(compores_results_spf_intersect, random_compores_results_spf_not_intersect,
                            threshold=0.05, smaller=True, N=len(compores_results_spf),
                            K=len(compores_results_spf_intersect))
    intersection_evaluation(compores_results_spf_intersect, random_compores_results_spf_not_intersect,
                            threshold=0.5, smaller=False, N=len(compores_results_spf),
                            K=len(compores_results_spf_intersect))
    intersection_evaluation(compores_results_spf_intersect, random_compores_results_spf_not_intersect,
                            threshold=compores_results_spf["correlation_p"].median(), smaller=False,
                            N=len(compores_results_spf),
                            K=len(compores_results_spf_intersect))
    times = 41
    # p_values= np.linspace(0, 1, times)
    relevant_values = compores_results_spf[f"{column_names[1]}_p"]
    values_0_1 = np.linspace(relevant_values.min(), relevant_values.quantile(0.1), 10,
                             endpoint=False)  # 10 values from 0 to 0.1
    values_1_9 = np.linspace(relevant_values.quantile(0.1), relevant_values.quantile(0.9), 20,
                             endpoint=False)  # 20 values from 0.1 to 0.9
    values_9_1 = np.linspace(relevant_values.quantile(0.9), relevant_values.max(), 11,
                             endpoint=True)  # 10 values from 0.9 to 1
    # Combine the values
    p_values = np.concatenate([values_0_1, values_1_9, values_9_1])
    std_values = np.zeros(len(p_values))
    intersection_sizes = np.zeros(len(p_values))
    subset_sizes = np.zeros(len(p_values))
    intersection_percent = np.zeros(len(p_values))
    random_intersection_sizes = np.zeros(len(p_values))
    random_intersection_sizes_std = np.zeros(len(p_values))
    random_intersection_percent = np.zeros(len(p_values))
    random_intersection_percent_std = np.zeros(len(p_values))
    for i, threshold in enumerate(p_values):
        print(threshold)
        correlated_genes = [ensmus_to_gene[mus] for mus in
                            compores_results_spf[compores_results_spf[f"{column_names[1]}_p"] < threshold][
                                column_names[0]].str.split('_').str[-1]]
        subset_sizes[i] = len(correlated_genes)

        # uncorrelated_genes = [ensmus_to_gene[mus] for mus in
        #                       compores_results_spf[compores_results_spf[f"{column_names[1]}_p"] < threshold][
        #                           column_names[0]].str.split('_').str[-1]]
        # all_genes = [ensmus_to_gene[mus] for mus in
        #              compores_results_spf[column_names[0]].str.split('_').str[-1]]

        correlated_intersection = list(set(correlated_genes).intersection(set(fmt_genes)))
        # uncorrelated_intersection = list(set(uncorrelated_genes).intersection(set(fmt_genes)))

        # df[df[f"{column_names[1]}_p"] < 0.05][column_names[0]].str.split('_').str[-1]]
        # intersection = list(set(uncorrelated_genes).intersection(set(gf_significant_genes)))
        intersection_sizes[i] = len(correlated_intersection)
        intersection_percent[i] = len(correlated_intersection) / len(correlated_genes) if len(
            correlated_genes) > 0 else 0
        # intersection_all = list(set(all_genes).intersection(set(gf_significant_genes)))
        # I = gf.index.intersection(set([ensmus_to_gene[gene] for gene in transcriptome.index]))
        print(len(correlated_genes), len(fmt_significant), len(correlated_intersection), correlated_intersection)
        # print('\n'.join(correlated_intersection))
        # save '\n'.join(intersection) to a file
        with open(f"./Private/CompoResVerification/intersection_genes_{abx}_{treat}.txt", "w") as file:
            file.write('\n'.join(correlated_intersection))
        # save the list(set(all_genes).union(set(gf_genes))) to a file
        with open(f"./Private/CompoResVerification/all_genes_{abx}_{treat}.txt", "w") as file:
            unique_genes = set(correlated_genes).union(fmt_significant)  # Combine the two sets
            file.write('\n'.join(str(gene) for gene in unique_genes))  # Write unique genes to the file

        size = np.zeros(1_000)
        percent = np.zeros_like(size)
        for j in range(len(size)):
            # choose randomly from compores_results_spf group of the same size as 'correlated_genes'
            random_choice = np.random.choice(
                compores_results_spf[column_names[0]].str.split('_').str[-1], len(correlated_genes), replace=False)
            random_choice = [ensmus_to_gene[mus] for mus in random_choice]
            random_intersection = list(set(random_choice).intersection(set(fmt_genes)))
            size[j] = len(random_intersection)
            percent[j] = len(random_intersection) / len(random_choice) if len(random_choice) > 0 else 0
        random_intersection_sizes[i] = np.mean(size)
        random_intersection_percent[i] = np.mean(size) / len(correlated_genes) if len(
            correlated_genes) > 0 else 0
        random_intersection_sizes_std[i] = np.std(size)
        random_intersection_percent_std[i] = np.std(percent)

        if vs_all:
            mean_random, std_random, p = simulate_intersections(
                N1=np.array([ensmus_to_gene[gene] for gene in transcriptome.index]), N2=fmt.gene_name,
                n1=len(correlated_genes),
                n2=len(fmt_significant), compare=len(correlated_intersection), p=threshold, iterations=1_00)
        else:
            mean_random, std_random, p = simulate_intersections(
                N1=np.array(
                    [ensmus_to_gene[gene] for gene in compores_results_spf[column_names[0]].str.split('_').str[-1]]),
                N2=fmt_significant.gene_name, n1=len(correlated_genes),
                n2=len(fmt_significant), compare=len(correlated_intersection), p=threshold, iterations=1_00)
            n = len(correlated_genes)
            K = len(
                set(np.array([ensmus_to_gene[gene] for gene in van_significant.index])).intersection(set(fmt_genes)))
            N = len(np.array([ensmus_to_gene[gene] for gene in van_significant.index]))
            print(f"expected mean {n * K / N}, variance {n * K * (N - K) * (N - n) / (N ** 2 * (N - 1))}")
        std_values[i] = std_random
        print(
            f"Random mean {mean_random}, std {std_random}: z-score = {abs((len(correlated_intersection) - mean_random) / std_random)}, p={p}")

    # plot the intersection sizes and the std values vs. p-value threshold
    plt.scatter(p_values, intersection_sizes, label="Intersection size", c='r')
    # plt.scatter(p_values, random_intersection_sizes, label="Random intersection size")
    plt.errorbar(p_values, random_intersection_sizes, yerr=random_intersection_sizes_std, fmt='o',
                 label="Random intersection size")
    plt.xlabel("p-value threshold")
    plt.ylabel("Intersection size")
    plt.title(f"Intersection size between {abx} {treat} and FMT as p-value threshold increases \n(more genes are "
              f"included as p increases)")
    plt.legend()
    plt.savefig(f"./Private/CompoResVerification/intersection_sizes_{abx}_{treat}.png")
    plt.show()

    # plot the intersection sizes and the std values vs. p-value threshold
    plt.scatter(subset_sizes, intersection_sizes, label="Intersection size", c='r')
    # plt.scatter(p_values, random_intersection_sizes, label="Random intersection size")
    plt.errorbar(subset_sizes, random_intersection_sizes, yerr=random_intersection_sizes_std, fmt='o',
                 label="Random intersection size")
    plt.xlabel("subset sizes")
    plt.ylabel("Intersection size")
    plt.title(f"Intersection size between {abx} {treat} and FMT as p-value threshold increases \n(more genes are "
              f"included as p increases)")
    plt.legend()
    plt.savefig(f"./Private/CompoResVerification/intersection_sizes_vs_subset_{abx}_{treat}.png")
    plt.show()

    plt.scatter(subset_sizes, intersection_sizes / random_intersection_sizes, label="Intersection size")
    plt.xlabel("subset sizes")
    plt.ylabel("Intersection size / random intersection size")
    # add vertical line at y=1
    plt.axhline(y=1, color='k', linestyle='--')
    plt.title(
        f"Intersection size / random intersection size between {abx} {treat} and FMT as p-value threshold increases \n(more genes are "
        f"included as p increases)")
    plt.legend()
    plt.savefig(f"./Private/CompoResVerification/intersection_fmt_fold_sizes_{abx}_{treat}.png")
    plt.show()

    plt.scatter(p_values, intersection_percent, label="Intersection", c='r')
    # plt.scatter(p_values, random_intersection_percent, label="Random intersection")
    plt.errorbar(p_values, random_intersection_percent, yerr=random_intersection_percent_std, fmt='o',
                 label="Random intersection")
    plt.xlabel("p-value threshold")
    plt.ylabel("Intersection percent out of uncorrelated genes")
    plt.title(f"Intersection percent of correlated genes between {abx} {treat} and FMT as p-value threshold increases "
              f"\n(more genes are included as p increases)")
    plt.legend()
    plt.savefig(f"./Private/CompoResVerification/intersection_percent_{abx}_{treat}.png")
    plt.show()

    plt.scatter(p_values, std_values, label="Std values")
    plt.xlabel("p-value threshold")
    plt.ylabel("Std values")
    plt.title(f"Std values of intersection between {abx} {treat} and FMT as p-value threshold increases \n(more genes "
              f"are included as p increases)")
    plt.savefig(f"./Private/CompoResVerification/std_values_{abx}_{treat}.png")
    plt.show()

    abx_data = metadata[(metadata['Drug'] == abx) & (metadata['Treatment'] == treat)]['ID'].values
    pbs_data = metadata[(metadata['Drug'] == 'PBS') & (metadata['Treatment'] == treat)]['ID'].values
    transcriptome["fold_change_spf"] = transcriptome.apply(
        lambda row: np.log2(np.median(row[abx_data]) / np.median(row[pbs_data])), axis=1)

    correlated_genes = [ensmus_to_gene[mus] for mus in
                        compores_results_spf[
                            compores_results_spf[f"{column_names[1]}_p"] < relevant_values.quantile(0.05)][
                            column_names[0]].str.split('_').str[-1]]
    uncorrelated_genes = [ensmus_to_gene[mus] for mus in
                          compores_results_spf[
                              compores_results_spf[f"{column_names[1]}_p"] >= relevant_values.quantile(0.05)][
                              column_names[0]].str.split('_').str[-1]]
    correlated_intersection = list(set(correlated_genes).intersection(set(fmt_significant)))
    uncorrelated_intersection = list(set(uncorrelated_genes).intersection(set(fmt_significant)))

    return intersection_fmt_spf, fmt, transcriptome, correlated_intersection, uncorrelated_intersection


def log2fc_plot(intersection, gf, spf, case):
    from scipy.stats import pearsonr, gaussian_kde
    from matplotlib.colors import LogNorm

    # from clusters_plot import plot_kde

    # import matplotlib.pyplot as plt
    # import numpy as np
    # Example data (replace with actual data)
    x = gf.loc[intersection]["fold_change_gf"]  # GF Van vs PBS log2(FC)
    y = spf.loc[intersection]["fold_change_spf"]  # SPF vs PBS log2(FC)

    # Filter out NaN and inf values from both x and y
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
    x = x[mask]
    y = y[mask]

    corr_coef, p_value = pearsonr(x, y)

    # Set up the figure and axis
    plt.figure(figsize=(6, 6))

    # colors = np.where(y > 0, 'blue', 'red')  # Use blue for positive y values, red for negative
    # plot it using kde
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=z, cmap='viridis', norm=LogNorm(), edgecolor='none', s=18)
    plt.colorbar(scatter, ax=ax, label='Density (log scale)')
    # # Scatter plot
    # plt.scatter(x, y, c=colors, edgecolor='black')

    # Add dashed lines at x=0 and y=0
    plt.axhline(0, color='black', linestyle='dotted')
    plt.axvline(0, color='black', linestyle='dotted')

    # # Set limits
    # plt.xlim([-4, 4])
    # plt.ylim([-4, 4])

    # plt.text(-3.5, -3.5, f'r = {corr_coef:.2f}\np = {p_value}', fontsize=12, color='black')
    # plt.text(2.5, -3.5, f'r = {corr_coef:.2f}\np = {p_value:.2f}', fontsize=12, color='black')

    # Labels and title
    plt.xlabel(r'GF Van vs. PBS log$_2$ (FC)', fontsize=12)
    plt.ylabel(r'SPF Van vs. PBS log$_2$ (FC)', fontsize=12)
    plt.title(f'Van IP: GF vs. SPF \n{case}\nr = {corr_coef:.2f}\np = {p_value}', fontsize=12)

    # Customize tick parameters
    plt.tick_params(axis='both', which='major', labelsize=10)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"./Private/selected_clusters/log2fc_{case}.png")
    plt.show()


def box_plot_compores_comparison():
    # read ./Private/CompoResVerification/{abx}-{treat}-intersect-GF.tsv for Van-PO and Van-IP
    path = "./Private/CompoResVerification/"
    van_ip = pd.read_csv(path + f"Van-IP-intersect-GF.tsv", sep="\t")
    van_po = pd.read_csv(path + f"Van-PO-intersect-FMT.tsv", sep="\t")
    # # boxplot the "correlation_p" column of both on the same plot
    # plt.boxplot([van_ip["correlation_p"], van_po["correlation_p"]], labels=["Van-IP", "Van-PO"])
    # violin plot instead of boxplot
    plt.violinplot([van_ip["correlation_p"], van_po["correlation_p"]])
    plt.xticks([1, 2], ["Van-IP", "Van-PO"])
    # add significance by t-test between groups
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(van_ip["correlation_p"], van_po["correlation_p"], alternative="greater")
    plt.text(1.5, 0.8, f"p={p_value}")

    plt.ylabel("p-value")
    plt.title("Van-IP vs. Van-PO CompoRes p-values")
    plt.savefig(path + "Van-IP_vs_Van-PO_CompoRes_p-values.png")
    plt.show()


def akiko_check():
    from ClusteringGO import build_tree, get_go_to_ensmusg, get_ancestor, get_go
    from typing import Set
    from anytree import PostOrderIter
    from goatools import obo_parser
    save_path = "./Private/CompoResVerification/neo_viral.csv"
    if os.path.exists(save_path):
        return set(pd.read_csv(save_path)["genes_id"])

    go = obo_parser.GODag(get_go())

    root, tree_size = build_tree(False)
    added: Set[str] = set()
    go_to_ensmbl_dict = get_go_to_ensmusg()
    progress_interval = 5_000
    viral_genes = {}
    ancestors = {}
    names = {}
    viral_genes_set = set()
    for i, node in enumerate(PostOrderIter(root)):
        node_genes = go_to_ensmbl_dict.get(node.go_id, set())
        if node_genes:
            if "vir" in node.name:
                if node.go_id not in viral_genes:
                    viral_genes[node.go_id] = set()
                    names[node.go_id] = node.name
                    all_ancestors = list(get_ancestor(go[node.go_id])) if node.go_id in go else None
                    # all_ancestors = list(get_ancestor(node))
                    if all_ancestors and len(all_ancestors) > 1:
                        print(node.go_id, [ancestor.name for ancestor in all_ancestors])
                    category_name = [ancestor.name for ancestor in all_ancestors] if all_ancestors else "NOT_BP"
                    ancestors[node.go_id] = category_name
                viral_genes[node.go_id] = viral_genes[node.go_id].union(node_genes)
                viral_genes_set = viral_genes_set.union(node_genes)
                added.add(node.go_id)
        if i % progress_interval == 0:
            print(f"### {i} nodes were updated ###")
    # for go_id in viral_genes:
    #     if ancestors[go_id] == "NOT_BP":
    #         print(f"{go_id} NOT BP")
    #     print(f"{go_id}", ancestors[go_id], names[go_id])
    #     # print(len(viral_genes[go_id]), viral_genes[go_id])
    print(len(viral_genes_set), viral_genes_set)
    # change viral_genes_set to pd.DF and save it
    viral_genes_set_df = pd.DataFrame(list(viral_genes_set), columns=["genes_id"])
    viral_genes_set_df.to_csv(save_path, index=False)
    return viral_genes_set


def neo_significance(threshold=0.05):
    from ClusteringGO import transform_data
    from compores_results_analysis import get_significant
    genome, metadata, partek, transcriptome = read_process_files(new=False)
    transcriptome, metadata = transform_data(transcriptome, metadata, "RASflow", skip=True)
    metadata["Category"] = metadata['Drug'] + "_" + metadata['Treatment']
    abx = "Neo"
    neo_significant = {}
    for treat in treatments:
        samples = metadata[
            ((metadata["Drug"] == abx) | (metadata["Drug"] == "PBS")) & (metadata["Treatment"] == treat)]
        curr = transcriptome[samples["ID"]]
        abx_data = metadata[(metadata['Drug'] == abx) & (metadata["Treatment"] == treat)]
        pbs_data = metadata[(metadata['Drug'] == 'PBS') & (metadata["Treatment"] == treat)]
        genes = get_significant(curr, abx_data, pbs_data, threshold=threshold)
        neo_significant[treat] = set(genes)
        print(f"Number of significant genes for {abx}-{treat}: {len(genes)}")
    return neo_significant


def neo_compores(viral_genes):
    abx = "Neo"
    path = r"D:\Master heavy files\CompoResAllConditions"
    plot_data = []

    for treat in treatments:
        data = read_and_print_pkl(path + rf'\{abx}-{treat}-feces\mean_log_p_value.pkl')
        data = data[f"{abx}-{treat}-feces"]
        index = read_and_print_pkl(path + rf'\{abx}-{treat}-feces\response_index.pkl')
        column_names = ['gene', 'correlation']
        compores_results_spf = pd.DataFrame({column_names[0]: index, column_names[1]: data})
        compores_results_spf[f'{column_names[1]}_p'] = np.exp(-compores_results_spf[column_names[1]])
        compores_results_spf["genes_name"] = compores_results_spf[column_names[0]].str.split('_').str[-1]
        # choose len(intersection) genes from the compores_results_spf that are not in intersection_gf_spf
        compores_results_spf_not_intersect = \
            compores_results_spf[~compores_results_spf["genes_name"].isin(viral_genes)]["correlation_p"]
        compores_results_spf_intersect = compores_results_spf[compores_results_spf["genes_name"].isin(viral_genes)][
            f"correlation_p"]

        # # print t-test p-value compores_results_spf_not_intersect and compores_results_spf_intersect
        # # Perform t-test
        # from scipy.stats import ttest_ind
        # ttest, p_val = ttest_ind(compores_results_spf_not_intersect, compores_results_spf_intersect,
        #                          alternative="less", equal_var=False)
        # print(f"{treat}: p-value = {p_val}")

        # Append data for plotting
        plot_data.append(pd.DataFrame({
            "group": ["Significant & Non-Viral"] * len(compores_results_spf_not_intersect) +
                     ["Significant & Viral"] * len(compores_results_spf_intersect),
            "correlation_p": pd.concat([compores_results_spf_not_intersect, compores_results_spf_intersect]),
            "treatment": treat
        }))

        # Combine all data for plotting
    combined_data = pd.concat(plot_data)

    # Create boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=combined_data,
        x="treatment",
        y="correlation_p",
        hue="group",
        palette="Set2",
        ax=ax
    )
    from scipy.stats import ttest_ind
    # Annotate significance for each treatment
    for treat_idx, treat in enumerate(treatments):
        # Filter data for the current treatment
        treat_data = combined_data[combined_data["treatment"] == treat]
        ttest, p_val = ttest_ind(treat_data[treat_data["group"] == "Significant & Non-Viral"]["correlation_p"],
                                 treat_data[treat_data["group"] == "Significant & Viral"]["correlation_p"],
                                 alternative="less", equal_var=False)

        y_max = treat_data["correlation_p"].max()
        annotation_y = y_max + 0.05  # Offset for annotation

        # Determine significance level
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        # Annotate above the boxplots
        x1 = treat_idx - 0.2  # Non-Viral boxplot
        x2 = treat_idx + 0.2  # Viral boxplot
        ax.plot(
            [x1, x1, x2, x2], [annotation_y, annotation_y + 0.02, annotation_y + 0.02, annotation_y],
            lw=1.5, c='k'
        )
        ax.text(
            (x1 + x2) / 2, annotation_y + 0.03, sig, ha='center', va='bottom', color='k'
        )

    # Customize plot
    ax.set_title(f"Neo CompoRes P-values by Treatment")
    ax.set_ylabel("CompoRes P-value")
    ax.set_xlabel("Treatment")
    ax.set_ylim(0, 1.2)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"./Private/compores_response_ranking/correlation_p_values_neo_viral.png")
    plt.show()

    # return
    # bins = 20
    # alpha = 0.3
    # # plt.hist(compores_results_spf["correlation_p"], bins=bins, color='black', edgecolor='black',
    # plt.hist(compores_results_spf["correlation_p"], bins=bins, color='black', edgecolor='black', density=True,
    #          alpha=alpha, label="All")
    # plt.hist(compores_results_spf_not_intersect, bins=bins, color='red', edgecolor='black',
    #          # alpha=alpha, label="Not intersecting")
    #          density=True, alpha=alpha, label="Not intersecting")
    # plt.hist(compores_results_spf_intersect, bins=bins, color='blue', edgecolor='black',
    #          # alpha=alpha, label="Intersecting")
    #          density=True, alpha=alpha, label="Intersecting")
    # plt.xlabel('p-value')
    # plt.ylabel('Frequency')
    # plt.title(f'p-value distribution for {abx}, {treat}')
    # plt.legend()
    # # plt.show()
    # # Do MWU test between compores_results_spf_not_intersect and compores_results_spf_intersect
    # from scipy.stats import mannwhitneyu
    # stat, p = mannwhitneyu(compores_results_spf_not_intersect, compores_results_spf_intersect)
    # print(f"{abx}-{treat} MWU test between not-viral and viral: p={p}")


def box_plot_compores_comparison_specific(genes, folder):
    path = fr"D:\Master heavy files\CompoResAllConditions\{folder}"
    significant_path = fr"D:\Master heavy files\CompoResAllConditions"
    n_abx = len(antibiotics)
    fig, axes = plt.subplots(nrows=1, ncols=n_abx, figsize=(10 * n_abx, 5), sharey=True)
    for idx, abx in enumerate(antibiotics):
        # Prepare data for plotting and t-tests
        plot_data = []
        for treat_idx, treat in enumerate(treatments):
            specific_compores_results_spf = get_compores_results(abx, path, treat)
            significant_compores_results_spf = get_compores_results(abx, significant_path, treat)
            compores_results_spf_not_specific = \
                significant_compores_results_spf[
                    # significant_compores_results_spf[~significant_compores_results_spf["genes_name"].isin(genes)][
                    "correlation_p"]
            compores_results_spf_specific = specific_compores_results_spf[f"correlation_p"]
            # assert set(genes) == set(specific_compores_results_spf["genes_name"].values)

            # Append to plot_data for combined plotting
            plot_data.append(pd.DataFrame({
                "group": ["Significant"] * len(compores_results_spf_not_specific) +
                         [folder] * len(compores_results_spf_specific),
                "correlation_p": pd.concat([compores_results_spf_not_specific, compores_results_spf_specific]),
                "treatment": treat
            }))

            # # Perform t-test
            # t_stat, p_val = ttest_ind(compores_results_spf_not_specific, compores_results_spf_specific)
            # print t-test p-value compores_results_spf_not_intersect and compores_results_spf_intersect
            from scipy.stats import ttest_ind
            ttest, p_val = ttest_ind(compores_results_spf_not_specific, compores_results_spf_specific, equal_var=False,
                                     alternative="less")
            print(treat, p_val)

            # Add annotations for the current subplot
            y_max = pd.concat([compores_results_spf_not_specific, compores_results_spf_specific]).max()
            annotation_y = y_max + 0.05
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            # Annotate significance on the plot
            x1 = treat_idx - 0.2  # Adjust for the position of "Significant"
            x2 = treat_idx + 0.2  # Adjust for the position of "Folder"
            axes[idx].plot(
                [x1, x1, x2, x2], [annotation_y, annotation_y + 0.02, annotation_y + 0.02, annotation_y],
                lw=1.5, c='k'
            )
            axes[idx].text(
                (x1 + x2) / 2, annotation_y + 0.03, sig, ha='center', va='bottom', color='k'
            )

            # Combine data for all treatments of the current antibiotic
        combined_data = pd.concat(plot_data)

        # Create a boxplot for the current antibiotic
        sns.boxplot(
            data=combined_data,
            x="treatment",
            y="correlation_p",
            hue="group",
            palette="Set2",
            ax=axes[idx]
        )

        # Set plot title and labels
        axes[idx].set_title(f"{abx}")
        axes[idx].set_ylabel("CompoRes P-value")
        axes[idx].set_xlabel("Treatment")
        axes[idx].set_ylim(0, 1.2)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"./Private/compores_response_ranking/correlation_p_values_{folder}_vs_not_{folder}.png")
    plt.close()


def compare_compores_all_antibiotics(genes, folder, threshold=False):
    """
    Compare CompoRes results for all antibiotics, calculating p-values and plotting results.

    Parameters:
    genes (list): List of viral genes or other specific genes of interest.
    folder (str): Folder name containing CompoRes results.
    """
    # Define a color palette for antibiotics (same color for each antibiotic in all plots)
    unique_colors = sns.color_palette("Set2", n_colors=len(antibiotics))
    color_map = {abx: unique_colors[i] for i, abx in enumerate(antibiotics)}

    adjust_font_sizes()
    folder_path = fr"D:\Master heavy files\CompoResAllConditions{'0_05' if threshold else ''}"
    # Create subplots
    n_treat = len(treatments)
    fig, axes = plt.subplots(nrows=1, ncols=n_treat, figsize=(10 * n_treat, 5), sharey=True)
    fig.suptitle('Box plots of Viral & Significant Genes\nBox width reflects sample size', fontsize=16)

    for idx, treat in enumerate(treatments):
        plot_data = []
        for abx in antibiotics:
            # Load data
            specific_data_path = os.path.join(folder_path, f"{abx}-{treat}-feces")
            specific_compores_results = read_and_print_pkl(os.path.join(specific_data_path, "mean_log_p_value.pkl"))
            specific_compores_results = pd.DataFrame({
                "gene": read_and_print_pkl(os.path.join(specific_data_path, "response_index.pkl")),
                "correlation": specific_compores_results[f"{abx}-{treat}-feces"]
            })
            specific_compores_results["correlation_p"] = np.exp(-specific_compores_results["correlation"])
            specific_compores_results["genes_name"] = specific_compores_results["gene"].str.split('_').str[-1]

            # Filter for specific group (e.g., "Significant & Viral")
            compores_intersect = specific_compores_results[
                specific_compores_results["genes_name"].isin(genes)
            ]["correlation_p"]

            # Convert to -log10(p)
            compores_intersect = -np.log10(compores_intersect)

            # Append to plot data
            plot_data.append(pd.DataFrame({
                "correlation_p": compores_intersect,
                "antibiotic": abx
            }))

        # Combine all data for this treatment
        combined_data = pd.concat(plot_data)

        # Calculate sample sizes and widths
        counts = combined_data.groupby('antibiotic').size().reset_index(name='count')
        max_count = counts['count'].max()
        counts['width'] = counts['count'] / max_count * 0.8  # Scale width for visibility

        # Merge width info back to combined_data
        combined_data = combined_data.merge(counts, on='antibiotic')

        # Plot for this treatment
        ax = axes[idx] if n_treat > 1 else axes
        antibiotics_order = counts['antibiotic'].tolist()

        # Custom boxplot with variable widths
        positions = np.arange(len(antibiotics_order))
        box_data = [combined_data[combined_data['antibiotic'] == abx]['correlation_p'].values for abx in
                    antibiotics_order]
        widths = counts['width'].values

        bp = ax.boxplot(box_data, positions=positions, widths=widths, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(markerfacecolor='gray', alpha=0.5))

        # Color each box based on antibiotic
        for patch, abx in zip(bp['boxes'], antibiotics_order):
            patch.set_facecolor(color_map[abx])

        # Set x-ticks and labels
        ax.set_xticks(positions)
        ax.set_xticklabels(antibiotics_order, rotation=45, ha='right')
        ax.set_title(f"{treat}")
        ax.set_ylim(0, 2)
        if idx == 0:
            ax.set_ylabel("-log10(P)")
        else:
            ax.set_ylabel("")

        # Add sample count above the top whisker
        for pos, count, box in zip(positions, counts['count'], bp['whiskers'][1::2]):
            whisker_top = box.get_ydata()[1]
            if threshold:
                ax.text(pos, 1.8, f"{count}", ha='center', va='bottom', fontsize=9, color='black')
            else:
                ax.text(pos, whisker_top + 0.2, f"{count}", ha='center', va='bottom', fontsize=9, color='black')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join("./Private", "compores_response_ranking",
                               f"correlation_p_values_{folder}_vs_not_{folder}_only_significant{'_0_05' if threshold else ''}.png")
    plt.savefig(output_path, dpi=600)
    plt.close()

    return
    # from scipy.stats import ttest_ind
    #
    # _, metadata, _, transcriptome = read_process_files(new=False)
    # # transcriptome, metadata = transform_data(transcriptome, metadata, "RASflow", skip=True)
    # genes_data = pd.DataFrame(columns=["genes_name", "treatment", "antibiotic", "fold_change", "t_test_p_value"])
    # missing = {}
    # for abx in ["Neo"]:
    #     for treat in treatments:
    #         missing[treat] = set()
    #         abx_samples = metadata[(metadata['Drug'] == abx) & (metadata['Treatment'] == treat)]['ID'].values
    #         pbs_samples = metadata[(metadata['Drug'] == 'PBS') & (metadata['Treatment'] == treat)]['ID'].values
    #         for gene in genes:
    #             # calculate fold change and t-test p-value for this genes
    #             if gene not in transcriptome.index:
    #                 missing[treat].add(gene)
    #                 continue
    #             gene_data_abx = transcriptome.loc[gene, abx_samples]
    #             gene_data_pbs = transcriptome.loc[gene, pbs_samples]
    #             # calculate fold change
    #             fold_change = np.mean(gene_data_abx) / np.mean(gene_data_pbs)
    #             # fold_change = np.mean(gene_data_abx) - np.mean(gene_data_pbs)
    #             # calculate t-test p-value
    #             t_test = ttest_ind(gene_data_abx, gene_data_pbs)
    #             # add the results to the genes_data
    #             new_row = pd.DataFrame({
    #                 "genes_name": gene,
    #                 "treatment": treat,
    #                 "antibiotic": abx,
    #                 "fold_change": fold_change,
    #                 "t_test_p_value": t_test[1]
    #             }, index=[0])
    #             genes_data = pd.concat([genes_data, new_row], ignore_index=True)
    #
    # data_list = []
    # for abx in ["Neo"]:
    #     for treat in treatments:
    #         significant_compores_results_spf = get_compores_results(abx, folder_path, treat)
    #         compores_results_spf_not_specific = \
    #             significant_compores_results_spf[
    #                 significant_compores_results_spf["genes_name"].isin(genes)]
    #         compores_results_spf_not_specific['treatment'] = treat
    #         compores_results_spf_not_specific['antibiotic'] = abx
    #         data_list.append(compores_results_spf_not_specific)
    #
    # # Combine all data
    # all_data = pd.concat(data_list, axis=0)
    # # drop column gene
    # all_data = all_data.drop(columns="gene")
    # # create a column "gene" by ensmus_dict[genes_name"]
    # all_data["gene"] = all_data["genes_name"].apply(lambda x: ensmus_dict[x])
    # # reorder: antibiotic, treatment, gene, genes_name, correlation, correlation_p
    # columns_order = ["antibiotic", "treatment", "gene", "genes_name", "correlation", "correlation_p"]
    # # rename correlation to -ln(p) CompoRes, correlation_p to CompoRes p-value
    # all_data = all_data[columns_order]
    # all_data = all_data.rename(columns={"correlation": "-ln(p) CompoRes", "correlation_p": "CompoRes p-value"})
    #
    # # merge the genes_data with all_data based on gene, treatment and antibiotic and keep all those columns
    # all_data = pd.merge(all_data, genes_data, on=["genes_name", "treatment", "antibiotic"], how="outer")
    #
    # # drop all rows where "CompoRes p-value" is NaN
    # all_data = all_data.dropna(subset=["CompoRes p-value"])
    #
    # # save all_data to a csv file
    # all_data.to_csv(f"./Private/compores_response_ranking/Neo_viral_raw.csv", index=False)


# def compare_compores_all_antibiotics(genes, folder):
#     """
#     Compare CompoRes results for all antibiotics, calculating p-values and plotting results.
#
#     Parameters:
#     genes (list): List of viral genes or other specific genes of interest.
#     folder (str): Folder name containing CompoRes results.
#     """
#     from scipy.stats import ttest_ind
#     adjust_font_sizes()
#     # Create subplots
#     n_abx = len(antibiotics)
#     fig, axes = plt.subplots(nrows=1, ncols=n_abx, figsize=(10 * n_abx, 5), sharey=True)
#     folder_path = r"D:\Master heavy files\CompoResAllConditions"
#
#     for idx, abx in enumerate(antibiotics):
#         plot_data = []
#
#         for treat_idx, treat in enumerate(treatments):
#             # Load data
#             specific_data_path = os.path.join(folder_path, f"{abx}-{treat}-feces")
#             specific_compores_results = read_and_print_pkl(os.path.join(specific_data_path, "mean_log_p_value.pkl"))
#             specific_compores_results = pd.DataFrame({
#                 "gene": read_and_print_pkl(os.path.join(specific_data_path, "response_index.pkl")),
#                 "correlation": specific_compores_results[f"{abx}-{treat}-feces"]
#             })
#             specific_compores_results["correlation_p"] = np.exp(-specific_compores_results["correlation"])
#             specific_compores_results["genes_name"] = specific_compores_results["gene"].str.split('_').str[-1]
#
#             # Divide into specific and non-specific groups
#             compores_not_intersect = specific_compores_results[~specific_compores_results["genes_name"].isin(genes)][
#                 "correlation_p"]
#             compores_intersect = specific_compores_results[specific_compores_results["genes_name"].isin(genes)][
#                 "correlation_p"]
#             # convert both to -log10
#             compores_not_intersect = -np.log10(compores_not_intersect)
#             compores_intersect = -np.log10(compores_intersect)
#
#             # Append to plot data
#             plot_data.append(pd.DataFrame({
#                 "group": ["Significant & Non-Viral"] * len(compores_not_intersect) +
#                          ["Significant & Viral"] * len(compores_intersect),
#                 "correlation_p": pd.concat([compores_not_intersect, compores_intersect]),
#                 "treatment": treat
#             }))
#
#             # Perform t-test
#             ttest, p_val = ttest_ind(compores_not_intersect, compores_intersect, alternative="greater", equal_var=False)
#             # ttest, p_val = ttest_ind(compores_not_intersect, compores_intersect, alternative="less", equal_var=False)
#             print(f"{abx} {treat}: p-value = {p_val}")
#
#             # Add annotations
#             y_max = pd.concat([compores_not_intersect, compores_intersect]).max()
#             annotation_y = y_max + 0.05
#
#             if p_val < 0.001:
#                 sig = '***'
#             elif p_val < 0.01:
#                 sig = '**'
#             elif p_val < 0.05:
#                 sig = '*'
#             else:
#                 sig = 'ns'
#
#             x1 = treat_idx - 0.2
#             x2 = treat_idx + 0.2
#             axes[idx].plot(
#                 [x1, x1, x2, x2], [annotation_y, annotation_y + 0.02, annotation_y + 0.02, annotation_y],
#                 lw=1.5, c='k'
#             )
#             axes[idx].text(
#                 (x1 + x2) / 2, annotation_y + 0.03, sig, ha='center', va='bottom', color='k'
#             )
#
#         # Combine data for plotting
#         combined_data = pd.concat(plot_data)
#
#         # Plot boxplot
#         sns.boxplot(
#             data=combined_data,
#             x="treatment",
#             y="correlation_p",
#             hue="group",
#             palette="Set2",
#             ax=axes[idx],
#             legend=False if idx < n_abx - 1 else True,
#         )
#
#         # Customize plot
#         axes[idx].set_title(f"{abx}")
#         axes[idx].set_ylabel("CompoRes -log10(P)")
#         # axes[idx].set_ylabel("CompoRes P-value")
#         axes[idx].set_xlabel("Treatment")
#         axes[idx].set_ylim(0, 2)
#         # axes[idx].set_ylim(0, 1.2)
#
#     handles, labels = axes[-1].get_legend_handles_labels()
#     axes[-1].legend(
#         handles, labels, title="Group",
#         loc='center left', bbox_to_anchor=(1, 0.5), frameon=False
#     )
#     # Adjust layout and save the figure
#     plt.tight_layout()
#     output_path = os.path.join("./Private", "compores_response_ranking",
#                                f"correlation_p_values_{folder}_vs_not_{folder}_all_significant.png")
#     plt.savefig(output_path, dpi=600)
#     plt.close()


def adjust_font_sizes():
    import matplotlib as mpl

    mpl.rcParams['font.size'] = 16  # Set global font size (e.g., 14)
    mpl.rcParams['axes.titlesize'] = 18  # Title font size
    mpl.rcParams['axes.labelsize'] = 16  # Axis label font size
    mpl.rcParams['xtick.labelsize'] = 14  # X-axis tick label size
    mpl.rcParams['ytick.labelsize'] = 14  # Y-axis tick label size
    mpl.rcParams['legend.fontsize'] = 14  # Legend font size
    mpl.rcParams['figure.titlesize'] = 20  # Overall figure title font size


def box_plot_compores_comparison_clock(genes, folder, names, significance_dict, names_dictionary):
    from matplotlib.lines import Line2D
    adjust_font_sizes()
    path = fr"D:\Master heavy files\CompoResAllConditions\{folder}"
    significant_path = fr"D:\Master heavy files\CompoResAllConditions"
    data_list = []
    # antibiotics = ["Neo", "Van"]
    for abx in antibiotics:
        for treat in treatments:
            significant_compores_results_spf = get_compores_results(abx, significant_path, treat)
            compores_results_spf_not_specific = \
                significant_compores_results_spf[~significant_compores_results_spf["genes_name"].isin(genes)]
            compores_results_spf_not_specific['treatment'] = treat
            compores_results_spf_not_specific['antibiotic'] = abx
            data_list.append(compores_results_spf_not_specific)
    # Combine all data
    all_data = pd.concat(data_list, axis=0)
    n_abx = len(antibiotics)
    # create a dictionary that assigns a color for each of the genes
    palette = sns.color_palette("Set2", n_colors=len(genes))
    gene_color_dict = {gene: palette[idx] for idx, gene in enumerate(genes)}
    # for gene, name in zip(genes, names):
    # Create a single figure with subplots for each antibiotic
    fig, axes = plt.subplots(nrows=1, ncols=n_abx, figsize=(10 * n_abx, 5), sharey=True)
    legend_handles = []
    # Plot for each antibiotic
    for idx, abx in enumerate(antibiotics):
        subset = all_data[all_data['antibiotic'] == abx]
        # convert subset values to -lop10(value)
        subset['correlation_p'] = -np.log10(subset['correlation_p'])
        p = sns.boxplot(data=subset, x='treatment', y='correlation_p', palette='Set2', ax=axes[idx])
        for patch in p.patches:
            patch.set_alpha(0.6)  # Set alpha to 0.5 for 50% transparency
        x_offsets = p.get_xticks()  # Correct x positions for the boxes
        # default_widths = [box.get_width() for box in p.patches]
        # sns.swarmplot(
        #     data=subset,
        #     x="treatment",
        #     y="correlation_p",
        #     palette="Set2",
        #     ax=axes[idx],
        #     size=5
        # )
        for treat_idx, treat in enumerate(treatments):
            significant_genes = [gene for gene in genes if significance_dict[treat][abx][gene] != 'ns']
            specific_compores_results_spf = get_compores_results(abx, path, treat)

            specific_values = [get_specific_value(gene, genes, specific_compores_results_spf) for gene in
                               significant_genes]
            # convert to -log10(value)
            specific_values = [-np.log10(value) for value in specific_values]

            markers = [u'\u2191' if d == 'Enhanced' else u'\u2193' for d in significant_genes]
            colors = [gene_color_dict[gene] for gene in significant_genes]
            # Add the specific value to the plot
            # axes[idx].scatter(
            #     x=[idx + (i - len(significant_genes) / 2) * 0.1 for i in range(len(significant_genes))],
            #     # Add offsets
            #     y=specific_values,
            #     color=colors,
            #     marker=markers,
            #     label=[significant_genes] if idx == 0 and treat == treatments[0] else "",
            #     # label=f"Gene: {gene}" if idx == 0 and treat == treatments[0] else "",
            #     zorder=3
            # )
            # x_offsets = [idx + (i - len(significant_genes) / 2) * 0.1 for i in range(len(significant_genes))]

            for i, (y_val, marker, color) in enumerate(zip(specific_values, markers, colors)):
                # Use scatter for points (optional) or directly text for Unicode markers
                x_val = x_offsets[treat_idx] + (i - len(significant_genes) / 2) * 0.1  # Add minor horizontal offsets
                axes[idx].text(
                    x_val, y_val, marker,  # Unicode marker as text
                    fontsize=25, color=color, ha="center", va="center", zorder=3
                )
                # plot also black dot in this location
                axes[idx].scatter(x_val, y_val, color='black', zorder=3, s=5)

                specific_value = y_val
                # Calculate p-value for the specific gene
                not_specific_values = subset[subset["treatment"] == treat]["correlation_p"]
                # p_val = np.sum(specific_value < not_specific_values.values) / len(not_specific_values)
                from scipy.stats import ttest_1samp

                # Perform a two-sided t-test
                t_stat, p_val = ttest_1samp(not_specific_values.values, specific_value)

                # Add significance level
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                # Annotate the p-value significance on the plot
                annotation_y = specific_value + 0.05
                # annotation_y = max(specific_value, not_specific_values.max()) + 0.05
                axes[idx].text(
                    x=x_val,
                    # x=treatments.index(treat),
                    y=annotation_y,
                    s=sig,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                    zorder=3,
                )

        # # edit x-ticks: add significance_dict[abx[treat][gene]] to the x-tick
        # labels = [f"{treat}\n{significance_dict[treat][abx][gene]}" for treat in treatments]
        # axes[idx].set_xticks(range(len(treatments)), labels=labels)
        axes[idx].set_title(f"{abx}")
        axes[idx].set_xlabel("Treatment")
        if idx == 0:
            axes[idx].set_ylabel("CompoRes -log10(P)")
        axes[idx].set_ylim(0, 2)
        # axes[idx].set_ylim(0, 1.2)
        # axes[idx].legend()
        # set subtitle for each subplot
        # plt.savefig(f"./Private/compores_response_ranking/correlation_p_values_{abx}.png")
    # Add label only for the first point in the first treatment and index
    for gene, color in gene_color_dict.items():
        legend_handles.append(
            Line2D([0], [0], color=color, markerfacecolor=color, markersize=12,
                   label=names_dictionary[gene])
        )
    axes[-1].legend(handles=legend_handles, title=names, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"./Private/compores_response_ranking/correlation_p_values_{folder}_{names}.png", dpi=600)
    # plt.savefig(f"./Private/compores_response_ranking/correlation_p_values_{folder}_{name}.png")
    # plt.show()
    plt.close()


def get_specific_value(gene, genes, specific_compores_results_spf):
    compores_results_spf_specific = \
        specific_compores_results_spf[specific_compores_results_spf["genes_name"] == gene][f"correlation_p"]
    # assert set(genes) == set(specific_compores_results_spf["genes_name"].values)
    specific_value = compores_results_spf_specific.values[0]
    return specific_value


def get_compores_results(abx, path, treat):
    data = read_and_print_pkl(path + rf'\{abx}-{treat}-feces\pairs\mean_log_p_value.pkl')
    data = data[f"{abx}-{treat}-feces"]
    index = read_and_print_pkl(path + rf'\{abx}-{treat}-feces\pairs\response_index.pkl')
    column_names = ['gene', 'correlation']
    compores_results_spf = pd.DataFrame({column_names[0]: index, column_names[1]: data})
    compores_results_spf[f'{column_names[1]}_p'] = np.exp(-compores_results_spf[column_names[1]])
    compores_results_spf["genes_name"] = compores_results_spf[column_names[0]].str.split('_').str[-1]
    return compores_results_spf


def significant_viral(viral_genes, significant_genes):
    for treat in treatments:
        print(f"Significant genes for {treat}")
        intersect = significant_genes[treat].intersection(viral_genes)
        print(len(intersect), intersect)


def significance(genes):
    from ClusteringGO import transform_data
    from scipy.stats import ttest_ind
    genome, metadata, partek, transcriptome = read_process_files(new=False)
    transcriptome, metadata = transform_data(transcriptome, metadata, "RASflow", skip=True)
    metadata["Category"] = metadata['Drug'] + "_" + metadata['Treatment']
    significant = {}
    for treat in treatments:
        significant[treat] = {}
        for abx in antibiotics:
            significant[treat][abx] = {}
            samples = metadata[
                ((metadata["Drug"] == abx) | (metadata["Drug"] == "PBS")) & (metadata["Treatment"] == treat)]
            curr = transcriptome[samples["ID"]]
            abx_samples = metadata[(metadata['Drug'] == abx) & (metadata["Treatment"] == treat)]
            pbs_samples = metadata[(metadata['Drug'] == 'PBS') & (metadata["Treatment"] == treat)]
            available_genes = [gene for gene in genes if gene in curr.index]
            print(f"{len(available_genes)} are available out of original {len(genes)}")
            for gene in available_genes:
                # get treat-test score for the gene
                abx_data = (curr.loc[gene][abx_samples['ID']])
                pbs_data = (curr.loc[gene][pbs_samples['ID']])
                t_abx, t_p_abx = ttest_ind(abx_data, pbs_data)
                if t_p_abx < 0.05:
                    if t_abx > 0:  # meaning the abx is enhanced
                        significant[treat][abx][gene] = "Enhanced"
                    else:
                        significant[treat][abx][gene] = "Suppressed"
                else:
                    significant[treat][abx][gene] = "ns"
    return significant


def plot_ip_po_distribution(log=True, threshold=0.05):
    from scipy.stats import ttest_ind
    from scipy.stats import genextreme

    # Load data
    ip_data = pd.read_csv(
        f"./Private/CompoResVerification/Van-IP-intersect-GF{'0_05' if threshold == 0.05 else ''}.tsv", sep="\t")
    po_data = pd.read_csv(
        f"./Private/CompoResVerification/Van-PO-intersect-FMT{'0_05' if threshold == 0.05 else ''}.tsv", sep="\t")

    ip_data["correlation"] = -np.log10(ip_data["correlation_p"])
    po_data["correlation"] = -np.log10(po_data["correlation_p"])

    # Prepare the data for seaborn
    ip_data['Group'] = 'IP'
    po_data['Group'] = 'PO'

    colors = {'IP': '#1f77b4', 'PO': '#808080'}

    col = 'correlation' if log else 'correlation_p'
    value = 'CompoRes -log(p value)' if log else 'CompoRes p value'
    threshold = -np.log10(0.05) if log else 0.05

    # Combine data
    combined_data = pd.concat([po_data, ip_data])

    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Density Plot (KDE)
    bw = 0.5 if log else 0.55
    sns.kdeplot(data=combined_data, x=col, hue='Group', fill=True, common_norm=False,
                palette=colors, alpha=0.6, ax=axs[0], bw_adjust=bw, legend=False)
    # axs[0].set_title('Density Plot')
    # axs[0].legend(labels=['IP', 'PO'])
    axs[0].set_ylabel('Relative Frequency')
    axs[0].set_xlabel(value)
    # # Manually add hatching to IP group violins
    # for patch in axs[0].collections:
    #     if isinstance(patch, plt.Polygon) and patch.get_facecolor() == colors["IP"]:  # IP color
    #         patch.set_hatch('..')  # Set the hatch pattern for IP

    if log:
        # Fit GEV distribution to IP data
        params_ip = genextreme.fit(ip_data[col])
        x_ip = np.linspace(0, 2, 1000)
        pdf_ip = genextreme.pdf(x_ip, *params_ip)
        # Fit GEV distribution to PO data
        params_po = genextreme.fit(po_data[col])
        x_po = np.linspace(0, 2, 1000)
        pdf_po = genextreme.pdf(x_po, *params_po)

        # Plot the GEV fit for IP
        axs[0].plot(x_ip, pdf_ip, color=colors['IP'], label='GEV Fit - IP', linestyle='--')
        # Plot the GEV fit for PO
        axs[0].plot(x_po, pdf_po, color=colors['PO'], label='GEV Fit - PO', linestyle='--')

    # Add arrow at x=0.05
    axs[0].annotate('p=0.05', xy=(threshold, axs[0].get_ylim()[1]), xytext=(threshold, axs[0].get_ylim()[1] * 1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.8, headwidth=8), ha='center', va='bottom',
                    fontsize=14)  # , fontweight='bold')
    axs[0].set_xlim(left=0, right=max(1, 1.2 * max(combined_data[col])))

    # Violin Plot
    sns.violinplot(data=combined_data, x='Group', y=col, inner='quartile',
                   palette=colors, ax=axs[1], bw=0.15)
    # axs[1].set_title('Violin Plot')
    # Statistical test for significance
    t_stat, p_value = ttest_ind(ip_data[col], po_data[col])

    # Add significance annotation
    # significance = '****' if p_value < 0.0001 else f'p = {p_value:.3e}'
    if p_value < 0.0001:
        significance = '****'
    elif p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'ns'  # Not significant
    top = max(1, 1.1 * max(combined_data[col]))
    axs[1].annotate(significance, xy=(0.5, top), xycoords='data', ha='center', va='bottom', fontsize=12)
    axs[1].plot([0, 1], [top] * 2, color='black')
    # get PO max col value
    space = 0.15 if log else 0.09
    po_max = po_data[col].max()
    axs[1].plot([0, 0], [po_max + space, top], color='black')
    ip_max = ip_data[col].max()
    axs[1].plot([1, 1], [ip_max + space, top], color='black')
    axs[1].set_ylim(bottom=0, top=max(1.05, 1.2 * max(combined_data[col])))
    axs[1].set_xlabel('')
    axs[1].set_ylabel(value)
    # axs[1].set_ylabel('CompoRes -log(p value)')

    # # Manually add hatching to IP group violins
    # for patch in axs[1].collections:
    #     # Check if patch is a polygon (the violin body) and if its color is close to the IP blue
    #     if isinstance(patch, plt.Polygon):
    #         # Get the RGB color of the patch
    #         patch_color = np.array(patch.get_facecolor())[:3]  # Ignore the alpha channel
    #         ip_color_rgb = np.array([31 / 255, 119 / 255, 180 / 255])  # RGB for #1f77b4
    #
    #         # Compare the colors by calculating the Euclidean distance between them
    #         if np.allclose(patch_color, ip_color_rgb, atol=0.1):
    #             patch.set_hatch('..')  # Set the hatch pattern for IP violins

    # Final adjustments
    anchor = (0.5, 0.8) if log else (0.516, 0.75)
    fig.legend(labels=['IP', 'PO'], loc='center', bbox_to_anchor=anchor, ncol=1, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)  # Adjust the vertical space between subplots
    plt.savefig(f"./Private/compores_response_ranking/IP_PO_distribution{'_log' if log else ''}.png", dpi=600)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    pass
    # # clock_genes = ["Nfil3", "Map1lc3a", "Dbp", "Ciart", "Atg9a", "Atg13", "Arntl"]
    # part_circadian_clock_genes = ["nfil3", "ciart", "dbp", "per3", "arntl"]
    # checking_genes = ["Isg15", "Oasl2", "Zbp1"]
    # part_circadian_clock_genes = ["per1", "per2", "per3", "cry1", "cry2",
    #                               "arntl", "nfil3", "nr1d1", "dbp", "clock",
    #                               "ciart"]  # "bmal1" is arntl. "chrono" is "ciart"?
    # autophagy_genes = ['Atg10', 'Atg101', 'Atg12', 'Atg13', 'Atg14', 'Atg16l1', 'Atg16l2', 'Atg2a', 'Atg2b',
    #                    'Atg3', 'Atg4a', 'Atg4b', 'Atg4c', 'Atg4d', 'Atg5', 'Atg7', 'Atg9a', 'Atg9b',
    #                    'Map1lc3a', 'Map1lc3b', 'Sqstm1', 'Gabarap', 'Gabarapl1', 'Gabarapl2',
    #                    'Becn1', 'Ulk1', 'Ulk2', 'Ulk3', 'Ulk4', 'Wipi2']
    # viral_genes = akiko_check()
    # part_circadian_clock_genes = ["per3"]
    # circadian_clock_genes = [gene.capitalize() for gene in checking_genes]
    # circadian_clock_genes = [gene.capitalize() for gene in viral_genes]
    # circadian_clock_genes = [gene.capitalize() for gene in part_circadian_clock_genes]
    # circadian_clock_genes = ["atg9a", "atg13", "map1lc3a", "per1", "per2", "cry1", "cry2",
    #                          "arntl", "nfil3", "nr1d1", "dbp", "clock",
    #                          "ciart"]  # "bmal1" is arntl. "chrono" is "ciart"?
    # # capitalize circadian_clock_genes
    # circadian_clock_genes = [gene.capitalize() for gene in circadian_clock_genes]
    # part_circadian_clock_genes = [gene.capitalize() for gene in part_circadian_clock_genes]
    # autophagy_genes = [gene.capitalize() for gene in autophagy_genes]
    # ensmus_dict = get_ensmus_dict()
    # # reverse this dictionary
    # names_dict = {v: k for k, v in ensmus_dict.items()}
    # ensmus_clock_genes = list(names_dict[gene] for gene in part_circadian_clock_genes if gene in names_dict)
    # ensmus_autophagy_genes = list(names_dict[gene] for gene in autophagy_genes if gene in names_dict)
    # ensmus_clock_genes = list(names_dict[gene] for gene in circadian_clock_genes if gene in names_dict)
    # groups = {
    #     "autophagy": ensmus_clock_genes[:3],
    #     "core_clock": ensmus_clock_genes[3:7],
    #     "other_clock": ensmus_clock_genes[7:],
    # }
    # missing = set(circadian_clock_genes) - set([ensmus_dict[gene] for gene in ensmus_clock_genes])
    # if missing:
    #     print(f"{missing} are missing from the original list")
    # significant_clock_genes = significance(ensmus_clock_genes)
    # prepare_genes_to_compores(threshold=0.05, by_genes=viral_genes, folder="viral")
    # prepare_genes_to_compores(threshold=0.05, by_genes=ensmus_clock_genes, folder="auroc")
    # prepare_genes_to_compores(threshold=0.05, by_genes=ensmus_clock_genes, folder="clock")
    # prepare_genes_to_compores(threshold=0.05, by_genes=ensmus_autophagy_genes, folder="autophagy_all")  # rerun 0.05
    # for group in groups:
    #     box_plot_compores_comparison_clock(groups[group], "clock", group, significant_clock_genes, ensmus_dict)
    # box_plot_compores_comparison_clock(ensmus_clock_genes, "clock", circadian_clock_genes, significant_clock_genes)
    #
    # viral_genes = akiko_check()
    # # significant_genes = neo_significance(threshold=0.01)
    # # significant_viral(viral_genes, significant_genes)
    # compare_compores_all_antibiotics(viral_genes, "viral", threshold=True)
    # compare_compores_all_antibiotics(viral_genes, "viral")
    # neo_compores(viral_genes)
    # # # NOTE: 525 are available out of original 586 viral genes
    # # prepare_genes_to_compores(threshold=0.05, by_genes=list(viral_genes), folder="viral")  # rerun 0.05
    # box_plot_compores_comparison_specific(viral_genes, "viral")
    # compare_correlation_fmt(vs_all=True, threshold=0.01)
    # compare_correlation_gf("Van", "IP", threshold=0.01)
    # # plot_ip_po_distribution(threshold=0.05)
    # plot_ip_po_distribution(threshold=0.01, log=False)
    # quit()
    #
    # # # prepare_clock_genes_to_compores(clock_genes)
    # # plot_clock_genes_compores()
    # # quit()
    # prepare_genes_to_compores(threshold=0.05, folder="zheniya")
    # # # show_case_correlated_genes()  # for all we ran 0.01
    # # compare_correlation_all()
    # # box_plot_compores_comparison()
    # # compare_correlation_fmt(vs_all=True, threshold=0.01)
    # quit()
    # ensmus_to_gene = get_ensmus_dict()
    # all_significant, gf, spf, correlated_intersection, uncorrelated_intersection = compare_correlation_gf("Van", "IP",
    #                                                                                                       threshold=0.01)
    # # quit()
    # spf.index = [ensmus_to_gene[gene] if gene in ensmus_to_gene else gene for gene in spf.index]
    # spf = spf.groupby(spf.index).sum()
    # # 1) All genes (log2 fc like you did), comparing the SPF and GF experiments.
    # log2fc_plot(gf.index.intersection(spf.index), gf, spf, "all_genes")
    # # 2) All genes that are significant (p-value) in both SPF and GF, regardless of CompoRes
    # log2fc_plot(all_significant, gf, spf, "all_significant_genes (no CompoRes)")
    # # 3) Genes that are significant in SPF and are uncorrelated with microbiota using CompoRes, vs GF significant genes
    # log2fc_plot(correlated_intersection, gf, spf, "GF_significant_SPF_microbiome_correlated")
    # # 4) Genes that are significant in SPF and are correlated with microbiota using CompoRes, vs GF significant genes
    # log2fc_plot(uncorrelated_intersection, gf, spf, "GF_significant_SPF_microbiome_uncorrelated")
    # # 5) Downsample (1) so that it’s the same size like (3) and like (4), to see how the p-values are after downsampling
    # random_all = np.random.choice(gf.index.intersection(spf.index), len(correlated_intersection), replace=False)
    # log2fc_plot(random_all, gf, spf, f"all_genes (down sampled to size of correlated {len(correlated_intersection)})")
    # random_all = np.random.choice(gf.index.intersection(spf.index), len(uncorrelated_intersection), replace=False)
    # log2fc_plot(random_all, gf, spf,
    #             f"all_genes (down sampled to size of uncorrelated {len(uncorrelated_intersection)})")
    # # 5) 6) Downsample (2) so that it’s the same size like (3) and like (4), to see how the p-values are after downsampling
    # random_all = np.random.choice(all_significant, len(correlated_intersection), replace=False)
    # log2fc_plot(random_all, gf, spf,
    #             f"all_significant_genes (down sampled to size of correlated {len(correlated_intersection)})")
    # random_all = np.random.choice(all_significant, len(uncorrelated_intersection), replace=False)
    # log2fc_plot(random_all, gf, spf,
    #             f"all_significant_genes (down sampled to size of uncorrelated {len(uncorrelated_intersection)})")
    # quit()
