import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from goatools import obo_parser
from scipy.stats import linregress

from ClusteringGO import (antibiotics, treatments, get_ancestor, get_go, get_metadata, transform_data,
                          private, path, read_process_files)


def parse_data(folder, type="", only_old=True):
    file = f"rpkm_named_genome-2023-09-26.tsv" if type else f"transcriptome_2023-09-17-genes_norm_named.tsv"
    df = pd.read_csv(folder + file, sep="\t")
    # # count number of appearances of strings from "gene_name"
    # count = Counter(data['gene_name'].values)
    # count = Counter(data['gene_id'].values)
    # # print all values that appear more than once
    # print([(key, value) for key, value in count.items() if value > 1])
    # drop gene_id column
    df = df.drop('gene_id', axis=1)
    # remove all samples that end with N from metadata and from data
    if only_old:
        df = df.drop([col for col in df.columns if col.endswith('N')], axis=1)
    return df


def normalize_raw_data(data_frame):
    """
    Normalize each column by the sum of the row
    """
    return data_frame.div(data_frame.sum(axis=0), axis=1)


data_path = os.path.join("..", "Data")
raw_data_path = os.path.join(data_path, "MultiAbx-16s", "MultiAbx-RPKM-RNAseq-B6",
                             "Partek_bell_all_Normalized_new_controls.csv")
# Alternative raw_data_path, commented out for example purposes:
# raw_data_path = os.path.join(data_path, "MultiAbx-16s", "MultiAbx-RPKM-RNAseq-B6", "New Partek_bell_all_Normalization_Normalized_counts1.csv")
# Another example, for an absolute path (commented out):
# raw_data_path = os.path.join("C:", "Users", "Yehonatan", "Desktop", "Master", "Git", "DEP_Compare16s", "Private", "imputed_all_log_zeros_removed.csv")

meta_data_path = os.path.join(data_path, "MultiAbx-16s", "MultiAbx-RPKM-RNAseq-B6", "all-samples-noC9C10-newC.csv")
# Alternative meta_data_path, commented out for example purposes:
# meta_data_path = os.path.join(data_path, "MultiAbx-16s", "MultiAbx-RPKM-RNAseq-B6", "all-samples-noC9.tsv")

# Assuming 'path' is defined earlier in your code as shown in the previous example:
all_path = os.path.join(path, "diff_abx", "top_correlated_GO_terms.tsv")

# For 'private' directory, assuming it's defined as shown in the previous response:
all_dir = os.path.join(private, "hist", "png", "")
treatments = np.array(treatments)
antibiotics = np.array(antibiotics)
treat_color = {'IP': 'red', 'IV': 'blue', 'PO': 'green'}
antibiotic_shape = {'Van': 'o', 'Met': '^', 'Amp': 's', 'Mix': 'd', 'Neo': 'p'}


def z_score_by_pbs(data, abx, pbs):
    """
    applied z_score normalization for all data by the values of the pbs mice columns
    """
    # get the pbs mice data
    pbs_data = data[pbs['ID']]
    # get the abx mice data
    abx_data = data[abx['ID']]
    # calculate the mean and std of the pbs mice
    pbs_mean = pbs_data.mean(axis=1)
    pbs_std = pbs_data.std(axis=1)
    # normalize the data by the mean and std of the pbs mice: subtract pbs_mean from every row and divide by std
    normalized_data = data.sub(pbs_mean, axis=0)
    normalized_data = normalized_data.div(pbs_std, axis=0)
    # return the normalized data
    return normalized_data


# def plot_clusters(raw_data):
#     for file in os.listdir(path):
#         if file.split(".")[-1] != "tsv":
#             continue
#         if len(file) < 33:
#             print(file)
#             continue
#         treat = (file.split("_")[-1]).split(".")[0]
#         abx = file.split("_")[-2]
#         # if treat != "IP" or abx != "Van":
#         #     continue
#         # if treat in ["IP", "IV"] or abx in ["Van", "Met"]:
#         #     continue
#         abx_mice = meta[(meta['Drug'] == abx) & (meta['Treatment'] == treat)]["ID"]
#         pbs_mice = meta[(meta['Drug'] == 'PBS') & (meta['Treatment'] == treat)]["ID"]
#         df = pd.read_csv(path + file, sep="\t").fillna(0)
#         best = df[(df['treat-test p-value'] < 0.05) & (df['MWU'] < 0.05) & (df['size'] > 5) & (
#                 df['better than parent'] != False) & (df['better than random correlation'] == 1)]
#         #  & (df["less than 5%"] == 1)]
#         # sort best by "distance" value
#         # # sort best from highest to lowest 'distance'
#         # best = best.sort_values(by="\"distance\"", ascending=False)
#         best = best.sort_values(by="MWU")
#         # best = best.sort_values(by="\"distance\"")
#         if best.empty:
#             print(abx, treat, "is empty")
#             continue
#         # iterate over rows in best dataframe and plot pcolormesh of "genes" column from raw all_data
#         for index, row in best.iterrows():
#             name = row["GO term"]
#             genes = row["genes"].split(",")
#             genes = [gene.strip("{").strip("}").strip(' ').strip("\"").strip("\'") for gene in genes]
#             genes_df = pd.DataFrame()
#             for i, gene in enumerate(genes):
#                 genes_raw = raw_data[raw_data.index == gene]
#                 genes_df = genes_df.append(pd.concat([genes_raw[abx_mice], genes_raw[pbs_mice]], axis=1))
#             # plot gene all_data using seaborn
#             if genes_df.empty:
#                 print(f"{name} in {abx} {treat} is empty")
#                 continue
#             plt.figure(figsize=(20, 10))
#             try:
#                 normalized_genes = z_score_by_pbs(genes_df, abx_mice, pbs_mice)
#                 sns.clustermap(data=normalized_genes, row_cluster=True, col_cluster=False,
#                                cmap='vlag')
#                 # z_score=0, cmap='vlag')
#             except ValueError:
#                 print(f"{name} condensed distance matrix {abx} {treat} is empty")
#                 plt.close()
#                 continue
#             plt.title(f"{abx} {treat} {name}")
#             enhance = "enhanced" if row['enhanced?'] == 1 else "suppressed"
#             curr_path = "./Private/analysis/"
#             plt.savefig(os.path.join(curr_path, f"{abx}_{treat}_GO{name.split(':')[1]}_{enhance}.png"))
#             plt.show()
#             plt.close()


def plot_correlation(df, title, x_name, y_name, folder=""):
    df = df[df['size'] < 800]
    # plt.scatter(df[x_name], df[y_name], cmap='viridis')
    plt.hist2d(df[x_name], df[y_name], bins=10, cmap='viridis')
    # plt.title(f"{title}{abx} {treat} {x_name} vs {y_name}")
    plt.xlabel(x_name.strip('\"'))
    plt.ylabel(y_name)
    x_name = x_name.strip("\"")
    y_name = y_name.strip("\"")
    plt.savefig(
        f"C:\\Users\\Yehonatan\\Desktop\\Master\\Git\\DEP_Compare16s\\Private\\{folder}{title}_{x_name}_{y_name}.png")
    # f"C:\\Users\\Yehonatan\\Desktop\\Master\\Git\\DEP_Compare16s\\Private\\{folder}{title}{abx}_{treat}_{x_name}_{y_name}.png")
    plt.show()
    plt.close()


# def plot_h2ab1(data):
#     # scatter plot for 'H2-Ab1' of distance vs. # genes in cluster (color by treatment, shape by abx)
#     h2ab1 = data[data['with H2-Ab1?'] == 1]
#     # plt.scatter(h2ab1['\"distance\"'], h2ab1['size'], cmap='viridis',
#     #             marker=h2ab1['Antibiotics'].map(antibiotic_shape),
#     #             c=h2ab1['Treatment'].map(treat_color))
#     for i in range(len(h2ab1)):
#         plt.scatter(h2ab1['\"distance\"'].iloc[i], h2ab1['size'].iloc[i], c=treat_color[h2ab1['Treatment'].iloc[i]],
#                     marker=antibiotic_shape[h2ab1['Antibiotics'].iloc[i]])  # , s=15, alpha=0.5)
#     plt.title(f"H2-Ab1 distance vs. # genes in cluster")
#     # write on the graph a legend of each treatment and antibiotics
#     color_name = [key for key in treat_color]
#     marker_name = [key for key in antibiotic_shape]
#     colors = [treat_color[key] for key in treat_color]
#     marks = [antibiotic_shape[key] for key in antibiotic_shape]
#     rows = [mpatches.Patch(color=colors[i]) for i in range(len(colors))]
#     columns = [plt.plot([], [], marks[i], markerfacecolor='w',
#                         markeredgecolor='k')[0] for i in range(len(marks))]
#     plt.legend(columns + rows, marker_name + color_name, loc=2)
#     plt.xlabel("cluster distance")
#     plt.ylabel("cluster size")
#     save_path = "C:\\Users\\Yehonatan\\Desktop\\Master\\Git\\DEP_Compare16s\\Private\\"
#     plt.savefig(save_path + f"analysis\\h2ab1_all_distance_vs_size.png")
#     plt.show()


def plot_medians(df, raw, abx_mice, pbs_mice, title, show=True, save=True):
    """
    iterate over the clusters in df, and for each cluster get the median of the genes by the raw all_data
    """
    if df.empty:
        return
    mice = pd.concat((abx_mice['ID'], pbs_mice['ID']))
    genes_df = pd.DataFrame()
    # iterate over the rows of the dataframe
    for index, row in df.iterrows():
        # get the genes in the cluster
        genes = row["genes"].split(",")
        genes = [gene.strip("[").strip("]").strip("{").strip("}").strip(' ').strip("\"").strip("\'") for gene in genes]
        if "=" in genes[0]:
            continue
            # TODO: fix for large GOs
        # todo: why???
        genes = [gene for gene in genes if gene in raw.index]
        # get the median of the genes in the cluster
        median = np.concatenate([np.array([row['GO term']]), np.median(raw[mice].loc[genes], axis=0)])
        # plot the median of the genes in the cluster
        # plt.scatter(median, row["size"], c=treat_color[row["Treatment"]], marker=antibiotic_shape[row["Antibiotics"]])
        # genes_df = genes_df.append(pd.Series(median), ignore_index=True)
        genes_df = pd.concat([genes_df, pd.Series(median).rename(row['GO term'], inplace=False)],
                             ignore_index=True, axis=1)
    genes_df = genes_df.T
    # sns.set(rc={'figure.figsize': (15, 5)})
    # genes_df.columns = pd.Series(['index']).append(mice)
    genes_df.columns = pd.concat([pd.Series(['index']), pd.Series(mice)])
    genes_df = genes_df.set_index('index').astype('f')
    normalized_df = z_score_by_pbs(genes_df, abx_mice, pbs_mice)  # todo: note this normalization!
    # normalized_df = genes_df
    # drop nan rows  # todo: why are there nan rows?
    normalized_df = normalized_df.dropna()
    if normalized_df.shape[0] > 1:
        cluster = sns.clustermap(data=normalized_df, row_cluster=True, col_cluster=False,
                                 cmap='vlag')  # , xticklabels=True, yticklabels=True)
        # z_score=0, cmap='vlag')  # , xticklabels=True, yticklabels=True)
        order = cluster.dendrogram_row.reordered_ind
        plt.close()
        return normalized_df.fillna(0).iloc[order]  # .apply(zscore, axis=1)
    return normalized_df.fillna(0)
    # return genes_df.iloc[order]


def plot_clusters_separately(df, raw, abx_mice, pbs_mice, title, show=True, save=True):
    """
    iterate over the clusters in df, and for each cluster get the median of the genes by the raw all_data
    """
    if df.empty:
        return
    mice = pd.concat((abx_mice['ID'], pbs_mice['ID']))
    # iterate over the rows of the dataframe
    for index, row in df.iterrows():
        # get the genes in the cluster
        genes = row["genes"].split(",")
        genes = [gene.strip("{").strip("}").strip(' ').strip("\"").strip("\'") for gene in genes]
        cluster = raw[mice].loc[genes]
        normalized_cluster = z_score_by_pbs(cluster, abx_mice, pbs_mice)
        sns.clustermap(data=normalized_cluster, row_cluster=True, col_cluster=False,
                       cmap='vlag')  # , xticklabels=True, yticklabels=True)
        # z_score=0, cmap='vlag')  # , xticklabels=True, yticklabels=True)
        suppress = "enhanced" if row["enhanced?"] == True else "suppressed"
        plt.title(f"{title}{row['GO term']} {row['name']} {suppress}")
        if save:
            plt.savefig(f"./Private/{row['Antibiotics']}_{row['Condition']}_GO_{row['GO term'][3:]}_{suppress}.png")
        if show:
            plt.show()


# def plot_all():
#     low_p = all_data[(all_data['treat-test p-value'] < 0.01)]
#     other = all_data[(all_data['MWU'] < 0.05) & (all_data['treat-test p-value'] < 0.05) & (all_data['size'] >= 10) & (
#             all_data['better than parent'] != False) & (all_data['better than random correlation'] == 1)]
#     abx_data = meta[meta['Drug'] != 'PBS']
#     pbs_data = meta[meta['Drug'] == 'PBS']
#     abx_data = abx_data[~abx_data['ID'].astype(str).str.startswith('S')]
#     plot_medians(low_p, raw, abx_data, pbs_data, "p-val_less_1")
#     plot_medians(other, raw, abx_data, pbs_data, "all")


def get_to_axis(axis, i, j, n, m):
    if n > 1 and m > 1:
        return axis[i, j]
    elif n ==1 and m == 1:
        return axis
    else:
        return axis[max(i, j)]


def get_clusters_names_dict(abx, treat, exp_type, space=50):
    file = pd.read_csv(rf"./Private/clusters_properties/{exp_type}/top_correlated_GO_terms_{abx}_{treat}.tsv", sep="\t")
    # create a dictionary from column GO term to name
    clusters_names_dict = dict(zip(file['GO term'], file['name']))
    clusters_names_dict = {key: value.split(":")[1] for key, value in clusters_names_dict.items()}
    truncated_dict = {
        key: ' '.join(value[:value[:space].rfind(' ')].split()) + ' [...]' if len(value) > 40 else value
        for key, value in clusters_names_dict.items()
    }
    return truncated_dict


def plot_median_all_conditions(meta_data, raw_data, antibiotics, treatments, condition, exp_type, run_type="",
                               labelsize=12, regular=True, cols_factor=6.0, rows_factor=5.0):
    matrices = get_median_matrices(antibiotics, condition, exp_type[1:], meta_data, raw_data, treatments, regular)
    axis = set_figure(treatments, antibiotics, cols_factor, rows_factor)
    GO_number = pd.DataFrame(index=antibiotics, columns=treatments, data=0)
    # for j, treat in enumerate(conditions):
    #     for i, abx in enumerate(antibiotics):
    for j, treat in enumerate(treatments):
        for i, abx in enumerate(antibiotics):
            curr_axis = get_to_axis(axis, i, j, len(treatments), len(antibiotics))
            curr_axis.set_title(f"{abx}, {treat}")
            cbar = False if i != 1 or j != 4 else True
            if matrices[treat][abx] is not None:
                cluster_names_dict = get_clusters_names_dict(abx, treat, exp_type)
                # matrices[treat][abx].T.to_csv(f"./Private/medians/only_medians/{abx}_{treat}.csv")
                curr_matrix = matrices[treat][abx]
                # sort columns and put all columns that ends with N in the end
                curr_matrix = curr_matrix.reindex(sorted(curr_matrix.columns, key=lambda x: x.endswith('N')), axis=1)
                # sns.heatmap(curr_matrix, vmin=-2.8, vmax=2, xticklabels=True, cmap="vlag", ax=curr_axis, cbar=cbar)
                GO_number.loc[abx, treat] = curr_matrix.shape[0]
                sns.heatmap(curr_matrix, xticklabels=True, cmap="vlag", ax=curr_axis, cbar=cbar, vmax=5, vmin=-5)
                # label_colors = ['blue' if label.startswith('C') else 'red' for label in curr_matrix.columns]
                label_colors = ['blue' if meta_data[meta_data["ID"] == label]["Drug"].values == "PBS" else 'red' for
                                label in curr_matrix.columns]
                bar_height = 0.01 * curr_matrix.shape[0]
                for k, color in enumerate(label_colors):
                    bar_width = 1  # Set the width to match a column (fixed at 1)
                    curr_axis.add_patch(
                        plt.Rectangle((k, curr_matrix.shape[0] - bar_height), bar_width, bar_height, color=color,
                                      fill=True))

                curr_axis.yaxis.label.set_visible(False)
                # replace y labels with cluster names, using dictionary above
                curr_axis.set_yticklabels(
                    [cluster_names_dict[label.get_text()] for label in curr_axis.get_yticklabels()], rotation=0)
                tl = curr_axis.get_xticklabels()
                curr_axis.set_xticklabels(tl, rotation=45, fontsize=labelsize)
                tl = curr_axis.get_yticklabels()
                curr_axis.set_yticklabels(tl, rotation=0)
            else:
                print(f"{abx} {treat} is empty")
    # increase vertical space between plots
    plt.subplots_adjust(wspace=1.5)
    # plt.subplots_adjust(hspace=2)
    # decrease all axis labels size
    plt.rc('xtick', labelsize=labelsize)
    # plt.title(" ")
    plt.savefig(private + fr"/analysis/{exp_type}/{exp_type}{run_type} medians_of_all.png", bbox_inches='tight')
    plt.show()
    plt.close()

    # save GO_number to a csv file
    GO_number.to_csv(private + fr"/analysis/{exp_type}/{exp_type}{run_type} GO_number.csv")


def get_median_matrices(antibiotics, condition, exp_type, meta_data, raw_data, treatments, regular=True):
    # meta_data = meta_data.drop(meta_data[meta_data['ID'] == 'V16'].index).drop(meta_data[meta_data['ID'] == 'V17'].
    #                                                                            index). \
    #     drop(meta_data[meta_data['ID'] == 'V18'].index).drop(meta_data[meta_data['ID'] == 'N18'].index)
    matrices = {}
    for treat in treatments:
        matrices[treat] = {}
        for abx in antibiotics:
            abx_data = meta_data[(meta_data['Drug'] == abx) & (meta_data[condition] == treat)]
            pbs_data = meta_data[(meta_data['Drug'] == 'PBS') & (meta_data[condition] == treat)]
            selected = get_selected_df(abx, treat, exp_type, regular)
            # selected = df[(df['size'] >= 5) & (df['better than parent'] != False) &
            #               (df['better than random correlation'] == 1)].sort_values(by='MWU').head(20)
            print(abx, treat, selected.shape)
            matrix = plot_medians(selected, raw_data, abx_data, pbs_data, f"{treat} {abx}", True, False)
            matrices[treat][abx] = matrix
    return matrices


def compare_to_gsea(meta_data, antibiotics, treatments, condition, exp_type):
    gsea_res = {
        "Van-IP": 957 + 0,
        "Van-IV": 35 + 0,
        "Van-PO": 76 + 0,
        "Mix-IP": 238 + 0,
        "Mix-IV": 148 + 0,
        "Mix-PO": 0,
        "Met-IP": 649 + 0,
        "Met-IV": 5 + 0,
        "Met-PO": 9 + 0,
        "Amp-IP": 460 + 0,
        "Amp-IV": 9 + 0,
        "Amp-PO": 805 + 0,
        "Neo-IP": 253 + 0,
        "Neo-IV": 380 + 0,
        "Neo-PO": 272 + 0,
    }
    # gsea_res_all = {
    #     "Van-IP": 957+135,
    #     "Van-IV": 35+6,
    #     "Van-PO": 76+19,
    #     "Mix-IP": 238+37,
    #     "Mix-IV": 148+40,
    #     "Mix-PO": 0,
    #     "Met-IP": 649+82,
    #     "Met-IV": 5+0,
    #     "Met-PO": 9+0,
    #     "Amp-IP": 460+76,
    #     "Amp-IV": 9+3,
    #     "Amp-PO": 805+141,
    #     "Neo-IP": 253+80,
    #     "Neo-IV": 380+98,
    #     "Neo-PO": 272+80,
    # }
    # gsea_res = {
    #     "Van-IP": 87 + 999,
    #     "Van-IV": 5 + 0,
    #     "Van-PO": 7 + 85,
    #     "Mix-IP": 99 + 179,
    #     "Mix-IV": 7 + 171,
    #     "Mix-PO": 62 + 164,
    #     "Met-IP": 1 + 721,
    #     "Met-IV": 1 + 3,
    #     "Met-PO": 0 + 9,
    #     "Amp-IP": 430 + 116,
    #     "Amp-IV": 4 + 8,
    #     "Amp-PO": 845 + 90,
    #     "Neo-IP": 17 + 310,
    #     "Neo-IV": 95 + 364,
    #     "Neo-PO": 5 + 350,
    # }
    # gsea_res_pos = {
    #     "Van-IP": 999,
    #     "Van-IV": 0,
    #     "Van-PO": 85,
    #     "Mix-IP": 179,
    #     "Mix-IV": 171,
    #     "Mix-PO": 164,
    #     "Met-IP": 721,
    #     "Met-IV": 3,
    #     "Met-PO": 9,
    #     "Amp-IP": 116,
    #     "Amp-IV": 8,
    #     "Amp-PO": 90,
    #     "Neo-IP": 310,
    #     "Neo-IV": 364,
    #     "Neo-PO": 350,
    # }
    # gsea_res_neg = {
    #     "Van-IP": 87,
    #     "Van-IV": 5,
    #     "Van-PO": 7,
    #     "Mix-IP": 99,
    #     "Mix-IV": 7,
    #     "Mix-PO": 62,
    #     "Met-IP": 1,
    #     "Met-IV": 1,
    #     "Met-PO": 0,
    #     "Amp-IP": 430,
    #     "Amp-IV": 4,
    #     "Amp-PO": 845,
    #     "Neo-IP": 17,
    #     "Neo-IV": 95,
    #     "Neo-PO": 5,
    # }
    plt.figure(figsize=(10, 10))
    for treat in treatments:
        for abx in antibiotics:
            selected = get_selected_df(abx, treat, exp_type)
            print(abx, treat, selected.shape)
            plt.scatter(len(selected), gsea_res[f"{abx}-{treat}"], c='blue')
            plt.text(len(selected), gsea_res[f"{abx}-{treat}"], f"{abx}-{treat}")
    plt.title(f"GSEA vs. our clustering")
    plt.xlabel("our clustering")
    plt.ylabel("GSEA")
    plt.savefig(private + fr"/analysis/{exp_type}/GSEA_vs_our_clustering.png", bbox_inches='tight')
    plt.show()


def set_figure(treats, antibiotics, cols_factor=6.0, rows_factor=5.0):
    rows, cols = len(antibiotics), len(treats)
    fig, axis = plt.subplots(rows, cols, figsize=(cols_factor * cols, rows_factor * rows))
    fig.tight_layout(pad=10.0)
    # font = {'family': 'Sans Serif',
    #         'size': 20}
    # plt.rc('font', **font)
    # plt.ylabel('antibiotics', size=20)
    # plt.xlabel('treatment', size=20)
    return axis


def save_colors_dictionary_as_txt(colors_dict, file_path):
    with open(file_path, "w") as f:
        for key, value in colors_dict.items():
            # Convert NumPy array to string representation
            value_str = np.array2string(value, separator=', ')
            f.write(f"{key}: {value_str}\n")


def load_colors_dictionary_from_txt(file_path):
    colors_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value_str = line.strip().split(": ")
            value = np.fromstring(value_str[1:-1], sep=', ')
            colors_dict[key] = value
    return colors_dict


def get_colors_dictionary(columns):
    # import matplotlib._color_data as mcd
    # colors = list(mcd.XKCD_COLORS.values())[::40]
    # if colors_dict.txt exist, return it
    colors_file_path = "C:\\Users\\Yehonatan\\Desktop\\Master\\Git\\DEP_Compare16s\\Private\\colors_dict.txt"

    # if os.path.exists(colors_file_path):
    if False:
        # if False:
        loaded_colors_dict = load_colors_dictionary_from_txt(colors_file_path)
        return loaded_colors_dict
    # else, create a new dictionary
    else:
        # colors = list(cm.rainbow(np.linspace(0, 1, len(columns))))
        colors = [
            [31 / 255, 120 / 255, 180 / 255, 1],  # Blue
            [51 / 255, 160 / 255, 44 / 255, 1],  # Green
            [227 / 255, 26 / 255, 28 / 255, 1],  # Red
            [255 / 255, 127 / 255, 0 / 255, 1],  # Orange
            [85 / 255, 85 / 255, 85 / 255, 1],  # Dark Gray
            [166 / 255, 206 / 255, 227 / 255, 1],  # Light Blue
            [178 / 255, 223 / 255, 138 / 255, 1],  # Light Green
            [251 / 255, 154 / 255, 153 / 255, 1],  # Light Red
            [0 / 255, 128 / 255, 128 / 255, 1],  # Teal
            [139 / 255, 69 / 255, 19 / 255, 1],  # Saddle Brown
            [138 / 255, 43 / 255, 226 / 255, 1],  # Blue Violet (with transparency)
            [218 / 255, 165 / 255, 32 / 255, 1],  # Goldenrod (with transparency)
            [75 / 255, 0 / 255, 130 / 255, 1],  # Indigo
            [32 / 255, 178 / 255, 170 / 255, 1],  # Light Sea Green
            [169 / 255, 169 / 255, 169 / 255, 1],  # Dark Gray
            [255 / 255, 20 / 255, 147 / 255, 1],  # Deep Pink
            [255 / 255, 215 / 255, 0 / 255, 1],  # Gold
            [154 / 255, 205 / 255, 50 / 255, 1],  # Yellow Green
            [199 / 255, 21 / 255, 133 / 255, 1],  # Medium Violet Red
        ]
        colors = [np.array(color) for color in colors]

        # shuffle colors
        import random
        # set a seed to get the same shuffle every time
        random.seed(0)
        random.shuffle(colors)
        # from matplotlib.cm import get_cmap
        # colors = list(get_cmap("tab20").colors)
        # other_colors = list(get_cmap("tab20b").colors)
        colors_dict = {col: colors[k] for k, col in enumerate(columns)}
        # save the colors dictionary to file
        save_colors_dictionary_as_txt(colors_dict, colors_file_path)

        return colors_dict


def plot_categories(antibiotics, treatments, exp_type, extra=False, loc='lower center', anchor=(0.5, -4.7),
                    regular=True, gsea=False, mix=True):
    size = 32
    go = obo_parser.GODag(get_go())
    categories_size = get_categories_size(go)

    counts_dict_enhanced = {}
    counts_dict_suppressed = {}
    all_go = set()
    # go = obo_parser.GODag(get_go())
    for i, treat in enumerate(treatments):
        counts_dict_enhanced[treat] = {}
        counts_dict_suppressed[treat] = {}
        for j, abx in enumerate(antibiotics):
            # abx_data = meta_data[(meta_data['Drug'] == abx) & (meta_data['Treatment'] == treat)]
            # pbs_data = meta_data[(meta_data['Drug'] == 'PBS') & (meta_data['Treatment'] == treat)]
            selected = get_selected_df(abx, treat, exp_type[1:], regular=regular) if not gsea else get_selected_gsea(
                abx, treat, go)
            unique_enhanced, counts_dict_enhanced[treat][abx] = get_categories(categories_size, go, selected,
                                                                               enhanced=True, regular=regular)
            all_go = all_go.union(unique_enhanced)
            unique_suppressed, counts_dict_suppressed[treat][abx] = get_categories(categories_size, go, selected,
                                                                                   enhanced=False, regular=regular)
            all_go = all_go.union(unique_suppressed)
            # print(abx, treat, counts_dict_high[treat][abx])
            # index = int(rows + cols + str(i * len(treats) + j + 1))
    # print(counts_dict_suppressed)
    # print(counts_dict_enhanced)

    if not all_go:
        print(f"No GO terms found in the selected data {exp_type}.")
        return {}, {}, []

    colors = get_colors_dictionary(all_go)
    enrichment = np.zeros((len(antibiotics), len(treatments)))
    axis = set_figure(treatments, antibiotics)
    for j, treat in enumerate(treatments):
        for i, abx in enumerate(antibiotics):
            curr_axis = get_to_axis(axis, i, j, len(treatments), len(antibiotics))
            curr_axis.set_title(f"{abx}, {treat}", size=size)
            first_loc, second_loc = 0, 0.8
            enhance_enrichment = plot_bar(curr_axis, colors, counts_dict_enhanced[treat][abx], first_loc)
            suppressed_enrichment = plot_bar(curr_axis, colors, counts_dict_suppressed[treat][abx], second_loc)
            curr_axis.set_xticks([first_loc, second_loc], ["enhanced", "suppressed"])
            curr_axis.set_xlim(first_loc - 0.4, second_loc + 0.4)
            curr_axis.tick_params(axis='both', labelsize=size)
            curr_axis.set_xticklabels(curr_axis.get_xticklabels(), rotation=10)
            enrichment[i, j] = enhance_enrichment + suppressed_enrichment

    # labels = np.append(all_go[[taxa in colors for taxa in all_go]], 'other')
    # labels = [key if len(key) < 35 else f"{key[:35]}\n{key[35:]}" for key in all_go]
    labels = list(colors.keys())
    labels.sort(reverse=True)
    orig_labels = labels.copy()
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    linebreak_cutoff = 43
    for i, label in enumerate(labels):
        if len(label) > linebreak_cutoff:
            labels[i] = f"{label[:linebreak_cutoff]}\n{label[linebreak_cutoff + 1:]}"
    lower_center = get_to_axis(axis, len(antibiotics) - 1, len(treatments) // 2, len(antibiotics), len(treatments))
    lower_center.legend(handles, labels, loc=loc, bbox_to_anchor=anchor, fontsize=size)
    # plt.suptitle(f"Categories of GO terms", fontsize=30)
    curr_path = os.path.join(".", "Private", "analysis")
    # verify that the path exists, if not create it
    if not os.path.exists(os.path.join(curr_path, exp_type[1:])):
        os.makedirs(os.path.join(curr_path, exp_type[1:]))
    plt.savefig(os.path.join(curr_path, exp_type[1:], f"{exp_type[1:]} categories.png"), bbox_inches='tight')
    plt.show()

    enhanced = plot_enrichment(exp_type, counts_dict_enhanced, orig_labels, "enhanced", mix)
    suppressed = plot_enrichment(exp_type, counts_dict_suppressed, orig_labels, "suppressed", mix)

    if extra:
        plot_extra(all_go, antibiotics, counts_dict_enhanced, counts_dict_suppressed, curr_path, enrichment, exp_type,
                   treatments)

    return enhanced, suppressed, orig_labels


def plot_enrichment(param, dict, categories, title, mix=True):
    # Flatten the nested dictionary
    flattened_dict = {}
    for outer_key, inner_dict in dict.items():
        for inner_key, value in inner_dict.items():
            flattened_dict[f"{inner_key}-{outer_key}"] = value

    for key in flattened_dict.keys():
        for cat in categories:
            if cat not in flattened_dict[key]:
                # print(f"{key} {cat} is missing")
                flattened_dict[key][cat] = 0

    # Create a DataFrame from the flattened dictionary
    df = pd.DataFrame(list(flattened_dict.values()), index=flattened_dict.keys())
    # print([col for col in df.columns if col not in categories])
    df = df.fillna(0)
    # sort the columns lexicographically
    df = df[categories]

    # if column name is longer than 35 characters, split it to two lines
    linebreak_cutoff = 35
    for i, col in enumerate(df.columns):
        if len(col) > linebreak_cutoff:
            df = df.rename(columns={col: f"{col[:linebreak_cutoff]}\n{col[linebreak_cutoff + 1:]}"})

    # create a figure of size 8x8 inches, 180 dots per inch
    plt.figure(figsize=(10, 8), dpi=180)

    # show all x and y labels
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)

    # Plot a heatmap
    # # sort the columns lexicographically
    # df = df.reindex(sorted(df.columns), axis=1)
    # sort the rows lexicographically, with Mix at the end
    index = [ind for ind in list(df.index) if "Mix" not in ind]
    # sort lexically
    index.sort()
    if mix:
        # add Mix-IP, Mix-IV, Mix-PO at the end
        index.append("Mix-IP")
        index.append("Mix-IV")
        index.append("Mix-PO")
    df = df.reindex(index)
    if param[9:] == "RASflow":
        vmax = 0.5 if title == "enhanced" else 0.15
        heatmap = sns.heatmap(df, cmap="GnBu", vmax=vmax)
        cbar = heatmap.collections[0].colorbar
        actual_vmin, actual_vmax = cbar.vmin, cbar.vmax
        # increase tick size
        cbar.ax.tick_params(labelsize=15)
        if actual_vmax == vmax:
            # Define explicit ticks, ensuring they cover your desired range
            ticks = np.linspace(actual_vmin, actual_vmax, num=5)  # Adjust the number of ticks as needed
            # Set the ticks on the colorbar
            cbar.set_ticks(ticks)
            # Customize tick labels, modifying the last label to indicate a limit
            last = f"{ticks[-1]:.02f}+" if actual_vmax == vmax else actual_vmax
            tick_labels = [f"{tick:.02f}" for tick in ticks[:-1]] + [last]  # Add '+' to the last label
            # Apply the customized tick labels
            cbar.set_ticklabels(tick_labels)
    else:
        sns.heatmap(df, cmap="GnBu")
    # remove y axis label
    plt.ylabel('')
    # Rotate the x-axis labels by 45 degrees
    # plt.xticks(rotation=45)
    plt.title(f"Enrichment of GO terms in {param[9:]} {title}")
    # plt.savefig(private + f"analysis/{param}/ enrichment {title}.png", bbox_inches='tight')
    plt.savefig(os.path.join(private, "analysis", param[1:], f"enrichment {title}.png"), bbox_inches='tight')
    plt.show()
    return df


def plot_extra(all_go, antibiotics, counts_dict_enhanced, counts_dict_suppressed, curr_path, enrichment, exp_type,
               treatments):
    sorted_go = list(all_go)
    sorted_go.sort()
    # sorted_go.insert(0, "treatment")
    # sorted_go.insert(0, "antibiotic")
    sorted_go.insert(0, "condition")
    # axis = set_figure(treatments, antibiotics)
    categories_enh = pd.DataFrame(columns=all_go, dtype=float)
    categories_supp = pd.DataFrame(columns=all_go, dtype=float)
    for i, treat in enumerate(treatments):
        for j, abx in enumerate(antibiotics):
            # curr_axis = get_to_axis(axis, i, j, len(treatments), len(antibiotics))
            # curr_axis.set_title(f"{abx}, {treat}", size=20)
            # counts_dict_suppressed[treat][abx]["treatment"] = treat
            # counts_dict_suppressed[treat][abx]["antibiotic"] = abx
            counts_dict_suppressed[treat][abx]["condition"] = abx + " " + str(treat)
            categories_supp = categories_supp.append(counts_dict_suppressed[treat][abx], ignore_index=True)
            # counts_dict_enhanced[treat][abx]["treatment"] = treat
            # counts_dict_enhanced[treat][abx]["antibiotic"] = abx
            counts_dict_enhanced[treat][abx]["condition"] = abx + " " + str(treat)
            categories_enh = categories_enh.append(counts_dict_enhanced[treat][abx], ignore_index=True)
    plot_categories_altogether(categories_enh, curr_path, exp_type, "enhanced")
    plot_categories_altogether(categories_supp, curr_path, exp_type, "suppressed")
    plot_enrichment_heatmap(enrichment, curr_path, antibiotics, treatments, exp_type)


def plot_categories_altogether(categories, curr_path, exp_type, title):
    categories = categories.set_index("condition")
    categories = categories.fillna(0)
    for col in categories.columns:
        split = col.split(" ")
        if len(split) > 2 and split[1] != "to":
            categories = categories.rename(columns={col: split[0] + " " + split[1]})
    sns.heatmap(categories, cmap="vlag", xticklabels=True, yticklabels=True)
    # plt.xticks(rotation=60)
    plt.title(title)
    # set x axis title to Category and y label to Condition
    plt.xlabel("Category")
    plt.ylabel("Condition")
    # make all fonts and sizes the same
    plt.rc('font', size=20)
    plt.savefig(curr_path + f"{exp_type}\\{exp_type} categories {title} heatmap.png", bbox_inches='tight')
    plt.show()


def get_category_size(term, go):
    tot_sum = len(term.children) + 1
    to_check = set(term.children)
    visited = set(term.id)
    while to_check:
        curr = to_check.pop().id
        if curr in visited:
            continue
        visited.add(curr)
        children = go[curr].children
        to_check = to_check.union(children)
        tot_sum += len(children)
    return tot_sum


def get_categories_size(go):
    # from ClusteringGO import build_genomic_tree
    # go_tree, _ = build_genomic_tree(go['GO:0008150'], go)
    categories_size = {}
    # for child in go_tree.all_children:
    for child in go['GO:0008150'].children:
        categories_size[child.name] = get_category_size(go[child.id], go)
    return categories_size


def plot_enrichment_heatmap(enrichment, curr_path, antibiotics, treatments, exp_type):
    enrichment = pd.DataFrame(enrichment)
    enrichment.columns = antibiotics
    enrichment['treatment'] = treatments
    enrichment = enrichment.set_index('treatment')
    sns.heatmap(enrichment, cmap="vlag", vmax=5)
    plt.title("Enrichment")
    # set x axis title to Antibiotic and y label to Treatment
    plt.xlabel("Antibiotic")
    plt.ylabel("Treatment")
    # make all fonts and sizes the same
    plt.rc('font', size=20)
    plt.savefig(curr_path + f"{exp_type}\\{exp_type} enrichment.png", bbox_inches='tight')
    plt.show()


def get_selected_df(abx, treat, exp_type, regular=True, fdr=True):
    df = pd.read_csv(os.path.join(path, exp_type, f"top_correlated_GO_terms_{abx}_{treat}.tsv"), sep="\t")
    if fdr:
        selected = df[(df['fdr correlation'] < 0.05)]
        # selected = df[(df['fdr GO significance'] < 0.05)]  #TODO
        # selected = df[(df['GO significance'] < 0.05)]  # TODO
        # selected = df[(df['p-value correlation'] < 0.05)]  # TODO
    elif regular:
        selected = df[(df['treat-test p-value'] < 0.05) & (df['size'] >= 2) &
                      # (df['better than parent'] != False) &
                      # (df['better than random'] == "True")]  # todo: change to True?
                      # (df['p-value distance'] <= 0.05)]  # todo: change to True?
                      (df['p-value correlation'] <= 0.05)]  # todo: change to True?
    else:
        selected = df[(df['treat-test p-value'] < 0.05) & (df['size'] >= 2) &
                      # (df['better than parent'] != False) &
                      # (df['better than random'] == True)]
                      # (df['p-value distance'] <= 0.05)]
                      (df['p-value correlation'] <= 0.05)]
    # selected.to_csv(f"./Private/analysis/Diff_abxyasmin/{abx}_{treat}.tsv", sep="\t", index=False)
    return selected


def get_selected_gsea(abx, treat, go):
    go_dict = create_go_term_dict(go)
    # iterate over folders in folder C:\Users\Yehonatan\Desktop\Master\Git\DEP_Compare16s\Private\GSEA and find the one starts with abx-treat
    selected = pd.DataFrame()
    for folder in os.listdir(os.path.join(private, "GSEA")):
        if folder.startswith(f"{abx}{treat}"):
            # read the csv file that starts with gsea_report_for_1
            for file in os.listdir(os.path.join(private, "GSEA", folder)):
                if file.startswith("gsea_report_for") and file.endswith(".tsv"):
                    results = pd.read_csv(os.path.join(private, "GSEA", folder, file), sep="\t")
                    # keep only rows where FDR q-val < 0.05
                    results = results[results['FDR q-val'] < 0.05]
                    results['GO term'] = results['NAME'].apply(lambda x: map_term_to_go_id(x, go_dict))
                    addition = "_enh" if "_1_" in file else "_sup"
                    results['GO term'] = results['GO term'] + addition
                    results['enhanced?'] = "_1_" in file
                    results = results.rename(columns={"SIZE": "size"})
                    if selected.empty:
                        selected = results
                    else:
                        selected = pd.concat([selected, results], ignore_index=True)
    if selected.empty:
        print(f"{abx} {treat} is empty")
        return pd.DataFrame(columns=['GO term', 'size', 'enhanced?'])
    # drop rows where GO term is None, and report the number of rows dropped
    dropped = selected[selected['GO term'].isnull()]
    selected = selected.dropna(subset=['GO term'])
    print(f"{abx} {treat} dropped {dropped.shape[0]} rows, {selected.shape[0]} rows left")
    return selected


def create_go_term_dict(go_dag):
    # Create a dictionary to map formatted term names to GO IDs
    go_term_dict = {
        go_term.name.lower(): go_term.id
        for go_term in go_dag.values()
    }
    return go_term_dict


def map_term_to_go_id(term, go_term_dict):
    # Remove prefix and format the term
    formatted_term = term.replace("GOBP_", "").replace("_", " ").lower()

    return go_term_dict.get(formatted_term, None)


def get_categories(categories_size, go, selected, enhanced, p=False, regular=True):
    # enhanced = "True" if enhanced else "False"
    category = []
    # iterate over rows in selected
    for index, row in selected.iterrows():
        if (regular and row['enhanced?'] != str(enhanced)) or (not regular and row['enhanced?'] != enhanced):
            # if row['enhanced?'] != enhanced:
            continue
        if p is not False and row['GO term'] not in p:
            continue
        term = row['GO term'].split("_")[0]
        number = int(row['size'])
        # add number times of ancestor to category
        if term not in go:
            continue
        ancestors = get_ancestor(go[term])
        for ancestor in ancestors:
            category.extend([ancestor.name for _ in range(number)])
    unique, counts = np.unique(np.array(category), return_counts=True)
    counts = counts.astype(np.double)
    for k in range(len(unique)):
        counts[k] /= categories_size[unique[k]]
    order = np.flip(counts.argsort())
    unique_ordered = unique[order]
    counts_ordered = counts[order]
    # category.sort()
    categories = {unique: counts for unique, counts in zip(unique_ordered, counts_ordered)}

    return unique, categories


def plot_bar(curr, colors, counts_dict, x):  # , alpha=1.0):
    if not len(counts_dict):
        return 0
    bottom = np.zeros(len(counts_dict))
    sorted_order = list(counts_dict.keys())
    sorted_order.sort()
    for k, col in enumerate(sorted_order):
        curr.bar(x, counts_dict[col], bottom=bottom,
                 color=colors.get(col, 'gray'))
        # color=colors.get(col, 'gray'), alpha=alpha)
        bottom += counts_dict[col]
    return bottom[0]


def pbs_zscore(vec: pd.Series, meta_data, condition, treatment):
    pbs_data = meta[(meta['Drug'] == 'PBS') & (meta_data[condition] == treatment)]
    pbs = (vec[pbs_data['ID']])  # .dropna()
    vec = (vec - pbs.mean()) / pbs.std()


def dimension_reduction(raw_data, meta_data, condition, log=False, pca=True, title=""):
    # raw_data = np.arcsinh(raw_data).apply(zscore, axis=1)
    if log:
        raw_data = np.log2(raw_data + 1)  # .apply(pbs_zscore, axis=1)
    data = raw_data.T
    # if 'C9' in raw_data.columns:
    #     data = data.drop('C9')  # .apply(zscore)
    data = data.loc[~data.index.str.startswith('S')]

    # # divide each column by its median
    # medians = data.median(axis=0)
    # data = data.div(medians)

    meta_data = meta_data.set_index('ID')
    meta_data = meta_data.loc[~meta_data.index.str.startswith('S')]
    # change type of condition to category
    data[condition] = meta_data.loc[data.index][condition].astype(str)
    data['drug'] = meta_data.loc[data.index]['Drug'].astype(str)
    if pca:
        data = plot_pca(data, condition)
    for perplexity in np.linspace(5, min(data.shape[0], data.shape[1]) // 4, 3):
        for pca_components in [5, 15]:
            data = plot_tsne(data, condition, True, perplexity, pca_components, title)


def plot_pca(data, condition, title=''):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # plot a pca of raw data in 2 dimensions
    pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(data[data.columns[:-2]].values) # todo: should be z-scored? no removal of genes

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[data.columns[:-2]].values)
    pca_result = pca.fit_transform(data_scaled)  # random_permutation = np.random.permutation(data.shape[0])
    data['pca-one'] = pca_result[:, 0]
    data['pca-two'] = pca_result[:, 1]
    plt.figure(figsize=(16, 10))
    # scatter plot pcs-one and pca-two of data where style is condition and hue is drug
    sns.scatterplot(x="pca-one", y="pca-two", hue="drug", style=condition, legend="full", alpha=0.7, data=data, s=70)
    plt.xlabel('Principal Component 1 ({}%)'.format(round(100 * pca.explained_variance_ratio_[0], 2)))
    plt.ylabel('Principal Component 2 ({}%)'.format(round(100 * pca.explained_variance_ratio_[1], 2)))

    # palette=sns.color_palette("hls", n_colors=len(antibiotics)), data=data)
    # plt.xlim(-200, 100)
    # plt.ylim(-500, 750)
    # drop the added columns
    data = data.drop(['pca-one', 'pca-two'], axis=1)
    # add to plot which part of the variance is explained by each component
    plt.title(f"PCA of raw data, {title}")
    plt.savefig(private + "dimension reduction/" + f"PCA of raw data {title}.png")
    plt.show()
    plt.close()
    return data


def plot_pca_pairs(data, condition, title='', n_components=2):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Plot PCA pairs of raw data in specified dimensions
    pca = PCA(n_components=n_components)

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[data.columns[:-2]].values)
    pca_result = pca.fit_transform(data_scaled)

    # Create column names based on the number of components
    pca_columns = [f'pca-{i}' for i in range(1, n_components + 1)]

    # Add PCA components to the dataframe
    for i in range(n_components):
        data[pca_columns[i]] = pca_result[:, i]

    # Create pairwise scatter plots for all combinations of PCA components
    plt.figure(figsize=(25, 15))
    for i in range(n_components):
        for j in range(i + 1, n_components):
            plt.subplot(n_components - 1, n_components - 1, i * (n_components - 1) + j)
            sns.scatterplot(x=pca_columns[i], y=pca_columns[j], hue="drug", style=condition, legend="full", alpha=0.7,
                            data=data, s=50)
            plt.xlabel(f'Principal Component {i + 1} ({round(100 * pca.explained_variance_ratio_[i], 2)}%)')
            plt.ylabel(f'Principal Component {j + 1} ({round(100 * pca.explained_variance_ratio_[j], 2)}%)')

    # Additional customization if needed
    # ...

    # Add title and save the plot
    plt.suptitle(f"PCA Pairs of raw data, {title}")
    plt.subplots_adjust(top=0.9)
    plt.savefig(private + "dimension reduction/" + f"PCA Pairs of raw data {title}.png")
    plt.show()
    plt.close()

    # Drop the added columns
    data = data.drop(pca_columns, axis=1)

    return data


def get_genes_from_df(df, go_cluster):
    # if len(df[(df["GO term"] == go_cluster)]["genes"].values) == 0:
    #     print(f"{go_cluster} is missing, filled by zeros")
    #     line = np.concatenate([np.array([go_cluster]), np.zeros(data.shape[1])])
    #     temp = temp.append(pd.Series(line), ignore_index=True)
    #     return temp
    return [gene.strip("{").strip("}").strip(' ').strip("\"").strip("\'") for gene in
            df[(df["GO term"] == go_cluster)]["genes"].values[0].split(",")]


def get_median_from_df(data, go_cluster, mice, temp, genes):
    relevant_genes = [gene for gene in genes if gene in data.index]
    median = np.median(data[mice].loc[relevant_genes], axis=0)
    # normalized_median = (median - np.median(pbs_data))  # / np.std(pbs_data) todo: make sure doesn't happen elsewhere
    line = np.concatenate([np.array([go_cluster]), median])
    temp = temp.append(pd.Series(line), ignore_index=True)
    return temp


def prepare_data(anti, condition, exp_type, meta_data, treat):
    temp = pd.DataFrame()
    df = pd.read_csv(f'./Private/clusters_properties/{exp_type}/top_correlated_GO_terms_{anti}_{treat}.tsv',
                     sep="\t")
    abx = meta_data[(meta_data['Drug'] == anti) & (meta_data[condition] == treat)]
    pbs = meta_data[(meta_data['Drug'] == 'PBS') & (meta_data[condition] == treat)]
    mice = pd.concat((abx['ID'], pbs['ID']))
    return abx, df, mice, pbs, temp


def plot_tsne(data, condition, pca=True, perplexity=7, pca_components=50, title=''):
    """
    hopefully: control are clustered together
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=perplexity,
                method='exact')  # todo: test for other perplexities and play with other parameters
    if pca:
        # reduce the number of dimensions before tSNE to "pca_components"
        pca_50 = PCA(n_components=pca_components)
        pca_result_50 = pca_50.fit_transform(data[data.columns[:-2]].values)
        tsne_results = tsne.fit_transform(pca_result_50)
    else:
        tsne_results = tsne.fit_transform(data[data.columns[:-2]].values)
        # add to plot which part of the variance is explained by each component tsne-2d-one and tsne-2d-two
        plt.text(0.5, 0.5, f"variance explained by each component: {tsne_results.explained_variance_ratio_}")
    data['tsne-2d-one'] = tsne_results[:, 0]
    data['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="drug", style=condition, legend="full", alpha=0.7,
                    data=data, s=150)
    # plt.xlim(-220, 220)
    # plt.ylim(-220, 220)
    plt.title(f"tSNE {f'with pca to {pca_components}' if pca else ''}, perplexity={perplexity} ")
    plt.savefig(
        private + "dimension reduction/tsne/" + f"PCA of raw data {title} perplexity={perplexity}, pca to {pca_components}.png")
    plt.show()
    data = data.drop(['tsne-2d-one', 'tsne-2d-two'], axis=1)

    return data


def get_selected(df):
    # selected = df[(df['MWU'] <= 0.05) & (df['size'] >= 2) & (
    #         df['better than parent'] != False) & (df['better than random correlation'] == 1)]
    selected = df[(df['MWU'] <= 0.05) & (df['size'] >= 2) & (
            df['p-value distance'] <= 0.05)]
    # df['better than parent'] is not False) & (df['better than random correlation'] is not False)]
    # iterate over the rows and get each cluster genes
    selected_genes = set()
    for _, row in selected.iterrows():
        # get the genes in the cluster
        genes = row["genes"].split(",")
        genes = set(gene.strip("{").strip("}").strip(' ').strip("\"").strip("\'") for gene in genes)
        selected_genes = selected_genes.union(genes)
    return list(selected_genes)


def intersection(antibiotics, conditions, exp_type):
    from venn import venn
    for abx in antibiotics:
        genes_times = {}
        for condition in conditions:
            df = pd.read_csv(os.path.join(path, exp_type, f"top_correlated_GO_terms_{abx}_{condition}.tsv"), sep="\t")
            genes_times[f"Time:{condition}"] = set(get_selected(df))
        venn(genes_times)
        plt.title(f"{exp_type.upper()} intersections")
        plt.savefig(os.path.join(private, "analysis", f"{exp_type} intersection.png"))
        plt.show()

    # high_spfs = {}
    # low_spfs = {}
    # for conditions in [5, 11, 17, 23]:
    #     high_spfs[conditions] = genes_times[f'Time:{conditions}'].intersection(high_spf[conditions])
    #     low_spfs[conditions] = genes_times[f'Time:{conditions}'].intersection(low_spf[conditions])
    #     assert len(high_spfs[conditions]) + len(low_spfs[conditions]) == len(genes_times[f'Time:{conditions}'])


# def get_all_go_raw(condition, data, exp_type, meta_data):
#     all = {}
#     for anti in antibiotics:
#         for treat in treatments:
#             df = pd.read_csv(f'./Private/clusters_properties/{exp_type}/top_correlated_GO_terms_{anti}_{treat}.tsv',
#                              sep="\t")
#             clusters = df[(df['MWU'] <= 0.05) & (df["better than random correlation"] == True) &
#                           (df["better than parent"] != False)]
#             print(exp_type, anti, treat, clusters.shape)
#             all[f"{condition}:{treat}"] = set(clusters["GO term"].values)
#     for anti in antibiotics:
#         for treat in treatments:
#             abx, df, mice, pbs, temp = prepare_data(anti, condition, exp_type, meta_data, treat)
#             for go_cluster in all[f"{condition}:{treat}"]:
#                 genes = get_genes_from_df(df, go_cluster)
#                 temp = get_median_from_df(data, go_cluster, mice, temp, genes)
#             # if not temp.empty:
#             #     temp = temp.set_index(0)
#             #     clustering = sns.clustermap(data=temp.astype(np.float), row_cluster=True, col_cluster=False)
#             #     order = clustering.dendrogram_row.reordered_ind
#             #     plt.close()
#             #     all[f"{condition}:{treat}"] = temp.iloc[order].index
#     all_listed = np.array([])
#     for treat in treatments:
#         all_listed = np.concatenate([all_listed, np.array(list(all[f"{condition}:{treat}"]))])
#     genes_df = pd.DataFrame()
#     for anti in antibiotics:
#         for treat in treatments:
#             abx, df, mice, pbs, temp = prepare_data(anti, condition, exp_type, meta_data, treat)
#             for go_cluster in all_listed:
#                 genes = get_genes_from_df(df, go_cluster)
#                 temp = get_median_from_df(data, go_cluster, mice, temp, genes)
#             temp = temp.set_index(0)
#             # temp.columns = mice
#             genes_df = pd.concat([genes_df, temp], axis=1)
#     return genes_df.astype('f').fillna(0)


def plot_ven5(data, title, dir):
    from venn import venn
    venn(data)
    plt.title(f"Venn Diagram for all antibiotics {title}")
    plt.savefig(private + f"analysis/{dir}/Venn5{title}.png", bbox_inches='tight')
    plt.show()
    plt.show()


def get_unique_random_genes(index, data, times=100_000):
    mix_unique = np.zeros(times)
    for i in range(times):
        mix_unique[i] = len(set(np.random.choice(index, len(data['Mix']), replace=False)) -
                            set.union(*[set(np.random.choice(index, len(data[abx2]), replace=False))
                                        for abx2 in antibiotics if abx2 != 'Mix']))
    return np.mean(mix_unique), np.std(mix_unique)


def ven_diagrams_plot(genes, param):
    from matplotlib_venn import venn3

    intersection_list = ""
    mix_list = ""
    # plot 3 van5 diagrams: one for each treatment
    for treat in treatments:
        # # create a dictionary of the number of significant genes for each antibiotic
        data = {abx: genes[abx][treat] for abx in antibiotics}
        # print the intersection of the significant genes
        sep_list = '\n'.join(set.intersection(*data.values()))
        temp = f"intersection of significant genes for {treat}: {len(set.intersection(*data.values()))}\n" \
               f"{sep_list}\n"
        print(temp)
        intersection_list += temp

        # print the number of mutual significant genes for each pair of antibiotic+mix and have no overlap with any antibiotic
        for abx in antibiotics:
            if abx == "Mix":
                continue
            intersection_genes = set.intersection(data[abx], data['Mix']) - set.union(
                *[data[abx2] for abx2 in antibiotics if ((abx2 != abx) and (abx2 != 'Mix'))])
            temp = (f"intersection of significant genes for {treat} {abx} and Mix (only): "
                    f"{len(intersection_genes)}\n")
            print(temp)
            mix_list += temp
            # save intersection_genes to csv
            with open(private + f"analysis/{param}/intersection_genes_{treat}_{abx}_Mix.csv", 'w') as file:
                file.write(",".join(intersection_genes))
        unique_genes = set(data['Mix']) - set.union(*[data[abx2] for abx2 in antibiotics if abx2 != 'Mix'])
        unique_random = get_unique_random_genes(raw.index, data)
        mix_list += f"Mix unique terms for {treat}: {len(unique_genes)}, {100 * len(unique_genes) / len(data['Mix'])}%\n"
        mix_list += (fr"vs. {unique_random[0]} $\pm$ {unique_random[1]} unique size for random groups, meaning "
                     f"{(len(unique_genes) - unique_random[0]) / unique_random[1]} SDs\n")
        mix_list += f"{unique_genes}\n"
        with open(private + f"analysis/{param}/Mix_unique_genes_{treat}.csv", 'w') as file:
            file.write(",".join(unique_genes))
        # # create a venn diagram
        # # plot_ven5(data, treat, param)
        # venn(data)
        # plt.title(f"Venn Diagram for {treat}")
        # plt.savefig(private + f"analysis/{param}/Venn5{treat}.png", bbox_inches='tight')
        # plt.show()
        # venn3 with Amp, Mix and each abx (with proportional areas)
        strings_to_remove = ['Amp', 'Mix']
        mask = np.isin(antibiotics, strings_to_remove, invert=True)
        antibiotic_partial = antibiotics[mask]
        for abx in antibiotic_partial:
            curr = [genes[anti][treat] for anti in strings_to_remove + [abx]]
            venn3(curr, set_labels=strings_to_remove + [abx])
            plt.title(f"Venn Diagram for {treat}; {', '.join(strings_to_remove + [abx])}")
            plt.savefig(private + f"analysis/{param}/ven/Venn3_{treat}_{abx}.png", bbox_inches='tight')
            # plt.show()
            plt.close()
            random_intersection(curr, strings_to_remove + [abx], raw.index, param, treat, abx)
    # save the intersection list
    with open(private + f'analysis/{param}/intersection_list.txt', 'w') as file:
        file.write(intersection_list)
    with open(private + f'analysis/{param}/mix_list.txt', 'w') as file:
        file.write(mix_list)


def plot_significant_genes_number(meta, raw, antibiotics, treatments, param, condition="Treatment"):
    import pickle
    # import matplotlib
    # matplotlib.use('Agg')

    # import venn
    threshold = 0.05  # todo: change all to 1%?
    # if the file private + f"analysis/{param}/statistics_genes.csv" doesn't exist, create it
    # if True:
    if not os.path.exists(os.path.join(private, f"analysis/{param}/statistics_genes.csv")):
        all_stats = pd.DataFrame()
        all_stats.index = raw.index
        from scipy.stats import ttest_ind

        genes_sum = {}
        genes = {}
        fold_change = {}
        for abx in antibiotics:
            genes_sum[abx] = {}
            fold_change[abx] = {}
            genes[abx] = {}
            for treat in treatments:
                abx_data = meta[(meta['Drug'] == abx) & (meta[condition] == treat)]['ID'].values
                pbs_data = meta[(meta['Drug'] == 'PBS') & (meta[condition] == treat)]['ID'].values
                # temp = raw.loc[np.concatenate(abx_data, pbs_data)]
                ttest_pvalues = raw.apply(lambda row: ttest_ind(row[abx_data], row[pbs_data])[1], axis=1)
                # Create a new pandas Series for the absolute t-test p-values
                abs_ttest_pvalues = ttest_pvalues.abs()
                # Count the number of significant values (p-value < threshold)
                genes_sum[abx][treat] = (abs_ttest_pvalues < threshold).sum()
                # save a set of the significant genes in genes
                genes[abx][treat] = set(raw.index[abs_ttest_pvalues < threshold])
                sub = raw.loc[abs_ttest_pvalues < threshold]
                # sub = raw

                # Calculate the fold changes
                fold_changes = sub.apply(lambda row: np.log2(np.median(row[abx_data]) / np.median(row[pbs_data])),
                                         axis=1)
                # calc the mean of the fold changes
                fold_change[abx][treat] = np.nanmean(np.abs(fold_changes))
                # add the p-values and fold changes to the all_stats df
                all_stats = pd.concat([all_stats, pd.DataFrame(
                    {f"p-value_{abx}_{treat}": ttest_pvalues, f"fold_change_{abx}_{treat}": fold_changes},
                    index=raw.index)], axis=1)

        # Create a DataFrame from the dictionary
        df_sum = pd.DataFrame(genes_sum)
        df_fold = pd.DataFrame(fold_change)
        # create new df: one col is abx, one is treat, one is number of significant genes and one is fold change
        df = pd.DataFrame(columns=["Antibiotic", "Treatment", "#significant_genes", "log_fold_change"])
        for abx in antibiotics:
            for treat in treatments:
                # df = df.append(
                #     {"Antibiotic": abx, "Treatment": treat, "#significant_genes": df_sum[abx][treat],
                #      "log_fold_change": df_fold[abx][treat]}, ignore_index=True)
                new_row = pd.DataFrame(
                    {"Antibiotic": abx, "Treatment": treat, "#significant_genes": df_sum[abx][treat],
                     "log_fold_change": df_fold[abx][treat]}, index=[0])
                df = pd.concat([df, new_row], ignore_index=True)
        # save df
        df.to_csv(private + f"/analysis/{param}/statistics_genes.csv", index=False)
        with open(private + f'/analysis/{param}/statistics_genes.pkl', 'wb') as file:
            pickle.dump(genes, file)
        all_stats.to_csv(private + f"/analysis/{param}/all_stats.csv", index=True)
    else:
        df = pd.read_csv(os.path.join(private, "analysis", param, "statistics_genes.csv"))
        with open(os.path.join(private, f'analysis', param, 'statistics_genes.pkl'), 'rb') as file:
            genes = pickle.load(file)

    # ven_diagrams_plot(genes, param)

    # rename the columns and rows
    df = df.rename_axis("Treatment", axis=1)
    df = df.rename_axis("Antibiotic", axis=0)

    import io
    from PIL import Image
    # plot heatmap of the number of significant genes
    plt.figure(figsize=(1.5, 2.5))
    # sns.heatmap(df_sum, cmap="vlag")
    # reverse the order of the treatments
    # sort by treatment
    df = df.sort_values(by='Treatment', ascending=False)
    df = df.loc[df.index[::-1]]
    # todo: check why fold change is too small
    sns.scatterplot(
        x='Treatment',
        y='Antibiotic',
        data=df,
        size='log_fold_change',  # Size mapped to size_value column
        sizes=(10, 250),  # Set the range of dot sizes
        hue='#significant_genes',  # Color mapped to color_value column
        palette='vlag',  # Choose a color palette
        edgecolor='w',  # White edges for better visibility
        linewidth=0.5,  # Edge linewidth
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Genes statistics for p-value<{threshold}")
    # plt.title(f"Genes statistics\nsize is number of significant genes, color is fold change")
    plt.ylabel("Treatment")
    plt.xlabel("Antibiotic")
    # Set x-axis and y-axis limits
    plt.xlim(-.5, len(df['Treatment'].unique()) - .5)  # Adjust the limits as needed
    plt.ylim(-.5, len(df['Antibiotic'].unique()) - .5)  # Adjust the limits as needed

    plt.savefig(os.path.join(private, "analysis", param, "genes stats.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(private, "analysis", param, "genes stats.pdf"), bbox_inches='tight')
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Use Pillow to save as TIFF
    with Image.open(buf) as img:
        img.save(os.path.join(private, "analysis", param, "genes stats.tiff"), format='TIFF')
    plt.show()
    return


def random_intersection(df, labels, genes, param, treat, abx, repeat=100_000):
    """Randomly selects groups of size as in df and computes the different intersections, repeat times
    the results are plotted in a venn diagram, where the value is the mean but the std is also reported"""
    from matplotlib_venn import venn3
    size_amp, size_mix, size_abx = len(df[0]), len(df[1]), len(df[2])
    amp_mix_only = np.zeros(repeat)
    abx_mix_only = np.zeros(repeat)
    abx_amp_only = np.zeros(repeat)
    abx_amp_mix = np.zeros(repeat)
    for i in range(repeat):
        rand_amp = set(np.random.choice(genes, size_amp, replace=False))
        rand_mix = set(np.random.choice(genes, size_mix, replace=False))
        rand_abx = set(np.random.choice(genes, size_abx, replace=False))
        amp_mix_only[i] = len(rand_mix.intersection(rand_amp) - rand_abx)
        abx_mix_only[i] = len(rand_mix.intersection(rand_abx) - rand_amp)
        abx_amp_only[i] = len(rand_abx.intersection(rand_amp) - rand_mix)
        abx_amp_mix[i] = len(rand_abx.intersection(rand_amp).intersection(rand_mix))
    # plot the venn diagram of the mean and std
    plt.figure(figsize=(5, 5))
    means = [size_amp - amp_mix_only.mean() - abx_amp_only.mean() - abx_amp_mix.mean(),
             size_mix - amp_mix_only.mean() - abx_mix_only.mean() - abx_amp_mix.mean(),
             np.mean(amp_mix_only),
             size_abx - abx_mix_only.mean() - abx_amp_only.mean() - abx_amp_mix.mean(),
             np.mean(abx_amp_only), np.mean(abx_mix_only),
             np.mean(abx_amp_mix)]
    means = [round(mean, 1) for mean in means]
    venn3(means, set_labels=labels)
    plt.title(f"Venn Diagram of average random size for {treat}; {', '.join(labels)}")
    plt.savefig(private + f"analysis/{param}/ven/Venn3_random_mean_{treat}_{abx}.png", bbox_inches='tight')
    plt.show()
    stds = [np.sqrt(amp_mix_only.std() ** 2 + abx_amp_only.std() ** 2 + abx_amp_mix.std() ** 2),
            np.sqrt(amp_mix_only.std() ** 2 + abx_mix_only.std() ** 2 + abx_amp_mix.std() ** 2),
            np.std(amp_mix_only),
            np.sqrt(abx_mix_only.std() ** 2 + abx_amp_only.std() ** 2 + abx_amp_mix.std() ** 2),
            np.std(abx_amp_only), np.std(abx_mix_only),
            np.std(abx_amp_mix)]
    stds = [round(std, 1) for std in stds]
    venn3(stds, set_labels=labels)
    plt.title(f"Venn Diagram of std random size for {treat}; {', '.join(labels)}")
    plt.savefig(private + f"analysis/{param}/ven/Venn3_random_std_{treat}_{abx}.png", bbox_inches='tight')
    plt.show()


def plot_kde(x, y, jitter=0.01, point_size=20):
    from scipy.stats import gaussian_kde, pearsonr
    from matplotlib.colors import LogNorm

    # Add minor jitter to the data points
    x_jittered = x + np.random.normal(scale=jitter, size=x.size)
    y_jittered = y + np.random.normal(scale=jitter, size=y.size)

    # Calculate the point density
    xy = np.vstack([x_jittered, y_jittered])
    try:
        z = gaussian_kde(xy)(xy)
    except:
        plot_kde(x, y, jitter=0.01)
        return

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x_jittered, y_jittered, z = x_jittered[idx], y_jittered[idx], z[idx]

    # Compute correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(x, y)
    # slope, intercept = np.polyfit(x, y, 1)
    slope = np.sum(x * y) / np.sum(x ** 2)
    correlation_text = f'Correlation: {correlation_coefficient:.2f}\nP-value: {p_value:.2e}\nSlope: {slope:.2f}'

    fig, ax = plt.subplots()
    scatter = ax.scatter(x_jittered, y_jittered, c=z, s=point_size, cmap='viridis', norm=LogNorm())
    plt.colorbar(scatter, ax=ax, label='Density (log scale)')

    # Add correlation and p-value text to the plot
    ax.text(0.05, 0.95, correlation_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.6))


y_limits = {
    "\"log(distance)\"": (-1.2, 0),
    "treat-test p-value": (-0.1, 2.5),
    "relative size": (0.1, 1.1),
    "fold change": (-2, 3),
    "mean variance between samples": (-0.1, 3),
}
x_limits = {
    "\"log(distance)\"": (-1.2, 0),
    "treat-test p-value": (-0.1, 2.5),
    "relative size": (0.1, 1.1),
    "fold change": (-1, 3),
    "mean variance between samples": (-0.1, 3),
}


def clusters_compare_mix(raw, meta, antibiotics, treatments, param):
    # TODO: same scale for all column and square plot
    # treat = "PO"
    # abx = "Amp"
    mix = "Mix"
    no_mix = [abx for abx in antibiotics if abx != mix]
    for treat in treatments:
        for to_mix in [True, False]:
            title = f"selected by {'mix' if to_mix else 'other abx'} threshold"
            compare_mix_single(mix, no_mix, param, treat, "\"log(distance)\"",
                               f"log(distance) of clusters in abx_{treat}", f"log(distance) of clusters in mix_{treat}",
                               f"log_distance_{treat}{'_to_mix' if to_mix else ''}", to_mix, title, jitter=0)
            compare_mix_single(mix, no_mix, param, treat, "treat-test p-value",
                               f"treat-test -log10(p-value) of clusters in abx_{treat}",
                               f"treat-test -log10(p-value) of clusters in mix_{treat}",
                               f"p-value_{treat}{'_to_mix' if to_mix else ''}", to_mix, title, log=True, jitter=0)
            compare_mix_single(mix, no_mix, param, treat, "relative size",
                               f"relative size of clusters in abx_{treat}", f"relative size of clusters in mix_{treat}",
                               f"relative_size_{treat}{'_to_mix' if to_mix else ''}", to_mix, title)
            compare_mix_single(mix, no_mix, param, treat, "fold change",
                               f"log10(fold change) in abx_{treat}", f"log10(fold change) of clusters in mix_{treat}",
                               f"fold_change_{treat}{'_to_mix' if to_mix else ''}", to_mix, title, log=True,
                               minus=False)
            # compare_mix_single(mix, no_mix, param, treat, "fold change",
            #                    f"fold change in abx_{treat}", f"-log10(fold change) of clusters in mix_{treat}",
            #                    f"fold_change_{treat}{'_to_mix' if to_mix else ''}_no_log", to_mix, title)
            compare_mix_single(mix, no_mix, param, treat, "mean variance between samples",
                               f"mean variance between samples in abx_{treat}",
                               f"mean variance between samples of clusters in mix_{treat}",
                               f"variance_{treat}{'_to_mix' if to_mix else ''}", to_mix, title)
    # return compare_mix


def compare_mix_single(mix, no_mix, param, treat, col, xlabel, ylabel, title, to_mix, by, log=False, minus=True,
                       jitter=0.01):
    plt.figure(figsize=(10, 10))
    selected_log_distances = []
    compare_mix_log_distances = []
    for abx in no_mix:
        compare_mix, selected = get_selected_df_plot_mix(abx, mix, param, treat) if to_mix else get_selected_df_plot(
            abx, mix, param, treat)
        # verify same order of clusters in both DFs
        assert sorted(selected["GO term"].values) == sorted(compare_mix["GO term"].values)
        print(treat, col, len(selected["GO term"]))
        # plot log(distance) of the clusters in abx_treat and mix_treat in a scatter plot
        # plt.scatter(selected["\"log(distance)\""], compare_mix["\"log(distance)\""])
        # Append log(distance) values from both DataFrames
        if log:
            if minus:
                selected_log_distances.extend(-np.log10(selected[col]))
                compare_mix_log_distances.extend(-np.log10(compare_mix[col]))
            else:
                selected_log_distances.extend(np.log10(selected[col]))
                compare_mix_log_distances.extend(np.log10(compare_mix[col]))
        else:
            if "log" in col:
                selected_log_distances.extend(selected[col] / np.log2(10))
                compare_mix_log_distances.extend(compare_mix[col] / np.log2(10))
            else:
                selected_log_distances.extend(selected[col])
                compare_mix_log_distances.extend(compare_mix[col])
    df = pd.DataFrame({
        'Selected': selected_log_distances,
        'Compare Mix': compare_mix_log_distances
    })
    df_filtered = df.dropna(subset=['Selected', 'Compare Mix']).reset_index(drop=True)
    if len(df_filtered) != len(df):
        print(title, f"dropped {len(df) - len(df_filtered)} nans")
        df = df_filtered
    # sns.kdeplot(data=df, x='Selected', y='Compare Mix', cmap="YlGnBu", shade=True, cbar=True)
    plot_kde(x=df['Selected'], y=df['Compare Mix'], jitter=jitter)
    # Add the y = x line
    # min_val = min(df['Selected'].min(), df['Compare Mix'].min())
    # max_val = max(df['Selected'].max(), df['Compare Mix'].max())
    min_val = df['Selected'].min()
    max_val = df['Selected'].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
    plt.legend()
    plt.xlim(x_limits[col])
    plt.ylim(y_limits[col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Comparison of clusters in abx_{treat} and {mix}_{treat},\n{by}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.axis('square')
    plt.savefig(f"./Private/met_comp/{title}.png")
    # plt.show()
    plt.close()


def get_selected_df_plot(abx, mix, param, treat):
    # read the clusters data for abx_treat and mix_treat
    df = pd.read_csv(path + f"\\{param}\\top_correlated_GO_terms_{abx}_{treat}.tsv", sep="\t")
    df_mix = pd.read_csv(path + f"\\{param}\\top_correlated_GO_terms_{mix}_{treat}.tsv", sep="\t")
    # get the selected clusters for abx_treat
    selected = df[(df['treat-test p-value'] <= 0.05) & (df['size'] >= 2) & (
            df['p-value distance'] <= 0.05)]
    # df['better than parent'] is not False) & (df['better than random'] is not False)]
    # get the selected clusters for mix_treat
    compare_mix = df_mix[df_mix["GO term"].isin(selected["GO term"])]
    # keep in selected only the clusters that are in compare_mix
    selected = selected[selected["GO term"].isin(compare_mix["GO term"])]
    return compare_mix, selected


def get_selected_df_plot_mix(abx, mix, param, treat):
    # read the clusters data for abx_treat and mix_treat
    df_other = pd.read_csv(path + f"\\{param}\\top_correlated_GO_terms_{abx}_{treat}.tsv", sep="\t")
    df = pd.read_csv(path + f"\\{param}\\top_correlated_GO_terms_{mix}_{treat}.tsv", sep="\t")
    # get the selected clusters for abx_treat
    selected = df[(df['treat-test p-value'] <= 0.05) & (df['size'] >= 2) & (
            df['p-value distance'] <= 0.05)]
    # df['better than parent'] is not False) & (df['better than random'] is not False)]
    # get the selected clusters for mix_treat
    compare_other = df_other[df_other["GO term"].isin(selected["GO term"])]
    # keep in selected only the clusters that are in compare_mix
    selected = selected[selected["GO term"].isin(compare_other["GO term"])]
    return selected, compare_other


def compare_significance_go(param):
    go_number = pd.read_csv(os.path.join(private, "analysis", param, "diff_abxRASflowRASflow GO_number.csv"),
                            index_col=0)
    significant_genes = pd.read_csv(private + f"/analysis/{param}/statistics_genes.csv")
    significant_genes["condition"] = significant_genes["Antibiotic"] + "_" + significant_genes["Treatment"]
    significant_genes["go_number"] = 0
    significant_genes["true_positive"] = 0
    for treat in treatments:
        confusion_matrix = pd.read_csv(f"./Private/YasminRandomForest/confusion_matrix_{treat}.csv", index_col=0)
        for abx in antibiotics:
            significant_genes.loc[significant_genes["condition"] == f"{abx}_{treat}", "go_number"] = go_number.loc[
                abx, treat]
            significant_genes.loc[significant_genes["condition"] == f"{abx}_{treat}", "true_positive"] = \
                confusion_matrix.loc[f"{abx}_{treat}", f"{abx}_{treat}"]

    plot_multiabx_scatter(param, significant_genes, x="#significant_genes", y="go_number")
    plot_multiabx_scatter(param, significant_genes, x="#significant_genes", y="true_positive")
    plot_multiabx_scatter(param, significant_genes, x="true_positive", y="go_number")


def plot_multiabx_scatter(param, significant_genes, x, y):
    # plot #significant_genes vs go_number
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=x, y=y, hue="Antibiotic", style="Treatment", data=significant_genes, s=30)
    plt.xlabel(x)
    plt.ylabel(y)

    # log scale and save
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(private + f"/analysis/{param}/{x}_vs_{y}_log.png")

    # remove Van IP
    significant_genes = significant_genes[
        (significant_genes["Antibiotic"] != "Van") & (significant_genes["Treatment"] != "IP")]

    # compute the linear regression
    slope, intercept, r_value, p_value, std_err = linregress(significant_genes[x], significant_genes[y])
    # sort by x
    significant_genes = significant_genes.sort_values(by=x)
    plt.plot(significant_genes[x], slope * significant_genes[x] + intercept, color='black')
    # plt.text(0.01, 0.98,
    #          f"r={round(r_value, 2)}, p={round(p_value, 2)}, slope={round(slope, 2)}, intercept={round(intercept, 2)}\n(no Van IP)",
    #          horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
    plt.title(f"r={round(r_value, 2)}, p={round(p_value, 2)}, slope={round(slope, 2)}, intercept={round(intercept, 2)}"
              f"\n(Regression done without Van IP)")
    plt.yscale("linear")
    plt.xscale("linear")
    # plt.title("Number of significant genes vs number of significant GO terms")
    plt.savefig(private + f"/analysis/{param}/{x}_vs_{y}.png")
    # plt.show()
    plt.close()


def effective_number_genes(data, metadata):
    effective_number_of_genes = pd.DataFrame(index=np.append(antibiotics, "PBS"), columns=treatments, data=0)
    hill_0 = pd.DataFrame(index=np.append(antibiotics, "PBS"), columns=treatments, data=0)
    hill_inf = pd.DataFrame(index=np.append(antibiotics, "PBS"), columns=treatments, data=0)
    # normalize each column to sum to 1
    data = data.div(data.sum(axis=0), axis=1)
    for treat in treatments:
        pbs = metadata[(metadata["Drug"] == "PBS") & (metadata["Treatment"] == treat)]["ID"]
        genes = data[pbs]
        effective_number_of_genes.loc["PBS", treat] = np.exp(np.mean(-np.sum(genes * np.log(genes), axis=0)))
        hill_0.loc["PBS", treat] = np.mean(np.sum(genes > 0, axis=0))
        hill_inf.loc["PBS", treat] = np.mean(genes.max(axis=0))
        for abx in antibiotics:
            samples = meta[(metadata["Drug"] == abx) & (metadata["Treatment"] == treat)]["ID"]
            # compute Hill number 1 for all samples in this condition
            genes = data[samples]
            effective_number_of_genes.loc[abx, treat] = np.exp(np.mean(-np.sum(genes * np.log(genes), axis=0)))
            hill_0.loc[abx, treat] = np.mean(np.sum(genes > 0, axis=0))
            hill_inf.loc[abx, treat] = np.mean(genes.max(axis=0))
    plot_heatmap_multiabx(effective_number_of_genes, "effective")
    plot_heatmap_multiabx(hill_0, "richness")
    plot_heatmap_multiabx(hill_inf, "inf")


def plot_heatmap_multiabx(effective_number_of_genes, title):
    # plot heatmap of effective_number_of_genes
    plt.figure(figsize=(10, 10))
    sns.heatmap(effective_number_of_genes, cmap="Blues")
    plt.title(f"{title} number of genes")
    plt.savefig(private + f"/analysis/{title}_number_of_genes.png")
    # plt.show()
    plt.close()


def plot_selected_clusters(raw, meta, param):
    # a. Autophagy in the Van IP
    selected = get_selected_df("Van", "IP", param, True)
    autophagy = selected[selected['name'].str.contains('autophag', case=False, na=False)]

    van_abx = meta[(meta['Drug'] == "Van") & (meta['Treatment'] == "IP")]['ID'].to_list()
    van_pbs = meta[(meta['Drug'] == 'PBS') & (meta['Treatment'] == "IP")]['ID'].to_list()
    all_samples = van_pbs + van_abx
    plot_clusters_heatmap(all_samples, autophagy, raw, "VanIP", van_pbs)

    # b. Antiviral in the Neo IP, IV
    selected = get_selected_df("Neo", "IP", param, True)
    viral = selected[selected['name'].str.contains('vir', case=False, na=False)]

    neo_abx = meta[(meta['Drug'] == "Neo") & (meta['Treatment'] == "IP")]['ID'].to_list()
    neo_pbs = meta[(meta['Drug'] == 'PBS') & (meta['Treatment'] == "IP")]['ID'].to_list()
    all_samples = neo_pbs + neo_abx
    plot_clusters_heatmap(all_samples, viral, raw, "NeoIP", neo_pbs)

    selected = get_selected_df("Neo", "IV", param, True)
    viral = selected[selected['name'].str.contains('vir', case=False, na=False)]

    neo_abx = meta[(meta['Drug'] == "Neo") & (meta['Treatment'] == "IV")]['ID'].to_list()
    neo_pbs = meta[(meta['Drug'] == 'PBS') & (meta['Treatment'] == "IV")]['ID'].to_list()
    all_samples = neo_pbs + neo_abx
    plot_clusters_heatmap(all_samples, viral, raw, "NeoIV", neo_pbs)


def plot_clusters_heatmap(all_samples, df, raw, abx, normalization):
    import ast
    # pbs_mean = raw[normalization].mean(axis=1)
    # pbs_std = raw[normalization].std(axis=1)
    # # normalize the data by the mean and std of the pbs mice: subtract pbs_mean from every row and divide by std
    # normalized_data = raw.sub(pbs_mean, axis=0).div(pbs_std, axis=0)
    from scipy.stats import zscore
    normalized_data = raw.apply(zscore, axis=1)  # Change axis=0 for column-wise normalization

    for i, row in df.iterrows():
        ensmus_list = ast.literal_eval(row['genes'])
        genes_list = ast.literal_eval(row['gene names'])

        cluster = normalized_data[all_samples].loc[ensmus_list]
        # Create the heatmap
        # sns.heatmap(cluster, cmap="RdBu_r", yticklabels=False)
        # Set the y-axis ticks to gene names
        # plt.yticks(range(len(genes_list)), genes_list, rotation=0)
        limit = max(cluster.max().max(), -cluster.min().min())
        # Generate the clustermap to get the row order
        g = sns.clustermap(cluster,
                           cmap="RdBu_r",
                           yticklabels=genes_list,
                           xticklabels=True,
                           col_cluster=False,
                           row_cluster=True,
                           vmin=-limit, vmax=limit,
                           dendrogram_ratio=(0, 0.1),  # This hides the row dendrogram
                           colors_ratio=(0.03, 0.97),  # Adjust space for labels
                           cbar_pos=(0.02, 0.8, 0.05, 0.18))

        # Extract the row order from the clustermap
        row_order = g.dendrogram_row.reordered_ind

        # Reorder the rows of the cluster matrix and gene names based on the clustermap
        reordered_cluster = cluster.iloc[row_order, :]
        reordered_genes_list = [genes_list[i] for i in row_order]
        plt.close()

        # Create the heatmap with the reordered rows
        plt.figure(figsize=(12, len(ensmus_list) / 2))
        sns.heatmap(reordered_cluster, cmap="RdBu_r", yticklabels=reordered_genes_list, xticklabels=True, vmin=-limit,
                    vmax=limit)

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.title(f"{row['GO term']}\n{row['name']}")
        plt.tight_layout()  # Adjust the layout to prevent cutting off labels
        plt.savefig(f"./Private/selected_clusters/{abx}_{row['GO term'].replace(':', '_')}.png")
        plt.show()
        plt.close()


def add_significance_indicators(ax, x, observed_values, p_values, y_offset=0.005):
    """
    Add significance indicators (ns, *, **, ***) above bars based on comparison with random values.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to plot on
    x : array-like
        x-coordinates of bars
    observed_values : array-like
        The actual correlation values
    random_mean : float
        Mean of random correlations
    random_std : float
        Standard deviation of random correlations
    width : float
        Width of the bars
    y_offset : float, optional
        Vertical offset for placing the significance indicators
    """
    # from scipy import stats

    # Function to determine significance symbol
    def get_significance_symbol(p_value):
        if p_value > 0.05:
            return "ns"
        elif p_value <= 0.001:
            return "***"
        elif p_value <= 0.01:
            return "**"
        else:
            return "*"

    # # Calculate z-scores and p-values
    # z_scores = (observed_values - random_mean) / random_std
    # p_values = 1 - stats.norm.cdf(z_scores)  # One-tailed test: only checking if higher

    # Get current y-axis limits
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin

    # Add annotations
    for i, (x_pos, value, p_value) in enumerate(zip(x, observed_values, p_values)):
        symbol = get_significance_symbol(p_value)
        # Position text above positive bars and below negative bars
        if value >= 0:
            y_pos = value + y_range * y_offset
            va = 'bottom'
        else:
            y_pos = value - y_range * y_offset
            va = 'top'

        ax.text(x_pos, y_pos, symbol, ha='center', va=va, fontsize=8)


def plot_correlation_gsea(gsea, our):
    # Create a single figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Pearson vs. Spearman Correlations for Enhanced and Suppressed", fontsize=14)

    # Iterate over directions and plot correlations
    for j, direction in enumerate(["enhanced", "suppressed"]):
        df1 = gsea[j]
        df2 = our[j]

        # Dictionaries to store row and column correlations
        correlations = {"row": {"Pearson": [], "Spearman": []}, "column": {"Pearson": [], "Spearman": []}}
        random_correlations = {"row": {"Pearson": [], "Spearman": []}, "column": {"Pearson": [], "Spearman": []}}
        p_values = {"row": {"Pearson": [], "Spearman": []}, "column": {"Pearson": [], "Spearman": []}}

        # Calculate row-wise and column-wise correlations for both methods
        for method in ["pearson", "spearman"]:
            # Row-wise correlations
            row_correlations = [df1.iloc[i].corr(df2.iloc[i], method=method) for i in range(df1.shape[0])]
            correlations["row"][method.capitalize()] = row_correlations

            # Column-wise correlations
            col_correlations = [df1[col].corr(df2[col], method=method) for col in df1.columns]
            correlations["column"][method.capitalize()] = col_correlations

            # compare to random correlation
            shuffles = 1_000
            temp_row_random_correlations = np.zeros(shuffles)
            temp_col_random_correlations = np.zeros(shuffles)
            for i in range(shuffles):
                # Shuffle rows and compute the mean of row-wise correlations
                shuffled_df1 = df1.sample(frac=1, replace=False).reset_index(drop=True)
                shuffled_df2 = df2.sample(frac=1, replace=False).reset_index(drop=True)
                temp_row_random_correlations[i] = shuffled_df1.corrwith(shuffled_df2, method=method).mean()

                # Shuffle columns in the same way as rows
                shuffled_df1 = df1.T.sample(frac=1, replace=False).reset_index(drop=True)
                shuffled_df2 = df2.T.sample(frac=1, replace=False).reset_index(drop=True)
                temp_col_random_correlations[i] = shuffled_df1.corrwith(shuffled_df2, method=method).mean()

            random_correlations["row"][method.capitalize()] = [np.nanmean(temp_row_random_correlations),
                                                               np.nanstd(temp_row_random_correlations)]
            random_correlations["column"][method.capitalize()] = [np.nanmean(temp_col_random_correlations),
                                                                  np.nanstd(temp_col_random_correlations)]

            p_values["row"][method.capitalize()] = [
                ((np.sum(temp_row_random_correlations >= row) + 1) / (shuffles + 1)) for row in row_correlations]
            p_values["column"][method.capitalize()] = [
                ((np.sum(temp_col_random_correlations >= col) + 1) / (shuffles + 1)) for col in col_correlations]

            # # plot a histogram of the random correlations
            # plt.hist(temp_row_random_correlations, bins=30, alpha=0.5, label="Random rows", color='gray')
            # plt.hist(temp_col_random_correlations, bins=30, alpha=0.5, label="Random columns", color='black')
            # plt.axvline(np.nanmean(temp_row_random_correlations), color='gray', linestyle='--')
            # plt.axvline(np.nanmean(temp_col_random_correlations), color='black', linestyle='--')
            # plt.xlabel(f"{method.capitalize()} correlation")
            # plt.ylabel("Frequency")
            # plt.title(f"{direction.capitalize()} - Random {method.capitalize()} Correlations")
            # plt.legend()
            # plt.show()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Plot row-wise correlations
        ax = axes[0, j]
        x_labels = df1.index
        x = np.arange(len(x_labels))
        width = 0.35  # Bar width
        ax.bar(x - width / 2, correlations["row"]["Pearson"], width, label="Pearson", alpha=0.7)
        ax.bar(x + width / 2, correlations["row"]["Spearman"], width, label="Spearman", alpha=0.7)
        # add ns, *, **, etc. for each of the bars in the bar plot in comparison to the shuffles
        # Add significance indicators for Pearson correlations
        add_significance_indicators(
            ax, x - width / 2,
            correlations["row"]["Pearson"],
            p_values["row"]["Pearson"],
        )

        # Add significance indicators for Spearman correlations
        add_significance_indicators(
            ax, x + width / 2,
            correlations["row"]["Spearman"],
            p_values["row"]["Spearman"],
        )
        ax.set_title(f"{direction.capitalize()} - Row-wise Correlation\n "
                     f"Mean Pearson: {np.nanmean(correlations['row']['Pearson']):.2f}, "
                     f"Mean Spearman: {np.nanmean(correlations['row']['Spearman']):.2f}\n"
                     f"Random rows correlation: Pearson: {random_correlations['row']['Pearson'][0]:.2f}±"
                     f"{random_correlations['row']['Pearson'][1]:.2f}, "
                     f"Spearman: {random_correlations['row']['Spearman'][0]:.2f}±"
                     f"{random_correlations['row']['Spearman'][1]:.2f}")
        # add thin line for random correlations
        ax.axhline(random_correlations['row']['Pearson'][0], color=colors[0], linestyle='--')
        ax.axhline(random_correlations['row']['Pearson'][0] + random_correlations['row']['Pearson'][1], color=colors[0],
                   linestyle=':')
        ax.axhline(random_correlations['row']['Spearman'][0], color=colors[1], linestyle='--')
        ax.axhline(random_correlations['row']['Spearman'][0] + random_correlations['row']['Spearman'][1],
                   color=colors[1], linestyle=':')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1, 1.2)  # Force y-axis to range from -1 to 1
        ax.legend()

        # Plot column-wise correlations
        ax = axes[1, j]
        x_labels = df1.columns
        x = np.arange(len(x_labels))
        ax.bar(x - width / 2, correlations["column"]["Pearson"], width, label="Pearson", alpha=0.7)
        ax.bar(x + width / 2, correlations["column"]["Spearman"], width, label="Spearman", alpha=0.7)
        # Add significance indicators for Pearson correlations
        add_significance_indicators(
            ax, x - width / 2,
            correlations["column"]["Pearson"],
            p_values["column"]["Pearson"],
        )

        # Add significance indicators for Spearman correlations
        add_significance_indicators(
            ax, x + width / 2,
            correlations["column"]["Spearman"],
            p_values["column"]["Spearman"],
        )
        ax.set_title(f"{direction.capitalize()} - Column-wise Correlation\n "
                     f"Mean Pearson: {np.nanmean(correlations['column']['Pearson']):.2f}, "
                     f"Mean Spearman: {np.nanmean(correlations['column']['Spearman']):.2f}\n"
                     f"Random columns correlation: Pearson: {random_correlations['column']['Pearson'][0]:.2f}±"
                     f"{random_correlations['column']['Pearson'][1]:.2f}, "
                     f"Spearman: {random_correlations['column']['Spearman'][0]:.2f}±"
                     f"{random_correlations['column']['Spearman'][1]:.2f}")
        ax.axhline(random_correlations['column']['Pearson'][0], color=colors[0], linestyle='--')
        ax.axhline(random_correlations['column']['Pearson'][0] + random_correlations['column']['Pearson'][1],
                   color=colors[0], linestyle=':')
        ax.axhline(random_correlations['column']['Spearman'][0], color=colors[1], linestyle='--')
        ax.axhline(random_correlations['column']['Spearman'][0] + random_correlations['column']['Spearman'][1],
                   color=colors[1], linestyle=':')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1, 1.2)  # Force y-axis to range from -1 to 1
        ax.legend()

    # Adjust layout for better visualization
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(private + "/analysis/correlation_gsea_our.png")
    plt.show()


if __name__ == "__main__":
    # temp_path = "C:\\Users\\Yehonatan\\Desktop\\Master\\Git\\DEP_Compare16s\\Private\\clusters_properties\\diff_abx\\"
    # for treat in treatments:
    #     for abx in antibiotics:
    #         df = pd.read_csv(temp_path+f"top_correlated_GO_terms_{abx}_{treat}.tsv", sep="\treat")
    #         df["GO term"] = df["GO term"].apply(lambda x: x.split("_")[0])
    #         df["suf"] = np.where(df["enhanced?"].astype(str) != "False", "_enh", "_supp")
    #         df["GO term"] += df["suf"]
    #         df = df.drop(['suf'], axis=1)
    #         df.to_csv(temp_path+f"top_correlated_GO_terms_{abx}_{treat}.tsv", sep="\treat", index=False)

    # raw = pd.read_csv(raw_data_path).set_index("Gene Symbol").fillna(0)  # changed to imputed
    # # raw = normalize_raw_data(raw)
    # # save normalized df
    # # raw.to_csv(raw_data_path[:-4] + "_normalized.csv")
    # meta = pd.read_csv(meta_data_path)
    # # meta = pd.read_csv(meta_data_path, sep="\t")
    # # iterate over all .csv files in "C:\Users\Yehonatan\Desktop\Master\Git\DEP_Compare16s\Private\hist\png"
    # # plot_clusters()
    # all_data = pd.read_csv(all_path, sep="\t")

    # plot_h2ab1(all_data)
    # for run_type in ["_normalize_cols", "_normalize_cols_and_rows"]:
    # for run_type in ["_B2_DES", "_B2_RPKM", "_Hi_DES", "_Hi_RPKM"]:
    # for run_type in ["_old_controls"]:
    for run_type in ["RASflow"]:
        all_data = pd.read_csv(os.path.join(path, f"diff_abx{run_type}\\top_correlated_GO_terms.tsv"), sep="\t")
        genome, meta, partek, transcriptome = read_process_files(new=False)

        if run_type == "_old_controls":
            # data = pd.read_csv(
            #     "../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/New Partek_bell_all_Normalization_Normalized_counts1.csv")
            # metadata = pd.read_csv("../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/all-samples-noC9C10.csv")
            raw = partek
        # if run_type == "_old_controls":
        #     raw = pd.read_csv(
        #         data_path + r"\MultiAbx-16s\MultiAbx-RPKM-RNAseq-B6\New Partek_bell_all_Normalization_Normalized_counts1.csv").set_index(
        #         "Gene Symbol").fillna(0)
        #     meta = pd.read_csv(data_path + r"\MultiAbx-16s\MultiAbx-RPKM-RNAseq-B6\all-samples-noC9.tsv", sep="\t")
        elif "B2" in run_type or "Hi" in run_type:
            folder = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
            raw = pd.read_csv(folder + f"Partek_Lilach_All_Abx_Normalization_Normalized_counts{run_type}.csv")
            # split data columns names by -,_ and take only first part
            # data.columns = [col.split('-')[0].split('_')[0] for col in data.columns]
            meta = pd.read_excel(folder + f"metadata.xlsx")
            # if 'New/Old' == N, add N to the ID
            meta['ID'] = meta.apply(lambda row: row['ID'] + 'N' if row['New/Old'] == 'N' else row['ID'], axis=1)
            # for Drug, replace mix with Mix, ampicillin with Amp, Control with PBS METRO with Met, NEO with Neo and
            # VANCO with Van
            meta['Drug'] = meta.apply(lambda row: row['Drug'].replace('mix', 'Mix').replace('ampicillin', 'Amp')
                                      .replace('Control ', 'PBS').replace('METRO', 'Met').replace('NEO', 'Neo')
                                      .replace('VANCO', 'Van'), axis=1)
            # for every column name of data that is in the column Sample in metadata, replace it with equivalent row from ID
            # column in metadata
            raw.columns = [meta[meta['Sample'] == col]['ID'].values[0] if col in meta['Sample'].values else col
                           for col in raw.columns]
            # remove all columns containing - or _
            raw = raw.drop([col for col in raw.columns if '-' in col or '_' in col if col != "gene_name"], axis=1)
            raw = raw.set_index("gene_name")
        elif run_type == "RASflow":
            # directory = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
            # data = parse_data(directory, only_old=True)
            # metadata = get_metadata(directory, only_old=True)
            # data.columns = [metadata[metadata['Sample'] == col]['ID'].values[0] if col in metadata['Sample'].values else col
            #                 for col in data.columns]
            # # remove samples with no metadata   # todo: check
            # data = data.drop([col for col in data.columns if '-' in col or '_' in col if col != "gene_name"], axis=1)
            # # sum rows with same gene_name but keep this gene_name column todo: note
            # data = data.groupby('gene_name').sum()
            # data['gene_name'] = data.index
            raw = transcriptome
        elif run_type == "RASflowRPKM":
            directory = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
            data = parse_data(directory, "rpkm", only_old=True)
            meta = get_metadata(directory, "rpkm", only_old=True)
            data.columns = [
                meta[meta['Sample'] == col]['ID'].values[0] if col in meta['Sample'].values else col
                for col in data.columns]
            # remove samples with no metadata   # todo: check
            data = data.drop([col for col in data.columns if '-' in col or '_' in col if col != "gene_name"], axis=1)
            # sum rows with same gene_name but keep this gene_name column todo: note
            data = data.groupby('gene_name').sum()
            data['gene_name'] = data.index
        else:
            raw = pd.read_csv(raw_data_path).set_index("Gene Symbol").fillna(0)  # changed to imputed
            meta = pd.read_csv(meta_data_path)
            raw = raw.div(raw.sum(axis=0), axis=1)
            if run_type == "_normalize_cols_and_rows":
                for abx in antibiotics:
                    for treatment in treatments:
                        abx_mice = meta[(meta['Drug'] == abx) & (meta['Treatment'] == treatment)]
                        pbs_mice = meta[(meta['Drug'] == 'PBS') & (meta['Treatment'] == treatment)]
                        all_mice = abx_mice.append(pbs_mice)
                        raw.loc[:, all_mice['ID']] = z_score_by_pbs(raw.loc[:, all_mice['ID']], abx_mice, pbs_mice)

        # raw = raw.replace(0, np.nan)
        # raw = np.log10(raw)
        # raw = impute_zeros(raw, meta, 'Treatment', run_type, skip_if_exist=True)
        # raw = np.log2(raw)

        raw, metadata = transform_data(raw, meta, run_type, skip=True)

        # # drop C9, C10 from metadata and from data # note
        # meta = meta.drop(meta[meta['ID'] == 'C9'].index).drop(meta[meta['ID'] == 'C10'].index)
        # raw = raw.drop('C9', axis=1).drop('C10', axis=1)

        # # Remove V11 from data, and remove row ID==V11 from metadata
        # # raw = raw.drop('V11', axis=1)
        # meta = meta.drop(meta[meta['ID'] == 'V11'].index)
        # save raw to csv
        raw.to_csv(private + f"/analysis/Diff_abx{run_type}/normalized_multiabx.csv")

        # plot_selected_clusters(raw, meta, "diff_abx" + run_type)
        # exit()

        # plot_significant_genes_number(meta, raw, antibiotics, treatments, "diff_abx" + run_type)
        # effective_number_genes(raw, meta)
        # compare_significance_go(param="diff_abx" + run_type)

        # clusters_compare_mix(raw, meta, antibiotics, treatments, "diff_abx" + run_type)

        # compare_to_gsea(meta, antibiotics, treatments, 'Treatment', "diff_abx" + run_type)
        # plot_median_all_conditions(meta, raw, antibiotics, treatments, 'Treatment', '\\diff_abx' + run_type, run_type,
        #                            regular=False)
        # todo: Document methods!!!
        # plot_all()
        # plot_clusters(np.log2(raw + 1))

        # save_median_all_conditions(meta, raw, antibiotics, treatments, "Treatment", "diff_abx" + run_type)
        our = plot_categories(antibiotics, treatments, "\\diff_abx" + run_type, False, regular=False)
        gsea = plot_categories(antibiotics, treatments, "\\diff_abx" + "GSEA", False, regular=False, gsea=True)
        plot_correlation_gsea(gsea, our)

    # selected_data = get_selected(all_data)
    # # # print(len(selected_data))
    # # dimension_reduction(raw.loc[selected_data], meta, 'Treatment')
    # all_go_concat = get_all_go_raw("Treatment", raw, "diff_abx", meta)
    # dimension_reduction(all_go_concat, meta, 'Treatment')

    # already happens in ClusteringGO
    # new_controls = pd.read_csv("../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/Partek_Lilach_New_controls_Shai_"
    #                            "Normalization_Normalized_counts_RPKM.tsv", sep='\t').set_index('Gene Symbol')
    # new_metadata = pd.read_csv("../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new-samples.tsv", sep='\t')
    # # concat all data and metadata
    # new_all_data = pd.concat([new_controls, raw], axis=1).fillna(0)
    # new_all_meta = pd.concat([new_metadata, meta], axis=0)

    # dimension_reduction(new_all_data, new_all_meta, "Treatment")

    # for treat in conditions:
    #     for abx in antibiotics:
    #         abx_data = meta[(meta['Drug'] == abx) & (meta['Treatment'] == treat)]
    #         pbs_data = meta[(meta['Drug'] == 'PBS') & (meta['Treatment'] == treat)]
    #         # specific_raw = raw[(raw["Antibiotics"] == abx) & (raw["Treatment"] == treat)]
    #         df = pd.read_csv(path + f"top_correlated_GO_terms_{abx}_{treat}.tsv", sep="\treat")
    #         # scatter plot for distance vs. # genes in cluster (color and shape by treatment)
    #         # plot(df, "", 'size', '\"distance\"')
    #
    #         selected = df[(df['MWU'] <= 0.05) & (df['size'] >= 2) & (
    #                 df['better than parent'] != False) & (df['better than random correlation'] == 1)]
    #         # selected = df[(df['size'] >= 5) & (df['better than parent'] != False) &
    #         #               (df['better than random correlation'] == 1)].sort_values(by='MWU').head(20)
    #         print(abx, treat, selected.shape)
    #         # ax = plt.subplot(rows, cols, i * cols + j + 1)
    #         matrix = plot_medians(selected, raw, abx_data, pbs_data, f"{treat} {abx}", True, False)
    #         # print(selected['MWU'].iloc[19])
    #
    #         # iterate over each row in selected and get the relevant GO object
    #         # for index, row in selected.iterrows():
    #
    #         # heatmap showing for each “surviving” cluster just the median expression, so each cluster is a single
    #         # row, and now have all 800 and ALL the samples. Let seaborn do a cluster-map (hierarchical clustering) on
    #         # the clusters, but organize the samples (columns) according to conditions.
    #         # plot_medians(df[df['treat-test p-value'] < 0.05], raw, abx_data, pbs_data, f"p-val {abx}_{treat}")
    # # plt.savefig(f"C:\\Users\\Yehonatan\\Desktop\\Master\\Git\\DEP_Compare16s\\Private\\all_conditions_median.png")
    # # plt.show()
