import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.special import comb
from venn import venn

from ClusteringGO import build_tree, calculate_correlation, get_ensmus_dict
from all_figures_plot import get_median_matrices
from clusters_plot import plot_categories, intersection, z_score_by_pbs, get_to_axis, set_figure

data_path = os.path.join("Data")
spf = os.path.join("SPF time points")
gf = os.path.join("GF time points")
private = os.path.join("Private")
path = os.path.join(private, "clusters_properties")


def get_clusters_names_dict(abx, treat, exp_type, space=50):
    file = pd.read_csv(os.path.join("Private", "clusters_properties", rf"top_correlated_GO_terms_{abx}_{treat}.tsv"), sep="\t")
    # create a dictionary from column GO term to name
    clusters_names_dict = dict(zip(file['GO term'], file['name']))
    clusters_names_dict = {key: value.split(":")[1] for key, value in clusters_names_dict.items()}
    truncated_dict = {
        key: ' '.join(value[:value[:space].rfind(' ')].split()) + ' [...]' if len(value) > 40 else value
        for key, value in clusters_names_dict.items()
    }
    return truncated_dict


def get_genes_from_df(df, go_cluster):
    return [gene.strip("{").strip("}").strip(' ').strip("\"").strip("\'") for gene in
            df[(df["GO term"] == go_cluster)]["genes"].values[0].split(",")]


def get_median_from_df(data, go_cluster, mice, temp, genes):
    relevant_genes = [gene for gene in genes if gene in data.index]
    median = np.median(data[mice].loc[relevant_genes], axis=0)
    line = np.concatenate([np.array([go_cluster]), median])
    temp = temp.append(pd.Series(line), ignore_index=True)
    return temp


def prepare_data(anti, condition, exp_type, meta_data, treat):
    temp = pd.DataFrame()
    df = pd.read_csv(os.path.join("Private", "clusters_properties", f'{exp_type}/top_correlated_GO_terms_{anti}_{treat}.tsv'),
                     sep="\t")
    abx = meta_data[(meta_data['Drug'] == anti) & (meta_data[condition] == treat)]
    pbs = meta_data[(meta_data['Drug'] == 'PBS') & (meta_data[condition] == treat)]
    mice = pd.concat((abx['ID'], pbs['ID']))
    return abx, df, mice, pbs, temp


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
            plt.savefig(os.path.join("Private", f"{row['Antibiotics']}_{row['Condition']}_GO_{row['GO term'][3:]}_{suppress}.png"))
        if show:
            plt.show()


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
    plt.savefig(private + fr"/{exp_type}{run_type} medians_of_all.png", bbox_inches='tight')
    plt.show()
    plt.close()

    # save GO_number to a csv file
    GO_number.to_csv(private + fr"/{exp_type}{run_type} GO_number.csv")


def all_path(x):
    return os.path.join(path, f"{x}", "top_correlated_GO_terms.tsv")


time_hr = [5, 11, 17, 23]
antibiotics = ['Vanco']

mucin_production = ["muc2", "tff3", "clca1", "fcgbp", "mep1b"]
antimicrobial_peptide_defense = ["zg16", "retnlb", "reg3g", "reg3b", "sprr2a"]
circadian_clock_genes = ["arntl", "bmal1", "chrono", "nfil3", "nr1d1", "per1", "per2", "cry1", "cry2", "dbp"]
genes_of_interest = set(mucin_production).union(set(antimicrobial_peptide_defense)).union(set(circadian_clock_genes))


def run_prep(is_gf, condition, median=False, clock_genes=False, get_categories=False, partek=False):
    # read csv file of metadata
    if partek:
        if not is_gf:
            meta = pd.read_csv(os.path.join(data_path, spf, "MICE 7-10.3.22.csv"))
            data = pd.read_csv(
                os.path.join(data_path, spf, "Partek_Lilach_20220407_Normalization_Normalized_counts.csv"))
            data = data.set_index('Gene Symbol')
        else:
            meta = pd.read_csv(os.path.join(data_path, gf, "GeneMat_RNAseq_ShaiBel.csv"))
            data = pd.read_csv(os.path.join(data_path, gf, "Partek-mouseRNA-20220516-_Lilach_SPF_GF_RPKM.csv"))
            data = data.set_index('Gene Symbol')
            data.columns = [name[:-3] for name in data.columns]
    else:
        condition_name = "GF" if is_gf else "SPF"
        data, meta = get_meta_data(condition_name)

    # sort meta by Time_hr
    meta = meta.sort_values(by=['Time_hr'], ignore_index=True)
    # data = np.log2(data + 1)
    mice = meta['Drug'] == condition if condition else [True] * meta.shape[0]
    data = data[meta[mice]['ID']]
    if median:
        mice_type = "gf" if is_gf else "spf"
        abx, genes_df, medians, genes_dict = collect_medians("Time_hr", data, mice_type, meta, False)
        data = genes_df
    # replace data.columns with concatenation of meta['ID'] and meta['Time_hr'] and meta['Drug']
    data.columns = [f"{ID}_{meta['Time_hr'][i]}_{meta['Drug'][i]}" for i, ID in enumerate(meta[mice]['ID'])]
    meta['ID'] = meta['ID'].astype(str) + "_" + meta['Time_hr'].astype(str) + "_" + meta['Drug'].astype(str)
    if median and get_categories:
        return data, meta, genes_df.index, genes_dict
    if clock_genes:
        circadian_clock_genes = ["Clock", "Arntl", "Bmal1", "Chrono", "Nfil3", "Nr1d1", "Per1", "Per2", "Cry1", "Cry2",
                                 "Dbp"]
        genes = [gene for gene in circadian_clock_genes if gene in data.index]
        data = data.loc[genes]
    mice_type = "GF" if is_gf else "SPF"
    # if folder ./Private/{mice_type.upper()} doesn't exist, create it
    if not os.path.exists(os.path.join("Private", mice_type.upper())):
        os.makedirs(os.path.join("Private", mice_type.upper()))
    data.to_csv(os.path.join("Private", mice_type.upper(), f"{mice_type} data.txt"), sep='\t')
    # save data.index to file and ignore the index of the line
    pd.Series(data.index).to_csv(os.path.join("Private", mice_type.upper(), "annot_data.txt"),
                                 sep='\t', index=False)
    return data, meta, data.index, {}


def reformat_meta(filename, meta_name):
    meta_data = pd.read_excel(data_path + spf + filename)
    meta_data.to_csv(data_path + spf + meta_name)
    return meta_data


def get_meta_data(cond, filter_threshold=0.55, remove_mitochondrial=True, normalized_only=False):
    meta = pd.read_csv(f"./Data/{cond.upper()} time points/metadata.csv")
    data = pd.read_csv(f"./Data/{cond.upper()} time points/genes_norm_named.tsv", sep="\t")
    stats = pd.read_csv(f"./Data/{cond.upper()} time points/stats_{cond.lower()}.csv")
    # create new column "aligned" from stats["% Aligned"] by removing last char (%) and converting to float
    stats["aligned"] = stats["% Aligned"].apply(lambda x: float(x[:-1]))
    assert (stats["aligned"].shape[0] == stats[stats["aligned"] > 80].shape[0]), "not all samples are aligned > 80%"
    samples = stats[stats['aligned'] > filter_threshold]['Sample Name']
    # # print the filtered out samples, sorted lexically
    # print(sorted([sample for sample in stats['Sample Name'] if sample not in samples.values]))

    # ensmus = data.set_index('gene_id')['gene_name'].to_dict()
    data = data.set_index("gene_id")
    data = data.drop("gene_name", axis=1)
    data = data[~data.index.isna()]
    data = data.groupby(data.index).sum()
    # replace data.columns with data.columns.split("_")[1] if it has "_"
    if cond == "SPF":
        data.columns = [col.split("_")[1] if (("_" in col) and ("id" not in col)) else col for col in data.columns]
        # replace meta["ID"] with meta["ID"].split("_")[1] if it has "_"
        meta["ID"] = [col.split("_")[1] if "_" in col else col for col in meta["ID"]]
    else:
        meta["ID"] = ["GF_" + col.split("_")[1] if "_" in col else col for col in meta["ID"]]

    # keep only metadata rows with Sample Name in Sample
    meta = meta[meta['ID'].isin(samples)]
    # keep only columns that are in metadata["ID"].values
    data = data[[col for col in data.columns if col in meta["ID"].values]]

    # remove sparse genes (more than 50% zeros in a row):
    # check all sparse genes (more than 50% zeros in a row) in each df, and check if the non-zero samples are the same
    # condition, using the metadata
    data_zeros = data[data == 0].count(axis=1)
    data_sparse = data_zeros[data_zeros > 0.5 * data.shape[1]]
    data = data.drop(data_sparse.index)

    if remove_mitochondrial:
        from ClusteringGO import mitochondrial_genes
        matching_indices = data.index[data.index.str.lower().isin(set(mitochondrial_genes))].tolist()
        # remove mitochondrial genes from the dataframes
        data = data.drop(matching_indices, errors='ignore')

    data = (data * 1_000_000).divide(data.sum(axis=0), axis=1)
    if normalized_only:
        return data, meta

    # transform data
    from ClusteringGO import impute_zeros
    data = data.replace(0, np.nan)
    data = impute_zeros(data, meta, 'Time_hr', f"time_points_{cond}", skip_if_exist=True)
    data = np.log2(data)
    data = zscore_all_by_pbs(data, meta)
    return data, meta


def zscore_all_by_pbs(data, metadata):
    for time in time_hr:
        pbs = metadata[((metadata['Drug'] == "PBS") & (metadata['Time_hr'] == time))]
        # get the pbs mice data
        pbs_data = data[pbs['ID']]
        # calculate the mean and std of the pbs mice
        pbs_mean = pbs_data.mean(axis=1)
        pbs_std = pbs_data.std(axis=1)
        # replace pbs_std 0 values by np.nanmin(pbs_std)
        pbs_std[pbs_std == 0] = np.nanmin(pbs_std[pbs_std != 0])
        data[pbs['ID']] = data[pbs['ID']].sub(pbs_mean, axis=0)
        data[pbs['ID']] = data[pbs['ID']].div(pbs_std, axis=0)
        abx = metadata[((metadata['Drug'] == "Vanco") & (metadata['Time_hr'] == time))]
        # normalize the data by the mean and std of the pbs mice: subtract pbs_mean from every row and divide by std
        data[abx['ID']] = data[abx['ID']].sub(pbs_mean, axis=0)
        data[abx['ID']] = data[abx['ID']].div(pbs_std, axis=0)
    # return the normalized data
    return data


def run_spf(to_cluster, plot, intersect):
    exp_type = 'spf'
    spf_data, spf_meta = get_meta_data("SPF")
    run(exp_type, spf_data, spf_meta, plot, to_cluster, intersect)


def run_gf(to_cluster, plot, intersect):
    exp_type = 'gf'
    gf_data, gf_meta = get_meta_data("GF")
    run(exp_type, gf_data, gf_meta, plot, to_cluster, intersect, regular=False)


def run(exp_type, data, meta, plot, to_cluster, intersect, regular=False, gene_to_check="H2-Ab1"):
    if to_cluster:
        # replace all zeros with nan
        tree, tree_size = build_tree()
        calculate_correlation(tree, data, meta, tree_size, antibiotics, time_hr, gene_to_check, exp_type, 'Time_hr',
                              significance_threshold=0.05)  # note
        # save_median_all_conditions(meta, data, antibiotics, time_hr, "Time_hr", exp_type)
    if plot:
        plot_median_all_conditions(meta, data, antibiotics, time_hr, 'Time_hr', "/" + exp_type, regular=regular,
                                   cols_factor=7.0, rows_factor=6.0)
        plot_categories(antibiotics, time_hr, "/" + exp_type, regular=regular, anchor=(0, -7.5))
        # selected_data = get_selected(pd.read_csv(all_path(exp_type), sep="\t"))
        # dimension_reduction(data.loc[selected_data], meta, "Time_hr")
    if intersect:
        intersection(antibiotics, time_hr, "/" + exp_type)


def collect_medians(condition, data, exp_type, meta_data, plot_intersection):
    all_clusters = {}
    for anti in antibiotics:
        for t in time_hr:
            df = pd.read_csv(os.path.join("Private", "clusters_properties", exp_type, f"top_correlated_GO_terms_{anti}_{t}.tsv"),
                             sep="\t")
            clusters = df[(df["better than random correlation"] == "True")  # & (df["\"distance\""] < 0.5)]
                          & (df["better than parent"] == True) & df["relative size"] >= 0.5]
            # clusters = df[(df['MWU'] <= 0.05) & (df["better than random correlation"] == "True") &
            #               (df["better than parent"] != "False")]
            print(exp_type, anti, t, clusters.shape)
            all_clusters[f"Time:{t}"] = set(clusters["GO term"].values)
        if plot_intersection:
            show_intersection(all_clusters, exp_type)
    all_listed = np.array(
        list(set(all_clusters[f"Time:{time_hr[0]}"]).intersection(*[all_clusters[f"Time:{t}"] for t in time_hr])))
    all_genes = get_genes_dict(all_clusters, condition, data, exp_type, meta_data)
    # for t in time_hr:
    # all_listed = np.concatenate([all_listed, np.array(list(all_clusters[f"Time:{t}"]))])
    abx, genes_df, median_dict = get_medians_given_categories(all_genes, all_listed, condition, data, exp_type,
                                                              meta_data)
    # create medians
    medians = np.array(
        [np.median(np.array([median_dict[antibiotics[0]][t][go_cluster] for t in time_hr]).ravel()) for
         go_cluster in all_listed])
    return abx, genes_df, medians, all_genes


def plot_genes(gf_data, spf_data, filter_type, go_dict, spf_all, first="GF", second="SPF", save=True,
               cat=["maintained", "lost", "gained"]):
    from random_forests import get_ensmus_dict
    ensmus_dict = get_ensmus_dict()
    # boxplot every row in df in separate graph.
    # divide columns to 4 time points, and in each time point plot the 4 mice of the abx separately than the pbs,
    # all in same figure
    res = pd.DataFrame(columns=["gene", "SPF PBS", "SPF ABX", "SPF PBS-abx", "GF PBS", "GF ABX", "GF PBS-abx"])
    # res = pd.DataFrame(columns=["GO number", "GO text name", "SPF PBS-abx", "GF PBS-abx"])
    for i, row_gf in gf_data.iterrows():
        df = pd.DataFrame(row_gf)
        gene = row_gf.name
        spf_row = pd.DataFrame(spf_data.loc[i])
        # concatenate the 2 dataframes to one long column
        df = pd.concat([df, spf_row], axis=0)
        # make a col "time" with index.split("_")[2] if index starts with "GF" else index.split("_")[1]
        df['time'] = [ind.split("_")[-2] for ind in df.index]
        df['abx'] = [ind.split("_")[-1] for ind in df.index]
        # convert df["abx"] == Vanco to ABX
        df['abx'] = df['abx'].apply(lambda x: "ABX" if x == "Vanco" else "PBS")
        # df["Type"] = "GF" if df.index.str.startswith("GF") else "SPF"
        df["Type"] = [ind.split("_")[0] for ind in df.index]
        df["Type"] = df["Type"].apply(lambda x: first if x == first else second)
        # make a col "Time_hr" with "int(time)-6 Type"
        df['Time_hr'] = df['time'].apply(lambda x: f"{int(x) - 6}")
        df["Time_hr"] = df["Time_hr"] + " " + df["Type"]

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(211)
        # Reverse the order of the "hue" variable's color palette
        palette = sns.color_palette("tab10", n_colors=2)
        palette = palette[::-1]

        # ax = sns.boxplot(data=df[df['Type'] == first], x="Time_hr", y=i, hue="abx", ax=ax)
        ax = sns.swarmplot(data=df[df['Type'] == first], x="Time_hr", y=i, hue="abx", ax=ax, palette=palette)
        sns.pointplot(x='Time_hr', y=i,
                      data=df[(df['Type'] == first) & (df['abx'] == "ABX")].groupby('Time_hr', as_index=False)[
                          gene].median(),
                      ax=ax, color=sns.color_palette()[1],
                      order=["-1 " + first, "5 " + first, "11 " + first, "17 " + first])
        sns.pointplot(x='Time_hr', y=i,
                      data=df[(df['Type'] == first) & (df['abx'] == "PBS")].groupby('Time_hr', as_index=False)[
                          gene].median(),
                      ax=ax, color=sns.color_palette()[0],
                      order=["-1 " + first, "5 " + first, "11 " + first, "17 " + first])
        gene_name = ensmus_dict[row_gf.name] if row_gf.name in ensmus_dict else row_gf.name
        ax.title.set_text(
            f"{first}, {gene_name}, abx={go_dict[0].loc[row_gf.name][1]}, pbs={go_dict[1].loc[row_gf.name][1]}\n "
            f"{filter_type}")
        # ax.title.set_text(f"{go_dict[row_gf.name.split('_')[0]]}\n{row_gf.name}, {filter_type}\n{first}")
        # remove x and y labels
        ax.set_xlabel('')
        ax.set_ylabel('')

        find = [row_gf.name in spf_all[i] for i in range(len(spf_all))]
        index = find.index(True) if find.count(True) >= 1 else -1
        spf_type = cat[index] if index != -1 else ""
        if spf_type:
            ax = fig.add_subplot(212)
            # sns.boxplot(data=df[df['Type'] == second], x="Time_hr", y=i, hue="abx", ax=ax)
            sns.swarmplot(data=df[df['Type'] == second], x="Time_hr", y=i, hue="abx", ax=ax, palette=palette)
            sns.pointplot(x='Time_hr', y=i,
                          data=df[(df['Type'] == second) & (df['abx'] == "ABX")].groupby('Time_hr',
                                                                                         as_index=False)[gene].median(),
                          ax=ax, color=sns.color_palette()[1], order=["-1 " + second, "5 " + second, "11 " + second,
                                                                      "17 " + second])
            sns.pointplot(x='Time_hr', y=i,
                          data=df[(df['Type'] == second) & (df['abx'] == "PBS")].groupby('Time_hr',
                                                                                         as_index=False)[gene].median(),
                          ax=ax, color=sns.color_palette()[0], order=["-1 " + second, "5 " + second, "11 " + second,
                                                                      "17 " + second])
            ax.title.set_text(f"{second}, {spf_type}")  # {row_gf.name}")
            # plt.title(f"{second} {filter_type} {row_gf.name}")
            # remove x and y labels
            ax.set_xlabel('')
            ax.set_ylabel('')
        plt.tight_layout()
        # if there's no folder in the name filer_type, create one
        if not os.path.exists(f"../../periodicity detection/results/{filter_type}"):
            os.makedirs(f"../../periodicity detection/results/{filter_type}")
        if save:
            print("saving")
            plt.savefig(
                f"../../periodicity detection/results/{filter_type}/{first} {filter_type} {gene_name.replace(':', '_')}.png")
        # f"../../periodicity detection/JTK/results/{filter_type}/{first} {filter_type} {row_gf.name[3:]}.png")
        plt.show()
        plt.close()

        # Create a new row as a DataFrame
        new_row = pd.DataFrame([{
            "gene": row_gf.name,
            f"{second} PBS-abx": spf_type,
            f"{first} PBS": go_dict[1].loc[row_gf.name][1],
            f"{first} ABX": go_dict[0].loc[row_gf.name][1],
            f"{first} PBS-abx": filter_type
        }])

        # Concatenate the new row to the existing DataFrame
        res = pd.concat([res, new_row], ignore_index=True)

        # res = res.append({"GO number": row_gf.name, "GO text name": go_dict[row_gf.name.split('_')[0]],
        #                   f"{second} PBS-abx": spf_type, f"{first} PBS-abx": filter_type}, ignore_index=True)
    return res


def get_genes_dict(all_clusters, condition, data, exp_type, meta_data):
    all_genes = {}
    for anti in antibiotics:
        for t in time_hr:
            abx, df, mice, pbs, temp = prepare_data(anti, condition, exp_type, meta_data, t)
            for go_cluster in all_clusters[f"Time:{t}"]:
                # cluster_name = go_cluster.split("_")[0] + "_" + go_cluster.split("_")[1]
                all_genes[go_cluster] = get_genes_from_df(df, go_cluster)
                temp = get_median_from_df(data, go_cluster, mice, temp, all_genes[go_cluster])
            if not temp.empty:
                temp = temp.set_index(0).astype(float)
                # drop all Nan rows
                temp = temp.dropna(axis=0)
                # replace all nans with 0
                temp = temp.fillna(0)
                clustering = sns.clustermap(data=temp, row_cluster=True, col_cluster=False)
                order = clustering.dendrogram_row.reordered_ind
                plt.close()
                all_clusters[f"Time:{t}"] = temp.iloc[order].index
    return all_genes


def get_medians_given_categories(cluster_genes_dict, all_listed, condition, data, exp_type, meta_data):
    median_dict = {}
    genes_df = pd.DataFrame()
    for anti in antibiotics:
        median_dict[anti] = {}
        for t in time_hr:
            median_dict[anti][t] = {}
            abx, df, mice, pbs, temp = prepare_data(anti, condition, exp_type, meta_data, t)
            for go_cluster in all_listed:
                temp = get_median_from_df(data, go_cluster, mice, temp, cluster_genes_dict[go_cluster])
                relevant_genes = [gene for gene in cluster_genes_dict[go_cluster] if gene in data.index]
                median_dict[anti][t][go_cluster] = np.median(data[pbs['ID']].loc[relevant_genes], axis=0)
            temp = temp.set_index(0)
            temp.columns = mice
            genes_df = pd.concat([genes_df, temp], axis=1)
    genes_df = genes_df.astype('f').fillna(0)
    return abx, genes_df, median_dict


def show_intersection(all_clusters, exp_type):
    print(set(all_clusters["Time:11"]).intersection(set(all_clusters["Time:23"])))
    print(set(all_clusters["Time:5"]).intersection(set(all_clusters["Time:23"])))
    print(set(all_clusters["Time:5"]).intersection(set(all_clusters["Time:17"])))
    venn(all_clusters)
    plt.title(f"{exp_type.upper()} intersections")
    plt.savefig(private + f"TP {exp_type} intersection.png")
    plt.show()


def plot_clock_genes(circadian_clock_genes, condition, data, exp_type, meta_data, z_score=False):
    clusters = {}
    for anti in antibiotics:
        clusters[anti] = {}
        for t in time_hr:
            df = pd.read_csv(os.path.join("Private", "clusters_properties", exp_type, f'top_correlated_GO_terms_{anti}_{t}.tsv'),
                             sep="\t")
            # iterate over rows and check if any of genes are there:
            # df['clock'] = df.apply(
            #     lambda r: any(gene.strip("{").strip("}").strip(' ').strip("\"").strip("\'") in r['genes'].split(",") for
            #                   gene in genes), axis=1)
            df["clock"] = False
            for index, row in df.iterrows():
                genes = row["genes"].split(",")
                genes = set([gene.strip("{").strip("}").strip(' ').strip("\"").strip("\'") for gene in genes])
                for gene in circadian_clock_genes:
                    if gene in genes:
                        df.at[index, "clock"] = True
                        continue
                # if any(gene in row['genes'].strip("{").strip("}").split(",") for gene in genes):
                #     print(row["GO cluster_res"])
            clusters[anti][t] = df[(df["clock"] == True) & (df["better than parent"] == True)]["GO term"]
    # plot clock genes data
    data = data.reindex(sorted(data.columns), axis=1)
    genes_df = pd.DataFrame()
    for anti in antibiotics:
        for t in time_hr:
            abx = meta_data[(meta_data['Drug'] == anti) & (meta_data[condition] == t)]
            pbs = meta_data[(meta_data['Drug'] == 'PBS') & (meta_data[condition] == t)]
            mice = pd.concat((abx['ID'], pbs['ID']))
            label = [f"{t - 6}" for _ in range(abx.shape[0])]
            circadian_clock_in = [gene for gene in circadian_clock_genes if gene.lower() in data.index.str.lower()]

            sub_data = z_score_by_pbs(data[mice].loc[circadian_clock_in], abx, pbs) if z_score else \
                data.loc[circadian_clock_in]
            abx_data = sub_data[abx['ID']]  # .loc[circadian_clock_in]
            abx_data.columns = label
            pbs_data = sub_data[pbs['ID']]  # .loc[circadian_clock_in]
            pbs_data.columns = label
            data_abx_pbs = pd.concat([abx_data, pbs_data], axis=0)
            genes_df = pd.concat([genes_df, data_abx_pbs], axis=1)
            # genes_df = genes_df.append(pd.Series(data[mice].loc[circadian_clock_in]), ignore_index=True)
    # yticks = np.linspace(0, abx.shape[0]*len(time_hr), 4, dtype=np.int)
    xticklabels = np.array([['', f"      {t - 6}", '', ''] for t in time_hr]).flatten()
    ax = sns.heatmap(np.trunc(genes_df), cmap="vlag", yticklabels=True, xticklabels=xticklabels, vmax=3, vmin=-3)
    ax.hlines([len(circadian_clock_in)], *ax.get_xlim())
    ax.vlines([abx.shape[0] * i for i in range(1, len(time_hr))], *ax.get_ylim())
    plt.ylabel('PBS                                  Abx')
    plt.xlabel('ZT')
    z_score_label = ", z-score by PBS" if z_score else ""
    plt.title(f"{exp_type.upper()} Clock genes log(1+X) expression{z_score_label}")
    plt.savefig(os.path.join("Private", "analysis", exp_type, f'{exp_type} {'z-score' if z_score else ''} clock_genes.png'),
                bbox_inches="tight")
    plt.show()


def find_cycles(df, other_df, loc):
    p_val_thresh = 0.05
    fold_change_thresh = np.log2(2)
    # fold_change_thresh = 2
    df = df.set_index(df.columns[0])
    other_df = other_df.set_index(other_df.columns[0])
    res = {}
    # iterate over all genes and find cycles
    df.columns = [f"{col.split('_')[loc]}" for col in df.columns]
    other_df.columns = [f"{col.split('_')[loc]}" for col in other_df.columns]
    # df.columns = [f"{col.split('_')[2]}" for col in df.columns]
    # other_df.columns = [f"{col.split('_')[2]}" for col in other_df.columns]
    for i, row in enumerate(df.iterrows()):
        # ttest each group of columns, if p-value is significant, then there is a change in expression
        # if p-value is not significant, then there is no change in expression
        res[row[0]] = ""
        # compare the value of the first time_hr to the other_df equivalent line
        other_at_time = other_df.loc[row[0]][f"{time_hr[0]}"]
        this_at_time = row[1][f"{time_hr[0]}"]
        _, p_val = stats.ttest_ind(this_at_time, other_at_time)
        med_this = np.median(this_at_time)
        med_other = np.median(other_at_time)
        score = abs(med_this - med_other)
        # score = max(med_this, med_other) / min(med_this, med_other)
        diff = med_this - med_other
        res[row[0]] += ("low" if diff < 0 else "high")[0] if p_val < p_val_thresh and abs(
            score) > fold_change_thresh else "same"[0]
        scores = np.zeros(int(comb(len(time_hr), 2)))
        p_vals = np.zeros(int(comb(len(time_hr), 2)))
        diffs = np.zeros(int(comb(len(time_hr), 2)))
        for j, time in enumerate(time_hr[:-1]):
            p_val, score, diff = statistical_analysis(j, 1, row, time_hr)
            scores[j] = score
            p_vals[j] = p_val
            if p_val < p_val_thresh and abs(score) > fold_change_thresh:
                direction = "up" if diff < 0 else "down"
                res[row[0]] += direction[0]
            else:
                res[row[0]] += "no change"[0]
        p_vals, scores, diffs = find_other_p(p_vals, row, scores, diffs, 0, 2, 3, time_hr)
        p_vals, scores, diffs = find_other_p(p_vals, row, scores, diffs, 1, 2, 4, time_hr)
        p_vals, scores, diffs = find_other_p(p_vals, row, scores, diffs, 0, 3, 5, time_hr)
        # check if there's an index where p_val is significant and fold change is higher than threshold
        both = np.where((p_vals < p_val_thresh) & (abs(scores) > fold_change_thresh))[0]
        if len(both) > 0:
            minimal = np.argmin(p_vals[both])
            direction = "up" if diffs[both[minimal]] < 0 else "down"
            res[row[0]] += direction[0]
        else:
            res[row[0]] += "no change"[0]
    return res


def find_other_p(p_vals, row, scores, diffs, n, k, j, time_hr):
    p_val, score, diff = statistical_analysis(n, k, row, time_hr)
    scores[j] = score
    p_vals[j] = p_val
    diffs[j] = diff
    return p_vals, scores, diffs


def statistical_analysis(j, k, row, time_hr):
    _, p_val = stats.ttest_ind(row[1][f"{time_hr[j]}"], row[1][f"{time_hr[j + k]}"])
    # compare fold change of the median of the two groups
    med_this = np.median(row[1][f"{time_hr[j]}"])
    score = max(med_this, np.median(row[1][f"{time_hr[j + k]}"])) / min(med_this,
                                                                        np.median(row[1][f"{time_hr[j + k]}"]))
    diff = med_this - np.median(row[1][f"{time_hr[j + k]}"])
    return p_val, abs(diff), diff
    # return p_val, score, diff


def get_clusters():
    go_dict = {}
    all_spf = pd.read_csv(os.path.join("Private", "clusters_properties", "spf", f"top_correlated_GO_terms.tsv"), sep="\t")
    # iterate over all rows:
    for i, row_spf in all_spf.iterrows():
        go = row_spf["GO term"].split('_')[0]
        if go not in go_dict:
            go_dict[go] = row_spf["name"]
    all_gf = pd.read_csv(os.path.join("Private", "clusters_properties", "gf", f"top_correlated_GO_terms.tsv"), sep="\t")
    for i, row_gf in all_gf.iterrows():
        go = row_gf["GO term"].split('_')[0]
        if go not in go_dict:
            go_dict[go] = row_gf.name
    return go_dict


def find_all_cycles():
    for mice_type in ["SPF", "GF"]:
        loc = 1 if mice_type == "SPF" else 2
        for anti in ["PBS", "Vanco"]:
            print(f"Starting {mice_type} {anti}")
            if os.path.exists(os.path.join("Private", mice_type.upper(), f"{mice_type} cycles {anti} results.txt")):
                print(f"Results for {mice_type} {anti} already exist, skipping...")
                continue
            # elif private/f"{mice_type.upper()} folder does not exist, create it
            if not os.path.exists(os.path.join("Private", mice_type.upper())):
                os.makedirs(os.path.join("Private", mice_type.upper()))
            data = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} data {anti}.txt", sep="\t")
            other_anti = "Vanco" if anti == "PBS" else "PBS"
            other_data = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} data {other_anti}.txt", sep="\t")
            results = find_cycles(data, other_data, loc)
            # save results to f"./Private/{mice_type.upper()}/{mice_type} cycles {anti} results.txt"
            with open(f"./Private/{mice_type.upper()}/{mice_type} cycles {anti} results.txt", "w") as f:
                for gene, cycle in results.items():
                    f.write(f"{gene}\t{cycle}\n")


def collect_stat(count, same, not_same, gene, abx, pbs):
    if abx.loc[gene][1] != pbs.loc[gene][1]:
        # print(f"{gene} {abx.loc[gene][1]} {pbs.loc[gene][1]}")
        count += 1
        if abx.loc[gene][1] not in not_same:
            if pbs.loc[gene][1] not in not_same:
                not_same[abx.loc[gene][1]] = set()
                not_same[abx.loc[gene][1]].add(pbs.loc[gene][1])
            else:
                not_same[pbs.loc[gene][1]].add(abx.loc[gene][1])
        else:
            not_same[abx.loc[gene][1]].add(pbs.loc[gene][1])
    else:
        if abx.loc[gene][1] not in same:
            same[abx.loc[gene][1]] = 1
        else:
            same[abx.loc[gene][1]] += 1


def plot_all_data(categories, median=False):
    if not median:
        run_prep(is_gf=True, condition=None, median=median, clock_genes=False, get_categories=True)
        run_prep(is_gf=False, condition=None, median=median, clock_genes=False, get_categories=True)
        gf = pd.read_csv(f"./Private/GF/GF data.txt", sep='\t').set_index('gene_id')
        spf = pd.read_csv(f"./Private/SPF/SPF data.txt", sep='\t').set_index('gene_id')
    else:
        gf, meta, _, genes_dict = run_prep(is_gf=True, condition=None, median=median,
                                           clock_genes=False, get_categories=True)
        spf, meta, _, genes_dict = run_prep(is_gf=False, condition=None, median=median,
                                            clock_genes=False, get_categories=True)

    # cluster_dict = get_clusters()
    tabular = pd.DataFrame(columns=["gene", "SPF PBS", "SPF ABX", "SPF PBS-abx", "GF PBS", "GF ABX", "GF PBS-abx"])
    spf_all = [categories["SPF"][category] for category in categories["SPF"]]
    gf_all = [categories["GF"][category] for category in categories["GF"]]
    for category in categories["GF"]:
        mice_type = "GF"
        abx = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} cycles Vanco results.txt", sep="\t",
                          header=None).set_index(0)
        pbs = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} cycles PBS results.txt", sep="\t",
                          header=None).set_index(0)
        print(category, "GF")
        gf_filtering = categories["GF"][category]
        relevant = [gene for gene in gf_filtering if
                    (gene in spf.index) and (gene in gf.index)]  # look for genes not in both
        if category == "maintained" or category == "mixed":
            relevant = [gene for gene in relevant if gene.lower() in genes_of_interest]
        # # print metadata
        # print(pd.read_csv(f"./Private/{mice_type.upper()}/GeneMat_RNAseq_ShaiBel.csv"))
        res = plot_genes(gf.loc[relevant], spf.loc[relevant], category, (abx, pbs), spf_all,
                         cat=list(categories["SPF"].keys()))
        # append res to tabular
        tabular = pd.concat([tabular, res], ignore_index=True)
        print("SPF")
        mice_type = "SPF"
        abx = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} cycles Vanco results.txt", sep="\t",
                          header=None).set_index(0)
        pbs = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} cycles PBS results.txt", sep="\t",
                          header=None).set_index(0)
        if category in categories["SPF"]:
            spf_filtering = categories["SPF"][category]
            relevant = [gene for gene in gf_filtering if
                        (gene in spf.index) and (gene in gf.index)]  # look for genes not in both
            if category == "maintained" or category == "mixed":
                relevant = [gene for gene in relevant if gene.lower() in genes_of_interest]
            res = plot_genes(spf.loc[relevant], gf.loc[relevant], category, (abx, pbs), gf_all, "SPF", "GF",
                             cat=list(categories["GF"].keys()))
            tabular = pd.concat([tabular, res], ignore_index=True)
    # merge all duplicate rows in tabular
    tabular = tabular.groupby(["gene", "SPF PBS-abx", "GF PBS-abx"]).size().reset_index(name='count')
    # drop the count column
    tabular = tabular.drop(columns=["count"])
    # save tabular to csv
    tabular.to_csv(f"../../periodicity detection/results/tabular-genes.csv", index=False)


def old_classification(gene, abx, pbs, categories, mice_type):
    global description
    changes = [abx.loc[gene][1][i] != pbs.loc[gene][1][i] for i in range(len(abx.loc[gene][1]))]
    # if difference is only in 1 char, cluster it as single_change
    if changes.count(True) == 1:
        # get index of the change
        change_index = changes.index(True)
        changed = f"{pbs.loc[gene][1][change_index]}_to_{abx.loc[gene][1][change_index]}"
        description = f"single_change_at_{-1 + change_index * 6}_{changed}"
        if description not in categories[mice_type]:
            categories[mice_type][description] = set()
        categories[mice_type][description].add(gene)
    # if difference is only in 2 chars, cluster it as double_change
    elif changes.count(True) == 2:
        # get index of the no-change
        change_index = changes.index(False)
        changed = f"{pbs.loc[gene][1][change_index]}"
        description = f"double_change_{changed}_not_at_{-1 + change_index * 6}"
        if description not in categories[mice_type]:
            categories[mice_type][description] = set()
        categories[mice_type][description].add(gene)
    # if difference is only in 3 chars, cluster it as triple_change
    elif changes.count(True) == 3:
        description = f"triple_change"
        if description not in categories[mice_type]:
            categories[mice_type][description] = set()
        categories[mice_type][description].add(gene)


def detect_change_hour(to_print=False):
    # global categories, mice_type, abx, pbs, gene, category_name
    categories = {}
    for mice_type in ["SPF", "GF"]:
        categories[mice_type] = {}
        abx = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} cycles Vanco results.txt", sep="\t",
                          header=None).set_index(0)
        pbs = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} cycles PBS results.txt", sep="\t",
                          header=None).set_index(0)
        # iterate over all genes and compare string in abx vs pbs
        for gene in abx.index:
            s_pbs = pbs.loc[gene][1]
            s_abx = abx.loc[gene][1]
            if s_pbs != s_abx:
                # classify to only suppression, only enhancement, mix
                if s_pbs[0] == 'h' and s_abx[0] == 'l':
                    categories = add_gene_to_category(f"suppressed_0", categories, gene, mice_type)
                elif s_pbs[0] == 'l' and s_abx[0] == 'h':
                    categories = add_gene_to_category(f"enhanced_0", categories, gene, mice_type)
                elif s_pbs[1] == "u" and s_abx[1] in ["n", "d"] or s_pbs[1] == "n" and s_abx[1] == "d":
                    categories = add_gene_to_category(f"suppressed_1", categories, gene, mice_type)
                elif s_pbs[1] == "d" and s_abx[1] in ["n", "u"] or s_pbs[1] == "n" and s_abx[1] == "u":
                    categories = add_gene_to_category(f"enhanced_1", categories, gene, mice_type)
                elif s_pbs[2] == "u" and s_abx[2] in ["n", "d"] or s_pbs[2] == "n" and s_abx[2] == "d":
                    categories = add_gene_to_category(f"suppressed_2", categories, gene, mice_type)
                elif s_pbs[2] == "d" and s_abx[2] in ["n", "u"] or s_pbs[2] == "n" and s_abx[2] == "u":
                    categories = add_gene_to_category(f"enhanced_2", categories, gene, mice_type)
                elif s_pbs[3] == "u" and s_abx[3] in ["n", "d"] or s_pbs[3] == "n" and s_abx[3] == "d":
                    categories = add_gene_to_category(f"suppressed_3", categories, gene, mice_type)
                elif s_pbs[3] == "d" and s_abx[3] in ["n", "u"] or s_pbs[3] == "n" and s_abx[3] == "u":
                    categories = add_gene_to_category(f"enhanced_3", categories, gene, mice_type)
                elif s_pbs[4] == "u" and s_abx[4] in ["n", "d"] or s_pbs[4] == "n" and s_abx[4] == "d":
                    # categories = add_gene_to_category(f"suppressed_4", categories, gene, mice_type)
                    categories = add_gene_to_category(f"suppressed_somewhere", categories, gene, mice_type)
                elif s_pbs[4] == "d" and s_abx[4] in ["n", "u"] or s_pbs[4] == "n" and s_abx[4] == "u":
                    categories = add_gene_to_category(f"enhanced__somewhere", categories, gene, mice_type)
                    # categories = add_gene_to_category(f"enhanced_g", categories, gene, mice_type)
            else:
                if s_pbs != "snnnn":
                    categories = add_gene_to_category(f"maintained_cycle", categories, gene, mice_type)
                categories = add_gene_to_category(f"maintained", categories, gene, mice_type)
    # print statistics of categories
    if to_print:
        print_stats(categories, "SPF")
        print_stats(categories, "GF")
    return categories


def print_stats(categories, mice_type):
    print(f"{mice_type}: {len(categories[mice_type])} categories")
    for category in categories[mice_type]:
        print(f"{category}: {len(categories[mice_type][category])} genes")
    # unite all sets from categories[mice_type]
    united = list(categories[mice_type].values())
    all_genes = set()
    for i in united:
        all_genes = all_genes.union(i)
    print(f"total {mice_type}: {len(all_genes)} genes")


def add_gene_to_category(category_name, categories, gene, mice_type):
    if category_name not in categories[mice_type]:
        categories[mice_type][category_name] = set()
    categories[mice_type][category_name].add(gene)
    return categories


def detect_change():
    # global categories, mice_type, abx, pbs, gene, category_name
    categories = {}
    for mice_type in ["SPF", "GF"]:
        categories[mice_type] = {}
        abx = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} cycles Vanco results.txt", sep="\t",
                          header=None).set_index(0)
        pbs = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} cycles PBS results.txt", sep="\t",
                          header=None).set_index(0)
        # iterate over all genes and compare string in abx vs pbs
        for gene in abx.index:
            # both[gene] = (abx.loc[gene][1], pbs.loc[gene][1])
            # collect_stat(count, same, not_same)
            s_pbs = pbs.loc[gene][1]
            s_abx = abx.loc[gene][1]
            if s_pbs != s_abx:
                # classify to only suppression, only enhancement, mix
                if ((s_pbs[0] == 'h' and s_abx[0] == 'l') or (s_pbs[0] == s_abx[0])) and \
                        (np.array([((s_pbs[i] == s_abx[i]) or (s_pbs[i] == "u" and s_abx[i] in ["n", "d"]) or
                                    (s_pbs[i] == "n" and s_abx[i] == "d")) for i in range(1, len(s_pbs))]).all()):
                    category_name = f"suppressed"
                    if category_name not in categories[mice_type]:
                        categories[mice_type][category_name] = set()
                    categories[mice_type][category_name].add(gene)
                elif ((s_pbs[0] == 'l' and s_abx[0] == 'h') or (s_pbs[0] == s_abx[0])) and \
                        (np.array([((s_pbs[i] == s_abx[i]) or (s_pbs[i] == "d" and s_abx[i] in ["n", "u"]) or
                                    (s_pbs[i] == "n" and s_abx[i] == "u")) for i in range(1, len(s_pbs))]).all()):
                    category_name = f"enhanced"
                    if category_name not in categories[mice_type]:
                        categories[mice_type][category_name] = set()
                    categories[mice_type][category_name].add(gene)
                else:
                    category_name = f"mixed"
                    if category_name not in categories[mice_type]:
                        categories[mice_type][category_name] = set()
                    categories[mice_type][category_name].add(gene)
                # old_classification()
            else:
                category_name = f"maintained"
                if category_name not in categories[mice_type]:
                    categories[mice_type][category_name] = set()
                categories[mice_type][category_name].add(gene)
    return categories


def translate_to_number(df, condition):
    translation = {"h": 1, "l": -1, "s": 0, "n": 0, "u": 1, "d": -1}
    # for i in range(len(df)):
    #     for k in range(len(df.iloc[i])):
    #         df.iloc[i][f'{k*6-1}'] = translation[df.iloc[i][1][k]]
    df[f"-1_{condition}"] = df[1].apply(lambda x: translation[x[0]])
    df[f"-1:5_{condition}"] = df[1].apply(lambda x: translation[x[1]])
    df[f"5:11_{condition}"] = df[1].apply(lambda x: translation[x[2]])
    df[f"11:17_{condition}"] = df[1].apply(lambda x: translation[x[3]])
    return df.drop(columns=[1])


def plot_heatmap(category_dict, title, median=False, classification=False):
    cols_spf = ['24_S24_5_Vanco', '23_S23_5_Vanco', '22_S22_5_Vanco', '21_S21_5_Vanco',
                '20_S20_5_PBS', '19_S19_5_PBS', '18_S18_5_PBS', '17_S17_5_PBS',
                '32_S32_11_Vanco', '29_S29_11_Vanco', '30_S30_11_Vanco', '31_S31_11_Vanco',
                '28_S28_11_PBS', '27_S27_11_PBS', '26_S26_11_PBS', '25_S25_11_PBS',
                '8_S8_17_Vanco', '7_S7_17_Vanco', '6_S6_17_Vanco', '5_S5_17_Vanco',
                '4_S4_17_PBS', '3_S3_17_PBS', '2_S2_17_PBS', '1_S1_17_PBS',
                '14_S14_23_Vanco', '13_S13_23_Vanco', '15_S15_23_Vanco', '16_S16_23_Vanco',
                '12_S12_23_PBS', '11_S11_23_PBS', '10_S10_23_PBS', '9_S9_23_PBS']
    cols_gf = ['10E_S17_5_PBS', '11E_S18_5_PBS', 'GF39_S19_5_PBS', 'GF40_S20_5_PBS',
               '13E_S24_5_Vanco', 'GF20_S23_5_Vanco', 'GF19_S22_5_Vanco', 'GF18_S21_5_Vanco',
               '14E_S25_11_PBS', '15E_S26_11_PBS', '16E_S27_11_PBS', '17E_S28_11_PBS',
               'GF49_S32_11_Vanco', 'GF48_S31_11_Vanco', 'GF47_S30_11_Vanco', 'GF26_S29_11_Vanco',
               '3E_S4_17_PBS', 'GF29_S3_17_PBS', 'GF28_S2_17_PBS', 'GF27_S1_17_PBS',
               'GF8_S6_17_Vanco', 'GF31_S8_17_Vanco', 'GF5_S5_17_Vanco', 'GF30_S7_17_Vanco',
               '6E_S12_23_PBS', 'GF34_S11_23_PBS', 'GF33_S10_23_PBS', 'GF32_S9_23_PBS',
               'GF12_S13_23_Vanco', 'GF13_S14_23_Vanco', 'GF14_S15_23_Vanco', 'GF37_S16_23_Vanco']
    cols_spf = [s.split('_', 1)[1] for s in cols_spf]
    cols_gf = [s.split('_', 1)[1] for s in cols_gf]
    for mice_type in ["SPF", "GF"]:
        # read original data
        data, meta, categories, genes_dict = run_prep(is_gf=mice_type == "GF", condition=None, median=median,
                                                      clock_genes=False, get_categories=True)
        if mice_type == "GF":
            data.columns = [s.split('_', 1)[1] for s in data.columns]
            meta.ID = meta.ID.str.replace("GF_", "")
        # meta["ID"] = meta["ID"] + "_" + meta["Time_hr"].astype(str) + "_" + meta["Drug"].astype(str)
        if classification:
            pbs = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} cycles PBS results.txt", sep="\t",
                              header=None).set_index(0)
            pbs = translate_to_number(pbs, "pbs")
            abx = pd.read_csv(f"./Private/{mice_type.upper()}/{mice_type} cycles Vanco results.txt", sep="\t",
                              header=None).set_index(0)
            abx = translate_to_number(abx, "abx")
            # go_categories = get_
            # concatenate pbs and abx, with proper column name
            combine = pd.concat([pbs, abx], axis=1)
        else:
            combine = data
        cols = cols_spf if mice_type == "SPF" else cols_gf
        combine = combine[cols]
        maintained_genes = category_dict[mice_type]["maintained"]
        meta_abx = meta.loc[meta["Drug"] == "Vanco"]
        meta_pbs = meta.loc[meta["Drug"] == "PBS"]
        # plot heatmap of maintained genes
        if median:
            data, meta, categories, genes_dict = run_prep(is_gf=mice_type == "GF", condition=None, median=True,
                                                          clock_genes=False, get_categories=True)
            combine = data

        if "maintained_cycle" in category_dict[mice_type]:
            maintained_cycle = category_dict[mice_type]["maintained_cycle"]
            plot_genes_heatmap(combine, maintained_cycle, mice_type, "maintained_cycle", meta_abx, meta_pbs,
                               classification, title)
        plot_genes_heatmap(combine, maintained_genes, mice_type, "maintained", meta_abx, meta_pbs, classification,
                           title)
        plot_change(meta_abx, meta_pbs, category_dict, combine, mice_type, "enhanced", classification, median, title)
        plot_change(meta_abx, meta_pbs, category_dict, combine, mice_type, "suppressed", classification, median, title)


def plot_change(abx, pbs, category_dict, combine, mice_type, cycle_type, classification, median, title):
    genes = np.array([])
    locs = []
    for time in range(5):
        # add f"{cycle_type}_{time}" to genes
        # add all genes from the set in category_dict[mice_type][f"{cycle_type}_{time}"] to genes as sstrings
        if f"{cycle_type}_{time}" in category_dict[mice_type]:
            new_genes = np.array(list(category_dict[mice_type][f"{cycle_type}_{time}"]))
            if not median and not combine.loc[new_genes].empty:
                order = sns.clustermap(combine.loc[new_genes], cmap="coolwarm", col_cluster=False,
                                       row_cluster=True).dendrogram_row.reordered_ind
                new_genes = new_genes[order]
                plt.close()
            genes = np.append(genes, new_genes).ravel()
            locs.append(len(genes))
    plot_genes_heatmap(combine, genes, mice_type, cycle_type, abx, pbs, classification, title, locs)


v_max_dict = {"SPF": {"maintained": 1.5, "maintained_cycle": 3, "enhanced": 2, "suppressed": 2, "clock": 2},
              "GF": {"maintained": 1, "maintained_cycle": 4, "enhanced": 2, "suppressed": 2, "clock": 2}}
v_min_dict = {"SPF": {"maintained": -1.5, "maintained_cycle": -2, "enhanced": -2, "suppressed": -2, "clock": -2},
              "GF": {"maintained": -1, "maintained_cycle": -4, "enhanced": -2, "suppressed": -2, "clock": -2}}


def plot_genes_heatmap(combine, genes, mice_type, genes_type, abx, pbs, classification, type_, vert_locs=None,
                       save_df=False):
    ensmus_dict = get_ensmus_dict()
    # sns.heatmap(combine.loc[genes], cmap="coolwarm", vmin=-1, vmax=1, z_score=0)
    genes = np.array([gene for gene in genes if gene in combine.index])

    orig_data, orig_meta = get_meta_data(mice_type, normalized_only=True)
    if mice_type == "GF":
        orig_data.columns = [col.split("_")[1] for col in orig_data.columns]
    orig_data = orig_data.loc[genes]
    to_show = z_score_by_pbs(combine.loc[genes], abx, pbs) if not classification else combine.loc[genes]
    if to_show.empty:
        print(mice_type, genes_type, "is empty")
        return
    # if not classification:
    #     to_show = np.log2(to_show + 1)
    index = 1  # if mice_type == "SPF" else 2
    sorted_cols = sorted(to_show.columns, key=lambda x: (int(x.split('_')[index]), x.split('_')[index + 1]))
    to_show = to_show[sorted_cols]

    # rename index using ensmbl to gene symbol
    to_show.index = [ensmus_dict[gene] for gene in to_show.index]

    # save
    orig_sorted_cols = [col.split("_")[0] for col in sorted_cols]
    orig_data = orig_data[orig_sorted_cols]
    orig_data.index = [ensmus_dict[gene] for gene in orig_data.index]
    orig_data.to_csv(f"./Private/{mice_type}_{genes_type}_data_{type_}.csv")
    to_show.to_csv(f"./Private/{mice_type}_{genes_type}_data_{type_}_z-scored.csv")

    if genes_type in v_max_dict[mice_type]:
        max_val = v_max_dict[mice_type][genes_type]
        min_val = v_min_dict[mice_type][genes_type]
        heatmap = sns.heatmap(to_show, cmap="coolwarm", vmax=max_val, vmin=min_val)
    else:
        heatmap = sns.heatmap(to_show, cmap="coolwarm")
    # show xticks every 4 columns, starting from 1, and show only .split('_')[2]+" "+.split('_')[3]
    plt.xticks(np.arange(2, len(to_show.columns), 4),
               [str(int(col.split('_')[index]) - 6) + " " + col.split('_')[index + 1] for col in
                to_show.columns[::4]], rotation=45)
    # show only half of the bar plot values
    # Get the colorbar object from the heatmap
    cbar = heatmap.collections[0].colorbar
    # Get current colorbar limits
    cmin, cmax = cbar.norm.vmin, cbar.norm.vmax
    if cmax >= 2:
        new_ticks = [cmin, 0, cmax / 2, cmax]
    else:
        new_ticks = [cmin, 0, cmax]
    # Set the new ticks
    cbar.set_ticks(new_ticks)

    # sns.clustermap(to_show, cmap="coolwarm", col_cluster=False, row_cluster=False, z_score=0)
    width = 2.5
    for i in range(1, 5):
        # draw a line between pbs and abx
        plt.axvline(x=8 * i - 4, color="white", linewidth=width, linestyle="--")
        # draw a dashed line between time points
        plt.axvline(x=8 * i, color="white", linewidth=width)
    if vert_locs:
        # draw a vertical line between maintained and changed genes
        for loc in vert_locs:
            plt.axhline(y=loc, color="white", linewidth=width)
    if save_df:
        combine.loc[genes].to_csv(f"./Private/{mice_type}_{genes_type}_genes.csv")
    # sns.clustermap(combine.loc[maintained_genes], cmap="coolwarm", col_cluster=False, row_cluster=False)
    plt.xlabel("Time point")
    plt.ylabel("Gene")
    plt.title(f"{mice_type} {genes_type} genes")
    plt.savefig(f"./Private/{mice_type}_{genes_type}_heatmap_{type_}.png", bbox_inches="tight")
    plt.show()
    plt.close()


def unite_go_clusters(median=True):
    data_spf, meta_spf, categories_spf, genes_dict_spf = run_prep(is_gf=False, condition=None, median=median,
                                                                  clock_genes=False, get_categories=True)
    data_gf, meta_gf, categories_gf, genes_dict_gf = run_prep(is_gf=True, condition=None, median=median,
                                                              clock_genes=False, get_categories=True)

    all_clusters = set(categories_gf).union(set(categories_spf))
    genes_dict = genes_dict_gf
    genes_dict.update(genes_dict_spf)
    for exp_type in ["gf", "spf"]:
        data = data_gf if exp_type == "gf" else data_spf
        meta_data = meta_gf if exp_type == "gf" else meta_spf

        genes_df = data
        for condition in ["Vanco", "PBS"]:
            temp = meta_data[meta_data["Drug"] == condition].reset_index()
            # temp["ID"] = temp["ID"].astype(str) + "_" + temp["Time_hr"].astype(str) + "_" + temp["Drug"].astype(str)
            mice = temp["ID"].values
            to_save = genes_df[mice]
            # add for each mouse ID the time and the drug
            # to_save.columns = [f"{mice[i]}_{temp['Time_hr'][i]}_{temp['Drug'][i]}" for i in range(len(mice))]
            if median:
                names = get_clusters()
                # update y labels from index to index + "\n" + to_save["name"].astype(str)
                to_save["name"] = to_save.apply(lambda x: names[x.name.split("_")[0]], axis=1)
                to_save.index = to_save.index  # + "_" + to_save["name"].astype(str)
                to_save = to_save.drop(columns=["name"])
            to_save.to_csv(f"./Private/{exp_type.upper()}/{exp_type.upper()} data {condition}.txt", sep='\t')
            # save data.index to file and ignore the index of the line
            pd.Series(genes_df.index).to_csv(f"./Private/{exp_type.upper()}/{exp_type.upper()} annot.txt",
                                             sep='\t', index=False)


def plot_clock_genes_all(ensmus_clock_genes):
    cols_spf = ['24_S24_5_Vanco', '23_S23_5_Vanco', '22_S22_5_Vanco', '21_S21_5_Vanco',
                '20_S20_5_PBS', '19_S19_5_PBS', '18_S18_5_PBS', '17_S17_5_PBS',
                '32_S32_11_Vanco', '29_S29_11_Vanco', '30_S30_11_Vanco', '31_S31_11_Vanco',
                '28_S28_11_PBS', '27_S27_11_PBS', '26_S26_11_PBS', '25_S25_11_PBS',
                '8_S8_17_Vanco', '7_S7_17_Vanco', '6_S6_17_Vanco', '5_S5_17_Vanco',
                '4_S4_17_PBS', '3_S3_17_PBS', '2_S2_17_PBS', '1_S1_17_PBS',
                '14_S14_23_Vanco', '13_S13_23_Vanco', '15_S15_23_Vanco', '16_S16_23_Vanco',
                '12_S12_23_PBS', '11_S11_23_PBS', '10_S10_23_PBS', '9_S9_23_PBS']
    cols_gf = ['10E_S17_5_PBS', '11E_S18_5_PBS', 'GF39_S19_5_PBS', 'GF40_S20_5_PBS',
               '13E_S24_5_Vanco', 'GF20_S23_5_Vanco', 'GF19_S22_5_Vanco', 'GF18_S21_5_Vanco',
               '14E_S25_11_PBS', '15E_S26_11_PBS', '16E_S27_11_PBS', '17E_S28_11_PBS',
               'GF49_S32_11_Vanco', 'GF48_S31_11_Vanco', 'GF47_S30_11_Vanco', 'GF26_S29_11_Vanco',
               '3E_S4_17_PBS', 'GF29_S3_17_PBS', 'GF28_S2_17_PBS', 'GF27_S1_17_PBS',
               'GF8_S6_17_Vanco', 'GF31_S8_17_Vanco', 'GF5_S5_17_Vanco', 'GF30_S7_17_Vanco',
               '6E_S12_23_PBS', 'GF34_S11_23_PBS', 'GF33_S10_23_PBS', 'GF32_S9_23_PBS',
               'GF12_S13_23_Vanco', 'GF13_S14_23_Vanco', 'GF14_S15_23_Vanco', 'GF37_S16_23_Vanco']
    cols_spf = [s.split('_', 1)[1] for s in cols_spf]
    cols_gf = [s.split('_', 1)[1] for s in cols_gf]
    for mice_type in ["SPF", "GF"]:
        # read original data
        data, meta, categories, genes_dict = run_prep(is_gf=mice_type == "GF", condition=None, median=False,
                                                      clock_genes=False, get_categories=True)
        if mice_type == "GF":
            data.columns = [s.split('_', 1)[1] for s in data.columns]
            meta.ID = meta.ID.str.replace("GF_", "")
            gf_order = ["nr1d1", "dbp", "clock", "cry1", "cry2", "ciart", "per1", "per2",  # "chrono" is "ciart"?
                        "nfil3", "arntl", ]
            circadian_clock_genes = [gene.capitalize() for gene in gf_order]
            ensmus_dict = get_ensmus_dict()
            # reverse this dictionary
            names_dict = {v: k for k, v in ensmus_dict.items()}
            ensmus_clock_genes = list(names_dict[gene] for gene in circadian_clock_genes if gene in names_dict)
        combine = data.loc[ensmus_clock_genes]
        cols = cols_spf if mice_type == "SPF" else cols_gf
        combine = combine[cols]

        meta_abx = meta.loc[meta["Drug"] == "Vanco"]
        meta_pbs = meta.loc[meta["Drug"] == "PBS"]
        plot_genes_heatmap(combine, ensmus_clock_genes, mice_type, "clock", meta_abx, meta_pbs, False,
                           "clock")


def clock_genes_phase():
    circadian_clock_genes = ["nr1d1", "dbp", "clock", "cry1", "cry2", "ciart", "per1", "per2",  # "chrono" is "ciart"?
                             "nfil3", "arntl", ]  # "bmal1" is arntl. autophagy: "atg9a", "atg13", "map1lc3a"
    # capitalize circadian_clock_genes
    circadian_clock_genes = [gene.capitalize() for gene in circadian_clock_genes]
    ensmus_dict = get_ensmus_dict()
    # reverse this dictionary
    names_dict = {v: k for k, v in ensmus_dict.items()}
    ensmus_clock_genes = list(names_dict[gene] for gene in circadian_clock_genes if gene in names_dict)
    plot_clock_genes_all(ensmus_clock_genes)
    clock = {"SPF": {"clock": ensmus_clock_genes}, "GF": {"clock": ensmus_clock_genes}}
    plot_all_data(clock, False)


def enrichment_analysis_tp(dict):
    excel_limit = 32745
    zt = ["ZT-1", "ZT5", "ZT11", "ZT17"]
    from random_forests import check_enrichment
    ensmus_dict = get_ensmus_dict()
    for mice_type in ["SPF", "GF"]:
        data, _, _, _ = run_prep(is_gf=mice_type == "GF", condition=None, median=False,
                                 clock_genes=False, get_categories=True)
        background = data.index.values
        background = [str(ensmus_dict[gene]) for gene in background if gene in ensmus_dict]
        enrichment_analysis = pd.DataFrame(
            columns=['Gene_set', 'Term', 'P-value', 'Adjusted P-value', 'Old P-value',
                     'Old adjusted P-value', 'Odds Ratio', 'Combined Score', 'Genes', "category", "# of_genes",
                     "All_Genes"])
        mice_dict = dict[mice_type]
        for category in mice_dict:
            check = [str(ensmus_dict[gene]) for gene in list(mice_dict[category]) if gene in ensmus_dict]
            enrichment = check_enrichment(check, background)
            category_name = category.split("_")[0] + "_" + zt[int(category.split("_")[1])] if ("_" in category and \
                                                                                               category.split("_")[
                                                                                                   1].isdigit()) else category
            cell_content = ";".join(
                [str(ensmus_dict[gene]) for gene in mice_dict[category] if isinstance(gene, str)]
            )
            if enrichment is not None:
                # print(enrichment)
                enrichment["category"] = category_name
                enrichment["# of_genes"] = enrichment["Genes"].str.count(";") + 1
                enrichment["All_Genes"] = cell_content if len(cell_content) < excel_limit else cell_content[
                                                                                               :excel_limit - 3] + "..."
                enrichment_analysis = pd.concat([enrichment_analysis, enrichment])

        # enrichment_analysis = enrichment_analysis.drop("index", axis=1)
        enrichment_analysis = enrichment_analysis.drop(["Old P-value", "Old adjusted P-value"], axis=1)
        enrichment_analysis["adj.P-val<5%"] = enrichment_analysis["Adjusted P-value"] < 0.05
        enrichment_analysis["P-val<5%"] = enrichment_analysis["P-value"] < 0.05
        enrichment_analysis.to_csv(
            f"./Private/time points/{mice_type}_enrichment_{'background' if background else ''}.csv",
            index=False)
        enrichment_analysis[enrichment_analysis["Adjusted P-value"] < 0.05].to_csv(
            f"./Private/time points/{mice_type}_enrichment_{'background' if background else ''}_filtered.csv",
            index=False)


def save_dictionary(categories_dict, txt=False):
    ensmus_dict = get_ensmus_dict()
    zt = ["ZT-1", "ZT5", "ZT11", "ZT17"]
    if txt:
        with open(f"./Private/categories_dict_hour.txt", "w") as f:
            for mice in categories_dict:
                # f.write(mice + "\n")
                # sort categories_dict[mice] lexico-graphically
                for category in sorted(categories_dict[mice]):
                    category_name = category.split("_")[0] + "_" + zt[int(category.split("_")[1])] if (
                            "_" in category and \
                            category.split("_")[
                                1].isdigit()) else category
                    # cell_content = "\n".join(
                    #     [str(ensmus_dict[gene]) for gene in categories_dict[mice][category] if isinstance(gene, str)]
                    # )
                    cell_content = str(len(categories_dict[mice][category]))
                    f.write(mice + "-" + category_name + "\n")
                    f.write(cell_content + "\n")
    else:
        import csv

        with open("./Private/time points/categories_dict_hour.csv", "w", newline="") as f:
            writer = csv.writer(f)

            # Write header row
            writer.writerow(["mice_type", "category", "gene"])

            # Write data
            for mice in categories_dict:
                for category in categories_dict[mice]:
                    category_name = category.split("_")[0] + "_" + zt[int(category.split("_")[1])] if (
                            "_" in category and category.split("_")[1].isdigit()
                    ) else category

                    for gene in categories_dict[mice][category]:
                        if isinstance(gene, str):  # Ensure gene is a string
                            writer.writerow([mice, category_name, ensmus_dict[gene]])


def get_significance(intersecting, all_spf_genes, all_gf_genes, num_spf, num_gf):
    from scipy.stats import hypergeom

    """
        Calculate the statistical significance of the intersection between SPF and GF gene sets.

        Parameters:
        intersecting (set): Genes common to both SPF and GF sets.
        all_spf_genes (pd.Index): All genes in SPF.
        all_gf_genes (pd.Index): All genes in GF.
        num_spf (int): Number of genes in the SPF gene set.
        num_gf (int): Number of genes in the GF gene set.

        Returns:
        float: p-value representing the significance of the overlap.
        """
    total_genes = len(set(all_spf_genes).union(set(all_gf_genes)))
    overlap_size = len(intersecting)

    # Hypergeometric test parameters
    M = total_genes  # Total population size
    n = num_spf  # Number of SPF genes
    N = num_gf  # Number of GF genes
    x = len(intersecting)  # Size of the specific intersection passed as argument

    # Compute p-value
    p_value = hypergeom.sf(x - 1, M, n, N)
    return p_value


def analyze_set_intersection(subset1, subset2, universe1, universe2):
    """
    Analyzes the statistical significance of the intersection between two sets,
    where each set is a subset of its own universe.

    Parameters:
    -----------
    subset1 : set
        First set of genes (e.g., spf_genes)
    subset2 : set
        Second set of genes (e.g., gf_genes)
    universe1 : set
        Universe set for subset1 (e.g., all_spf)
    universe2 : set
        Universe set for subset2 (e.g., all_gf)

    Returns:
    --------
    dict
        Contains p_value, odds_ratio, and intersection statistics
    """
    from scipy import stats
    # Verify subsets
    if not subset1.issubset(universe1) or not subset2.issubset(universe2):
        raise ValueError("Subsets must be contained within their respective universes")

    common_universe = universe1.intersection(universe2)
    intersecting = subset1.intersection(subset2)

    subset1_in_common = subset1.intersection(common_universe)
    subset2_in_common = subset2.intersection(common_universe)

    in_both = len(intersecting)
    in_set1_only = len(subset1_in_common - subset2)
    in_set2_only = len(subset2_in_common - subset1)
    in_neither = len(common_universe) - in_both - in_set1_only - in_set2_only

    contingency_table = np.array([[in_both, in_set1_only],
                                  [in_set2_only, in_neither]])

    odds_ratio, p_value = stats.fisher_exact(contingency_table)

    expected_overlap = (len(subset1_in_common) * len(subset2_in_common)) / len(common_universe)
    fold_enrichment = in_both / expected_overlap if expected_overlap > 0 else float('inf')

    # Get enrichment direction and significance
    significance_annotation = get_significance_annotation(p_value, in_both, expected_overlap)

    return {
        'p_value': p_value,
        'significance_annotation': significance_annotation,
        'odds_ratio': odds_ratio,
        'intersection_size': in_both,
        'expected_overlap': expected_overlap,
        'fold_enrichment': fold_enrichment,
        'subset1_size': len(subset1),
        'subset2_size': len(subset2),
        'universe1_size': len(universe1),
        'universe2_size': len(universe2),
        'common_universe_size': len(common_universe),
        'contingency_table': contingency_table.tolist()
    }


def get_significance_annotation(p_value, observed, expected):
    """
    Get significance stars and direction of enrichment.
    Returns both stars and a symbol indicating if intersection is higher (↑) or lower (↓) than expected.
    """
    # Get stars based on p-value
    if p_value < 0.0001:
        stars = "****"
    elif p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        return "ns"  # If not significant, don't show direction

    # Calculate padding to center the arrow above the stars
    half_stars_len = len(stars) / 2
    left_pad = " " * int(half_stars_len - 0.5)  # Subtract 0.5 to account for arrow width

    # Add direction arrow, centered above stars
    if observed > expected:
        return f"{left_pad}↑\n{stars}"  # Significantly higher
    else:
        return f"{left_pad}↓\n{stars}"  # Significantly lower


def time_intersections(all_dict):
    """
    Analyzes and visualizes intersections between SPF and GF genes across time points
    using matplotlib plots to create Venn diagrams with proportional sizes.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    all_spf = get_meta_data("SPF")[0].index
    all_gf = get_meta_data("GF")[0].index
    results = {}

    # Determine the maximum set size across all time points for consistent scaling
    all_set_sizes = []
    for direction in ["enhanced", "suppressed"]:
        for time in range(4):
            spf_genes = all_dict["SPF"][f"{direction}_{time}"]
            gf_genes = all_dict["GF"][f"{direction}_{time}"]
            all_set_sizes.append(len(spf_genes))
            all_set_sizes.append(len(gf_genes))

    # Function to create Venn diagram using plots
    def create_plot_venn(set1, set2, ax, set_labels=None):
        """
        Creates a Venn diagram as a scatter plot with consistent scaling.
        Parameters:
        - set1, set2: The two sets to compare
        - ax: Matplotlib axis to draw on
        - set_labels: Tuple of labels for the two sets
        Returns:
        - Dictionary with relevant statistics and plot elements
        """
        set1_size = len(set1)
        set2_size = len(set2)
        intersection = set1.intersection(set2)
        intersection_size = len(intersection)
        only_set1 = set1_size - intersection_size
        only_set2 = set2_size - intersection_size

        # Calculate circle sizes based on set sizes
        radius1 = np.sqrt(set1_size / np.pi) if set1_size > 0 else 0.1
        radius2 = np.sqrt(set2_size / np.pi) if set2_size > 0 else 0.1

        # Calculate overlap proportion to determine circle positioning
        total_size = set1_size + set2_size
        if total_size == 0 or intersection_size == 0:
            overlap_ratio = 0
            distance = (radius1 + radius2) * 2  # No overlap
        else:
            overlap_ratio = min(0.8, intersection_size / min(set1_size, set2_size))
            max_distance = radius1 + radius2
            min_distance = abs(radius1 - radius2)
            # Adjust distance to create proper visual overlap
            distance = max_distance - overlap_ratio * (max_distance - min_distance)
            # Ensure a minimum separation if overlap is very small
            if overlap_ratio < 0.1 and overlap_ratio > 0:
                distance = max_distance * 0.9

        # Generate points for circles
        center1 = (-distance / 2, 0)
        center2 = (distance / 2, 0)

        # Create scatter plots instead of patches
        # Generate points for SPF circle (set1)
        theta = np.linspace(0, 2 * np.pi, 100)
        x1 = center1[0] + radius1 * np.cos(theta)
        y1 = center1[1] + radius1 * np.sin(theta)

        # Generate points for GF circle (set2)
        x2 = center2[0] + radius2 * np.cos(theta)
        y2 = center2[1] + radius2 * np.sin(theta)

        # Plot filled circles
        ax.fill(x1, y1, '#3333FF', alpha=0.5, edgecolor='none')
        ax.fill(x2, y2, '#FFCC00', alpha=0.5, edgecolor='none')

        # Set labels
        if set_labels:
            ax.text(center1[0], -radius1 - 0.2, set_labels[0], ha='center', va='top', fontsize=12)
            ax.text(center2[0], -radius2 - 0.2, set_labels[1], ha='center', va='top', fontsize=12)

        # Add count labels
        # For first set unique elements
        ax.text(center1[0], 0, str(only_set1), ha='center', va='center', fontsize=12)
        # For second set unique elements
        ax.text(center2[0], 0, str(only_set2), ha='center', va='center', fontsize=12)
        # For intersection elements
        if intersection_size > 0:
            # Position the intersection label at the center of the overlapping region
            mid_x = (center1[0] * radius2 + center2[0] * radius1) / (radius1 + radius2)
            ax.text(mid_x, 0, str(intersection_size), ha='center', va='center', fontsize=12)

        # Return important values
        return {
            'center1': center1,
            'center2': center2,
            'radius1': radius1,
            'radius2': radius2,
            'distance': distance,
            'intersection_size': intersection_size
        }

    # Set global font size
    plt.rcParams.update({'font.size': 16})

    # Universal axis limits for consistent scaling
    limit = 46
    universal_x_min = -limit
    universal_x_max = limit
    universal_y_min = -limit
    universal_y_max = limit

    for direction in ["enhanced", "suppressed"]:
        results[direction] = {}
        for time in range(4):
            spf_genes = set(all_dict["SPF"][f"{direction}_{time}"])
            gf_genes = set(all_dict["GF"][f"{direction}_{time}"])

            # Analyze intersection
            stats = analyze_set_intersection(spf_genes, gf_genes, all_spf, all_gf)
            results[direction][time] = stats

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(4.5, 5))

            # Create Venn diagram
            venn_elements = create_plot_venn(
                spf_genes, gf_genes,
                ax,
                set_labels=("SPF", "GF")
            )

            # Set title
            plt.title(f"{direction.capitalize()} ZT {time * 6 - 1}\n"
                      f"(SPF and GF common genes: {stats['common_universe_size']})",
                      fontsize=16)

            # Add significance stars with improved positioning
            if venn_elements['intersection_size'] > 0:
                # If there's an intersection, place significance above the intersection count
                mid_x = (venn_elements['center1'][0] * venn_elements['radius2'] +
                         venn_elements['center2'][0] * venn_elements['radius1']) / (
                                venn_elements['radius1'] + venn_elements['radius2'])
                if stats['significance_annotation'] != "ns":
                    ax.text(mid_x, 0.4 * min(venn_elements['radius1'], venn_elements['radius2']),
                            stats['significance_annotation'],
                            ha='center', va='center', fontsize=12)
                else:
                    ax.text(mid_x, 0.4 * min(venn_elements['radius1'], venn_elements['radius2']),
                            stats['significance_annotation'],
                            ha='center', va='center', fontsize=9)
            else:
                # If there's no intersection, place stars between the two circles
                mid_x = (venn_elements['center1'][0] + venn_elements['center2'][0]) / 2
                mid_y = (venn_elements['center1'][1] + venn_elements['center2'][1]) / 2
                ax.text(mid_x, mid_y, stats['significance_annotation'],
                        ha='center', va='center', fontsize=12)

            # Use universal axis limits for consistency
            ax.set_aspect('equal')
            ax.set_xlim(universal_x_min, universal_x_max)
            ax.set_ylim(universal_y_min, universal_y_max)

            # Hide axes
            ax.axis('off')
            # Make the plot take up more of the figure area by reducing margins
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

            # set_plot_defaults()
            # plt.tight_layout()
            plt.savefig(f"./Private/{direction}{time * 6 - 1}_venn.png", dpi=300)
            plt.savefig(f"./Private/{direction}{time * 6 - 1}_venn.svg", dpi=300)
            plt.show()
            plt.close()

    return results


if __name__ == "__main__":
    # run_spf(to_cluster=True, plot=True, intersect=False)
    # run_gf(to_cluster=True, plot=True, intersect=False)
    # quit()

    unite_go_clusters(False)
    find_all_cycles()
    # categories_dict = detect_change()
    # # plot_all_data(categories_dict, True)
    # plot_heatmap(categories_dict, "total", False, False)
    categories_dict = detect_change_hour()
    save_dictionary(categories_dict, txt=True)
    # enrichment_analysis_tp(categories_dict)
    # plot_all_data(categories_dict, False)
    plot_heatmap(categories_dict, "hour", False, False)

    clock_genes_phase()
    time_intersections(categories_dict)
