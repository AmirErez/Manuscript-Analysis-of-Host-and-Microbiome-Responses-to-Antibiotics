import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from goatools import obo_parser
from scipy.stats import linregress

from ClusteringGO import (antibiotics, treatments, get_ancestor, get_go, private, path, set_plot_defaults)


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


def plot_correlation(df, title, x_name, y_name, folder=""):
    df = df[df['size'] < 800]
    # plt.scatter(df[x_name], df[y_name], cmap='viridis')
    plt.hist2d(df[x_name], df[y_name], bins=10, cmap='viridis')
    # plt.title(f"{title}{abx} {treat} {x_name} vs {y_name}")
    plt.xlabel(x_name.strip('\"'))
    plt.ylabel(y_name)
    x_name = x_name.strip("\"")
    y_name = y_name.strip("\"")
    plt.savefig(os.path.join("Private", f"{folder}{title}_{x_name}_{y_name}.png"))
    plt.show()
    plt.close()


def get_to_axis(axis, i, j, n, m):
    if n > 1 and m > 1:
        return axis[i, j]
    else:
        return axis[max(i, j)]


def set_figure(treats, antibiotics, cols_factor=6.0, rows_factor=5.0):
    rows, cols = len(antibiotics), len(treats)
    fig, axis = plt.subplots(rows, cols, figsize=(cols_factor * cols, rows_factor * rows))
    fig.tight_layout(pad=1.5)
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
    colors_file_path = os.path.join("Private", "colors_dict.txt")

    if os.path.exists(colors_file_path):
        # if False:
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
                    regular=True, gsea=False):
    size = 10
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

    set_plot_defaults()
    colors = get_colors_dictionary(all_go)
    enrichment = np.zeros((len(antibiotics), len(treatments)))
    axis = set_figure(treatments, antibiotics, cols_factor=4 / len(treatments), rows_factor=8 / len(antibiotics))
    for j, treat in enumerate(treatments):
        for i, abx in enumerate(antibiotics):
            curr_axis = get_to_axis(axis, i, j, len(treatments), len(antibiotics))
            curr_axis.set_title(f"{abx}, {treat}", size=size)  # , weight="bold")
            first_loc, second_loc = 0, 0.8
            enhance_enrichment = plot_bar(curr_axis, colors, counts_dict_enhanced[treat][abx], first_loc)
            suppressed_enrichment = plot_bar(curr_axis, colors, counts_dict_suppressed[treat][abx], second_loc)
            curr_axis.set_xticks([first_loc, second_loc], ["Enh.", "Supp."])
            curr_axis.set_xlim(first_loc - 0.4, second_loc + 0.4)
            curr_axis.tick_params(axis='both', labelsize=size - 2)
            # curr_axis.tick_params(axis='x', labelsize=size-2)
            curr_axis.set_xticklabels(curr_axis.get_xticklabels())  # , rotation=90)
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
    # lower_center.legend(handles, labels, loc=loc, bbox_to_anchor=anchor, fontsize=size)
    # plt.suptitle(f"Categories of GO terms", fontsize=30)
    curr_path = os.path.join(".", "Private", "analysis")
    plt.savefig(os.path.join(curr_path, exp_type[1:], f"{exp_type[1:]} categories.png"), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(curr_path, exp_type[1:], f"{exp_type[1:]} categories.svg"), bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    # Create a separate figure just for the legend
    fig_legend = plt.figure(figsize=(3, 4), dpi=300)
    # Add an empty plot to attach the legend
    ax = fig_legend.add_subplot(111)
    ax.axis('off')  # Hide the empty plot's axes
    # Create the legend in this separate figure
    legend = ax.legend(handles, labels, loc='center', fontsize=size)
    # Show the legend-only figure
    plt.savefig(os.path.join(curr_path, exp_type[1:], f"{exp_type[1:]} categories legend.svg"), bbox_inches='tight')
    plt.show()

    suppressed = plot_enrichment(antibiotics, treatments, exp_type, counts_dict_suppressed, orig_labels, "suppressed")
    enhanced = plot_enrichment(antibiotics, treatments, exp_type, counts_dict_enhanced, orig_labels, "enhanced")

    return enhanced, suppressed, orig_labels


def plot_enrichment(antibiotics, treatments, param, dict, categories, title):
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
    linebreak_cutoff = 36
    for i, col in enumerate(df.columns):
        if len(col) > linebreak_cutoff:
            df = df.rename(columns={col: f"{col[:linebreak_cutoff]}\n{col[linebreak_cutoff:]}"})

    # create a figure of size 3*3 inches, 180 dots per inch
    set_plot_defaults()
    plt.figure(figsize=(4, 4), dpi=300)
    # plt.figure(figsize=(10, 8), dpi=180)

    # Plot a heatmap
    # # sort the columns lexicographically
    # df = df.reindex(sorted(df.columns), axis=1)
    # sort the rows lexicographically, with Mix at the end
    index = [ind for ind in list(df.index) if "Mix" not in ind]
    # sort lexically
    index.sort()
    if type(treatments[0]) != int:
        # add Mix-IP, Mix-IV, Mix-PO at the end
        index.append("Mix-IP")
        index.append("Mix-IV")
        index.append("Mix-PO")
    df = df.reindex(index)
    df = df.T
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
        heatmap = sns.heatmap(df, cmap="GnBu")
    # remove y axis label
    plt.ylabel('')
    # Rotate the x-axis labels by 45 degrees
    # plt.xticks(rotation=45)
    plt.title(f"Enrichment of GO terms\n{param[9:]} {title}")
    # plt.savefig(private + f"analysis/{param}/ enrichment {title}.png", bbox_inches='tight')
    # Ensure all ticks are shown
    heatmap.set_yticks(np.arange(len(df.index)) + 0.5)  # One tick per row
    heatmap.set_yticklabels(df.index, fontsize=6)  # Force the label size
    heatmap.set_xticks(np.arange(len(df.columns)) + 0.5)  # One tick per row
    heatmap.set_xticklabels(df.columns, fontsize=8)  # Force the label size
    plt.gca().yaxis.set_tick_params(pad=5)  # Add some space
    # # show all x and y labels
    # plt.rc('xtick', labelsize=8)
    # plt.rc('ytick', labelsize=6)
    # plt.tight_layout()

    plt.savefig(os.path.join(private, "analysis", param[1:], f"enrichment {title}.png"), bbox_inches='tight')
    plt.show()
    plt.close()
    # save df to csv
    df.to_csv(os.path.join(private, "analysis", param[1:], f"enrichment {param[9:]} {title}.csv"))
    return df


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

    elif regular:
        selected = df[(df['treat-test p-value'] < 0.05) & (df['size'] >= 2) &
                      (df['p-value correlation'] <= 0.05)]
    else:
        selected = df[(df['treat-test p-value'] < 0.05) & (df['size'] >= 2) &
                      (df['p-value correlation'] <= 0.05)]
    return selected


def get_selected_gsea(abx, treat, go):
    go_dict = create_go_term_dict(go)
    selected = pd.DataFrame()
    for folder in os.listdir(os.path.join(private, "GSEA")):
        if folder.startswith(f"{abx}{treat}"):
            # read the csv file that starts with gsea_report_for_1
            for file in os.listdir(os.path.join(private, "GSEA", folder)):
                if file.startswith("gsea_report_for") and file.endswith(".tsv"):
                    results = pd.read_csv(os.path.join(private, "GSEA", folder, file), sep="\t")
                    addition = "_enh" if "_1_" in file else "_sup"
                    # save this table to a csv file with the name abx_treat_GSEA_addition.csv
                    results.to_csv(os.path.join(private, "GSEA", "all_results", f"{abx}_{treat}_GSEA{addition}.tsv"),
                                   sep="\t",
                                   index=False)
                    # keep only rows where FDR q-val < 0.05
                    results = results[results['FDR q-val'] < 0.05]
                    results['GO term'] = results['NAME'].apply(lambda x: map_term_to_go_id(x, go_dict))
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
        go_term.name.lower().replace("-", " "): go_term.id
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


def plot_tsne(data, condition, pca=True, perplexity=7, pca_components=50, title=''):
    """
    hopefully: control are clustered together
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=perplexity,
                method='exact')
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


def plot_significant_genes_number(meta, raw, antibiotics, treatments, param, condition="Treatment"):
    import pickle
    # import matplotlib
    # matplotlib.use('Agg')

    # import venn
    threshold = 0.05
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
    plt.savefig(os.path.join(private, "analysis", param, "genes stats.svg"), bbox_inches='tight')
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # plt.savefig(buf, format='svg')
    buf.seek(0)

    # Use Pillow to save as TIFF
    with Image.open(buf) as img:
        img.save(os.path.join(private, "analysis", param, "genes stats.tiff"), format='TIFF')
    plt.show()


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


def plot_kde(x, y, shape, legend, jitter=0.01, point_size=20):
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
        if jitter != 0.02:
            plot_kde(x, y, legend=legend, shape=shape, jitter=0.02)
        return

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x_jittered, y_jittered, z = x_jittered[idx], y_jittered[idx], z[idx]
    shapes = np.array(shape)[idx]
    # Compute correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(x, y)
    # slope, intercept = np.polyfit(x, y, 1)
    slope = np.sum(x * y) / np.sum(x ** 2)
    correlation_text = f'Correlation: {correlation_coefficient:.2f}\nP-value: {p_value:.2e}\nSlope: {slope:.2f}'

    fig, ax = plt.subplots()
    # scatter = ax.scatter(x_jittered, y_jittered, c=z, s=point_size, cmap='viridis', norm=LogNorm())
    nique_shapes = np.unique(shapes)
    for shape in np.unique(shapes):
        mask = shapes == shape
        scatter = ax.scatter(x_jittered[mask], y_jittered[mask], c=z[mask], s=point_size,
                             cmap='viridis', norm=LogNorm(), marker=shape, label=legend[shape])
    plt.colorbar(scatter, ax=ax, label='Density (log scale)')

    # Add correlation and p-value text to the plot
    ax.text(0.55, 0.25, correlation_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.6))
    ax.legend(title="Shapes")


def clusters_compare_mix(antibiotics, treatments, param):
    mix = "Mix"
    no_mix = [abx for abx in antibiotics if abx != mix]
    for treat in treatments:
        for to_mix in [True, False]:
            title = f"selected by {'mix' if to_mix else 'other abx'} threshold"
            compare_mix_single(mix, no_mix, param, treat, "correlation",
                               f"mean pairwise correlation of clusters in abx_{treat}",
                               f"mean pairwise correlation of clusters in mix_{treat}",
                               f"mean_pairwise_correlation_{treat}{'_to_mix' if to_mix else ''}", to_mix, title,
                               jitter=0)
            compare_mix_single(mix, no_mix, param, treat, "p-value correlation",
                               f"mean pairwise -log(p-value) correlation of clusters in abx_{treat}",
                               f"mean pairwise -log(p-value) of clusters in mix_{treat}",
                               f"p_val_correlation_{treat}{'_to_mix' if to_mix else ''}", to_mix, title, jitter=0,
                               log=True, minus=True)
            compare_mix_single(mix, no_mix, param, treat, "GO significance",
                               f"-log(p-value) of enrichment of significant\n genes clusters in abx_{treat}",
                               f"-log(p-value) of enrichment of significant\n genes clusters in mix_{treat}",
                               f"go_p_val_{treat}{'_to_mix' if to_mix else ''}", to_mix, title, jitter=0,
                               log=True, minus=True)
            compare_mix_single(mix, no_mix, param, treat, "relative size",
                               f"relative size of clusters in abx_{treat}", f"relative size of clusters in mix_{treat}",
                               f"relative_size_{treat}{'_to_mix' if to_mix else ''}", to_mix, title)
            compare_mix_single(mix, no_mix, param, treat, "mean variance between samples",
                               f"mean variance between samples in abx_{treat}",
                               f"mean variance between samples of clusters in mix_{treat}",
                               f"variance_{treat}{'_to_mix' if to_mix else ''}", to_mix, title)


def compare_mix_single(mix, no_mix, param, treat, col, xlabel, ylabel, title, to_mix, by, log=False, minus=True,
                       jitter=0.01):
    plt.figure(figsize=(10, 10))
    selected_log_distances = []
    compare_mix_log_distances = []
    markers = {no_mix[0]: 'o', no_mix[1]: '^', no_mix[2]: 's', no_mix[3]: 'D'}
    opposite = {value: key for key, value in markers.items()}
    shapes = []
    for abx in no_mix:
        compare_mix, selected = get_selected_df_plot_mix(abx, mix, param, treat) if to_mix else get_selected_df_plot(
            abx, mix, param, treat)
        # verify same order of clusters in both DFs
        assert sorted(selected["GO term"].values) == sorted(compare_mix["GO term"].values)
        print(treat, col, len(selected["GO term"]))
        # plot log(distance) of the clusters in abx_treat and mix_treat in a scatter plot
        # plt.scatter(selected["\"log(distance)\""], compare_mix["\"log(distance)\""])
        # Append log(distance) values from both DataFrames
        shapes.extend([markers[abx]] * len(selected[col]))
        if log:
            if minus:
                # replace the zeros from selected[col] with 1e-10 and also from compare_mix[col]
                selected[col] = selected[col].replace(0, 1e-10)
                compare_mix[col] = compare_mix[col].replace(0, 1e-10)
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
                # if "p-value" in xlabel:
                #     print(abx, treat, selected[col].min(), "Mix", compare_mix[col].min())
    df = pd.DataFrame({
        'Selected': selected_log_distances,
        'Compare Mix': compare_mix_log_distances,
        'Shape': shapes,
    })
    df_filtered = df.dropna(subset=['Selected', 'Compare Mix']).reset_index(drop=True)
    if len(df_filtered) != len(df):
        print(title, f"dropped {len(df) - len(df_filtered)} nans")
        df = df_filtered
    # sns.kdeplot(data=df, x='Selected', y='Compare Mix', cmap="YlGnBu", shade=True, cbar=True)
    plot_kde(x=df['Selected'], y=df['Compare Mix'], shape=df['Shape'], jitter=jitter, legend=opposite)

    x_limits = {
        "go": (-0.1, 3),
        "p-value correlation": (-0.1, 1.1),
        "GO significance": (-0.1, 1.1),
    }
    y_limits = x_limits
    if col not in x_limits:
        min_val = 0
        max_val = 1.1
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
    elif "p-value" in col:  # or
        min_val = -0.1
        max_val = 3.5
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
    elif "significance" in col:  # or
        min_val = -0.1
        max_val = 5
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
    else:
        min_val = df['Selected'].min()
        max_val = df['Selected'].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Comparison of clusters in abx_{treat} and {mix}_{treat},\n{by}")

    plt.tight_layout()
    # if ./Private/met_comp does not exist, create it
    if not os.path.exists("./Private/met_comp"):
        os.makedirs("./Private/met_comp")
    plt.savefig(f"./Private/met_comp/{title}.png")
    plt.savefig(f"./Private/met_comp/{title}.svg")
    # plt.show()
    plt.close()


def get_selected_df_plot(abx, mix, param, treat):
    # read the clusters data for abx_treat and mix_treat
    df = pd.read_csv(os.path.join("data", "clusters_properties", f"top_correlated_GO_terms_{abx}_{treat}.tsv"), sep="\t")
    df_mix = pd.read_csv(os.path.join("data", "clusters_properties", f"top_correlated_GO_terms_{mix}_{treat}.tsv"), sep="\t")
    # get the selected clusters for abx_treat
    # selected = df[(df['treat-test p-value'] <= 0.05) & (df['size'] >= 2) & (
    #         df['p-value distance'] <= 0.05)]
    selected = df[(df['fdr correlation'] < 0.05)]

    # df['better than parent'] is not False) & (df['better than random'] is not False)]
    # get the selected clusters for mix_treat
    compare_mix = df_mix[df_mix["GO term"].isin(selected["GO term"])]
    # keep in selected only the clusters that are in compare_mix
    selected = selected[selected["GO term"].isin(compare_mix["GO term"])]
    return compare_mix, selected


def get_selected_df_plot_mix(abx, mix, param, treat):
    # read the clusters data for abx_treat and mix_treat
    df_other = pd.read_csv(os.path.join("data", "clusters_properties", f"top_correlated_GO_terms_{abx}_{treat}.tsv"),
                           sep="\t")
    df = pd.read_csv(os.path.join("data", "clusters_properties", f"top_correlated_GO_terms_{mix}_{treat}.tsv"),
                     sep="\t")
    # get the selected clusters for abx_treat
    selected = df[(df['fdr correlation'] < 0.05)]
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
    set_plot_defaults()
    plt.figure(figsize=(5, 5))
    # plt.figure(figsize=(3, 3))
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
    plt.savefig(private + f"/analysis/{param}/{x}_vs_{y}.svg")
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
    set_plot_defaults()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # fig, axes = plt.subplots(2, 2, figsize=(3, 3))
    fig.suptitle("Pearson and Spearman Correlations for Enhanced and Suppressed", fontsize=14)
    labels_order = gsea[0].columns
    # Iterate over directions and plot correlations
    for j, direction in enumerate(["enhanced", "suppressed"]):
        df1 = gsea[j][labels_order]
        df2 = our[j][labels_order]

        # Dictionaries to store row and column correlations
        correlations = {"row": {"Pearson": [], "Spearman": []}, "column": {"Pearson": [], "Spearman": []}}
        random_correlations = {"row": {"Pearson": [], "Spearman": []}, "column": {"Pearson": [], "Spearman": []}}
        p_values = {"row": {"Pearson": [], "Spearman": []}, "column": {"Pearson": [], "Spearman": []}}
        shuffle_values = {"row": {"Pearson": [], "Spearman": []}, "column": {"Pearson": [], "Spearman": []}}

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
            orig_columns1 = df1.columns
            orig_columns2 = df2.columns
            orig_index1 = df1.index
            orig_index2 = df2.index
            for i in range(shuffles):
                # get shuffle order of columns
                shuffle1 = np.random.permutation(orig_columns1)
                shuffle2 = np.random.permutation(orig_columns2)
                shuffled_df1 = df1[shuffle1]
                shuffled_df2 = df2[shuffle2]
                shuffled_df1.columns = orig_columns1
                shuffled_df2.columns = orig_columns2
                temp_col_random_correlations[i] = shuffled_df1.corrwith(shuffled_df2, method=method).mean()
                # # Shuffle rows and compute the mean of row-wise correlations
                # shuffled_df1 = df1.sample(frac=1, replace=False).reset_index(drop=True)
                # shuffled_df2 = df2.sample(frac=1, replace=False).reset_index(drop=True)
                # temp_col_random_correlations[i] = shuffled_df1.corrwith(shuffled_df2, method=method).mean()

                shuffle1 = np.random.permutation(orig_index1)
                shuffle2 = np.random.permutation(orig_index2)
                shuffled_df1 = df1.loc[shuffle1]
                shuffled_df2 = df2.loc[shuffle2]
                shuffled_df1.index = orig_index1
                shuffled_df2.index = orig_index2
                temp_row_random_correlations[i] = shuffled_df1.corrwith(shuffled_df2, method=method).mean()
                # # Shuffle columns in the same way as rows
                # shuffled_df1 = df1.T.sample(frac=1, replace=False).reset_index(drop=True)
                # shuffled_df2 = df2.T.sample(frac=1, replace=False).reset_index(drop=True)
                # temp_row_random_correlations[i] = shuffled_df1.corrwith(shuffled_df2, method=method).mean()

            random_correlations["row"][method.capitalize()] = [np.nanmean(temp_row_random_correlations),
                                                               np.nanstd(temp_row_random_correlations)]
            random_correlations["column"][method.capitalize()] = [np.nanmean(temp_col_random_correlations),
                                                                  np.nanstd(temp_col_random_correlations)]

            p_values["row"][method.capitalize()] = [
                ((np.sum(temp_row_random_correlations >= row) + 1) / (shuffles + 1)) for row in row_correlations]
            p_values["column"][method.capitalize()] = [
                ((np.sum(temp_col_random_correlations >= col) + 1) / (shuffles + 1)) for col in col_correlations]
            shuffle_values["row"][method.capitalize()] = [temp_row_random_correlations for _ in row_correlations]
            shuffle_values["column"][method.capitalize()] = [temp_col_random_correlations for _ in col_correlations]

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
        pearson_boxplot = ax.boxplot(shuffle_values["row"]["Pearson"], positions=x - width / 2, widths=width / 2,
                                     patch_artist=True, boxprops=dict(facecolor=colors[0], alpha=0.3, color=colors[0]))
        spearman_boxplot = ax.boxplot(shuffle_values["row"]["Spearman"], positions=x + width / 2, widths=width / 2,
                                      patch_artist=True, boxprops=dict(facecolor=colors[1], alpha=0.3, color=colors[1]))
        ax.scatter(x - width / 2, correlations["row"]["Pearson"], label="Pearson", color=colors[0], s=50,
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.scatter(x + width / 2, correlations["row"]["Spearman"], label="Spearman", color=colors[1], s=50,
                   edgecolor="black", linewidth=0.5, zorder=3)
        # add ns, *, **, etc. for each of the bars in the bar plot in comparison to the shuffles
        # Add significance indicators for Pearson correlations
        top_whisker_y = np.array([max(whisker.get_ydata()) for whisker in pearson_boxplot['whiskers'][1::2]])
        if not np.isnan(top_whisker_y.any()):
            add_significance_indicators(
                ax, x - width / 2,
                [max(correlations["row"]["Pearson"][i], top_whisker_y[i]) for i in range(len(x))],
                p_values["row"]["Pearson"],
            )

        # Add significance indicators for Spearman correlations
        top_whisker_y = np.array([max(whisker.get_ydata()) for whisker in spearman_boxplot['whiskers'][1::2]])
        if not np.isnan(top_whisker_y.any()):
            add_significance_indicators(
                ax, x + width / 2,
                [max(correlations["row"]["Spearman"][i], top_whisker_y[i]) for i in range(len(x))],
                p_values["row"]["Spearman"],
            )
        ax.set_title(f"{direction.capitalize()} - Row-wise Correlation\n "
                     f"Mean Pearson: {np.nanmean(correlations['row']['Pearson']):.2f}, "
                     f"Mean Spearman: {np.nanmean(correlations['row']['Spearman']):.2f}\n"
                     # f"Random rows correlation: Pearson: {random_correlations['row']['Pearson'][0]:.2f}±"
                     # f"{random_correlations['row']['Pearson'][1]:.2f}, "
                     # f"Spearman: {random_correlations['row']['Spearman'][0]:.2f}±"
                     # f"{random_correlations['row']['Spearman'][1]:.2f}"
                     )
        # add thin line for random correlations
        # ax.axhline(random_correlations['row']['Pearson'][0], color=colors[0], linestyle='--')
        # ax.axhline(random_correlations['row']['Pearson'][0] + random_correlations['row']['Pearson'][1], color=colors[0],
        #            linestyle=':')
        # ax.axhline(random_correlations['row']['Spearman'][0], color=colors[1], linestyle='--')
        # ax.axhline(random_correlations['row']['Spearman'][0] + random_correlations['row']['Spearman'][1],
        #            color=colors[1], linestyle=':')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1, 1.2)  # Force y-axis to range from -1 to 1
        ax.legend()

        # Plot column-wise correlations
        ax = axes[1, j]
        x_labels = df1.columns
        x = np.arange(len(x_labels))
        pearson_boxplot = ax.boxplot(shuffle_values["column"]["Pearson"], positions=x - width / 2, widths=width / 2,
                                     patch_artist=True, boxprops=dict(facecolor=colors[0], alpha=0.3, color=colors[0]))
        spearman_boxplot = ax.boxplot(shuffle_values["column"]["Spearman"], positions=x + width / 2, widths=width / 2,
                                      patch_artist=True, boxprops=dict(facecolor=colors[1], alpha=0.3, color=colors[1]))
        ax.scatter(x - width / 2, correlations["column"]["Pearson"], color=colors[0], s=50, edgecolor="black",
                   linewidth=0.5, zorder=3, label="Pearson")
        ax.scatter(x + width / 2, correlations["column"]["Spearman"], color=colors[1], s=50, edgecolor="black",
                   linewidth=0.5, zorder=3, label="Spearman")
        # Add significance indicators for Pearson correlations
        top_whisker_y = np.array([max(whisker.get_ydata()) for whisker in pearson_boxplot['whiskers'][1::2]])
        if not np.isnan(top_whisker_y.any()):
            add_significance_indicators(
                ax, x - width / 2,
                [max(correlations["column"]["Pearson"][i], top_whisker_y[i]) for i in range(len(x))],
                p_values["column"]["Pearson"],
            )

        # Add significance indicators for Spearman correlations
        top_whisker_y = np.array([max(whisker.get_ydata()) for whisker in spearman_boxplot['whiskers'][1::2]])
        if not np.isnan(top_whisker_y.any()):
            add_significance_indicators(
                ax, x + width / 2,
                [max(correlations["column"]["Spearman"][i], top_whisker_y[i]) for i in range(len(x))],
                p_values["column"]["Spearman"],
            )
        ax.set_title(f"{direction.capitalize()} - Column-wise Correlation\n "
                     f"Mean Pearson: {np.nanmean(correlations['column']['Pearson']):.2f}, "
                     f"Mean Spearman: {np.nanmean(correlations['column']['Spearman']):.2f}\n"
                     # f"Random columns correlation: Pearson: {random_correlations['column']['Pearson'][0]:.2f}±"
                     # f"{random_correlations['column']['Pearson'][1]:.2f}, "
                     # f"Spearman: {random_correlations['column']['Spearman'][0]:.2f}±"
                     # f"{random_correlations['column']['Spearman'][1]:.2f}"
                     )
        # ax.axhline(random_correlations['column']['Pearson'][0], color=colors[0], linestyle='--')
        # ax.axhline(random_correlations['column']['Pearson'][0] + random_correlations['column']['Pearson'][1],
        #            color=colors[0], linestyle=':')
        # ax.axhline(random_correlations['column']['Spearman'][0], color=colors[1], linestyle='--')
        # ax.axhline(random_correlations['column']['Spearman'][0] + random_correlations['column']['Spearman'][1],
        #            color=colors[1], linestyle=':')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1, 1.2)  # Force y-axis to range from -1 to 1
        ax.legend()

    # set x and y ticks of all subplots to 8
    for ax in axes.flat:
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)

    # Adjust layout for better visualization
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(private + "/analysis/correlation_gsea_our.png")
    plt.savefig(private + "/analysis/correlation_gsea_our.svg")
    plt.show()


def merge_results(path):
    for treat in treatments:
        for j, abx in enumerate(antibiotics):
            # abx_data = meta_data[(meta_data['Drug'] == title) & (meta_data['Treatment'] == treat)]
            # pbs_data = meta_data[(meta_data['Drug'] == 'PBS') & (meta_data['Treatment'] == treat)]
            selected = pd.read_csv(path, sep="\t")
            gsea_selected = pd.DataFrame()
            for folder in os.listdir(os.path.join(private, "gsea")):
                if folder.startswith(f"{abx.capitalize()}{treat.upper()}"):
                    # read the csv file that starts with gsea_report_for_1
                    for file in os.listdir(os.path.join(private, "GSEA", folder)):
                        if file.startswith("gsea_report_for") and file.endswith(".tsv"):
                            results = pd.read_csv(os.path.join(private, "GSEA", folder, file), sep="\t")
                            results['GO term'] = results['NAME']
                            addition = "_enh" if "_1_" in file else "_supp"
                            results['GO term'] = results['GO term'] + addition
                            results['enhanced?'] = "_1_" in file
                            results = results.rename(columns={"SIZE": "size"})
                            gsea_selected = pd.concat([gsea_selected, results], ignore_index=True)
            # merge the two dataframes based on index
            merged = pd.merge(selected, gsea_selected, on='GO term', how='outer', suffixes=('_our', '_gsea'))
            merged.to_csv(os.path.join(private, "clusters_properties", "diff_abxRASflow", f"merged_{abx}_{treat}.tsv"),
                          sep="\t", index=False)
            clean = merged[
                ["Antibiotics", "Condition", "GO term", "name", "genes", "gene names", "size_our", "size_gsea",
                 "fdr GO significance",
                 "fdr correlation", "FDR q-val"]]
            clean.to_csv(
                os.path.join(private, "clusters_properties", "diff_abxRASflow", f"clean_merged_{abx}_{treat}.tsv"),
                sep="\t", index=False)

# if __name__ == "__main__":
#     run_type = "RASflow"
#     all_data = pd.read_csv(os.path.join(path, f"diff_abx{run_type}\\top_correlated_GO_terms.tsv"), sep="\t")
#     genome, meta, partek, transcriptome = read_process_files(new=False)
#     raw = transcriptome
#     raw, metadata = transform_data(raw, meta, run_type, skip=True)
#     # save raw to csv
#     raw.to_csv(private + f"/analysis/Diff_abx{run_type}/normalized_multiabx.csv")
#
#     plot_selected_clusters(raw, meta, "diff_abx" + run_type)
#     plot_significant_genes_number(meta, raw, antibiotics, treatments, "diff_abx" + run_type)
#
#     compare_significance_go(param="diff_abx" + run_type)
#     clusters_compare_mix(antibiotics, treatments, "diff_abx" + run_type)
#
#     merge_results()
#     # save_median_all_conditions(meta, raw, antibiotics, treatments, "Treatment", "diff_abx" + run_type)
#     our = plot_categories(antibiotics, treatments, "\\diff_abx" + run_type, False, regular=False)
#     gsea = plot_categories(antibiotics, treatments, "\\diff_abx" + "GSEA", False, regular=False, gsea=True,
#                            anchor=(0.5, -5.2))
#     plot_correlation_gsea(gsea, our)
