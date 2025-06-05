import os

import gseapy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ClusteringGO import treatments, transform_data, antibiotics
from all_figures_plot import read_process_files

light_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
orange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]


def plot_confusion_matrix(addition="", factor=1, path=os.path.join("Private"), order=None, random=False,
                          size=(10, 10)):
    # plot it as a heatmap, make x label "predicted", y label "true"
    import matplotlib.pyplot as plt
    import seaborn as sns
    forest_confusion_matrix = pd.read_csv(path + f"/confusion_matrix{addition}{'_random' if random else ''}.csv",
                                          index_col=0)
    if order:
        forest_confusion_matrix = forest_confusion_matrix.loc[order, order]
    fig, ax = plt.subplots(figsize=size)
    # print(forest_confusion_matrix.sum(axis=1).values, forest_confusion_matrix.sum(axis=0).values)
    print(forest_confusion_matrix.sum().sum())

    # Set the center point
    center = 1 / len(forest_confusion_matrix)
    # Create a custom normalization
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=0, vcenter=center, vmax=1)

    # increase size of number on the heatmap
    if factor != 1:
        heatmap = sns.heatmap(forest_confusion_matrix / factor, annot=True, fmt=".1%", ax=ax, annot_kws={"size": 20},
                              vmin=0,
                              vmax=1, norm=norm)
        to_save = forest_confusion_matrix / factor
    else:
        heatmap = sns.heatmap(forest_confusion_matrix, annot=True, fmt=".1%", ax=ax, annot_kws={"size": 20},
                              cmap="RdBu_r", vmin=0, vmax=1, norm=norm)
        to_save = forest_confusion_matrix

    to_save.to_csv(path + f"/confusion_matrix{addition}{'_random' if random else ''}.csv")
    # Add center to cbar
    cbar = heatmap.collections[0].colorbar

    # # Get the current tick locations and labels
    # tick_locs = cbar.get_ticks()
    # if center not in tick_locs:
    #     tick_locs = sorted(list(tick_locs) + [center])
    # # tick_locs = [loc for loc in tick_locs if abs(loc - center) > 1e-6]
    tick_locs = [center * i for i in range(int(1 / center) + 1)]

    cbar.set_ticks(tick_locs)
    tick_labels = [f'{loc * 100:.0f}%' for loc in tick_locs]
    # center_index = tick_locs.index(center)
    # tick_labels[center_index] = f'{center:.3f} (center)'
    cbar.set_ticklabels(tick_labels)

    ax.tick_params(labelsize=18)
    ax.set_xlabel("Predicted Category", fontsize=25)
    ax.set_ylabel("True Category", fontsize=25)
    plt.tight_layout()
    plt.savefig(path + f"/confusion_matrix{addition}{'_random' if random else ''}.png")
    # plt.show()
    plt.close()


def plot_heatmap_colors(cluster, col_cluster, save_name, top_df, path=os.path.join("Private"), normalize=True,
                        colors=None, title="", jump=7, xticks=[3.5, 10.5, 17.5, 24.5], show_all_y=True, hline=None,
                        vline=False, sort=True, set_max=True, multiabx=False):
    if col_cluster:
        mice = cluster.dendrogram_col.reordered_ind
        top_df = top_df.iloc[:, mice]
    if normalize:
        # remove from each row its mean
        top_df = top_df.sub(top_df.mean(axis=1), axis=0)
        # normalize each row by its standard deviation
        top_df = top_df.div(top_df.std(axis=1), axis=0)
    if sort:
        # sort top_df columns lexically
        columns = np.argsort(top_df.columns)
        top_df = top_df.iloc[:, columns]

    # plot the heatmap
    if set_max:
        vmax = 10 if show_all_y else set_max
        # heatmap = sns.heatmap(top_df, vmax=vmax, cmap="RdBu_r")
        heatmap = sns.heatmap(top_df, vmax=vmax, vmin=-10, cmap="RdBu_r")
    elif multiabx:
        vmax = 2
        vmin = -2
        heatmap = sns.heatmap(top_df, vmax=vmax, vmin=vmin, cmap="RdBu_r")
    else:
        heatmap = sns.heatmap(top_df, cmap="RdBu_r")
    # create a dictionary 'PBS DONOR': 'blue', 'PBS RECIPIENT': 'light blue',
    # 'Van DONOR': 'red', 'Van RECIPIENT': 'light red'

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
        last = f"{ticks[-1]:.0f}+" if actual_vmax == vmax else actual_vmax
        first = f"{ticks[0]:.0f}+" if actual_vmin == vmin else actual_vmin
        tick_labels = [first] + [f"{tick:.0f}" for tick in ticks[1:-1]] + [last]  # Add '+' to the last label
        # Apply the customized tick labels
        cbar.set_ticklabels(tick_labels)

    if not colors:
        colors = {'Donor PBS': 'blue', 'Recipient PBS': light_blue, 'Donor Van': 'red', 'Recipient Van': orange}
    label_colors = [colors[label] for label in top_df.columns]
    if hline:
        plt.axhline(y=hline, color='black', linewidth=1)
    if vline:
        for i in range(1, 3):
            # draw a line between pbs and abx
            plt.axvline(x=14 * i - 7, color="black", linewidth=1, linestyle="--")
            # draw a dashed line between time points
            plt.axvline(x=14 * i, color="black", linewidth=1)
    bar_height = 0.01 * top_df.shape[0]
    ax = plt.gca()
    for k, color in enumerate(label_colors):
        bar_width = 1  # Set the width to match a column (fixed at 1)
        ax.add_patch(
            plt.Rectangle((k, top_df.shape[0] - bar_height), bar_width, bar_height, color=color,
                          fill=True))
    if jump:
        # show only the 4th, 11th, 18th, 25th columns
        ax.set_xticks(xticks)
        ax.set_xticklabels(top_df.columns[::jump])
        plt.xticks(rotation=45, ha='center', size=20)
    else:
        plt.xticks(np.arange(len(top_df.columns)), top_df.columns, rotation=0, fontsize=8)
        plt.xticks(rotation=90, ha='center', size=20)
    if title:
        plt.title(title)
    # plt.xticks(np.arange(len(top_df.columns)), top_df.columns, rotation=45, ha='right', fontsize=12)
    if show_all_y:
        # show all yticks
        plt.yticks(np.arange(len(top_df.index)), top_df.index, rotation=0, fontsize=8)
    plt.ylabel("")
    # increase all font sizes
    plt.rc('font', size=25)
    plt.gcf().set_size_inches(2.5 * (len(colors) + 1), 15 * top_df.shape[0] / 100)
    plt.tight_layout()
    if top_df.shape[0] < 400:
        # plt.savefig(f"./Private/YasminRandomForest/{save_name}.png", dpi=600)
        plt.savefig(path + f"/{save_name}.pdf", dpi=600)
        plt.savefig(path + f"/{save_name}.png", dpi=600)
    else:
        plt.savefig(path + f"/{save_name}.pdf")
        plt.savefig(path + f"/{save_name}.png")
    plt.close()
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def classification_report_to_df(report_str):
    import re

    lines = report_str.split('\n')

    # Find the column names
    column_names = re.findall(r'\b\w+\b', lines[0])

    # Initialize an empty list to store data
    data = []

    # Iterate through the lines and extract data
    for line in lines[2:-5]:
        values = re.findall(r'\b\d+\.?\d*\b', line)
        if values:
            data.append(values)

    # Create a Pandas DataFrame
    df = pd.DataFrame(data, columns=column_names)

    # set precision to index, rename recall to precision, f1 to recall, score to f1-score
    df = df.rename(columns={"precision": "index", "recall": "precision", "f1": "recall", "score": "f1-score"})
    df = df.set_index("index")
    # convert all values to float
    df = df.astype(float)

    return df


def four_way_forest(df, feature_columns, target_column, test_size=8 / 28, random_state=42):
    """
    Perform classification with a random forest classifier for four classes.

    Parameters:
    - df: DataFrame with features and target variable.
    - feature_columns: List of column names for features.
    - target_column: Name of the target variable column.
    - test_size: Proportion of the data to include in the test split (default is 0.2).
    - random_state: Seed for random number generation (default is 42).

    Returns:
    - clf: Trained random forest classifier.
    - conf_matrix: Confusion matrix.
    - classification_rep: Classification report.
    """

    # Split the data into features (X) and target variable (y)
    X = df[feature_columns]
    y = df[target_column]

    # Encode target variable if it's not numeric
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state + 2,
                                                        stratify=y)

    # Build a random forest classifier
    clf = RandomForestClassifier(random_state=random_state + 1)
    clf.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = clf.predict(X_test)

    # Get actual labels before encoding
    actual_labels = label_encoder.inverse_transform(y_test)

    # Evaluate the classifier
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # create a dictionary of the labels using the label encoder
    labels_dict = {i: label for i, label in enumerate(label_encoder.classes_)}

    # convert classification report to a DataFrame
    report = classification_report_to_df(classification_rep)
    # get features importance
    importance = pd.Series(clf.feature_importances_, index=feature_columns)

    return conf_matrix, report.values, importance, labels_dict


def four_way_random_forest_multiabx(abx_data, abx_metadata, title, column, abx=True, reps=10000,
                                    path=os.path.join("Private", "AbxRandomForest")):
    # add to data the group column from metadata
    abx_data = abx_data.T
    # intersecting_genes = fmt_data.index.intersection(abx_data.index)

    if abx:
        # abx_data = abx_data.loc[intersecting_genes]
        abx_metadata[column] = abx_metadata["Drug"] + "_" + abx_metadata["Treatment"]

    abx_data = pd.merge(abx_data, abx_metadata[["ID", column]], left_index=True, right_on="ID", how='outer').set_index(
        "ID")
    n = abx_metadata[column].nunique()
    confusion_matrix = np.zeros((n, n))
    classification_report = np.zeros((n, 4))
    importance = pd.Series(np.zeros(len(abx_data.columns[:-1])), index=abx_data.columns[:-1])
    # replace na with 0
    for i in range(reps):
        result = four_way_forest(abx_data, abx_data.columns[:-1], column,
                                 test_size=n * 1.5 / abx_data.shape[0], random_state=i)
        confusion_matrix += result[0]
        classification_report += result[1]
        importance += result[2]
        labels_dict = result[3]
    confusion_matrix /= reps
    classification_report /= reps
    importance /= reps
    print("Confusion Matrix:")
    print(confusion_matrix)
    confusion_matrix = pd.DataFrame(confusion_matrix)

    # normalizing confusion matrix by the sum of each row
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)

    confusion_matrix.index = [labels_dict[i] for i in range(confusion_matrix.shape[0])]
    confusion_matrix.columns = [labels_dict[i] for i in range(confusion_matrix.shape[0])]
    # save the confusion matrix
    # if path does not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)
    confusion_matrix.to_csv(path + f"/confusion_matrix_{title}.csv", index=True)

    classification_report = pd.DataFrame(classification_report, columns=["precision", "recall", "f1-score", "support"])
    print("\nClassification Report:")
    print(classification_report)
    # save feature importance
    importance = importance.sort_values(ascending=False)
    importance.to_csv(path + f"/feature_importance_{title}.csv", index=True)
    return confusion_matrix


def four_way_random_forest_multitreat(abx_data, abx_metadata, title, column, abx=True, reps=10000,
                                      path=os.path.join("Private", "AbxRandomForest"), random=False):
    # add to data the group column from metadata
    abx_data = abx_data.T
    # intersecting_genes = fmt_data.index.intersection(abx_data.index)

    if abx:
        # abx_data = abx_data.loc[intersecting_genes]
        abx_metadata[column] = abx_metadata.apply(
            lambda row: row["Drug"] + "_" + row["Treatment"] if row["Drug"] != "PBS" else "PBS", axis=1
        )

    abx_data = pd.merge(abx_data, abx_metadata[["ID", column]], left_index=True, right_on="ID").set_index("ID")
    n = abx_metadata[column].nunique()
    confusion_matrix = np.zeros((n, n))
    classification_report = np.zeros((n, 4))
    importance = pd.Series(np.zeros(len(abx_data.columns[:-1])), index=abx_data.columns[:-1])
    from collections import Counter
    smallest = min(list(Counter([group_name for group_name in abx_data[column] if group_name != "PBS"]).values()))
    for i in range(reps):
        if random:
            # abx_data.index = np.random.permutation(abx_data.index)
            # abx_data[column] = np.random.RandomState(seed=42).permutation(abx_data[column].values)
            abx_data[column] = np.random.permutation(abx_data[column].values)
        # Down sample:
        indexes = []
        for treat in treatments:
            indexes.extend(
                abx_data[abx_data["group"].str.endswith(treat)].sample(n=smallest, replace=False, random_state=i).index)
        indexes.extend(abx_data[abx_data["group"] == "PBS"].sample(n=smallest, replace=False, random_state=i).index)
        result = four_way_forest(abx_data.loc[indexes], abx_data.loc[indexes].columns[:-1], column,
                                 test_size=n * 1.5 / abx_data.loc[indexes].shape[0], random_state=i)
        confusion_matrix += result[0]
        classification_report += result[1]
        importance += result[2]
        labels_dict = result[3]
    confusion_matrix /= reps
    classification_report /= reps
    importance /= reps
    print(f"Confusion Matrix {'(random)' if random else ''}:")
    print(confusion_matrix)
    confusion_matrix = pd.DataFrame(confusion_matrix)

    # normalizing confusion matrix by the sum of each row
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)

    # make a df from the confusion matrix with columns |  PBS_DONOR   |  PBS_RECIPIENT  |  Vanco_DONOR  | Vanco_RECIPIENT |
    # and rows |  PBS_DONOR   |  PBS_RECIPIENT  |  Vanco_DONOR  | Vanco_RECIPIENT |
    # confusion_matrix = pd.DataFrame(confusion_matrix,
    #                                 columns=[labels_dict[i] for i in range(confusion_matrix.shape[0])],
    #                                 index=[labels_dict[i] for i in range(confusion_matrix.shape[0])])
    confusion_matrix.index = [labels_dict[i] for i in range(confusion_matrix.shape[0])]
    confusion_matrix.columns = [labels_dict[i] for i in range(confusion_matrix.shape[0])]
    # save the confusion matrix
    confusion_matrix.to_csv(path + f"/confusion_matrix_{title}{'_random' if random else ''}.csv", index=True)

    classification_report = pd.DataFrame(classification_report, columns=["precision", "recall", "f1-score", "support"])
    print("\nClassification Report:")
    print(classification_report)
    # save feature importance
    importance = importance.sort_values(ascending=False)
    importance.to_csv(path + f"/feature_importance_{title}{'_random' if random else ''}.csv", index=True)
    return confusion_matrix


def background_analysis(background):
    background = background[~background.isnull()]
    base_url = "https://maayanlab.cloud/speedrichr"

    res = requests.post(
        base_url + '/api/addbackground',
        data=dict(background='\n'.join(background)),
    )
    if res.ok:
        background_response = res.json()
        return background_response["backgroundid"]


def multi_abx_forest():
    metadata, data = read_process_files(new=False)
    data, metadata = transform_data(data, metadata, "RASflow",
                                    skip=True)  # note we don't do that due to 324 np.inf values caused by division by 0

    metadata["group"] = metadata["Drug"] + "_" + metadata["Treatment"]
    ensmus_to_gene = get_ensmus_dict()
    # background_id = background_analysis(data.rename(index=ensmus_to_gene).index)
    background_id = data.rename(index=ensmus_to_gene).index.to_list()
    # for treat in ["IP"]:
    for treat in treatments:
        sub_metadata = metadata[metadata["Treatment"] == treat]
        sub_data = data[sub_metadata['ID']]

        four_way_random_forest_multiabx(sub_data, sub_metadata, treat, "group", abx=True, reps=10_000)

        plot_confusion_matrix(f"_{treat}", factor=1, path=os.path.join("Private", "AbxRandomForest"),
                              order=[abx + f"_{treat}" for abx in ["PBS"] + antibiotics])
        analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(400,), background=background_id,
                        treat=treat)


def multi_treat_forest(random=False):
    metadata, data = read_process_files(new=False)
    data, metadata = transform_data(data, metadata, "RASflow", skip=True)

    metadata["group"] = metadata["Drug"] + "_" + metadata["Treatment"]
    ensmus_to_gene = get_ensmus_dict()
    # background_id = background_analysis(data.rename(index=ensmus_to_gene).index)
    background_id = data.rename(index=ensmus_to_gene).index.to_list()
    for abx in antibiotics:
        sub_metadata = metadata[(metadata["Drug"] == abx) | (metadata["Drug"] == "PBS")]
        sub_data = data[sub_metadata['ID']]

        four_way_random_forest_multitreat(sub_data, sub_metadata, abx, "group", abx=True, reps=10_000, random=random)

        plot_confusion_matrix(f"_{abx}", factor=1, path=os.path.join("Private", "AbxRandomForest"),
                              order=[abx + f"_{treat}" for treat in treatments] + ["PBS"], random=random)
        analyze_results(sub_data, sub_metadata, f"_{abx}", sizes=(400,), background=background_id,
                        treat=abx)


def get_ensmus_dict():
    folder_dir = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
    df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
    return df.set_index('gene_id')['gene_name'].to_dict()


factors = {
    "IP": 50,
    "IV": 70,
    "PO": 25,
    "Amp": 100,
    "Met": 23,
    "Neo": 140,
    "Van": 45,
    "Mix": 18,
}


def plot_cumsum(feature_importance, title):
    # Sort the feature importance in descending order
    feature_importance_sorted = feature_importance.sort_values('importance', ascending=False)

    # Calculate the cumulative sum of importance
    feature_importance_sorted['cumsum'] = feature_importance_sorted['importance'].cumsum()

    # Calculate the percentage of total importance
    total_importance = feature_importance_sorted['importance'].sum()
    feature_importance_sorted['cumsum_percent'] = feature_importance_sorted['cumsum'] / total_importance * 100

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=feature_importance_sorted, x=range(len(feature_importance_sorted)), y='cumsum_percent')

    plt.title(f'Cumulative Sum of Feature Importance {title.split("_")[-1]}')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance (%)')

    # Add reference lines
    plt.axhline(y=80, color='r', linestyle='--', label='80% Importance')
    plt.axhline(y=90, color='g', linestyle='--', label='90% Importance')

    plt.legend()
    plt.grid(True)
    plt.savefig(path=os.path.join("Private", "AbxRandomForest", "feature_importance_cum-sum{title}.png"))

    # Show the plot
    # plt.show()
    plt.close()


def analyze_results(data, metadata, title, background, treat, sizes=(100, 200, 400), abx=None, random=False):
    samples_order = []
    if treat in treatments:
        for abx in ["PBS"] + antibiotics:
            samples_order.extend(metadata[(metadata["Drug"] == abx) & (metadata["Treatment"] == treat)]["ID"].values)
    else:
        for treat_ in treatments:
            samples_order.extend(metadata[(metadata["Drug"] == "PBS") & (metadata["Treatment"] == treat_)]["ID"].values)
        for treat_ in treatments:
            samples_order.extend(metadata[(metadata["Drug"] == treat) & (metadata["Treatment"] == treat_)]["ID"].values)
    data = data[samples_order]
    # change group column to group (ID)
    feature_importance = pd.read_csv(os.path.join("Private", "AbxRandomForest", f"feature_importance{title}.csv"),
                                     index_col=0)
    ensmus_to_gene = get_ensmus_dict()
    # rename index to "gene" and rename first column to "importance"
    feature_importance.index.name = "gene"

    feature_importance = feature_importance.rename(columns={feature_importance.columns[0]: "importance"})
    # Print Feature importance cumsum
    # plot_cumsum(feature_importance, title)
    for num_top in sizes:
        top = feature_importance.head(num_top).index.values
        if abx:
            top = [gene for gene in top if gene in data.index]
            if random:
                top = np.random.choice(data.index, 100, replace=False)

            pbs_columns = [col for col in data.columns if "C" in col]
            non_pbs_columns = [col for col in data.columns if "C" not in col]
            ordered_columns = pbs_columns + non_pbs_columns
            data = data[ordered_columns]
        # create a df with only those genes
        top_df = data.loc[top]

        # rename the columns to the group
        top_df = top_df.rename(columns=metadata.set_index('ID')['group'].to_dict())
        top_df.rename(index=ensmus_to_gene, inplace=True)

        # background = background_analysis(top_df.index)
        dynamic_tree_plot(top_df, [gene for gene in background if type(gene) is str], factors[treat],
                          title + f"_{num_top}")
        continue


def calculate_between_cluster_variance(cluster_indices_1, cluster_indices_2, distance_matrix):
    between_cluster_dists = distance_matrix[np.ix_(cluster_indices_1, cluster_indices_2)]
    return np.var(between_cluster_dists.flatten())


def dynamic_tree_cut(link, distances, size_penalty_factor=50, min_cluster_size=3):
    n_samples = link.shape[0] + 1
    distance_matrix = squareform(distances)

    # Initialize each sample as its own cluster
    clusters = [{i} for i in range(n_samples)]

    def calculate_variance(cluster):
        if len(cluster) <= 1:
            return 0
        cluster_distances = distance_matrix[np.ix_(list(cluster), list(cluster))]
        return np.var(cluster_distances[np.triu_indices(len(cluster), k=1)])

    def should_merge(cluster1, cluster2):
        if not cluster1 or not cluster2:
            return True
        combined_cluster = cluster1.union(cluster2)
        within_var = calculate_variance(combined_cluster)
        between_var = np.var(distance_matrix[np.ix_(list(cluster1), list(cluster2))])

        # Size penalty
        size_penalty = size_penalty_factor * (1 / len(cluster1) + 1 / len(cluster2))

        # Merge if within variance (plus size penalty) is less than or equal to between variance,
        # or if either cluster is smaller than min_cluster_size
        return (within_var <= between_var + size_penalty) or \
            (len(cluster1) <= min_cluster_size) or \
            (len(cluster2) <= min_cluster_size)

    # Traverse the tree and merge clusters
    for i in range(n_samples - 1):
        left, right, _, _ = link[i]
        left, right = int(left), int(right)

        if should_merge(clusters[left], clusters[right]):
            # Merge the clusters
            new_cluster = clusters[left].union(clusters[right])
            clusters.append(new_cluster)
        else:
            # Keep clusters separate
            clusters.append(set())

    # Assign cluster labels
    cluster_labels = np.zeros(n_samples, dtype=int)
    # valid_clusters = [c for c in clusters if c]  # Remove empty clusters
    for i, cluster in enumerate(clusters):
        for sample in cluster:
            cluster_labels[sample] = i

    return cluster_labels


def check_enrichment(genes, background):
    db = ["GO_Biological_Process_2023", "KEGG_2019_Mouse"]
    if background is not None:
        return gseapy.enrichr(gene_list=genes, gene_sets=db, organism="mouse", outdir=None,
                              background=background).results
    return gseapy.enrichr(gene_list=genes, gene_sets=db, organism="mouse", outdir=None).results


def remap_clusters(clusters, order):
    # # Get the unique cluster labels
    # unique_labels = np.unique(clusters)

    # Create a mapping from original labels to new labels
    counter = 0
    label_mapping = {}
    for old_label in order:
        if old_label != -1:
            label_mapping[old_label] = counter
            counter += 1

    # label_mapping = {old_label: new_label for new_label, old_label in enumerate(order) if old_label != -1}
    label_mapping[-1] = -1
    # other = -1
    # assert label_mapping[other] == 0

    # Apply the mapping to the clusters array
    remapped_clusters = np.array([label_mapping[label] for label in clusters])

    return remapped_clusters, label_mapping


def change_group_number(numbers):
    seen_groups = set()
    new_numbers = []
    next_new_number = max(numbers) + 1

    i = 0
    while i < len(numbers):
        current_number = numbers[i]

        # Skip -1 values
        if current_number == -1:
            new_numbers.append(current_number)
            i += 1
            continue

        # Find the end of the current group
        j = i + 1
        while j < len(numbers) and numbers[j] == current_number:
            j += 1

        # Check if this group has been seen before
        group = tuple(numbers[i:j])
        index = set(numbers[i:j])
        assert len(index) == 1
        index = [ind for ind in index][0]
        if index in seen_groups:
            # Replace with a new number
            new_numbers.extend([next_new_number] * (j - i))
            next_new_number += 1
            # print(f"Group {index} has been seen before")
        else:
            # Keep the original numbers
            new_numbers.extend(group)
            seen_groups.add(index)

        i = j

    return new_numbers


def dynamic_tree_plot(top_df, background, factor, title):
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    # top_df = np.transpose(np.arange(1, 10001).reshape(100, 100))
    # distances = pdist(top_df, "euclidean")
    # normalized_df = (top_df - top_df.mean()) / top_df.std()

    # link = linkage(normalized_df, "average", metric="euclidean")
    g = sns.clustermap(top_df, row_cluster=True, metric="euclidean", method="average",
                       col_cluster=False, z_score=0)

    # Access the linkage matrix used for the rows
    orig_link = g.dendrogram_row.linkage
    # # Get the leaf order
    # leaf_order = sch.leaves_list(orig_link)
    # # Reorder the DataFrame
    # ordered_df = top_df.iloc[leaf_order]
    # link = sch.linkage(ordered_df, "average")
    # Cut the dendrogram dynamically
    # clusters = dynamic_tree_cut(top_df)  # , factor)
    clusters = dynamic_tree_cut(orig_link, pdist(top_df), size_penalty_factor=factor)  # , factor)

    ordered = hierarchy.dendrogram(orig_link, no_plot=True)['leaves']
    reordered_clusters = [clusters[i] for i in ordered]
    remapped_clusters = change_group_number(reordered_clusters)
    # Map reordered clusters back to original positions
    for i, original_index in enumerate(ordered):
        clusters[original_index] = remapped_clusters[i]
    # Iterate over clusters and collect indexes
    top_df['Cluster'] = clusters
    cluster_dict = {}
    cluster_dict[-1] = []
    for cluster_id in np.unique(clusters):
        cluster_indexes = top_df[top_df['Cluster'] == cluster_id].index.tolist()
        if len(cluster_indexes) < 5:
            cluster_dict[-1] += cluster_indexes
            clusters[clusters == cluster_id] = -1
        else:
            cluster_dict[cluster_id] = cluster_indexes
    clusters, mapping = remap_clusters(clusters, list(dict.fromkeys([clusters[i] for i in ordered])))
    top_df = top_df.drop("Cluster", axis=1)

    # names = {}
    # Print cluster indexes
    enrichment_analysis = pd.DataFrame(
        columns=['Gene_set', 'Term', 'P-value', 'Adjusted P-value', 'Old P-value',
                 'Old adjusted P-value', 'Odds Ratio', 'Combined Score', 'Genes', "cluster_id"])
    # columns=["cluster_id", "index", "Name", "P-value", "Odds Ratio", "Combined score", "Overlap Genes",
    #          "Adjusted p-value", "old p-value", "old Adjusted p-value", "genes"])
    for cluster_id, indexes in cluster_dict.items():
        if len(indexes) < 2 or cluster_id == -1:
            continue
        # print(f"Cluster {cluster_id}: {indexes}")
        enrichment = check_enrichment([gene for gene in indexes if type(gene) is str], background)
        if enrichment is not None:
            enrichment["cluster_id"] = mapping[cluster_id]
            enrichment["#of_genes"] = enrichment["Genes"].str.count(";") + 1
            enrichment["All_Genes"] = ",".join([gene for gene in indexes if type(gene) is str])
            enrichment_analysis = pd.concat([enrichment_analysis, enrichment])

    # enrichment_analysis = enrichment_analysis.drop("index", axis=1)
    enrichment_analysis = enrichment_analysis.drop(["Old P-value", "Old adjusted P-value"], axis=1)
    enrichment_analysis["adj.P-val<5%"] = enrichment_analysis["Adjusted P-value"] < 0.05
    enrichment_analysis["P-val<5%"] = enrichment_analysis["P-value"] < 0.05
    enrichment_analysis.to_csv(os.path.join("Private", "AbxRandomForest",
                                            f"cluster_dynamic{title}_enrichment_{'background' if background else ''}.csv"),
                               index=False)
    enrichment_analysis[enrichment_analysis["Adjusted P-value"] < 0.05].to_csv(
        os.path.join("Private", "AbxRandomForest",
                     f"cluster_dynamic{title}_enrichment_{'background' if background else ''}_filtered.csv"),
        index=False)

    # Create a color palette for the clusters
    n_clusters = len(np.unique(clusters))
    print(n_clusters)

    palette = sns.color_palette("tab10")
    # palette = palette + [(0, 0, 0)]  # Black in RGB is (0, 0, 0)
    # palette = sns.color_palette(palette)
    cluster_colors = [palette[c % len(palette)] if c != -1 else (0, 0, 0) for c in clusters]
    # cluster_colors[-1] = [(0, 0, 0)]  # Black in RGB is (0, 0, 0)
    # Standardize the data
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(top_df)
    # Create the clustermap
    plt.figure(figsize=(12, 8))
    from matplotlib.colors import LinearSegmentedColormap

    # Define a linear gradient colormap with the specified colors
    colors = [
        (0, "#5C00A6"),  # Purple at the maximum (positive end)
        (0.25, "#78C6FF"),  # Blue transitioning to white
        (0.5, "white"),  # White at zero (neutral point)
        (0.75, "#F79D00"),  # Orange transitioning from white
        (1, "#A11400"),  # Dark Red at the minimum (negative end)
    ]

    # Create a custom colormap using the defined colors
    single_axis_cmap = LinearSegmentedColormap.from_list("SingleAxisDiverging", colors)
    limit = 6
    # limit = max(abs(top_df.values.min()), abs(top_df.values.max()))
    g = sns.clustermap(top_df,
                       row_colors=cluster_colors,
                       col_cluster=False,
                       row_linkage=orig_link,
                       # cmap="vlag",
                       cmap=single_axis_cmap,
                       # figsize=(12, 8),
                       dendrogram_ratio=0.2,
                       vmax=limit,
                       vmin=-limit,
                       # z_score=0
                       # cbar_pos=(0.02, 0.8, 0.05, 0.18)
                       )

    # save
    plotted_data = g.data2d
    plotted_data.to_csv(os.path.join("Private", "AbxRandomForest", f"cluster_dynamic{title}.csv"))

    # # Adjust the colorbar label
    # g.ax_cbar.set_ylabel('Standardized Values')
    # Add cluster labels to the dendrogram
    for i, c in enumerate(np.unique(clusters)):
        color_cluster = palette[c % len(palette)] if c != -1 else (0, 0, 0)
        g.ax_row_dendrogram.bar(0, 0, color=color_cluster, label=f'Cluster {c}', linewidth=0)
    g.ax_row_dendrogram.legend(title='Clusters', loc="center")  # , bbox_to_anchor=(0.5, 0.8))
    plt.title('Clustered Heatmap with Dendrogram', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join("Private", "AbxRandomForest", f"cluster_dynamic{title}.png"))
    # plt.show()
    plt.close()


if __name__ == "__main__":
    multi_abx_forest()
    multi_treat_forest()
