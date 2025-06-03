import gseapy
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
# from Yasmin_analysis import four_way_random_forest
from Yasmin_analysis import four_way_forest, classification_report_to_df, plot_heatmap_colors, orange, light_blue, \
    plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ClusteringGO import read_process_files, treatments, transform_data, antibiotics


def class_forest(df, feature_columns, target_column, test_size=8 / 28, random_state=42):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)

    # Build a random forest classifier
    clf = RandomForestClassifier(random_state=random_state)
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

    # print("Actual Labels:")
    # print(labels_dict)
    # print("\nConfusion Matrix:")
    # print(conf_matrix)
    # print("\nClassification Report:")
    # print(classification_rep)
    # convert classification report to a DataFrame
    report = classification_report_to_df(classification_rep)
    # get features importance
    importance = pd.Series(clf.feature_importances_, index=feature_columns)

    return conf_matrix, report.values, importance


def four_way_random_forest_multiabx(abx_data, abx_metadata, title, column, abx=True, reps=10000,
                                    path="./Private/AbxRandomForest"):
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

    # make a df from the confusion matrix with columns |  PBS_DONOR   |  PBS_RECIPIENT  |  Vanco_DONOR  | Vanco_RECIPIENT |
    # and rows |  PBS_DONOR   |  PBS_RECIPIENT  |  Vanco_DONOR  | Vanco_RECIPIENT |
    # confusion_matrix = pd.DataFrame(confusion_matrix,
    #                                 columns=[labels_dict[i] for i in range(confusion_matrix.shape[0])],
    #                                 index=[labels_dict[i] for i in range(confusion_matrix.shape[0])])
    confusion_matrix.index = [labels_dict[i] for i in range(confusion_matrix.shape[0])]
    confusion_matrix.columns = [labels_dict[i] for i in range(confusion_matrix.shape[0])]
    # save the confusion matrix
    confusion_matrix.to_csv(path + f"/confusion_matrix_{title}.csv", index=True)

    classification_report = pd.DataFrame(classification_report, columns=["precision", "recall", "f1-score", "support"])
    print("\nClassification Report:")
    print(classification_report)
    # save feature importance
    importance = importance.sort_values(ascending=False)
    importance.to_csv(path + f"/feature_importance_{title}.csv", index=True)
    return confusion_matrix


def four_way_random_forest_multitreat(abx_data, abx_metadata, title, column, abx=True, reps=10000,
                                      path="./Private/AbxRandomForest", random=False):
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
    _, metadata, _, data = read_process_files(new=False)
    data, metadata = transform_data(data, metadata, "RASflow",
                                    skip=True)  # note we don't do that due to 324 np.inf values caused by division by 0
    # data.fillna(0, inplace=True)

    metadata["group"] = metadata["Drug"] + "_" + metadata["Treatment"]
    ensmus_to_gene = get_ensmus_dict()
    background_id = background_analysis(data.rename(index=ensmus_to_gene).index)
    background_id = data.rename(index=ensmus_to_gene).index.to_list()
    # for treat in ["IP"]:
    for treat in treatments:
        sub_metadata = metadata[metadata["Treatment"] == treat]
        sub_data = data[sub_metadata['ID']]

        four_way_random_forest_multiabx(sub_data, sub_metadata, treat, "group", abx=True, reps=10_000)

        plot_confusion_matrix(f"_{treat}", factor=1, path="./Private/AbxRandomForest",
                              order=[abx + f"_{treat}" for abx in ["PBS"] + antibiotics])
        # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(800, 1600), background=background_id,
        # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(50,), background=background_id,
        # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(400,), background=background_id,
        #                 # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(50, 100, 200, 400, ), background=background_id,
        #                 # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(1600, ), background=background_id,
        #                 treat=treat)
        # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(200,))
        # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(100, 200, 400))
        # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(50, 100, 200, 400, ))


def van_forest():
    _, metadata, _, data = read_process_files(new=False)
    data, metadata = transform_data(data, metadata, "RASflow",
                                    skip=True)
    metadata["group"] = metadata["Drug"] + "_" + metadata["Treatment"]
    ensmus_to_gene = get_ensmus_dict()
    # background_id = background_analysis(data.rename(index=ensmus_to_gene).index)
    background_id = data.rename(index=ensmus_to_gene).index.to_list()
    # save as csv
    data.rename(index=ensmus_to_gene).index.to_csv("./Private/all_genes_multiabx.csv")
    # for abx in ["Van"]:
    abx = "Van"
    abx_metadata = metadata[(metadata["Drug"] == abx) | (metadata["Drug"] == "PBS")]
    abx_data = data[abx_metadata['ID']]
    column = "group"
    abx_metadata[column] = abx_metadata.apply(
        lambda row: row["Drug"] + "_" + row["Treatment"] if (
                (row["Drug"] == "Van") & (row["Treatment"] == "IP")) else "other", axis=1)

    path = "./Private/AbxRandomForest"
    feature_importance = pd.read_csv(f"{path}/feature_importance_VanIPvsAll.csv")
    feature_importance = feature_importance.set_index("gene id")
    # plot a heatmap of top 100 features
    genes = feature_importance.index[:100]
    # sort abx_data columns lexically
    # abx_data = abx_data.reindex(sorted(abx_data.columns), axis=1)
    # other = abx_metadata[abx_metadata["group"] == "other"]["ID"]
    # van = abx_metadata[abx_metadata["group"] == "Van_IP"]["ID"]
    # to_plot = abx_data[np.concatenate([other.values, van.values])].loc[genes]
    # # zscore rows
    # to_plot = to_plot.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    # sns.heatmap(data=to_plot, cmap="coolwarm")
    # plt.show()

    # four_way_random_forest_multitreat(sub_data, sub_metadata, abx, "group", abx=True, reps=10_000)
    # van_random_forest(abx, abx_data, abx_metadata, column)

    # plot_confusion_matrix(f"_{abx}IPvsAll", factor=1, path="./Private/AbxRandomForest",
    #                       order=["Van_IP", "other"])
    abx_metadata["group"] = metadata["Drug"] + "_" + metadata["Treatment"]

    # # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(800,), background=background_id,
    analyze_results(abx_data, abx_metadata, f"_{abx}IPvsAll", sizes=(400,), background=background_id,
                    # analyze_results(sub_data, sub_metadata, f"_{abx}", sizes=(50, 100, 200, 400, ), background=background_id,
                    treat=abx)


def van_random_forest(abx, abx_data, abx_metadata, column):
    title = abx
    reps = 10_000
    path = "./Private/AbxRandomForest"
    # add to data the group column from metadata
    abx_data = abx_data.T
    # intersecting_genes = fmt_data.index.intersection(abx_data.index)
    abx_data = pd.merge(abx_data, abx_metadata[["ID", column]], left_index=True, right_on="ID").set_index("ID")
    n = abx_metadata[column].nunique()
    confusion_matrix = np.zeros((n, n))
    classification_report = np.zeros((n, 4))
    importance = pd.Series(np.zeros(len(abx_data.columns[:-1])), index=abx_data.columns[:-1])
    from collections import Counter
    smallest = min(list(Counter([group_name for group_name in abx_data[column] if group_name != "PBS"]).values()))
    for i in range(reps):
        # Down sample:
        indexes = []
        indexes.extend(
            abx_data[abx_data["group"].str.startswith("Van")].sample(n=smallest, replace=False, random_state=i).index)
        indexes.extend(abx_data[abx_data["group"] == "other"].sample(n=smallest, replace=False, random_state=i).index)
        result = four_way_forest(abx_data.loc[indexes], abx_data.loc[indexes].columns[:-1], column,
                                 test_size=n * 1.5 / abx_data.loc[indexes].shape[0], random_state=i)
        confusion_matrix += result[0]
        classification_report += result[1]
        importance += result[2]
        labels_dict = result[3]
    confusion_matrix /= reps
    classification_report /= reps
    importance /= reps
    print(f"Confusion Matrix:")
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
    confusion_matrix.to_csv(path + f"/confusion_matrix_{title}IPvsAll.csv", index=True)
    classification_report = pd.DataFrame(classification_report, columns=["precision", "recall", "f1-score", "support"])
    print("\nClassification Report:")
    print(classification_report)
    # save feature importance
    importance = importance.sort_values(ascending=False)
    importance.to_csv(path + f"/feature_importance_{title}IPvsAll.csv", index=True)
    # return confusion_matrix
    return abx_data


def multi_treat_forest(random=False):
    _, metadata, _, data = read_process_files(new=False)
    data, metadata = transform_data(data, metadata, "RASflow", skip=True)
    # data.fillna(0, inplace=True)

    metadata["group"] = metadata["Drug"] + "_" + metadata["Treatment"]
    ensmus_to_gene = get_ensmus_dict()
    # background_id = background_analysis(data.rename(index=ensmus_to_gene).index)
    background_id = data.rename(index=ensmus_to_gene).index.to_list()
    # for abx in ["Van"]:
    for abx in antibiotics:
        sub_metadata = metadata[(metadata["Drug"] == abx) | (metadata["Drug"] == "PBS")]
        sub_data = data[sub_metadata['ID']]

        four_way_random_forest_multitreat(sub_data, sub_metadata, abx, "group", abx=True, reps=10_000, random=random)

        plot_confusion_matrix(f"_{abx}", factor=1, path="./Private/AbxRandomForest",
                              order=[abx + f"_{treat}" for treat in treatments] + ["PBS"], random=random)
        # # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(800,), background=background_id,
        # analyze_results(sub_data, sub_metadata, f"_{abx}", sizes=(400,), background=background_id,
        #                 # analyze_results(sub_data, sub_metadata, f"_{abx}", sizes=(50, 100, 200, 400, ), background=background_id,
        #                 treat=abx)
        # analyze_results(sub_data, sub_metadata, f"_{abx}", sizes=(800, 1600, ), background=background_id,


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
    # "AMP": 4,
    # "MIX": 4,
    # "NEO": 4,
    # "VAN": 4,
    # "MET": 4,
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
    plt.savefig(f"./Private/AbxRandomForest/feature_importance_cum-sum{title}.png")

    # Show the plot
    plt.show()

    # # Print the number of features needed for 80% and 90% importance
    # features_80 = len(feature_importance_sorted[feature_importance_sorted['cumsum_percent'] <= 80])
    # features_90 = len(feature_importance_sorted[feature_importance_sorted['cumsum_percent'] <= 90])
    #
    # print(f"Number of features needed for 80% importance: {features_80}")
    # print(f"Number of features needed for 90% importance: {features_90}")


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

    feature_importance = pd.read_csv(f"./Private/AbxRandomForest/feature_importance{title}.csv", index_col=0)
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

        # plt.scatter(top_df.columns, top_df.loc["Gsk3b"])
        # options = ["PBS_IP", "Amp_IP", "Met_IP", "Neo_IP", "Van_IP", "Mix_IP"]
        # means = [np.mean(top_df.loc["Gsk3b"][abx]) for abx in options]
        # plt.scatter(options, means, c="r", marker="x")
        # plt.title("Gsk3b")
        # plt.show()
        # background = background_analysis(top_df.index)
        dynamic_tree_plot(top_df, [gene for gene in background if type(gene) is str], factors[treat],
                          title + f"_{num_top}")
        continue

        # dynamic_tree_plot(top_df, False, factors[treat], title)
        # dynamic_tree_plot(top_df, False, factors[treat], title)
        # continue
        if "Cluster" in top_df.columns:
            top_df = top_df.drop("Cluster", axis=1)

        # cluster the genes using hierarchical clustering
        cluster = sns.clustermap(top_df, row_cluster=True, metric="euclidean", method="average",
                                 col_cluster=False, z_score=0)
        # increase x labels font size
        plt.xticks(fontsize=16)
        # get the order of the genes after clustering
        # genes = cluster.data2d.index
        genes = cluster.dendrogram_row.reordered_ind
        # show all y labels
        plt.yticks(np.arange(len(top_df.index)), top_df.index, rotation=0, fontsize=8)
        # remove y label
        plt.ylabel("")
        # increase figure size
        plt.gcf().set_size_inches(15, 15 * num_top / 100)
        # save the clustermap
        plt.savefig(
            f"./Private/AbxRandomForest/feature_importance_{num_top}_clustermap{'_' + abx if abx else ''}{title}.png")
        plt.close()
        top_df = top_df.iloc[genes]
        top_df.to_csv(
            f"./Private/AbxRandomForest/feature_importance_{num_top}_clustered{'_' + abx if abx else ''}{title}.csv")
        treat = title.split("_")[-1].upper()
        colors = {f'PBS_{treat}': light_blue, f'Amp_{treat}': 'red',
                  f'Met_{treat}': orange, f'Van_{treat}': 'purple',
                  f'Neo_{treat}': 'yellow', f'Mix_{treat}': 'pink'} if treat in treatments else {
            "PBS_IP": light_blue, 'PBS_IV': "blue", "PBS_PO": "green",
            f"{treat.capitalize()}_IP": "red", f'{treat.capitalize()}_IV': "orange",
            f"{treat.capitalize()}_PO": "yellow",
        }

        plot_heatmap_colors(cluster, False, f"feature_importance_{num_top}{title}_heatmap", top_df, colors=colors,
                            jump=False, path="./Private/AbxRandomForest", set_max=False, multiabx=True)


# def dynamic_tree_cut(link, dist, factor, depth=4):
#     from scipy.cluster.hierarchy import inconsistent, fcluster
#
#     # Calculate inconsistency statistics
#     incons = inconsistent(link)
#
#     # Determine cut threshold based on inconsistency
#     threshold = np.mean(incons[:, -1]) + factor * np.std(incons[:, -1])
#
#     clusters = fcluster(link, t=threshold, criterion='inconsistent', depth=depth)
#     # Perform clustering
#     return clusters


# def dynamic_tree_cut(link, distances, factor, depth=4):
#     # Calculate inconsistency statistics
#     incons = inconsistent(link)
#
#     # Determine initial cut threshold based on inconsistency
#     threshold = np.mean(incons[:, -1]) + 3 * np.std(incons[:, -1])
#     # threshold = np.mean(incons[:, -1]) + factor * np.std(incons[:, -1])
#
#     # Perform initial clustering
#     initial_clusters = fcluster(link, t=threshold, criterion='inconsistent', depth=depth)
#
#     # Calculate distances between points in the original data
#     distance_matrix = squareform(distances)
#
#     # Create a linkage tree
#     tree, nodes = to_tree(link, rd=True)
#
#     # Convert clusters to a dictionary format for easy access
#     clusters = {i: np.where(initial_clusters == i)[0].tolist() for i in np.unique(initial_clusters)}
#
#     merged = True
#     while merged:
#         merged = False
#         # Traverse the dendrogram tree and check adjacent clusters
#         for node in nodes:
#             if node.is_leaf():
#                 continue
#             left_cluster = get_leaves(node.get_left(), initial_clusters, clusters)
#             right_cluster = get_leaves(node.get_right(), initial_clusters, clusters)
#
#             left_cluster_id = initial_clusters[left_cluster[0]]
#             right_cluster_id = initial_clusters[right_cluster[0]]
#             if right_cluster_id == left_cluster_id:
#                 continue
#
#             within_left_var = calculate_within_cluster_variance(left_cluster, distance_matrix)
#             within_right_var = calculate_within_cluster_variance(right_cluster, distance_matrix)
#             between_var = calculate_between_cluster_variance(left_cluster, right_cluster, distance_matrix)
#
#             if within_left_var < between_var and within_right_var < between_var:
#                 # Merge right cluster into left cluster
#                 clusters[left_cluster_id] += right_cluster
#                 for index in right_cluster:
#                     initial_clusters[index] = left_cluster_id
#                 del clusters[right_cluster_id]
#                 merged = True
#
#     # Final cluster labels
#     final_clusters = np.zeros(len(initial_clusters), dtype=int)
#     for cluster_id, indices in clusters.items():
#         final_clusters[indices] = cluster_id
#
#     return final_clusters

import numpy as np
from scipy.spatial.distance import squareform


def calculate_within_cluster_variance(cluster_indices, distance_matrix):
    if len(cluster_indices) <= 1:
        return 0
    within_cluster_dists = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
    return np.var(within_cluster_dists[np.triu_indices(len(cluster_indices), k=1)])


def calculate_between_cluster_variance(cluster_indices_1, cluster_indices_2, distance_matrix):
    between_cluster_dists = distance_matrix[np.ix_(cluster_indices_1, cluster_indices_2)]
    return np.var(between_cluster_dists.flatten())


def get_leaves(node):
    if node.is_leaf():
        return [node.id]
    else:
        return get_leaves(node.left) + get_leaves(node.right)


def dynamic_tree_cut_old(data, distances, max_clusters=None, max_intra_to_inter_ratio=1.0):
    from scipy.spatial.distance import squareform
    """
    Perform clustering by comparing in-group variance to between-group variance.

    Parameters:
    -----------
    data : numpy.ndarray
        Input data matrix where each row is a sample
    max_clusters : int, optional
        Maximum number of clusters to create. If None, determined dynamically.
    max_intra_to_inter_ratio : float, optional
        Threshold for merging clusters based on variance ratio

    Returns:
    --------
    numpy.ndarray
        Cluster labels for each sample
    """
    # Compute pairwise distances
    dist_matrix = squareform(distances)
    n_samples = len(data)

    # Initialize each sample as its own cluster
    clusters = [{i} for i in range(n_samples)]

    def calculate_cluster_variance(cluster_indices):
        if len(cluster_indices) <= 1:
            return 0

        # Select distances within the cluster
        cluster_dist = dist_matrix[np.ix_(list(cluster_indices), list(cluster_indices))]
        return np.var(cluster_dist[np.triu_indices(len(cluster_indices), k=1)])

    def calculate_between_cluster_variance(cluster1, cluster2):
        # Compute distances between all points in two clusters
        between_dist = dist_matrix[np.ix_(list(cluster1), list(cluster2))]
        return np.var(between_dist)

    # Clustering process
    while len(clusters) > (max_clusters or 1):
        # Find the two closest clusters to merge
        best_merge = None
        best_merge_score = float('inf')

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Compute cluster variances
                intra_var1 = calculate_cluster_variance(clusters[i])
                intra_var2 = calculate_cluster_variance(clusters[j])
                between_var = calculate_between_cluster_variance(clusters[i], clusters[j])

                # Compute merge score
                # Lower score means more similar clusters
                merge_score = (intra_var1 + intra_var2) / (between_var + 1e-10)

                if merge_score < best_merge_score:
                    best_merge = (i, j)
                    best_merge_score = merge_score

        # Check if merging is beneficial
        if best_merge_score > max_intra_to_inter_ratio:
            break

        # Merge the best pair of clusters
        i, j = best_merge
        merged_cluster = clusters[i].union(clusters[j])

        # Remove old clusters and add new merged cluster
        clusters = [
            cluster for k, cluster in enumerate(clusters)
            if k not in {i, j}
        ]
        clusters.append(merged_cluster)

    # Assign cluster labels
    cluster_labels = np.zeros(n_samples, dtype=int)
    for label, cluster in enumerate(clusters):
        for sample in cluster:
            cluster_labels[sample] = label

    return cluster_labels


def dynamic_tree_cut_ready_made(df, size_penalty_factor=5, min_cluster_size=5):
    # from sklearn.cluster import HDBSCAN
    # hdb = HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=0.2)
    # hdb.fit(df)
    # return hdb.labels_
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=5).fit(df)
    return clustering.labels_


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


# def check_enrichment(genes, background):
#     import json
#     genes = [gene for gene in genes if type(gene) is str]
#     description = ",".join(genes)
#
#     db = "GO_Biological_Process_2023"
#     if background:
#         base_url = "https://maayanlab.cloud/speedrichr"
#         res = requests.post(
#             base_url + '/api/addList',
#             files=dict(
#                 list=(None, '\n'.join(genes)),
#                 description=(None, description),
#             )
#         )
#         if res.ok:
#             userlist_response = res.json()
#             response = userlist_response["userListId"]
#             ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/view?userListId=%s'
#             req_res = requests.get(ENRICHR_URL % response)
#             if req_res.ok:
#                 uploaded_genes = json.loads(req_res.text)["genes"]
#                 if sorted([gene.lower() for gene in genes]) != sorted([gene.lower() for gene in uploaded_genes]):
#                     print(sorted([gene.lower() for gene in genes]), '\n', sorted([gene.lower() for gene in uploaded_genes]))
#                 # assert sorted([gene.lower() for gene in genes]) == sorted([gene.lower() for gene in uploaded_genes])
#         else:
#             return
#         base_url = "https://maayanlab.cloud/speedrichr"
#
#         res = requests.post(
#             base_url + '/api/backgroundenrich',
#             data=dict(
#                 userListId=response,
#                 backgroundid=background,
#                 backgroundType=db,
#             )
#         )
#         if res.ok:
#             results = res.json()
#             return results[db]
#     else:
#         ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/addList'
#         payload = {
#             'list': (None, '\n'.join(genes)),
#             'description': (None, description)
#         }
#
#         res = requests.post(ENRICHR_URL, files=payload)
#         if res.ok:
#             userlist_response = json.loads(res.text)
#             response = userlist_response["userListId"]
#             ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/view?userListId=%s'
#             req_res = requests.get(ENRICHR_URL % response)
#             if req_res.ok:
#                 uploaded_genes = json.loads(req_res.text)["genes"]
#                 assert sorted([gene.lower() for gene in genes]) == sorted([gene.lower() for gene in uploaded_genes])
#         else:
#             return
#
#         ENRICHR_URL = 'https://maayanlab.cloud/Enrichr/enrich'
#         query_string = '?userListId=%s&backgroundType=%s'
#         res = requests.get(
#             ENRICHR_URL + query_string % (response, db)
#         )
#         if res.ok:
#             results = json.loads(res.text)
#             return results[db]


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
            # print(cluster_id, ": ", indexes, "\n", enrichment)
            # enrichment = pd.DataFrame(enrichment, columns=["index", "Name", "P-value", "Odds Ratio", "Combined score",
            #                                                 "Overlap Genes", "Adjusted p-value",
            #                                                 "old p-value", "old Adjusted p-value"])
            # names[cluster_id] = enrichment
            enrichment["cluster_id"] = mapping[cluster_id]
            enrichment["#of_genes"] = enrichment["Genes"].str.count(";") + 1
            enrichment["All_Genes"] = ",".join([gene for gene in indexes if type(gene) is str])
            enrichment_analysis = pd.concat([enrichment_analysis, enrichment])

    # enrichment_analysis = enrichment_analysis.drop("index", axis=1)
    enrichment_analysis = enrichment_analysis.drop(["Old P-value", "Old adjusted P-value"], axis=1)
    enrichment_analysis["adj.P-val<5%"] = enrichment_analysis["Adjusted P-value"] < 0.05
    enrichment_analysis["P-val<5%"] = enrichment_analysis["P-value"] < 0.05
    enrichment_analysis.to_csv(
        f"./Private/AbxRandomForest/cluster_dynamic{title}_enrichment_{'background' if background else ''}.csv",
        index=False)
    enrichment_analysis[enrichment_analysis["Adjusted P-value"] < 0.05].to_csv(
        f"./Private/AbxRandomForest/cluster_dynamic{title}_enrichment_{'background' if background else ''}_filtered.csv",
        index=False)

    # Create a color palette for the clusters
    n_clusters = len(np.unique(clusters))
    print(n_clusters)
    # if n_clusters < 20:
    #     palette = sns.color_palette("tab20")
    # elif n_clusters < 25:
    #     custom_palette = ['#FFF7EC', '#FEE8C8', '#FDD49E', '#FDBB84', '#FC8D59', '#EF6548', '#D7301F', '#990000',
    #                       '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F',
    #                       '#BCBD22', '#17BECF', '#EDF8FB', '#BFD3E6', '#9EBCDA', '#8C96C6', '#8C6BB1', '#88419D', '#6E016B']
    #     np.random.shuffle(custom_palette)
    #     palette = sns.color_palette(custom_palette[:n_clusters])
    # else:
    #     palette = sns.color_palette("husl", n_colors=n_clusters + 1)
    #     np.random.shuffle(palette)
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
    plotted_data.to_csv(f"./Private/AbxRandomForest/cluster_dynamic{title}.csv")

    # # Adjust the colorbar label
    # g.ax_cbar.set_ylabel('Standardized Values')
    # Add cluster labels to the dendrogram
    for i, c in enumerate(np.unique(clusters)):
        color_cluster = palette[c % len(palette)] if c != -1 else (0, 0, 0)
        g.ax_row_dendrogram.bar(0, 0, color=color_cluster, label=f'Cluster {c}', linewidth=0)
    g.ax_row_dendrogram.legend(title='Clusters', loc="center")  # , bbox_to_anchor=(0.5, 0.8))
    plt.title('Clustered Heatmap with Dendrogram', y=1.02)
    plt.tight_layout()
    plt.savefig(
        f"./Private/AbxRandomForest/cluster_dynamic{title}.png")
    # plt.show()
    plt.close()


def sonia_forest():
    data = pd.read_csv("../Data/Sonia/Partek_Sonia_GF_exp13_Normalization_Normalized_counts_for_Deseq2.csv",
                       index_col=0)
    metadata = pd.read_csv("../Data/Sonia/metadata for Random forest.txt", sep='\t')
    # rename "Sample" to "ID" column
    metadata.columns = ["ID", "Condition"]
    four_way_random_forest_multiabx(data, metadata, "Sonia", "Condition", abx=False, reps=10000)


if __name__ == "__main__":
    # multi_abx_forest()
    multi_treat_forest()
    # van_forest()
    # multi_treat_forest(random=True)
    # sonia_forest()
