import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform  # For efficient distance calculation
from scipy.stats import gmean  # For your CLR function
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ClusteringGO import treatments, antibiotics


def pca_fastspar_prepare(data, metadata, groups, column, addition='', to_cond=True):
    for group in groups:
        # treat_df = pd.DataFrame()
        treat = group if group != "PO" else "gavage"
        cond = metadata[column] == treat if column == "treatment" else (
                (metadata[column] == treat) | (metadata[column] == "PBS"))
        if to_cond:
            samples = metadata[(cond) & (metadata["Type"] == "feces")]['#SampleID']
        else:
            samples = metadata[cond]['#SampleID']
        # for file in os.listdir("./Private/selbal-OTU/OTU/"):
        #     # if file is not a csv file, pass
        #     if file.split(".")[-1] != "tsv" or file.split("-")[1] != treatment or file.split("-")[2] != "feces.tsv":
        #         continue
        #     df = pd.read_csv(f"./Private/selbal-OTU/OTU/{file}", sep="\t", index_col=0)
        #     # merge rows with by column name
        #     treat_df = pd.concat([treat_df, df], axis=0)
        # # remove redundant PBS (added multiple times)
        # df = df.drop_duplicates()
        in_data = [sample for sample in samples if sample in data.columns]
        treat_df = data[in_data]
        # remove empty rows
        treat_df = treat_df.loc[:, (treat_df != 0).any(axis=0)]
        # optional = pd.read_csv(f'../Data/Abx_16s_data/export_tables/merged_table_all_d4-{treat}-feces-only_exported-feature-table/merged_table_all_d4-{treat}-feces-only.csv').set_index("OTU ID")

        # replace nan with 0
        df = treat_df.fillna(0)

        df = df.T
        # remove all columns with only 0 values
        df = df.loc[:, (df != 0).any(axis=0)]
        # df = df.T
        # change the index name to #OTU ID
        df.index.name = "#OTU ID"
        df.to_csv(f"./Private/selbal-OTU/OTU/fastspar/{group}{addition}.tsv", sep="\t")


def explain_var(matrix, title, path):
    # calculate the variance explained by each PC
    pca = PCA()
    pca.fit(matrix)
    var = pca.explained_variance_ratio_
    plt.scatter(range(1, len(var) + 1), np.cumsum(var))
    plt.title(f"Variance Explained by PCs, {title}")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.savefig(path + f"variance_explained_{title}.png")
    plt.show()


def cailliez_correction(matrix):
    # from skbio.stats.distance import DistanceMatrix

    # Calculate the F matrix
    n = matrix.shape[0]
    ones_n = np.ones((n, n))
    identity_n = np.eye(n)
    F = -0.5 * (matrix ** 2 - np.dot(matrix ** 2, ones_n) / n - np.dot(ones_n, matrix ** 2) / n + np.dot(
        ones_n, np.dot(matrix ** 2, ones_n)) / (n ** 2))

    # Perform eigen decomposition on F
    eigenvalues, eigenvectors = np.linalg.eigh(F)

    # Find the smallest eigenvalue. If it's negative, that's our correction factor.
    k = np.min(eigenvalues)
    if k < 0:
        # Adjust F to make the smallest eigenvalue non-negative
        F_corrected = F + (np.abs(k) + 0.0001) * identity_n  # Small constant to ensure positivity
    else:
        F_corrected = F

    # Recompute the corrected dissimilarity matrix using the corrected F
    corrected_dissimilarity_matrix = np.sqrt(np.max(F_corrected - np.diag(np.diag(F_corrected)), 0))

    # # Now, you can use this corrected dissimilarity matrix for PCoA
    # # Convert corrected dissimilarity matrix to DistanceMatrix object required by skbio's PCoA
    # corrected_distance_matrix = DistanceMatrix(squareform(corrected_dissimilarity_matrix),
    #                                            ids=[str(i) for i in range(corrected_dissimilarity_matrix.shape[0])])
    # return corrected_distance_matrix
    return corrected_dissimilarity_matrix


def get_default_colors(types):
    """
    Returns the first n default colors from the matplotlib color cycle.

    Parameters:
    - n: The number of colors to return.

    Returns:
    - A list of color codes.
    """
    n = len(types)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {types[i]: color_cycle[i] for i in range(n)}


def correction(matrix, correction):
    if correction == "const":
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Find the smallest eigenvalue
        min_eigenvalue = np.min(eigenvalues)

        # If the smallest eigenvalue is negative, add its absolute value plus a small constant to all eigenvalues
        if min_eigenvalue < 0:
            correction = np.abs(min_eigenvalue) + 0.0001  # Small constant to ensure positivity
            corrected_eigenvalues = eigenvalues + correction
        else:
            corrected_eigenvalues = eigenvalues

        # Recompute the dissimilarity matrix with corrected eigenvalues
        res = np.dot(eigenvectors, np.dot(np.diag(corrected_eigenvalues), eigenvectors.T))
        # make res a df with original index and columns
        matrix = pd.DataFrame(res, index=matrix.index, columns=matrix.columns)
    elif correction == "cailliez":
        res = cailliez_correction(matrix)
        matrix = pd.DataFrame(res, index=matrix.index, columns=matrix.columns)


from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def get_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """
    Create an ellipse patch representing the covariance of x and y
    """
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(np.mean(x), np.mean(y))

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def pcoa(matrix, metadata_df, title, color_group, d0=False, correction=None, days=None, addition=''):
    # Principal Coordinate Analysis (PCoA)
    pca = PCA(n_components=2)
    pcoa_result = pca.fit_transform(matrix)

    # Merge metadata with PCoA results
    pcoa_df = pd.DataFrame(pcoa_result, index=matrix.index, columns=['PCoA1', 'PCoA2'])

    merged_pcoa_df = pd.merge(metadata_df, pcoa_df, left_on='#SampleID', right_index=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.title(title, size=16)
    # plt.title(f'PCoA for All {title} Antibiotics', size=16)

    # x_factor = 20 if title != "IP" else 200
    # # change xlim and y lim to be the closest half integer to the max and min values
    # plt.xlim((round(x_factor * min(merged_pcoa_df['PCoA1'])) / x_factor) - (1 / (10 * x_factor)),
    #          (round(x_factor * max(merged_pcoa_df['PCoA1'])) / x_factor) + (1 / (10 * x_factor)))
    # plt.ylim((np.floor(200 * min(merged_pcoa_df['PCoA2'])) / 200) - 0.0005,
    #          (round(200 * max(merged_pcoa_df['PCoA2'])) / 200) + 0.005)
    # add amount of variance explained by each PC
    plt.xlabel(f'PCoA1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)', size=16)
    plt.ylabel(f'PCoA2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)', size=16)

    if not d0:
        # Create a color cycle
        colors = list(mcolors.TABLEAU_COLORS.values())
        color_cycle = (colors * (len(merged_pcoa_df[color_group].unique()) // len(colors) + 1))[
                      :len(merged_pcoa_df[color_group].unique())]
        shapes = {"IP": "^", "IV": "s", "gavage": "h"}
        abx_color = {"PBS": '#1f77b4', "abx": '#ff7f0e'}
        not_printed_pbs = True
        for (abx, group), color in zip(merged_pcoa_df.groupby(color_group), color_cycle):
            # ax.scatter(group['PCoA1'], group['PCoA2'], label=abx)
            # Scatter plot
            if color_group != "antibiotic_treatment":
                ax.scatter(group['PCoA1'], group['PCoA2'], label=abx, color=color)
                # Add ellipse
                get_ellipse(group['PCoA1'], group['PCoA2'], ax, n_std=2.0,
                            facecolor=color, edgecolor=color, alpha=0.2)
            else:
                ax.scatter(group['PCoA1'], group['PCoA2'], label=abx, color=abx_color.get(abx.split("_")[0], '#ff7f0e'),
                           marker=shapes[abx.split("_")[1]])
                if "PBS" not in abx:
                    # Add ellipse
                    get_ellipse(group['PCoA1'], group['PCoA2'], ax, n_std=2.0,
                                facecolor=color, edgecolor=color, alpha=0.2)
                elif not_printed_pbs:
                    pbs_samples = merged_pcoa_df[merged_pcoa_df["antibiotic"] == "PBS"]
                    get_ellipse(pbs_samples['PCoA1'], pbs_samples['PCoA2'], ax, n_std=2.0,
                                facecolor=color, edgecolor=color, alpha=0.2)
                    not_printed_pbs = False

        # Insert the spectrum plot
        # calculate the variance explained by each PC
        pca = PCA()
        pca.fit(matrix)
        var = pca.explained_variance_ratio_
        sub_loc = {
            "IV": [0.57, 0.66, 0.25, 0.2],
            "IP": [0.58, 0.66, 0.25, 0.25],
            "PO": [0.55, 0.55, 0.25, 0.25],

            "amp": [0.5, 0.67, 0.25, 0.2],
            "van": [0.5, 0.45, 0.25, 0.2],
            "met": [0.7, 0.48, 0.25, 0.2],
            "neo": [0.45, 0.5, 0.25, 0.2],
            "mix": [0.4, 0.62, 0.25, 0.2]
        }
        ax.legend()
    else:
        # colors = get_default_colors(merged_pcoa_df[color_group].unique())
        # Create a color cycle
        colors = list(mcolors.TABLEAU_COLORS.values())
        color_cycle = (colors * (len(merged_pcoa_df[color_group].unique()) // len(colors) + 1))[
                      :len(merged_pcoa_df[color_group].unique())]
        shapes = {"IP": "^", "IV": "s", "gavage": "h"}
        abx_color = {"PBS": '#1f77b4', "abx": '#ff7f0e'}

        day0 = merged_pcoa_df.set_index("#SampleID").loc[days[0]]
        # for abx, group in day0.groupby(color_group):
        #     ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (d0)", edgecolor=colors[abx], facecolors='none')
        for (abx, group), color in zip(day0.groupby(color_group), color_cycle):
            # ax.scatter(group['PCoA1'], group['PCoA2'], label=abx)
            # Scatter plot
            if color_group != "antibiotic_treatment":
                ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (d0)", edgecolor=color, facecolors='none')
            else:
                ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (d0)",
                           edgecolor=abx_color.get(abx.split("_")[0], '#ff7f0e'),
                           marker=shapes[abx.split("_")[1]], facecolors='none')
        # Add ellipse for all d0
        get_ellipse(day0['PCoA1'], day0['PCoA2'], ax, n_std=2.0,
                    facecolor='gray', edgecolor='gray', alpha=0.2)
        day4 = merged_pcoa_df.set_index("#SampleID").loc[days[1]]
        # for abx, group in day4.groupby(color_group):
        #     ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (d4)", color=colors[abx])
        #     # Add ellipse
        #     get_ellipse(group['PCoA1'], group['PCoA2'], ax, n_std=2.0,
        #                 facecolor=colors[abx], edgecolor=colors[abx], alpha=0.2)
        not_printed_pbs = True
        for (abx, group), color in zip(day4.groupby(color_group), color_cycle):
            # ax.scatter(group['PCoA1'], group['PCoA2'], label=abx)
            # Scatter plot
            if color_group != "antibiotic_treatment":
                ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (EP)", color=color)
                # ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (d4)", color=color)
                # Add ellipse
                get_ellipse(group['PCoA1'], group['PCoA2'], ax, n_std=2.0,
                            facecolor=color, edgecolor=color, alpha=0.2)
            else:
                ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (EP)",
                # ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (d4)",
                           color=abx_color.get(abx.split("_")[0], '#ff7f0e'),
                           marker=shapes[abx.split("_")[1]])
                if "PBS" not in abx:
                    # Add ellipse
                    get_ellipse(group['PCoA1'], group['PCoA2'], ax, n_std=2.0,
                                facecolor=color, edgecolor=color, alpha=0.2)
                elif not_printed_pbs:
                    pbs_samples = merged_pcoa_df[merged_pcoa_df["antibiotic"] == "PBS"]
                    get_ellipse(pbs_samples['PCoA1'], pbs_samples['PCoA2'], ax, n_std=2.0,
                                facecolor=color, edgecolor=color, alpha=0.2)
                    not_printed_pbs = False
        if len(days) > 2:
            day1 = merged_pcoa_df.set_index("#SampleID").loc[days[2]]
            for abx, group in day1.groupby(color_group):
                ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (d1)", edgecolor=colors[abx],
                           facecolors='gray')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    loc = "best" if title != "IV" else "center right"
    # ax.legend(loc=loc)
    plt.tight_layout()
    plt.savefig(f"./Private/dimension reduction/{title}_pcoa{addition}.png", dpi=300)
    plt.show()


def geometric_mean(x):
    """Calculate the geometric mean of a vector"""
    return np.exp(np.mean(np.log(x)))


def clr_transformation(x, epsilon=1e-5):
    """Compute the centered log-ratio transformation of a composition vector,
    adding a small constant to handle zeros."""
    x_modified = np.where(x == 0, epsilon, x)
    gm = geometric_mean(x_modified)
    return np.log(x_modified / gm)


def arcsin_sqrt_transformation(x, epsilon=None):
    """
    Compute the arcsin-sqrt transformation.
    Assumes x is a vector of proportions (sums to 1).
    """
    # Clip to [0, 1] to avoid domain errors from floating-point inaccuracies
    # x_clipped = np.clip(x, 0, 1)
    # convert values to proportions
    x_clipped = x / x.sum()
    return np.arcsin(np.sqrt(x_clipped))


def log_transformation(x, epsilon=1e-9):
    """
    Compute the log transformation with a small pseudocount.
    Often used as log(x + 1) for counts, but log(x + epsilon)
    is better for proportions.
    """
    return np.log(x + epsilon)


def logit_transformation(x, epsilon=1e-9):
    """
    Compute the element-wise logit transformation.
    Assumes x is a vector of proportions.
    Handles 0s and 1s by clipping to [epsilon, 1-epsilon].
    """
    # x_clipped = np.clip(x, epsilon, 1 - epsilon)
    x_clipped = x / x.sum()
    # replace 0s with epsilon
    x_clipped[x_clipped == 0] = epsilon
    return np.log(x_clipped / (1 - x_clipped))


def aitchison_distance(x, y):
    """Calculate the Aitchison distance between two composition vectors"""
    clr_x = clr_transformation(x)
    clr_y = clr_transformation(y)
    return np.sqrt(np.sum((clr_x - clr_y) ** 2))


def clr_transform(matrix):
    # CLR transformation
    matrix += 1e-8
    clr_matrix = np.log(matrix / matrix.mean(axis=0))
    return clr_matrix


def plot_pca(matrix, name):
    # Perform CLR transformation
    clr_matrix = clr_transform(matrix[matrix.columns[:-3]])

    explain_var(clr_matrix, name, f"./Private/dimension reduction/fastspar_results/PCA/")

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(clr_matrix)

    # Plot the first two principal components
    # plt.scatter(pca_result[:, 0], pca_result[:, 1])
    matrix['PC1'] = pca_result[:, 0]
    matrix['PC2'] = pca_result[:, 1]
    sns.scatterplot(data=matrix, x='PC1', y='PC2', hue='antibiotic')
    plt.title(f'PCA of CLR Transformed Matrix, {name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(f"./Private/dimension reduction/fastspar_results/PCA/{name}_pca.png")
    plt.show()


def clr_pca_trial(data, metadata, taxonomy):
    for treat in treatments:
        treat = treat if treat != "PO" else "gavage"
        sub_meta = metadata[(metadata['treatment'] == treat) & (metadata["Type"] == "feces")]
        samples = [sample for sample in sub_meta['#SampleID'] if sample in data.index]
        sub_data = data.loc[samples]
        sub_data = sub_data.fillna(0)
        # drop empty columns
        sub_data = sub_data.loc[:, (sub_data != 0).any(axis=0)]
        plot_pca(sub_data, f"{taxonomy} level, {treat}")


def plot_hist_fastspar(df, x_label, title, reduce_identity, x_lim):
    if reduce_identity:
        df = df - np.eye(len(corr))
    if x_lim:
        # plt.xlim(x_lim)
        # make all values below and above the limit to be the Nan
        df[df < x_lim[0]] = np.nan
        df[df > x_lim[1]] = np.nan
    plt.hist(df.values.flatten(), bins=100, density=True)
    plt.title(f"Correlation Distribution, {treat}\n "
              f"mean={round(np.nanmean(df.values.flatten()), 3)}, std={round(np.nanstd(df.values.flatten()), 3)}\n"
              f"mean square value={round(np.nanmean((df.values.flatten() ** 2)), 10)}\n"
              f"positives:{np.nansum(df.values.flatten() > 0)}, negatives:{np.nansum(df.values.flatten() < 0)}")
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.savefig(f"./Private/dimension reduction/fastspar_results/{treat}_{title}_dist.png")
    plt.show()


def calculate_pearson_correlation(X, Y):
    # Ensure the lists have the same length
    if len(X) != len(Y):
        raise ValueError("Lists X and Y must have the same length.")

    # Step 1: Calculate the means of X and Y
    mean_X = sum(X) / len(X)
    mean_Y = sum(Y) / len(Y)

    # Step 2 and 3: Compute deviations from the mean, their product, and sum of these products
    deviation_products = [(x - mean_X) * (y - mean_Y) for x, y in zip(X, Y)]
    sum_of_products = sum(deviation_products)

    # Step 4 and 5: Calculate squared deviations and their sums
    squared_deviations_X = [(x - mean_X) ** 2 for x in X]
    squared_deviations_Y = [(y - mean_Y) ** 2 for y in Y]
    sum_of_squared_deviations_X = sum(squared_deviations_X)
    sum_of_squared_deviations_Y = sum(squared_deviations_Y)

    # Step 6: Compute the Pearson correlation coefficient
    r_XY = sum_of_products / ((sum_of_squared_deviations_X * sum_of_squared_deviations_Y) ** 0.5)

    return r_XY


def run_pcoa(metadata, run_treats, cov, fastspar, bad, d0, qiime):
    iterable = treatments if run_treats else [abx.lower() for abx in antibiotics]
    for treat in iterable:
        if fastspar:
            corr = pd.read_csv(f"./Private/dimension reduction/fastspar_results/median_correlation_{treat}.tsv",
                               sep="\t")
            corr = corr.set_index("#OTU ID")
        elif d0 and not qiime:
            treat_name = treat if treat != "PO" else "gavage"
            data = pd.read_csv(f"../Data/MultiAbx-16s/otu_table_feces.txt", sep="\t").set_index("OTU ID")
            metadata = pd.read_csv("../Data/MultiAbx-16s/metadata_feces.tsv", sep="\t")
            metadata['antibiotic_treatment'] = metadata['antibiotic'] + "_" + metadata['treatment']
            samples = metadata[metadata['antibiotic'] == treat_name.lower()]['#SampleID']
            # samples = metadata[metadata['treatment'] == treat_name]['#SampleID']
            in_data = [sample for sample in samples if ((sample in data.columns) and ("d1" not in sample))]
            # in_data = [sample for sample in samples if sample in data.columns]
            data = data[in_data]
            distance = pd.DataFrame(index=data.columns, columns=data.columns)
            for col1 in data.columns:
                for col2 in data.columns:
                    # corr.loc[col1, col2] = calculate_pearson_correlation(data[col1], data[col2])
                    distance.loc[col1, col2] = aitchison_distance(data[col1], data[col2])
        elif qiime:
            data = pd.read_csv(f"./Private/selbal-OTU/OTU/fastspar/{treat}_qiime.tsv", sep="\t").set_index("#OTU ID").T
            d4_cols = [col for col in data.columns if
                       (metadata[metadata["#SampleID"] == col]["day"] == 4).values[0]]
            if d0:
                d0_cols = [col for col in data.columns if
                           (metadata[metadata["#SampleID"] == col]["day"] == 0).values[0]]
                d4_cols += d0_cols
                in_data = d4_cols
            data = data[d4_cols]

            corr = pd.DataFrame(index=data.columns, columns=data.columns)
            distance = pd.DataFrame(index=data.columns, columns=data.columns)
            # calc correlations between rows
            # corr = data.T.corr()
            # Calculate all-to-all correlations manually
            for col1 in data.columns:
                for col2 in data.columns:
                    # corr.loc[col1, col2] = calculate_pearson_correlation(data[col1], data[col2])
                    distance.loc[col1, col2] = aitchison_distance(data[col1], data[col2])
        else:
            data = pd.read_csv(f"./Private/selbal-OTU/OTU/fastspar/{treat}.tsv", sep="\t").set_index("#OTU ID").T
            # todo: check 2 shows w/o .d4
            # todo: remove not-d4 samples
            # if there's a sample without d4, and there's another sample named the same with d4, remove the one without
            # d4. Otherwise keep it
            for sample in data.columns:
                parts = sample.split(".")
                if len(parts) > 1 and parts[0] in data.columns:
                    data = data.drop(parts[0], axis=1)
            # data.columns = [col.split(".")[0] for col in data.columns]
            if not bad:
                # remove C14, C17, C18
                if "C14" in data.columns:
                    data = data.drop(["C14", "C17", "C18"], axis=1)
            # # apply clr transformation
            # data = clr_transform(data)
            # Initialize an empty DataFrame for the correlation matrix
            corr = pd.DataFrame(index=data.columns, columns=data.columns)
            distance = pd.DataFrame(index=data.columns, columns=data.columns)
            # calc correlations between rows
            # corr = data.T.corr()
            # Calculate all-to-all correlations manually
            for col1 in data.columns:
                for col2 in data.columns:
                    # corr.loc[col1, col2] = calculate_pearson_correlation(data[col1], data[col2])
                    distance.loc[col1, col2] = aitchison_distance(data[col1], data[col2])

        # distance = np.identity(len(corr)) - corr
        # distance = 1 - corr
        # distance = np.sqrt(1 - (corr.astype('float64') ** 2))
        # # print eigenvalues of distance matrix, sorted
        # print(np.sort(np.linalg.eigvals(corr - np.eye(len(corr)))))
        # distance = np.sqrt((corr**2) - np.identity(len(corr)))

        if d0:
            group_d0 = [sample for sample in in_data if "d0" in sample]
            group_d4 = [sample for sample in in_data if "d0" not in sample]

            if run_treats:
                pcoa(distance, metadata, treat + " (including d0)", 'antibiotic', d0=d0,
                     days=[group_d0, group_d4])
            else:
                pcoa(distance, metadata, treat + " (including d0)", 'antibiotic_treatment', d0=d0,
                     days=[group_d0, group_d4])
            # group_d1 = [sample for sample in in_data if "d1" in sample]
            # pcoa(distance, metadata, treat + " (including d0-d1)", 'antibiotic_treatment', d0=d0, days=[group_d0, group_d4, group_d1])
        else:
            if run_treats:
                pcoa(distance, metadata, treat, 'antibiotic', addition="_qiime" if qiime else "")
            else:
                pcoa(distance, metadata, treat, 'antibiotic_treatment', addition="_qiime" if qiime else "")

        # # plot distribution of correlations
        # plot_hist_fastspar(corr, "Correlation", "corr", True, None)

        if cov:
            cov = pd.read_csv(f"./Private/dimension reduction/fastspar_results/median_covariance_{treat}.tsv", sep="\t")
            cov = cov.set_index("#OTU ID")
            pcoa(cov, metadata, treat)

            # # plot distribution of correlations
            # plot_hist_fastspar(cov, "Covariance", "cov", False, None)
            # plot_hist_fastspar(cov.copy(), "Covariance", "cov_low", False, (-0.2, 0.2))
            # plot_hist_fastspar(cov.copy(), "Covariance", "cov_high", False, (1, 2))
    print("done")


def run_referee_pcoa(metadata, run_treats):
    iterable = treatments if run_treats else [abx.lower() for abx in antibiotics]
    for treat in iterable:
        data = pd.read_csv(
            f"./Private/qiime/{treat}_qiime.tsv",
            sep="\t").set_index("#OTU ID").T
        d4_cols = [col for col in data.columns if
                   (metadata[metadata["#SampleID"] == col]["day"] == 4).values[0]]
        d0_cols = [col for col in data.columns if
                   (metadata[metadata["#SampleID"] == col]["day"] == 0).values[0]]
        d4_cols += d0_cols
        in_data = d4_cols
        data = data[d4_cols]

        data_transposed = data.T

        # A dictionary mapping the method name to its function
        transformations = {
            "Aitchison (CLR)": clr_transformation,
            "Arcsin-Sqrt": arcsin_sqrt_transformation,
            "Log": log_transformation,
            "Logit": logit_transformation
        }
        proportions = data_transposed.div(data_transposed.sum(axis=1), axis=0)
        minimal_read = proportions[proportions > 0].min().min()
        print(f"the minimal read is {minimal_read}")

        # --- 3. Run the Loop ---
        for method_name, transform_func in transformations.items():

            # 3a. Apply the transformation
            # We apply the function row-wise (axis=1) to each sample
            transformed_df = data_transposed.apply(
                lambda x: transform_func(x.values, minimal_read),
                axis=1,
                result_type='expand'
            )
            # Re-assign original column names (features) if lost
            transformed_df.columns = data_transposed.columns

            # 3b. Calculate the Euclidean distance matrix
            # pdist calculates a condensed distance matrix
            condensed_dist = pdist(transformed_df, 'euclidean')

            # squareform converts it into a full square matrix
            dist_matrix_values = squareform(condensed_dist)

            # Convert to a labeled DataFrame (required by many PCoA functions)
            distance_matrix = pd.DataFrame(
                dist_matrix_values,
                index=data_transposed.index,
                columns=data_transposed.index
            )

            group_d0 = [sample for sample in in_data if "d0" in sample]
            group_d4 = [sample for sample in in_data if "d0" not in sample]

            treats_dict = {
                "IP": "Intraperitoneal (IP)",
                "IV": "Intravenous (IV)",
                "PO": "Oral (PO)",
                "amp": "Ampicillin",
                "met": "Metronidazole",
                "neo": "Neomycin",
                "van": "Vancomycin",
                "mix": "Mix"
            }
            if run_treats:
                pcoa(distance_matrix, metadata, treats_dict[treat] + " [" + method_name + "]", 'antibiotic', d0=True,
                     days=[group_d0, group_d4])
            else:
                pcoa(distance_matrix, metadata, treats_dict[treat] + " [" + method_name + "]", 'antibiotic_treatment',
                     d0=True,
                     days=[group_d0, group_d4])


def referees_request():
    metadata = pd.read_csv("../Data/QIIME/qiime_metadata.tsv", sep="\t")
    # for d0 in [True, False]:
    run_referee_pcoa(metadata, True)
    run_referee_pcoa(metadata, False)


if __name__ == "__main__":
    referees_request()
    exit()
    metadata = pd.read_csv('../Data/Abx_16s_data/elad_metadata_merge_all.tsv', sep='\t')
    # for taxonomy in ["class", "family", "genus", "taxon"]:
    # for taxonomy in ["species"]:
    #     data = pd.read_csv(f'./Private/otu_merged_feces_{taxonomy}.tsv', sep='\t').set_index("sample_id")
    #     clr_pca_trial(data, metadata, taxonomy)
    #

    metadata = pd.read_csv('../Data/Abx_16s_data/elad_metadata_merge_all.tsv', sep='\t')
    data = pd.read_csv(
        '../Data/Abx_16s_data/export_tables/merged_table_all_d4_exported-feature-table/merged_table_all_d4.tsv',
        sep='\t').set_index("OTU ID")
    pca_fastspar_prepare(data, metadata, treatments, "treatment")
    pca_fastspar_prepare(data, metadata, [abx.lower() for abx in antibiotics], "antibiotic")

    qiime_metadata_orig = pd.read_csv(
        "./Data/mf_ok122_2.tsv",
        sep="\t")
    qiime_metadata_SB1 = pd.read_csv(
        "./Data/metadata_SB1.tsv",
        sep="\t")
    qiime_metadata_SB1["day"] = [4] * qiime_metadata_SB1.shape[0]
    # rename "TREATMENT" to "antibiotic" and "INJECTION" to "treatment"
    qiime_metadata_SB1 = qiime_metadata_SB1.rename(columns={"TREATMENT": "antibiotic", "INJECTION": "treatment"})
    # replace ampicilin with amp, pbs with PBS, vancomycin with van, metranedazol with met, neomycin with neo
    qiime_metadata_SB1['antibiotic'] = qiime_metadata_SB1['antibiotic'].replace(
        {"ampicilin": "amp", "pbs": "PBS", "vancomycin": "van", "metranedazol": "met", "neomycin": "neo"})
    qiime_metadata_SB1 = qiime_metadata_SB1[qiime_metadata_SB1["PART"] == "COLON"]
    # replace PO by gavage
    qiime_metadata_SB1['treatment'] = qiime_metadata_SB1['treatment'].replace({"PO": "gavage"})
    qiime_metadata = pd.concat([qiime_metadata_orig, qiime_metadata_SB1], axis=0)
    qiime_metadata['antibiotic_treatment'] = qiime_metadata['antibiotic'] + "_" + qiime_metadata['treatment']

    qiime_data_orig = pd.read_csv(
        "./Data/orig_species.tsv",
        sep="\t", skiprows=[0]).set_index("#OTU ID")
    qiime_data_SB1 = pd.read_csv(
        "./Data/SB1_species.tsv",
        sep="\t", skiprows=[0]).set_index("#OTU ID")
    qiime_data_SB1 = qiime_data_SB1[
        [col for col in qiime_data_SB1.columns if col in qiime_metadata_SB1["#SampleID"].values]]
    # if there's a sample without d4, and there's another sample named the same with d4, remove the one with d4.
    print("Dropping: ", end="")
    for sample in qiime_data_orig.columns:
        parts = sample.split(".")
        if len(parts) > 1 and parts[1] == "d4" and parts[0] in qiime_data_SB1.columns and \
                qiime_metadata[qiime_metadata["#SampleID"] == parts[0]]["PART"].values[0] == "COLON":
            qiime_data_orig = qiime_data_orig.drop(sample, axis=1)
            print(f"{sample}, ", end="")
    # merge the two dfs by index
    qiime_data = pd.concat([qiime_data_orig, qiime_data_SB1], axis=1)

    pca_fastspar_prepare(qiime_data, qiime_metadata, treatments, "treatment", "_qiime", False)
    pca_fastspar_prepare(qiime_data, qiime_metadata, [abx.lower() for abx in antibiotics], "antibiotic", "_qiime",
                         False)

    # save qiime_data to a csv
    qiime_data.to_csv("../Data/QIIME/qiime_data.tsv", sep="\t")
    # normalize each column to sum to 100_000
    qiime_data_norm = qiime_data.div(qiime_data.sum(axis=0), axis=1) * 100_000
    # save the normalized data
    qiime_data_norm.to_csv("../Data/QIIME/qiime_data_normalized.tsv", sep="\t")
    # remove all columns with .d0, .d1
    qiime_data_d4 = qiime_data[[col for col in qiime_data.columns if ".d0" not in col and ".d1" not in col]]
    # save the data without d0 and d1
    qiime_data_d4.to_csv("../Data/QIIME/qiime_data_d4.tsv", sep="\t")
    # normalize each column to sum to 100_000
    qiime_data_d4_norm = qiime_data_d4.div(qiime_data_d4.sum(axis=0), axis=1) * 100_000
    # save the normalized data
    qiime_data_d4_norm.to_csv("../Data/QIIME/qiime_data_d4_normalized.tsv", sep="\t")

    # save qiime_metadata to a csv
    qiime_metadata.to_csv("../Data/QIIME/qiime_metadata.tsv", sep="\t", index=False)

    # run fastspar
    cov = False
    fastspar = False
    bad = False
    # d0 = True
    qiime = True

    if qiime:
        metadata = qiime_metadata
    # for d0 in [True, False]:
    for d0 in [True]:
        run_pcoa(metadata, True, cov, fastspar, bad, d0, qiime)
        run_pcoa(metadata, False, cov, fastspar, bad, d0, qiime)
