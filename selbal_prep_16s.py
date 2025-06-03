import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ClusteringGO import antibiotics, treatments, set_plot_defaults


# check about Julia (programing language), MRSA, Sara Mitri
# gavage -> PO
# get statistics from selbal (Bonferroni, FDR)
# !van PO feces clusters 3,4: plot manually and mark PBS and Van

# import dexplot as dxp


def example():
    # Import Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    # Prepare all_data
    x_var = 'manufacturer'
    groupby_var = 'class'
    df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [df[x_var].values.tolist() for i, df in df_agg]
    # Draw
    plt.figure(figsize=(16, 9), dpi=80)
    colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, df[x_var].unique().__len__(), stacked=True, density=False,
                                color=colors[:len(vals)])
    # Decoration
    plt.legend({group: col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
    plt.xlabel(x_var)
    plt.ylabel("Frequency")
    plt.ylim(0, 40)
    plt.xticks(ticks=bins, labels=np.unique(df[x_var]).tolist(), rotation=90, horizontalalignment='left')
    plt.show()


counter = 0


# species_number = 10


def renaming(name):
    global counter
    # if not isinstance(name, str):
    #     print(name)
    name = str(name) + "-" + str(counter)
    counter += 1
    return name


def fill_otu(df, impute=True):
    # df.tax_taxon = np.where(df.tax_taxon.isnull(), df.tax_taxon, "t_" + df.tax_taxon + "_" + str(df.index))
    df["number"] = df.index.astype(str)
    df.tax_taxon = np.where((df.tax_taxon.isnull()) & (df.tax_species.notnull()),
                            "s_" + df.tax_species + "_" + df.number, df.tax_taxon)
    df.tax_taxon = np.where((df.tax_taxon.isnull()) & (df.tax_genus.notnull()),
                            "g_" + df.tax_genus + "_" + df.number, df.tax_taxon)
    df.tax_taxon = np.where((df.tax_taxon.isnull()) & (df.tax_family.notnull()),
                            "f_" + df.tax_family + "_" + df.number, df.tax_taxon)
    df.tax_taxon = np.where((df.tax_taxon.isnull()) & (df.tax_order.notnull()),
                            "o_" + df.tax_order + "_" + df.number, df.tax_taxon)
    df.tax_taxon = np.where((df.tax_taxon.isnull()) & (df.tax_class.notnull()),
                            "c_" + df.tax_class + "_" + df.number, df.tax_taxon)
    df.tax_taxon = np.where((df.tax_taxon.isnull()) & (df.tax_phylum.notnull()),
                            "p_" + df.tax_phylum + "_" + df.number, df.tax_taxon)
    df.tax_taxon = np.where((df.tax_taxon.isnull()) & (df.tax_kingdom.notnull()),
                            "k_" + df.tax_kingdom + "_" + df.number, df.tax_taxon)
    df.tax_taxon = np.where(df.tax_taxon.isnull(), "unknown_" + df.number, df.tax_taxon)
    # remove the "number" column
    df = df.drop(columns=['number'])
    return df


def fill_species_qiime(df, impute=True):
    df["number"] = df.index.astype(str)
    df.replace("__", np.nan, inplace=True)
    df.replace("s__", np.nan, inplace=True)
    df.replace("g__", np.nan, inplace=True)
    df.replace("f__", np.nan, inplace=True)

    # Impute missing taxon values based on the hierarchy
    df['species'] = np.where((df['species'].isnull()) & (df['genus'].notnull()),
                             df['genus'] + "_" + df['number'], df['species'])
    df['species'] = np.where((df['species'].isnull()) & (df['family'].notnull()),
                             df['family'] + "_" + df['number'], df['species'])
    df['species'] = np.where((df['species'].isnull()) & (df['order'].notnull()),
                             df['order'] + "_" + df['number'], df['species'])
    df['species'] = np.where((df['species'].isnull()) & (df['class'].notnull()),
                             df['class'] + "_" + df['number'], df['species'])
    df['species'] = np.where((df['species'].isnull()) & (df['phylum'].notnull()),
                             df['phylum'] + "_" + df['number'], df['species'])
    df['species'] = np.where((df['species'].isnull()) & (df['kingdom'].notnull()),
                             df['kingdom'] + "_" + df['number'], df['species'])
    df['species'] = np.where(df['species'].isnull(), "unknown_" + df['number'], df['species'])

    # Remove the "number" column
    df = df.drop(columns=['number'])

    assert not df.species.isnull().values.any()

    return df


def fill_genus_qiime(df, impute=True):
    df["number"] = df.index.astype(str)
    df.replace("__", np.nan, inplace=True)
    df.replace("g__", np.nan, inplace=True)

    # Impute missing taxon values based on the hierarchy
    df['genus'] = np.where((df['genus'].isnull()) & (df['family'].notnull()),
                           df['family'] + "_" + df['number'], df['genus'])
    df['genus'] = np.where((df['genus'].isnull()) & (df['order'].notnull()),
                           df['order'] + "_" + df['number'], df['genus'])
    df['genus'] = np.where((df['genus'].isnull()) & (df['class'].notnull()),
                           df['class'] + "_" + df['number'], df['genus'])
    df['genus'] = np.where((df['genus'].isnull()) & (df['phylum'].notnull()),
                           df['phylum'] + "_" + df['number'], df['genus'])
    df['genus'] = np.where((df['genus'].isnull()) & (df['kingdom'].notnull()),
                           df['kingdom'] + "_" + df['number'], df['genus'])
    df['genus'] = np.where(df['genus'].isnull(), "unknown_" + df['number'], df['genus'])

    # Remove the "number" column
    df = df.drop(columns=['number'])

    assert not df.genus.isnull().values.any()

    return df


def fill_family_qiime(df, impute=True):
    df["number"] = df.index.astype(str)
    df.replace("__", np.nan, inplace=True)
    df.replace("f__", np.nan, inplace=True)

    # Impute missing taxon values based on the hierarchy
    df['family'] = np.where((df['family'].isnull()) & (df['order'].notnull()),
                            df['order'] + "_" + df['number'], df['family'])
    df['family'] = np.where((df['family'].isnull()) & (df['class'].notnull()),
                            df['class'] + "_" + df['number'], df['family'])
    df['family'] = np.where((df['family'].isnull()) & (df['phylum'].notnull()),
                            df['phylum'] + "_" + df['number'], df['family'])
    df['family'] = np.where((df['family'].isnull()) & (df['kingdom'].notnull()),
                            df['kingdom'] + "_" + df['number'], df['family'])
    df['family'] = np.where(df['family'].isnull(), "unknown_" + df['number'], df['family'])

    # Remove the "number" column
    df = df.drop(columns=['number'])

    assert not df.family.isnull().values.any()

    return df


def fill_genus(df, impute=True):
    df["number"] = df.index.astype(str)
    # add g for genus
    df.tax_genus = np.where(df.tax_genus.isnull(), df.tax_genus, "g_" + df.tax_genus)

    # unknown = 'unknown_genus'
    df.tax_genus = np.where(df.tax_genus.isnull(), "f_" + df.tax_family + ("" if impute else "_" + df.number),
                            df.tax_genus)
    df.tax_genus = np.where(df.tax_genus.isnull(), "o_" + df.tax_order + ("" if impute else "_" + df.number),
                            df.tax_genus)
    df.tax_genus = np.where(df.tax_genus.isnull(), "c_" + df.tax_class + ("" if impute else "_" + df.number),
                            df.tax_genus)
    df.tax_genus = np.where(df.tax_genus.isnull(), "p_" + df.tax_phylum + ("" if impute else "_" + df.number),
                            df.tax_genus)
    df.tax_genus = np.where(df.tax_genus.isnull(), "k_" + df.tax_kingdom + ("" if impute else "_" + df.number),
                            df.tax_genus)
    df.tax_genus = np.where(df.tax_genus.isnull(), "unknown_" + df.number, df.tax_genus)
    df = df.drop(columns=['number'])
    return df


def fill_family(df, impute=True):
    df["number"] = df.index.astype(str)
    # add f for family
    df.tax_family = np.where(df.tax_family.isnull(), df.tax_family, "f_" + df.tax_family)

    df.tax_family = np.where(df.tax_family.isnull(), "o_" + df.tax_order + ("" if impute else "_" + df.number),
                             df.tax_family)
    df.tax_family = np.where(df.tax_family.isnull(), "c_" + df.tax_class + ("" if impute else "_" + df.number),
                             df.tax_family)
    df.tax_family = np.where(df.tax_family.isnull(), "p_" + df.tax_phylum + ("" if impute else "_" + df.number),
                             df.tax_family)
    df.tax_family = np.where(df.tax_family.isnull(), "k_" + df.tax_kingdom + ("" if impute else "_" + df.number),
                             df.tax_family)
    df.tax_family = np.where(df.tax_family.isnull(), "unknown_" + df.number, df.tax_family)
    df = df.drop(columns=['number'])
    return df


def fill_class(df, impute=True):
    df["number"] = df.index.astype(str)
    # add c for class
    df.tax_class = np.where(df.tax_class.isnull(), df.tax_class, "c_" + df.tax_class)

    df.tax_class = np.where(df.tax_class.isnull(), "p_" + df.tax_phylum + ("" if impute else "_" + df.number),
                            df.tax_class)
    df.tax_class = np.where(df.tax_class.isnull(), "k_" + df.tax_kingdom + ("" if impute else "_" + df.number),
                            df.tax_class)
    df.tax_class = np.where(df.tax_class.isnull(), "unknown_" + df.number, df.tax_class)
    df = df.drop(columns=['number'])
    return df


missing = {}


# def test(x, gene_expression, cluster_genes, name):
#     if x.name.split('.')[0] in gene_expression.columns:
#         return gene_expression.set_index('gene_name').loc[cluster_genes][x.name.split('.')[0]].median()
#     else:
#         # print(x.name.split('.')[0])
#         missing[x.name.split('.')[0]] = name
#         return -1


def add_median(df, place, by_cluster=True, genes=[]):
    # treatments = ['IP', 'IV', 'PO']
    # antibiotics = ['Met', 'Van', 'Amp', 'Mix', 'Neo']
    gene_expression = pd.read_csv(
        "../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/Partek_bell_all_Normalization_Normalized_counts.tsv", sep='\t')
    # from collections import Counter
    # print(Counter(gene_expression.columns.values))
    for anti in antibiotics:
        for treatment in treatments:
            drug = anti.lower()
            df_ = df[(df['treatment'] == treatment) & ((df['antibiotic'] == drug) | (df['antibiotic'] == 'PBS'))]
            # cols = [col.replace("[", "").replace("]", "") for col in df_.columns]
            # df_.columns = cols
            if by_cluster:
                clusters = pd.read_csv(f"../B6SPF-experiment/Private/{anti}-{treatment}-clusters.tsv", sep="\t")
                for cluster in set(clusters.Cluster):
                    cluster_genes = clusters[clusters.Cluster == cluster]['Gene']
                    df_[f'cluster_{cluster}'] = df_.apply(
                        lambda x: gene_expression.set_index('gene_name').loc[cluster_genes][
                            x.name.split('.')[0]].median()
                        if x.name.split('.')[0] in gene_expression.columns else -1, axis=1)
                    # lambda x: test(x, gene_expression, cluster_genes, f"{anti}-{treatment}-{place}"), axis=1)
                    # not in Partek_bell_all_Normalization_Normalized_counts.tsv:
                    # {'V18': 'Van-IV-SI_mucus', 'N18': 'Neo-IV-SI_mucus', 'M': 'Met-IP-SI_lumen', 'V17': 'Van-IV-SI_mucus',
                    # 'V6': 'Van-PO-SI_mucus', 'V16': 'Van-IV-SI_mucus'}
                    # twice V4?
                tags = len(set(clusters.Cluster)) + 3
            else:
                for gene in genes:
                    if gene.capitalize() in gene_expression['gene_name'].values:
                        df_[f'{gene}'] = df_.apply(
                            lambda x: gene_expression.set_index('gene_name').loc[gene.capitalize()][
                                x.name.split('.')[0]]
                            if x.name.split('.')[0] in gene_expression.columns else -1, axis=1)
                    # else:
                    #     print(place, anti, gene)
                tags = len(set(genes)) + 3
            # remove columns with less than 2 values
            # threshold = 0.8
            # check = df_[df_.columns[:-tags]]
            # col_condition = ((check[check > 0].count()) / len(check) < threshold).to_numpy().nonzero()
            # df_ = df_.drop(df_.columns[col_condition], axis=1)
            add = "" if by_cluster else "specific_genes/"
            median = "median-" if by_cluster else ""
            # drop the columns "treatment", "antibiotic", "is_treatment"
            df_ = df_.drop(columns=["treatment", "antibiotic", "is_treatment"])
            df_.to_csv(f"./Private/medians/{add}{anti}-{treatment}-{median}{place}.tsv", sep="\t")


def filter_d4(df):
    # get all columns containing d4 and remove ".d4" from the name
    d4_cols = df.columns[df.columns.str.contains("d4")]
    # d4_cols = df.columns[(df.columns.str.contains("d4")) | (~df.columns.str.contains("d"))]
    d4_cols = [col.replace(".d4", "") for col in d4_cols]
    columns = df.columns[~((df.columns.str.contains("d")) & (~df.columns.str.contains("d4")))]
    # for any column not containing d4, drop it if it has same name otherwise keep it
    for col in df.columns:
        if "d4" not in col:
            if col in d4_cols:
                columns = columns.drop(col)
    return columns


def filter_d0(df):
    return df.columns[df.columns.str.contains("d0")]


def get_qiime():
    metadata = pd.read_csv(f"../Data/QIIME/qiime_metadata.tsv", sep="\t")
    metadata["Type"] = "feces"
    metadata = metadata.set_index("#SampleID")
    df = pd.read_csv(f"../Data/QIIME/qiime_data_normalized.tsv", sep="\t")

    # # remove V11
    # metadata = metadata[~metadata.index.str.contains('V11')]
    # columns_to_drop = [col for col in df.columns if 'V11' in col]
    # df = df.drop(columns=columns_to_drop)

    # Split the column by ';'
    split_columns = df['#OTU ID'].str.split(';', expand=True)

    # Rename the columns
    split_columns.columns = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    # # replace "__" with NaN
    # split_columns.replace("__", np.nan, inplace=True)

    # Join the new columns with the original DataFrame
    df = df.join(split_columns)
    df.rename(columns={'#OTU ID': 'OTU ID'}, inplace=True)

    # save df to file
    df.to_csv(f"./private/to_publish/multi_abx_qiime.csv", index=False)
    # metadata = metadata["TYPE"] !=
    metadata.drop(["BarcodeSequence", "LinkerPrimerSequence", "ReversePrimer", "TYPE", "PART", "MOUSE NUM", "PLATE"],
                  axis=1).to_csv(f"./private/to_publish/multi_abx_qiime_metadata.csv")

    return df, metadata


def create_csv(level='taxon', impute=True, qiime=True, d0=False):
    taxonomy = ['tax_kingdom', 'tax_phylum', 'tax_class', 'tax_order', 'tax_family', 'tax_genus',
                'tax_species', 'tax_taxon']
    taxonomy.remove(f'tax_{level}')
    if qiime:
        taxonomy.remove(f'tax_taxon')
        taxonomy = [col.split("_")[1] for col in taxonomy]

    for place in ["feces", "SI_lumen", "SI_mucus"]:
        if not qiime:
            metadata = pd.read_csv(f"../Data/Abx_16s_data/elad_metadata_merge_all.tsv", sep="\t").set_index("#SampleID")
            df = pd.read_csv(f"./private/otu_merged_{place}.tsv", sep="\t")
        else:
            df, metadata = get_qiime()
        names = 7 if qiime else 8
        value_cols = df.columns[:-names]
        df[value_cols] = df[value_cols].fillna(0)
        if not qiime:
            if level == 'taxon':
                df = fill_otu(df, impute)
            elif level == 'genus':
                df = fill_genus(df, impute)
            elif level == 'family':
                df = fill_family(df, impute)
            elif level == 'class':
                df = fill_class(df, impute)
        else:
            if level == 'species':
                df = fill_species_qiime(df, impute)
            elif level == 'genus':
                df = fill_genus_qiime(df, impute)
            elif level == 'family':
                df = fill_family_qiime(df, impute)

        # remove redundant columns
        col = f"tax_{level}" if not qiime else level
        df = df.set_index(col).drop(taxonomy, axis=1).drop(["OTU ID"], axis=1)

        if not d0:
            columns = filter_d4(df) if place == "feces" else df.columns
        else:
            columns = filter_d0(df)
        # columns = df.columns
        # columns = data_frame.columns[data_frame.columns.str.contains("d0")] if place == "feces" else data_frame.columns
        # columns = data_frame.columns
        # columns = columns[~columns.str.contains("C9")]
        df = df[columns]
        # normalize each column to be fractional part
        # data_frame = data_frame * 1000 / data_frame.sum()
        df = df * 100 / df.sum()
        df = df.T

        # sum all columns with same name
        df = df.groupby(df.columns, axis=1).sum()
        assert len(set(df.columns)) == df.columns.shape[0]

        threshold = "no_threshold"
        # classification = data_frame.pop('is_treatment')
        # # data_frame = data_frame.rename(columns=renaming)
        # threshold = 2
        # # threshold = data_frame.shape[0] // 2
        # print(data_frame.shape)
        # counter, counter_less = 0, 0
        # for col_name, col_val in data_frame.iteritems():
        #     if not np.any(col_val):
        #         counter += 1
        #     elif np.sum(col_val.astype(float) > 0) < threshold:
        #         counter_less += 1
        # # # data_frame = data_frame.loc[:, (data_frame != 0).any(axis=0)]
        # col_condition = (data_frame[data_frame > 0].count() < threshold).to_numpy().nonzero()
        # data_frame = data_frame.drop(data_frame.columns[col_condition], axis=1)
        # print(data_frame.shape, counter, counter_less)
        # # data_frame = data_frame.rename(columns=lambda x: x.split("-")[0])
        data_name = "SI_Mucus_ok134" if place == 'SI_mucus' else place
        # meta_data = pd.read_csv(f"../Data/MultiAbx-16s/metadata_{data_name}.tsv", sep="\treat").set_index("#SampleID")
        place_name = "SI-Lumen" if place.endswith("lumen") else ("SI-MUCUS" if place.endswith("mucus") else place)
        meta = metadata[metadata['Type'] == place_name]
        # data_frame['is_treatment'] = data_frame.apply(lambda x: "ABX" if meta_data.loc[x.name].antibiotic != "PBS" else "PBS", axis=1)

        # add metadata to file:
        df['is_treatment'] = df.apply(lambda x: "ABX" if not meta.loc[x.name].antibiotic.startswith("PBS") else "PBS",
                                      axis=1)
        df['antibiotic'] = df.apply(lambda x: meta.loc[x.name].antibiotic, axis=1)
        df['treatment'] = df.apply(
            lambda x: "PO" if meta.loc[x.name].treatment == "gavage" else meta.loc[x.name].treatment, axis=1)
        df.index.name = 'sample_id'
        # add_median(df, place, by_cluster, genes) # shouldn't be here anymore!
        # add_median(data_frame, place)
        # data_frame.to_csv(f"./private/otu_merged_{place}_reduced_{threshold}_sum-1000_fam.tsv", sep="\treat")

        df.to_csv(f"./private/otu_merged_{place}_{level}{'_qiime' if qiime else ''}{'_d0' if d0 else ''}.tsv", sep="\t")
        # print(data_frame)
        for abx in df.antibiotic.unique():
            if abx == "PBS":
                continue
            for treat in df.treatment.unique():
                treat = "gavage" if treat == "PO" else treat
                print(f"{place} {abx} {treat}")
                samples = meta[((meta["antibiotic"] == abx.lower()) | (meta["antibiotic"] == "PBS")) & (
                        meta["treatment"] == treat)]
                if not d0:
                    samples = samples[samples["day"] == 4]
                samples = [sample for sample in samples.index.values if sample in df.index]
                temp = df.loc[samples]
                # replace sample_id with sample_id.split(".")[0]
                temp.index = temp.index.str.split(".").str[0]
                temp = temp[temp.columns[:-3]]

                temp.to_csv(
                    f"./private/CompoResGenes/microbiome/{level}/{abx.capitalize()}-{treat if treat != "gavage" else "PO"}-{data_name}.tsv",
                    sep="\t")
        if qiime:
            return
    # print(missing)


def get_colors_dictionary(columns):
    # from matplotlib.cm import get_cmap
    # # colors = list(plt.cm.rainbow(np.linspace(0, 1, len(columns))))
    # # get len(df.columns) colors
    # colors = [cm.tab20b(i) for i in np.linspace(0, 1, len(columns))]
    # # colors = list(get_cmap("tab20").colors)
    # other_colors = list(get_cmap("tab20b").colors)
    # # change color order for aesthetics
    # temp = colors[0]
    # colors[0] = colors[6]
    # colors[6] = temp
    # colors[14] = other_colors[14]

    # cmap = plt.get_cmap('jet')
    # colors = cmap(np.linspace(0, 1.0, len(columns)))
    # colors = sns.color_palette("Spectral", len(columns))
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Yellow-green
        '#17becf',  # Cyan
        '#ff9896',  # Light red
        '#98df8a',  # Light green
        '#ffbb78',  # Light orange
        '#c5b0d5',  # Light purple
        '#f7b6d2',  # Light pink
        '#9edae5',  # Light blue
        '#ad494a',  # Dark red
        '#4B0082',  # Indigo
        '#FF1493',  # Deep pink
        '#00CED1',  # Dark turquoise
        '#FF4500',  # Orange-red
        '#32CD32',  # Lime green
        '#4169E1',  # Royal blue
        '#800080',  # Purple
        '#FFD700'  # Gold
    ]
    # import seaborn as sns
    # import colorcet as cc
    # colors = sns.color_palette(cc.glasbey, n_colors=len(columns))

    return {col: colors[k] for k, col in enumerate(columns)}


def plot_composition(data, loc, thresh, colors, level, labels, qiime=True):
    # data_name = "SI_Mucus_ok134" if loc == 'SI_mucus' else loc
    # merged_data_meta = pd.read_csv(f"../Data/Abx_16s_data/metadata_{data_name}.tsv", sep="\treat")
    meta = pd.read_csv(f"../Data/Abx_16s_data/elad_metadata_merge_all.tsv", sep="\t")
    if qiime:
        _, meta = get_qiime()
        meta = meta.reset_index()
    place_name = "SI-Lumen" if loc.endswith("lumen") else ("SI-MUCUS" if loc.endswith("mucus") else loc)
    meta = meta[meta['Type'] == place_name][["#SampleID", 'antibiotic', 'treatment']]
    meta = meta[np.in1d(meta["#SampleID"], data.index)]
    merged_data_meta = data.merge(meta, left_on='sample_id', right_on='#SampleID')
    # merged_data_meta['treat'] = merged_data_meta['treatment']
    # merged_data_meta['drug'] = merged_data_meta['antibiotic']
    # fig, axs = plt.subplots(len(treats), len(drugs), sharex='all', sharey='all')
    # plt.figure(num=None, figsize=(6, 16))
    # ax = fig.add_subplot(111)  # The big subplot
    # font = {'family': 'Sans Serif',
    #         'weight': 'bold',
    #         'size': 20}
    # plt.rc('font', **font)
    plt.figure()
    # plt.figure(constrained_layout=True)
    fig = plt.gcf()
    rows, cols = len(drugs), len(treats)
    eff_num = np.zeros((len(treats), len(drugs)))
    barplot_data = []
    for i, treat in enumerate(treats):
        for j, drug in enumerate(drugs):
            sub = merged_data_meta[
                (merged_data_meta['treatment'] == treat) & (merged_data_meta['antibiotic'] == drug)].copy()
            # if #SampleID contains ., remove it and everything after it
            sub['#SampleID'] = sub['#SampleID'].str.split('.').str[0]
            species = sub[sub.columns[:-3]] / 100
            bottom = np.zeros(sub.shape[0])

            # Prepare data for the bar plot and save it
            for col in sub.columns:
                if col in ['#SampleID', 'antibiotic', 'treatment']:
                    continue
                barplot_data.extend([
                    {'Treatment': treat, 'Antibiotic': drug, '#SampleID': sample, 'Category': col, 'Value': value}
                    for sample, value in zip(sub['#SampleID'], sub[col])
                ])

            # index = int(rows + cols + str(i * len(treats) + j + 1))
            for col in labels:
                plt.subplot(rows, cols, j * cols + i + 1).bar(sub['#SampleID'], sub[col], bottom=bottom,
                                                              color=colors.get(col, 'gray'), label=col)
                bottom += sub[col]
            for k, col in enumerate(data.columns):
                if col == '#SampleID' or col == 'antibiotic' or col == 'treatment':
                    continue
                if col not in labels:
                    plt.subplot(rows, cols, j * cols + i + 1).bar(sub['#SampleID'], sub[col], bottom=bottom,
                                                                  color='gray', label="other")
                    bottom += sub[col]
            treat_name = treat if treat != "gavage" else "PO"
            drug_name = drug.capitalize() if drug != "PBS" else drug
            # calc Hill number of effective species: N_1 = exp(H), H is shannon's entropy
            # mean_s = np.mean(species, axis=0)
            # create a histogram of species size for row 0

            eff_num[i, j] = np.exp(np.mean(-np.sum(species * np.log(species), axis=1)))
            std = np.exp(np.std(-np.sum(species * np.log(species), axis=1)))
            # N_1_form = '{:.2e}'.format(N_1)
            plt.subplot(rows, cols, j * cols + i + 1).set_title(f"{drug_name}, {treat_name} " +
                                                                r"[$\left\langle n_{eff}\right\rangle =" +
                                                                f"{np.round(eff_num[i, j], 2)}"
                                                                fr"\pm{np.round(std, 2)}$]", fontsize=30)
            plt.subplot(rows, cols, j * cols + i + 1).tick_params('x', labelsize=30)
            # plt.subplot(rows, cols, j * cols + i + 1).tick_params('x', labelrotation=20, labelsize=30)
    # Convert barplot_data to a DataFrame and save to CSV
    df_barplot = pd.DataFrame(barplot_data)
    df_barplot.to_csv(f"./private/compositional_microbiome_population_{loc}_{thresh}_{level}_barplot_values.csv",
                      index=False)

    # plt.subplot(rows, cols, cols * rows - (cols // 2)).set_xlabel("Sample")
    # plt.subplot(rows, cols, cols + 1).set_ylabel("Population")
    # best = data.columns.values
    # basic = best[[taxa in colors for taxa in best]]
    # Create the legend with sorted labels
    handles, current_labels = plt.gca().get_legend_handles_labels()
    handles = [handles[current_labels.index(label)] for label in labels]
    # handles = [colors[label] for label in basic]
    if qiime:
        basic = [val.replace("__", "_") for val in labels]
    # if label contains '_' and then a number, remove the number
    basic = [taxa.split('_')[0] + "_" + taxa.split('_')[1] if taxa.split('_')[0] != level[1] else taxa for taxa in
             basic]
    labels = np.append(basic, 'other')
    # handles.append("gray")
    colors['other'] = 'gray'
    # Create a custom handle for the "gray" color
    import matplotlib.patches as mpatches
    gray_handle = mpatches.Patch(color='gray', label='other')
    handles.append(gray_handle)

    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 4), fontsize=30, reverse=True)
    # plt.legend(labels, loc='center left', bbox_to_anchor=(1, 4), fontsize=30, reverse=True)

    # plt.subplot(rows, cols, 6).legend(labels, loc='center right', bbox_to_anchor=(3, 0), fontsize=20)
    level_name = level.replace('_', ' ') if level != 'species' else 'OTU'
    plt.suptitle(f"Compositional Microbiome Population, {loc.replace('_', ' ')}, {level_name} level",
                 fontsize=45)
    fig.set_size_inches(8 * (cols + 1), 6 * rows)
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    # plt.tight_layout()

    # set title location to be above the plot
    fig.subplots_adjust(top=0.95)
    set_plot_defaults()
    plt.savefig(f"./private/compositional_microbiome_population_{loc}_{thresh}_same_labels_all{level}.svg",
                format='svg', dpi=180)
    plt.savefig(f"./private/compositional_microbiome_population_{loc}_{thresh}_same_labels_all{level}.png",
                format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return eff_num


def plot_effective_number_heatmap(eff_num, loc, thresh, drugs, treats, maximal, minimal, level,
                                  title="Effective number of"):
    eff_num = pd.DataFrame(eff_num)
    eff_num.columns = [drug.capitalize() if drug != "PBS" else drug for drug in drugs]
    eff_num[" "] = [treat if treat != "gavage" else "PO" for treat in treats]
    eff_num = eff_num.set_index(" ")
    # transpose
    eff_num = eff_num.T

    # sort
    eff_num = eff_num[np.sort(treatments)]
    order = ["Amp", "Met", "Neo", "Van", "Mix", "PBS"]
    eff_num = eff_num.loc[order]

    sns.set(font_scale=1.4)
    sns.heatmap(eff_num, cmap="Blues", vmin=minimal, vmax=maximal)
    name = level if not level.endswith('y') else level[:-1] + 'i'
    name += 'es' if name != 'taxa' else ''
    name = "genera" if name == "genuses" else name
    name = "species" if name == "specieses" else name
    # name = "OTU" if name == "specieses" else name
    plt.title(f"{title} {name}, {loc.replace('_', ' ')}")
    # increase font size of axis labels and title
    # plt.rc('font', size=16)
    plt.savefig(f"./private/effective number of {level}_{loc}_{thresh}.png")
    plt.show()


# treats = ['gavage', 'IV', 'IP']
# drugs = ['amp', 'mix', 'van', 'met', 'neo', 'PBS']
treats = ['IP', 'IV', 'gavage']
drugs = ['amp', 'met', 'neo', 'van', 'mix', 'PBS']


def create_figures(level='genus', impute=True, qiime=True):
    threshold = "no_threshold"
    types = ["feces"]
    cutoff = 25
    biggest_bacteria = np.empty(cutoff * len(types), dtype=object)
    for i, place in enumerate(types):
        df = pd.read_csv(f"./private/otu_merged_{place}_{level}{'_qiime' if qiime else ''}.tsv", sep="\t").set_index(
            'sample_id')
        # df = df.reindex(df.mean().sort_values().index[::-1], axis=1)
        df = df.reindex(df[df.columns[:-8]].mean().sort_values().index[::-1], axis=1)
        print([val + ": " + str(i) for i, val in enumerate(df.columns.values) if "Porph" in val])
        return
        biggest_bacteria[i * cutoff:i * cutoff + cutoff] = df.columns.values[:cutoff]
    biggest_bacteria_set = set(biggest_bacteria)
    colors = get_colors_dictionary(biggest_bacteria_set)
    to_plot = np.zeros((3, len(treats), len(drugs)))
    for i, place in enumerate(types):
        df = pd.read_csv(f"./private/otu_merged_{place}_{level}{'_qiime' if qiime else ''}.tsv", sep="\t").set_index(
            'sample_id')
        # data_frame.index.name = "sample_id"
        # df = df.reindex(df.mean().sort_values().index[::-1], axis=1)
        df = df.reindex(df[df.columns[:-8]].mean().sort_values().index[::-1], axis=1)
        # data_frame = data_frame.reset_index()
        addition = f'_{level}'
        if impute:
            addition += "_imputed"
        to_plot[i, :, :] = plot_composition(df, place, threshold, colors, addition, biggest_bacteria)
    for i, place in enumerate(types):
        plot_effective_number_heatmap(to_plot[i, :, :], place, threshold, drugs, treats, np.max(to_plot),
                                      np.min(to_plot), level)
    return to_plot


def d0_entropy(level, limits):
    # data, meta = get_qiime()
    merged_data_meta = pd.read_csv(f"./private/otu_merged_feces_{level}_qiime_d0.tsv", sep="\t")
    merged_data_meta.rename(columns={'sample_id': '#SampleID'}, inplace=True)
    # meta = meta.reset_index()
    # meta = meta[meta['Type'] == "feces"][["#SampleID"]]
    # meta = meta[np.in1d(meta["#SampleID"], data.index)]
    # merged_data_meta = data.merge(meta, left_on='sample_id', right_on='#SampleID')
    eff_num = np.zeros((len(treats), len(drugs)))
    for i, treat in enumerate(treats):
        for j, drug in enumerate(drugs):
            treat = treat if treat != "gavage" else "PO"
            sub = merged_data_meta[
                (merged_data_meta['treatment'] == treat) & (merged_data_meta['antibiotic'] == drug)].copy()
            # if #SampleID contains ., remove it and everything after it
            sub['#SampleID'] = sub['#SampleID'].str.split('.').str[0]
            species = sub[sub.columns[1:-3]] / 100
            eff_num[i, j] = np.exp(np.mean(-np.sum(species * np.log(species), axis=1)))
    maximal = np.max(limits)
    minimal = np.min(limits)
    plot_effective_number_heatmap(eff_num, "feces_d0", "no_threshold", drugs, treats, maximal,
                                  minimal, level)
    # Create an empty DataFrame with the same columns
    eff_num = pd.DataFrame(eff_num)
    eff_num.columns = [drug.capitalize() + " d0" if drug != "PBS" else drug + " d0" for drug in drugs]
    eff_num[" "] = [treat if treat != "gavage" else "PO" for treat in treats]
    eff_num = eff_num.set_index(" ")
    eff_num = eff_num.T
    limits = pd.DataFrame(limits)
    limits.columns = [drug.capitalize() + " d4" if drug != "PBS" else drug + " d4" for drug in drugs]
    limits[" "] = [treat if treat != "gavage" else "PO" for treat in treats]
    limits = limits.set_index(" ")
    limits = limits.T
    result = pd.DataFrame(columns=eff_num.columns)
    # Initialize a counter for the new DataFrame's index
    counter = 0
    index = []
    # Interleave rows from both DataFrames
    for i in range(len(eff_num)):
        result.loc[counter] = limits.iloc[i]
        index.append(limits.index[i])
        counter += 1
        result.loc[counter] = eff_num.iloc[i]
        index.append(eff_num.index[i])
        counter += 1
    result.index = index
    result = result[np.sort(treatments)]
    order = ["Amp d4", "Amp d0", "Met d4", "Met d0", "Neo d4", "Neo d0", "Van d4", "Van d0", "Mix d4", "Mix d0",
             "PBS d4", "PBS d0"]
    result = result.loc[order]

    sns.set(font_scale=1.4)
    sns.heatmap(result, cmap="Blues", vmin=minimal, vmax=maximal)
    name = level if not level.endswith('y') else level[:-1] + 'i'
    name += 'es' if name != 'taxa' else ''
    name = "genera" if name == "genuses" else name
    name = "species" if name == "specieses" else name
    # name = "OTU" if name == "specieses" else name
    plt.title(f"Effective number of {name}")
    # increase font size of axis labels and title
    # plt.rc('font', size=16)
    set_plot_defaults()
    plt.tight_layout()
    plt.savefig(f"./private/effective number of {level}_d0+4.png")
    plt.show()


if __name__ == '__main__':
    get_qiime()
    # for level in ['genus', 'family']:
    # for level in ['taxon', 'genus', 'class', 'species']:
    # for level in ['family']:
    for level in ['species', 'genus', 'family']:
        # for level in ['species']:
        impute = False
        # mucin_production = ["muc2", "tff3", "clca1", "fcgbp", "mep1b"]
        # antimicrobial_peptide_defense = ["zg16", "retnlb", "reg3g", "reg3b", "sprr2a1", "sprr2a2", "sprr2a3"]
        # genes = ["arntl", "ciart", "nfil3", "nr1d1", "per1", "per2", "cry1", "cry2", "dbp"]
        # genes = mucin_production + antimicrobial_peptide_defense + genes
        # create_csv(level, impute, by_cluster=False, genes=genes)

        # create_csv(level, impute)
        # create_csv(level, impute, d0=True)
        # # example()

        all_values = create_figures(level, impute)
        # d0_entropy(level, all_values[0, :, :])
