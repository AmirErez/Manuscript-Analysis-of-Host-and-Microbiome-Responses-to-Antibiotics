import json
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import wget
from goatools import obo_parser
from matplotlib.patches import Ellipse
from scipy.stats import linregress
from scipy.stats.mstats import gmean

from clusters_plot import clusters_compare_mix

treatments = ['IP', 'IV', 'PO']
antibiotics = ['Amp', 'Met', 'Neo', 'Van', 'Mix']

mitochondrial_genes_translation = {
    "ND1": "mt-nd1",
    "ND2": "mt-nd2",
    "ND3": "mt-nd3",
    "ND4": "mt-nd4",
    "ND4L": "mt-nd4l",
    "ND5": "mt-nd5",
    "ND6": "mt-nd6",
    "COX1": "mt-co1",
    "COX2": "mt-co2",
    "COX3": "mt-co3",
    "CYTB": "mt-cytb",
    "ATP6": "mt-atp6",
    "ATP8": "mt-atp8",
    "tRNA-F": "mt-TF",
    "tRNA-V": "mt-TV",
    "tRNA-L": "mt-TL1",
    "tRNA-I": "mt-TI",
    "tRNA-Q": "mt-TQ",
    "tRNA-M": "mt-TM",
    "tRNA-W": "mt-TW",
    "tRNA-A": "mt-TA",
    "tRNA-N": "mt-TN",
    "tRNA-C": "mt-TC",
    "tRNA-Y": "mt-TY",
    "tRNA-S1": "mt-TS1",
    "tRNA-D": "mt-TD",
    "tRNA-K": "mt-TK",
    "tRNA-G": "mt-TG",
    "tRNA-R": "mt-TR",
    "tRNA-H": "mt-TH",
    "tRNA-S2": "mt-TS2",
    "tRNA-L2": "mt-TL2",
    "tRNA-E": "mt-TE",
    "tRNA-T": "mt-TT",
    "tRNA-P": "mt-TP",
    "12S rRNA": "mt-rnr1",
    "16S rRNA": "mt-rnr2"
}

# save the values of the dictionary in a list
mitochondrial_genes = list(mitochondrial_genes_translation.values())

private = os.path.join(".", "Private")
path = os.path.join(private, "clusters_properties\\")


# data_folder = os.path.join("..", "Data", "MultiAbx-16s", "MultiAbx-RPKM-RNAseq-B6", "new normalization")


def set_plot_defaults():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    # sns.set_theme(rc=plt.rcParams)


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
    metadata = pd.read_csv(os.path.join("Data", "qiime_metadata.tsv"), sep="\t")
    metadata["Type"] = "feces"
    metadata = metadata.set_index("#SampleID")
    df = pd.read_csv(os.path.join("Data", "qiime_data_normalized.tsv"), sep="\t")

    # # remove V11
    # metadata = metadata[~metadata.index.str.contains('V11')]
    # columns_to_drop = [col for col in df.columns if 'V11' in col]
    # df = df.drop(columns=columns_to_drop)

    # Split the column by ';'
    split_columns = df['#OTU ID'].str.split(';', expand=True)

    # Rename the columns
    split_columns.columns = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    # Join the new columns with the original DataFrame
    df = df.join(split_columns)
    df.rename(columns={'#OTU ID': 'OTU ID'}, inplace=True)

    return df, metadata


def create_csv(level='taxon', impute=True, qiime=True, d0=False):
    taxonomy = ['tax_kingdom', 'tax_phylum', 'tax_class', 'tax_order', 'tax_family', 'tax_genus',
                'tax_species', 'tax_taxon']
    taxonomy.remove(f'tax_{level}')
    if qiime:
        taxonomy.remove(f'tax_taxon')
        taxonomy = [col.split("_")[1] for col in taxonomy]

    for place in ["feces", "SI_lumen", "SI_mucus"]:
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

        df.to_csv(
            os.path.join("Private", f"otu_merged_{place}_{level}{'_qiime' if qiime else ''}{'_d0' if d0 else ''}.tsv"),
            sep="\t")
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
                    f"./private/CompoResGenes/{abx.capitalize()}-{treat if treat != 'gavage' else 'PO'}-{data_name}.tsv",
                    sep="\t")
        if qiime:
            return
    # print(missing)


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


def load_colors_dictionary_from_txt(file_path):
    colors_dict = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value_str = line.strip().split(": ")
            value = np.fromstring(value_str[1:-1], sep=', ')
            colors_dict[key] = value
    return colors_dict


def save_colors_dictionary_as_txt(colors_dict, file_path):
    with open(file_path, "w") as f:
        for key, value in colors_dict.items():
            # Convert NumPy array to string representation
            value_str = np.array2string(value, separator=', ')
            f.write(f"{key}: {value_str}\n")


def plot_composition(data, loc, thresh, colors, level, labels, qiime=True):
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
            plt.subplot(rows, cols, j * cols + i + 1).tick_params('x', labelrotation=20, labelsize=30)
    # Convert barplot_data to a DataFrame and save to CSV
    df_barplot = pd.DataFrame(barplot_data)
    df_barplot.to_csv(os.path.join("Private", f"compositional_microbiome_population_{loc}_{thresh}_{level}_barplot_values.csv"),
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
    # plt.savefig(f"./private/compositional_microbiome_population_{loc}_{thresh}_same_labels_all{level}.svg",
    #             format='svg', dpi=180)
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
    # save eff_num to csv
    eff_num.to_csv(f"./private/effective_number_of_{level}_{loc}_{thresh}.csv")
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


def get_colors_dictionary_bact(columns):
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

    return {col: colors[k] for k, col in enumerate(columns)}


treats = ['IP', 'IV', 'gavage']
drugs = ['amp', 'met', 'neo', 'van', 'mix', 'PBS']


def create_figures(level='genus', impute=True, qiime=True):
    threshold = "no_threshold"
    types = ["feces"]
    cutoff = 25
    biggest_bacteria = np.empty(cutoff * len(types), dtype=object)
    for i, place in enumerate(types):
        df = pd.read_csv(f"./Data/otu_merged_{place}_{level}{'_qiime' if qiime else ''}.tsv", sep="\t").set_index(
            'sample_id')
        # df = df.reindex(df.mean().sort_values().index[::-1], axis=1)
        df = df.reindex(df[df.columns[:-8]].mean().sort_values().index[::-1], axis=1)
        biggest_bacteria[i * cutoff:i * cutoff + cutoff] = df.columns.values[:cutoff]
    biggest_bacteria_set = set(biggest_bacteria)
    colors = get_colors_dictionary_bact(biggest_bacteria_set)
    to_plot = np.zeros((3, len(treats), len(drugs)))
    for i, place in enumerate(types):
        df = pd.read_csv(f"./Data/otu_merged_{place}_{level}{'_qiime' if qiime else ''}.tsv", sep="\t").set_index(
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
        df.to_csv(f"./Private/{group}{addition}.tsv", sep="\t")


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


def pcoa(matrix, metadata_df, title, color_group, correction=None, days=None, addition=''):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # explain_var(matrix, title, "./Private/dimension reduction/fastspar_results/")

    if correction:
        matrix = correction(matrix, correction)
    # Principal Coordinate Analysis (PCoA)
    pca = PCA(n_components=2)
    pcoa_result = pca.fit_transform(matrix)

    # Merge metadata with PCoA results
    pcoa_df = pd.DataFrame(pcoa_result, index=matrix.index, columns=['PCoA1', 'PCoA2'])

    merged_pcoa_df = pd.merge(metadata_df, pcoa_df, left_on='#SampleID', right_index=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.title(f'PCoA for All {title} Antibiotics', size=16)

    # x_factor = 20 if title != "IP" else 200
    # # change xlim and y lim to be the closest half integer to the max and min values
    # plt.xlim((round(x_factor * min(merged_pcoa_df['PCoA1'])) / x_factor) - (1 / (10 * x_factor)),
    #          (round(x_factor * max(merged_pcoa_df['PCoA1'])) / x_factor) + (1 / (10 * x_factor)))
    # plt.ylim((np.floor(200 * min(merged_pcoa_df['PCoA2'])) / 200) - 0.0005,
    #          (round(200 * max(merged_pcoa_df['PCoA2'])) / 200) + 0.005)
    # add amount of variance explained by each PC
    plt.xlabel(f'PCoA1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)', size=16)
    plt.ylabel(f'PCoA2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)', size=16)

    colors = get_default_colors(merged_pcoa_df[color_group].unique())
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
            ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (d4)", color=color)
            # Add ellipse
            get_ellipse(group['PCoA1'], group['PCoA2'], ax, n_std=2.0,
                        facecolor=color, edgecolor=color, alpha=0.2)
        else:
            ax.scatter(group['PCoA1'], group['PCoA2'], label=abx + " (d4)",
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
    # update legend labels: replace "_" with " "
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        new_labels.append(label.replace("_", " ").replace("gavage", "PO"))
    ax.legend(handles, new_labels, loc='upper left', bbox_to_anchor=(1, 1))  # , fontsize=8)
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"./Private/{title}_pcoa{addition}.png", dpi=300)
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


def aitchison_distance(x, y):
    """Calculate the Aitchison distance between two composition vectors"""
    clr_x = clr_transformation(x)
    clr_y = clr_transformation(y)
    return np.sqrt(np.sum((clr_x - clr_y) ** 2))


def run_pcoa(metadata, run_treats):
    iterable = treatments if run_treats else [abx.lower() for abx in antibiotics]
    for treat in iterable:
        data = pd.read_csv(f"./Data/fastspar/{treat}_qiime.tsv", sep="\t").set_index("#OTU ID").T
        d4_cols = [col for col in data.columns if
                   (metadata[metadata["#SampleID"] == col]["day"] == 4).values[0]]
        d0_cols = [col for col in data.columns if
                   (metadata[metadata["#SampleID"] == col]["day"] == 0).values[0]]
        d4_cols += d0_cols
        in_data = d4_cols
        data = data[d4_cols]
        distance = pd.DataFrame(index=data.columns, columns=data.columns)
        # calc correlations between rows
        # Calculate all-to-all correlations manually
        for col1 in data.columns:
            for col2 in data.columns:
                distance.loc[col1, col2] = aitchison_distance(data[col1], data[col2])
        group_d0 = [sample for sample in in_data if "d0" in sample]
        group_d4 = [sample for sample in in_data if "d0" not in sample]

        if run_treats:
            pcoa(distance, metadata, treat + " (including d0)", 'antibiotic', days=[group_d0, group_d4])
        else:
            pcoa(distance, metadata, treat + " (including d0)", 'antibiotic_treatment', days=[group_d0, group_d4])
    print("done")


def run_pcoa_functions():
    qiime_metadata_orig = pd.read_csv(
        fr"./Data/mf_ok122_2.tsv",
        sep="\t")
    print("Changing IP to IV (correcting classification of #12): ", end="")
    mask = qiime_metadata_orig["#SampleID"].str.contains("12") & (qiime_metadata_orig["treatment"] != "IP")
    changed_samples = qiime_metadata_orig.loc[mask, "#SampleID"].tolist()

    print(", ".join(changed_samples))
    qiime_metadata_orig.loc[mask, "treatment"] = "IP"

    qiime_metadata_SB1 = pd.read_csv(fr"./Data/metadata_SB1.tsv", sep="\t")
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

    qiime_data_orig = pd.read_csv(fr"./Data/orig_species.tsv", sep="\t", skiprows=[0]).set_index("#OTU ID")
    qiime_data_SB1 = pd.read_csv(fr"./Data/SB1_species.tsv", sep="\t", skiprows=[0]).set_index("#OTU ID")
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
    qiime_data.to_csv("./private/qiime_data.tsv", sep="\t")
    # normalize each column to sum to 100_000
    qiime_data_norm = qiime_data.div(qiime_data.sum(axis=0), axis=1) * 100_000
    # save the normalized data
    qiime_data_norm.to_csv("./private/qiime_data_normalized.tsv", sep="\t")
    # remove all columns with .d0, .d1
    qiime_data_d4 = qiime_data[[col for col in qiime_data.columns if ".d0" not in col and ".d1" not in col]]
    # save the data without d0 and d1
    qiime_data_d4.to_csv("./private/qiime_data_d4.tsv", sep="\t")
    # normalize each column to sum to 100_000
    qiime_data_d4_norm = qiime_data_d4.div(qiime_data_d4.sum(axis=0), axis=1) * 100_000
    # save the normalized data
    qiime_data_d4_norm.to_csv("./private/qiime_data_d4_normalized.tsv", sep="\t")

    # save qiime_metadata to a csv
    qiime_metadata.to_csv("./private/qiime_metadata.tsv", sep="\t", index=False)

    # run fastspar

    metadata = qiime_metadata

    run_pcoa(metadata, True)
    run_pcoa(metadata, False)


def figure1():
    for level in ['species', 'genus', 'family']:
        impute = False
        create_figures(level, impute)
    run_pcoa_functions()


def plot_correlation_gsea(gsea, our):
    # Create a single figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
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

    # Adjust layout for better visualization
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    set_plot_defaults()
    plt.savefig(private + "/correlation_gsea_our.png")
    plt.show()


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


def get_go(download_anyway=False):
    # go_obo_url = 'http://current.geneontology.org/ontology/go-basic.obo'
    go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
    data_folder = os.getcwd() + '/all_data'
    # Check if we have the ./all_data path already
    if not os.path.isfile(data_folder):
        # Emulate mkdir -p (no error if folder exists)
        try:
            os.mkdir(data_folder)
        except OSError as e:
            if e.errno != 17:
                raise e
    else:
        raise Exception('Data path (' + data_folder + ') exists as a file. Please rename, remove or change the desired '
                                                      'location of the all_data path.')
    # Check if the file exists already
    if not os.path.isfile(data_folder + '/go-basic.obo') or download_anyway:
        go_obo = wget.download(go_obo_url, data_folder + '/go-basic.obo')
    else:
        go_obo = data_folder + '/go-basic.obo'
    # print(go_obo)
    return go_obo


def get_selected_df(abx, treat, exp_type, regular=True, fdr=True):
    df = pd.read_csv(os.path.join(private, exp_type, f"top_correlated_GO_terms_{abx}_{treat}.tsv"), sep="\t")
    selected = df[(df['fdr correlation'] < 0.05)]
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


def get_ancestor(go_term):
    last = set()
    to_check = {go_term}
    checked = set()
    while to_check:
        term = to_check.pop()
        if term not in checked:
            for parent in term.parents:
                if parent.id == "GO:0008150":
                    last.add(term)
                else:
                    to_check.add(parent)
        checked.add(term)
    return last


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


def get_selected_gsea(abx, treat, go):
    go_dict = create_go_term_dict(go)
    # iterate over folders in folder .\Private\GSEA and find the one starts with abx-treat
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


def set_figure(treats, antibiotics, cols_factor=6.0, rows_factor=5.0):
    rows, cols = len(antibiotics), len(treats)
    fig, axis = plt.subplots(rows, cols, figsize=(cols_factor * cols * 1.55, rows_factor * rows * 1.55))
    fig.tight_layout(pad=10.0)
    # font = {'family': 'Sans Serif',
    #         'size': 20}
    # plt.rc('font', **font)
    # plt.ylabel('antibiotics', size=20)
    # plt.xlabel('treatment', size=20)
    return axis


def get_to_axis(axis, i, j, n, m):
    if n > 1 and m > 1:
        return axis[i, j]
    else:
        return axis[max(i, j)]


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
    curr_path = os.path.join(".", "Private")
    plt.savefig(os.path.join(curr_path, f"{exp_type[1:]} categories.png"), bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(curr_path, f"{exp_type[1:]} categories.svg"), bbox_inches='tight', dpi=600)
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
    plt.savefig(os.path.join(curr_path, f"{exp_type[1:]} categories legend.svg"), bbox_inches='tight')
    plt.show()

    suppressed = plot_enrichment(antibiotics, treatments, exp_type, counts_dict_suppressed, orig_labels,
                                 f"suppressed {'GSEA' if gsea else ''}")
    enhanced = plot_enrichment(antibiotics, treatments, exp_type, counts_dict_enhanced, orig_labels,
                               f"enhanced {'GSEA' if gsea else ''}")

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
    plt.figure(figsize=(3, 4), dpi=300)
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
    set_plot_defaults()
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

    plt.savefig(os.path.join(private, f"enrichment {title}.png"), bbox_inches='tight')
    plt.show()
    plt.close()
    return df


def plot_significant_genes_number(meta, raw, antibiotics, treatments, param, condition="Treatment"):
    import pickle
    # import matplotlib
    # matplotlib.use('Agg')

    # import venn
    threshold = 0.05
    # if the file private + f"analysis/{param}/statistics_genes.csv" doesn't exist, create it
    # if True:
    if not os.path.exists(os.path.join(private, f"statistics_genes.csv")):
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
        df.to_csv(private + f"statistics_genes.csv", index=False)
        with open(private + f'statistics_genes.pkl', 'wb') as file:
            pickle.dump(genes, file)
        all_stats.to_csv(private + f"all_stats.csv", index=True)
    else:
        df = pd.read_csv(os.path.join(private, "statistics_genes.csv"))
        with open(os.path.join(private, 'statistics_genes.pkl'), 'rb') as file:
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

    plt.savefig(os.path.join(private, "genes stats.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(private, "genes stats.pdf"), bbox_inches='tight')
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Use Pillow to save as TIFF
    with Image.open(buf) as img:
        img.save(os.path.join(private, "genes stats.tiff"), format='TIFF')
    plt.show()


def read_process_files(new=False, filter_value=0.55, merge_big_abx=True, remove_mitochondrial=True, gene_name=False):
    partek_df = pd.read_csv(
        "../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/New Partek_bell_all_Normalization_Normalized_counts1.csv")
    partek_df = partek_df.set_index("Gene Symbol")
    folder_dir = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
    genome_df = pd.read_csv(folder_dir + "rpkm_named_genome-2023-09-26.tsv", sep="\t")
    transcriptome_df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")

    if gene_name:
        # replace all empty cells in gene_name with the value in gene_id
        genome_df["gene_name"] = genome_df.apply(
            lambda row: row["gene_id"] if pd.isna(row["gene_name"]) else row["gene_name"], axis=1)
        transcriptome_df["gene_name"] = transcriptome_df.apply(
            lambda row: row["gene_id"] if pd.isna(row["gene_name"]) else row["gene_name"], axis=1)
        genome_df = genome_df.set_index("gene_name")
        transcriptome_df = transcriptome_df.set_index("gene_name")
        genome_df = genome_df.drop("gene_id", axis=1)
        transcriptome_df = transcriptome_df.drop("gene_id", axis=1)
    else:
        genome_df = genome_df.drop("gene_name", axis=1)
        transcriptome_df = transcriptome_df.drop("gene_name", axis=1)
        genome_df.rename(columns={'gene_id': 'gene_name'}, inplace=True)
        transcriptome_df.rename(columns={'gene_id': 'gene_name'}, inplace=True)
        genome_df = genome_df.set_index("gene_name")
        transcriptome_df = transcriptome_df.set_index("gene_name")

    # replace partek nans with 0
    partek_df = partek_df.fillna(0)

    metadata = get_metadata(type="", only_old=not new, filter=filter_value)

    # change genome and transcriptome column names using metadata: replace the name which is 'Sample' to the
    # equivalent 'ID'
    genome_df = genome_df.rename(columns=metadata.set_index('Sample')['ID'].to_dict())
    transcriptome_df = transcriptome_df.rename(columns=metadata.set_index('Sample')['ID'].to_dict())

    # keep in all 3 DFs only columns that are in metadata["ID"].values
    genome_df = genome_df[[col for col in genome_df.columns if col in metadata["ID"].values]]
    transcriptome_df = transcriptome_df[[col for col in transcriptome_df.columns if col in metadata["ID"].values]]
    partek_df = partek_df[[col for col in partek_df.columns if col in metadata["ID"].values]]

    if merge_big_abx:
        new_path = r"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/mRNA_NEBNext_20200908/"
        new_data = pd.read_csv(new_path + "mRNA_NEBNext_20200908_genes_norm_named.tsv", sep="\t")
        # sum rows with the same gene_name and drop the gene_id column
        # new_data = new_data.drop("gene_id", axis=1).groupby("gene_name").sum()
        new_stats = pd.read_csv(new_path + r"big_abx_stats.csv")
        # remove all samples with "aligned" < 0.5
        columns_to_keep = new_stats[new_stats["aligned"] > filter_value]["Sample Name"]
        # new_data = new_data[columns_to_keep.append(pd.Series(["gene_name", "gene_id"]))]
        columns_to_keep = columns_to_keep.tolist()  # Convert to list if needed
        columns_to_keep.append("gene_name")  # Append to the list
        columns_to_keep.append("gene_id")
        new_data.columns = [col.split("_")[-1] if "gene" not in col else col for col in new_data.columns]
        # drop columns C1, C2, C3 as they already exist in the other df
        new_data = new_data.drop(["C1", "C2", "C3"], axis=1)

        if gene_name:
            new_data["gene_name"] = new_data.apply(
                lambda row: row.name if pd.isna(row["gene_name"]) else row["gene_name"], axis=1)
            new_data = new_data.set_index("gene_name").drop("gene_id", axis=1)
        else:
            new_data = new_data.drop("gene_name", axis=1)
            new_data.rename(columns={'gene_id': 'gene_name'}, inplace=True)
            new_data = new_data.set_index("gene_name")
        transcriptome_df = pd.merge(transcriptome_df, new_data, left_index=True, right_index=True)
        new_metadata = get_metadata(type="", only_old=not new, filter=False)
        new_metadata = new_metadata[new_metadata["ID"].isin(new_data.columns)]
        metadata = pd.concat([metadata, new_metadata])

    # print indexes that appear twice in genome and transcriptome
    if len(genome_df.index[genome_df.index.duplicated()]) > 0:
        print("indexes that appear twice in genome:\n", genome_df.index[genome_df.index.duplicated()])
        print("and transcriptome:\n", transcriptome_df.index[transcriptome_df.index.duplicated()])
    genome_df = genome_df.groupby(genome_df.index).sum()
    transcriptome_df = transcriptome_df.groupby(transcriptome_df.index).sum()

    # remove sparse genes (more than 50% zeros in a row):
    # check all sparse genes (more than 50% zeros in a row) in each df, and check if the non-zero samples are the same
    # condition, using the metadata
    partek_zeros = partek_df[partek_df == 0].count(axis=1)
    partek_sparse = partek_zeros[partek_zeros > 0.5 * partek_df.shape[1]]
    genome_zeros = genome_df[genome_df == 0].count(axis=1)
    genome_sparse = genome_zeros[genome_zeros > 0.5 * genome_df.shape[1]]
    transcriptome_zeros = transcriptome_df[transcriptome_df == 0].count(axis=1)
    transcriptome_sparse = transcriptome_zeros[transcriptome_zeros > 0.5 * transcriptome_df.shape[1]]
    partek_df = partek_df.drop(partek_sparse.index)
    genome_df = genome_df.drop(genome_sparse.index)
    transcriptome_df = transcriptome_df.drop(transcriptome_sparse.index)

    if remove_mitochondrial:
        matching_indices = transcriptome_df.index[
            transcriptome_df.index.str.lower().isin(set(mitochondrial_genes))].tolist()

        # remove mitochondrial genes from the dataframes
        genome_df = genome_df.drop(matching_indices, errors='ignore')
        transcriptome_df = transcriptome_df.drop(matching_indices, errors='ignore')
        partek_df = partek_df.drop(matching_indices, errors='ignore')

    partek_df = (partek_df * 1000000).divide(partek_df.sum(axis=0), axis=1)
    genome_df = (genome_df * 1000000).divide(genome_df.sum(axis=0), axis=1)
    transcriptome_df = (transcriptome_df * 1000000).divide(transcriptome_df.sum(axis=0), axis=1)

    # NOTICE! drop C9, C10, C18, M13, V14 from all DFs and metadata
    to_remove = ["C9", "C10", "C18", "M13", "V14", "V11"]
    transcriptome_df = transcriptome_df.drop(to_remove, axis=1)
    metadata = metadata[~metadata["ID"].isin(to_remove)]

    return genome_df, metadata, partek_df, transcriptome_df


def get_metadata(type="", only_old=True, filter=0.55):
    meta = pd.read_excel(os.path.join("./Data", "metadata.xlsx"))
    meta['ID'] = meta.apply(lambda row: row['ID'] + 'N' if row['New/Old'] == 'N' else row['ID'], axis=1)
    meta['Drug'] = meta.apply(lambda row: row['Drug'].replace('mix', 'Mix').replace('ampicillin', 'Amp')
                              .replace('Control ', 'PBS').replace('METRO', 'Met').replace('NEO', 'Neo')
                              .replace('VANCO', 'Van'), axis=1)
    if filter:
        file = "RASflow stats 2023_09_26.csv" if type else "RASflow stats 2023_09_17.csv"
        qc = pd.read_csv(os.path.join("./Data", file))
        # get Sample Name from qc if aligned > filter
        samples = qc[qc['aligned'] > filter]['Sample Name']
        # print the filtered out samples, sorted lexically
        print(sorted([sample for sample in qc['Sample Name'] if sample not in samples.values]))
        # keep only metadata rows with Sample Name in Sample
        meta = meta[meta['Sample'].isin(samples)]
    # # print samples that are in samples and not in meta.Sample
    # print([sample for sample in samples if sample not in meta['Sample'].values])
    if only_old:
        # remove all samples that end with N from metadata and from data
        meta = meta[~meta['ID'].str.endswith('N')]
    return meta


def zscore_all_by_pbs(data, metadata):
    for treat in treatments:
        pbs = metadata[(metadata['Drug'] == "PBS") & (metadata["Treatment"] == treat)]
        # get the pbs mice data
        pbs_data = data[pbs['ID']]
        # calculate the mean and std of the pbs mice
        pbs_mean = pbs_data.mean(axis=1)
        pbs_std = pbs_data.std(axis=1)
        # replace pbs_std 0 values by np.nanmin(pbs_std)
        pbs_std[pbs_std == 0] = np.nanmin(pbs_std[pbs_std != 0])
        data[pbs['ID']] = data[pbs['ID']].sub(pbs_mean, axis=0)
        data[pbs['ID']] = data[pbs['ID']].div(pbs_std, axis=0)
        for anti in antibiotics:
            abx = metadata[(metadata['Drug'] == anti) & (metadata["Treatment"] == treat)]
            # normalize the data by the mean and std of the pbs mice: subtract pbs_mean from every row and divide by std
            data[abx['ID']] = data[abx['ID']].sub(pbs_mean, axis=0)
            data[abx['ID']] = data[abx['ID']].div(pbs_std, axis=0)
    # return the normalized data
    return data


def zscore_all_by_pbs_gf(data_gf, metadata_gf):
    pbs = metadata_gf[metadata_gf['Drug'] == "PBS"]
    # get the pbs mice data
    pbs_data = data_gf[pbs['ID']]
    # calculate the mean and std of the pbs mice
    pbs_mean = pbs_data.mean(axis=1)
    pbs_std = pbs_data.std(axis=1)
    # replace pbs_std 0 values by np.nanmin(pbs_std)
    pbs_std[pbs_std == 0] = np.nanmin(pbs_std[pbs_std != 0])
    data_gf[pbs['ID']] = data_gf[pbs['ID']].sub(pbs_mean, axis=0)
    data_gf[pbs['ID']] = data_gf[pbs['ID']].div(pbs_std, axis=0)
    abx = metadata_gf[metadata_gf['Drug'] == "Van"]
    # normalize the data by the mean and std of the pbs mice: subtract pbs_mean from every row and divide by std
    data_gf[abx['ID']] = data_gf[abx['ID']].sub(pbs_mean, axis=0)
    data_gf[abx['ID']] = data_gf[abx['ID']].div(pbs_std, axis=0)
    # return the normalized data
    return data_gf


def get_mean_all(df, meta, condition, skip, run_type):
    # if the file f'./Private/means{run_type}.json' exists, return it and the stds
    if skip and os.path.exists(f'./Private/means{run_type}.json'):
        with open(f"./Private/means{run_type}.json", "r") as f:
            all_means = json.load(f)
        with open(f"./Private/stds{run_type}.json", "r") as f:
            all_stds = json.load(f)
        return all_means, all_stds
    all_means = {}
    seen = {}
    for i in range(df.shape[0]):
        if i % 100 == 0:
            print(f"calculating mean for {i}/{df.shape[0]}")
        for j in range(df.shape[1]):
            name = df.columns[j]
            antibiotic = meta[meta['ID'] == name]['Drug'].values[0]
            treatment = meta[meta['ID'] == name][condition].values[0]
            if antibiotic not in all_means:
                all_means[antibiotic] = {}
                seen[antibiotic] = {}
            if treatment not in all_means[antibiotic]:
                all_means[antibiotic][treatment] = []
                seen[antibiotic][treatment] = set()
            if df.index[i] not in seen[antibiotic][treatment]:
                seen[antibiotic][treatment].add(df.index[i])
                mice = meta[(meta['Drug'] == antibiotic) & (meta[condition] == treatment)]['ID']
                all_means[antibiotic][treatment].append(gmean(df.iloc[i][mice]))
    # keep mean and std of all lists for each antibiotic and treatment
    all_stds = {}
    for antibiotic in all_means:
        all_stds[antibiotic] = {}
        for treatment in all_means[antibiotic]:
            all_stds[antibiotic][treatment] = np.std(all_means[antibiotic][treatment])
            all_means[antibiotic][treatment] = np.mean(all_means[antibiotic][treatment])
    try:
        # save the means and stds
        with open(f"./Private/means{run_type}.json", "w") as f:
            json.dump(all_means, f)
        with open(f"./Private/stds{run_type}.json", "w") as f:
            json.dump(all_stds, f)
    except:
        print("couldn't save means and stds")
    return all_means, all_stds


def impute_zeros(to_impute, meta_data, condition, run_type='', skip_if_exist=False, mean=False):
    """
    replaces all zeros by mean of other gene expression of same treatment and same antibiotic
    """
    # if the file f'./Private/imputed_all_log_zeros_removed{run_type}.csv' exists, return it
    if skip_if_exist and os.path.exists(f'./Private/imputed_all_zeros_removed{run_type}.csv'):
        return pd.read_csv(f'./Private/imputed_all_zeros_removed{run_type}.csv', index_col=0)
    if mean:
        means, stds = get_mean_all(to_impute, meta_data, condition, skip_if_exist, run_type)
    # add column of nan counts of the row
    to_impute['nans'] = to_impute.isnull().sum(axis=1)
    to_impute = to_impute.drop(to_impute[to_impute['nans'] >= 0.2 * to_impute.shape[1]].index).drop('nans', axis=1)
    row, col = np.where(to_impute.isnull())
    total = len(row)
    counter = 1
    all_other_are_zeros = 0
    all_other_are_zeros_conditions = 0
    too_big = 0
    zeros = set()
    for i, j in zip(row, col):
        # assert it is nan
        assert np.isnan(to_impute.iloc[i, j])
        # print(f"replacing {to_impute[i][j]} with {to_impute[i][j]}")
        name = to_impute.columns[j]
        antibiotic = meta_data[meta_data['ID'] == name]['Drug'].values[0]
        treatment = meta_data[meta_data['ID'] == name][condition].values[0]
        # print(name, antibiotic, treatment)
        if counter % 5000 == 0:
            print(f"{counter}/{total} zeros imputed")
        counter += 1
        mice = meta_data[(meta_data['Drug'] == antibiotic) & (meta_data[condition] == treatment) &
                         (meta_data['ID'] != name)]['ID']
        if mean:
            # replace the zero with the geometric mean of the other mice
            mean = np.nanmean(to_impute.iloc[i][mice])
            if np.isnan(mean) or (antibiotic, treatment, to_impute.index[i]) in zeros:
                zeros.add((antibiotic, treatment, to_impute.index[i]))
                all_other_are_zeros += 1
                all_other_are_zeros_conditions += 1 / (len(mice) + 1)
                # set mean to be the min non-zero value of this sample (column)
                mean = np.min(to_impute[name][~to_impute[name].isnull()])
            if abs(mean - means[antibiotic][treatment]) > stds[antibiotic][treatment]:
                too_big += 1
                continue
            to_impute.iloc[i, j] = mean
        else:
            to_impute.iloc[i, j] = np.nanmin(to_impute.iloc[i][mice])

    row, col = np.where(to_impute.isnull())
    print(
        f"Now left with {len(row)} zeros, {set([to_impute.columns[c] for c in col])}")  # , but {all_other_are_zeros} are zeros in all other mice")
    # replace na with 1
    to_impute = to_impute.fillna(1)
    # print(f"in {int(all_other_are_zeros_conditions)} conditions. {too_big} were too big")
    #     to_impute = impute_zeros(to_impute, meta_data)
    to_impute.to_csv(f'./Private/imputed_all_zeros_removed{run_type}.csv')
    return to_impute


def transform_data(data, metadata, run_type, skip=False, save=False, gf=False):
    # replace all zeros with nan
    data = data.replace(0, np.nan)
    # # Remove V11 from data, and remove row ID==V11 from metadata
    # data = data.drop('V11', axis=1)
    # metadata = metadata.drop(metadata[metadata['ID'] == 'V11'].index)
    data = impute_zeros(data, metadata, 'Treatment', run_type, skip_if_exist=skip)
    data = np.log2(data)
    # z-score by PBS
    data = zscore_all_by_pbs(data, metadata) if not gf else zscore_all_by_pbs_gf(data, metadata)
    return data, metadata


def plot_multiabx_scatter(param, significant_genes, x, y):
    # plot #significant_genes vs go_number
    plt.figure(figsize=(4, 4))
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
    set_plot_defaults()
    # plt.title("Number of significant genes vs number of significant GO terms")
    plt.savefig(private + f"/analysis/{param}/{x}_vs_{y}.png")
    plt.show()
    plt.close()


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
    normalized_df = z_score_by_pbs(genes_df, abx_mice, pbs_mice)
    # normalized_df = genes_df
    # drop nan rows
    normalized_df = normalized_df.dropna()
    if normalized_df.shape[0] > 1:
        cluster = sns.clustermap(data=normalized_df, row_cluster=True, col_cluster=False,
                                 cmap='vlag')  # , xticklabels=True, yticklabels=True)
        # z_score=0, cmap='vlag')  # , xticklabels=True, yticklabels=True)
        order = cluster.dendrogram_row.reordered_ind
        plt.close()
        return normalized_df.fillna(0).iloc[order]  # .apply(zscore, axis=1)
    return normalized_df.fillna(0)


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


def plot_median_all_conditions(meta_data, raw_data, antibiotics, treatments, condition, exp_type,
                               labelsize=12, regular=True, cols_factor=6.0, rows_factor=5.0):
    matrices = get_median_matrices(antibiotics, condition, exp_type[1:], meta_data, raw_data, treatments, regular)
    axis = set_figure(treatments, antibiotics, cols_factor, rows_factor)
    GO_number = pd.DataFrame(index=antibiotics, columns=treatments, data=0)
    for j, treat in enumerate(treatments):
        for i, abx in enumerate(antibiotics):
            curr_axis = get_to_axis(axis, i, j, len(treatments), len(antibiotics))
            curr_axis.set_title(f"{abx}, {treat}")
            if matrices[treat][abx] is not None:
                curr_matrix = matrices[treat][abx]
                # sort columns and put all columns that ends with N in the end
                curr_matrix = curr_matrix.reindex(sorted(curr_matrix.columns, key=lambda x: x.endswith('N')), axis=1)
                # sns.heatmap(curr_matrix, vmin=-2.8, vmax=2, xticklabels=True, cmap="vlag", ax=curr_axis, cbar=cbar)
                GO_number.loc[abx, treat] = curr_matrix.shape[0]
            else:
                print(f"{abx} {treat} is empty")
    # save GO_number to a csv file
    GO_number.to_csv(private + fr"/GO_number.csv")


def compare_significance_go(raw, meta, param):
    plot_median_all_conditions(meta, raw, antibiotics, treatments, 'Treatment', '\\diff_abxRASflow',
                               regular=False)
    go_number = pd.read_csv(os.path.join(private, "GO_number.csv"),
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


def figure2():
    run_type = "RASflow"
    genome, meta, _, raw = read_process_files(new=False)
    raw, metadata = transform_data(raw, meta, run_type, skip=True)
    plot_significant_genes_number(meta, raw, antibiotics, treatments, "diff_abx" + run_type)
    our = plot_categories(antibiotics, treatments, "\\diff_abx" + run_type, False, regular=False)
    gsea = plot_categories(antibiotics, treatments, "\\diff_abx" + "GSEA", False, regular=False, gsea=True,
                           anchor=(0.5, -5.2))
    plot_correlation_gsea(gsea, our)
    compare_significance_go(raw, meta, param="diff_abx" + run_type)
    clusters_compare_mix(antibiotics, treatments, "diff_abx" + run_type)


def plot_auroc_vs_noise(result: dict, exp_name, response_tag):
    """
       Plots the mean AUROC vs. noise level.

       :param auroc_values_dict: AUROC values extracted using the `collect_auroc_data` method.
   """

    num_otus = result['otu']
    number_of_responses = result['response']
    coda_method = result['case']
    realizations = 15
    response_based = 'balance'
    base_dir = 'auroc-van/to_publish'
    noise_levels = [float(val) for val in result['noises']]
    auroc_values_dict = result['data']
    sample_size = result['num']

    # sort noise level by size (increasing) and sort auroc values accordingly
    order = np.argsort(noise_levels)
    noise_levels = np.array(noise_levels)[order]
    auroc_values_dict = np.array(auroc_values_dict, dtype=np.float64)[order]

    mean_data = {}
    std_data = {}
    for key, auroc_array in zip(noise_levels, auroc_values_dict):
        mean_data[key] = float(np.nanmean(auroc_array))
        std_data[key] = float(np.nanstd(auroc_array))

    # noise_levels = sorted(set(float(k[2]) for k in mean_data.keys()))
    # sample_sizes = sorted(set(k[1] for k in mean_data.keys()))

    # Create a figure with two vertical subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), gridspec_kw={'height_ratios': [3, 1]}, sharex="all")

    # Plot AUROC vs. noise level
    means = [val for val in mean_data.values()]
    ax1.plot(noise_levels, means, color='#e69b00', label=f'Processed sample size {sample_size}')
    # Add standard deviation as shadowed area
    stds = [val for val in std_data.values()]
    ax1.fill_between(noise_levels, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                     alpha=0.2, color='#e69b00', label=f'+- 1 SD')
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_xlabel('')
    ax1.set_ylabel('Mean AUROC')
    ax1.set_title(
        f'AUROC averaged across {realizations} realizations vs. Noise level'
        f'\nMicrobiome: {exp_name}; Response: {response_tag}'
        # f'\nLR transformation: OCU {coda_method} {response_based}'
    )
    ax1.legend()
    ax1.grid(True)

    # Prepare data for the box plot
    values = [float(noise) for noise in noise_levels]

    # Create a twin y-axis to preserve the y-axis of the line plot
    ax2.boxplot(values, vert=False, boxprops=dict(color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                medianprops=dict(color='red'))
    ax2.set_yticks([])
    ax2.set_xlabel('RMSE Values')

    # Add scatter plot for the original response RMSE values
    ax2.scatter(values, np.ones(len(values)), color='blue', alpha=0.1, s=30)
    # Set the x-axis limits for both plots
    ax1.set_xlim(noise_levels[0] * 0.95, noise_levels[-1] * 1.05)

    plt.tight_layout()
    set_plot_defaults()

    fig.savefig(
        f"./Private/{base_dir}/{exp_name}_{response_tag}_mean_auroc_vs_noise_{coda_method}_{response_based}_with_noise_analysis.png")
    fig.savefig(
        f"./Private/{base_dir}/{exp_name}_{response_tag}_mean_auroc_vs_noise_{coda_method}_{response_based}_with_noise_analysis.svg")

    plt.close(fig)


def extract_auroc_data(path):
    # Read the CSV, skipping the first unnamed column
    df = pd.read_csv(path, index_col=0, header=None)

    # Extract and assert values
    otu = df.iloc[0, :].unique()
    assert len(otu) == 1, "OTU values are not identical"
    otu = otu[0]

    num = df.iloc[1, :].unique()
    assert len(num) == 1, "NUM values are not identical"
    num = num[0]

    noises = df.iloc[2, :].tolist()

    response = df.iloc[3, :].unique()
    assert len(response) == 1, "Response values are not identical"
    response = response[0]

    case = df.iloc[4, :].unique()
    assert len(case) == 1 and case[0] == 'pairs', "Case values are not all 'pairs'"
    case = case[0]

    # Extract remaining columns as lists
    data = [df.iloc[5:, i].astype(float).tolist() for i in range(len(df.columns))]

    # Final results
    result = {
        'otu': otu,
        'num': num,
        'noises': noises,
        'response': response,
        'case': case,
        'data': data
    }

    return result


def figure_s():
    # tags = ["Isg15", "Oasl2", "Zbp1"]
    tags = ["Nfil3", "Ciart", "Dbp"]
    # otu = {"IP": 133, "PO": 129}
    # for treat in ["PO"]:
    for treat in ["PO", "IP"]:
        for response in range(1, 4):
            results = extract_auroc_data(f"./Private/auroc-van/to_publish/auroc_{treat}_{response}.csv")
            plot_auroc_vs_noise(results, f"Van-{treat}", tags[response - 1])


if __name__ == '__main__':
    figure1()
    figure2()
    figure_s()
