import os
from scipy.stats import ttest_ind
from ClusteringGO import transform_data
from groups_comparison import read_data_metadata

import pandas as pd

path = os.path.join(".", "Data", "Pairs")


def transform_name(col):
    # Standardize timepoint suffix
    timepoint = ".d0" if col.endswith("D0") else ".ep"

    # Map prefix to Group ID
    name_part = col.replace("D0", "").replace("EP", "")

    mapping = {
        "con-": "G1_",
        "vanco-": "G2_",
        "neo-": "G3_",
        "metro-": "G4_",
        "met-neo": "G5_",
        "met-van": "G6_"
    }

    name_abx = name_part.split("-")[0] + "-"
    if name_abx in mapping:
        col = mapping[name_abx] + name_part.split("-")[1]
    if "met-van" in name_part:
        name_abx = "met-van"
        col = mapping[name_abx] + name_part.split("van")[1]
    if "met-neo" in name_part:
        name_abx = "met-neo"
        col = mapping[name_abx] + name_part.split("neo")[1]
    return f"{col}{timepoint}"


def get_metagenomics():
    # Load the data
    # sep='\t' is crucial as MetaPhlAn outputs are tab-separated
    # comment='#' ignores the versioning line at the top
    df = pd.read_csv(path + 'merged_taxa_metaphlan.tsv', sep='\t', comment='#').dropna(how='all')

    # Optional: Set the clade_name as the index for easier lookups
    df.set_index('clade_name', inplace=True)
    # missing = [col for col in df.columns if col.endswith("_1") and col[:-1]+"2" not in df.columns]
    # print(f"Missing pairs for: {missing}")
    # missing = [col for col in df.columns if col.endswith("_2") and col[:-1]+"1" not in df.columns]
    # print(f"Missing pairs for: {missing}")
    # replace columns that end with "_1" to end with ".d0" and columns that end with "_2" to end with ".ep"
    df.columns = [col.replace("G-", "G") for col in df.columns]
    df.columns = [col[:-2] + ".d0" if col.endswith("_1") else (col[:-2] + ".ep" if col.endswith("_2") else col) for col
                  in df.columns]
    # families =  [fam for fam in df.index if "f__" in fam]
    # genura = [gen for gen in df.index if "g__" in gen]
    # species = [spec for spec in df.index if "s__" in spec]
    # print(f"Families: {len(families)}, Genura: {len(genura)}, Species: {len(species)}")
    return df


def get_taxonomic_level(df, level_prefix):
    """
    Filters the MetaPhlAn dataframe for a specific taxonomic level.
    level_prefix: 'f__' for family, 'g__' for genus, 's__' for species
    """
    # 1. Filter: Find rows containing the prefix but NOT the level below it
    # Use regex to find rows where the prefix is the last one in the string
    regex_pattern = f'{level_prefix}[^|]+$'
    filtered_df = df[df.index.str.contains(regex_pattern, regex=True)].copy()

    # 2. Re-normalization
    # Because 'UNCLASSIFIED' or omitted low-abundance taxa might exist,
    # the sum might not be exactly 100. We re-scale so the selected level sums to 100.
    normalized_df = filtered_df.div(filtered_df.sum(axis=0), axis=1) * 100_000

    return normalized_df


def get_significant(current, abx_data, pbs_data, threshold=0.05):
    genes = []
    for gene in current.index:
        # get treat-test score for the gene
        abx = (current.loc[gene][abx_data['ID']])
        pbs = (current.loc[gene][pbs_data['ID']])
        t_pbs, t_p_pbs = ttest_ind(pbs, abx)
        if abs(t_p_pbs) < threshold:
            genes.append(gene)
    return genes


def prepare_genes_to_compores(threshold=0.05, by_genes=False, folder=None, metagenomics=False, level='genus'):
    data, metadata, id_to_name = read_data_metadata(remove_mitochondrial=False)
    data.to_csv(f"../../DIABLO/genes.tsv", sep="\t")
    metadata.to_csv(f"../../DIABLO/metadata.tsv", sep="\t")
    antibiotics = metadata['Drug'].unique().tolist()
    antibiotics.remove('PBS')
    run_type = "_pairs"
    # todo: run without z-scoring, see also other clock genes like per2
    data, metadata = transform_data(data, metadata, run_type, skip=True)
    metadata["Category"] = metadata['Drug'] + "_" + metadata['Treatment']

    if metagenomics:
        folder = f"{folder}_metagenomics_{level}" if folder else None
        from pairs_microbiom import get_metagenomics, get_taxonomic_level
        metagenomics_df = get_metagenomics()
        if level == 'genus':
            microbiome = get_taxonomic_level(metagenomics_df, 'g__')
        elif level == 'species':
            microbiome = get_taxonomic_level(metagenomics_df, 's__')
        elif level == 'family':
            microbiome = get_taxonomic_level(metagenomics_df, 'f__')
        else:
            raise ValueError("level must be one of 'genus', 'species', 'family'")
    else:
        microbiome = pd.read_csv(path + "feature_table_genus_level.tsv", sep='\t')
        microbiome = microbiome.set_index("Genus_Level")
        microbiome = microbiome.drop(columns=["Confidence"])
        microbiome.columns = [transform_name(col) for col in microbiome.columns]
    # keep only columns with ep
    microbiome = microbiome[
        [col for col in microbiome.columns if (("ep" in col) and (col.split(".")[0] in data.columns))]]
    microbiome.columns = [col.split(".")[0] for col in microbiome.columns]
    microbiome = microbiome.T
    microbiome.index = microbiome.index.set_names("sample_id")
    microbiome.to_csv(f"../../DIABLO/microbiome{'_metagenomics_' + level if metagenomics else ''}.tsv", sep="\t")

    # if f"./Private/CompoResGenes/{folder}" does not exist, create it
    if folder and os.path.exists(f"./Private/CompoResGenes/{folder}") is False:
        os.makedirs(f"./Private/CompoResGenes/{folder}")
        # create folder "response" and "metadata"
        os.makedirs(f"./Private/CompoResGenes/{folder}/response", exist_ok=True)
        os.makedirs(f"./Private/CompoResGenes/{folder}/metadata", exist_ok=True)
        os.makedirs(f"./Private/CompoResGenes/{folder}/microbiome", exist_ok=True)
    for abx in antibiotics:
        samples = metadata[((metadata["Drug"] == abx) | (metadata["Drug"] == "PBS"))]
        curr_microbiome = microbiome.loc[samples["ID"]]
        curr_microbiome.to_csv(f"./Private/CompoResGenes/{folder}/microbiome/{abx}-pairs-feces.tsv", sep="\t")
        curr = data[samples["ID"]]
        if not by_genes:
            abx_data = metadata[metadata['Drug'] == abx]
            pbs_data = metadata[metadata['Drug'] == 'PBS']
            genes = get_significant(curr, abx_data, pbs_data, threshold=threshold)
        else:
            genes = [gene for gene in by_genes if gene in curr.index]
            print(f"{len(genes)} are available out of original {len(by_genes)}")
        curr_genes = curr.T[genes]
        # curr_genes.to_csv(f"./Private/feeding/{abx}-{treat}.tsv", sep="\t")
        addition = f"{folder}/response/" if folder else ""
        curr_genes.to_csv(f"./Private/CompoResGenes/{addition}{abx}-pairs.tsv", sep="\t")
        print(f"Number of significant genes for {abx}-pairs: {len(genes)}")
        save_meta = samples.set_index("ID")['Category']
        addition = f"{folder}/metadata/" if folder else ""
        save_meta.to_csv(f"./Private/CompoResGenes/{addition}{abx}-pairs-metadata.tsv", sep="\t")


if __name__ == "__main__":
    # Example usage
    # prepare_genes_to_compores(threshold=0.05, folder="pairs")

    # follow prev results
    from compores_results_analysis import akiko_check, get_ensmus_dict

    # part_circadian_clock_genes = ["nfil3", "ciart", "dbp", "per3"]  # , "arntl"]
    part_circadian_clock_genes = ["Clock", "Arntl", "Bmal1", "Chrono", "Nfil3", "Nr1d1", "Per1", "Per2",
                                  "Cry1", "Cry2", "Dbp"]
    autophagy_genes = ['Atg10', 'Atg101', 'Atg12', 'Atg13', 'Atg14', 'Atg16l1', 'Atg16l2', 'Atg2a', 'Atg2b',
                       'Atg3', 'Atg4a', 'Atg4b', 'Atg4c', 'Atg4d', 'Atg5', 'Atg7', 'Atg9a', 'Atg9b',
                       'Map1lc3a', 'Map1lc3b', 'Sqstm1', 'Gabarap', 'Gabarapl1', 'Gabarapl2',
                       'Becn1', 'Ulk1', 'Ulk2', 'Ulk3', 'Ulk4', 'Wipi2']
    # todo: see clock+neo PO that they agree between DIABLO & CompoRes, 16S and metagenomics
    # todo: Spearman correlation between 16S and metagenomics for all significant genes, and plot -plog p-val
    viral_genes = akiko_check()
    # capitalize the genes
    part_circadian_clock_genes = [gene.capitalize() for gene in part_circadian_clock_genes]
    autophagy_genes = [gene.capitalize() for gene in autophagy_genes]
    viral_genes = [gene.capitalize() for gene in viral_genes]
    ensmus_dict = get_ensmus_dict()
    # reverse this dictionary
    names_dict = {v: k for k, v in ensmus_dict.items()}
    ensmus_clock_genes = list(names_dict[gene] for gene in part_circadian_clock_genes if gene in names_dict)
    ensmus_autophagy_genes = list(names_dict[gene] for gene in autophagy_genes if gene in names_dict)
    ensmus_viral_genes = list(names_dict[gene] for gene in viral_genes if gene in names_dict)
    prepare_genes_to_compores(threshold=0.05, by_genes=ensmus_clock_genes, folder="pairs_clock")
    # prepare_genes_to_compores(threshold=0.05, by_genes=ensmus_autophagy_genes, folder="pairs_autophagy_all")
    # prepare_genes_to_compores(threshold=0.05, by_genes=ensmus_viral_genes, folder="pairs_viral")
    for level in ["genus", "species", "family"]:
        prepare_genes_to_compores(threshold=0.05, by_genes=ensmus_clock_genes, folder="pairs_clock", metagenomics=True,
                                  level=level)
        # prepare_genes_to_compores(threshold=0.05, by_genes=ensmus_autophagy_genes, folder="pairs_autophagy_all", metagenomics=True, level=level)
        # prepare_genes_to_compores(threshold=0.05, by_genes=ensmus_viral_genes, folder="pairs_viral", metagenomics=True, level=level)
        # prepare_genes_to_compores(threshold=0.05, folder="pairs", metagenomics=True, level=level)
