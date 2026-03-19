import json
import os
from collections import defaultdict, deque
from json import JSONEncoder
from typing import Dict, Set, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wget
from anytree import NodeMixin, PostOrderIter
from anytree.importer import JsonImporter
from goatools import obo_parser
from scipy import stats
# from goatools.base import download_go_basic_obo, download_ncbi_associations
# from line_profiler import LineProfiler
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.stats.mstats import gmean
from statsmodels.stats.multitest import fdrcorrection

# TODO: add option to use another root, not BP - might be one node below or anything below
# See if GSEA has data to compare to


# profiler = LineProfiler()

# # Get http://geneontology.org/ontology/go-basic.obo
# obo_fname = download_go_basic_obo()
# # Get ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz
# gene2go = download_ncbi_associations()
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

# For an absolute path (commented out, use if needed)
# private = os.path.join("C:", "Users", "Yehonatan", "Desktop", "Master", "Git", "DEP_Compare16s", "Private")

directory = private
path = os.path.join(private, "clusters_properties")

# Note the use of ".." to go up a directory from the current location
data_folder = os.path.join("..", "Data", "MultiAbx-16s", "MultiAbx-RPKM-RNAseq-B6", "new normalization")


class GeneNode(NodeMixin, JSONEncoder):  # Add Node feature
    def __init__(self, go_id, level, name, go_obj, parents=None, children=None):
        super().__init__()
        self.go_id = go_id
        self.level = level
        self.name = name
        category = list(get_ancestor(go_obj))
        self.category = category[0].name if len(category) else "biological process"
        # self.go_object = go_obj
        self.parents = parents if parents else set()
        self.parent = parents
        self.children = children if children else set()
        # self.all_children = children if children else set()
        self.gene_set = set()
        self.pearson_corr = None
        self.spearman_corr = None
        self.dist = np.inf

    def __repr__(self):
        return self.go_id

    def __str__(self):
        return self.go_id

    def serialize(self):
        self.parents = list(self.parents)
        self.children = list(self.children)
        # self.all_children = list(self.children)
        self.gene_set = list(self.gene_set)

    def unserialize(self):
        self.parents = set(self.parents)
        self.children = set(self.children)
        # self.all_children = set(self.children)
        self.gene_set = set(self.gene_set)

    def toJson(self, o):
        if isinstance(o, GeneNode):
            return o.__dict__
        else:
            raise TypeError


def get_ancestor(go_term):
    last = set()
    to_check = {go_term}
    while to_check:
        term = to_check.pop()
        for parent in term.parents:
            if parent.id == "GO:0008150":
                last.add(term)
            else:
                to_check.add(parent)
    return last


# def convert_to_ensembl(gene_to_go_mapping):
#     import mygene
#     mg = mygene.MyGeneInfo()
#     all_genes = list(gene_to_go_mapping.keys())
#
#     # Query the mygene API to get Ensembl IDs for the genes
#     gene_info = mg.querymany(all_genes, scopes='symbol,refseq,uniprot', fields='ensembl.gene', species='mouse',
#                              returnall=True)
#
#     ensembl_gene_to_go_mapping = {}
#
#     for entry in gene_info['out']:
#         gene_symbol = entry['query']
#
#         ensembl_data = entry.get('ensembl', {})
#         if isinstance(ensembl_data, list):
#             # If there are multiple Ensembl entries, take the first one
#             ensembl_id = ensembl_data[0].get('gene')
#         else:
#             ensembl_id = ensembl_data.get('gene')
#
#         # If there's a valid Ensembl ID, add it to the new mapping
#         if ensembl_id:
#             ensembl_gene_to_go_mapping[ensembl_id] = gene_to_go_mapping[gene_symbol]
#
#     return ensembl_gene_to_go_mapping


# def create_gene_id_dict():
#     import wget
#     import gzip
#     import shutil
#     # from goatools.obo_parser import GODag
#     # 1. Download necessary files using wget
#     # Download the Gene Ontology obo file
#     go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
#     wget.download(go_obo_url, directory + 'go-basic.obo')
#     # Download the gene association file for Mus musculus from EBI
#     association_url_ebi = 'ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/MOUSE/goa_mouse.gaf.gz'
#     wget.download(association_url_ebi, directory + 'goa_mouse.gaf.gz')
#     # 2. Decompress the goa_mouse.gaf.gz file
#     with gzip.open(directory + 'goa_mouse.gaf.gz', 'rb') as f_in:
#         with open(directory + 'goa_mouse.gaf', 'wb') as f_out:
#             shutil.copyfileobj(f_in, f_out)
#     # 3. Clean the goa_mouse.gaf file to skip header lines
#     with open(directory + 'goa_mouse.gaf', 'r', encoding='utf-8') as f_in:
#         with open(directory + 'cleaned_goa_mouse.gaf', 'w', encoding='utf-8') as f_out:
#             for line in f_in:
#                 if not line.startswith('!'):
#                     f_out.write(line)
#     # # 4. Load the Gene Ontology
#     # obodag = GODag("go-basic.obo")
#     # 5. Manually parse the cleaned_goa_mouse.gaf file
#     gene_to_go_mapping = {}
#     with open(directory + 'cleaned_goa_mouse.gaf', 'r', encoding='utf-8') as f:
#         for line in f:
#             columns = line.strip().split('\t')
#             if len(columns) > 5:
#                 gene_id = columns[1]
#                 go_term = columns[4]
#                 if gene_id not in gene_to_go_mapping:
#                     gene_to_go_mapping[gene_id] = []
#                 gene_to_go_mapping[gene_id].append(go_term)
#     return gene_to_go_mapping


# def gene_id_to_go():
#     import json
#     import os
#
#     directory = "./Private/gene to GO"
#     # File path
#     file_path = directory + 'ensembl_gene_to_go_mapping.json'
#
#     # Check if the file exists
#     if os.path.exists(file_path):
#         # Load the dictionary from the JSON file
#         with open(file_path, 'r') as infile:
#             ensembl_gene_to_go_mapping = json.load(infile)
#     else:
#         gene_to_go_mapping = create_gene_id_dict()
#
#         print("genes:", len(gene_to_go_mapping))
#         ensembl_gene_to_go_mapping = convert_to_ensembl(gene_to_go_mapping)
#
#         # # Print the first few items in the dictionary for verification
#         # for gene_id, gos in list(ensembl_gene_to_go_mapping.items())[:5]:
#         #     print(f"{gene_id} is associated with GO terms {', '.join(gos)}")
#         print("ensmble:", len(ensembl_gene_to_go_mapping))
#         # Create the mapping
#         ensembl_gene_to_go_mapping = convert_to_ensembl(gene_to_go_mapping)
#
#         # Save the dictionary to a JSON file
#         with open(file_path, 'w') as outfile:
#             json.dump(ensembl_gene_to_go_mapping, outfile, indent=4)
#     return ensembl_gene_to_go_mapping


# def convert_to_go_to_ensembl(ensembl_gene_to_go_mapping):
#     go_to_ensembl_gene_mapping = {}
#
#     for ensembl_id, go_terms in ensembl_gene_to_go_mapping.items():
#         for go_term in go_terms:
#             if go_term not in go_to_ensembl_gene_mapping:
#                 go_to_ensembl_gene_mapping[go_term] = []
#             go_to_ensembl_gene_mapping[go_term].append(ensembl_id)
#
#     return go_to_ensembl_gene_mapping


# def go_to_gene_id():
#     import json
#     import os
#
#     directory = "./Private/gene to GO"
#     # File path for GO to Ensembl mapping
#     file_path = directory + 'go_to_ensembl_gene_mapping.json'
#
#     # Check if the file exists
#     if os.path.exists(file_path):
#         # Load the dictionary from the JSON file
#         with open(file_path, 'r') as infile:
#             go_to_ensembl_gene_mapping = json.load(infile)
#     else:
#         ensembl_gene_to_go_mapping = gene_id_to_go()
#         go_to_ensembl_gene_mapping = convert_to_go_to_ensembl(ensembl_gene_to_go_mapping)
#
#         # Save the dictionary to a JSON file
#         with open(file_path, 'w') as outfile:
#             json.dump(go_to_ensembl_gene_mapping, outfile, indent=4)
#
#     return go_to_ensembl_gene_mapping


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


def build_genomic_tree(biological_processes: Any, go: Dict) -> Tuple[GeneNode, int]:
    """
    BFS traverse of the DAG
    """
    visited: Set[str] = set()
    root = GeneNode(go_id=biological_processes.id, level=biological_processes.level,
                    name=biological_processes.name, go_obj=biological_processes, parents=None)
    to_visit = deque([root])
    id_to_node: Dict[str, GeneNode] = {biological_processes.id: root}
    nodes = 0

    while to_visit:
        current = to_visit.popleft()
        if current.go_id not in visited:
            visited.add(current.go_id)
            nodes += 1

            if current.go_id in go:
                for child in go[current.go_id].children:
                    if child.id not in id_to_node:
                        temp_node = GeneNode(go_id=child.id, level=child.level, name=child.name, go_obj=child)
                        id_to_node[child.id] = temp_node
                        to_visit.append(temp_node)
                    else:
                        temp_node = id_to_node[child.id]
                    current.children += (temp_node,)
                    temp_node.parents.add(current)

        if nodes % 5000 == 0:
            print(f"Processed {nodes} nodes")

    print(f"Total {nodes} nodes for {biological_processes.id}")
    bio_terms = [term for term in go if go[term].namespace == 'biological_process']
    missing_terms = set(bio_terms) - visited
    print(f"missing {len(missing_terms)}, Examples: {list(missing_terms)[:5]}")

    return root, nodes


# def build_genomic_tree(biological_processes, go):
#     """
#     BFS traverse of the DAG
#     """
#     visited = set()
#     # visited = visited.add(biological_processes.id)
#     root = GeneNode(go_id=biological_processes.id, level=biological_processes.level, name=biological_processes.name,
#                     go_obj=biological_processes, parents=None)
#     to_visit = [root]
#     id_to_node = {biological_processes.id: root}
#     nodes = 0
#     while len(to_visit):
#         current = to_visit.pop()
#         visited.add(current.go_id)
#         nodes += 1
#         for child in go[current.go_id].children:
#             if child.id not in visited:
#                 temp_node = GeneNode(go_id=child.id, level=child.level, name=child.name, go_obj=child)
#                 to_visit.append(temp_node)
#                 id_to_node[temp_node.go_id] = temp_node
#             else:
#                 temp_node = id_to_node[child.id]
#             current.children += tuple([temp_node])
#             temp_node.parents.add(current)
#             # current.all_children.add(temp_node)
#         # else:
#         #     current.children = None
#         # check if to_visit is empty
#         if nodes % 5000 == 0:
#             print(nodes)
#     print(f"total {nodes} nodes for {biological_processes.id}")
#     return root, nodes


def add_genes_names(root, genes_df):
    empty_nodes_counter = 0
    # Create a dictionary mapping GO IDs to sets of gene IDs
    go_to_genes = genes_df.groupby('go_id')['gene'].apply(set).to_dict()
    for i, node in enumerate(PostOrderIter(root)):
        node_genes = go_to_genes.get(node.go_id, set())
        if node_genes:
            node.gene_set = node.gene_set.union(node_genes)
        else:
            empty_nodes_counter += 1
        if i % 500 == 0:
            print(f"### {i} nodes were updated ###")
    print(f"{empty_nodes_counter} empty nodes")
    return root


def get_random_corr(size, df, plot=False, times=10_000):
    # Create an array to store the random samples
    sample_genes = np.array([np.random.choice(df.index, size=size, replace=False) for _ in range(times)])

    # Compute the standard deviation of the selected rows, take the mean across columns, and store the result
    # random_dist = np.array([np.nanmean(np.nanstd(df.loc[sample_genes[i]], axis=0)) for i in range(times)])
    random_dist = np.array([average_pairwise_spearman(df.loc[sample_genes[i]]) for i in range(times)])

    ecdf_data = save_ecdf_efficient(random_dist, tail_threshold=0.05, mid_step=0.05)
    # np.save('bootstrap_ecdf_efficient.npy', ecdf_data)
    #
    # # Later, to calculate p-value
    # ecdf_data = np.load('bootstrap_ecdf_efficient.npy', allow_pickle=True).item()

    if plot:
        log_dist = np.log(random_dist[random_dist > 0])
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        ax.hist(log_dist, bins=50, density=True, alpha=0.7, color='skyblue')

        # Plot kernel density estimation
        # sns.kdeplot(log_dist, ax=ax, color='navy')
        mean = np.mean(log_dist)
        std = np.std(log_dist)
        x = np.linspace(-1, 6, 100)
        plt.plot(x, (1 / (np.sqrt(2 * np.pi * std ** 2))) * np.exp(-np.power(x - mean, 2) / (2 * (std ** 2))))
        # Add labels and title
        ax.set_xlabel('Mean Standard Deviation')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Random Correlations')

        # Add text with mean and standard deviation
        mean = np.mean(log_dist)
        std = np.std(log_dist)
        ax.text(0.95, 0.95, f'Mean: {mean:.4f}\nStd: {std:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.title(f"size {size}")

        # Show the plot
        plt.show()

    # Return the mean and standard deviation of the computed distribution
    return random_dist.mean(), random_dist.std(), ecdf_data


def mean_mwu(anti, genes_data, expression, meta, treatment, condition):
    # # get median gene for genes_data
    # mean = np.mean(current.loc[genes_data], axis=1)
    # gene_index = np.argmin(np.array(np.abs(np.median(mean) - mean)))
    # gene = mean.index[gene_index]
    mw = np.zeros(len(genes_data))
    for i, gene in enumerate(genes_data):
        # get Mann–Whitney score for the gene
        abx, pbs = get_abx_pbs(anti, expression, gene, meta, treatment, condition)

        # maximal = len(abx) * len(pbs)
        if len(pbs) == 0 or len(abx) == 0 or (np.sum(abx) == 0 and np.sum(pbs) == 0):
            mw[i] = np.nan
            continue
        if np.array_equal(abx.values, pbs.values):
            print(f"abx and pbs are the same for {anti}_{treatment}_{gene}")
            mw[i] = np.nan
            continue
        # MWU_pbs = mannwhitneyu(pbs, abx)
        # MWU_abx = mannwhitneyu(abx, pbs)
        # mw[i] = max(MWU_abx[1], MWU_pbs[1])
        # Perform Mann-Whitney U test once
        _, mw[i] = mannwhitneyu(abx, pbs, alternative='two-sided')
    return gmean(mw)


def mean_fold(anti, genes_data, expression, meta, treatment, condition):
    # # get median gene for genes_data
    # mean = np.mean(current.loc[genes_data], axis=1)
    # gene_index = np.argmin(np.array(np.abs(np.median(mean) - mean)))
    # gene = mean.index[gene_index]
    fold_change = np.zeros(len(genes_data))
    for i, gene in enumerate(genes_data):
        abx, pbs = get_abx_pbs(anti, expression, gene, meta, treatment, condition)
        if len(pbs) == 0 or len(abx) == 0 or (np.sum(abx) == 0 and np.sum(pbs) == 0) or np.array_equal(abx.values,
                                                                                                       pbs.values):
            fold_change[i] = np.nan
            continue
        if pbs.median():
            fold_change[i] = abx.median() / pbs.median()
        else:
            print(f"pbs median is 0 for {anti}_{treatment}_{gene}")
            fold_change[i] = np.nan
    return np.nanmean(fold_change)


def geomean_t_test(anti, genes_data, current, meta, treatment, condition):
    """
    get the average treat-test over all genes
    """
    t_test = np.zeros(len(genes_data))
    for i, gene in enumerate(genes_data):
        # get treat-test score for the gene
        abx_data = meta[(meta['Drug'] == anti) & (meta[condition] == treatment)]
        pbs_data = meta[(meta['Drug'] == 'PBS') & (meta[condition] == treatment)]
        abx = (current.loc[gene][abx_data['ID']])  # .dropna()
        pbs = (current.loc[gene][pbs_data['ID']])  # .dropna()
        t_pbs, t_p_pbs = ttest_ind(pbs, abx)
        # t_abx, t_p_abx = ttest_ind(abx, pbs)
        # if t_abx != t_pbs:
        #     print("not equal ", t_abx, t_pbs)
        #     print("p value not equal? ", t_p_abx, t_p_pbs)
        # t_test[i] = max(t_p_abx, t_p_pbs)
        t_test[i] = abs(t_p_pbs)
    # print(f"geometric mean of treat-test for {anti}_{treatment} is {gmean(t_test)}"
    #       f" while the mean is {np.nanmean(t_test)}"
    #       f" genes are {genes_data if len(genes_data) < 10 else len(genes_data)}")
    if np.nanmean(t_test) is np.nan or gmean(t_test) is np.nan:
        print(anti, treatment, t_test)
    return gmean(t_test)


def median_t_test(anti, genes_data, current, meta, treatment, condition):
    """
    get the average treat-test over all genes
    """
    # get treat-test score for the median
    abx_samples = meta[(meta['Drug'] == anti) & (meta[condition] == treatment)]
    pbs_samples = meta[(meta['Drug'] == 'PBS') & (meta[condition] == treatment)]
    gene = current.loc[genes_data].median()
    abx = gene[abx_samples['ID']]  # .dropna()
    pbs = gene[pbs_samples['ID']]  # .dropna()
    t_pbs, t_p_pbs = ttest_ind(pbs, abx)
    return t_p_pbs


def median_mwu(anti, genes_data, current, meta, treatment, condition):
    """
    get the average mwu over all genes
    """
    genes_data_in = [gene for gene in genes_data if gene in current.index]
    if len(genes_data_in) != len(genes_data):
        print(f"Only {len(genes_data_in)}/{len(genes_data)} are in file")
    # get treat-test score for the median
    abx_samples = meta[(meta['Drug'] == anti) & (meta[condition] == treatment)]
    pbs_samples = meta[(meta['Drug'] == 'PBS') & (meta[condition] == treatment)]
    gene = current.loc[genes_data_in].median()
    abx = gene[abx_samples['ID']]  # .dropna()
    pbs = gene[pbs_samples['ID']]  # .dropna()
    t_pbs, t_p_pbs = mannwhitneyu(pbs, abx)
    return t_p_pbs


def median_fold_change(anti, genes_data, current, meta, treatment, condition):
    """
    Calculate the difference in median z-scores between antibiotic and PBS groups.

    Steps:
    1. Calculate median z-score per sample for the selected genes
    2. Calculate mean of median z-scores per group (antibiotic and PBS)
    3. Calculate the difference between these means
    4. Return the z-score difference
    """
    # get FC for the median
    abx_samples = meta[(meta['Drug'] == anti) & (meta[condition] == treatment)]
    pbs_samples = meta[(meta['Drug'] == 'PBS') & (meta[condition] == treatment)]
    # Get gene expression data for the selected genes
    gene_data = current.loc[genes_data]
    # Step 1: Calculate median per sample
    abx_medians = gene_data[abx_samples['ID']].median()
    pbs_medians = gene_data[pbs_samples['ID']].median()
    # Step 2: Calculate mean of medians per group
    abx_mean = np.mean(abx_medians)
    pbs_mean = np.mean(pbs_medians)
    # Step 3: Calculate fold change
    z_score_difference = abx_mean - pbs_mean
    return z_score_difference


def genes_data_split(anti, genes_data, current, meta, treatment, condition):
    enhanced, suppressed = set(), set()
    significant_genes = {}
    for gene in genes_data:
        # get treat-test score for the gene
        abx, pbs = get_abx_pbs(anti, current, gene, meta, treatment, condition)
        t_abx, t_p_abx = ttest_ind(abx, pbs)
        if t_abx > 0:  # meaning the abx is enhanced
            enhanced.add(gene)
        else:
            suppressed.add(gene)
        if t_p_abx < 0.05:
            significant_genes[gene] = t_p_abx
    return enhanced, suppressed, significant_genes
    # return list(enhanced), list(suppressed)


def get_abx_pbs(anti, current, gene, meta, treatment, condition):
    abx_data = meta[(meta['Drug'] == anti) & (meta[condition] == treatment)]
    pbs_data = meta[(meta['Drug'] == 'PBS') & (meta[condition] == treatment)]
    # if len(current.loc[gene]) > 1, take the row with the fewer nan values
    row = current.loc[gene]
    if len(row.shape) > 1 and len(row) > 1:
        row = row.dropna()
        print(f"for some reason gene {gene} appears twice")
    abx = (row[abx_data['ID']]).dropna()
    pbs = (row[pbs_data['ID']]).dropna()
    return abx, pbs


# @profiler

def plot_curve(random_cutoff, random_std, path):
    # Extract keys, values, and standard deviations
    keys = list(random_cutoff.keys())
    values = [random_cutoff[key] for key in keys]
    std_devs = [random_std[key] for key in keys]

    # Plotting
    plt.errorbar(keys, values, yerr=std_devs, fmt='o', capsize=5, capthick=1, ecolor='red')
    plt.xlabel('Genes number (group size)')
    plt.ylabel('Average mean-std of the group')
    plt.title('Values with Error Bars')
    # verify if path exists, if not create it
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(path + ".png")
    # plt.show()
    plt.close()


def save_ecdf_efficient(bootstrap_results, tail_threshold=0.05, mid_step=0.05):
    sorted_data = np.sort(bootstrap_results)
    n = len(sorted_data)

    # Calculate full ECDF
    ecdf_values = np.arange(1, n + 1) / n

    # Initialize lists to store selected points
    selected_data = []
    selected_ecdf = []

    # Save tail data (both lower and upper)
    lower_idx = np.searchsorted(ecdf_values, tail_threshold)
    upper_idx = np.searchsorted(ecdf_values, 1 - tail_threshold)

    selected_data.extend(sorted_data[:lower_idx])
    selected_ecdf.extend(ecdf_values[:lower_idx])

    # Save middle data with larger steps
    current_value = tail_threshold
    while current_value < (1 - tail_threshold):
        idx = np.searchsorted(ecdf_values, current_value)
        selected_data.append(sorted_data[idx])
        selected_ecdf.append(ecdf_values[idx])
        current_value += mid_step

    selected_data.extend(sorted_data[upper_idx:])
    selected_ecdf.extend(ecdf_values[upper_idx:])

    return {'data': np.array(selected_data), 'ecdf': np.array(selected_ecdf)}


def calculate_pvalue_ecdf_efficient_lower_tail(observed_value, ecdf_data, tail):
    data = ecdf_data['data']
    ecdf_values = ecdf_data['ecdf']
    n = len(data)  # Number of bootstrap samples

    # Small continuity correction
    correction = 1 / (2 * n)

    if observed_value >= 0:
        # Upper-tail for positive correlations
        idx = np.searchsorted(data, observed_value, side='right')
        if idx == 0:
            p_value = 1.0 - correction  # Avoid p-value of 1
        elif idx == len(data):
            p_value = correction  # Avoid p-value of 0
        else:
            # Interpolate between the two nearest points
            x0, x1 = data[idx - 1], data[idx]
            y0, y1 = ecdf_values[idx - 1], ecdf_values[idx]
            p_value = 1.0 - (y0 + (observed_value - x0) * (y1 - y0) / (x1 - x0))
            p_value = max(min(p_value, 1.0 - correction), correction)  # Apply continuity correction
    else:
        # Lower-tail for negative correlations
        idx = np.searchsorted(data, observed_value, side='left')
        if idx == 0:
            p_value = correction  # Avoid p-value of 0
        elif idx == len(data):
            p_value = 1.0 - correction  # Avoid p-value of 1
        else:
            # Interpolate between the two nearest points
            x0, x1 = data[idx - 1], data[idx]
            y0, y1 = ecdf_values[idx - 1], ecdf_values[idx]
            p_value = y0 + (observed_value - x0) * (y1 - y0) / (x1 - x0)
            p_value = max(min(p_value, 1.0 - correction), correction)  # Apply continuity correction

    # Return the lower-tail p-value directly
    return p_value


def average_pairwise_spearman(gene_data):
    """
    Calculate the average pairwise Spearman correlation for all pairs of genes.

    Parameters:
    gene_data (pd.DataFrame): A DataFrame where rows are genes and columns are samples.

    Returns:
    float: The average pairwise Spearman correlation.
    """
    # correlations = []
    #
    # for gene1, gene2 in combinations(gene_data.index, 2):
    #     corr, _ = stats.spearmanr(gene_data.loc[gene1], gene_data.loc[gene2])
    #     correlations.append(corr)
    #
    # return np.mean(correlations)
    corr_matrix = gene_data.T.corr(method='spearman')
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_values = upper_tri.values[np.triu_indices(corr_matrix.shape[0], k=1)]
    # corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).dropna().values
    return np.mean(corr_values)


def calculate_hypergeometric_pvalue(N, K, n, k):
    """
    Calculate the hypergeometric p-value.

    Parameters:
    N : int
        Total number of genes
    K : int
        Total number of significant genes
    n : int
        Number of genes in the GO term
    k : int
        Number of significant genes in the GO term

    Returns:
    float
        The p-value
    """
    # Calculate the probability of getting k or more successes
    pvalue = 1 - stats.hypergeom.cdf(k - 1, N, K, n)

    return pvalue


# def calculate_correlation(root, expression, meta, size, antis, treats, gene_to_check, exp_type, condition, id_to_name,
def calculate_correlation(expression, meta, antis, treats, gene_to_check, exp_type, condition, id_to_name,
                          remove=['N18']):
    sigmas_cutoff = 2
    print_cluster = True
    count_prints = 0
    go_to_ensmbl_dict = get_go_to_ensmusg()
    go = obo_parser.GODag(get_go())
    size = len(go_to_ensmbl_dict)

    # pearson = {}
    # spearman = {}

    dist = {}
    # expression = np.log2(gene_expression.fillna(0) + 1)
    top = pd.DataFrame()
    for anti in antis:
        # pearson[anti] = {}
        # spearman[anti] = {}
        dist[anti] = {}
        for treatment in treats:
            print(f"starting {anti} {treatment}")
            temp = pd.DataFrame()
            # pearson[anti][treatment] =  np.zeros(size)
            # spearman[anti][treatment] = np.zeros(size)
            # instead of couples correlation, calculate mean (log(x+1) - mean(log(x+1)))
            dist[anti][treatment] = np.zeros(size)
            # drug = anti.lower()
            samples = meta[((meta['Drug'] == anti) | (meta['Drug'] == 'PBS')) & (meta[condition] == treatment)]
            # remove missing samples
            for sample in remove:
                if sample in samples['ID'].values:
                    samples = samples.drop(samples[samples['ID'] == sample].index)
                    print(f"{sample} removed")
            iter_samples = samples['ID'].to_list()
            for sample in iter_samples:
                if sample not in expression.columns:
                    print(f"{sample} not in expression")
                    samples = samples[samples['ID'] != sample]
                    meta = meta[meta['ID'] != sample]
                    # samples = samples.drop(samples[samples['ID'] == id].index)
            current = expression[samples['ID']]
            abx_samples = samples[samples['Drug'] == anti]
            pbs_samples = samples[samples['Drug'] == 'PBS']
            current_abx = expression[abx_samples['ID']]
            current_pbs = expression[pbs_samples['ID']]
            # Calculate trend for all genes for this condition
            genes_enhanced_all, genes_suppressed_all, significant_genes = genes_data_split(anti, current.index, current,
                                                                                           meta,
                                                                                           treatment, condition)
            # significant_gos = chances_of_significance(anti, treatment, go_to_ensmbl_dict, significant_genes)
            counter = 0
            random_cutoff_enh = {}
            random_std_enh = {}
            ecdf_storage_enh = {}
            random_cutoff_supp = {}
            random_std_supp = {}
            ecdf_storage_supp = {}
            # mwu_cutoff = {}
            for i, node in enumerate(go_to_ensmbl_dict):
                # for i, node in enumerate(PreOrderIter(root)):
                if len(set(go_to_ensmbl_dict[node])) == 0:
                    continue
                if node not in go:
                    continue
                genes_not_in_data = set(set(go_to_ensmbl_dict[node]) - set(current.index))
                no_genes = len(genes_not_in_data)
                # for gene in genes_not_in_data:
                #     node.gene_set.remove(gene)
                go_to_ensmbl_dict[node] = [gene for gene in go_to_ensmbl_dict[node] if gene not in genes_not_in_data]
                # if no_genes:
                #     print(
                #         f"{no_genes} genes were not in all_data ({anti} {treatment}) for {go[node].name if node in go else 'NO_NAME'}")
                # genes_enhanced, genes_suppressed = genes_data_split(anti, go_to_ensmbl_dict[node], current, meta,
                #                                                     treatment, condition)
                genes_enhanced = [gene for gene in go_to_ensmbl_dict[node] if gene in genes_enhanced_all]
                genes_suppressed = [gene for gene in go_to_ensmbl_dict[node] if gene in genes_suppressed_all]

                GO_significance = calculate_hypergeometric_pvalue(len(current.index), len(significant_genes),
                                                                  len(go_to_ensmbl_dict[node]),
                                                                  len([gene for gene in go_to_ensmbl_dict[node] if
                                                                       gene in significant_genes]))
                enhanced = True
                for genes_data in [genes_enhanced, genes_suppressed]:
                    # if dist is not np.nan and distance > 0 and len(genes_data) > 0:
                    # if distance > 0 and len(genes_data) > 0:
                    if len(genes_data) > 1:  # 0?
                        category_size = round(len(genes_data) / 10) * 10 if len(genes_data) > 50 else len(genes_data)
                        if category_size == 0:
                            continue
                        if enhanced:
                            if category_size not in random_cutoff_enh:
                                # print(f"adding {len(genes_data)} to random")
                                # calculate random groups pairwise correlation
                                random_cutoff_enh[category_size], random_std_enh[category_size], ecdf_storage_enh[
                                    category_size] = get_random_corr(category_size,
                                                                     current.loc[list(genes_enhanced_all)])
                                # random_cutoff[category_size] = rand_distance
                                # random_std[category_size] = std
                                # ecdf_storage[category_size] = ecdf_temp
                        else:
                            if category_size not in random_cutoff_supp:
                                # calculate random groups pairwise correlation
                                random_cutoff_supp[category_size], random_std_supp[category_size], ecdf_storage_supp[
                                    category_size] = get_random_corr(category_size,
                                                                     current.loc[list(genes_suppressed_all)])

                        distance = np.nanmean(np.nanstd(current.loc[genes_data], axis=0))
                        distance_abx = np.nanmean(np.nanstd(current_abx.loc[genes_data], axis=0))
                        distance_pbs = np.nanmean(np.nanstd(current_pbs.loc[genes_data], axis=0))

                        correlation = average_pairwise_spearman(current.loc[genes_data])
                        correlation_abx = average_pairwise_spearman(current_abx.loc[genes_data])
                        correlation_pbs = average_pairwise_spearman(current_pbs.loc[genes_data])

                        # variance = np.nanmean(np.nanvar(current.loc[genes_data], axis=1))
                        # variance_abx = np.nanmean(np.nanvar(current_abx.loc[genes_data], axis=1))
                        # variance_pbs = np.nanmean(np.nanvar(current_pbs.loc[genes_data], axis=1))
                        # ddof=0 ensures it matches np.nanvar default behavior
                        variance = current.loc[genes_data].var(axis=1, ddof=0).mean()
                        variance_abx = current_abx.loc[genes_data].var(axis=1, ddof=0).mean()
                        variance_pbs = current_pbs.loc[genes_data].var(axis=1, ddof=0).mean()

                        dist[anti][treatment][counter] = correlation
                        counter += 1
                        # best = mean_mwu(anti, genes_data, current, meta_data, treatment)
                        # best = geomean_t_test(anti, genes_data, current, meta, treatment, condition)
                        median_ttest = median_t_test(anti, genes_data, current, meta, treatment, condition)
                        mwu = mean_mwu(anti, genes_data, current, meta, treatment, condition)
                        fold_change = mean_fold(anti, genes_data, current, meta, treatment, condition)
                        median_zscore_diff = median_fold_change(anti, genes_data, current, meta, treatment, condition)

                        # log2fc = np.log2(fold_change)
                        # Create an array of NaNs with the same shape
                        log_fc = np.full_like(fold_change, np.nan)
                        # Only calculate log2 where data is positive (> 0), 'out' stores the result, 'where' tells it which indices to calculate
                        log2fc = np.log2(fold_change, out=log_fc, where=(fold_change > 0))

                        # if len(genes_data) > 10:
                        #     color = 'red' if distance > random_cutoff[len(genes_data)] else 'blue'
                        #     if node.parent and node.parent.dist and node.parent.dist < node.dist:
                        #         color = 'hotpink' if distance > random_cutoff[len(genes_data)] else 'cyan'
                        #     # marker = 'o' if best > mwu_cutoff[len(genes_data)] else 'x'
                        #     marker = 'o' if best < 0.05 else 'x'
                        #     plt.scatter(best, distance, s=5, color=color, marker=marker)
                        # if distance > 2.5:
                        genes_id_to_write = [id_to_name[gene] for gene in genes_data if
                                             ((gene in id_to_name) and (len(genes_data) < 3000))]
                        genes_to_write = genes_data if len(genes_data) < 300 else f"size = {len(genes_data)}"
                        # parent_dist = node.parent.dist if node.parent and node.parent.dist else np.nan
                        # parent_dist = np.nanmin(np.array([parent.dist for parent in node.parents])) if node.parents \
                        #     else np.inf
                        suf = "_enh" if enhanced else "_supp"
                        all_ancestors = list(get_ancestor(go[node])) if node in go else None
                        category_name = all_ancestors[0].name if all_ancestors else "NOT_FOUND"
                        curr_storage = ecdf_storage_enh[category_size] if enhanced else ecdf_storage_supp[category_size]
                        line = {'Antibiotics': anti, 'Condition': treatment, 'GO term': node + suf,
                                'name': f"{category_name}:{go[node].name if node in go else 'NO_NAME'}",
                                'genes': genes_to_write, 'gene names': genes_id_to_write,
                                'GO significance': GO_significance, 'correlation': correlation,
                                'correlation_pbs': correlation_pbs, 'corrlation_abx': correlation_abx,
                                'distance': distance_abx, '\"log(distance)\"': np.log2(distance),
                                'mean variance between samples': variance_abx, 'distance_all': distance,
                                'distance_pbs': distance_pbs, 'mean variance between samples all mice': variance,
                                'mean variance between samples pbs': variance_pbs, "size": len(genes_data),
                                # 'better than random': correlation > random_cutoff_enh[
                                #     category_size] if enhanced else correlation < random_cutoff_supp[category_size],
                                f'p-value correlation': calculate_pvalue_ecdf_efficient_lower_tail(correlation,
                                                                                                   curr_storage,
                                                                                                   tail='upper' if enhanced else 'lower'),
                                'random correlation': random_cutoff_enh[category_size] if enhanced else
                                random_cutoff_supp[category_size],
                                'std correlation': random_std_enh[category_size] if enhanced else random_std_supp[
                                    category_size],
                                'median t-test p-value': median_ttest, 't-test less than 5%': median_ttest < 0.05,
                                'fold change': fold_change, 'log2 fold change': log2fc,
                                'median zscore diff': median_zscore_diff,
                                "enhanced?": enhanced, "relative size": len(genes_data) / len(go_to_ensmbl_dict[node]),
                                'MWU': mwu, 'MWU less than 5%': mwu < 0.05, }
                        # removed: 'MWU': mwu, 'MWU less than 5%': mwu < 0.05, f"with {gene_to_check}?": gene_to_check in genes_data,
                        # 'better than parent': node.parent and node.parent.dist and node.parent.dist < node.dist,
                        # 'parent dist': parent_dist,
                        # 'better than random mwu': best > mwu_cutoff[len(genes_data)]}
                        # f'better than random by {sigmas_cutoff} sigma': distance < random_cutoff[
                        #     category_size] - sigmas_cutoff * random_std[category_size],
                        # f'better than random by 1 sigma': distance < random_cutoff[
                        #     category_size] - random_std[category_size],

                        # top = top.append(line, ignore_index=True)
                        # temp = temp.append(line, ignore_index=True)
                        line_df = pd.DataFrame([line])  # Convert the dictionary to a DataFrame

                        # top = pd.concat([top, line_df], ignore_index=True)
                        temp = pd.concat([temp, line_df], ignore_index=True)
                    enhanced = False
            # plt.xlabel("treat-test")
            # plt.ylabel("genes dissimilarity")
            # plt.title(f"{anti}, {treatment}, cluster > 10 genes")
            # ax = plt.gca()
            # plt.text(.3, .95, 'red: worse than random, better than parent\n'
            #                   'blue: better than random, better than parent\n'
            #                   'hotpink: worse than random, worse than parent\n'
            #                   'cyan: better than random, worse than parent',
            #          ha='left', va='top', transform=ax.transAxes)
            # plt.savefig(f"./Private/mwu-dist/{exp_type}/{anti}_{treatment}.png")
            # plt.show()
            # plt.close()
            # print(
            #     f"{anti} {treatment} for {node.go_id}, with genes {genes_data} has 'correlation' {distance}")
            # node.pearson_corr =
            # node.spearman_corr =
            # pearson[anti][treatment][i] = node.pearson_corr
            # spearman[anti][treatment][i] = node.spearman_corr
            # return pearson, spearman
            print(f"{no_genes} were not in all_data ({anti} {treatment})")
            temp["fdr GO significance"] = fdrcorrection(temp["GO significance"])[1]

            temp["fdr correlation"] = np.nan
            filtered_p_values = \
                temp[(temp["fdr GO significance"] < 0.05) & temp["correlation"].notna()][
                    "correlation"]
            # Apply FDR correction to the filtered p-values
            fdr_corrected = fdrcorrection(filtered_p_values.to_list())[1]
            # temp["fdr correlation"] = fdrcorrection(temp["p-value correlation"])[1]
            temp.loc[(temp["fdr GO significance"] < 0.05) & temp["correlation"].notna(), "fdr correlation"] = fdr_corrected
            # temp["fdr t-test"] = fdrcorrection(temp["treat-test p-value"])[1]
            # Filter the rows where p-value correlation is less than 0.05
            filtered_p_values = temp[(temp["fdr correlation"] < 0.05) & temp["median t-test p-value"].notna()]["median t-test p-value"]
            # Apply FDR correction to the filtered p-values
            fdr_corrected = fdrcorrection(filtered_p_values)[1]
            # Create a new column with NaN values
            temp["fdr median t-test"] = np.nan
            # Assign the FDR corrected values back to the DataFrame
            # temp.loc[temp["fdr correlation"] < 0.05, "fdr median t-test"] = fdr_corrected
            temp.loc[(temp["fdr correlation"] < 0.05) & temp["median t-test p-value"].notna(), "fdr median t-test"] = fdr_corrected

            # verify output directory exists
            os.makedirs(f'./Private/clusters_properties/{exp_type}/', exist_ok=True)
            temp.to_csv(f'./Private/clusters_properties/{exp_type}/top_correlated_GO_terms_{anti}_{treatment}.tsv',
                        sep='\t', index=False)
            top = pd.concat([top, temp], ignore_index=True)
            plot_curve(random_cutoff_enh, random_std_enh,
                       f'./Private/random_tightness/{exp_type}_{anti}_{treatment}_corr-vs-size_enh')
            plot_curve(random_cutoff_supp, random_std_supp,
                       f'./Private/random_tightness/{exp_type}_{anti}_{treatment}_corr-vs-size_supp')
    top.to_csv(f'./Private/clusters_properties/{exp_type}/top_correlated_GO_terms.tsv', sep='\t', index=False)
    return dist


#
#
# def calculate_correlation(root, expression, meta, size, antis, treats, gene_to_check, exp_type, condition,
#                           remove=('N18')):
#     sigmas_cutoff = 2
#     print_cluster = True
#     count_prints = 0
#     # pearson = {}
#     # spearman = {}
#     folder_dir = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
#     df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
#     id_to_name = df.set_index('gene_id')['gene_name'].to_dict()
#     dist = {}
#     # todo: maybe replace 0s with random number and average over multiple runs to avoid a correlation of zeros
#     # expression = np.log2(gene_expression.fillna(0) + 1)
#     top = pd.DataFrame()
#     for anti in antis:
#         # pearson[anti] = {}
#         # spearman[anti] = {}
#         dist[anti] = {}
#         for treatment in treats:
#             print(f"starting {anti} {treatment}")
#             temp = pd.DataFrame()
#             # pearson[anti][treatment] =  np.zeros(size)
#             # spearman[anti][treatment] = np.zeros(size)
#             # instead of couples correlation, calculate mean (log(x+1) - mean(log(x+1)))
#             dist[anti][treatment] = np.zeros(size)
#             # drug = anti.lower()
#             samples = meta[((meta['Drug'] == anti) | (meta['Drug'] == 'PBS')) & (meta[condition] == treatment)]
#             # remove missing samples
#             for sample in remove:
#                 if sample in samples['ID'].values:
#                     samples = samples.drop(samples[samples['ID'] == sample].index)
#                     print(f"{sample} removed")
#             iter_samples = samples['ID'].to_list()
#             for sample in iter_samples:
#                 if sample not in expression.columns:
#                     print(f"{sample} not in expression")
#                     samples = samples[samples['ID'] != sample]
#                     meta = meta[meta['ID'] != sample]
#                     # samples = samples.drop(samples[samples['ID'] == id].index)
#             current = expression[samples['ID']]
#             abx_samples = samples[samples['Drug'] == anti]
#             pbs_samples = samples[samples['Drug'] == 'PBS']
#             current_abx = expression[abx_samples['ID']]
#             current_pbs = expression[pbs_samples['ID']]
#             counter = 0
#             random_cutoff = {}
#             random_std = {}
#             ecdf_storage = {}
#             # mwu_cutoff = {}
#             for i, node in enumerate(PreOrderIter(root)):
#                 if len(set(node.gene_set)) == 0:
#                     continue
#                 genes_not_in_data = set(set(node.gene_set) - set(current.index))
#                 no_genes = len(genes_not_in_data)
#                 # for gene in genes_not_in_data:
#                 #     node.gene_set.remove(gene)
#                 node.gene_set = [gene for gene in node.gene_set if gene not in genes_not_in_data]
#                 if no_genes:
#                     print(f"{no_genes} genes were not in all_data ({anti} {treatment})")
#                 genes_enhanced, genes_suppressed = genes_data_split(anti, node.gene_set, current, meta, treatment,
#                                                                     condition)
#
#                 enhanced = True
#                 for genes_data in [genes_enhanced, genes_suppressed]:
#                     category_size = round(len(genes_data) / 10) * 10 if len(genes_data) > 50 else len(genes_data)
#                     if category_size == 0:
#                         continue
#                     if category_size not in random_cutoff:
#                         # print(f"adding {len(genes_data)} to random")
#                         random_cutoff[category_size], random_std[category_size], ecdf_storage[category_size] = get_random_corr(category_size, current)
#                         # random_cutoff[category_size] = rand_distance
#                         # random_std[category_size] = std
#                         # ecdf_storage[category_size] = ecdf_temp
#
#                     # mwu_range = 50
#                     # if len(genes_data) not in mwu_cutoff:
#                     #     print(f"adding {len(genes_data)} to mwu, rounded to {mwu_range}")
#                     #     if len(genes_data) > mwu_range and (len(genes_data) // mwu_range) * mwu_range in mwu_cutoff:
#                     #         print(f"{len(genes_data)} rounded to {(len(genes_data) // mwu_range) * mwu_range}")
#                     #         mwu_cutoff[len(genes_data)] = mwu_cutoff[(len(genes_data) // mwu_range) * mwu_range]
#                     #     else:
#                     #         random_mwu = get_random_mwu(len(genes_data), current, anti, meta_data, treatment)
#                     #         mwu_cutoff[len(genes_data)] = random_mwu
#                     #         mwu_cutoff[(len(genes_data) // mwu_range) * mwu_range] = random_mwu
#                     # mean = np.mean(current.loc[genes_data], axis=0)
#                     distance = np.nanmean(np.nanstd(current.loc[genes_data], axis=0))
#                     distance_abx = np.nanmean(np.nanstd(current_abx.loc[genes_data], axis=0))
#                     distance_pbs = np.nanmean(np.nanstd(current_pbs.loc[genes_data], axis=0))
#                     # if dist is not np.nan and distance > 0 and len(genes_data) > 0:
#                     if distance > 0 and len(genes_data) > 0:
#                         node.dist = distance
#                         dist[anti][treatment][counter] = node.dist
#                         counter += 1
#                         # best = mean_mwu(anti, genes_data, current, meta_data, treatment)
#                         best = geomean_t_test(anti, genes_data, current, meta, treatment, condition)
#                         mwu = mean_mwu(anti, genes_data, current, meta, treatment, condition)
#                         fold_change = mean_fold(anti, genes_data, current, meta, treatment, condition)
#
#                         # if len(genes_data) > 10:
#                         #     color = 'red' if distance > random_cutoff[len(genes_data)] else 'blue'
#                         #     if node.parent and node.parent.dist and node.parent.dist < node.dist:
#                         #         color = 'hotpink' if distance > random_cutoff[len(genes_data)] else 'cyan'
#                         #     # marker = 'o' if best > mwu_cutoff[len(genes_data)] else 'x'
#                         #     marker = 'o' if best < 0.05 else 'x'
#                         #     plt.scatter(best, distance, s=5, color=color, marker=marker)
#                         # if distance > 2.5:
#                         genes_id_to_write = [id_to_name[gene] for gene in genes_data if ((gene in id_to_name) and (len(genes_data) < 3000))]
#                         genes_to_write = genes_data if len(genes_data) < 3000 else f"size = {len(genes_data)}"
#                         # parent_dist = node.parent.dist if node.parent and node.parent.dist else np.nan
#                         parent_dist = np.nanmin(np.array([parent.dist for parent in node.parents])) if node.parents \
#                             else np.inf
#                         # todo: better than parent in 2 sigmas (if too strict relax to 1)
#                         suf = "_enh" if enhanced else "_supp"
#                         line = {'Antibiotics': anti, 'Condition': treatment, 'GO term': node.go_id + suf,
#                                 'name': f"{node.category}:{node.name}", 'genes': genes_to_write, 'gene names': genes_id_to_write,
#                                 '\"distance\"': distance, '\"log(distance)\"': np.log2(distance),
#                                 'distance_abx': distance_abx, 'distance_pbs': distance_pbs,
#                                 'MWU': mwu, "size": len(genes_data),
#                                 f"with {gene_to_check}?": gene_to_check in genes_data,
#                                 'better than random': distance < random_cutoff[category_size],
#                                 f'p-value distance': calculate_pvalue_ecdf_efficient_lower_tail(distance, ecdf_storage[category_size]),
#                                 # f'better than random by {sigmas_cutoff} sigma': distance < random_cutoff[
#                                 #     category_size] - sigmas_cutoff * random_std[category_size],
#                                 # f'better than random by 1 sigma': distance < random_cutoff[
#                                 #     category_size] - random_std[category_size],
#                                 'random distance': random_cutoff[category_size],
#                                 'std distance': random_std[category_size],
#                                 'better than parent': node.parent and node.parent.dist and node.parent.dist < node.dist,
#                                 'parent dist': parent_dist, 'treat-test p-value': best,
#                                 'treat-test less than 5%': best < 0.05, 'MWU less than 5%': mwu < 0.05,
#                                 'fold change': fold_change, 'log2 fold change': np.log2(fold_change),
#                                 "enhanced?": enhanced, "relative size": len(genes_data) / len(node.gene_set)}
#                         # 'better than random mwu': best > mwu_cutoff[len(genes_data)]}
#                         # top = top.append(line, ignore_index=True)
#                         # temp = temp.append(line, ignore_index=True)
#                         line_df = pd.DataFrame([line])  # Convert the dictionary to a DataFrame
#
#                         # top = pd.concat([top, line_df], ignore_index=True)
#                         temp = pd.concat([temp, line_df], ignore_index=True)
#                     enhanced = False
#             # plt.xlabel("treat-test")
#             # plt.ylabel("genes dissimilarity")
#             # plt.title(f"{anti}, {treatment}, cluster > 10 genes")
#             # ax = plt.gca()
#             # plt.text(.3, .95, 'red: worse than random, better than parent\n'
#             #                   'blue: better than random, better than parent\n'
#             #                   'hotpink: worse than random, worse than parent\n'
#             #                   'cyan: better than random, worse than parent',
#             #          ha='left', va='top', transform=ax.transAxes)
#             # plt.savefig(f"./Private/mwu-dist/{exp_type}/{anti}_{treatment}.png")
#             # plt.show()
#             # plt.close()
#             # print(
#             #     f"{anti} {treatment} for {node.go_id}, with genes {genes_data} has 'correlation' {distance}")
#             # node.pearson_corr =
#             # node.spearman_corr =
#             # pearson[anti][treatment][i] = node.pearson_corr
#             # spearman[anti][treatment][i] = node.spearman_corr
#             # return pearson, spearman
#             print(f"{no_genes} were not in all_data ({anti} {treatment})")
#             temp["fdr distance"] = fdrcorrection(temp["p-value distance"])[1]
#             temp["fdr t-test"] = fdrcorrection(temp["treat-test p-value"])[1]
#             temp.to_csv(f'./Private/clusters_properties/{exp_type}/top_correlated_GO_terms_{anti}_{treatment}.tsv',
#                         sep='\t', index=False)
#             top = pd.concat([top, temp], ignore_index=True)
#             plot_curve(random_cutoff, random_std,
#                        f'./Private/random_tightness/{exp_type}_{anti}_{treatment}_dist-vs-size')
#     top.to_csv(f'./Private/clusters_properties/{exp_type}/top_correlated_GO_terms.tsv', sep='\t', index=False)
#     return dist


def save_median_all_conditions(meta_data, raw_data, antibiotics, treatments, condition, exp_type):
    import os
    # meta_data = meta_data.drop(meta_data[meta_data['ID'] == 'V16'].index).drop(meta_data[meta_data['ID'] == 'V17'].
    #                                                                            index). \
    #     drop(meta_data[meta_data['ID'] == 'V18'].index).drop(meta_data[meta_data['ID'] == 'N18'].index)
    for treat in treatments:
        for abx in antibiotics:
            abx_data = meta_data[(meta_data['Drug'] == abx) & (meta_data[condition] == treat)]
            pbs_data = meta_data[(meta_data['Drug'] == 'PBS') & (meta_data[condition] == treat)]
            df = pd.read_csv(os.path.join(path, exp_type, f"top_correlated_GO_terms_{abx}_{treat}.tsv")
                             , sep="\t")
            save_all_medians(df, raw_data, abx_data, pbs_data, abx, treat, exp_type)


def save_all_medians(df, raw, abx_mice, pbs_mice, abx, treat, exp_type):
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
        # import re
        # genes = [re.sub(r'[^A-Za-z0-9]', '', gene) for gene in genes]
        # get the median of the genes in the cluster
        median = np.concatenate([np.array([row['GO term']]), np.median(raw[mice].loc[genes], axis=0)])
        line_df = pd.DataFrame([median])  # Convert the dictionary to a DataFrame
        genes_df = pd.concat([genes_df, line_df], ignore_index=True)
    # genes_df.columns = pd.Series(['index']).append(mice)
    genes_df.columns = pd.concat([pd.Series(['index']), pd.Series(mice)])
    genes_df = genes_df.set_index('index').astype('f')
    genes_df.to_csv(f"./Private/medians/only_medians/{exp_type}/{abx}_{treat}.csv")


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
        # if name == 'C10' or name == 'C9':  # todo: should be removed?
        #     continue
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
            to_impute.iloc[i, j] = np.min(to_impute.iloc[i][mice])

    row, col = np.where(to_impute.isnull())
    print(f"Now left with {len(row)} zeros, but {all_other_are_zeros} are zeros in all other mice")
    print(f"in {int(all_other_are_zeros_conditions)} conditions. {too_big} were too big")
    #     to_impute = impute_zeros(to_impute, meta_data)
    to_impute.to_csv(f'./Private/imputed_all_zeros_removed{run_type}.csv')
    return to_impute


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


def check_importance_missing_genes(missing_genes, transcriptome_df, meta):
    # create a matrix not_significant with antibiotics as index and treatments as columns
    significant = pd.DataFrame(index=antibiotics, columns=treatments, data=0)
    not_significant = pd.DataFrame(index=antibiotics, columns=treatments, data=0)
    not_significant_set = set()
    significant_set = set()
    # check significance of all those genes in transcriptome_df
    for anti in antibiotics:
        for treatment in treatments:
            print(f"starting {anti} {treatment}")
            # compare the significance of each missing gene in the transcriptome_df, for abx&treat vs PBS&treat
            abx_data = meta[(meta['Drug'] == anti) & (meta['Treatment'] == treatment)]
            pbs_data = meta[(meta['Drug'] == 'PBS') & (meta['Treatment'] == treatment)]
            for gene in missing_genes:
                if not gene in transcriptome_df.index:
                    print(f"no data for {gene}")
                    continue
                abx = transcriptome_df.loc[gene, abx_data['Sample']]
                pbs = transcriptome_df.loc[gene, pbs_data['Sample']]
                if len(abx.shape) > 1:
                    continue
                if len(pbs) == 0 or len(abx) == 0 or (np.sum(abx) == 0 and np.sum(pbs) == 0):
                    # print(f"no data for {anti} {treatment} {gene}")
                    not_significant.loc[anti, treatment] += 1
                    not_significant_set.add(gene)
                    continue
                if np.array_equal(abx.values, pbs.values):
                    print(f"abx and pbs are the same for {anti} {treatment} {gene}")
                    continue
                MWU_pbs = mannwhitneyu(pbs, abx)
                MWU_abx = mannwhitneyu(abx, pbs)
                if MWU_pbs[1] < 0.05 or MWU_abx[1] < 0.05:
                    # print(f"{gene} is significant for {anti} {treatment} with pbs {MWU_pbs} and abx {MWU_abx}")
                    significant.loc[anti, treatment] += 1
                    significant_set.add(gene)

    # plot a heatmap of the not_significant matrix and the significant matrix
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(not_significant, annot=True, ax=ax[0], xticklabels=treatments, yticklabels=antibiotics)
    ax[0].set_title("empty genes")
    sns.heatmap(significant, annot=True, ax=ax[1], xticklabels=treatments, yticklabels=antibiotics)
    ax[1].set_title("significant genes")
    plt.savefig(f"./Private/missing_genes.png")
    plt.show()
    print(f"significant genes: {significant}")
    print(f"not significant genes: {not_significant}")
    return significant_set, not_significant_set


def plot_histogram_counts(df, type_of):
    value_counts = df['transcript_biotype'].value_counts()
    percentages = (value_counts / value_counts.sum()) * 100
    filtered_percentages = percentages[percentages >= 1]

    ordered_counts = filtered_percentages.sort_index()
    sns.barplot(x=ordered_counts.index, y=ordered_counts.values, palette='viridis')
    plt.xticks(rotation=90)  # Rotate x labels if needed for better readability
    plt.xlabel('Transcript Biotype')
    plt.ylabel('Percentage (%)')
    plt.title(f'Histogram of Transcript Biotype Counts for {type_of}')
    plt.savefig(f"./Private/Genes/{type_of}_transcript_biotype.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()


def get_go_to_ensmusg():
    from biomart import BiomartServer

    # Connect to the BioMart server
    server = BiomartServer("http://www.ensembl.org/biomart")
    # server = BiomartServer("https://www.uswest.ensembl.org/biomart")
    # server = BiomartServer("https://www.useast.ensembl.org/biomart")
    # server = BiomartServer("https://www.asia.ensembl.org/biomart")

    # Choose the Ensembl database
    mart = server.datasets['mmusculus_gene_ensembl']

    # Define the attributes you want to retrieve
    attributes = [
        'ensembl_gene_id',
        'go_id'
    ]
    filters = {
        'go_parent_term': 'GO:0008150'  # This is the root term for Biological Process
    }

    # Query BioMart
    response = mart.search({
        'filters': filters,
        'attributes': attributes
    })

    # Parse the response
    go_to_ensmusg = defaultdict(set)
    for line in response.iter_lines():
        decoded_line = line.decode('utf-8')
        ensembl_gene_id, go_id = decoded_line.split("\t")
        if go_id:
            go_to_ensmusg[go_id].add(ensembl_gene_id)
    return go_to_ensmusg


# def add_genes_ids(root, go_to_ensmbl_dict):
#     empty_nodes_counter = 0
#     added = set()
#     for i, node in enumerate(PostOrderIter(root)):
#         node_genes = go_to_ensmbl_dict.get(node.go_id, set())
#         if node_genes:
#             node.gene_set = node.gene_set.union(node_genes)
#             added.add(node.go_id)
#         else:
#             empty_nodes_counter += 1
#         if i % 500 == 0:
#             print(f"### {i} nodes were updated ###")
#     print(f"{empty_nodes_counter} empty nodes")
#     print(f"Out of {len(go_to_ensmbl_dict)} mmusculus_gene_ensembl GOs, {len(added)} were added")
#     missing = set(go_to_ensmbl_dict.keys()) - added
#     print(f"Examples:")
#     folder_dir = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
#     df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
#     id_to_name = df.set_index('gene_id')['gene_name'].to_dict()
#     for i, go in enumerate(missing):
#         print(go, [id_to_name[name] for name in go_to_ensmbl_dict[go] if name in id_to_name])
#         if i == 5:
#             break
#     return root
def add_genes_ids(root: Any, go_to_ensmbl_dict: Dict[str, Set[str]],
                  progress_interval: int = 1000,
                  max_examples: int = 5,
                  gene_name_file: str = "./Data/transcriptome_2023-09-17-genes_norm_named.tsv") -> Any:
    empty_nodes_counter = 0
    added: Set[str] = set()

    for i, node in enumerate(PostOrderIter(root)):
        node_genes = go_to_ensmbl_dict.get(node.go_id, set())
        if node_genes:
            node.gene_set = node.gene_set.union(node_genes)
            added.add(node.go_id)
        else:
            empty_nodes_counter += 1

        if i % progress_interval == 0:
            print(f"### {i} nodes were updated ###")

    print(f"{empty_nodes_counter} empty nodes")
    print(f"Out of {len(go_to_ensmbl_dict)} mmusculus_gene_ensembl GOs, {len(added)} were added")

    missing = set(go_to_ensmbl_dict.keys()) - added
    print("Examples:")

    try:
        df = pd.read_csv(gene_name_file, sep="\t")
        id_to_name = df.set_index('gene_id')['gene_name'].to_dict()

        for i, go in enumerate(missing):
            if i >= max_examples:
                break
            gene_names = [id_to_name.get(name, name) for name in go_to_ensmbl_dict[go]]
            print(go, gene_names)
    except FileNotFoundError:
        print(f"Warning: Gene name file not found at {gene_name_file}")
    except KeyError as e:
        print(f"Warning: Expected column not found in gene name file: {e}")

    return root


def build_tree(download=False):
    go = obo_parser.GODag(get_go(download_anyway=download))
    # cell_cycle_genes()
    filename = "genomic_tree.json"
    # check if file "genomic_tree.jason" is in current path
    if os.path.exists(f"./Private/{filename}"):
        tree = JsonImporter().read(open(f"./Private/{filename}"))
        for node in PostOrderIter(tree):
            node.unserialize()
    else:
        tree, tree_size = build_genomic_tree(go['GO:0008150'], go)
        bio_terms = [term for term in go if go[term].namespace == 'biological_process']
        print(f"{tree_size} nodes were built out of {len(bio_terms)} biological process GO terms")
        # tree1 = build_genomic_tree(go['GO:0007582'])
        # tree2 = build_genomic_tree(go['GO:0044699'])
        # tree3 = build_genomic_tree(go['GO:0000004'])
        # genes = pd.read_csv("./Go_terms.csv", dtype=np.dtype(str))
        # genes = pd.read_csv("./Private/Genes/mgi.gaf", dtype=np.dtype(str), skiprows=36, sep="\t")
        # # check_existence_all_genes(genes)
        # add_genes_names(tree, genes)
        go_to_ensmbl_dict = get_go_to_ensmusg()
        print(f"GO genes number {len(go_to_ensmbl_dict)}")
        print(f"terms not in dictionary file: {len([term for term in go.values() if term not in go_to_ensmbl_dict])}")
        add_genes_ids(tree, go_to_ensmbl_dict)

        # # use anytree iterator to go over GO terms
        # add_genes_names(tree, go_to_gene_id())

        # # save the go_tree
        # for node in PostOrderIter(tree):
        #     node.serialize()
        # exporter = JsonExporter(indent=2, default=MyEncoder)
        # exporter.write(go_tree, open(f"./Private/{filename}", "w"))
    # load the go_tree
    # JsonImporter().read(open("./Private/genomic_tree.json"))

    return tree, tree_size


def read_process_files(new=False, filter_value=0.55, merge_big_abx=True, remove_mitochondrial=True, gene_name=False):
    partek_df = pd.read_csv(
        "../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/New Partek_bell_all_Normalization_Normalized_counts1.csv")
    partek_df = partek_df.set_index("Gene Symbol")
    folder_dir = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
    genome_df = pd.read_csv(folder_dir + "rpkm_named_genome-2023-09-26.tsv", sep="\t")
    transcriptome_df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
    id_to_name = dict(zip(transcriptome_df['gene_id'], transcriptome_df['gene_name']))

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
    partek_df = partek_df.fillna(0)  # todo: check why

    metadata = get_metadata(data_folder, type="", only_old=not new, filter=filter_value)

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
        new_metadata = get_metadata(data_folder, type="", only_old=not new, filter=False)
        new_metadata = new_metadata[new_metadata["ID"].isin(new_data.columns)]
        metadata = pd.concat([metadata, new_metadata])

    # sum rows from transcriptome and genome with the same index TODO
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
        mito_ids = [
            gid for gid, gname in id_to_name.items()
            if str(gname).lower() in mitochondrial_genes
        ]
        # todo: see statistics of the removed genes
        matching_indices = transcriptome_df.index[
            transcriptome_df.index.str.lower().isin(set(mito_ids))].tolist()
            # transcriptome_df.index.str.lower().isin(set(mitochondrial_genes))].tolist()

        # remove mitochondrial genes from the dataframes
        genome_df = genome_df.drop(matching_indices, errors='ignore')
        transcriptome_df = transcriptome_df.drop(matching_indices, errors='ignore')
        partek_df = partek_df.drop(matching_indices, errors='ignore')

    partek_df = (partek_df * 1000000).divide(partek_df.sum(axis=0), axis=1)
    genome_df = (genome_df * 1000000).divide(genome_df.sum(axis=0), axis=1)
    transcriptome_df = (transcriptome_df * 1000000).divide(transcriptome_df.sum(axis=0), axis=1)

    # NOTICE! drop C9, C10, C18, M13, V14 from all DFs and metadata
    to_remove = ["C9", "C10", "C18", "M13", "V14"]
    transcriptome_df = transcriptome_df.drop(to_remove, axis=1)
    metadata = metadata[~metadata["ID"].isin(to_remove)]

    return genome_df, metadata, partek_df, transcriptome_df


def get_metadata(folder, type="", only_old=True, filter=0.55):
    meta = pd.read_excel(os.path.join(folder, "metadata.xlsx"))
    meta['ID'] = meta.apply(lambda row: row['ID'] + 'N' if row['New/Old'] == 'N' else row['ID'], axis=1)
    meta['Drug'] = meta.apply(lambda row: row['Drug'].replace('mix', 'Mix').replace('ampicillin', 'Amp')
                              .replace('Control ', 'PBS').replace('METRO', 'Met').replace('NEO', 'Neo')
                              .replace('VANCO', 'Van'), axis=1)
    if filter:
        file = "RASflow stats 2023_09_26.csv" if type else "RASflow stats 2023_09_17.csv"
        qc = pd.read_csv(os.path.join(folder, file))
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
    treatments = metadata['Treatment'].unique()
    antibiotics = [abx for abx in metadata['Drug'].unique() if abx != "PBS"]
    for treat in treatments:
        pbs = metadata[(metadata['Drug'] == "PBS") & (metadata["Treatment"] == treat)]
        # get the pbs mice data
        pbs_data = data[pbs['ID']]
        # calculate the mean and std of the pbs mice
        pbs_mean = pbs_data.mean(axis=1)
        pbs_std = pbs_data.std(axis=1)
        data[pbs['ID']] = data[pbs['ID']].sub(pbs_mean, axis=0)
        data[pbs['ID']] = data[pbs['ID']].div(pbs_std, axis=0)
        for anti in antibiotics:
            abx = metadata[(metadata['Drug'] == anti) & (metadata["Treatment"] == treat)]
            # normalize the data by the mean and std of the pbs mice: subtract pbs_mean from every row and divide by std
            data[abx['ID']] = data[abx['ID']].sub(pbs_mean, axis=0)
            data[abx['ID']] = data[abx['ID']].div(pbs_std, axis=0)
    # return the normalized data
    return data


def transform_data(data, metadata, run_type, skip=False, save=False, skip_norm=False):
    # replace all zeros with nan
    data = data.replace(0, np.nan)
    # # Remove V11 from data, and remove row ID==V11 from metadata
    # data = data.drop('V11', axis=1)
    # metadata = metadata.drop(metadata[metadata['ID'] == 'V11'].index)
    if save:
        folder_dir = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
        df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
        id_to_name = df.set_index('gene_id')['gene_name'].to_dict()
        data['gene_original_name'] = data.index.map(id_to_name)

        data.to_csv("./Private/data process/no_v11.csv")
        metadata.to_csv("./Private/data process/metada.csv")
        # metadata.to_csv("./Private/data process/metada_no_v11.csv")
        data = data.drop('gene_original_name', axis=1)
    data = impute_zeros(data, metadata, 'Treatment', run_type, skip_if_exist=skip)
    if save:
        data['gene_original_name'] = data.index.map(id_to_name)
        data.to_csv("./Private/data process/imputed.csv")
        # data.to_csv("./Private/data process/no_v11_imputed.csv")
        data = data.drop('gene_original_name', axis=1)
    data = np.log2(data)
    if save:
        data['gene_original_name'] = data.index.map(id_to_name)
        data.to_csv("./Private/data process/imputed_log.csv")
        # data.to_csv("./Private/data process/no_v11_imputed_log.csv")
        data = data.drop('gene_original_name', axis=1)
    if not skip_norm:
        # z-score by PBS
        data = zscore_all_by_pbs(data, metadata)
        if save:
            data['gene_original_name'] = data.index.map(id_to_name)
            data.to_csv("./Private/data process/imputed_log_zscore.csv")
            # data.to_csv("./Private/data process/no_v11_imputed_log_zscore.csv")
            data = data.drop('gene_original_name', axis=1)
    return data, metadata


if __name__ == "__main__":
    import sys

    run_type = sys.argv[1]

    genome, metadata, partek, transcriptome = read_process_files(new=False)

    # save metadata and transcriptome as csv files
    metadata.to_csv("./Private/metadata.csv", index=False)
    transcriptome.to_csv("./Private/transcriptome.csv")

    data = transcriptome
    data, metadata = transform_data(data, metadata, run_type, skip=False)
    data.to_csv("./Private/transcriptome_transformed.csv")
    tree, tree_size = build_tree(True)
    # make any value smaller than log10(5) to be 0 todo
    # data[data < np.log10(1)] = 0

    folder_dir = f"../Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/"
    df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
    id_to_name = df.set_index('gene_id')['gene_name'].to_dict()

    # corr = calculate_correlation(tree, data, metadata, tree_size, antibiotics, treatments, "H2-Ab1",
    corr = calculate_correlation(data, metadata, antibiotics, treatments, "H2-Ab1",
                                 f"diff_abx{run_type}", 'Treatment', id_to_name)

    # build 2 trees and compare?
    # todo add tests for the functions

    # save_median_all_conditions(metadata, data, antibiotics, treatments, "Treatment", "diff_abx" + run_type)
