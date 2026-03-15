import sys
import os
import argparse

# Get the absolute path of the target directory
# Assuming the current file is two levels below the root
module_path = os.path.abspath(os.path.join('..', '..', 'CorrEnrich'))

# Add the directory to sys.path if it's not already there
if module_path not in sys.path:
    sys.path.append(module_path)

# Now import run_analysis
from run_analysis import run_analysis

"""
Example script to run the full ClusteringGO analysis pipeline.
"""
import os
import argparse
import pandas as pd
from anytree import PreOrderIter
from tqdm import tqdm
from goatools import obo_parser
from groups_comparison import read_data_metadata
from groups_comparison import transform_data as transform
from clusteringgo.tree import build_tree, get_go_to_ensmusg, get_go
from clusteringgo.data_processing import read_process_files, transform_data
from clusteringgo.tree import build_tree, get_go_to_ensmusg
from clusteringgo.stats import (
    average_pairwise_spearman, get_random_corr, median_mwu,
    genes_data_split, calculate_hypergeometric_pvalue, calculate_pvalue_from_ecdf
)
from clusteringgo.utils import plot_random_corr_curve, save_results, get_gene_name_map

def run_analysis(output_dir, primary_condition_col, control_value, secondary_condition_col=None, significance_threshold=0.05):
    """
    Main function to execute the gene clustering and correlation analysis.

    Args:
        output_dir (str): Path to save results and plots.
        primary_condition_col (str): The main metadata column to test (e.g., 'Drug').
        control_value (str): The value in `primary_condition_col` that is the control (e.g., 'PBS').
        secondary_condition_col (str, optional): A second column for nested analysis (e.g., 'Treatment').
        significance_threshold (float, optional): P-value cutoff for individual gene t-tests.
    """
    print("Step 1: Reading and processing data...")
    os.makedirs(output_dir, exist_ok=True)
    # transcriptome, metadata = read_process_files(data_folder)
    data, metadata, id_to_name = read_data_metadata(remove_mitochondrial=False)
    # antibiotics = metadata['Drug'].unique().tolist()
    # antibiotics.remove('PBS')
    # treatments = metadata['Treatment'].unique().tolist()
    run_type = "_pairs"


    print("Step 2: Transforming data (impute, log2, z-score)...")
    data, metadata = transform(data, metadata, run_type, skip=True)

    print("Step 3: Building Gene Ontology tree...")
    # tree, _ = build_tree(data_dir=output_dir)
    # Note: Using get_go_to_ensmusg directly as ClusteringGO.py does
    go_to_ensmbl_dict = get_go_to_ensmusg()

    print(f"Loaded {len(go_to_ensmbl_dict)} GO terms with gene mappings.")

    # This loads the official GO database to filter against.
    print("Loading GO-DAG from obo file...")
    # Use output_dir to store the downloaded go-basic.obo file
    go_dag_file = get_go(data_dir=output_dir, download_anyway=False)
    if go_dag_file is None:
        print("ERROR: Could not download or find go-basic.obo file. Exiting.")
        return
    go_dag = obo_parser.GODag(go_dag_file)
    print("GO-DAG loaded.")

    print("Step 4: Calculating correlations and significance...")
    all_results = []

    # gene_map_path = os.path.join(data_folder, "new normalization", "transcriptome_2023-09-17-genes_norm_named.tsv")
    # id_to_name = get_gene_name_map(gene_map_path)

    primary_conditions = [c for c in metadata[primary_condition_col].unique() if c != control_value]
    secondary_conditions = metadata[secondary_condition_col].unique() if secondary_condition_col else [None]

    # Convert tree/GO iterator to a list ONCE to get the total count for the progress bar
    nodes_to_process = list(go_to_ensmbl_dict.keys())
    # nodes_to_process = list(PreOrderIter(tree))
    total_nodes = len(nodes_to_process)

    for primary_val in primary_conditions:
        for secondary_val in secondary_conditions:
            if secondary_val:
                current_meta = metadata[
                    ((metadata[primary_condition_col] == primary_val) | (metadata[primary_condition_col] == control_value)) &
                    (metadata[secondary_condition_col] == secondary_val)
                ]
                condition_name_desc = f"{primary_val}-{secondary_val}"
            else:
                current_meta = metadata[
                    (metadata[primary_condition_col] == primary_val) | (metadata[primary_condition_col] == control_value)
                ]
                condition_name_desc = primary_val

            print(f"\n--- Analyzing: {condition_name_desc} ---")

            current_sample_ids = [s for s in current_meta['ID'] if s in data.columns]
            current_expression = data[current_sample_ids]
            # Get all genes present in the current expression matrix

            all_genes_in_data = current_expression.index

            if current_expression.empty or len(current_meta[primary_condition_col].unique()) < 2:
                print(f"Skipping {condition_name_desc} due to insufficient data.")
                continue

            # 1. Pre-calculate significance for ALL genes in the current dataset
            print(f"Calculating significance for all {len(all_genes_in_data)} genes...")
            enhanced_all, suppressed_all, significant_genes_all = genes_data_split(
                primary_val, control_value, all_genes_in_data, current_expression, metadata,
                primary_condition_col, secondary_condition_col, secondary_val,
                threshold=significance_threshold
            )
            print(f"Found {len(significant_genes_all)} significant genes.")

            results_list = []
            random_cutoff, random_std, ecdf_storage = {}, {}, {}

            # --- WRAP YOUR LOOP WITH TQDM ---
            # This shows a progress bar: e.g., "Amp-IP: 25%|██▌ | 5000/20000 [00:10<00:30, 499.50term/s]"
            for node in tqdm(nodes_to_process, desc=condition_name_desc, unit="term", total=total_nodes):
                if not go_to_ensmbl_dict.get(node):
                # if not node.gene_set:
                    continue
                # This check ensures we only process terms that are in the
                # official go-basic.obo file, matching ClusteringGO.py
                if node not in go_dag:
                    print(f"{node} not in go-basic.obo file. Skipping.")
                    continue

                genes_in_data = list(go_to_ensmbl_dict.get(node).intersection(all_genes_in_data))
                # genes_in_data = list(node.gene_set.intersection(current_expression.index))
                if len(genes_in_data) < 2:
                    continue

                # enhanced, suppressed = split_genes_by_trend(
                #     primary_val, control_value, genes_in_data, current_expression, metadata,
                #     primary_condition_col, secondary_condition_col, secondary_val
                # )
                # 2. Calculate GO Significance (Hypergeometric P-value)
                genes_significant_in_go = [g for g in genes_in_data if g in significant_genes_all]
                go_significance = calculate_hypergeometric_pvalue(
                    N=len(all_genes_in_data),  # Total genes in background
                    K=len(significant_genes_all),  # Total significant genes in background
                    n=len(genes_in_data),  # Genes in this GO term
                    k=len(genes_significant_in_go)  # Significant genes in this GO term
                )
                # 3. Filter gene lists to ONLY include significant genes
                # enhanced = [g for g in genes_in_data if g in enhanced_all and g in significant_genes_all]
                # suppressed = [g for g in genes_in_data if g in suppressed_all and g in significant_genes_all]
                enhanced = [g for g in genes_in_data if g in enhanced_all]
                suppressed = [g for g in genes_in_data if g in suppressed_all]

                for trend_genes, trend_label in [(enhanced, 'enhanced'), (suppressed, 'suppressed')]:
                    if len(trend_genes) < 2:
                        continue

                    correlation = average_pairwise_spearman(current_expression.loc[trend_genes])
                    if pd.isna(correlation):
                        continue

                    size_category = round(len(trend_genes) / 10) * 10 if len(trend_genes) > 50 else len(trend_genes)
                    if size_category > 1 and size_category not in random_cutoff:
                        # This expensive step is now parallelized (see stats.py)
                        # rc, rs, ecdf = get_random_corr(size_category, current_expression)
                        # random_cutoff[size_category], random_std[size_category], ecdf_storage[size_category] = rc, rs, ecdf
                        # Use all enhanced/suppressed genes for the random background, as in ClusteringGO.py
                        background_genes = enhanced_all if trend_label == 'enhanced' else suppressed_all
                        background_genes_in_data = list(set(background_genes).intersection(all_genes_in_data))
                        if len(background_genes_in_data) > size_category:
                            rc, rs, ecdf = get_random_corr(size_category,
                                                           current_expression.loc[background_genes_in_data])
                            random_cutoff[size_category], random_std[size_category], ecdf_storage[size_category] = rc, rs, ecdf

                    _, mwu_p_value = median_mwu(
                        primary_val, control_value, trend_genes, current_expression, metadata,
                        primary_condition_col, secondary_condition_col, secondary_val
                    )

                    p_val_corr = 1.0
                    if size_category in ecdf_storage:
                        tail = 'upper'
                        # tail = 'upper' if trend_label == 'enhanced' else 'lower'
                        p_val_corr = calculate_pvalue_from_ecdf(correlation, ecdf_storage[size_category], tail=tail)

                    gene_names = [id_to_name.get(g, g) for g in trend_genes]

                    results_list.append({
                        primary_condition_col: primary_val,
                        secondary_condition_col if secondary_condition_col else 'Group': secondary_val if secondary_val else 'All',
                        'GO_Term': node,
                        # 'GO_Term': node.go_id,
                        # 'GO_Name': node.name,
                        'GO_Significance': go_significance,
                        'Trend': trend_label,
                        'N_Genes': len(trend_genes),
                        'Correlation': correlation,
                        'Correlation_PValue': p_val_corr,
                        'Random_Corr_Mean': random_cutoff.get(size_category),
                        'MWU_PValue': mwu_p_value,
                        'Genes': ','.join(trend_genes),
                        'Gene_Names': ','.join(gene_names)
                    })

            if results_list:
                results_df = pd.DataFrame(results_list)
                condition_name = f"{primary_val}_{secondary_val}" if secondary_val else primary_val
                condition_results_path = os.path.join(output_dir, f'results_{condition_name}.tsv')
                save_results(results_df, condition_results_path)
                all_results.append(results_df)

                plot_path = os.path.join(output_dir, f'random_corr_{condition_name}.png')
                plot_random_corr_curve(random_cutoff, random_std, plot_path)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_path = os.path.join(output_dir, 'all_go_term_results.tsv')
        save_results(final_df, final_path)
        print(f"\nAnalysis complete. All results saved to {final_path}")
    else:
        print("\nAnalysis complete. No results were generated.")


def run_analysis_pairs(output_dir, primary_condition_col, control_value, secondary_condition_col=None, significance_threshold=0.05):
    """
    Main function to execute the gene clustering and correlation analysis.

    Args:
        output_dir (str): Path to save results and plots.
        primary_condition_col (str): The main metadata column to test (e.g., 'Drug').
        control_value (str): The value in `primary_condition_col` that is the control (e.g., 'PBS').
        secondary_condition_col (str, optional): A second column for nested analysis (e.g., 'Treatment').
        significance_threshold (float, optional): P-value cutoff for individual gene t-tests.
    """
    print("Step 1: Reading and processing data...")
    os.makedirs(output_dir, exist_ok=True)
    # transcriptome, metadata = read_process_files(data_folder)
    data, metadata, id_to_name = read_data_metadata(remove_mitochondrial=False)
    # antibiotics = metadata['Drug'].unique().tolist()
    # antibiotics.remove('PBS')
    # treatments = metadata['Treatment'].unique().tolist()
    run_type = "_pairs"


    print("Step 2: Transforming data (impute, log2, z-score)...")
    data, metadata = transform(data, metadata, run_type, skip=True)

    print("Step 3: Building Gene Ontology tree...")
    # tree, _ = build_tree(data_dir=output_dir)
    # Note: Using get_go_to_ensmusg directly as ClusteringGO.py does
    go_to_ensmbl_dict = get_go_to_ensmusg()

    print(f"Loaded {len(go_to_ensmbl_dict)} GO terms with gene mappings.")

    # This loads the official GO database to filter against.
    print("Loading GO-DAG from obo file...")
    # Use output_dir to store the downloaded go-basic.obo file
    go_dag_file = get_go(data_dir=output_dir, download_anyway=False)
    if go_dag_file is None:
        print("ERROR: Could not download or find go-basic.obo file. Exiting.")
        return
    go_dag = obo_parser.GODag(go_dag_file)
    print("GO-DAG loaded.")

    print("Step 4: Calculating correlations and significance...")

    # gene_map_path = os.path.join(data_folder, "new normalization", "transcriptome_2023-09-17-genes_norm_named.tsv")
    # id_to_name = get_gene_name_map(gene_map_path)

    primary_conditions = [c for c in metadata[primary_condition_col].unique() if c != control_value]
    secondary_conditions = metadata[secondary_condition_col].unique() if secondary_condition_col else [None]

    # Convert tree/GO iterator to a list ONCE to get the total count for the progress bar
    nodes_to_process = list(go_to_ensmbl_dict.keys())
    # nodes_to_process = list(PreOrderIter(tree))
    total_nodes = len(nodes_to_process)
    output_dir_orig = output_dir
    primary_conditions_orig = primary_conditions

    # for control_value in ["Met"]:
    for control_value in ["Van", "Neo", "Met"]:
        all_results = []
        output_dir = output_dir_orig + f"_{control_value}"
        primary_conditions = [c for c in primary_conditions_orig if ("+" in c and control_value in c)]
        print(control_value, primary_conditions)
        for primary_val in primary_conditions:
            for secondary_val in secondary_conditions:
                if secondary_val:
                    current_meta = metadata[
                        ((metadata[primary_condition_col] == primary_val) | (metadata[primary_condition_col] == control_value)) &
                        (metadata[secondary_condition_col] == secondary_val)
                    ]
                    condition_name_desc = f"{primary_val}-{secondary_val}"
                else:
                    current_meta = metadata[
                        (metadata[primary_condition_col] == primary_val) | (metadata[primary_condition_col] == control_value)
                    ]
                    condition_name_desc = primary_val

                print(f"\n--- Analyzing: {condition_name_desc} ---")

                current_sample_ids = [s for s in current_meta['ID'] if s in data.columns]
                current_expression = data[current_sample_ids]
                # Get all genes present in the current expression matrix

                all_genes_in_data = current_expression.index

                if current_expression.empty or len(current_meta[primary_condition_col].unique()) < 2:
                    print(f"Skipping {condition_name_desc} due to insufficient data.")
                    continue

                # 1. Pre-calculate significance for ALL genes in the current dataset
                print(f"Calculating significance for all {len(all_genes_in_data)} genes...")
                enhanced_all, suppressed_all, significant_genes_all = genes_data_split(
                    primary_val, control_value, all_genes_in_data, current_expression, metadata,
                    primary_condition_col, secondary_condition_col, secondary_val,
                    threshold=significance_threshold
                )
                print(f"Found {len(significant_genes_all)} significant genes.")

                results_list = []
                random_cutoff, random_std, ecdf_storage = {}, {}, {}

                # --- WRAP YOUR LOOP WITH TQDM ---
                # This shows a progress bar: e.g., "Amp-IP: 25%|██▌ | 5000/20000 [00:10<00:30, 499.50term/s]"
                for node in tqdm(nodes_to_process, desc=condition_name_desc, unit="term", total=total_nodes):
                    if not go_to_ensmbl_dict.get(node):
                    # if not node.gene_set:
                        continue
                    # This check ensures we only process terms that are in the
                    # official go-basic.obo file, matching ClusteringGO.py
                    if node not in go_dag:
                        print(f"{node} not in go-basic.obo file. Skipping.")
                        continue

                    genes_in_data = list(go_to_ensmbl_dict.get(node).intersection(all_genes_in_data))
                    # genes_in_data = list(node.gene_set.intersection(current_expression.index))
                    if len(genes_in_data) < 2:
                        continue

                    # enhanced, suppressed = split_genes_by_trend(
                    #     primary_val, control_value, genes_in_data, current_expression, metadata,
                    #     primary_condition_col, secondary_condition_col, secondary_val
                    # )
                    # 2. Calculate GO Significance (Hypergeometric P-value)
                    genes_significant_in_go = [g for g in genes_in_data if g in significant_genes_all]
                    go_significance = calculate_hypergeometric_pvalue(
                        N=len(all_genes_in_data),  # Total genes in background
                        K=len(significant_genes_all),  # Total significant genes in background
                        n=len(genes_in_data),  # Genes in this GO term
                        k=len(genes_significant_in_go)  # Significant genes in this GO term
                    )
                    # 3. Filter gene lists to ONLY include significant genes
                    # enhanced = [g for g in genes_in_data if g in enhanced_all and g in significant_genes_all]
                    # suppressed = [g for g in genes_in_data if g in suppressed_all and g in significant_genes_all]
                    enhanced = [g for g in genes_in_data if g in enhanced_all]
                    suppressed = [g for g in genes_in_data if g in suppressed_all]

                    for trend_genes, trend_label in [(enhanced, 'enhanced'), (suppressed, 'suppressed')]:
                        if len(trend_genes) < 2:
                            continue

                        correlation = average_pairwise_spearman(current_expression.loc[trend_genes])
                        if pd.isna(correlation):
                            continue

                        size_category = round(len(trend_genes) / 10) * 10 if len(trend_genes) > 50 else len(trend_genes)
                        if size_category > 1 and size_category not in random_cutoff:
                            # This expensive step is now parallelized (see stats.py)
                            # rc, rs, ecdf = get_random_corr(size_category, current_expression)
                            # random_cutoff[size_category], random_std[size_category], ecdf_storage[size_category] = rc, rs, ecdf
                            # Use all enhanced/suppressed genes for the random background, as in ClusteringGO.py
                            background_genes = enhanced_all if trend_label == 'enhanced' else suppressed_all
                            background_genes_in_data = list(set(background_genes).intersection(all_genes_in_data))
                            if len(background_genes_in_data) > size_category:
                                rc, rs, ecdf = get_random_corr(size_category,
                                                               current_expression.loc[background_genes_in_data])
                                random_cutoff[size_category], random_std[size_category], ecdf_storage[size_category] = rc, rs, ecdf

                        _, mwu_p_value = median_mwu(
                            primary_val, control_value, trend_genes, current_expression, metadata,
                            primary_condition_col, secondary_condition_col, secondary_val
                        )

                        p_val_corr = 1.0
                        if size_category in ecdf_storage:
                            tail = 'upper'
                            # tail = 'upper' if trend_label == 'enhanced' else 'lower'
                            p_val_corr = calculate_pvalue_from_ecdf(correlation, ecdf_storage[size_category], tail=tail)

                        gene_names = [id_to_name.get(g, g) for g in trend_genes]

                        results_list.append({
                            primary_condition_col: primary_val,
                            secondary_condition_col if secondary_condition_col else 'Group': secondary_val if secondary_val else 'All',
                            'GO_Term': node,
                            # 'GO_Term': node.go_id,
                            # 'GO_Name': node.name,
                            'GO_Significance': go_significance,
                            'Trend': trend_label,
                            'N_Genes': len(trend_genes),
                            'Correlation': correlation,
                            'Correlation_PValue': p_val_corr,
                            'Random_Corr_Mean': random_cutoff.get(size_category),
                            'MWU_PValue': mwu_p_value,
                            'Genes': ','.join(trend_genes),
                            'Gene_Names': ','.join(gene_names)
                        })

                if results_list:
                    results_df = pd.DataFrame(results_list)
                    condition_name = f"{primary_val}_{secondary_val}" if secondary_val else primary_val
                    condition_results_path = os.path.join(output_dir, f'results_{condition_name}.tsv')
                    save_results(results_df, condition_results_path)
                    all_results.append(results_df)

                    plot_path = os.path.join(output_dir, f'random_corr_{condition_name}.png')
                    plot_random_corr_curve(random_cutoff, random_std, plot_path)

        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            final_path = os.path.join(output_dir, 'all_go_term_results.tsv')
            save_results(final_df, final_path)
            print(f"\nAnalysis complete. All results saved to {final_path}")
        else:
            print("\nAnalysis complete. No results were generated.")


def main():
    parser = argparse.ArgumentParser(description="Run the ClusteringGO analysis pipeline.")
    parser.add_argument("output_dir", help="Path to save results and plots.")
    parser.add_argument("--primary_col", required=True, help="The main metadata column to test (e.g., 'Drug').")
    parser.add_argument("--control_val", required=True, help="The control value in the primary column (e.g., 'PBS').")
    parser.add_argument("--secondary_col", default=None, help="Optional second column for nested analysis (e.g., 'Treatment').")
    parser.add_argument("--significance_threshold", type=float, default=0.05,
                        help="P-value cutoff for individual gene t-tests (default: 0.05).")
    args = parser.parse_args()

    run_analysis_pairs(args.output_dir, args.primary_col, args.control_val, args.secondary_col, args.significance_threshold)
    # run_analysis(args.output_dir, args.primary_col, args.control_val, args.secondary_col, args.significance_threshold)


def plot_res():
    from groups_comparison import plot_categories
    from statsmodels.stats.multitest import fdrcorrection
    import numpy as np
    all_antis = {
        "Van": ["Met+Van"],
        "Neo": ["Met+Neo"],
        "Met": ["Met+Van", "Met+Neo"]
    }
    treatments = ["PO"]

    # convert results from /Private/PairsCorrEnrichResults to /Private/clusters_properties/diff_abx" + run_type
    # convert = True
    convert = False
    for run_type in ["Van", "Neo", "Met"]:
        antibiotics = all_antis[run_type]
        if convert:
            dir_path = "./Private/PairsCorrEnrichResults_" + run_type
            # Non-recursive (efficient)
            with os.scandir(dir_path) as it:
                for file in it:
                    if file.is_file() and file.name.endswith(".tsv"):
                        temp = pd.read_csv(os.path.join(dir_path, file.name), sep="\t")
                        temp["fdr GO significance"] = fdrcorrection(temp["GO_Significance"])[1]

                        temp["fdr correlation"] = np.nan
                        filtered_p_values = \
                            temp[(temp["fdr GO significance"] < 0.05) & temp["Correlation_PValue"].notna()][
                                "Correlation_PValue"]
                        # Apply FDR correction to the filtered p-values
                        fdr_corrected = fdrcorrection(filtered_p_values.to_list())[1]
                        # temp["fdr correlation"] = fdrcorrection(temp["p-value correlation"])[1]
                        temp.loc[(temp["fdr GO significance"] < 0.05) & temp[
                            "Correlation_PValue"].notna(), "fdr correlation"] = fdr_corrected
                        # temp["fdr t-test"] = fdrcorrection(temp["treat-test p-value"])[1]
                        # Filter the rows where p-value correlation is less than 0.05
                        filtered_p_values = temp[(temp["fdr correlation"] < 0.05) & temp["MWU_PValue"].notna()][
                            "MWU_PValue"]
                        # Apply FDR correction to the filtered p-values
                        fdr_corrected = fdrcorrection(filtered_p_values)[1]
                        # Create a new column with NaN values
                        temp["fdr median t-test"] = np.nan
                        # Assign the FDR corrected values back to the DataFrame
                        # temp.loc[temp["fdr correlation"] < 0.05, "fdr median t-test"] = fdr_corrected
                        temp.loc[(temp["fdr correlation"] < 0.05) & temp["MWU_PValue"].notna(), "fdr MWU"] = fdr_corrected
                        temp["enhanced?"] = temp.apply(lambda row: True if row["Trend"] == "enhanced" else False, axis=1)
                        # rename column GO_term to GO term
                        temp = temp.rename(columns={"GO_Term": "GO term", "N_Genes": "size"})
                        # sort by fdr correlation (smaller first)
                        temp = temp.sort_values(by="fdr correlation", ascending=True)
                        new_name = os.path.join("./Private/clusters_properties", "diff_abx" + run_type,
                                                file.name.replace("results", "top_correlated_GO_terms"))
                        if "all_go_term" in new_name:
                            new_name = new_name.replace("all_go_term_", "")
                        # verify that the directory exists, if not create it
                        dir_name = os.path.dirname(new_name)
                        if not os.path.exists(dir_name):
                            os.makedirs(dir_name)
                        temp.to_csv(new_name, sep="\t", index=False)

        # our = plot_categories(antibiotics, treatments, "/diff_abx" + run_type, False, regular=False, mix=False)
        # gsea_abx = [abx.replace("+", "_") for abx in antibiotics]
    gsea_abx = ["MetNeo_Neo", "MetVan_Van", "MetNeo_Met", "MetVan_Met"]
    for abx in gsea_abx:
        gsea = plot_categories(gsea_abx, [''], "/diff_abx" + abx + "GSEA", False, regular=False,
                               gsea=True, mix=False)
        # plot_correlation_gsea(gsea, our)


if __name__ == "__main__":
    # main()
    plot_res()

