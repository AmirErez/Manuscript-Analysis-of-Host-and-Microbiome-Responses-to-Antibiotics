import os
import argparse
import pandas as pd
import numpy as np
import glob
import requests
import io
import time
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# Constants moved to run_analysis script or derived dynamically
MITOCHONDRIAL_GENES = [
    "mt-nd1", "mt-nd2", "mt-nd3", "mt-nd4", "mt-nd4l", "mt-nd5", "mt-nd6",
    "mt-co1", "mt-co2", "mt-co3", "mt-cytb", "mt-atp6", "mt-atp8", "mt-tf",
    "mt-tv", "mt-tl1", "mt-ti", "mt-tq", "mt-tm", "mt-tw", "mt-ta", "mt-tn",
    "mt-tc", "mt-ty", "mt-ts1", "mt-td", "mt-tk", "mt-tg", "mt-tr", "mt-th",
    "mt-ts2", "mt-tl2", "mt-te", "mt-tt", "mt-tp", "mt-rnr1", "mt-rnr2"
]

# --- Define the cutoffs ---
# FDR_CUTOFF = 0.1
FDR_CUTOFF = 0.05
LOG2FC_CUTOFF = 1.0
# LOG2FC_CUTOFF = np.log2(1.5)

# Define the Enrichr API endpoints
ENRICHR_URL_ADDLIST = 'https://maayanlab.cloud/Enrichr/addList'
ENRICHR_URL_EXPORT = 'https://maayanlab.cloud/Enrichr/export'

# Define the specific library you want to query
# You can find the exact name in the Enrichr URL (e.g., library=MGI_...)
LIBRARY_NAME = 'MGI_Mammalian_Phenotype_Level_4_2021'


def run_enrichment_on_file(gene_list_path, output_dir):
    """
    Submits a gene list from a file to Enrichr and saves the
    results for the specified MGI library.
    """
    base_name = os.path.basename(gene_list_path)
    print(f"--- Processing: {base_name} ---")

    # --- 1. Read the gene list from the .txt file ---
    try:
        with open(gene_list_path, 'r') as f:
            # Read genes, strip whitespace, and filter out empty lines
            genes = [line.strip() for line in f if line.strip()]

        if not genes:
            print(f"Skipping {base_name}: No genes found in file.")
            return

        # Format the gene list as a single string, one gene per line
        gene_list_str = "\n".join(genes)
        print(f"Submitting {len(genes)} genes...")

    except Exception as e:
        print(f"Error reading {gene_list_path}: {e}")
        return

    # --- 2. Submit the gene list to Enrichr (POST) ---
    payload = {
        'list': (None, gene_list_str),
        'description': (None, base_name)
    }

    try:
        response_add = requests.post(ENRICHR_URL_ADDLIST, files=payload)

        if not response_add.ok:
            print(f"Error submitting gene list: {response_add.text}")
            return

        # Get the 'userListId' from the response
        result_json = response_add.json()
        user_list_id = result_json.get('userListId')

        if not user_list_id:
            print("Error: 'userListId' not found in Enrichr response.")
            return

    except requests.exceptions.RequestException as e:
        print(f"Network error on submit: {e}")
        return

    # --- 3. Get the results from Enrichr (GET) ---
    print(f"Fetching results for {LIBRARY_NAME}...")

    query_params = {
        'userListId': user_list_id,
        'filename': 'enrichr_results',
        'backgroundType': LIBRARY_NAME
    }

    try:
        response_export = requests.get(ENRICHR_URL_EXPORT, params=query_params)

        if not response_export.ok:
            print(f"Error fetching results: {response_export.text}")
            return

    except requests.exceptions.RequestException as e:
        print(f"Network error on export: {e}")
        return

    # --- 4. Save the results to a CSV file ---

    # Use pandas to easily parse the tab-separated text response
    try:
        # The response text is raw, tab-separated data.
        # We read it into a pandas DataFrame.
        results_text = response_export.text
        results_df = pd.read_csv(io.StringIO(results_text), sep='\t')

        # Define a clean output filename
        output_filename = os.path.join(
            output_dir,
            f"Enrichr_MGI_{base_name.replace('.txt', '.csv')}"
        )

        # Save as a CSV
        results_df.to_csv(output_filename, index=False)
        print(f"Successfully saved results to {output_filename}")

    except pd.errors.EmptyDataError:
        print("No enrichment results returned (empty response).")
    except Exception as e:
        print(f"Error saving results: {e}")


def filter_de_files(input_dir, output_dir):
    """
    Finds all DE_results_*.csv files in the input_dir, filters them
    for significance, and saves gene lists to the output_dir.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all DE result files
    search_pattern = os.path.join(input_dir, "DE_results_*.csv")
    de_files = glob.glob(search_pattern)

    if not de_files:
        print(f"Warning: No files found matching '{search_pattern}'")
        return

    print(f"Found {len(de_files)} files to process...")

    for f_path in de_files:
        try:
            # Extract the condition name from the filename
            # e.g., "DE_results_Mix-High.csv" -> "Mix-High"
            base_name = os.path.basename(f_path)
            condition_name = base_name.replace("DE_results_", "").replace(".csv", "")

            print(f"--- Processing: {condition_name} ---")

            # Read the CSV, assuming the first column is the gene index
            df = pd.read_csv(f_path, index_col=0)

            # 1. Apply the standard cutoffs
            df_filtered = df[
                (df['fdr_bh'] < FDR_CUTOFF) &
                (df['log2fc'].abs() > LOG2FC_CUTOFF)
                ]

            if df_filtered.empty:
                print(f"No significant genes found for {condition_name} with current cutoffs.")
                continue

            # 2. Separate into enhanced (up-regulated) and suppressed (down-regulated)
            # Use the index (which contains the gene names)
            enhanced_genes = df_filtered[df_filtered['log2fc'] > 0].index.tolist()
            suppressed_genes = df_filtered[df_filtered['log2fc'] < 0].index.tolist()
            significant_genes = df_filtered.index.tolist()

            # 3. Save the lists to .txt files
            if significant_genes:
                filename = os.path.join(output_dir, f"{condition_name}_genes.txt")
                with open(filename, 'w') as f:
                    for gene in significant_genes:
                        f.write(f"{gene}\n")
                print(f"Saved {len(significant_genes)} genes to {filename}")

            # if enhanced_genes:
            #     enh_filename = os.path.join(output_dir, f"{condition_name}_enhanced_genes.txt")
            #     with open(enh_filename, 'w') as f:
            #         for gene in enhanced_genes:
            #             f.write(f"{gene}\n")
            #     print(f"Saved {len(enhanced_genes)} enhanced genes to {enh_filename}")
            #
            # if suppressed_genes:
            #     sup_filename = os.path.join(output_dir, f"{condition_name}_suppressed_genes.txt")
            #     with open(sup_filename, 'w') as f:
            #         for gene in suppressed_genes:
            #             f.write(f"{gene}\n")
            #     print(f"Saved {len(suppressed_genes)} suppressed genes to {sup_filename}")

        except Exception as e:
            print(f"Error processing file {f_path}: {e}")

    print("\nProcessing complete.")

def genes_data_split(primary_value, control_value, genes_data, expression, meta,
                     primary_col, secondary_col=None, secondary_val=None):
    """
    Splits genes into enhanced or suppressed based on expression trend AND
    returns a dictionary of genes that are individually significant.
    (Vectorized for efficiency)
    Returns a full DataFrame with t-stat, p-val, and log2fc for all genes.
    """

    # 1. Identify control (pbs) and primary (abx) sample groups ONCE
    # Find sample IDs from the metadata
    pbs_mask = (meta[primary_col] == control_value)
    abx_mask = (meta[primary_col] == primary_value)

    if secondary_col and secondary_val is not None:
        secondary_mask = (meta[secondary_col] == secondary_val)
        pbs_mask &= secondary_mask
        abx_mask &= secondary_mask

    # We use .loc[mask, 'ID'] which is equivalent to meta.query(...)[ID]
    pbs_samples = meta.loc[pbs_mask, 'ID']
    abx_samples = meta.loc[abx_mask, 'ID']

    # 2. Filter expression data ONCE

    # Find samples that exist in *both* the metadata and expression columns
    # We must convert expression.columns to a pd.Index for .intersection()
    valid_pbs_samples = pbs_samples[pbs_samples.isin(expression.columns)]
    valid_abx_samples = abx_samples[abx_samples.isin(expression.columns)]

    # Find genes that exist in *both* genes_data and the expression index
    expression_genes_index = pd.Index(expression.index)
    common_genes = expression_genes_index.intersection(genes_data)

    # Check for empty groups after filtering
    if valid_pbs_samples.empty or valid_abx_samples.empty or common_genes.empty:
        if valid_pbs_samples.empty:
            print("Warning: No valid control (pbs) samples found.")
        if valid_abx_samples.empty:
            print("Warning: No valid primary (abx) samples found.")
        if common_genes.empty:
            print("Warning: No common genes found.")
        return pd.DataFrame()

    # Create the two final data matrices for comparison
    pbs_data = expression.loc[common_genes, valid_pbs_samples]
    abx_data = expression.loc[common_genes, valid_abx_samples]

    # 3. Calculate Log2 Fold Change (Data is already log2 transformed)
    mean_pbs = pbs_data.mean(axis=1)
    mean_abx = abx_data.mean(axis=1)
    log2fc = mean_abx - mean_pbs

    # 4. Perform t-test for all genes at once
    # axis=1 compares data along the rows (i.e., compares sample groups for each gene)
    # nan_policy='omit' handles NaNs within sample groups
    # equal_var=False performs Welch's T-test, which is generally safer
    try:
        t_stat, p_val = ttest_ind(abx_data, pbs_data, axis=1, nan_policy='omit', equal_var=True)
    except ValueError as e:
        print(f"T-test failed (e.g., all-NaN slice): {e}. Returning empty results.")
        return pd.DataFrame()

    # 5. Combine results into a DataFrame
    results_df = pd.DataFrame({
        't_stat': t_stat,
        'p_val': p_val
    }, index=common_genes)

    # Drop genes where t-test failed (e.g., no variance in both groups)
    results_df = results_df.dropna()

    if results_df.empty:
        return pd.DataFrame()

    # Add log2fc, aligning with the genes that passed the t-test
    results_df['log2fc'] = log2fc.loc[results_df.index]
    # Reorder columns for clarity
    results_df = results_df[['log2fc', 't_stat', 'p_val']]
    # 6. Return the full results DataFrame
    return results_df


def impute_zeros(genes_df, meta_data, condition, missing_threshold=0.2):
    """
    Efficiently imputes missing values in a gene expression DataFrame using a vectorized approach.

    The process is as follows:
    1. Replaces all 0s with NaN.
    2. (Preprocessing) Removes genes (rows) with a high percentage of missing values.
    3. Imputes NaNs with the minimum value of their corresponding sample group.
       A group is defined by the unique combination of 'Drug' and the specified 'condition' column.
    4. Handles any remaining NaNs by first using the gene's row-minimum, and finally a small constant.

    Args:
        genes_df (pd.DataFrame): DataFrame of gene expressions (genes x samples).
        meta_data (pd.DataFrame): Metadata where rows correspond to samples. Must contain
                                  'ID', 'Drug', and the column specified by `condition`.
        condition (str): The metadata column to group by along with 'Drug' (e.g., 'Treatment').
        missing_threshold (float): The proportion of missing values (e.g., 0.2 for 20%)
                                   above which a gene will be removed.

    Returns:
        pd.DataFrame: A new DataFrame with missing values imputed, without the filtered-out genes.
    """
    # 1. Prepare the DataFrame for imputation
    imputed_df = genes_df.copy()
    imputed_df.replace(0, np.nan, inplace=True)

    # 2. Pre-processing: Remove rows with too many missing values
    n_samples = imputed_df.shape[1]
    # Keep only the rows where the number of nulls is below the threshold
    imputed_df = imputed_df.loc[imputed_df.isnull().sum(axis=1) < n_samples * missing_threshold]

    # --- Vectorized Imputation ---
    # Transpose the DataFrame so we can group the samples (columns) efficiently
    imputed_T = imputed_df.T  # Now it's samples x genes

    # Create a grouper series from the metadata that aligns with the transposed DataFrame's index
    # This creates a unique group label (e.g., ('DrugA', 'TreatmentX')) for each sample ID
    meta_data_indexed = meta_data.set_index('ID')
    grouper = meta_data_indexed.loc[imputed_T.index].apply(
        lambda x: (x['Drug'], x[condition]), axis=1
    )

    # 3. Use groupby().transform() to calculate group minimums
    # .transform('min') calculates the min for each group and broadcasts the result
    # back to the original shape of imputed_T. This is the key to vectorization. 🚀
    group_mins = imputed_T.groupby(grouper).transform('min')

    # Fill the NaNs in our data with the calculated group minimums
    imputed_T.fillna(group_mins, inplace=True)

    # Transpose back to the original orientation (genes x samples)
    imputed_df = imputed_T.T

    # 4. Handle any remaining NaNs (for cases where a whole group was NaN for a gene)
    # First fallback: fill with the row (gene) minimum
    imputed_df = imputed_df.apply(lambda row: row.fillna(row.min()), axis=1)

    # Second fallback: fill any remaining NaNs (from all-NaN rows) with a tiny value
    imputed_df.fillna(1e-6, inplace=True)

    return imputed_df


def zscore_all_by_pbs(data, metadata):
    """
    Calculates z-score for each gene based on the PBS control group for each treatment.
    Args:
        data (pd.DataFrame): The gene expression data.
        metadata (pd.DataFrame): The sample metadata.
    Returns:
        pd.DataFrame: Z-scored data.
    """
    zscored_data = data.copy()
    for treat in metadata['Treatment'].unique():
        pbs_samples = metadata[(metadata['Drug'] == "PBS") & (metadata["Treatment"] == treat)]['ID']

        # Ensure we only use samples present in the data columns
        pbs_samples_in_data = [s for s in pbs_samples if s in data.columns]
        if not pbs_samples_in_data:
            continue

        pbs_data = data[pbs_samples_in_data]
        pbs_mean = pbs_data.mean(axis=1)
        pbs_std = pbs_data.std(axis=1)
        # Avoid division by zero for groups with no variance
        pbs_std[pbs_std == 0] = 1
        pbs_std[np.isnan(pbs_std)] = 1

        treatment_samples = metadata[metadata["Treatment"] == treat]['ID']
        treatment_samples_in_data = [s for s in treatment_samples if s in data.columns]

        for sample in treatment_samples_in_data:
            zscored_data[sample] = (data[sample] - pbs_mean) / pbs_std

    return zscored_data.fillna(0)  # Fill any NaNs that may result from missing groups


def transform_data(data, metadata):
    """
    Applies a full transformation pipeline: impute zeros, log2 transform, and z-score.
    Args:
        data (pd.DataFrame): Raw (but normalized) data.
        metadata (pd.DataFrame): Sample metadata.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Transformed data and metadata.
    """
    data_imputed = impute_zeros(data, metadata, 'Treatment')
    data_log = np.log2(data_imputed)
    data_zscored = zscore_all_by_pbs(data_log, metadata)
    return data_zscored, metadata


def get_metadata(folder, qc_file_suffix="", only_old=True, filter_threshold=0.55):
    """
    Reads and filters metadata based on QC stats.
    Args:
        folder (str): Path to the data folder containing metadata.xlsx.
        qc_file_suffix (str): Suffix for the QC stats file.
        only_old (bool): If True, only include 'Old' samples.
        filter_threshold (float): Minimum 'aligned' value to keep a sample.
    Returns:
        pd.DataFrame: Filtered metadata.
    """
    meta_path = os.path.join(folder, "metadata.xlsx")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")
    meta = pd.read_excel(meta_path)
    meta['ID'] = meta.apply(lambda row: row['ID'] + 'N' if row['New/Old'] == 'N' else row['ID'], axis=1)
    meta['Drug'] = meta['Drug'].replace({
        'mix': 'Mix', 'ampicillin': 'Amp', 'Control ': 'PBS',
        'METRO': 'Met', 'NEO': 'Neo', 'VANCO': 'Van'
    })

    if filter_threshold:
        qc_file = f"RASflow stats {qc_file_suffix}.csv"
        qc_path = os.path.join(folder, qc_file)
        if not os.path.exists(qc_path):
            raise FileNotFoundError(f"QC file not found at {qc_path}")
        qc = pd.read_csv(qc_path)
        samples = qc[qc['aligned'] > filter_threshold]['Sample Name']
        meta = meta[meta['Sample'].isin(samples)]

    if only_old:
        meta = meta[~meta['ID'].str.endswith('N')]
    return meta


def read_process_files(data_folder, new=False, filter_value=0.55, remove_mitochondrial=True, use_gene_name=True):
    """
    Reads and processes the raw gene expression files.
    Args:
        data_folder (str): The root folder containing the data files.
        new (bool): Whether to include new data.
        filter_value (float): QC filter threshold.
        remove_mitochondrial (bool): If True, removes mitochondrial genes.
        use_gene_name (bool): If True, use gene names as index, otherwise gene IDs.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed transcriptome data and metadata.
    """
    norm_folder = os.path.join(data_folder, "new normalization")
    transcriptome_path = os.path.join(norm_folder, "transcriptome_2023-09-17-genes_norm_named.tsv")

    if not os.path.exists(transcriptome_path):
        raise FileNotFoundError(f"Transcriptome file not found at: {transcriptome_path}")

    transcriptome_df = pd.read_csv(transcriptome_path, sep="\t")

    index_col = "gene_name" if use_gene_name else "gene_id"
    drop_col = "gene_id" if use_gene_name else "gene_name"

    transcriptome_df[index_col] = transcriptome_df.apply(
        lambda row: row["gene_id"] if pd.isna(row[index_col]) else row[index_col], axis=1)
    transcriptome_df = transcriptome_df.set_index(index_col).drop(drop_col, axis=1)

    qc_suffix = "2023_09_17"
    metadata = get_metadata(norm_folder, qc_file_suffix=qc_suffix, only_old=not new, filter_threshold=filter_value)

    id_map = metadata.set_index('Sample')['ID'].to_dict()
    transcriptome_df = transcriptome_df.rename(columns=id_map)
    transcriptome_df = transcriptome_df[[col for col in transcriptome_df.columns if col in metadata["ID"].values]]

    # merge big abx:
    new_path = os.path.join(norm_folder, r"mRNA_NEBNext_20200908")
    new_data = pd.read_csv(os.path.join(new_path, "mRNA_NEBNext_20200908_genes_norm_named.tsv"), sep="\t")
    # sum rows with the same gene_name and drop the gene_id column
    # new_data = new_data.drop("gene_id", axis=1).groupby("gene_name").sum()
    new_stats = pd.read_csv(os.path.join(new_path, r"big_abx_stats.csv"))
    # remove all samples with "aligned" < 0.5
    columns_to_keep = new_stats[new_stats["aligned"] > filter_value]["Sample Name"]
    # new_data = new_data[columns_to_keep.append(pd.Series(["gene_name", "gene_id"]))]
    columns_to_keep = columns_to_keep.tolist()  # Convert to list if needed
    columns_to_keep.append("gene_name")  # Append to the list
    columns_to_keep.append("gene_id")
    new_data.columns = [col.split("_")[-1] if "gene" not in col else col for col in new_data.columns]
    # drop columns C1, C2, C3 as they already exist in the other df
    new_data = new_data.drop(["C1", "C2", "C3"], axis=1)

    new_data[index_col] = new_data.apply(lambda row: row["gene_id"] if pd.isna(row[index_col]) else row[index_col],
                                         axis=1)
    new_data = new_data.set_index(index_col).drop(drop_col, axis=1)

    transcriptome_df = pd.merge(transcriptome_df, new_data, left_index=True, right_index=True)
    new_metadata = get_metadata(norm_folder, qc_file_suffix=qc_suffix, only_old=not new, filter_threshold=False)
    new_metadata = new_metadata[new_metadata["ID"].isin(new_data.columns)]
    metadata = pd.concat([metadata, new_metadata])

    transcriptome_df = transcriptome_df.groupby(transcriptome_df.index).sum()

    # Remove sparse genes
    transcriptome_zeros = transcriptome_df[transcriptome_df == 0].count(axis=1)
    transcriptome_sparse = transcriptome_zeros[transcriptome_zeros > 0.5 * transcriptome_df.shape[1]]
    transcriptome_df = transcriptome_df.drop(transcriptome_sparse.index)

    if remove_mitochondrial:
        matching_indices = transcriptome_df.index[
            transcriptome_df.index.str.lower().isin(set(MITOCHONDRIAL_GENES))].tolist()
        transcriptome_df = transcriptome_df.drop(matching_indices, errors='ignore')

    # Normalize to TPM/RPM
    transcriptome_df = (transcriptome_df * 1e6).divide(transcriptome_df.sum(axis=0), axis=1)

    # Clean up samples
    to_remove = ["C9", "C10", "C18", "M13", "V14"]
    transcriptome_df = transcriptome_df.drop(to_remove, axis=1, errors='ignore')
    metadata = metadata[~metadata["ID"].isin(to_remove)]

    return transcriptome_df, metadata


def run_analysis(data_folder, output_dir, primary_condition_col, control_value, secondary_condition_col=None,
                 significance_threshold=0.05):
    """
    Main function to execute the gene clustering and correlation analysis.
    Calculates DE, applies FDR correction, and saves all results to CSV.

    Args:
        data_folder (str): Path to the root data directory.
        output_dir (str): Path to save results and plots.
        primary_condition_col (str): The main metadata column to test (e.g., 'Drug').
        control_value (str): The value in `primary_condition_col` that is the control (e.g., 'PBS').
        secondary_condition_col (str, optional): A second column for nested analysis (e.g., 'Treatment').
        significance_threshold (float, optional): P-value cutoff for individual gene t-tests.
    """
    print("Step 1: Reading and processing data...")
    os.makedirs(output_dir, exist_ok=True)
    transcriptome, metadata = read_process_files(data_folder)

    print("Step 2: Transforming data (impute, log2, z-score)...")
    data, metadata = transform_data(transcriptome, metadata)

    primary_conditions = [c for c in metadata[primary_condition_col].unique() if c != control_value]
    secondary_conditions = metadata[secondary_condition_col].unique() if secondary_condition_col else [None]

    for primary_val in primary_conditions:
        for secondary_val in secondary_conditions:
            if secondary_val:
                current_meta = metadata[
                    ((metadata[primary_condition_col] == primary_val) | (
                            metadata[primary_condition_col] == control_value)) &
                    (metadata[secondary_condition_col] == secondary_val)
                    ]
                condition_name_desc = f"{primary_val}-{secondary_val}"
            else:
                current_meta = metadata[
                    (metadata[primary_condition_col] == primary_val) | (
                            metadata[primary_condition_col] == control_value)
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

            # 1. Calculate DE stats for ALL genes in the current dataset
            print(f"Calculating DE stats for all {len(all_genes_in_data)} genes...")
            results_df = genes_data_split(
                primary_val, control_value, all_genes_in_data, current_expression, metadata,
                primary_condition_col, secondary_condition_col, secondary_val,
            )
            if results_df.empty:
                print(f"No results generated for {condition_name_desc}.")
                continue

            # Keep only genes where logfold > 1
            results_df = results_df[results_df['log2fc'].abs() > LOG2FC_CUTOFF]

            # 2. Add FDR (BH) correction
            # multipletests returns: (reject, pvals_corrected, alphacSidak, alphacBonf)
            reject, pvals_adj, _, _ = multipletests(
                results_df['p_val'],
                alpha=significance_threshold,
                method='fdr_bh'
            )
            results_df['fdr_bh'] = pvals_adj
            results_df['significant_fdr'] = reject

            # 3. Add trend based on t-statistic (or log2fc)
            results_df['trend'] = np.where(results_df['t_stat'] > 0, 'enhanced', 'suppressed')

            # 4. Sort by FDR for easy viewing
            results_df = results_df.sort_values(by='fdr_bh', ascending=True)

            # 5. Save all results to a file
            output_filename = os.path.join(output_dir, f"DE_results_{condition_name_desc}.csv")
            results_df.to_csv(output_filename, index=True)  # index=True saves the gene names/ID
            print(f"Found {results_df['significant_fdr'].sum()} significant genes (FDR < {significance_threshold}).")
            print(f"Saved full DE results to: {output_filename}")




def main():
    parser = argparse.ArgumentParser(description="Run the ClusteringGO analysis pipeline.")
    parser.add_argument("data_dir", help="Path to the root data directory.")
    parser.add_argument("output_dir", help="Path to save results and plots.")
    parser.add_argument("--primary_col", required=True, help="The main metadata column to test (e.g., 'Drug').")
    parser.add_argument("--control_val", required=True, help="The control value in the primary column (e.g., 'PBS').")
    parser.add_argument("--secondary_col", default=None,
                        help="Optional second column for nested analysis (e.g., 'Treatment').")
    parser.add_argument("--significance_threshold", type=float, default=0.05,
                        help="P-value cutoff for FDR (default: 0.05).")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found at '{args.data_dir}'")
    else:
        run_analysis(args.data_dir, args.output_dir, args.primary_col, args.control_val, args.secondary_col,
                     args.significance_threshold)

        # You can adjust the cutoffs here if you want
        print(f"Using cutoffs: FDR < {FDR_CUTOFF} and |Log2FC| > {LOG2FC_CUTOFF}")
        os.makedirs(os.path.join(args.output_dir, "filtered"), exist_ok=True)
        filter_de_files(args.output_dir, os.path.join(args.output_dir, "filtered1_5"))

        os.makedirs(os.path.join(args.output_dir, "enrichr1_5"), exist_ok=True)

        # Find all .txt files in the input directory
        search_pattern = os.path.join(os.path.join(args.output_dir, "filtered1_5"), "*.txt")
        gene_list_files = glob.glob(search_pattern)

        if not gene_list_files:
            print(f"No '.txt' files found in {os.path.join(args.output_dir, 'filtered')}")
            return

        print(f"Found {len(gene_list_files)} gene lists to analyze.")

        for f_path in gene_list_files:
            run_enrichment_on_file(f_path, os.path.join(args.output_dir, "enrichr1_5"))
            # Be polite to the server: wait 1 second between requests
            time.sleep(1)

        print("\nEnrichment analysis complete.")


if __name__ == "__main__":
    main()
