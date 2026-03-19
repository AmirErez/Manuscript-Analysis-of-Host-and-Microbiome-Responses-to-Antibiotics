import pandas as pd
import glob
import ast
import re
import os

# SET THIS: path to your CompoRes output directory (containing CompoRes_Clock results)
COMPORES_PATH = "./Private"

# SET THIS: path to the DIABLO analysis outputs directory
DIABLO_OUTPUT_PATH = ""


# --- 1. Helper Function to Clean Taxa Names ---
def extract_core_genus(taxon_str):
    """
    Cleans strings like "s__Duncaniella dubosii" or "g_UBA7173"
    to return just the core genus: "Duncaniella" and "UBA7173".
    """
    # Remove standard brackets/quotes
    clean = taxon_str.strip("[]'\"")
    # Remove typical prefix markers (g__, s__, g_, etc.)
    clean = re.sub(r'^[kpcofgs]_+?', '', clean)
    # If it's a species (contains a space), just take the genus (first word)
    clean = clean.split(' ')[0]
    return clean

summary_table = pd.DataFrame(columns=[
    "Treatment",
    "Gene",
    "Total_Possible_Edges",
    "CompoRes_Significant_Edges",
    "DIABLO_Selected_Edges",
    "Overlapping_Edges",
    "Jaccard_Index_Percent",
    "Random_Expectation_Percent",
    "Precision_Percent",
    "Expected_Precision_Percent"
])
for treatment in ["IP", "IV", "PO"]:
    # Build the path using os.path.join so that glob patterns work reliably
    path = os.path.join(
        COMPORES_PATH,
        "CompoRes_Clock",
        f"compores_all_Van{treatment}_clock_119",
        "balance_calculation_results",
        f"Van-{treatment}-feces",
        # "CLR",
        "pairs",
        "ocu_level_results",
        "regular",
    )

    # --- 2. Load CompoRes Edges (The Ground Truth) ---
    # Use os.path.join to create the glob pattern (directory + pattern)
    pattern = os.path.join(path, "*combined_ocu_level_results.csv")
    compores_files = glob.glob(pattern)

    if not compores_files:
        # Helpful warning so the user knows the directory/pattern might be wrong
        print(f"Warning: no compores files found for treatment '{treatment}' using pattern: {pattern}")

    compores_edges = set()

    # Note: Update this mapping if your DIABLO output uses gene symbols instead of Ensembl IDs
    # format: {'ENSMUSG00000059824': 'dbp', ...}
    ensembl_to_symbol = {
        'ENSMUSG00000059824': 'dbp',
        'ENSMUSG00000028957': 'ciart',
        'ENSMUSG00000056749': 'nfil3',
        'ENSMUSG00000038550': 'per3'
    }

    for f in compores_files:
        # Extract Ensembl ID from filename and map to symbol
        ens_id = re.search(r'(ENSMUSG\d+)', f).group(1)
        gene = ensembl_to_symbol.get(ens_id, ens_id)
        curr_gene = gene
        if gene not in ensembl_to_symbol.values():
            print(f"Warning: Ensembl ID '{ens_id}' is not in our analysis.")
            continue

        df = pd.read_csv(f)
        # Filter for significance (p < 0.05)
        sig_df = df[df['rho_p_value'] < 0.05]

        for _, row in sig_df.iterrows():
            # Parse NUM and DEN string lists into actual lists
            try:
                num_list = ast.literal_eval(row['NUM_Taxa_List'])
                den_list = ast.literal_eval(row['DEN_Taxa_List']) if "DEN_Taxa_List" in row else []
            except:
                continue

            # Create Gene-Taxon edge tuples
            for taxon in (num_list + den_list):
                genus = extract_core_genus(taxon)
                compores_edges.add((gene, genus.lower()[1:]))

        # --- 3. Load DIABLO Edges ---
        # Replace with your actual DIABLO correlation network filename
        base = os.path.join(DIABLO_OUTPUT_PATH, f"DIABLO_Compare_Van-{treatment}_vs_PBS-{treatment}1component")
        diablo_df = pd.read_csv(base+"_correlation_network.csv")
        diablo_edges = set()

        for _, row in diablo_df.iterrows():
            gene = row['Transcriptome_Feature']  # E.g., 'dbp'
            gene = gene.split("_")[0].lower()  # Remove any suffixes
            if gene != curr_gene:
                continue  # Skip if this gene is not the one we're currently analyzing from CompoRes
            genus = extract_core_genus(row['Microbiome_Feature'])
            diablo_edges.add((gene, genus.lower()))

        # --- 4. Load Background Stats ---
        stats_df = pd.read_csv(base+"_network_background_stats.csv")
        total_possible_edges = stats_df['Total_Possible_Edges'].iloc[0]

        # --- 5. Calculate Metrics ---
        # Jaccard Index
        intersection = compores_edges.intersection(diablo_edges)
        union = compores_edges.union(diablo_edges)

        if len(union) == 0:
            jaccard = 0.0
        else:
            jaccard = len(intersection) / len(union)

        if len(diablo_edges) == 0:
            precision = 0.0
        else:
            precision = len(intersection) / len(diablo_edges)

        # Random Expectation (YY%)
        total_possible_edges = 328
        # The probability of a random edge being a CompoRes edge
        expected_precision = len(compores_edges) / total_possible_edges
        # expected_precision = len(compores_edges) / total_possible_edges

        # Expected Jaccard (Random Selection)
        # If DIABLO randomly picked len(diablo_edges) out of total_possible_edges,
        # the expected intersection with CompoRes is:
        expected_intersection = (len(diablo_edges) * len(compores_edges)) / total_possible_edges
        expected_union = len(diablo_edges) + len(compores_edges) - expected_intersection
        expected_jaccard = expected_intersection / expected_union

        to_append = pd.DataFrame([{
            "Treatment": treatment,
            "Gene": curr_gene,
            "Total_Possible_Edges": total_possible_edges,
            "CompoRes_Significant_Edges": len(compores_edges),
            "DIABLO_Selected_Edges": len(diablo_edges),
            "Overlapping_Edges": len(intersection),
            "Jaccard_Index_Percent": jaccard * 100,
            "Random_Expectation_Percent": expected_jaccard * 100,
            "Precision_Percent": precision * 100,
            "Expected_Precision_Percent": expected_precision * 100
        }])
        summary_table = pd.concat([summary_table, to_append], ignore_index=True)

        # --- 6. Print Results ---
        print("=== Overall Network Agreement ===")
        print(f"Total Possible Edges in Model: {total_possible_edges}")
        print(f"CompoRes Significant Edges:    {len(compores_edges)}")
        print(f"DIABLO Selected Edges:         {len(diablo_edges)}")
        print(f"Overlapping Edges:             {len(intersection)}")
        print("-" * 33)
        print(f"Jaccard Index (XX%):           {jaccard * 100:.2f}%")
        print(f"Random Expectation (YY%):      {expected_jaccard * 100:.2f}%")
        print("-" * 33)

        print("\nSpecific Overlaps Found:")
        for edge in intersection:
            print(f"  Gene: {edge[0]:<6} | Genus: {edge[1]}")

# --- 7. Save Summary Table ---
output_path = os.path.join(COMPORES_PATH, "CompoRes_Clock")
summary_table.to_csv(output_path + "/DIABLO_vs_CompoRes_Summary.csv", index=False)
print(f"\nSummary table saved as '{output_path}DIABLO_vs_CompoRes_Summary.csv'")