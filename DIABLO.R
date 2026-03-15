#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# DIABLO Multi-Omics Integration Pipeline (Adapted for 5x3 Design)
#
# Author: Adapted by Bioinformatician
# Date: October 24, 2025
#
# Description:
# This script performs a DIABLO analysis on microbiome and transcriptome
# data, based on a 5-Antibiotic x 3-Treatment experimental design.
#
# v3: Includes gene name translation AND saves the circos plot
#     correlation network as a CSV file.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#=============================================================================
# 1. SETUP: LOAD LIBRARIES
#=============================================================================

# Install missing packages if necessary
if (!requireNamespace("mixOmics", quietly = TRUE)) install.packages("mixOmics")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("readr", quietly = TRUE)) install.packages("readr") # For read_csv/read_tsv
if (!requireNamespace("matrixStats", quietly = TRUE)) install.packages("matrixStats") # For variance filtering
if (!requireNamespace("svglite", quietly = TRUE)) install.packages("svglite") # <-- ADD THIS

library(mixOmics)
library(tidyverse)
library(readr)
library(matrixStats) # For rowVars
library(svglite)

# --- CONFIGURATION ---
# Define the paths to your files
main_path <- "/Users/yonchlevin/Desktop/ErezLab/MouseAbxBel/Git/DEP_Compare16s/Private/"
PATH_METADATA <- paste0(main_path, "metadata.csv")
# PATH_TRANSCRIPTOME <- paste0(main_path, "transcriptome_transformed.csv") 
# TODO: note that this option (above) might have been used. In this case, replace the log2 section with the section above
PATH_TRANSCRIPTOME <- paste0(main_path, "transcriptome.csv")
PATH_MICROBIOME <- paste0(main_path, "otu_merged_feces_genus_qiime.tsv")

# *** NEW: Path to your gene mapping file ***
PATH_GENE_MAP <- "/Users/yonchlevin/Desktop/ErezLab/MouseAbxBel/Git/Data/MultiAbx-16s/MultiAbx-RPKM-RNAseq-B6/new normalization/transcriptome_2023-09-17-genes_norm_named.tsv"

# Define the output directory for plots and tables
# PLEASE UPDATE THIS to your desired output folder
OUTPUT_DIR <- "/Users/yonchlevin/Desktop/ErezLab/MouseAbxBel/DIABLO/DIABLO_Analysis_Outputs"
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR)
}

cat("Setup complete. Libraries loaded.\n")

#=============================================================================
# 2. DATA LOADING
#=============================================================================
cat("Step 2: Loading data...\n")

# --- Load Metadata ---
metadata <- read_csv(PATH_METADATA) %>%
  mutate(ID = as.character(ID))

# --- Load Transcriptome Data (Features as Rows) ---
transcriptome_raw <- read_csv(PATH_TRANSCRIPTOME)

# --- Load 16S Microbiome Data ---
microbiome_raw <- read_tsv(PATH_MICROBIOME)

# --- Load Gene Mapping File ---
tryCatch({
  gene_map_raw <- read_tsv(PATH_GENE_MAP)
  gene_map_df <- gene_map_raw %>%
    select(gene_id, gene_name) %>%
    distinct()
  cat(sprintf("Successfully loaded gene map with %d unique gene_id entries.\n", nrow(gene_map_df)))
}, error = function(e) {
  stop(paste("FATAL ERROR: Could not load gene mapping file:", PATH_GENE_MAP, "\nError:", e$message))
})

cat("Data loading complete.\n")


#=============================================================================
# 3. PRE-PROCESSING: 16S MICROBIOME DATA
#=============================================================================
cat("Step 3: Pre-processing 16S microbiome data...\n")

microbiome_matrix <- microbiome_raw %>%
  mutate(sample_id = str_remove(sample_id, "\\.d4$")) %>%
  column_to_rownames("sample_id") %>%
  select(-any_of(c("is_treatment", "antibiotic", "treatment")))

n_samples <- nrow(microbiome_matrix)
min_prevalence <- 0.10
keep_genera <- colSums(microbiome_matrix > 5) > (n_samples * min_prevalence)
microbiome_filtered <- microbiome_matrix[, keep_genera]

cat(sprintf("Filtered microbiome data: %d samples, %d genera remaining.\n",
            nrow(microbiome_filtered), ncol(microbiome_filtered)))

microbiome_clr <- logratio.transfo(microbiome_filtered, logratio = 'CLR', offset = 1)
cat("CLR transformation complete.\n")


#=============================================================================
# 4. PRE-PROCESSING: TRANSCRIPTOME DATA
#=============================================================================
cat("Step 4: Pre-processing Transcriptome data...\n")

transcriptome_transposed <- transcriptome_raw %>%
  column_to_rownames("gene_name") %>% # gene_name column here is actually ENSMUSG IDs
  t() %>%
  as.data.frame()

transcriptome_log2 <- as.matrix(log2(transcriptome_transposed + 1))

# gene_variances <- rowVars(t(as.matrix(transcriptome_transposed)))
# names(gene_variances) <- colnames(transcriptome_transposed)
# 
# N_GENES_TO_KEEP <- 5000
# top_genes <- names(sort(gene_variances, decreasing = TRUE))[1:N_GENES_TO_KEEP]
# transcriptome_processed <- transcriptome_transposed[, top_genes]
gene_variances <- rowVars(t(as.matrix(transcriptome_log2)))
names(gene_variances) <- colnames(transcriptome_log2)

N_GENES_TO_KEEP <- 5000
top_genes <- names(sort(gene_variances, decreasing = TRUE))[1:N_GENES_TO_KEEP]
transcriptome_processed <- transcriptome_log2[, top_genes]

#=============================================================================
# 5. DATA ALIGNMENT (SIMPLIFIED)
#=============================================================================
cat("Step 5: Aligning all data...\n")

common_samples <- intersect(
  rownames(microbiome_clr),
  rownames(transcriptome_processed)
)

common_samples <- intersect(
  common_samples,
  metadata$ID
)

cat(sprintf("Found %d common samples across all 3 files.\n", length(common_samples)))

# --- Create Master Datasets ---
microbiome_master <- microbiome_clr[common_samples, ]
transcriptome_master_unnamed <- transcriptome_processed[common_samples, ]

metadata_master <- metadata %>%
  filter(ID %in% common_samples) %>%
  arrange(match(ID, common_samples))

cat("Renaming transcriptome features with gene names...\n")
# Create a dataframe of the current column names (ENSMUSG IDs)
ensmus_ids_df <- data.frame(gene_id = colnames(transcriptome_master_unnamed))
# Join with the gene map
name_map <- ensmus_ids_df %>%
  left_join(gene_map_df, by = "gene_id")
# Handle missing names (if any) by keeping the original ENSMUSG ID
name_map$gene_name <- ifelse(is.na(name_map$gene_name), name_map$gene_id, name_map$gene_name)
# Make names unique (e.g., if two IDs map to "Gm4728", they become "Gm4728" and "Gm4728.1")
new_gene_names <- make.unique(name_map$gene_name, sep = ".")
cat(sprintf("Total %d features, %d unique gene names found.\n", 
            length(new_gene_names), length(unique(new_gene_names))))
# Apply the new, readable names to the master matrix
transcriptome_master <- transcriptome_master_unnamed
colnames(transcriptome_master) <- new_gene_names

if (!all(rownames(microbiome_master) == metadata_master$ID) ||
    !all(rownames(transcriptome_master) == metadata_master$ID)) {
  stop("FATAL ERROR: Sample IDs are not aligned after filtering. Aborting.")
}

X_master <- list(
  microbiome = as.matrix(microbiome_master),
  transcriptome = as.matrix(transcriptome_master)
)

design <- matrix(1, ncol = length(X_master), nrow = length(X_master),
                 dimnames = list(names(X_master), names(X_master)))
diag(design) <- 0

cat("Master X and metadata objects created and aligned.\n")


#=============================================================================
# 6. RUN ALL DIABLO ANALYSES
#=============================================================================
cat("Step 6: Starting DIABLO analyses...\n")

# --- Define a helper function to run one analysis ---
run_diablo_analysis <- function(X_data, Y_variable, analysis_name, n_components = 2) {
  
  cat(sprintf("\n\n--- Running Analysis: %s ---\n", analysis_name))
  cat(sprintf("Number of groups in Y: %d\n", length(unique(Y_variable))))
  
  Y <- as.factor(Y_variable)
  order_indices <- order(Y)
  # Re-order Y
  Y_sorted <- Y[order_indices]
  # Re-order X data
  X_data_sorted <- list(
    microbiome = X_data$microbiome[order_indices, , drop = FALSE],
    transcriptome = X_data$transcriptome[order_indices, , drop = FALSE]
  )
  
  # --- Dynamic list.keepX generation ---
  n_features_microbiome <- ncol(X_data_sorted$microbiome)
  n_features_transcriptome <- ncol(X_data_sorted$transcriptome)
  
  cat(sprintf("Microbiome features: %d, Transcriptome features: %d\n", 
              n_features_microbiome, n_features_transcriptome))
  
  test_keep_values <- c(seq(10, 50, 10))
  
  keepX_microbiome <- test_keep_values[test_keep_values <= n_features_microbiome]
  keepX_transcriptome <- test_keep_values[test_keep_values <= n_features_transcriptome]
  
  if (length(keepX_microbiome) == 0) {
    keepX_microbiome <- c(n_features_microbiome)
  }
  if (length(keepX_transcriptome) == 0) {
    keepX_transcriptome <- c(n_features_transcriptome)
  }
  
  list.keepX <- list(
    microbiome = keepX_microbiome,
    transcriptome = keepX_transcriptome
  )
  
  cat("Using dynamic keepX list for tuning:\n")
  print(list.keepX)
  
  # --- Model Tuning ---
  cat("Running model tuning (this may take a while)...\n")
  # set.seed(321)
  tune.model <- tune.block.splsda(
    X = X_data_sorted, Y = Y_sorted, ncomp = n_components,
    test.keepX = list.keepX, design = design,
    validation = 'Mfold', folds = 5, nrepeat = 3,
    dist = "mahalanobis.dist", progressBar = TRUE
  )
  
  optimal.keepX <- tune.model$choice.keepX
  cat("Tuning complete. Optimal keepX chosen:\n")
  print(optimal.keepX)
  
  # --- Final Model ---
  final.model <- block.splsda(
    X = X_data_sorted, Y = Y_sorted, ncomp = n_components,
    keepX = optimal.keepX, design = design
  )
  
  # --- Save Results ---
  output_prefix <- file.path(OUTPUT_DIR, paste0("DIABLO_", analysis_name))
  
  # Save plots as SVG
  cat("Saving sample plot as SVG...\n")
  svglite(paste0(output_prefix, "_sample_plot_comp1.svg"), width = 10, height = 8)
  print(plotDiablo(final.model, ncomp = 1, legend = TRUE, title = analysis_name))
  dev.off()
  
  cat("Saving circos plot as SVG...\n")
  svglite(paste0(output_prefix, "_circos_plot.svg"), width = 10, height = 10)
  circosPlot(final.model, cutoff = 0.7, line = TRUE,
             color.blocks = c('brown1', 'royalblue1'),
             color.cor = c("chocolate3", "grey20"), size.labels = 1.5,
             size.variables = 0.9)
  dev.off()
  
  cat("Saving heatmaps as SVG...\n")
  svglite(paste0(output_prefix, "_heatmap.svg"), width = 20, height = 25)
  cimDiablo(final.model, margin = c(8, 12))
  dev.off()
  
  svglite(paste0(output_prefix, "_heatmap_grouped.svg"), width = 20, height = 25)
  cimDiablo(final.model, margin = c(8, 12), cluster = "col")
  dev.off()
  
  # --- Save Loadings Tables (Feature Lists) ---
  for (comp in 1:n_components) {
    loadings_list <- list()
    for (block_name in names(final.model$loadings)) {
      block_loadings <- final.model$loadings[[block_name]][, comp, drop = FALSE]
      
      if (nrow(block_loadings) > 0) {
        df <- as.data.frame(block_loadings) %>%
          rownames_to_column(var = "feature") %>%
          rename(loading = paste0("comp", comp)) %>%
          mutate(block = block_name) %>%
          filter(loading != 0)
        
        loadings_list[[block_name]] <- df
      }
    }
    loadings_table <- bind_rows(loadings_list)
    
    # loadings_table_named <- loadings_table %>%
      # left_join(gene_map, by = c("feature" = "gene_id"))
    
    # if (nrow(loadings_table_named) > 0) {
    if (nrow(loadings_table) > 0) {
        write.csv(loadings_table,
                file = paste0(output_prefix, "_loadings_comp", comp, ".csv"),
                row.names = FALSE)
    }
  }
  
  # --- Save Correlation Network Table ---
  cat("Generating correlation network table...\n")
  
  # Define the cutoff (same as circosPlot)
  cor_cutoff <- 0.7 
  
  # Use pdf(NULL) to prevent 'network' from opening a plot window
  pdf(NULL) 
  correlation_matrix <- network(final.model, blocks = c('microbiome', 'transcriptome'),
                                color.node = c('brown1', 'royalblue1'), cutoff = cor_cutoff, plot = FALSE)
  dev.off()
  
  if (!is.null(correlation_matrix) && !is.null(correlation_matrix$M)) {
    correlation_table <- as.data.frame(as.table(correlation_matrix$M))
    colnames(correlation_table) <- c("Microbiome_Feature", "Transcriptome_Feature", "Correlation")
    
    correlation_table_filtered <- correlation_table %>%
      filter(Correlation != 0) %>%
      arrange(desc(abs(Correlation)))
    
    # # Also join with gene map to make this table readable
    # correlation_table_named <- correlation_table_filtered %>%
    #   left_join(gene_map, by = c("Transcriptome_Feature" = "gene_id"))
    
    # write.csv(correlation_table_named, 
    write.csv(correlation_table_filtered, 
                        file = paste0(output_prefix, "_correlation_network.csv"), 
              row.names = FALSE)
    
    # cat(sprintf("Saved correlation network table with %d links.\n", nrow(correlation_table_named)))
    cat(sprintf("Saved correlation network table with %d links.\n", nrow(correlation_table_filtered)))
} else {
    cat(sprintf("No correlations found above cutoff %f. Skipping network table.\n", cor_cutoff))
  }
  # --- *** END OF NEW SECTION *** ---
  
  
  cat(sprintf("--- Analysis '%s' complete. Results saved. ---\n", analysis_name))
}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ANALYSIS 1: Compare 3 Treatments (pooling all antibiotics)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
run_diablo_analysis(
  X_data = X_master,
  Y_variable = metadata_master$Treatment,
  analysis_name = "Compare_Treatments_All-Abx",
  # gene_map = gene_map_df,
  n_components = 4
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ANALYSIS 2: Compare 5 Antibiotics (pooling all treatments)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
run_diablo_analysis(
  X_data = X_master,
  Y_variable = metadata_master$Drug,
  analysis_name = "Compare_Antibiotics_All-Trt",
  # gene_map = gene_map_df,
  n_components = 4
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ANALYSIS 3: Compare 5 Antibiotics *within* each Treatment
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cat("\n\n--- Starting Analysis 3: Abx within each Trt ---\n")
all_treatments <- unique(metadata_master$Treatment)

for (t_level in all_treatments) {
  
  t_level_clean <- gsub("[^A-Za-z0-9]", "_", t_level)
  
  indices <- which(metadata_master$Treatment == t_level)
  metadata_sub <- metadata_master[indices, ]
  X_sub <- list(
    microbiome = X_master$microbiome[indices, ],
    transcriptome = X_master$transcriptome[indices, ]
  )
  
  run_diablo_analysis(
    X_data = X_sub,
    Y_variable = metadata_sub$Drug,
    analysis_name = paste0("Compare_Abx_within_Trt-", t_level_clean),
    # gene_map = gene_map_df,
    n_components = 4
  )
}
cat("--- Analysis 3 Complete ---\n")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ANALYSIS 4: Compare 3 Treatments *within* each Antibiotic
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cat("\n\n--- Starting Analysis 4: Trt within each Abx ---\n")
all_drugs <- unique(metadata_master$Drug)

for (a_level in all_drugs) {
  
  a_level_clean <- gsub("[^A-Za-z0-9]", "_", a_level)
  
  indices <- which(metadata_master$Drug == a_level)
  metadata_sub <- metadata_master[indices, ]
  X_sub <- list(
    microbiome = X_master$microbiome[indices, ],
    transcriptome = X_master$transcriptome[indices, ]
  )
  
  run_diablo_analysis(
    X_data = X_sub,
    Y_variable = metadata_sub$Treatment,
    analysis_name = paste0("Compare_Trt_within_Abx-", a_level_clean),
    # gene_map = gene_map_df,
    n_components = 4
  )
}
cat("--- Analysis 4 Complete ---\n")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ANALYSIS 5: Compare specific Drug-Treatment vs. matched PBS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cat("\n\n--- Starting Analysis 5: Specific Drug vs Matched PBS ---\n")

# Define the 6 specific comparisons you want to run
comparisons <- list(
  list(drug1 = "Van", drug2 = "PBS", trt = "IP", name = "Compare_Van-IP_vs_PBS-IP"),
  list(drug1 = "Van", drug2 = "PBS", trt = "PO", name = "Compare_Van-PO_vs_PBS-PO"),
  list(drug1 = "Van", drug2 = "PBS", trt = "IV", name = "Compare_Van-IV_vs_PBS-IV"),
  list(drug1 = "Neo", drug2 = "PBS", trt = "IP", name = "Compare_Neo-IP_vs_PBS-IP"),
  list(drug1 = "Neo", drug2 = "PBS", trt = "PO", name = "Compare_Neo-PO_vs_PBS-PO"),
  list(drug1 = "Neo", drug2 = "PBS", trt = "IV", name = "Compare_Neo-IV_vs_PBS-IV")
)

# Loop through each defined comparison
for (comp in comparisons) {
  
  cat(sprintf("\n--- Running sub-analysis: %s ---\n", comp$name))
  
  # 1. Find the indices for the two groups
  # We need samples that are (Drug == drug1 AND Treatment == trt) OR (Drug == drug2 AND Treatment == trt)
  indices <- which(
    (metadata_master$Drug == comp$drug1 & metadata_master$Treatment == comp$trt) |
      (metadata_master$Drug == comp$drug2 & metadata_master$Treatment == comp$trt)
  )
  
  # 2. Check if we have enough data (at least 2 samples)
  if (length(indices) < 2) {
    cat(sprintf("Skipping %s: Not enough samples found for this comparison.\n", comp$name))
    next # Skip to the next iteration of the loop
  }
  
  # 3. Create the subsetted data
  metadata_sub <- metadata_master[indices, ]
  X_sub <- list(
    microbiome = X_master$microbiome[indices, , drop = FALSE],
    transcriptome = X_master$transcriptome[indices, , drop = FALSE]
  )
  
  # 4. Check if we have two groups to compare
  # (e.g., if we only found "PBS-IP" but no "Van-IP" samples)
  if (length(unique(metadata_sub$Drug)) < 2) {
    cat(sprintf("Skipping %s: Only one of the two groups was found (e.g., only PBS, no Drug).\n", comp$name))
    next
  }
  
  # 5. Run the analysis
  # The Y_variable is the Drug column, which will have two levels (e.g., "Van" and "PBS")
  run_diablo_analysis(
    X_data = X_sub,
    Y_variable = metadata_sub$Drug, 
    analysis_name = paste0(comp$name, "1component"),
    # analysis_name = paste0(comp$name, "2components"),
    n_components = 1  # For a 2-group comparison, we only need 1 component
    # n_components = 2  # For a 2-group comparison, we only need 1 component
  )
}

cat("--- Analysis 5 Complete ---\n")

#=============================================================================
# 7. SCRIPT FINISHED
#=============================================================================
cat("\n\n==============================================\n")
cat("All DIABLO analyses finished successfully!\n")
cat(paste("Check your output directory:", OUTPUT_DIR, "\n"))
cat("Loading tables and Correlation tables are now saved.\n")
cat("==============================================\n")