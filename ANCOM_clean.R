library(ANCOMBC)
library(phyloseq)
path <- "C:/Users/Yehonatan/Desktop/Master/Git/DEP_Compare16s/Private/CompoResGenes/"
antibiotics_list <- c("Amp", "Van", "Neo", "Met", "Mix")
treatments_list  <- c("IP", "IV", "PO")
all_results_clean <- list()
taxa_all <- list()

for (antibiotic in antibiotics_list) {
  for (treatment in treatments_list) {
    
    # 1) Load feature table & metadata
    feat_fp <- file.path(path, "microbiome", "genus",
                         paste0(antibiotic, "-", treatment, "-feces.tsv"))
    meta_fp <- file.path(path, "metadata",
                         paste0(antibiotic, "-", treatment, "-metadata.tsv"))
    
    feature_data <- read.delim(feat_fp, sep="\t", header=TRUE,
                               stringsAsFactors=FALSE, row.names=1)
    metadata     <- read.delim(meta_fp, sep="\t", header=TRUE,
                               stringsAsFactors=FALSE, row.names=1)
    metadata$Category <- factor(metadata$Category, levels = c(paste0(antibiotic, "_", treatment), paste0("PBS_", treatment)))
    
    
    # 2) Transpose
    feature_table <- t(feature_data)
    
    # 3) Build phyloseq object
    otu  <- otu_table(feature_table, taxa_are_rows=TRUE)
    samp <- sample_data(metadata)
    pseq <- phyloseq(otu, samp)
    
    # 4) Identify taxa with zero variance within any group
    group_levels <- unique(metadata$Category)
    otu_mat <- as(otu_table(pseq), "matrix")
    
    zero_var_taxa <- apply(otu_mat, 1, function(taxon_counts) {
      any(sapply(group_levels, function(group) {
        group_samples <- rownames(metadata[metadata$Category == group, , drop=FALSE])
        var(taxon_counts[group_samples], na.rm=TRUE) == 0
      }))
    })
    
    # Keep only taxa with non-zero variance in all groups
    pseq <- prune_taxa(!zero_var_taxa, pseq)
    
    
    # 5) Run ANCOMBC2
    out <- ancombc2(
      data         = pseq,
      fix_formula  = "Category",
      p_adj_method = "BH",
      verbose      = TRUE,
      lib_cut      = 0
    )
    
    # 6) Auto-detect which element of out$res matches this treatment
    # Try expected ANCOMBC2 result name
    res_name <- paste0("q_CategoryPBS_", treatment)
    if (!res_name %in% names(out$res)) {
      stop("Expected result name not found: ", res_name, 
           "\nAvailable: ", paste(names(out$res), collapse=", "))
    }
    
    # 8) Extract q-values and count significance
    q_values <- out$res[[ res_name ]]
    # Fix: This line was using res_df which isn't defined, using out$res$taxon instead
    taxa_names <- out$res$taxon
    significant_count <- sum(q_values < 0.05, na.rm=TRUE)
    
    #print(antibiotic)
    #print(treatment)
    #print(out$res$taxon[q_values < 0.05])
    
    # 9) Build one-row summary + all taxa-columns
    summary_df <- data.frame(
      antibiotic        = antibiotic,
      treatment         = treatment,
      significant_count = significant_count,
      stringsAsFactors  = FALSE
    )
    
    # Create a data frame for significant taxa with their q-values and p-values
    # Add the p-value column name based on treatment
    p_value_col <- paste0("p_CategoryPBS_", treatment)
    
    # Create data frame with taxon information for significant results
    sig_idx <- which(q_values > -1)
    if(length(sig_idx) > 0) {
      summary_taxa <- data.frame(
        antibiotic = antibiotic,
        treatment = treatment,
        taxon = out$res$taxon[sig_idx],
        q_value = q_values[sig_idx],
        p_value = out$res[[p_value_col]][sig_idx],
        lfc = -1*out$res[[paste0("lfc_CategoryPBS_", treatment)]][sig_idx],
        stringsAsFactors = FALSE
      )
    } else {
      # Create empty data frame with correct structure if no significant results
      summary_taxa <- data.frame(
        antibiotic = character(0),
        treatment = character(0),
        taxon = character(0),
        q_value = numeric(0),
        p_value = numeric(0),
        lfc = numeric(0),
        stringsAsFactors = FALSE
      )
    }
    
    all_results_clean[[ paste(antibiotic, treatment, sep="_") ]] <- summary_df
    taxa_all[[ paste(antibiotic, treatment, sep="_") ]] <- summary_taxa
  }
}

# 10) Combine all into your final table
final_table <- do.call(rbind, all_results_clean)

# Combine all taxa information
all_taxa_table <- do.call(rbind, taxa_all)

# Inspect or save
print(final_table)
print(head(all_taxa_table))

# Save summary results
write.table(final_table,
            file.path(path, "ANCOMBC_qvalues_summary_genus.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Save significant taxa results
write.table(all_taxa_table,
            file.path(path, "ANCOMBC_significant_taxa_genus.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
