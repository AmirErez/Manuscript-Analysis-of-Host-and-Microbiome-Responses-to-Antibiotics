## Microbiome PCoA and PERMANOVA Analysis Script - Modified Version - MULTI-TRANSFORM
## CORRECTED for data format: Samples as Rows, Microbes as Columns
##
## This script iterates over multiple options (amp, van, neo, met, mix, IP, IV, PO)
## Modified to focus on comparing PBS groups with each other and with matching treatment groups
## NOW iterates over four transformation methods

# Load required packages
required_packages <- c("vegan", "dplyr", "compositions")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing package:", pkg, "\n")
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# ==============================================================================
# START: Transformation Functions
# ==============================================================================

# Helper function to calculate geometric mean
geom_mean_r <- function(x) {
  # x is expected to be all positive (after zero-replacement)
  exp(mean(log(x)))
}

# 1. Aitchison (CLR) transformation
#    Operates on counts, replaces 0s with epsilon
transform_clr <- function(x, epsilon) {
  x_mod <- ifelse(x == 0, epsilon, x)
  gm <- geom_mean_r(x_mod)
  log(x_mod / gm)
}

# 2. Arcsin-Sqrt transformation
#    Operates on counts, converts to proportions internally
transform_arcsin_sqrt <- function(x, epsilon) {
  s <- sum(x)
  if (s == 0) return(rep(0, length(x))) # Handle all-zero rows
  x_prop <- x / s
  asin(sqrt(x_prop))
}

# 3. Log transformation
#    Operates on counts + epsilon
transform_log <- function(x, epsilon) {
  log(x + epsilon)
}

# 4. Logit transformation
#    Operates on counts, converts to proportions internally
transform_logit <- function(x, epsilon) {
  s <- sum(x)
  if (s == 0) return(rep(0, length(x))) # Handle all-zero rows
  x_prop <- x / s
  x_prop <- ifelse(x_prop == 0, epsilon, x_prop)
  x_prop <- ifelse(x_prop == 1, 1 - epsilon, x_prop) # Handle 1s
  log(x_prop / (1 - x_prop))
}

# A list mapping the method name to its function
transformation_list <- list(
  "Aitchison" = transform_clr,
  "Arcsin_Sqrt" = transform_arcsin_sqrt,
  "Log" = transform_log,
  "Logit" = transform_logit
)

# ==============================================================================
# END: Transformation Functions
# ==============================================================================


# Define the options to iterate over
# options_to_process <- c("amp")
options_to_process <- c("amp", "van", "neo", "met", "mix", "IP", "IV", "PO")
permutations_num = 1e+05

# Set base file paths
metadata_path <- "./Data/"
data_path <- "./Private/qiime/"
metadata_file <- paste0(metadata_path, "qiime_metadata.tsv")
out_path <- "./Private/adonis/"

# Check if metadata file exists
if (!file.exists(metadata_file)) {
  stop("ERROR: Metadata file not found at: ", metadata_file)
}

# Read metadata once
metadata <- read.table(metadata_file, header = TRUE, sep = "\t", check.names = FALSE, comment.char = "")
metadata <- as.data.frame(metadata)
cat("Metadata loaded: ", nrow(metadata), "rows x", ncol(metadata), "columns\n")

# Process metadata once
metadata <- metadata %>%
  mutate(
    condition = ifelse(
      day == 0,
      "d0",
      paste(antibiotic_treatment, "_d", day, sep = "")
    )
  )

# Check required columns in metadata
if (!"#SampleID" %in% colnames(metadata)) {
  stop("ERROR: Metadata file must contain a column named '#SampleID'")
}
if (!"antibiotic" %in% colnames(metadata)) {
  stop("ERROR: Metadata file must contain a column named 'antibiotic'")
}

# Ensure sample IDs are character type
metadata$`#SampleID` <- as.character(metadata$`#SampleID`)

# Custom function to perform PBS-specific comparisons
perform_pbs_comparisons <- function(d4_metadata, d4_distance, method_out_path, current_option) {
  # Get list of all conditions
  all_conditions <- unique(d4_metadata$condition)
  
  # Extract PBS conditions
  pbs_conditions <- all_conditions[grepl("^PBS_", all_conditions)]
  
  # Output file for all results
  all_results_file <- paste0(method_out_path, current_option, "_pbs_comparison_results.txt")
  if (file.exists(all_results_file)) {
    file.remove(all_results_file) # Clear old results for this method
  }
  file.create(all_results_file)
  
  # Counter for results
  comparison_count <- 0
  
  # For each PBS condition
  for (pbs_condition in pbs_conditions) {
    cat("\nAnalyzing PBS condition:", pbs_condition, "\n")
    
    # Extract the suffix (number or identifier after PBS_)
    pbs_suffix <- sub("^PBS_", "", pbs_condition)
    
    # Find matching treatment condition with same suffix
    matching_treatments <- all_conditions[grepl(paste0("_", pbs_suffix, "$"), all_conditions) & 
                                            !grepl("^PBS_", all_conditions)]
    
    # Find other PBS conditions
    other_pbs <- setdiff(pbs_conditions, pbs_condition)
    
    # Create lists to store comparison results
    comparison_results <- list()
    
    # Compare with other PBS groups
    for (other_condition in other_pbs) {
      cat("  Comparing", pbs_condition, "vs", other_condition, "\n")
      
      # Select only the two conditions being compared
      subset_metadata <- d4_metadata[d4_metadata$condition %in% c(pbs_condition, other_condition), ]
      subset_samples <- subset_metadata$`#SampleID`
      
      # Skip if not enough samples
      if (length(subset_samples) < 2 || length(unique(subset_metadata$condition)) < 2) {
        cat("    Not enough samples for comparison. Skipping.\n")
        next
      }
      
      # Create distance matrix for these samples
      subset_distance <- as.dist(as.matrix(d4_distance)[subset_samples, subset_samples])
      
      # Run PERMANOVA
      perm_result <- adonis2(subset_distance ~ condition, data = subset_metadata, permutations = permutations_num)
      
      # Store result
      comparison_name <- paste(pbs_condition, "vs", other_condition)
      comparison_results[[comparison_name]] <- perm_result
      
      # Append to results file
      cat("\nComparison:", comparison_name, "\n", file = all_results_file, append = TRUE)
      capture.output(print(perm_result), file = all_results_file, append = TRUE)
      
      # Save detailed result to CSV
      result_df <- as.data.frame(perm_result)
      result_df$comparison <- comparison_name
      write.csv(result_df, paste0(method_out_path, current_option, "_", gsub(" ", "_", comparison_name), "_single.csv"))
      
      comparison_count <- comparison_count + 1
    }
    
    # Compare with matching treatment groups
    for (treatment_condition in matching_treatments) {
      cat("  Comparing", pbs_condition, "vs", treatment_condition, "\n")
      
      # Select only the two conditions being compared
      subset_metadata <- d4_metadata[d4_metadata$condition %in% c(pbs_condition, treatment_condition), ]
      subset_samples <- subset_metadata$`#SampleID`
      
      # Skip if not enough samples
      if (length(subset_samples) < 2 || length(unique(subset_metadata$condition)) < 2) {
        cat("    Not enough samples for comparison. Skipping.\n")
        next
      }
      
      # Create distance matrix for these samples
      subset_distance <- as.dist(as.matrix(d4_distance)[subset_samples, subset_samples])
      
      # Run PERMANOVA
      perm_result <- adonis2(subset_distance ~ condition, data = subset_metadata, permutations = permutations_num)
      
      # Store result
      comparison_name <- paste(pbs_condition, "vs", treatment_condition)
      comparison_results[[comparison_name]] <- perm_result
      
      # Append to results file
      cat("\nComparison:", comparison_name, "\n", file = all_results_file, append = TRUE)
      capture.output(print(perm_result), file = all_results_file, append = TRUE)
      
      # Save detailed result to CSV
      result_df <- as.data.frame(perm_result)
      result_df$comparison <- comparison_name
      write.csv(result_df, paste0(method_out_path, current_option, "_", gsub(" ", "_", comparison_name), "_single.csv"))
      
      comparison_count <- comparison_count + 1
    }
  }
  
  cat("\nCompleted", comparison_count, "comparisons for PBS conditions\n")
  return(comparison_count)
}

# Iterate over each option
for (current_option in options_to_process) {
  cat("\n\n========================================\n")
  cat("Processing", current_option, "\n")
  cat("========================================\n")
  
  # Update file paths for current option
  abundance_file <- paste0(data_path, current_option, "_qiime.tsv")
  
  # Check if abundance file exists
  if (!file.exists(abundance_file)) {
    cat("WARNING: Abundance file not found at:", abundance_file, "\nSkipping to next option.\n")
    next
  }
  
  # Set base results directory for current option
  option_out_path <- paste0(out_path, current_option, "/")
  if (!dir.exists(option_out_path)) {
    dir.create(option_out_path, recursive = TRUE)
  }
  
  # --- CORRECTED DATA LOADING ---
  cat("Loading abundance data for", current_option, "...\n")
  abundance_data_raw <- read.table(abundance_file, sep = "\t", header = TRUE, check.names = FALSE, comment.char = "")
  abundance_data_raw <- as.data.frame(abundance_data_raw)
  
  # Check for '#OTU ID' column
  if (!"#OTU ID" %in% colnames(abundance_data_raw)) {
    cat("ERROR: Abundance file must contain a column named '#OTU ID' with sample names.\nSkipping.\n")
    next
  }
  
  # Set sample names as rownames
  sample_names <- abundance_data_raw[["#OTU ID"]]
  abundance_data <- abundance_data_raw[, -1] # Remove the #OTU ID column
  rownames(abundance_data) <- sample_names
  
  # Convert all data to numeric
  abundance_data[] <- lapply(abundance_data, function(x) as.numeric(as.character(x)))
  
  # Remove samples (rows) that have a sum of 0, as they are uninformative
  initial_sample_count <- nrow(abundance_data)
  abundance_data <- abundance_data[rowSums(abundance_data, na.rm = TRUE) > 0, ]
  final_sample_count <- nrow(abundance_data)
  
  cat("Abundance data loaded: ", final_sample_count, "samples x", ncol(abundance_data), "OTUs\n")
  if (initial_sample_count > final_sample_count) {
    cat("Removed", initial_sample_count - final_sample_count, "samples with 0 total abundance.\n")
  }
  # --- END CORRECTED DATA LOADING ---
  
  # --- Calculate Epsilon (minimal read) ---
  # 'sweep' with MARGIN=1 works on rows (samples)
  proportions_matrix <- sweep(abundance_data, 1, rowSums(abundance_data), FUN = "/")
  proportions_matrix[is.na(proportions_matrix)] <- 0 # Handle rows with sum 0
  minimal_read <- min(proportions_matrix[proportions_matrix > 0])
  
  if (is.infinite(minimal_read)) {
    cat("WARNING: No positive values found in abundance data. Using default epsilon 1e-9.\n")
    minimal_read <- 1e-9 # A small default
  }
  cat("Calculated minimal read (epsilon):", minimal_read, "\n")
  
  # Define sample groups
  cat("\nIdentifying sample groups...\n")
  group_d0_all <- metadata$`#SampleID`[grepl("d0", metadata$`#SampleID`)]
  group_d4_all <- metadata$`#SampleID`[!grepl("d0", metadata$`#SampleID`)]
  group_d1_all <- metadata$`#SampleID`[grepl("d1", metadata$`#SampleID`)]
  
  # Intersect with rownames(abundance_data), which are now the sample IDs
  group_d0 <- intersect(group_d0_all, rownames(abundance_data))
  group_d4 <- intersect(group_d4_all, rownames(abundance_data))
  group_d1 <- intersect(group_d1_all, rownames(abundance_data))
  
  group_d4 <- setdiff(group_d4, group_d1)
  cat("Day 0 samples in data:", length(group_d0), "\n")
  cat("Day 4 samples in data:", length(group_d4), "\n")
  
  # --- START NEW TRANSFORMATION LOOP ---
  for (method_name in names(transformation_list)) {
    transform_func <- transformation_list[[method_name]]
    
    cat("\n------------------------------------\n")
    cat("Running analysis for:", current_option, "with transformation:", method_name, "\n")
    cat("------------------------------------\n")
    
    # Create method-specific output directory
    method_out_path <- paste0(option_out_path, method_name, "/")
    if (!dir.exists(method_out_path)) {
      dir.create(method_out_path, recursive = TRUE)
    }
    
    # --- 3a. Apply the transformation ---
    cat("Applying transformation...\n")
    transformed_data <- NULL
    try({
      # Apply function to each row (sample)
      # 'apply' with MARGIN=1 returns a matrix where columns are the results
      transformed_data_t <- apply(abundance_data, 1, transform_func, epsilon = minimal_read)
      
      # Transpose back so that rows are samples and columns are features
      transformed_data <- t(transformed_data_t)
      
      # Ensure row and column names are preserved
      rownames(transformed_data) <- rownames(abundance_data)
      colnames(transformed_data) <- colnames(abundance_data)
    })
    
    if (is.null(transformed_data)) {
      cat("ERROR: Transformation", method_name, "failed. Skipping.\n")
      next # Skip to next transformation
    }
    
    # Handle potential -Inf/Inf/NaN from transformations
    if (any(!is.finite(transformed_data))) {
      cat("WARNING: Non-finite values (-Inf, Inf, NaN) produced by transformation. Replacing with 0.\n")
      transformed_data[!is.finite(transformed_data)] <- 0
    }
    
    # --- Calculate Distance ---
    cat("Calculating distance matrix...\n")
    # 'dist' calculates distances between rows (samples)
    distance_matrix <- dist(transformed_data, method = "euclidean")
    cat("Distance matrix calculated successfully\n")
    
    # Run PBS-specific comparisons
    cat("Running PBS-specific comparisons...\n")
    
    # Prepare metadata for day 4 samples only
    d4_metadata <- metadata[metadata$`#SampleID` %in% group_d4, ]
    d4_samples <- d4_metadata$`#SampleID`
    
    # Skip if not enough samples
    if (length(d4_samples) < 2 || nrow(d4_metadata) < 2) {
      cat("Not enough day 4 samples to perform analysis. Skipping option.\n")
    } else {
      d4_distance <- as.dist(as.matrix(distance_matrix)[d4_samples, d4_samples])
      
      # Perform PBS-specific comparisons
      num_comparisons <- perform_pbs_comparisons(d4_metadata, d4_distance, method_out_path, current_option)
      
      if (num_comparisons == 0) {
        cat("No PBS comparisons were performed. Check if PBS conditions exist in the metadata.\n")
      }
    }
    
    cat("\nCompleted analysis for", current_option, "-", method_name, "\n")
    
  } # --- END NEW TRANSFORMATION LOOP ---
  
} # --- END OPTION LOOP ---

cat("\n\nAll analyses completed!\n")