
# Microbiome and Gene Ontology (GO) Analysis Toolkit

This repository provides complementary tools for analyzing microbiome data in conjunction with gene ontology (GO) clustering, PCA, and other statistical and machine learning techniques used in the manuscript "Analysis of Host and Microbiome Responses to Antibiotics". The scripts are designed to facilitate the analysis of microbiome datasets, perform clustering based on GO annotations, and visualize results through various plots.

## Repository Structure

```
.
├── Data/                                       # Folder containing datasets for all_figures_plot.py and clusters_plot.py
├── .gitignore                                  # Specifies intentionally untracked files to ignore
├── init_project.py                             # Creates the required Private/ output directory
│
├── -- Python Scripts --
├── ClusteringGO.py                             # Performs clustering of genes using GO annotations
├── all_figures_plot.py                         # Aggregates and plots major figures
├── clusters_plot.py                            # Generates plots for specific clusters
├── compores_results_analysis.py                # Analyzes results from CompoRes (Compositional microbiome Response)
├── DE_expression.py                            # Differential expression analysis using DESeq2 (pydeseq2)
├── diablo_vs_compores.py                       # Compares DIABLO multi-omics results against CompoRes output
├── groups_comparison.py                        # PCA and statistical comparisons across experimental groups
├── metagenomics_16s_comparison.py              # Compares metagenomics and 16S microbiome data
├── microbiome_pca.py                           # PCA visualization of microbiome composition
├── pairs_compores.py                           # CompoRes analysis for paired antibiotic/PBS samples
├── pairs_RF.py                                 # Random Forest classification for paired antibiotic data
├── PairsCorrEnrich.py                          # Pairwise correlation and enrichment analysis
├── pcoa_referee.py                             # PCoA visualization and helper utilities
├── random_forests.py                           # Runs Random Forest classifier on host transcriptome data
├── rat_neurons.py                              # Analysis of rat neuron gene expression data
├── time_points.py                              # Handles and analyzes temporal gene expression data
│
└── -- R Scripts --
├── ANCOM_clean.R                               # ANCOM-BC differential abundance analysis across antibiotic/treatment groups
├── DIABLO.R                                    # DIABLO multi-omics integration (5-antibiotic × 3-treatment design)
├── DIABLO_pairs.R                              # DIABLO multi-omics integration for paired (antibiotic vs. PBS) design
└── PERMANOVA_single_groups_transformations.R   # PCoA and PERMANOVA analysis with multiple CLR/transformation methods
```

## Installation

Clone the repository:

```bash
git clone https://github.com/AmirErez/Manuscript-Analysis-of-Host-and-Microbiome-Responses-to-Antibiotics.git
cd Manuscript-Analysis-of-Host-and-Microbiome-Responses-to-Antibiotics
```

Create a virtual environment and install dependencies (example using `venv` and `pip`):

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
python init_project.py
```

For R scripts, install the required packages from CRAN/Bioconductor:

```r
install.packages(c("vegan", "dplyr", "compositions"))
BiocManager::install(c("ANCOMBC", "phyloseq", "mixOmics"))
```

## Usage

You can run individual scripts based on your analysis needs:

**Figure generation:**
- `all_figures_plot.py`: Generate all main paper figures.
- `clusters_plot.py`: Plot specific gene or sample clusters.

**Transcriptomics:**
- `ClusteringGO.py`: Cluster genes by GO term annotation.
- `DE_expression.py`: Run differential expression analysis (DESeq2).
- `time_points.py`: Analyze gene expression data across time points.
- `rat_neurons.py`: Analyze rat neuron transcriptome data.

**Microbiome:**
- `microbiome_pca.py`: PCA of microbiome composition data.
- `metagenomics_16s_comparison.py`: Compare metagenomics and 16S datasets.
- `compores_results_analysis.py`: Perform statistical analysis of CompoRes output.
- `pairs_compores.py`: CompoRes analysis for antibiotic/PBS paired samples.
- `ANCOM_clean.R`: ANCOM-BC differential abundance testing.
- `PERMANOVA_single_groups_transformations.R`: PCoA and PERMANOVA with multiple transformations.

**Multi-omics integration:**
- `DIABLO.R`: DIABLO integration across the full 5×3 experimental design.
- `DIABLO_pairs.R`: DIABLO integration for paired antibiotic vs. PBS comparisons.
- `diablo_vs_compores.py`: Compare DIABLO and CompoRes results.
- `PairsCorrEnrich.py`: Pairwise correlation and enrichment between omics layers.
- `groups_comparison.py`: Statistical comparisons and PCA across experimental groups.
- `pcoa_referee.py`: PCoA analysis and visualization utilities.

**Machine learning:**
- `random_forests.py`: Run Random Forest classification on transcriptome data.
- `pairs_RF.py`: Random Forest classification for paired antibiotic data.

## Dependencies

### Python

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
goatools
wget
anytree
statsmodels
requests
gseapy
venn
pydeseq2
```

### R

```
vegan
dplyr
compositions
ANCOMBC
phyloseq
mixOmics
```

## License

[MIT](LICENSE)

## Contact

For questions or issues, feel free to contact Yehonatan at yehonatan.levin@mail.huji.ac.il.
