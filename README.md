
# Microbiome and Gene Ontology (GO) Analysis Toolkit

This repository provides complementary tools for analyzing microbiome data in conjunction with gene ontology (GO) clustering, PCA, and other statistical and machine learning techniques usd in the manuscript "Analysis of Host and Microbiome Responses to Antibiotics". The scripts are designed to facilitate the analysis of microbiome datasets, perform clustering based on GO annotations, and visualize results through various plots.

## Repository Structure

```
.
├── Data/                        # Folder containing datasets for all_figures_plot.py and clusters_plot.py
├── .gitignore                   # Specifies intentionally untracked files to ignore
├── ClusteringGO.py              # Performs clustering of genes using GO annotations
├── all_figures_plot.py          # Aggregates and plots major figures
├── clusters_plot.py             # Generates plots for specific clusters
├── compores_results_analysis.py # Analyzes results from CompoRes (Compositional microbiome Response)
├── microbiome_pca.py            # Performs PCoA on microbiome datasets
├── random_forests.py            # Runs Random Forest classifier
├── selbal_prep_16s.py           # Prepares data
├── time_points.py               # Handles and analyzes temporal data
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
```


## Usage

You can run individual scripts based on your analysis needs:

- `ClusteringGO.py`: Cluster genes by GO term.
- `microbiome_pca.py`: Reduce dimensionality of microbiome features using PCoA.
- `random_forests.py`: Run classification using a Random Forest model.
- `selbal_prep_16s.py`: Preprocessing.
- `all_figures_plot.py`: Generate summary plots.
- `clusters_plot.py`: Plot specific gene or sample clusters.
- `time_points.py`: Analyze gene expression data across time points.
- `compores_results_analysis.py`: Perform statistical analysis of CompoRes output.

## Dependencies

required packages:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
goatools
json
os
wget
collections
typing
anytree
statsmodels
requests
gseapy
venn
```

## License

[MIT](LICENSE)

## Contact

For questions or issues, feel free to contact Yehonatan at yehonatan.levin@mail.huji.ac.il.
