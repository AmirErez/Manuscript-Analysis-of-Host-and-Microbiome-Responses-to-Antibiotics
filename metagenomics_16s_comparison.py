import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from scipy.stats import pearsonr, gaussian_kde
from compores_results_analysis import read_and_print_pkl
from groups_comparison import read_data_metadata


def get_compores_results(abx, path, treat, opt):
    data = read_and_print_pkl(path + rf'/{abx}-{treat}-feces/{opt}/mean_log_p_value.pkl')
    # data = read_and_print_pkl(path + rf'/{abx}-{treat}-feces/pairs/mean_log_p_value.pkl')
    data = data[f"{abx}-{treat}-feces"]
    index = read_and_print_pkl(path + rf'/{abx}-{treat}-feces/{opt}/response_index.pkl')
    # index = read_and_print_pkl(path + rf'/{abx}-{treat}-feces/pairs/response_index.pkl')
    column_names = ['gene', 'correlation']
    compores_results_spf = pd.DataFrame({column_names[0]: index, column_names[1]: data})
    # compores_results_spf[f'{column_names[1]}_p'] = np.exp(compores_results_spf[column_names[1]])
    compores_results_spf[f'{column_names[1]}_p'] = np.exp(-compores_results_spf[column_names[1]])
    compores_results_spf["genes_name"] = compores_results_spf[column_names[0]].str.split('_').str[-1]
    return compores_results_spf


significant_path = fr"/Users/yonchlevin/Desktop/ErezLab/MouseAbxBel/CompoResults/metagenomicsVS16s"

path_meta = significant_path + fr"/metagenomics"
path_16s = significant_path + fr"/16S"

abx_dict = {
    "clock": "Van",
    "viral": "Neo",
    "all": "Van",
}
data, metadata, id_to_name = read_data_metadata(remove_mitochondrial=False)

# Loop for options and comparisons
for log in [True, False]:
    for opt in ["pairs", "CLR"]:
        for comp in ["clock", "viral", "all"]:
            compores_results_metagenomics = get_compores_results(abx_dict[comp], path_meta + f"/{comp}", "pairs", opt)
            compores_results_16s = get_compores_results(abx_dict[comp], path_16s + f"/{comp}", "pairs", opt)

            # Merge results
            merged = pd.merge(
                compores_results_metagenomics,
                compores_results_16s,
                on="genes_name",
                how="inner",
                suffixes=("_meta", "_16s"),
            )
            merged["comparison"] = comp
            merged["gene id"] = merged["genes_name"].map(id_to_name)

            # Get p-values
            pcol_meta = 'correlation_p_meta' if 'correlation_p_meta' in merged.columns else 'correlation_p'
            pcol_16s = 'correlation_p_16s' if 'correlation_p_16s' in merged.columns else 'correlation_p'

            orig_x = pd.to_numeric(merged.get(pcol_meta), errors='coerce')
            orig_y = pd.to_numeric(merged.get(pcol_16s), errors='coerce')

            # Filter valid
            valid_idx = orig_x.notna() & orig_y.notna()
            if len(valid_idx) != len(merged):
                print("missing: ", len(valid_idx), len(merged))
            plot_df = merged.loc[valid_idx].copy()
            orig_x = orig_x.loc[valid_idx]
            orig_y = orig_y.loc[valid_idx]

            # Log transform data
            tiny = 1e-300
            orig_x = orig_x.clip(lower=tiny)
            orig_y = orig_y.clip(lower=tiny)
            if log:
                tx = -np.log2(orig_x)
                ty = -np.log2(orig_y)
            else:
                tx = orig_x
                ty = orig_y

            # --- DENSITY CALCULATION ---
            # Calculate point density using Gaussian KDE
            xy = np.vstack([tx, ty])
            if xy.shape[1] > 1:  # Only calculate if we have enough points
                z = gaussian_kde(xy)(xy)
                # Sort points by density so densest points are plotted last (on top)
                idx = z.argsort()
                tx_sorted, ty_sorted, z_sorted = tx.iloc[idx], ty.iloc[idx], z[idx]
            else:
                # Fallback for empty or single point
                tx_sorted, ty_sorted, z_sorted = tx, ty, np.ones_like(tx)

            # Experiment with both linear and log density scales
            for scale in ['linear', 'log']:
                plt.figure(figsize=(6, 6))

                # Choose normalization and colormap
                norm = None
                if scale == 'log':
                    # Use LogNorm for color mapping (protect against 0 in density if any)
                    norm = mcolors.LogNorm(vmin=np.max([z_sorted.min(), 1e-10]), vmax=z_sorted.max())

                scatter = plt.scatter(tx_sorted, ty_sorted, c=z_sorted, s=30,
                                      alpha=0.8, cmap='viridis', norm=norm, edgecolor='none')

                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label(f'Density ({scale} scale)')


                if log:
                    # Plot diagonal
                    thr_p = 0.05
                    thr_log = -np.log2(thr_p)
                    mx = max(tx.max(), ty.max(), thr_log)
                    plt.plot([0, mx], [0, mx], linestyle='--', color='gray', linewidth=1)

                    # Threshold lines
                    plt.axvline(thr_log, color='red', linestyle='--', linewidth=1)
                    plt.axhline(thr_log, color='red', linestyle='--', linewidth=1)
                    plt.xlabel('-log2(p-value) [CompoRes on metagenomics]')
                    plt.ylabel('-log2(p-value) [CompoRes on 16S]')
                else:
                    # Plot diagonal
                    thr_p = 0.05
                    mx = max(tx.max(), ty.max(), thr_p)
                    plt.plot([0, mx], [0, mx], linestyle='--', color='gray', linewidth=1)

                    # Threshold lines
                    plt.axvline(thr_p, color='red', linestyle='--', linewidth=1)
                    plt.axhline(thr_p, color='red', linestyle='--', linewidth=1)

                    plt.xlabel('p-value [CompoRes on metagenomics]')
                    plt.ylabel('p-value [CompoRes on 16S]')

                plt.title(f'Metagenomics vs 16S ({comp}, {scale} density) — n={len(plot_df)}')

                # Stats annotation
                r, pval = pearsonr(tx.values, ty.values)
                xpos = 0.05 * mx
                ypos = 0.95 * mx
                plt.text(xpos, ypos, f"Pearson r={r:.3f}\n p={pval:.2g}",
                         fontsize=8, color='black',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                # Gene annotation (using original unsorted indices to map back to plot_df)
                if comp != "all":
                    annotate_mask = (orig_x < thr_p) & (orig_y < thr_p)
                    for idx_row, row in plot_df[annotate_mask].iterrows():
                        gx = float(tx.loc[idx_row])
                        gy = float(ty.loc[idx_row])
                        gid = row.get('gene id', row.get('genes_name'))
                        plt.text(gx, gy, str(gid), fontsize=6, alpha=0.9, color='black')

                plt.tight_layout()
                out_dir = './Private/seq_comp'
                os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/metagenomics_vs_16s_scatter_{comp}_{opt}_{scale}{'_pval' if not log else ''}.png"
                plt.savefig(out_path, dpi=300)
                # plt.show() # Optional: comment out if running in batch to avoid window popups
                plt.close()
            # save the merged table
            merged.to_csv(f"{out_dir}/metagenomics_vs_16s_scatter_{comp}_{opt}.csv")
