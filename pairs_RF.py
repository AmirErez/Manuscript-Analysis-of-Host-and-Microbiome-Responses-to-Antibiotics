from random_forests import four_way_random_forest_multiabx, plot_confusion_matrix, analyze_results, background_analysis
from groups_comparison import read_data_metadata, transform_data


def multi_abx_forest():
    data, metadata, id_to_name = read_data_metadata(remove_mitochondrial=False)
    antibiotics = metadata['Drug'].unique().tolist()
    antibiotics.remove('PBS')
    treatments = metadata['Treatment'].unique().tolist()
    run_type = "_pairs"
    data, metadata = transform_data(data, metadata, run_type, skip=True, skip_norm=True)

    metadata["group"] = metadata["Drug"]
    # ensmus_to_gene = id_to_name
    # background_id = background_analysis(data.rename(index=ensmus_to_gene).index)
    background_id = data.rename(index=id_to_name).index.to_list()
    for treat in treatments:
        sub_metadata = metadata[metadata["Treatment"] == treat]
        sub_data = data[sub_metadata['ID']]

        # # # four_way_random_forest_multiabx(sub_data, sub_metadata, treat, "group", abx=True, reps=10, path="./Private/AbxRandomForestPairs")
        # four_way_random_forest_multiabx(sub_data, sub_metadata, treat, "group", abx=True, reps=10_000, path="./Private/AbxRandomForestPairs")
        #
        plot_confusion_matrix(f"_{treat}", factor=1, path="./Private/AbxRandomForestPairs",
                              order=[f"{abx}_{treat}" for abx in ["PBS"] + antibiotics])
        exit()
        # plot_data = sub_data
        plot_data, _ = transform_data(data, metadata, run_type, skip=True)
        subplot_data = plot_data[sub_metadata['ID']]
        analyze_results(subplot_data, sub_metadata, f"_{treat}", sizes=(800, 1600), background=background_id,
        # analyze_results(subplot_data, sub_metadata, f"_{treat}", sizes=(50, 100, ), background=background_id,
        # analyze_results(subplot_data, sub_metadata, f"_{treat}", sizes=(200, 400, ), background=background_id,
        # analyze_results(subplot_data, sub_metadata, f"_{treat}", sizes=(50, 100, 200, 400, ), background=background_id,
        # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(1600,), background=background_id,
                        treat=treat, path="AbxRandomForestPairs")
        # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(200,))
        # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(100, 200, 400))
        # analyze_results(sub_data, sub_metadata, f"_{treat}", sizes=(50, 100, 200, 400, ))


if __name__ == "__main__":
    multi_abx_forest()
