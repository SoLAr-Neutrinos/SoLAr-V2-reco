#!/usr/bin/env python

from tools import *

from argparse import ArgumentParser


def main(metrics):
    warnings.filterwarnings("ignore", category=Warning, module="numpy")

    # 1
    plot_track_stats(
        metrics,
    )
    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 2
    for event_idx in tqdm(params.individual_plots, leave=False):
        if event_idx in metrics:
            for track_idx, values in metrics[event_idx].items():
                if not isinstance(track_idx, str) and track_idx > 0:
                    dQ_array = values["dQ"]
                    dh = values["dx"]
                    plot_dQ(dQ_array, event_idx, track_idx, dh, interpolate=False)

                    if params.show_figures:
                        plt.show()
                    else:
                        plt.close("all")

    # 3
    plot_light_geo_stats(metrics)

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 4
    light_vs_charge(metrics)

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 5
    plot_voxel_data(metrics)

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 6
    plot_light_fit_stats(metrics)

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    warnings.filterwarnings("default", category=Warning)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--folder",
        "-f",
        help="Folder name for specific metrics file",
        default="combined",
    )
    parser.add_argument(
        "--filter", help="Tag number of filter file within folder", default=None
    )
    parser.add_argument("--save", "-s", help="Save images", action="store_true")
    parser.add_argument(
        "--display", "-d", help="Display images (not recomended)", action="store_true"
    )

    args = parser.parse_args()

    params.file_label = args.folder
    filter_tag = args.filter
    params.show_figures = args.display
    params.save_figures = args.save

    if filter_tag is not None:
        filter_file = f"{params.file_label}/filter_parameters_{filter_tag}.json"

    metrics_file = f"{params.file_label}/metrics_{params.file_label}.pkl"

    recal_params()

    if params.file_label == "combined" and not os.path.isfile(metrics_file):
        metrics = combine_metrics()
    elif not os.path.isdir(params.file_label):
        print(f"Folder {params.file_label} not found. Exiting...")
        exit(1)
    else:
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)

    if filter_tag is not None and os.path.isfile(filter_file):
        with open(filter_file, "r") as f:
            filter_settings = json.load(f)
            params.__dict__.update(filter_settings)

    print(len(metrics), "metrics loaded")
    metrics = filter_metrics(metrics)

    main(metrics)
