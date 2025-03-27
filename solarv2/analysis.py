#!/usr/bin/env python

from .tools import *

from argparse import ArgumentParser
import inspect


def analysis(metrics, **kwargs):
    warnings.filterwarnings("ignore", category=Warning, module="numpy")

    methods = [
        plot_track_stats,
        plot_track_angles,
        plot_dQ,
        plot_light_geo_stats,
        plot_light_vs_charge,
        plot_voxel_data,
        plot_light_fit_stats,
    ]
    method_kwargs = {}
    for method in methods:
        method_kwargs[method.__qualname__] = {}
        sig = inspect.signature(method)
        # Prepare the arguments for the function
        for key, param in sig.parameters.items():
            if key in kwargs:
                try:
                    method_kwargs[method.__qualname__][key] = literal_eval(kwargs[key])
                except ValueError:
                    method_kwargs[method.__qualname__][key] = kwargs[key]

    # 1 - Track statistical plots
    print("\nPlotting track statistics\n")
    plot_track_stats(metrics, **method_kwargs["plot_track_stats"])
    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 2 - Track angular distribution plots
    print("\nPlotting track angular distribution\n")
    plot_track_angles(metrics, **method_kwargs["plot_track_angles"])
    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 3 - Individual dQ/dx plots
    print("\nPlotting individual dQ/dx plots\n")
    for event_idx in tqdm(params.individual_plots, leave=False, desc="Individual dQ/dx"):
        if event_idx in metrics:
            for track_idx, values in metrics[event_idx].items():
                if not isinstance(track_idx, str) and track_idx > 0:
                    dQ_series = values["dQ"]
                    dx_series = values["dx"]
                    plot_dQ(
                        dQ_series=dQ_series,
                        dx_series=dx_series,
                        event_idx=event_idx,
                        track_idx=track_idx,
                        **method_kwargs["plot_dQ"],
                    )

                    if params.show_figures:
                        plt.show()
                    else:
                        plt.close("all")

    # 4 - Light geometrical properties to charge tracks statistics
    print("\nPlotting light geometrical properties to charge tracks statistics\n")
    plot_light_geo_stats(metrics, **method_kwargs["plot_light_geo_stats"])

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 5 - Event level light vs charge statistics
    print("\nPlotting event level light vs charge statistics\n")
    plot_light_vs_charge(metrics, **method_kwargs["plot_light_vs_charge"])

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 6 - Voxelized charge and light data
    print("\nPlotting voxelized charge and light data\n")
    plot_voxel_data(metrics, **method_kwargs["plot_voxel_data"])

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 7 - Light fit statistics
    print("\nPlotting light fit statistics\n")
    plot_light_fit_stats(metrics, **method_kwargs["plot_light_fit_stats"])

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    warnings.filterwarnings("default", category=Warning)


def main(folder, filter=None, display=False, save=True, parameters=None):
    print("\nAnalysis started...")

    params.output_folder = folder
    filter_tag = filter
    params.show_figures = display
    params.save_figures = save

    kwargs = load_params(parameters)

    search_path = os.path.join(params.work_path, f"{params.output_folder}")

    if filter_tag is not None:
        filter_file = os.path.join(search_path, f"filter_parameters_{filter_tag}.json")

    metrics_file = os.path.join(search_path, f"metrics_{params.output_folder}.pkl")

    recal_params()

    if "combined" in params.output_folder and not os.path.isfile(metrics_file):
        metrics = combine_metrics()
    elif not os.path.isdir(search_path):
        print(f"Folder {params.output_folder} not found. Exiting...")
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

    if params.lifetime_correction>0:
        metrics = apply_lifetime(metrics)

    analysis(metrics, **kwargs)

    print("\nAnalysis finished.\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "folder",
        help="Folder name for specific metrics file",
        default="combined",
        nargs="?",
    )
    parser.add_argument("-f", "--filter", help="Tag number of filter file within folder", default=None)
    # parser.add_argument("--save", "-s", help="Save images", action="store_true")
    parser.add_argument("-d", "--display", help="Display images (not recomended)", action="store_true")
    parser.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for aditional parameters or json file containing parameters",
        required=False,
    )

    args = parser.parse_args()
