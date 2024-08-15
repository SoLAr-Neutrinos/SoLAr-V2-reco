#!/usr/bin/env python

from tools import *

from argparse import ArgumentParser
import inspect


def main(metrics, **kwargs):
    warnings.filterwarnings("ignore", category=Warning, module="numpy")

    methods = [
        plot_track_stats,
        plot_dQ,
        plot_light_geo_stats,
        light_vs_charge,
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

    # 2 - Individual dQ/dx plots
    print("\nPlotting individual dQ/dx plots\n")
    for event_idx in tqdm(params.individual_plots, leave=False):
        if event_idx in metrics:
            for track_idx, values in metrics[event_idx].items():
                if not isinstance(track_idx, str) and track_idx > 0:
                    dQ_array = values["dQ"]
                    dh = values["dx"]
                    plot_dQ(
                        dQ_array, event_idx, track_idx, dh, **method_kwargs["plot_dQ"]
                    )

                    if params.show_figures:
                        plt.show()
                    else:
                        plt.close("all")

    # 3 - Light geometrical properties to charge tracks statistics
    print("\nPlotting light geometrical properties to charge tracks statistics\n")
    plot_light_geo_stats(metrics, **method_kwargs["plot_light_geo_stats"])

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 4 - Event level light vs charge statistics
    print("\nPlotting event level light vs charge statistics\n")
    light_vs_charge(metrics, **method_kwargs["light_vs_charge"])

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 5 - Voxelized charge and light data
    print("\nPlotting voxelized charge and light data\n")
    plot_voxel_data(metrics, **method_kwargs["plot_voxel_data"])

    if params.show_figures:
        plt.show()
    else:
        plt.close("all")

    # 6 - Light fit statistics
    print("\nPlotting light fit statistics\n")
    plot_light_fit_stats(metrics, **method_kwargs["plot_light_fit_stats"])

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
    parser.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for aditional parameters",
        required=False,
    )

    args = parser.parse_args()

    params.file_label = args.folder
    filter_tag = args.filter
    params.show_figures = args.display
    params.save_figures = args.save

    kwargs = {}
    if args.parameters is not None:
        # Check if parameters are provided in a JSON file
        if (
            len(args.parameters) == 1
            and args.parameters[0].endswith(".json")
            and os.path.isfile(args.parameters[0])
        ):
            with open(args.parameters[0], "r") as f:
                param = json.load(f)
        else:
            # Convert command line parameters to dictionary
            param = {
                key: value
                for param in args.parameters
                for key, value in [param.split("=") if "=" in param else (param, None)]
            }

        # Now process the parameters in a single for loop
        for key, value in param.items():
            if key in params.__dict__:
                params.__dict__[key] = value
            else:
                kwargs[key] = value

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

    main(metrics, **kwargs)