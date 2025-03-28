#!/usr/bin/env python

from argparse import ArgumentParser

from .tools import (
    fit_events,
    load_params,
    load_data,
    os,
    params,
    pickle,
    process_root,
)


def main(charge, light, folder, parameters=None):
    params.reload_files = bool(charge and light)

    print("\nReconstruction started...")

    kwargs = load_params(parameters)

    if params.reload_files:
        input_charge = charge
        input_light = light
        params.output_folder = "_".join(input_light.split("_")[-2:]).split(".")[0]
        charge_df, light_df, match_dict = process_root(input_charge, input_light)
    else:
        params.output_folder = os.path.split(folder)[-1]
        # Load charge file
        charge_df, light_df, match_dict = load_data(folder, return_metrics=False)

    metrics = fit_events(charge_df, light_df, match_dict)

    output_path = os.path.join(params.work_path, f"{params.output_folder}")
    os.makedirs(output_path, exist_ok=True)

    metrics_file = os.path.join(output_path, f"metrics_{params.output_folder}.pkl")
    if params.lifetime > 0:
        metrics_file = metrics_file.replace(".pkl", f"_lt{params.lifetime:.3}.pkl")

    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)

    print("\nReconstruction finished.\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--charge", help="Path to charge file")
    parser.add_argument("-l", "--light", help="Path to light file")
    parser.add_argument("-f", "--folder", help="Folder name for processing")
    parser.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for additional parameters or json file containing parameters",
        required=False,
    )

    args = parser.parse_args()

    # Validate arguments
    if args.reload_files:
        if not args.charge or not args.light:
            parser.error("Both '--charge' and '--light' arguments are required when not using the '--folder' argument.")
    else:
        if not args.folder:
            parser.error("The '--folder' argument is required when not using the '--charge' and '--light' arguments.")

    main(args)
