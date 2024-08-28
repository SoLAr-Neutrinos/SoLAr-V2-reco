#!/usr/bin/env python

from argparse import ArgumentParser

from tools import (
    fit_events,
    json,
    literal_eval,
    load_data,
    os,
    params,
    pickle,
    process_root,
)

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
    params.reload_files = bool(args.charge and args.light)

    # Validate arguments
    if params.reload_files:
        if not args.charge or not args.light:
            parser.error(
                "Both '--charge' and '--light' arguments are required when not using the '--folder' argument."
            )
    else:
        if not args.folder:
            parser.error(
                "The '--folder' argument is required when not using the '--charge' and '--light' arguments."
            )

    print("\nReconstruction started...\n")

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
                try:
                    params.__dict__[key] = (
                        literal_eval(value)
                        if not isinstance(params.__dict__[key], str)
                        else value
                    )
                except ValueError:
                    params.__dict__[key] = value
            # else:
            #     try:
            #         kwargs[key] = literal_eval(value)
            #     except ValueError:
            #         kwargs[key] = value

    if params.reload_files:
        input_charge = args.charge
        input_light = args.light
        params.file_label = "_".join(input_light.split("_")[-2:]).split(".")[0]
        charge_df, light_df, match_dict = process_root(input_charge, input_light)
    else:
        params.file_label = os.path.split(args.folder)[-1]
        # Load charge file
        charge_df, light_df, match_dict = load_data(args.folder, return_metrics=False)

    metrics = fit_events(charge_df, light_df, match_dict)

    output_path = os.path.join(params.work_path, f"{params.file_label}")
    os.makedirs(output_path, exist_ok=True)
    metrics_file = os.path.join(output_path, f"metrics_{params.file_label}.pkl")
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)

    print("\nReconstruction finished.\n")
