#!/usr/bin/env python

from argparse import ArgumentParser
from tools import params, pickle, process_root, fit_events

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("light", help="Path to light file")
    parser.add_argument("charge", help="Path to charge file")
    parser.parser.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for aditional parameters or json file containing parameters",
        required=False,
    )
    args = parser.parse_args()

    input_charge = args.charge
    input_light = args.light
    params.file_label = "_".join(input_light.split("_")[-2:]).split(".")[0]

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

    charge_df, light_df, match_dict = process_root(input_charge, input_light)

    metrics = fit_events(charge_df, light_df, match_dict)

    with open(f"{params.file_label}/metrics_{params.file_label}.pkl", "wb") as f:
        pickle.dump(metrics, f)
