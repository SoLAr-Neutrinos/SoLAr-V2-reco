#!/usr/bin/env python

import sys
from argparse import ArgumentParser

sys.path.append("..")

from tools import (
    json,
    literal_eval,
    load_charge,
    montecarlo,
    os,
    params,
    pickle,
    recal_params,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("charge", help="Path to charge file")
    parser.add_argument(
        "--dead-areas", "-d", help="Simulate dead areas", action="store_true"
    )
    parser.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for aditional parameters or json file containing parameters",
        required=False,
    )

    args = parser.parse_args()

    print("\nReconstruction started...")

    input_charge = args.charge
    params.simulate_dead_area = args.dead_areas
    params.output_folder = input_charge.split("_")[-1].split(".")[0]

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

    charge_df = load_charge(input_charge)
    charge_df = montecarlo.rotate_coordinates(charge_df)

    if not params.simulate_dead_area:
        params.detector_x = params.quadrant_size * 8
        params.detector_y = params.quadrant_size * 8
        print(
            f"Not simulating dead areas. Detector x and y dimensions reset to {params.quadrant_size * 8}"
        )
    else:
        if not params.output_folder.endswith("DA"):
            params.output_folder += "_DA"
        if (
            params.simulate_dead_area
            and not os.path.split(params.work_path)[-1] == "DA"
        ):
            params.work_path = os.path.join(params.work_path, "DA")

        # Cut SiPMs from the anode
        charge_df = montecarlo.cut_sipms(charge_df)
        # Cut dead chips from anode
        charge_df = montecarlo.cut_chips(charge_df)

    recal_params()

    translation = montecarlo.get_translation()
    if all([t != 0 for t in translation]):
        charge_df = montecarlo.translate_coordinates(charge_df, translation)

    charge_df = montecarlo.cut_volume(charge_df)

    metrics = montecarlo.fit_events(charge_df)

    output_charge = f"{params.output_folder}/charge_df_{params.output_folder}.bz2"
    metrics_file = f"{params.output_folder}/metrics_{params.output_folder}.pkl"

    # Save files
    os.makedirs(params.output_folder, exist_ok=True)
    charge_df.to_csv(output_charge)

    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)
