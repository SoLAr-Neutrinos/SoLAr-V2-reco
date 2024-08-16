#!/usr/bin/env python

from argparse import ArgumentParser
from tools import params, pickle, load_charge, recal_params, fit_events, os, montecarlo

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("charge", help="Path to charge file")
    parser.add_argument(
        "--dead-areas", "-d", help="Simulate dead areas", action="store_true"
    )

    args = parser.parse_args()

    input_charge = args.charge
    params.file_label = input_charge.split("_")[-1].split(".")[0]
    params.simulate_dead_area = args.dead_areas

    charge_df = load_charge(input_charge)
    charge_df = montecarlo.rotate_coordinates(charge_df)

    if not params.simulate_dead_area:
        params.detector_x = params.quadrant_size * 8
        params.detector_y = params.quadrant_size * 8
        print(
            f"Not simulating dead areas. Detector x and y dimensions set to {params.quadrant_size * 8}"
        )
    else:
        # Cut SiPMs from the anode
        charge_df = montecarlo.cut_sipms(charge_df)
        # Cut dead chips from anode
        charge_df = montecarlo.cut_chips(charge_df)

    recal_params()

    translation = montecarlo.get_translation()
    if all([t != 0 for t in translation]):
        charge_df = montecarlo.translate_coordinates(charge_df, translation)

    charge_df = montecarlo.cut_volume(charge_df)

    # Save files
    os.makedirs(params.file_label, exist_ok=True)
    output_charge = f"{params.file_label}/charge_df_{params.file_label}.bz2"
    charge_df.to_csv(output_charge)

    metrics = montecarlo.fit_events(charge_df)

    with open(f"{params.file_label}/metrics_{params.file_label}.pkl", "wb") as f:
        pickle.dump(metrics, f)
