#!/usr/bin/env python

from argparse import ArgumentParser
from tools import params, pickle, process_root, fit_events

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("light", help="Path to light file")
    parser.add_argument("charge", help="Path to charge file")

    args = parser.parse_args()

    input_charge = args.charge
    input_light = args.light
    params.file_label = "_".join(input_light.split("_")[-2:]).split(".")[0]

    charge_df, light_df, match_dict = process_root.process_root(
        input_charge, input_light
    )

    metrics = fit_events.fit_events(charge_df, light_df, match_dict)

    with open(f"{params.file_label}/metrics_{params.file_label}.pkl", "wb") as f:
        pickle.dump(metrics, f)
