#!/usr/bin/env python

import sys
from argparse import ArgumentParser

sys.path.append("..")

from tools import (event_display, load_data, params, plt, prepare_event,
                   recal_params, tqdm)


def display_events(events, charge_df, light_df=None, match_dict=None, metrics=None):
    for event in tqdm(events):
        charge_event, light_event, _ = prepare_event(
            event, charge_df, light_df, match_dict
        )

        if charge_event is None and light_event is None:
            continue

        event_display(
            event_idx=event,
            charge_df=charge_event,
            light_df=light_event,
            metrics=metrics[event] if int(event) in metrics else None,
        )

        if params.show_figures:
            plt.show()
        else:
            plt.close("all")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", help="Folder name for specific data file")
    parser.add_argument("-e", "--events", help="Event number", type=int, nargs="+")
    parser.add_argument("--save", "-s", help="Save images", action="store_true")
    parser.add_argument(
        "--no-display", "-n", help="Don't display images", action="store_false"
    )
    parser.add_argument(
        "--dead-areas", "-d", help="Simulate dead areas", action="store_true"
    )

    args = parser.parse_args()

    params.show_figures = args.no_display
    params.output_folder = args.file
    params.save_figures = args.save
    params.simulate_dead_area = args.dead_areas
    if args.events:
        params.individual_plots = args.events

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

    if args.events:
        params.individual_plots = args.events

    recal_params()

    charge_df, light_df, match_dict, metrics = load_data(
        params.output_folder, return_metrics=True
    )

    display_events(params.individual_plots, charge_df, light_df, match_dict, metrics)
