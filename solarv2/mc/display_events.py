#!/usr/bin/env python

import sys
from argparse import ArgumentParser

from ..tools import (
    os,
    event_display,
    load_data,
    params,
    plt,
    prepare_event,
    recal_params,
    load_params,
    tqdm,
)


def display_events(
    events, charge_df, light_df=None, match_dict=None, metrics=None, **kwargs
):
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
            **kwargs,
        )

        if dq:
            for track_idx, values in metrics[event].items():
                if not isinstance(track_idx, str) and track_idx > 0:
                    dQ_series = values["dQ"]
                    dx_series = values["dx"]
                    plot_dQ(
                        dQ_series=dQ_series,
                        dx_series=dx_series,
                        event_idx=event,
                        track_idx=track_idx,
                    )

                    if params.show_figures:
                        plt.show(block=False)
                    else:
                        plt.close("all")

        if params.show_figures:
            plt.show(block=False)
        else:
            plt.close("all")


def main(folder, events, no_display=True, save=False, dead_areas=False, dq=False, parameters=None):
    print("\nEvent display started...")

    params.show_figures = no_display
    params.output_folder = folder
    params.save_figures = save
    params.simulate_dead_area = dead_areas
    if events:
        params.individual_plots = events

    if params.simulate_dead_area:
        params.detector_x = params.quadrant_size * 4
        params.detector_y = params.quadrant_size * 5
        print(
            f"Simulating dead areas. Detector x and y dimensions reset to ({params.detector_x}, {params.detector_y})"
        )
        if not params.output_folder.endswith("DA"):
            params.output_folder += "_DA"
        if (
            params.simulate_dead_area
            and not os.path.split(params.work_path)[-1] == "DA"
        ):
            params.work_path = os.path.join(params.work_path, "DA")
    else:
        params.detector_x = params.quadrant_size * 8
        params.detector_y = params.quadrant_size * 8
        print(
            f"Not simulating dead areas. Detector x and y dimensions reset to ({params.detector_x}, {params.detector_y})"
        )

    if events:
        params.individual_plots = events

    kwargs = load_params(parameters)

    recal_params()

    search_path = os.path.join(params.work_path, f"{params.output_folder}")

    charge_df, light_df, match_dict, metrics = load_data(
        search_path, return_metrics=True
    )

    if params.lifetime > 0:
        metrics = apply_lifetime(metrics)

    display_events(
        params.individual_plots, charge_df, light_df, match_dict, metrics, dq=dq, **kwargs
    )

    print("\nEvent display finished.\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("folder", help="Folder name for specific data file")
    parser.add_argument("-e", "--events", help="Event number", type=int, nargs="+")
    parser.add_argument("-s", "--save", help="Save images", action="store_true")
    parser.add_argument(
        "-n", "--no-display", help="Don't display images", action="store_false"
    )
    parser.add_argument(
        "-d", "--dead-areas", help="Simulate dead areas", action="store_true"
    )
    parser.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for aditional parameters or json file containing parameters",
        required=False,
    )
    parser.add_argument("-q", "--dq", help="Make dQ/dx plots", action="store_true")

    args = parser.parse_args()

    main(**vars(args))
