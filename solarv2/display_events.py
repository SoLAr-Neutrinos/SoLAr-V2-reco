#!/usr/bin/env python

from argparse import ArgumentParser

from .tools import (
    event_display,
    plt,
    recal_params,
    params,
    prepare_event,
    tqdm,
    load_data,
    load_params,
)


def display_events(
    events, charge_df, light_df=None, match_dict=None, metrics=None, dq=False, **kwargs
):
    for event in tqdm(events, desc="Event display"):
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


def main(folder, events=None, save=False, no_display=True, dq=False, parameters=None):
    print("\nEvent display started...")

    params.show_figures = no_display
    params.output_folder = folder
    params.save_figures = save

    kwargs = load_params(parameters)

    if events:
        params.individual_plots = events

    recal_params()

    charge_df, light_df, match_dict, metrics = load_data(
        params.output_folder, return_metrics=True
    )

    if params.lifetime>0:
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
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for aditional parameters or json file containing parameters",
        required=False,
    )
    parser.add_argument("-q", "--dq", help="Make dQ/dx plots", action="store_true")

    args = parser.parse_args()

    main(**vars(args))
