#!/usr/bin/env python

from argparse import ArgumentParser

from tools import (
    event_display,
    plt,
    recal_params,
    params,
    prepare_event,
    tqdm,
    load_data,
)


def display_events(events, charge_df, light_df=None, match_dict=None, metrics=None):
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
        )

        if params.show_figures:
            plt.show()
        else:
            plt.close("all")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("folder", help="Folder name for specific data file")
    parser.add_argument("-e", "--events", help="Event number", type=int, nargs="+")
    parser.add_argument("--save", "-s", help="Save images", action="store_true")
    parser.add_argument(
        "--no-display", "-n", help="Don't display images", action="store_false"
    )

    args = parser.parse_args()

    print("\nEvent display started...")

    params.show_figures = args.no_display
    params.output_folder = args.folder
    params.save_figures = args.save

    if args.events:
        params.individual_plots = args.events

    recal_params()

    charge_df, light_df, match_dict, metrics = load_data(
        params.output_folder, return_metrics=True
    )

    display_events(params.individual_plots, charge_df, light_df, match_dict, metrics)

    print("\nEvent display finished.\n")
