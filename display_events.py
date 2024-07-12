#!/usr/bin/env python

from argparse import ArgumentParser

from tools import (
    os,
    pickle,
    pd,
    event_display,
    json,
    literal_eval,
    plt,
    recal_params,
    params,
    prepare_event,
)


def display_events(events, charge_df, light_df, match_dict, metrics=None):
    for event in events:
        charge_event, light_event, _ = prepare_event(
            charge_df, light_df, match_dict, event
        )

        if charge_event is None and light_event is None:
            continue

        recal_params()

        event_display(
            event_idx=event,
            charge_df=charge_event,
            light_df=light_event,
            metrics=metrics[event] if int(event) in metrics else None,
        )

        if params.show_figures:
            plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", help="Folder name for specific data file")
    parser.add_argument("events", help="Event number", type=int, nargs="+")
    parser.add_argument("--save", "-s", help="Save images", action="store_true")
    parser.add_argument(
        "--no-display", "-n", help="Don't display images", action="store_false"
    )

    args = parser.parse_args()

    params.show_figures = args.no_display
    params.file_label = args.file
    params.save_figures = args.save
    recal_params()

    # Load charge file
    charge_df = pd.read_csv(
        f"{params.file_label}/charge_df_{params.file_label}.bz2", index_col="eventID"
    )
    charge_df[charge_df.columns[9:]] = charge_df[charge_df.columns[9:]].map(
        lambda x: literal_eval(x) if isinstance(x, str) else x
    )

    # Load light file
    light_df = pd.read_csv(f"{params.file_label}/light_df_{params.file_label}.bz2")

    # Load match dictionary
    match_dict = json.load(
        open(f"{params.file_label}/match_dict_{params.file_label}.json")
    )
    match_dict = {int(key): value for key, value in match_dict.items()}

    # Load metrics file
    if os.path.isfile(f"{params.file_label}/metrics_{params.file_label}.pkl"):
        with open(f"{params.file_label}/metrics_{params.file_label}.pkl", "rb") as f:
            metrics = pickle.load(f)
    else:
        metrics = None

    display_events(args.events, charge_df, light_df, match_dict, metrics)
