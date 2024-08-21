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
    tqdm,
)


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

    args = parser.parse_args()

    params.show_figures = args.no_display
    params.file_label = args.file
    params.save_figures = args.save

    if args.events:
        params.individual_plots = args.events

    recal_params()

    # Load charge file
    charge_input = f"{params.file_label}/charge_df_{params.file_label}.bz2"
    charge_df = pd.read_csv(charge_input, index_col="eventID")
    charge_df[charge_df.columns] = charge_df[charge_df.columns].map(
        lambda x: (
            literal_eval(x)
            if isinstance(x, str) and (x[0] == "[" or x[0] == "(")
            else x
        )
    )

    # Load light file
    light_input = f"{params.file_label}/light_df_{params.file_label}.bz2"
    light_df = None
    match_dict = None
    if os.path.isfile(light_input):
        light_df = pd.read_csv()

        # Load match dictionary
        match_dict = json.load(
            open(f"{params.file_label}/match_dict_{params.file_label}.json")
        )
        match_dict = {int(key): value for key, value in match_dict.items()}

    # Load metrics file
    input_metrics = f"{params.file_label}/metrics_{params.file_label}.pkl"
    if os.path.isfile(input_metrics):
        with open(input_metrics, "rb") as f:
            metrics = pickle.load(f)
    else:
        metrics = None

    display_events(params.individual_plots, charge_df, light_df, match_dict, metrics)
