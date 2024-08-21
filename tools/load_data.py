#!/usr/bin/env python

from .methods import ak, json, literal_eval, np, os, pd, pickle


# Load charge and light files, match dictionary and optionally metrics
def load_data(folder, return_metrics=False):
    charge_input = f"{folder}/charge_df_{folder}.bz2"
    charge_df = pd.read_csv(charge_input, index_col="eventID")
    charge_df[charge_df.columns] = charge_df[charge_df.columns].map(
        lambda x: (
            literal_eval(x)
            if isinstance(x, str) and (x[0] == "[" or x[0] == "(")
            else x
        )
    )

    # Load light file
    light_input = f"{folder}/light_df_{folder}.bz2"
    light_df = None
    if os.path.isfile(light_input):
        light_df = pd.read_csv(light_input, index_col=0)

    # Load match dictionary
    match_input = f"{folder}/match_dict_{folder}.json"
    match_dict = None
    if os.path.isfile(match_input):
        with open(match_input, "r") as f:
            match_dict = json.load(f)

        match_dict = {int(key): value for key, value in match_dict.items()}
        # Remove charge events without associated light event
        charge_df = charge_df.loc[list(match_dict.keys())]
        # Remove light events without charge event match
        light_events = np.unique(ak.flatten(match_dict.values()))
        light_df = light_df[light_df["event"].isin(light_events)]

    metrics_file = f"{folder}/metrics_{folder}.pkl"
    metrics = None
    if return_metrics and os.path.isfile(metrics_file):
        metrics_file = f"{folder}/metrics_{folder}.pkl"
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)

        return charge_df, light_df, match_dict, metrics
    else:
        return charge_df, light_df, match_dict
