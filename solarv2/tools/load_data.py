#!/usr/bin/env python

from .methods import ak, json, literal_eval, np, os, pd, pickle, params


# Load charge and light files, match dictionary and optionally metrics
def load_data(folder, return_metrics=False):
    label = os.path.split(folder)[-1]
    charge_input = os.path.join(folder, f"charge_df_{label}.pkl")
    charge_df = pd.read_pickle(charge_input)

    # Load light file
    light_input = os.path.join(folder, f"light_df_{label}.pkl")
    light_df = None
    if os.path.isfile(light_input):
        light_df = pd.read_pickle(light_input)

    # Load match dictionary
    match_input = os.path.join(folder, f"match_dict_{label}.json")
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

    metrics_file = os.path.join(folder, f"metrics_{label}.pkl")
    if params.lifetime > 0:
        metrics_file = metrics_file.replace(".pkl", f"_lt{params.lifetime:.3}.pkl")

    metrics = None
    if return_metrics and os.path.isfile(metrics_file):
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)

        return charge_df, light_df, match_dict, metrics
    else:
        return charge_df, light_df, match_dict
