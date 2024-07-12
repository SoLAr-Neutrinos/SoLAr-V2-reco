#!/usr/bin/env python

from argparse import ArgumentParser

from .methods import (
    cluster_hits,
    fit_hit_clusters,
    json,
    light_geometry,
    literal_eval,
    params,
    pd,
    pickle,
    prepare_event,
    recal_params,
    voxelize_hits,
)


def fit_events(charge_df, light_df, match_dict):
    metrics = {}
    for event in charge_df.index:
        charge_event, light_event, mask = prepare_event(
            charge_df, light_df, match_dict, event
        )

        if charge_event is None and light_event is None:
            continue

        # Create a design matrix
        labels = cluster_hits(charge_event[["x", "y", "z"]].to_numpy())
        # Fit clusters
        metrics[event] = fit_hit_clusters(
            charge_event[["x", "y", "z"]].to_numpy(),
            charge_event["q"].to_numpy(),
            labels,
        )

        # Light to track geometry metrics
        track_lines = []
        for track_idx, values in metrics[event].items():
            if "Fit_line" not in values:
                continue
            values["SiPM"] = light_geometry(
                track_line=values["Fit_line"],
                track_norm=values["Fit_norm"],
                sipm_df=light_event,
                light_variable=params.light_variable,
            )
            track_lines.append(values["Fit_line"])

        # Light and charge voxelization and fitting
        metrics[event]["SiPM"] = voxelize_hits(
            charge_event,
            light_event,
            params.light_variable,
            charge_lines=track_lines,
        )

        metrics[event][
            "Pixel_mask"
        ] = mask.to_numpy()  # Save masks to original dataframe for reference
        metrics[event]["Total_light"] = light_event[params.light_variable].sum()
        metrics[event]["Total_charge"] = charge_event["q"].sum()
    return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", help="Folder name for folder containing data files")
    parser.add_argument(
        "--no-display", "-n", help="Don't display images", action="store_false"
    )

    args = parser.parse_args()

    params.file_label = args.file
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

    metrics = fit_events(charge_df, light_df, match_dict)

    with open(f"{params.file_label}/metrics_{params.file_label}.pkl", "wb") as f:
        pickle.dump(metrics, f)
