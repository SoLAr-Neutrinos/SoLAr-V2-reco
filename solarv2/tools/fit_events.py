#!/usr/bin/env python

import json
import os
import pickle


def fit_events(charge_df, light_df, match_dict):
    metrics = {}
    for event in tqdm(charge_df.index, desc="Fitting events"):
        charge_event, light_event, mask = prepare_event(event, charge_df, light_df, match_dict)

        if charge_event is None and light_event is None:
            continue
        if len(charge_event) <= 2:
            # tqdm.write(f"Event {event} has 2 or less entries. Skipping...")
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
            if isinstance(track_idx, str) or track_idx <= 0:
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

        metrics[event]["Pixel_mask"] = mask.to_numpy()  # Save masks to original dataframe for reference
        metrics[event]["Total_light"] = light_event[params.light_variable].sum()
        metrics[event]["Total_charge"] = charge_event["q"].sum()
    return metrics


if __name__ == "__main__":
    from argparse import ArgumentParser

    from methods import (
        cluster_hits,
        fit_hit_clusters,
        light_geometry,
        literal_eval,
        params,
        pd,
        tqdm,
        prepare_event,
        recal_params,
        tqdm,
        voxelize_hits,
    )

    parser = ArgumentParser()
    parser.add_argument("file", help="Folder name for folder containing data files")
    # parser.add_argument(
    #     "--no-display", "-n", help="Don't display images", action="store_false"
    # )

    args = parser.parse_args()

    params.output_folder = args.file
    recal_params()

    # Load charge file
    charge_df = pd.read_pickle(os.path.join(params.work_path, params.output_folder, "charge_df_{params.output_folder}.pkl"))

    # Load light file
    light_df = pd.read_pickle(
        os.path.join(
            params.work_path,
            params.output_folder,
            f"light_df_{params.output_folder}.pkl",
        )
    )

    # Load match dictionary
    match_dict = json.load(
        open(
            os.path.join(
                params.work_path,
                params.output_folder,
                f"match_dict_{params.output_folder}.json",
            )
        )
    )
    match_dict = {int(key): value for key, value in match_dict.items()}

    metrics = fit_events(charge_df, light_df, match_dict)

    metrics_file = os.path.join(
        params.work_path,
        params.output_folder,
        f"metrics_{params.output_folder}.pkl",
    )
    if params.lifetime > 0:
        metrics_file = metrics_file.replace(".pkl", f"_lt{params.lifetime:.3}.pkl")

    with open(
        metrics_file,
        "wb",
    ) as f:
        pickle.dump(metrics, f)


else:
    from .methods import (
        cluster_hits,
        fit_hit_clusters,
        light_geometry,
        literal_eval,
        params,
        pd,
        tqdm,
        prepare_event,
        recal_params,
        voxelize_hits,
    )
