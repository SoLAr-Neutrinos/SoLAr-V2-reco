#!/usr/bin/env python

import pickle

import awkward as ak
import numpy as np
import uproot


def metrics_to_root(metrics, output_file):
    # Extract data
    event_indices = []
    track_indices = []
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    fit_norms = []
    fit_mses = []
    ransac_scores = []
    q_effs = []
    dQs = []
    dxs = []

    fit_labels_list = []
    total_charges = []
    event_indices_events = []

    for event_idx, (event_key, event_value) in enumerate(metrics.items()):
        for track_key, track_value in event_value.items():
            if isinstance(track_key, np.int64):
                event_indices.append(int(event_key))
                track_indices.append(track_key)
                x.append(track_value["Fit_line"].point[0])
                y.append(track_value["Fit_line"].point[1])
                z.append(track_value["Fit_line"].point[2])
                vx.append(track_value["Fit_line"].direction[0])
                vy.append(track_value["Fit_line"].direction[1])
                vz.append(track_value["Fit_line"].direction[2])
                fit_norms.append(track_value["Fit_norm"])
                fit_mses.append(track_value["Fit_mse"])
                ransac_scores.append(track_value["RANSAC_score"])
                q_effs.append(track_value["q_eff"])
                dQs.append(track_value["dQ"])
                dxs.append(track_value["dx"])
            else:
                fit_labels_list.append(event_value["Fit_labels"])
                total_charges.append(event_value["Total_charge"])
                event_indices_events.append(int(event_key))

    # Convert to numpy arrays where possible
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)
    fit_norms = np.array(fit_norms)
    fit_mses = np.array(fit_mses)
    ransac_scores = np.array(ransac_scores)
    q_effs = np.array(q_effs)
    dxs = np.array(dxs)
    event_indices = np.array(event_indices)
    track_indices = np.array(track_indices)

    # Convert variable-length arrays to awkward arrays
    fit_labels_awk = ak.Array(fit_labels_list)
    dQs_awk = ak.Array(dQs)
    total_charges = np.array(total_charges)

    # Writing to ROOT file
    with uproot.recreate(output_file) as file:
        # Tracks tree
        file["tracks"] = {
            "event": event_indices,
            "track": track_indices,
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "length": fit_norms,
            "mse": fit_mses,
            "score": ransac_scores,
            "qeff": q_effs,
            "dx": dxs,
            "dQ": dQs_awk,
        }

        # Events tree
        file["events"] = {
            "event": np.array(event_indices_events),
            "Fit_labels": fit_labels_awk,
            "Total_charge": total_charges,
        }

    print("Data successfully written to output.root")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input_file", help="Path to input file")
    parser.add_argument(
        "output_file", help="Path to output file", default="output.root"
    )
    args = parser.parse_args()
    with open(args.input_file, "rb") as f:
        metrics = pickle.load(f)

    metrics_to_root(metrics, args.output_file)
