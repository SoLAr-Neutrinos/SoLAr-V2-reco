#!/usr/bin/env python

from ..tools.methods import (
    params,
    np,
    pd,
    cluster_hits,
    fit_hit_clusters,
    tqdm,
)


def get_translation():
    translation = (
        (-np.power(-1, params.flip_x) * params.quadrant_size * 2.0 * params.simulate_dead_area),
        (params.quadrant_size * 0.5 * params.simulate_dead_area),
        -155,
    )
    return translation


def translate_coordinates(hit_df, translation, inverse=False):
    hit_df["hit_x"] = hit_df["hit_x"].apply(lambda x: [round(i - translation[0] * np.power(-1, inverse), 1) for i in x])
    hit_df["hit_y"] = hit_df["hit_y"].apply(lambda x: [round(i - translation[1] * np.power(-1, inverse), 1) for i in x])
    hit_df["hit_z"] = hit_df["hit_z"].apply(lambda x: [round(i - translation[2] * np.power(-1, inverse), 1) for i in x])

    return hit_df


def rotate_coordinates(hit_df):
    df = hit_df.copy()
    df["hit_x"] = hit_df["hit_z"].apply(lambda x: [-i for i in x])
    df["hit_y"] = hit_df["hit_y"]
    df["hit_z"] = hit_df["hit_x"]

    return df


def cut_volume(hit_df):
    additional_columns = [col for col in hit_df.columns if col not in ["hit_x", "hit_y", "hit_z"]]
    columns_to_zip = ["hit_x", "hit_y", "hit_z"] + additional_columns

    def filter_hits(hit_row):
        filtered_hits = [
            hit
            for hit in zip(*[hit_row[col] for col in columns_to_zip])
            if all(
                abs(coord) <= detector / 2
                for coord, detector in zip(
                    hit[:3],
                    (params.detector_x, params.detector_y, 2 * params.detector_z),
                )
            )
        ]
        return pd.Series(zip(*filtered_hits))

    hit_df[columns_to_zip] = hit_df.apply(filter_hits, axis=1)
    hit_df.dropna(subset=["hit_x", "hit_y", "hit_z"], inplace=True)

    return hit_df


def cut_sipms(hit_df):
    additional_columns = [col for col in hit_df.columns if col not in ["hit_x", "hit_y", "hit_z"]]
    columns_to_zip = ["hit_x", "hit_y", "hit_z"] + additional_columns

    def filter_hits(hit_row):
        filtered_hits = [
            hit
            for hit in zip(*[hit_row[col] for col in columns_to_zip])
            if not any(
                sipm_x - params.sipm_size / 2 <= hit[0] <= sipm_x + params.sipm_size / 2
                and sipm_y - params.sipm_size / 2 <= hit[1] <= sipm_y + params.sipm_size / 2
                for sipm_x, sipm_y in [
                    (
                        -params.quadrant_size * 4 + params.quadrant_size * (i + 0.5),
                        -params.quadrant_size * 4 + params.quadrant_size * (j + 0.5),
                    )
                    for i in range(8)
                    for j in range(8)
                ]
            )
        ]
        return pd.Series(zip(*filtered_hits))

    hit_df[columns_to_zip] = hit_df.apply(filter_hits, axis=1)
    hit_df.dropna(subset=["hit_x", "hit_y", "hit_z"], inplace=True)

    return hit_df


def cut_chips(hit_df):
    additional_columns = [col for col in hit_df.columns if col not in ["hit_x", "hit_y", "hit_z"]]
    columns_to_zip = ["hit_x", "hit_y", "hit_z"] + additional_columns

    def filter_hits(hit_row):
        filtered_hits = [
            hit
            for hit in zip(*[hit_row[col] for col in columns_to_zip])
            if not (
                any(
                    chip_x - params.quadrant_size / 2
                    <= hit[0] * np.power(-1, params.flip_x)
                    <= chip_x + params.quadrant_size / 2
                    and chip_y - params.quadrant_size / 2 <= hit[1] <= chip_y + params.quadrant_size / 2
                    for chip_x, chip_y in [
                        (
                            -params.quadrant_size * 4 + params.quadrant_size * (j - 1 + 0.5),
                            params.quadrant_size * 4 - params.quadrant_size * (i - 1 + 0.5),
                        )
                        for (i, j) in [(3, 3), (4, 4), (5, 4), (6, 4)]
                    ]
                )
                or (
                    hit[0] * np.power(-1, params.flip_x)
                    + params.quadrant_size * 4
                    - params.quadrant_size
                    + hit[1]
                    - params.quadrant_size * 4
                    + params.quadrant_size * 4
                    <= params.quadrant_size
                    and 0
                    <= hit[0] * np.power(-1, params.flip_x) + params.quadrant_size * 4 - params.quadrant_size
                    <= params.quadrant_size
                    and 0 <= hit[1] - params.quadrant_size * 4 + params.quadrant_size * 4 <= params.quadrant_size
                )
            )
        ]
        return pd.Series(zip(*filtered_hits))

    hit_df[columns_to_zip] = hit_df.apply(filter_hits, axis=1)
    hit_df.dropna(subset=["hit_x", "hit_y", "hit_z"], inplace=True)

    return hit_df


def fit_events(charge_df):
    metrics = {}
    for event in tqdm(charge_df.index, desc="Fitting events"):
        charge_values = pd.DataFrame(
            charge_df.loc[
                event,
                [
                    "hit_x",
                    "hit_y",
                    "hit_z",
                    "hit_q",
                ],
            ].to_list(),
            index=["x", "y", "z", "q"],
        ).T

        # charge_values["q"] = charge_values["q"] * charge_gain  # Convert mV to ke

        # Create a design matrix
        labels = cluster_hits(
            charge_values[["x", "y", "z"]].to_numpy(),
        )
        # Fit clusters
        metrics[event] = fit_hit_clusters(
            charge_values[["x", "y", "z"]].to_numpy(),
            charge_values["q"].to_numpy(),
            labels,
        )

        # metrics[idx][
        #     "Pixel_mask"
        # ] = mask.to_numpy()  # Save masks to original dataframe for reference
        metrics[event]["Total_charge"] = charge_values["q"].sum()

    return metrics


# if __name__ == "__main__":
#     from argparse import ArgumentParser

#     parser = ArgumentParser()
#     parser.add_argument("charge", help="Path to charge file")
#     parser.add_argument(
#         "--dead-areas", "-d", help="Simulate dead areas", action="store_false"
#     )

#     args = parser.parse_args()

#     input_charge = args.charge
#     params.simulate_dead_area = args.dead_areas
#     params.output_folder = input_charge.split("_")[-1].split(".")[0]
#     recal_params()
