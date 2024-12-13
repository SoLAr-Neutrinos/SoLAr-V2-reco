import uproot
import pandas as pd
import numpy as np

if __package__:
    from . import params
    from .methods import get_sipm_mask, integrate_peaks, sipm_to_xy
else:
    import params
    from methods import get_sipm_mask, integrate_peaks, sipm_to_xy


# Uproot
def load_charge(file_name, events=None):
    with uproot.open(file_name) as f:
        if "events" in f:
            charge_df = f["events"].arrays(library="pd")
        elif "HitTree" in f:
            charge_df = f["HitTree"].arrays(library="pd")
            charge_df.rename({"eid": "event", "iev": "eventID"}, axis=1, inplace=True)

        charge_df.set_index("eventID", inplace=True)
        if events is not None:
            charge_df = charge_df.loc[events]

    return charge_df


def load_light(file_name, deco=True, events=None, mask=True, keep_rwf=False):
    light_df = pd.DataFrame()
    with uproot.open(file_name) as f:
        if deco:
            tree = f["decowave"]
        else:
            tree = f["rwf_array"]

        scaling_par = 655.340
        if "scaling_par" in f:
            scaling_par = f["scaling_par"].value

        for idx, arrays in enumerate(tree.iterate(library="np")):
            df = pd.DataFrame.from_dict(arrays, orient="index").T
            df.dropna()
            if events is not None:
                df = df[df["event"].isin(events)]

            df[["sn", "ch"]] = df[["sn", "ch"]].astype(int)

            if mask:
                df = df[
                    df[["sn", "ch"]].apply(
                        lambda x: get_sipm_mask(x.iloc[0], x.iloc[1]), axis=1
                    )
                ]

            if df.empty:
                continue

            df[["x", "y"]] = df[["sn", "ch"]].apply(
                lambda x: pd.Series(sipm_to_xy(x.iloc[0], x.iloc[1])), axis=1
            )

            if deco:
                df["rwf"] = df["decwfm"]

            df["rwf"] = df["rwf"].apply(lambda x: x / scaling_par)

            df[["integral", "properties"]] = df["rwf"].apply(
                lambda x: pd.Series(
                    (np.nan, {}) if any(np.isnan(x)) else integrate_peaks(x)
                )
            )
            df["peak"] = df["properties"].apply(
                lambda x: (
                    max(x["peak_heights"])
                    if "peak_heights" in x and len(x["peak_heights"]) > 0
                    else np.nan
                )
            )

            columns = ["event", "tai_ns", "sn", "ch", "peak", "integral", "x", "y"]
            if keep_rwf:
                columns.append("rwf")

            df = df[columns]
            light_df = pd.concat([light_df, df], ignore_index=True)

    return light_df
