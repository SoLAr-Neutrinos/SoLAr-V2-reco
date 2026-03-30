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
        else:
            raise ValueError("No suitable tree found in the file")

        charge_df.set_index("eventID", inplace=True)
        if events is not None:
            charge_df = charge_df.loc[events]

    return charge_df


def load_light(file_name, events=None, mask=True, keep_rwf=False):
    light_df = pd.DataFrame()
    with uproot.open(file_name) as f:
        if "flash_tree" in f:
            tree = f["flash_tree"]
            for idx, arrays in enumerate(tree.iterate(library="np")):
                df = _sipm_flash_table(arrays)
                light_df = pd.concat([light_df, df], ignore_index=True)

            return light_df

        elif "decowave" in f:
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
                df = df[df[["sn", "ch"]].apply(lambda x: get_sipm_mask(x.iloc[0], x.iloc[1]), axis=1)]

            if df.empty:
                continue

            df[["x", "y"]] = df[["sn", "ch"]].apply(lambda x: pd.Series(sipm_to_xy(x.iloc[0], x.iloc[1])), axis=1)

            if "flash_tree" in f:
                df["integral"] = df["sipm_charge"]
            else:
                df["rwf"] = df["decwfm"]

                df["rwf"] = df["rwf"].apply(lambda x: x / scaling_par)

                df[["integral", "properties"]] = df["rwf"].apply(
                    lambda x: pd.Series((np.nan, {}) if any(np.isnan(x)) else integrate_peaks(x))
                )

            columns = ["event", "tai_ns", "sn", "ch", "integral", "x", "y"]
            if keep_rwf:
                columns.append("rwf")

            df = df[columns]
            light_df = pd.concat([light_df, df], ignore_index=True)

    return light_df

# From decowave. If decowave changes, this list may need to be updated.
SIPM_CHANNEL_LIST = [
        {"sn": 215553018, "ch": 4}, {"sn": 215553018, "ch": 5}, {"sn": 215553018, "ch": 6},
        {"sn": 215553018, "ch": 7}, {"sn": 215553018, "ch": 8}, {"sn": 215553018, "ch": 9},
        {"sn": 215553018, "ch": 10}, {"sn": 215553018, "ch": 11}, {"sn": 215553018, "ch": 12},
        {"sn": 215553018, "ch": 13}, {"sn": 215553018, "ch": 14}, {"sn": 215553018, "ch": 15},
        {"sn": 215553018, "ch": 20}, {"sn": 215553018, "ch": 21}, {"sn": 215553018, "ch": 22},
        {"sn": 215553018, "ch": 23}, {"sn": 215553018, "ch": 36}, {"sn": 215553018, "ch": 37},
        {"sn": 215553018, "ch": 38}, {"sn": 215553018, "ch": 39}, {"sn": 215553018, "ch": 40},
        {"sn": 215553018, "ch": 41}, {"sn": 215553018, "ch": 42}, {"sn": 215553018, "ch": 43},
        {"sn": 215553018, "ch": 44}, {"sn": 215553018, "ch": 45}, {"sn": 215553018, "ch": 46},
        {"sn": 215553018, "ch": 47}, {"sn": 215553018, "ch": 52}, {"sn": 215553018, "ch": 53},
        {"sn": 215553018, "ch": 54}, {"sn": 215553018, "ch": 55},
        {"sn": 215537201, "ch": 4}, {"sn": 215537201, "ch": 5}, {"sn": 215537201, "ch": 6},
        {"sn": 215537201, "ch": 7}, {"sn": 215537201, "ch": 8}, {"sn": 215537201, "ch": 9},
        {"sn": 215537201, "ch": 10}, {"sn": 215537201, "ch": 11}, {"sn": 215537201, "ch": 12},
        {"sn": 215537201, "ch": 13}, {"sn": 215537201, "ch": 14}, {"sn": 215537201, "ch": 15},
        {"sn": 215537201, "ch": 20}, {"sn": 215537201, "ch": 21}, {"sn": 215537201, "ch": 22},
        {"sn": 215537201, "ch": 23}, {"sn": 215537201, "ch": 36}, {"sn": 215537201, "ch": 37},
        {"sn": 215537201, "ch": 38}, {"sn": 215537201, "ch": 39}, {"sn": 215537201, "ch": 40},
        {"sn": 215537201, "ch": 41}, {"sn": 215537201, "ch": 42}, {"sn": 215537201, "ch": 43},
        {"sn": 215537201, "ch": 44}, {"sn": 215537201, "ch": 45}, {"sn": 215537201, "ch": 46},
        {"sn": 215537201, "ch": 47}, {"sn": 215537201, "ch": 52}, {"sn": 215537201, "ch": 53},
        {"sn": 215537201, "ch": 54}, {"sn": 215537201, "ch": 55}
    ]

def _sipm_flash_table(arrays):
    events = arrays['event'][:, 0]  # pick first value
    tai_ns = arrays['tai_ns']
    sipm_charge = arrays['sipm_charge']
    n_events, n_channels = sipm_charge.shape
    sipm_list = SIPM_CHANNEL_LIST
    assert len(sipm_list) == n_channels, "Number of channels does not match expected number of SiPMs ()".format(len(sipm_list))
    data = {
        'event': [],
        'tai_ns': [],
        'sn': [],
        'ch': [],
        'x': [],
        'y': [],
        'integral': []
    }
    for i in range(n_events):
        for ch_idx in range(n_channels):
            sn = sipm_list[ch_idx]['sn']
            ch = sipm_list[ch_idx]['ch']
            xy = sipm_to_xy(sn, ch)
            if xy is None:
                continue
            x, y = xy
            data['event'].append(events[i])
            data['tai_ns'].append(tai_ns[i])
            data['sn'].append(sn)
            data['ch'].append(ch)
            data['x'].append(x)
            data['y'].append(y)
            data['integral'].append(sipm_charge[i, ch_idx])

    return pd.DataFrame(data)