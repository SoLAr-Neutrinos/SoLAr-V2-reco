#!/usr/bin/env python


def process_root(input_charge, input_light):
    params.output_folder = "_".join(input_light.split("_")[-2:]).split(".")[0]

    output_path = os.path.join(params.work_path, f"{params.output_folder}")
    output_charge = os.path.join(output_path, f"charge_df_{params.output_folder}.pkl")

    output_light = os.path.join(output_path, f"light_df_{params.output_folder}.pkl")

    output_match = os.path.join(output_path, f"match_dict_{params.output_folder}.json")

    print("\nLoading charge file...")
    charge_df = load_charge(input_charge)

    # Remove events with no charge hits or triggers
    charge_mask = (charge_df["event_hits_q"].apply(tuple).explode().groupby("eventID").min() > 0) * (
        charge_df["trigID"].apply(len) > 0
    )
    charge_df = charge_df[charge_mask]

    print("\nLoading light file...")
    light_df = load_light(input_light)

    print("\nMatching events...")
    match_dict, dt = match_events(charge_df, light_df, return_dt=True)

    # Remove light events without charge event match
    light_events = np.unique(ak.flatten(match_dict.values()))
    light_df = light_df[light_df["event"].isin(light_events)]

    # Remove charge events without associated light event
    charge_df = charge_df.loc[list(match_dict.keys())]

    # Flip x axis
    charge_df["event_hits_x"] = charge_df["event_hits_x"].apply(lambda x: [np.power(-1, params.flip_x) * i for i in x])
    light_df["x"] = light_df["x"].apply(lambda x: np.power(-1, params.flip_x) * x)

    # Save files
    os.makedirs(output_path, exist_ok=True)
    charge_df.to_pickle(output_charge)
    light_df.to_pickle(output_light)
    with open(output_match, "w") as f:
        json.dump(match_dict, f)

    print("Root processing completed.")

    return charge_df, light_df, match_dict


if __name__ == "__main__":
    from argparse import ArgumentParser

    from methods import (
        ak,
        json,
        pickle,
        load_charge,
        load_light,
        match_events,
        np,
        os,
        params,
    )
    from uproot import load_charge, load_light

    parser = ArgumentParser()
    parser.add_argument("light", help="Path to light file")
    parser.add_argument("charge", help="Path to charge file")
    parser.add_argument("-o", "--output", help="Output folder", default=None)

    args = parser.parse_args()

    input_charge = args.charge
    input_light = args.light
    if args.output is not None:
        params.work_path = args.output

    _, _, _ = process_root(input_charge, input_light)

else:
    from .methods import (
        ak,
        json,
        match_events,
        np,
        os,
        params,
    )
    from .uproot import load_charge, load_light
