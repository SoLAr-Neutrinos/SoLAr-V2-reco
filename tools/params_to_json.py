#!/usr/bin/env python

import json


def params_to_json(json_file):
    with open(f"{json_file}.json", "w") as f:
        variables = {
            key: value if not isinstance(value, params.np.ndarray) else value.tolist()
            for key, value in params.__dict__.items()
            if not key.startswith("__") and not type(value) is type(params)
        }
        json.dump(variables, f)


if __name__ == "__main__":
    import params
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to JSON file (no extension needed)")
    args = parser.parse_args()

    params_to_json(args.json_file.split(".")[0])

else:
    from . import params
