import argparse

from . import analysis, display_events, reconstruction


def main():
    parser = argparse.ArgumentParser(
        description="Solar event reconstruction and analysis"
    )
    subparsers = parser.add_subparsers(dest="subparser_name")

    # Reconstruction
    parser_reconstruction = subparsers.add_parser("reco", help="Reconstruct events")
    parser_reconstruction.add_argument("-c", "--charge", help="Path to charge file")
    parser_reconstruction.add_argument("-l", "--light", help="Path to light file")
    parser_reconstruction.add_argument(
        "-f", "--folder", help="Folder name for processing"
    )
    parser_reconstruction.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for additional parameters or json file containing parameters",
        required=False,
    )

    # Analysis
    parser_analysis = subparsers.add_parser("ana", help="Analyze events")
    parser_analysis.add_argument(
        "folder",
        help="Folder name for specific metrics file",
        default="combined",
        nargs="?",
    )
    parser_analysis.add_argument(
        "--filter", help="Tag number of filter file within folder", default=None
    )
    # parser_analysis.add_argument(
    #     "--save", "-s", help="Save images", action="store_true"
    # )
    parser_analysis.add_argument(
        "--display", help="Display images (not recomended)", action="store_true"
    )
    parser_analysis.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for aditional parameters or json file containing parameters",
        required=False,
    )

    # Display events
    parser_display = subparsers.add_parser("display", help="Display events")
    parser_display.add_argument("folder", help="Folder name for specific data file")
    parser_display.add_argument(
        "-e", "--events", help="Event number", type=int, nargs="+"
    )
    parser_display.add_argument("-s", "--save", help="Save images", action="store_true")
    parser_display.add_argument(
        "-n", "--no-display", help="Don't display images", action="store_false"
    )
    parser_display.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for aditional parameters or json file containing parameters",
        required=False,
    )

    args = parser.parse_args()

    if args.subparser_name == "reco":
        reconstruction.main(**vars(args))

    elif args.subparser_name == "ana":
        analysis.main(
            **vars(args),
        )

    elif args.subparser_name == "display":
        display_events.main(**vars(args))


if __name__ == "__main__":
    main()
