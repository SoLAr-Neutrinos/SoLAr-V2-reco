import argparse

from . import analysis, display_events, reconstruction


def main():
    parser = argparse.ArgumentParser(description="Solar Monte Carlo event reconstruction and analysis")
    subparsers = parser.add_subparsers(dest="subparser_name", required=True)

    # Reconstruction
    parser_reconstruction = subparsers.add_parser("reco", help="Reconstruct events")
    parser_reconstruction.add_argument("charge", type=str, help="Path to the charge file")
    parser_reconstruction.add_argument("--dead-areas", "-d", action="store_true", help="Simulate dead areas")
    parser_reconstruction.add_argument("--parameters", "-p", help="Path to the parameters file")

    # Analysis
    parser_analysis = subparsers.add_parser("ana", help="Analyze events")
    parser_analysis.add_argument(
        "folder",
        type=str,
        help="Folder name for specific metrics file",
        default="combined",
        nargs="?",
    )
    parser_analysis.add_argument("--filter", help="Tag number of filter file within folder", default=None)
    parser_analysis.add_argument("--save", "-s", help="Save images", action="store_true")
    parser_analysis.add_argument("--display", help="Display images (not recomended)", action="store_true")
    parser_analysis.add_argument("--dead-areas", "-d", help="Simulate dead areas", action="store_true")
    parser_analysis.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for aditional parameters or json file containing parameters",
        required=False,
    )

    # Display events
    parser_display = subparsers.add_parser("display", help="Display events")
    parser_display.add_argument("folder", type=str, help="Folder name for specific data file")
    parser_display.add_argument("-e", "--events", help="Event number", type=int, nargs="+")
    parser_display.add_argument("-s", "--save", help="Save images", action="store_true")
    parser_display.add_argument("-n", "--no-display", help="Don't display images", action="store_false")
    parser_display.add_argument("-d", "--dead-areas", help="Simulate dead areas", action="store_true")
    parser_display.add_argument(
        "-p",
        "--parameters",
        action="append",
        help="Key=value pairs for aditional parameters or json file containing parameters",
        required=False,
    )

    args = parser.parse_args()

    if args.subparser_name == "reco":
        reconstruction.main(charge=args.charge, dead_areas=args.dead_areas, parameters=args.parameters)
    elif args.subparser_name == "ana":
        analysis.main(
            folder=args.folder,
            filter=args.filter,
            save=args.save,
            display=args.display,
            dead_areas=args.dead_areas,
            parameters=args.parameters,
        )
    elif args.subparser_name == "display":
        display_events.main(
            folder=args.folder,
            events=args.events,
            save=args.save,
            no_display=args.no_display,
            dead_areas=args.dead_areas,
            parameters=args.parameters,
        )


if __name__ == "__main__":
    main()
