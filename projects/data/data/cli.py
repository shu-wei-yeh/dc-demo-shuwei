import os

from data.find import DataQualityDict
from gwpy.timeseries import TimeSeriesDict
from jsonargparse import ActionConfigFile, ArgumentParser


def fetch(
    start: float, end: float, channels: list[str], sample_rate: float
) -> TimeSeriesDict:
    """
    Simple wrapper to annotate and simplify
    the kwargs so that jsonargparse can build
    a parser out of them.
    """

    X = TimeSeriesDict.fetch(start=start, end=end, channels=channels)
    return X.resample(sample_rate)


def main(args=None):
    query_parser = ArgumentParser()
    query_parser.add_method_arguments(DataQualityDict, "query_segments")
    query_parser.add_argument("--output-file", "-o", type=str)

    fetch_parser = ArgumentParser()
    fetch_parser.add_function_arguments(fetch)
    fetch_parser.add_argument("--sample-rate", type=float)
    fetch_parser.add_argument("--output-directory", "-o", type=str)
    fetch_parser.add_argument("--prefix", "-p", type=str, default="deepclean")

    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("query", query_parser)
    subcommands.add_subcommand("fetch", fetch_parser)

    args = parser.parse_args(args)
    if args.subcommand == "query":
        args = args.query.as_dict()
        output_file = args.pop("output_file")
        segments = DataQualityDict.query_segments(**args)
        segments.write(output_file)
    elif args.subcommand == "fetch":
        args = args.fetch.as_dict()
        output_directory = args.pop("output_directory")
        prefix = args.pop("prefix")
        X = fetch(**args)

        fname = "{prefix}-{start}-{end}.hdf5".format(prefix, **args)
        fname = os.path.join(output_directory, fname)
        X.write(fname, format="hdf5")


if __name__ == "__main__":
    main()
