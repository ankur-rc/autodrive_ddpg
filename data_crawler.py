import argparse
import os
import logging


def main(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    root = args.root
    output = args.output
    extensions = args.extensions
    op_file = "files.csv"

    root = os.path.abspath(root)
    if not os.path.exists(root):
        logger.error("Root folder does not exist!")

    output = os.path.join(os.path.abspath(output), op_file)
    if os.path.exists(os.path.basename(output)):
        logger.info("Writing to {}..".format(output))
    else:
        logger.error("Path '{}' is not writable. Please re-check.")

    with open(output, "w") as op:
        for root, _, files in os.walk(root):
            logger.info("Accessing.. {}".format(root))
            for fil in files:
                if os.path.splitext(fil)[1] in extensions:
                    op.write(os.path.abspath(fil)+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Crawls the directory structure to accumulate file paths.")
    parser.add_argument("root", metavar="ROOT",
                        help="Root directory of your dataset.")
    parser.add_argument("output", metavar="OUTPUT",
                        help="Output folder directory. Save file will be 'files.csv'")
    parser.add_argument(
        "--extensions",  metavar="EXT", nargs="+", help="Filename extensions to be considered (default=.jpg)", default=[".jpg"])

    args = parser.parse_args()
    main(args)
