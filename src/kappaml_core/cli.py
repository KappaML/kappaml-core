"""
This is a skeleton file that can serve as a starting point for a Python
console script.
"""

import argparse
import logging
import sys

from kappaml_core import __version__

__author__ = "Alex Imbrea"
__copyright__ = "Alex Imbrea"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from kappaml_core.skeleton import fib`,
# when using this Python module as a library.


def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for _i in range(n - 1):
        a, b = b, a + b
    return a


def demo(demo_name):
    """Demo all the KappaML models"""
    if demo_name == "baseline":
        print("Baseline model")
    elif demo_name == "greedy":
        print("Greedy model selection")
    elif demo_name == "epsilon_greedy":
        print("Epsilon-Greedy model selection")
    pass


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="KappaML Core CLI")
    subparsers = parser.add_subparsers(dest="command")
    fib_parser = subparsers.add_parser("fib", help="Fibonacci example")
    fib_parser.add_argument("n", help="n-th Fibonacci number", type=int)
    demo_parse = subparsers.add_parser("demo", help="Demo all the KappaML models")
    demo_parse.add_argument(
        "demo_name",
        help="Name of the model to be used",
        type=str,
        choices=["baseline", "greedy", "epsilon_greedy"],
    )

    parser.add_argument(
        "--version",
        action="version",
        version="kappaml-core {ver}".format(ver=__version__),
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="loglevel",
        help="set loglevel to ERROR",
        action="store_const",
        const=logging.ERROR,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` and :func:`demo` to be called with string arguments
    in a CLI fashion

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting kappaml-core %s", __version__)
    _logger.debug("Arguments: %s", args)

    if args.command == "fib":
        _logger.debug("Starting crazy calculations...")
        print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
        _logger.info("Done.")
    elif args.command == "demo":
        _logger.debug("Starting demo...")
        demo(args.demo_name)
        _logger.info("Done.")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
