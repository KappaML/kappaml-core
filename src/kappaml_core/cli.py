"""
This is a skeleton file that can serve as a starting point for a Python
console script.
"""

import argparse
import logging
import sys

from river import compose, datasets, facto, metrics, optim, preprocessing
from river.evaluate import progressive_val_score
from river.linear_model import LinearRegression
from river.model_selection import GreedyRegressor
from river.reco import Baseline, BiasedMF, FunkMF

from kappaml_core import __version__, meta

__author__ = "Alex Imbrea"
__copyright__ = "Alex Imbrea"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


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


def evaluate(model):
    X_y = datasets.Phishing()
    metric = metrics.MAE() + metrics.RMSE()
    return progressive_val_score(
        X_y, model, metric, print_every=100, show_time=True, show_memory=True
    )


MODEL_CHOICES = [
    "baseline",
    "funk_mf",
    "biased_mf",
    "fm",
    "greedy",
    "meta_regressor",
]


def demo(demo_name):
    """Demo all the KappaML models"""
    if demo_name == "baseline":
        print("Baseline model")
        baseline_params = {
            "optimizer": optim.SGD(0.025),
            "l2": 0.0,
            "initializer": optim.initializers.Zeros(),
        }

        model = preprocessing.PredClipper(
            regressor=Baseline(**baseline_params), y_min=1, y_max=5
        )
        evaluate(model)
    elif demo_name == "funk_mf":
        print("FunkMF model")
        funk_mf_params = {
            "n_factors": 10,
            "optimizer": optim.SGD(0.05),
            "l2": 0.1,
            "initializer": optim.initializers.Normal(mu=0.0, sigma=0.1, seed=73),
        }

        model = preprocessing.PredClipper(
            regressor=FunkMF(**funk_mf_params), y_min=1, y_max=5
        )
        evaluate(model)
    elif demo_name == "biased_mf":
        biased_mf_params = {
            "n_factors": 10,
            "bias_optimizer": optim.SGD(0.025),
            "latent_optimizer": optim.SGD(0.05),
            "weight_initializer": optim.initializers.Zeros(),
            "latent_initializer": optim.initializers.Normal(mu=0.0, sigma=0.1, seed=73),
            "l2_bias": 0.0,
            "l2_latent": 0.0,
        }

        model = preprocessing.PredClipper(
            regressor=BiasedMF(**biased_mf_params), y_min=1, y_max=5
        )
        evaluate(model)
    elif demo_name == "fm":
        print("Facto Machine")
        fm_params = {
            "n_factors": 10,
            "weight_optimizer": optim.SGD(0.025),
            "latent_optimizer": optim.SGD(0.05),
            "sample_normalization": False,
            "l1_weight": 0.0,
            "l2_weight": 0.0,
            "l1_latent": 0.0,
            "l2_latent": 0.0,
            "intercept": 3,
            "intercept_lr": 0.01,
            "weight_initializer": optim.initializers.Zeros(),
            "latent_initializer": optim.initializers.Normal(mu=0.0, sigma=0.1, seed=73),
        }

        regressor = compose.Select("user", "item")
        regressor |= facto.FMRegressor(**fm_params)

        model = preprocessing.PredClipper(regressor=regressor, y_min=1, y_max=5)
        evaluate(model)
    elif demo_name == "greedy":
        print("Greedy model selection")
        models = [
            preprocessing.PredClipper(
                LinearRegression(optimizer=optim.SGD(lr=lr)),
                y_min=1,
                y_max=5,
            )
            for lr in [0.025, 0.05, 0.1]
        ]

        model = GreedyRegressor(models=models)
        evaluate(model)
    elif demo_name == "meta_regressor":
        print("Meta regressor model selection")
        models = [
            preprocessing.PredClipper(
                LinearRegression(optimizer=optim.SGD(lr=lr)),
                y_min=1,
                y_max=5,
            )
            for lr in [0.025, 0.05, 0.1]
        ]

        model = meta.MetaRegressor(models=models)
        evaluate(model)
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
        choices=MODEL_CHOICES,
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
