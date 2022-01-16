from river import stream
from river.datasets import base


class MovieLens25M(base.RemoteDataset):
    """MovieLens 1M dataset.

    Source: https://grouplens.org/datasets/movielens/25m/

    References
    ----------
    [^1]: [The MovieLens Datasets](http://dx.doi.org/10.1145/2827872)

    """

    def __init__(self):
        super().__init__(
            n_samples=100_000,
            n_features=10,
            task=base.REG,
            url="https://files.grouplens.org/datasets/movielens/ml-25m.zip",
            size=261_978_986,
            filename="movie.csv",
        )

    def _iter(self):
        return stream.iter_csv(
            self.path,
            target="rating",
            converters={
                "timestamp": int,
                "release_date": int,
                "age": float,
                "rating": float,
            },
            delimiter="\t",
        )
