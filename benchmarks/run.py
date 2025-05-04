import json

from river import (
    datasets,
    dummy,
    evaluate,
    linear_model,
    metrics,
    preprocessing,
    stats,
    tree,
)
from tqdm import tqdm

from kappaml_core import meta

TRACKS = {
    "Regression": {
        evaluate.RegressionTrack(),
        # evaluate.Track(
        #     name="Regression - Synthetic datasets",
        #     datasets=[
        #         datasets.synth.Friedman(),
        #         datasets.synth.FriedmanDrift(),
        #     ],
        #     metric=metrics.MAE(),
        # )
    },
    "Classification": {
        evaluate.BinaryClassificationTrack(),
    },
    "Forecasting": {
        evaluate.Track(
            name="Forecasting",
            datasets=[
                datasets.AirlinePassengers(),
            ],
            metric=metrics.MAE(),
        )
    },
}

MODELS = {
    "Regression": {
        "BASELINE - Mean": dummy.StatisticRegressor(stats.Mean()),
        "KappaML - MetaRegressor": meta.MetaRegressor(
            models=[
                preprocessing.StandardScaler() | linear_model.LinearRegression(),
                preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor(),
            ],
        ),
    },
    "Classification": {
        "BASELINE": dummy.NoChangeClassifier(),
        # "Aggregated Mondrian Forest": forest.AMFClassifier(seed=42),
    },
    "Forecasting": {
        "BASELINE - Mean": dummy.StatisticRegressor(stats.Mean()),
        # "SNARIMAX": time_series.SNARIMAX(p=1, d=1, q=1),
    },
}


def run_track(track, models):
    results = {}
    for dataset in track:
        results[dataset.__class__.__name__] = {}
        for key, model in models.items():
            results[dataset.__class__.__name__][key] = []
            time = 0.0
            for i in tqdm(
                track.run(model, dataset),
                total=10,
                desc=f"{key} on {dataset.__class__.__name__}",
            ):
                time += i["Time"].total_seconds()
                res = {
                    "step": i["Step"],
                    "track": track.name,
                    "model": key,
                    "dataset": dataset.__class__.__name__,
                }
                for k, v in i.items():
                    if isinstance(v, metrics.base.Metric):
                        res[k] = v.get()
                res["Memory in Mb"] = i["Memory"] / 1024**2
                res["Time in s"] = time
                results[dataset.__class__.__name__][key].append(res)
    return results


if __name__ == "__main__":
    """Run all benchmark tracks."""
    results = {}
    for track_type in TRACKS:
        results[track_type] = {}
        for track in TRACKS[track_type]:
            results[track_type][track.name] = run_track(track, MODELS[track_type])

    # Print overview of final results
    print("\nBenchmark Results Overview:")
    print("=" * 80)

    for track_type, tracks in results.items():
        print(f"\n{track_type}:")
        print("-" * 40)

        for track_name, sets in tracks.items():
            print(f"\n{track_name}:")

            for dataset, models in sets.items():
                print(f"\n  Dataset: {dataset}")

                for model, metrics_list in models.items():
                    # Get the final metrics
                    final_metrics = metrics_list[-1]
                    print(f"    {model}:")

                    # Print all metrics except metadata
                    for metric, value in final_metrics.items():
                        if metric not in ["step", "track", "model", "dataset"]:
                            print(f"      {metric}: {value:.4f}")

    with open("results.json", "w") as f:
        json.dump(results, f)
