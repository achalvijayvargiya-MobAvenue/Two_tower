from __future__ import annotations


def setup_mlflow(tracking_uri: str | None, experiment_name: str) -> None:
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

