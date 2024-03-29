from typing import Any, TypedDict

from feature_engineer.experiment_config.experiment import Experiment


class ExperimentResult(TypedDict):
    features_used: list[str]
    train_test_metric_results: dict[str, list[float]]
    train_test_metric_results_summary: dict[str, dict[str, float]]
    holdout_metric_results: dict[str, float]
    classification_report: str

    extra_data: dict[str, Any]


class ExperimentRun(TypedDict):
    run_number: int
    time_taken: float
    seed: int

    results: dict[str, ExperimentResult]


class ExperimentInfo(TypedDict):
    experiment: Experiment
    runs: list[ExperimentRun]
