from typing import Any, TypedDict

from feature_engineer.experiment_config.experiment import Experiment


class ExperimentResult(TypedDict):
    best_model: Any
    features_used: list[str]

    train_test_metric_results: dict[str, list[float]]
    train_test_metric_results_summary: dict[str, dict[str, float]]
    holdout_metric_results: dict[str, float]
    classification_report: str

    extra_data: dict[str, Any]


class LabelledResults(TypedDict):
    label: str
    result: ExperimentResult


class ExperimentRun(TypedDict):
    run_number: int
    time_taken: float
    seed: int

    results: LabelledResults


class ExperimentInfo(TypedDict):
    experiment: Experiment
    runs: list[ExperimentRun]
