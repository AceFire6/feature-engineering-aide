from collections import defaultdict
from typing import Any, Callable, Iterable, Optional, Protocol, Type, TypedDict, Union
from pathlib import Path

from numpy.typing import ArrayLike
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
import toml

from feature_engineer.experiment_runner.settings import (
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_FEATURE_PREPROCESSORS,
    SUPPORTED_METRICS,
)


class FeatureMap(TypedDict):
    categorical: Optional[list[str]]
    bool: Optional[list[str]]
    ordinal: Optional[dict[str, list[str]]]


class ScorerProtocol(Protocol):
    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike, sample_weight: Optional[ArrayLike] = None) -> float:
        ...


class Experiment:
    seed: int
    start_time: Optional[str] = None
    prediction_data_file: Path

    prediction_true_values: list[str]
    prediction_false_values: list[str]

    feature_map: FeatureMap
    boolean_features: list[str]
    ordinal_features: dict[str, list[str]]
    categorical_features: list[str]
    dtypes: dict[str, str]

    prediction_data_columns: list[str]
    target_column: str
    split_column: str
    holdout_split_condition: Any

    prediction_data: pd.DataFrame
    ordinal_encoded_cols: defaultdict[str, pd.Index]

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    holdout_x: pd.DataFrame
    holdout_y: pd.Series

    name: str
    metric_results: dict
    metric_results_labels: dict

    metrics: dict[str, ScorerProtocol]
    feature_preprocessors: dict[str, Optional[Callable]]
    classifiers: dict[str, Type[ClassifierMixin]]
    file_path: Path

    def __init__(self, experiment_config: dict[str, Any], file_path: Path):
        self.seed = experiment_config['random_seed']
        np.random.seed(self.seed)

        self.prediction_data_file = Path(experiment_config['data_source'])

        self.prediction_true_values = experiment_config['data']['true_values']
        self.prediction_false_values = experiment_config['data']['false_values']

        self.feature_map = experiment_config['data']['features']
        self.boolean_features = self.feature_map.get('bool', [])
        self.ordinal_features = self.feature_map.get('ordinal', {})
        self.categorical_features = self.feature_map.get('categorical', [])

        # We exclude booleans because we set those after the NA values are removed
        self._categorical_columns = self.categorical_features + list(self.ordinal_features.keys())
        dtype_map = {
            'category': self._categorical_columns,
        }
        self.dtypes = {}
        for dtype, feat_list in dtype_map.items():
            for column in feat_list:
                self.dtypes[column] = dtype

        self.prediction_data_columns = experiment_config['data']['selected_features']
        self.target_column = experiment_config['data']['target']
        self.split_column = experiment_config['data']['data_split_column']
        self.holdout_split_condition = experiment_config['data']['holdout_split_condition']

        self.prediction_data: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self.prediction_data_file,
            dtype=self.dtypes,
            usecols=self.prediction_data_columns + [self.target_column, self.split_column],
            true_values=self.prediction_true_values,
            false_values=self.prediction_false_values,
            na_values=experiment_config['data']['na_values'],
        )
        self.prediction_data.dropna(inplace=True)
        self.prediction_data = self.prediction_data.astype(
            {column: 'bool' for column in self.boolean_features}
        )

        # Do OrdinalEncoding
        self.ordinal_encoded_cols = defaultdict()
        for column, category_ordering in self.ordinal_features.items():
            # Skip holdout column
            if column == self.split_column:
                continue

            self.prediction_data[column].cat.set_categories(
                category_ordering,
                ordered=True,
                inplace=True,
            )
            self.ordinal_encoded_cols[column] = self.prediction_data[column].cat.categories
            self.prediction_data[column] = self.prediction_data[column].cat.codes

        # Do OneHotEncoding
        if self.categorical_features:
            # Remove holdout column before encoding
            if self.split_column in self.categorical_features:
                self.categorical_features.remove(self.split_column)

            self.prediction_data = pd.get_dummies(
                self.prediction_data,
                columns=self.categorical_features,
                prefix=self.categorical_features,
            )

            for feature in self.categorical_features:
                encoded_features = [
                    column for column in self.prediction_data.columns
                    if feature in column
                ]
                self.prediction_data_columns.remove(feature)
                self.prediction_data_columns.extend(encoded_features)

        holdout_index = self.prediction_data[self.split_column] == self.holdout_split_condition

        self.X = self.prediction_data[~holdout_index][self.prediction_data_columns]
        self.y = self.prediction_data[~holdout_index][self.target_column]
        self.groups = self.prediction_data[~holdout_index][self.split_column]

        self.holdout_x = self.prediction_data[holdout_index][self.prediction_data_columns]
        self.holdout_y = self.prediction_data[holdout_index][self.target_column]

        self.name = experiment_config['experiment']
        self.metric_results = {}
        self.metric_results_labels = {}

        self._metrics = experiment_config['metrics']
        self.metrics = {metric: SUPPORTED_METRICS[metric] for metric in self._metrics}

        self._feature_preprocessors = experiment_config['feature_selection_strategies']
        self.feature_preprocessors = {
            preprocessor: SUPPORTED_FEATURE_PREPROCESSORS[preprocessor]
            for preprocessor in self._feature_preprocessors
        }

        self._classifiers = experiment_config['classifiers']
        self.classifiers = {
            classifier: SUPPORTED_CLASSIFIERS[classifier]
            for classifier in self._classifiers
        }
        self.file_path = file_path

    def training_set_sample_size(self) -> int:
        return self.prediction_data[self.target_column].size

    def add_result(self, metric: str, result: float, label: str = None) -> None:
        if metric not in self.metric_results:
            self.metric_results[metric] = []
            self.metric_results_labels[metric] = []

        self.metric_results[metric].append(result)
        self.metric_results_labels[metric].append(label)

    @staticmethod
    def get_split(
        dataset: Union[pd.DataFrame, pd.Series],
        train_indices: pd.Series,
        test_indices: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return dataset.iloc[train_indices], dataset.iloc[test_indices]


def parse_experiment_paths(experiment_input_paths: Iterable[str]) -> list[Experiment]:
    experiment_file_paths = [Path(experiment_file) for experiment_file in experiment_input_paths]
    experiments = []

    for experiment_file_path in experiment_file_paths:
        if experiment_file_path.is_dir():
            print(f'Cannot handle {experiment_file_path.absolute()} as it is a directory!')
            continue

        experiment_config = toml.load(experiment_file_path)
        experiment = Experiment(experiment_config, file_path=experiment_file_path)
        experiments.append(experiment)

    return experiments
