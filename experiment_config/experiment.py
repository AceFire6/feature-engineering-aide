from collections import defaultdict
from typing import Any, Dict, List
from pathlib import Path

import pandas as pd
import numpy as np
import toml

from experiment_config.settings import SUPPORTED_CLASSIFIERS, SUPPORTED_METRICS


class Experiment:
    def __init__(self, experiment_config: Dict[str, Any], file_path: Path = None):
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
            self.prediction_data[column].cat.set_categories(
                category_ordering,
                ordered=True,
                inplace=True,
            )
            self.ordinal_encoded_cols[column] = self.prediction_data[column].cat.categories
            self.prediction_data[column] = self.prediction_data[column].cat.codes

        # Do OneHotEncoding
        if self.categorical_features:
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

        self.name = experiment_config['experiment']
        self.metric_results = {}
        self.metric_results_labels = {}

        self._metrics = experiment_config['metrics']
        self.metrics = {metric: SUPPORTED_METRICS[metric] for metric in self._metrics}

        self._classifiers = experiment_config['classifiers']
        self.classifiers = {
            classifier: SUPPORTED_CLASSIFIERS[classifier]
            for classifier in self._classifiers
        }
        self.file_path = file_path

    def training_set_sample_size(self):
        return self.prediction_data[self.target_column].size

    def get_x_train_test_split(self, train_indices, test_indices):
        return self.X.iloc[train_indices], self.X.iloc[test_indices]

    def get_y_train_test_split(self, train_indices, test_indices):
        return self.y.iloc[train_indices], self.y.iloc[test_indices]

    def add_result(self, metric: str, result: float, label: str = None) -> None:
        if metric not in self.metric_results:
            self.metric_results[metric] = []
            self.metric_results_labels[metric] = []

        self.metric_results[metric].append(result)
        self.metric_results_labels[metric].append(label)


def parse_experiment_paths(experiment_input_paths: List[str]) -> List[Experiment]:
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
