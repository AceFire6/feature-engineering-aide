from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Any, Dict

from experiment_config.settings import SUPPORTED_CLASSIFIERS, SUPPORTED_METRICS
from experiment_config.utils import get_encoding_from_label


class Experiment:
    def __init__(self, experiment_config: Dict[str, Any]):
        np.random.seed(experiment_config['random_seed'])

        self.prediction_data_file = Path(experiment_config['data_source'])

        self.prediction_true_values = experiment_config['data']['true_values']
        self.prediction_false_values = experiment_config['data']['false_values']

        self.boolean_features = experiment_config['data']['bool_type_features']
        self.other_data_type_features = experiment_config['data']['other_type_features']

        self.prediction_data_rows = experiment_config['data']['selected_features']
        self.target_column = experiment_config['data']['target']
        self.split_column = experiment_config['data']['data_split_column']

        self.prediction_data: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self.prediction_data_file,
            dtype=self.other_data_type_features,
            usecols=self.prediction_data_rows + [self.target_column, self.split_column],
            true_values=self.prediction_true_values,
            false_values=self.prediction_false_values,
            na_values=experiment_config['data']['na_values'],
        )
        self.prediction_data.dropna(inplace=True)
        self.prediction_data = self.prediction_data.astype(self.boolean_features)

        self.label_encoded_cols = defaultdict(LabelEncoder)

        self.training_data = self.prediction_data.apply(
            lambda x: self.label_encoded_cols[x.name].fit_transform(x)
        )

        self.holdout_split_condition = get_encoding_from_label(
            column=self.split_column,
            label=experiment_config['data']['holdout_split_condition'],
            encoders=self.label_encoded_cols,
        )
        holdout_index = self.training_data[self.split_column] == self.holdout_split_condition

        self.X = self.training_data[~holdout_index][self.prediction_data_rows]
        self.y = self.training_data[~holdout_index][self.target_column]
        self.groups = self.training_data[~holdout_index][self.split_column]

        self.name = experiment_config['experiment']
        self.metric_results = {}

        self._metrics = experiment_config['metrics']
        self.metrics = {metric: SUPPORTED_METRICS[metric] for metric in self._metrics}

        self._classifiers = experiment_config['classifiers']
        self.classifiers = {
            classifier: SUPPORTED_CLASSIFIERS[classifier]
            for classifier in self._classifiers
        }

    def training_set_sample_size(self):
        return self.training_data[self.target_column].size

    def get_x_train_test_split(self, train_indices, test_indices):
        return self.X.iloc[train_indices], self.X.iloc[test_indices]

    def get_y_train_test_split(self, train_indices, test_indices):
        return self.y.iloc[train_indices], self.y.iloc[test_indices]

    def add_result(self, metric: str, result: float) -> None:
        if metric not in self.metric_results:
            self.metric_results[metric] = []

        self.metric_results[metric].append(result)
