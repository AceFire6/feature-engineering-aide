from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from experiment_config.settings import SUPPORTED_CLASSIFIERS, SUPPORTED_METRICS
from experiment_config.utils import (
    get_encoding_from_label,
    parse_experiment_paths,
    print_metric_results_five_number_summary,
)

experiment_input_paths = sys.argv[1:]
if not experiment_input_paths:
    print('Please pass in experiment files as arguments to this script')

experiment_configs = parse_experiment_paths(experiment_input_paths)

for experiment_config in experiment_configs:
    np.random.seed(experiment_config.random_seed)

    prediction_data_file = Path(experiment_config.data_source)

    prediction_true_values = experiment_config.data.true_values
    prediction_false_values = experiment_config.data.false_values

    boolean_features = experiment_config.data.bool_type_features
    other_data_type_features = experiment_config.data.other_type_features

    prediction_data_rows = experiment_config.data.selected_features
    target_column = experiment_config.data.target
    split_column = experiment_config.data.data_split_column

    prediction_data: pd.DataFrame = pd.read_csv(
        filepath_or_buffer=prediction_data_file,
        dtype=other_data_type_features,
        usecols=prediction_data_rows + [target_column, split_column],
        true_values=prediction_true_values,
        false_values=prediction_false_values,
        na_values=experiment_config.data.na_values,
    )
    prediction_data.dropna(inplace=True)
    prediction_data = prediction_data.astype(boolean_features)

    label_encoded_cols = defaultdict(LabelEncoder)

    training_data = prediction_data.apply(
        lambda x: label_encoded_cols[x.name].fit_transform(x)
    )

    holdout_split_condition = get_encoding_from_label(
        column=split_column,
        label=experiment_config.data.holdout_split_condition,
        encoders=label_encoded_cols,
    )
    holdout_index = training_data[split_column] == holdout_split_condition

    X = training_data[~holdout_index][prediction_data_rows]
    y = training_data[~holdout_index][target_column]
    groups = training_data[~holdout_index][split_column]

    classifier_result_metrics = {
        classifier: {metric: [] for metric in experiment_config.metrics}
        for classifier in experiment_config.classifiers
    }
    classifiers = {
        classifier: SUPPORTED_CLASSIFIERS[classifier]
        for classifier in experiment_config.classifiers
    }

    leave_one_out = LeaveOneGroupOut()

    for classifier_name, classifier_class in classifiers.items():
        print(f'Running {classifier_name}')
        for train_index, test_index in tqdm(leave_one_out.split(X, y, groups)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            classifier = classifier_class()
            classifier.fit(X_train, y_train)

            y_hat = classifier.predict(X_test)

            for metric in experiment_config.metrics:
                metric_function = SUPPORTED_METRICS[metric]
                metric_result = metric_function(y_test, y_hat)
                classifier_result_metrics[classifier_name][metric].append(metric_result)

    now = f'{datetime.utcnow():%Y-%m-%d_%H:%M:%S}'
    results_file_name = f'{experiment_config.experiment}_automl_mcc_results_{now}.txt'

    with open(results_file_name, 'w') as results_file:
        print(f'n = {training_data[target_column].size}', file=results_file)

        for classifier_name, result_metrics in classifier_result_metrics.items():
            print(classifier_name, file=results_file)

            print_metric_results_five_number_summary(result_metrics, results_file)
