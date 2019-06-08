from collections import defaultdict
from pathlib import Path
import sys

from dotmap import DotMap
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import toml

from experiment_config.settings import SUPPORTED_CLASSIFIERS, SUPPORTED_METRICS


experiment_input_paths = sys.argv[1:]
if not experiment_input_paths:
    print('Please pass in experiment files as arguments to this script')

experiment_file_paths = [Path(experiment_file) for experiment_file in experiment_input_paths]

experiment_configs = []

for experiment_file_path in experiment_file_paths:
    if experiment_file_path.is_dir():
        print(f'Cannot handle {experiment_file_path.absolute()} as it is a directory!')
        continue

    experiment_config = toml.load(experiment_file_path)
    experiment_config_dot_dict = DotMap(experiment_config)
    experiment_configs.append(experiment_config_dot_dict)

for experiment_config in experiment_configs:
    prediction_data_file = Path(experiment_config.data_source)

    prediction_true_values = experiment_config.data.true_values
    prediction_false_values = experiment_config.data.false_values

    boolean_features = experiment_config.data.bool_type_features
    other_data_type_features = experiment_config.data.other_type_features

    prediction_data_rows = experiment_config.data.selected_features
    target_column = experiment_config.data.target

    prediction_data: pd.DataFrame = pd.read_csv(
        filepath_or_buffer=prediction_data_file,
        dtype=other_data_type_features,
        usecols=prediction_data_rows + [target_column],
        true_values=prediction_true_values,
        false_values=prediction_false_values,
        na_values=experiment_config.data.na_values,
        keep_default_na=False,
    )
    prediction_data = prediction_data.astype(boolean_features)
    prediction_data.dropna(inplace=True)

    label_encoded_cols = defaultdict(LabelEncoder)

    training_data = prediction_data.apply(
        lambda x: label_encoded_cols[x.name].fit_transform(x)
    )

    X = training_data[prediction_data_rows]
    y = training_data[target_column]

    leave_one_out = LeaveOneGroupOut()

    result_metrics = {metric: [] for metric in experiment_config.metrics}
    classifiers = {
        classifier: SUPPORTED_CLASSIFIERS[classifier]
        for classifier in experiment_config.classifiers
    }

    for classifier_name, classifier_class in classifiers.items():
        print(f'Running {classifier_name}')
        for train_index, test_index in tqdm(leave_one_out.split(X, y, X['Application Year'])):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            classifier = classifier_class()
            classifier.fit(X_train, y_train)

            y_hat = classifier.predict(X_test)

            for metric in result_metrics.keys():
                metric_function = SUPPORTED_METRICS[metric]
                result_metrics[metric].append(metric_function(y_test, y_hat))

        print(classifier_name, result_metrics)
