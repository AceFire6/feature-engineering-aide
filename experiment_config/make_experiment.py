#! /usr/bin/env python
from datetime import datetime
from pathlib import Path

import pandas as pd
import toml
from InquirerPy import inquirer

from experiment_config.settings import (
    DATA_TYPE_CHOICES,
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_METRICS,
    ORDINAL,
)
from experiment_config.utils import (
    get_entries_from_csv_row,
    has_headings,
    is_not_empty,
)

experiment_name = inquirer.text(
    message='Specify the name of the experiment',
    validate=lambda answer: not Path(f'{answer}.toml').exists(),
    invalid_message='Experiment with that name already exists!',
).execute()

data_source = inquirer.filepath(
    message='Specify path to a CSV data file',
    validate=lambda answer: Path(answer).exists() and Path(answer).is_file(),
    invalid_message='Enter a valid CSV file path!',
).execute()

if not has_headings(data_source):
    file_headings = inquirer.text(
        message='Headings not found in file. Enter field titles as comma-separated values',
        filter=lambda answer: get_entries_from_csv_row(answer),
        validate=is_not_empty,
        invalid_message='Headings are required!',
    ).execute()
    dataset = pd.read_csv(data_source, names=file_headings)
else:
    # Headings are inferred
    dataset = pd.read_csv(data_source)

dataset_columns = list(dataset.columns)

selected_columns = inquirer.checkbox(
    message='Select which of the columns you wish to use for this experiment',
    choices=dataset_columns,
    validate=lambda answer: len(answer) > 1,
    invalid_message='You must select at least two columns!',
).execute()

true_values = inquirer.text(
    message='Values to consider True (as CSV)',
    filter=get_entries_from_csv_row,
    validate=is_not_empty,
).execute()
false_values = inquirer.text(
    message='Values to consider False (as CSV)',
    filter=get_entries_from_csv_row,
    validate=is_not_empty,
).execute()

possible_na_values = {}
for feature in selected_columns:
    possible_na_values[feature] = []

    unique_values = dataset[feature].unique()
    for value in unique_values:
        if isinstance(value, str) and not value.isalnum():
            possible_na_values[feature].append(value)

    # Remove features with not possible NA values
    if not possible_na_values[feature]:
        del possible_na_values[feature]

additional_na_values = {}
for feature, na_values in possible_na_values.items():
    selected_na_values = inquirer.checkbox(
        message=f'Select additional NA values for: {feature}',
        choices=na_values,
    ).execute()

    if selected_na_values:
        additional_na_values[feature] = selected_na_values

# Replace additional NA value options from dataset with NA
dataset = dataset.replace(additional_na_values, pd.NA)

selected_features = inquirer.checkbox(
    message='Select which of the columns you wish to use as training features',
    choices=selected_columns,
    validate=is_not_empty,
    invalid_message='You must select at least one column!',
).execute()
remaining_columns = [col for col in selected_columns if col not in selected_features]

target = inquirer.select(
    message='Select which column you wish to use as the target',
    choices=remaining_columns,
    validate=is_not_empty,
    invalid_message='You must select at least one column!',
).execute()
remaining_columns.remove(target)

holdout_split_column = inquirer.select(
    message='Select the column on which to test the condition to create the holdout dataset',
    choices=remaining_columns,
).execute()

possible_split_values = dataset[holdout_split_column].unique()
holdout_split_condition = inquirer.select(
    message='Enter condition on which to test the condition to create the holdout dataset',
    choices=possible_split_values,
).execute()

selectable_features = selected_features + [target, holdout_split_column]
feature_dtype_map = {}
for feature_name, feature_type in DATA_TYPE_CHOICES.items():
    dtype_features = inquirer.checkbox(
        message=f'Select the columns that are {feature_name}',
        choices=selectable_features,
    ).execute()

    if dtype_features:
        feature_dtype_map[feature_type] = dtype_features
        # Reduce the options as we select features
        selectable_features = [feature for feature in selectable_features if feature not in dtype_features]

if ORDINAL in feature_dtype_map:
    ordinals = {}
    for ordinal_feature in feature_dtype_map[ORDINAL]:
        unique_ord_values = dataset[ordinal_feature].dropna().unique()
        options = ', '.join(unique_ord_values)

        ordinal_ordering = inquirer.text(
            message=f'Select the order of the ordinal values in column {ordinal_feature} - {options=} (as CSV)',
            filter=get_entries_from_csv_row,
            validate=lambda answer: len(answer.split(',')) == len(unique_ord_values),
            invalid_message=f'You need to include all {len(unique_ord_values)} value(s)!',
        ).execute()
        ordinals[ordinal_feature] = ordinal_ordering

    feature_dtype_map[ORDINAL] = ordinals


classifier_choices = inquirer.checkbox(
    message='Select Classifiers',
    choices=SUPPORTED_CLASSIFIERS.keys(),
    validate=is_not_empty,
    invalid_message='Choose at least 1 classifier!',
).execute()

metric_choices = inquirer.checkbox(
    message='Select Metrics',
    choices=SUPPORTED_METRICS.keys(),
    validate=is_not_empty,
    invalid_message='Choose at least 1 metric!',
).execute()

now = datetime.now()
config = {
    'timestamp': now.isoformat(),
    'random_seed': 0,
    'experiment': experiment_name,
    'data_source': data_source,
    'data': {
        'selected_features': selected_features,
        'target': target,
        'holdout_split_column': holdout_split_column,
        'holdout_split_condition': holdout_split_condition,
        'na_values': additional_na_values,
        'true_values': true_values,
        'false_values': false_values,
        'features': feature_dtype_map,
    },
    'classifiers': classifier_choices,
    'metrics': metric_choices,
}

with open(f'{experiment_name}.toml', 'w') as config_file:
    toml.dump(config, config_file)
