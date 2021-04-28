#! /usr/bin/env python
from datetime import datetime
from pathlib import Path

import pandas as pd
import toml
from InquirerPy import prompt

from experiment_config.settings import (
    DATA_TYPE_CHOICES,
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_METRICS,
    ORDINAL,
)
from experiment_config.utils import (
    extract_headings,
    get_entries_from_csv_row,
    set_selected_features,
)

# These are set up in process_data_file
data_source_headings = []
selected_features = []

data_source_questions = [
    {
        'type': 'input',
        'name': 'experiment_name',
        'message': 'Specify the name of the experiment',
        'validate': lambda answer: (
            'Experiment with that name already exists'
            if Path(f'{answer}.toml').exists()
            else True
        ),
    },
    {
        'type': 'filepath',
        'name': 'data_source',
        'message': 'Specify path to a CSV data file',
        'validate': lambda answer: Path(answer).exists(),
        'invalid_message': 'Enter a valid path',
        'filter': lambda answer: extract_headings(answer, data_source_headings),
    },
    {
        'type': 'input',
        'name': 'field_titles',
        'message': (
            'Headings not found in file. '
            'Enter field titles as comma-separated values'
        ),
        'when': lambda answers: len(data_source_headings) == 0,
        'filter': lambda answer: data_source_headings.extend(get_entries_from_csv_row(answer)),
    },
    {
        'type': 'checkbox',
        'name': 'selected_features',
        'message': 'Select which of the columns you wish to use as features',
        'choices': data_source_headings,
        'filter': lambda answer: set_selected_features(answer, selected_features),
    },
    {
        'type': 'list',
        'name': 'target',
        'message': 'Select which column you wish to use as the target',
        'choices': (
            name
            for name in data_source_headings
            if name not in selected_features
        ),
    },
    {
        'type': 'list',
        'name': 'data_split_column',
        'message': 'Select the column on which to test the condition to create the holdout dataset',
        'choices': data_source_headings,
    },
    {
        'type': 'input',
        'name': 'holdout_split_condition',
        'message': 'Enter condition on which to test the condition to create the holdout dataset',
    },
]

data_source_answers = prompt(data_source_questions)

all_features = (
    selected_features
    + [data_source_answers['target'], data_source_answers['data_split_column']]
)
feature_data_type_questions = [
    *[
        {
            'type': 'checkbox',
            'name': feature_type,
            'message': f'Select the columns that are {feature_name}',
            'choices': all_features,
        } for feature_name, feature_type in DATA_TYPE_CHOICES.items()
    ],
]
feature_type_answers = prompt(feature_data_type_questions)

feature_questions = [
    {
        'type': 'input',
        'name': 'true_values',
        'message': 'Values to consider True (as CSV)',
        'filter': get_entries_from_csv_row,
    },
    {
        'type': 'input',
        'name': 'false_values',
        'message': 'Values to consider False (as CSV)',
        'filter': get_entries_from_csv_row,
    },
]
feature_answers = prompt(feature_questions)

data = pd.read_csv(
    filepath_or_buffer=data_source_answers['data_source'],
    usecols=all_features,
    true_values=feature_answers['true_values'],
    false_values=feature_answers['false_values'],
)

possible_na_values = {}
for feature in all_features:
    possible_na_values[feature] = []

    unique_values = data[feature].unique()
    for value in unique_values:
        if isinstance(value, str):
            if not value.isalnum():
                possible_na_values[feature].append(value)

na_value_questions = [
    *[
        {
            'type': 'checkbox',
            'name': feature,
            'message': f'Select NA values for: {feature}',
            'choices': values,
        } for feature, values in possible_na_values.items() if values
    ],
]
na_value_answers = prompt(na_value_questions)

data = data.replace(na_value_answers, pd.NA).dropna()

if ORDINAL in feature_type_answers:
    ordinals = {}
    for ordinal_feature in feature_type_answers[ORDINAL]:
        options = ', '.join(option for option in data[ordinal_feature].unique())
        ordinal_ordering = prompt({
            'type': 'input',
            'name': ordinal_feature,
            'message': f'Select the order of the ordinal values in column {ordinal_feature} - {options=} (as CSV)',
            'filter': get_entries_from_csv_row,
        })
        ordinals.update(ordinal_ordering)

    feature_type_answers[ORDINAL] = ordinals


classifier_question = [
    {
        'type': 'checkbox',
        'name': 'classifiers',
        'message': 'Select Classifiers',
        'choices': SUPPORTED_CLASSIFIERS.keys(),
        'validate': lambda result: len(result) > 0,
        'invalid_message': 'Choose at least 1 classifier',
    },
]
classifier_choices = prompt(classifier_question)

metric_question = [
    {
        'type': 'checkbox',
        'name': 'metrics',
        'message': 'Select Metrics',
        'choices': SUPPORTED_METRICS.keys(),
        'validate': lambda result: len(result) > 0,
        'invalid_message': 'Choose at least 1 metric',
    },
]
metric_choices = prompt(metric_question)

experiment_name = data_source_answers['experiment_name']

now = datetime.now()

config = {
    'timestamp': now.isoformat(),
    'random_seed': 0,
    'experiment': experiment_name,
    'data_source': data_source_answers['data_source'],
    'data': {
        'selected_features': data_source_answers['selected_features'],
        'target': data_source_answers['target'],
        'data_split_column': data_source_answers['data_split_column'],
        'holdout_split_condition': data_source_answers['holdout_split_condition'],
        'na_values': na_value_answers,
        'features': feature_type_answers,
        **feature_answers,
    },
    **classifier_choices,
    **metric_choices,
}

with open(f'{experiment_name}.toml', 'w') as config_file:
    toml.dump(config, config_file)
