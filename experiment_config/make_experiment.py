#! /usr/bin/env python
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas
import toml

from experiment_config.settings import (
    DATA_TYPE_CHOICES,
    styled_prompt,
    SUPPORTED_CLASSIFIERS,
    SUPPORTED_METRICS,
)
from experiment_config.utils import (
    extract_headings,
    get_entries_from_csv_row,
    get_random_seed,
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
        'type': 'input',
        'name': 'data_source',
        'message': 'Specify path to a CSV data file',
        'validate': lambda answer: 'Enter a valid path' if not Path(answer).exists() else True,
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
        'choices': ({'name': name} for name in data_source_headings),
        'filter': lambda answer: set_selected_features(answer, selected_features),
    },
    {
        'type': 'list',
        'name': 'target',
        'message': 'Select which column you wish to use as the target',
        'choices': (
            {'name': name}
            for name in data_source_headings
            if name not in selected_features
        ),
    },
]

data_source_answers = styled_prompt(data_source_questions)

all_features = selected_features + [data_source_answers['target']]
feature_data_type_questions = [
    *[
        {
            'type': 'list',
            'name': feature,
            'message': f'Select the data type for column: {feature}',
            'choices': DATA_TYPE_CHOICES.keys(),
            'filter': lambda answer: DATA_TYPE_CHOICES[answer],
        } for feature in all_features
    ],
]
feature_type_answers = styled_prompt(feature_data_type_questions)

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
feature_answers = styled_prompt(feature_questions)

bool_type_features = {}
other_type_features = {}
for feature, type_name in feature_type_answers.items():
    if type_name == DATA_TYPE_CHOICES['Boolean']:
        bool_type_features[feature] = type_name
    else:
        other_type_features[feature] = type_name

data = pandas.read_csv(
    filepath_or_buffer=data_source_answers['data_source'],
    usecols=all_features,
    true_values=feature_answers['true_values'],
    false_values=feature_answers['false_values'],
)

possible_na_values = {}
for feature in feature_type_answers.keys():
    possible_na_values[feature] = []

    unique_values = data[feature].unique()
    for value in unique_values:
        if isinstance(value, str):
            if not value.isalnum():
                possible_na_values[feature].append(value)

# for feature in feature_type_answers.keys():
#     unique_values = data[feature].value_counts(normalize=True).to_dict()
#
#     unique_value_type_frequency = defaultdict(lambda: {'values': [], 'frequency': 0})
#     for value, frequency in unique_values.items():
#         value_type = type(value)
#         unique_value_type_frequency[value_type]['frequency'] += frequency
#         unique_value_type_frequency[value_type]['values'].append(value)
#
#     possible_na_values[feature] = []
#     for values_type, values_aggs in unique_value_type_frequency.items():
#         if values_aggs['frequency'] > 0.2:
#             continue
#
#         possible_na_values[feature].extend(values_aggs['values'])


na_value_questions = [
    *[
        {
            'type': 'checkbox',
            'name': feature,
            'message': f'Select NA values for: {feature}',
            'choices': [{'name': name} for name in values],
        } for feature, values in possible_na_values.items() if values
    ],
]
na_value_answers = styled_prompt(na_value_questions)


def select_classifiers() -> Dict[str, Any]:
    error_message = 'Choose at least 1 classifier'

    classifier_questions = [
        {
            'type': 'checkbox',
            'name': 'classifiers',
            'message': 'Select Classifiers',
            'choices': [{'name': name} for name in SUPPORTED_CLASSIFIERS.keys()],
        },
    ]

    classifier_answers = None

    while classifier_answers is None or len(classifier_answers['classifiers']) == 0:
        classifier_answers = styled_prompt(classifier_questions)

        if len(classifier_answers['classifiers']) == 0:
            print(error_message)

    return classifier_answers


classifier_choices = select_classifiers()


def select_metrics() -> Dict[str, Any]:
    error_message = 'Choose at least 1 metric'

    classifier_questions = [
        {
            'type': 'checkbox',
            'name': 'metrics',
            'message': 'Select Metrics',
            'choices': [{'name': name} for name in SUPPORTED_METRICS.keys()],
        },
    ]

    classifier_answers = None

    while classifier_answers is None or len(classifier_answers['metrics']) == 0:
        classifier_answers = styled_prompt(classifier_questions)

        if len(classifier_answers['metrics']) == 0:
            print(error_message)

    return classifier_answers


metric_choices = select_metrics()

experiment_name = data_source_answers['experiment_name']

now = datetime.now()

config = {
    'timestamp': now.isoformat(),
    'random_seed': get_random_seed(now),
    'experiment': experiment_name,
    'data_source': data_source_answers['data_source'],
    'data': {
        'selected_features': data_source_answers['selected_features'],
        'target': data_source_answers['target'],
        'bool_type_features': bool_type_features,
        'other_type_features': other_type_features,
        'na_values': na_value_answers,
        **feature_answers,
    },
    **classifier_choices,
    **metric_choices,
}

with open(f'{experiment_name}.toml', 'w') as config_file:
    toml.dump(config, config_file)
