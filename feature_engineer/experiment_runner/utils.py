from collections import Sized
import csv
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TextIO, TypeVar

import functools
import numpy as np

from feature_engineer.experiment_config.experiment import Experiment


DecoratedFunctionResult = TypeVar('DecoratedFunctionResult')
DecoratedFunction = Callable[..., DecoratedFunctionResult]
Decorator = Callable[[DecoratedFunction], Callable[..., DecoratedFunctionResult]]


def has_headings(data_source: str) -> bool:
    csv_sniffer = csv.Sniffer()

    data_source_path = Path(data_source)

    # Use newline='' to preserve newlines
    with data_source_path.open(newline='') as data_file:
        file_sample = data_file.read(1024)

    return csv_sniffer.has_header(file_sample)


def is_not_empty(value: Sized) -> bool:
    return len(value) > 0


def get_entries_from_csv_row(headings_csv: str) -> list[str]:
    return list(map(str.strip, headings_csv.split(',')))


def get_random_seed(now: datetime) -> int:
    # Max value is 10 digits long
    # Only experiments generated at the same second with have the same seed
    return int(now.timestamp())


def print_metric_results_five_number_summary(
    result_metrics: dict[str, list[float]],
    results_file: TextIO,
) -> None:
    for metric, results in result_metrics.items():
        min_result = min(results)
        max_result = max(results)
        q1, median, q3 = np.percentile(results, [25, 50, 75])

        print(f'\t{metric}', file=results_file)
        print(f'\t\t{results}', file=results_file)
        print(
            f'\t\tMin: {min_result}',
            f'Q1: {q1}',
            f'Median: {median}',
            f'Q3: {q3}',
            f'Max: {max_result}',
            sep='\n\t\t',
            end='\n',
            file=results_file,
        )


def make_results_path(experiment: Experiment) -> Path:
    start_time = experiment.start_time

    # Make results directory if it doesn't exist
    results_file_path = experiment.file_path.parent / 'results' / experiment.name
    results_file_path.mkdir(parents=True, exist_ok=True)

    results_file_name = f'{experiment.name}_automl_results_{start_time}.txt'
    results_file_path_with_name = results_file_path / results_file_name

    return results_file_path_with_name


def write_results(experiment: Experiment, *result_lines: str) -> None:
    result_path = make_results_path(experiment)

    with result_path.open('a') as result_file:
        result_file.writelines(result_lines)


def hook_function(pre_hook: Optional[Callable] = None, post_hook: Optional[Callable] = None) -> Decorator:
    def decorator(func: DecoratedFunction) -> Callable[..., DecoratedFunctionResult]:
        @functools.wraps(func)
        def _func_wrapper(*args, **kwargs) -> DecoratedFunctionResult:
            if pre_hook is not None:
                pre_hook(*args, **kwargs)

            result = func(*args, **kwargs)

            if post_hook is not None:
                post_hook(*args, **kwargs)

            return result
        return _func_wrapper
    return decorator


def get_metric_results_five_number_summary(result_metrics: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    metric_results = {}

    for metric, results in result_metrics.items():
        min_result = min(results)
        max_result = max(results)
        q1, median, q3 = np.percentile(results, [25, 50, 75])

        metric_results[metric] = {
            'min': min_result,
            'q1': q1,
            'median': median,
            'q3': q3,
            'max': max_result,
        }

    return metric_results
