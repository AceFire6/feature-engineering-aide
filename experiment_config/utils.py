from collections import Sized
import csv
from datetime import datetime
from pathlib import Path

from typing import TextIO

import numpy as np


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
