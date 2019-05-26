import csv
from pathlib import Path

from typing import List


def extract_headings(data_source: str, headings: List[str]) -> str:
    csv_sniffer = csv.Sniffer()

    data_source_path = Path(data_source)

    # Use newline='' to preserve newlines
    with data_source_path.open(newline='') as data_file:
        file_sample = data_file.read(1024)
        has_headings = csv_sniffer.has_header(file_sample)

        if not has_headings:
            return data_source

        dialect = csv_sniffer.sniff(file_sample)
        data_file.seek(0)
        reader = csv.reader(data_file, dialect)

        csv_headings = next(reader)

        headings.extend(csv_headings)

    return data_source


def get_entries_from_csv_row(headings_csv: str) -> List[str]:
    return [part.strip() for part in headings_csv.split(',')]


def set_selected_features(input_features: List[str], selected_features: List) -> List[str]:
    selected_features.extend(input_features)
    return input_features
