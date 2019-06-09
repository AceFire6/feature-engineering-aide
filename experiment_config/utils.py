import csv
from datetime import datetime
from pathlib import Path

from typing import Dict, List

from sklearn.preprocessing import LabelEncoder


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


def get_random_seed(now: datetime) -> int:
    # Max value is 10 digits long
    # Only experiments generated at the same second with have the same seed
    return int(now.timestamp())


def get_encoding_from_label(column: str, label: str, encoders: Dict[str, LabelEncoder]) -> str:
    return encoders[column].transform([label])[0]