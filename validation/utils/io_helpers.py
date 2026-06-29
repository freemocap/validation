import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import plotly.graph_objects as go


def return_path_only(path: Path) -> Path:
    return path

def empty_saver(path:Path, data):
    return path

def load_csv(path: Path):
    return pd.read_csv(path)

def save_csv(path: Path, df:pd.DataFrame):
    # path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def save_parquet(path:Path, df: pd.DataFrame):
    df.to_parquet(path)

def save_numpy(path:Path, array:np.array):
    np.save(path, array)

def load_numpy(path:Path):
    return np.load(path)

def save_plotly_fig(path:Path, fig:go.Figure):
    fig.write_html(path)

def load_qualisys_tsv(path_to_tsv:Path):
    header_length = get_header_length(path_to_tsv)
    data = pd.read_csv(
        path_to_tsv,
        delimiter='\t',
        skiprows=header_length
    )
    return data

def load_qualisys_timestamp_from_tsv(path_to_tsv:Path):
    with open(path_to_tsv, 'r') as file:
        for line in file:
            if line.startswith('TIME_STAMP'):
                timestamp_str = line.strip().split('\t')[1]
                datetime_obj = datetime.strptime(timestamp_str, '%Y-%m-%d, %H:%M:%S.%f')
                return datetime_obj.timestamp()
    raise ValueError(f"No TIME_STAMP found in file: {path_to_tsv}")


def get_header_length(path:Path) -> int:
    """Determine the length of the header in the TSV file."""
    with path.open('r') as file:
        for i, line in enumerate(file):
            if line.startswith('TRAJECTORY_TYPES'):
                return i + 1
    raise ValueError("Header not found in the TSV file.")
