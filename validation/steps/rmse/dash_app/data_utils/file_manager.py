from pathlib import Path
import numpy as np
import pandas as pd

class FileManager:
    def __init__(self, path_to_recording_folder: Path):
        self.path_to_recording_folder = path_to_recording_folder
        self.run()

    def run(self):
        self._build_paths()
        self._load_data()

    def _build_paths(self):
        path_to_output_data_folder = self.path_to_recording_folder/'output_data'
        self.path_to_freemocap_array = path_to_output_data_folder/'mediapipe_body_3d_xyz_transformed.npy'
        self.path_to_qualisys_array = self.path_to_recording_folder/'qualisys'/'clipped_qualisys_skel_3d.npy'
        self.rmse_csv_path = path_to_output_data_folder/'rmse_dataframe.csv'
        self.absolute_error_csv_path = path_to_output_data_folder/'absolute_error_dataframe.csv'

    def _load_data(self):
        self.freemocap_data = self._load_if_exists(self.path_to_freemocap_array)
        self.qualisys_data = self._load_if_exists(self.path_to_qualisys_array)
        self.rmse_dataframe = self._load_csv_if_exists(self.rmse_csv_path)
        self.absolute_error_dataframe = self._load_csv_if_exists(self.absolute_error_csv_path)

    def _load_if_exists(self, path: Path):
        if path.exists():
            return np.load(path)
        else:
            print(f"Warning: File {path} does not exist.")
            return None

    def _load_csv_if_exists(self, path: Path):
        if path.exists():
            return pd.read_csv(path)
        else:
            print(f"Warning: File {path} does not exist.")
            return None