import numpy as np
import pandas as pd
from .convert_data import extract_and_convert_data
from .marker_lists import mediapipe_markers, qualisys_markers, markers_to_extract

def load_npy_data(file_path):
    try:
        data = np.load(file_path)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    
def combine_freemocap_and_qualisys_into_dataframe(freemocap_3d_data:np.ndarray, qualisys_3d_data:np.ndarray):
    # Extract markers and convert to dataframe
    freemocap_dataframe = extract_and_convert_data(freemocap_3d_data, mediapipe_markers, markers_to_extract)
    qualisys_dataframe = extract_and_convert_data(qualisys_3d_data, qualisys_markers, markers_to_extract)
    
    # Add system labels
    freemocap_dataframe['system'] = 'freemocap'
    qualisys_dataframe['system'] = 'qualisys'
    
    # Combine dataframes
    dataframe_of_3d_data = pd.concat([freemocap_dataframe, qualisys_dataframe], ignore_index=True)
    
    return dataframe_of_3d_data


def load_and_process_data(path_to_freemocap_array, path_to_qualisys_array):
    # Load 3D data arrays
    freemocap_3d_data = load_npy_data(path_to_freemocap_array)
    qualisys_3d_data = load_npy_data(path_to_qualisys_array)
    
    # Check if data is loaded successfully
    if freemocap_3d_data is None or qualisys_3d_data is None:
        raise ValueError("Data could not be loaded.")

    # Extract markers and convert to dataframe
    freemocap_dataframe = extract_and_convert_data(freemocap_3d_data, mediapipe_markers, markers_to_extract)
    qualisys_dataframe = extract_and_convert_data(qualisys_3d_data, qualisys_markers, markers_to_extract)
    
    # Add system labels
    freemocap_dataframe['system'] = 'freemocap'
    qualisys_dataframe['system'] = 'qualisys'
    
    # Combine dataframes
    dataframe_of_3d_data = pd.concat([freemocap_dataframe, qualisys_dataframe], ignore_index=True)
    
    return dataframe_of_3d_data