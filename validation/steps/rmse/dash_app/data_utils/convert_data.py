import numpy as np
import pandas as pd

def extract_and_convert_data(data, markers_list, markers_to_extract):
    extracted_data = extract_specific_markers(data_marker_dimension=data, list_of_markers=markers_list, markers_to_extract=markers_to_extract)
    dataframe = convert_3d_array_to_dataframe(data_3d_array=extracted_data, data_marker_list=markers_to_extract)
    return dataframe

def convert_3d_array_to_dataframe(data_3d_array:np.ndarray, data_marker_list:list):
    """
    Convert the FreeMoCap data from a numpy array to a pandas DataFrame.

    Parameters:
    - data_3d_array (numpy.ndarray): The 3d data array. Shape should be (num_frames, num_markers, 3).
    - data_marker_list (list): List of marker names 

    Returns:
    - data_frame_marker_dim_dataframe (pandas.DataFrame): DataFrame containing FreeMoCap data with columns ['Frame', 'Marker', 'X', 'Y', 'Z'].

    """
    num_frames = data_3d_array.shape[0]
    num_markers = data_3d_array.shape[1]

    frame_list = []
    marker_list = []
    x_list = []
    y_list = []
    z_list = []

    for frame in range(num_frames):
        for marker in range(num_markers):
            frame_list.append(frame)
            marker_list.append(data_marker_list[marker])
            x_list.append(data_3d_array[frame, marker, 0])
            y_list.append(data_3d_array[frame, marker, 1])
            z_list.append(data_3d_array[frame, marker, 2])

    data_frame_marker_dim_dataframe = pd.DataFrame({
        'frame': frame_list,
        'marker': marker_list,
        'x': x_list,
        'y': y_list,
        'z': z_list
    })

    return data_frame_marker_dim_dataframe


def extract_specific_markers(data_marker_dimension:np.ndarray, list_of_markers:list, markers_to_extract:list):
    """
    Extracts specific markers for a frame of a 3D data array based on the given indices and markers to extract.

    Parameters:
    - data (numpy.ndarray): The 3D data array containing all markers. Shape should be (num_markers, 3).
    - indices (list): The list of marker names corresponding to the columns in the data array.
    - markers_to_extract (list): The list of marker names to extract.

    Returns:
    - extracted_data (numpy.ndarray): A new 3D data array containing only the extracted markers. 
      Shape will be (num_frames, num_extracted_markers, 3).
    """
    # Identify the column indices that correspond to the markers to extract
    col_indices = [list_of_markers.index(marker) for marker in markers_to_extract if marker in list_of_markers]
    
    # Extract the relevant columns from the data array
    extracted_data = data_marker_dimension[:,col_indices, :]

    return extracted_data