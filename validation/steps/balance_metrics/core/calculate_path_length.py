import numpy as np

def calculate_path_length(sliced_freemocap_data):
    diffs = np.diff(sliced_freemocap_data, axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=-1))
    return path_length 


def calculate_path_lengths(com_data, condition_info_dict:dict):
    path_lengths = {}

    for condition_name, condition_data in condition_info_dict.items():
        start_frame, end_frame = condition_data["frames"]
        label = condition_data["label"]

        sliced_com = com_data[start_frame:end_frame - 1]

        path_lengths[condition_name] = calculate_path_length(
            sliced_com
        )

    return path_lengths