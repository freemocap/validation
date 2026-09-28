import numpy as np
from scipy.stats import chi2


def calculate_ellipse_area(com_data, condition_info_dict:dict):
    ellipse_areas = {}

    for condition_name, condition_data in condition_info_dict.items():
        start_frame, end_frame = condition_data["frames"]

        com_data_sliced = com_data[start_frame: end_frame]

        x = com_data_sliced[:, 0, 0]
        y = com_data_sliced[:, 0, 1]

        x = x - np.mean(x)
        y = y - np.mean(y)

        cov_matrix = np.cov(x, y)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)

        chi2_val = chi2.ppf(0.95, df=2)

        a = np.sqrt(eigenvalues.max() * chi2_val)
        b = np.sqrt(eigenvalues.min() * chi2_val)

        area = np.pi * a * b

        ellipse_areas[condition_name] = area

    return ellipse_areas