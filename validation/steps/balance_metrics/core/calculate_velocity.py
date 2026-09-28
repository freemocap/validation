import numpy as np


def calculate_condition_velocities(
    com_data,
    condition_info_dict: dict,
    sampling_rate: float,
):
    velocities = {}

    for condition_name, condition_data in condition_info_dict.items():
        start_frame, end_frame = condition_data["frames"]

        com_data_sliced = com_data[start_frame:end_frame - 1]

        # mm/frame -> mm/s
        velocity = np.diff(com_data_sliced, axis=0) * sampling_rate

        velocity = np.squeeze(velocity, axis=1)

        velocities[condition_name] = velocity

    return velocities


def calculate_mean_2d_velocity(
    condition_velocities: dict,
):
    mean_velocities = {}

    for condition_name, velocity in condition_velocities.items():
        velocity_x = velocity[:, 0]
        velocity_y = velocity[:, 1]

        velocity_2d = np.sqrt(
            velocity_x**2 + velocity_y**2
        )

        mean_velocities[condition_name] = np.mean(velocity_2d)

    return mean_velocities