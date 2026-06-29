from validation.datatypes.data_component import DataComponent
from validation.utils.io_helpers import load_csv, load_numpy, save_numpy, save_csv, return_path_only, empty_saver
from validation.utils.save_trcs import save_as_trc

FREEMOCAP_TIMESTAMPS = DataComponent(
    name="freemocap_timestamps",
    filename="{recording_name}_timestamps.csv",
    relative_path="synchronized_videos/timestamps",
    loader=load_csv,
)

FREEMOCAP_PREALPHA_TIMESTAMPS = DataComponent(
    name="freemocap_timestamps",
    filename = "unix_synced_timestamps.csv",
    relative_path = "synchronized_videos/timestamps",
    loader=load_csv,
)

FREEMOCAP_PRE_SYNC_JOINT_CENTERS = DataComponent(
    name = "freemocap_pre_synced",
    filename = '{tracker}_body_3d_xyz.npy',
    relative_path="output_data/{tracker}",
    loader=load_numpy
)

TRANSFORMATION_MATRIX = DataComponent(
    name = "transformation_matrix",
    filename = "transformation_3d.npy",
    relative_path = "{tracker}/aligned_3d_data",
    loader= load_numpy,
    saver = save_numpy
)

FREEMOCAP_JOINT_CENTERS = DataComponent(
    name = "freemocap_aligned_3d",
    filename = "{tracker}_body_3d_xyz.npy",
    relative_path = "validation/{tracker}",
    loader = load_numpy,
    saver = save_numpy
)

FREEMOCAP_RIGID_JOINT_CENTERS = DataComponent(
    name = "freemocap_rigid_aligned_3d",
    filename = "{tracker}_body_rigid_3d_xyz.npy",
    relative_path = "validation/{tracker}",
    loader = load_numpy,
    saver = save_numpy
)

FREEMOCAP_PARQUET = DataComponent(
    name = "freemocap_parquet",
    filename = "freemocap_data_by_frame.parquet",
    relative_path = "{tracker}/aligned_3d_data",
    loader = return_path_only,
    saver = empty_saver
)


FREEMOCAP_LAG = DataComponent(
    name = "freemocap_lag",
    filename = "freemocap_lag.csv",
    relative_path = "validation",
    loader = load_csv,
    saver = save_csv
)

FREEMOCAP_JOINT_ANGLES = DataComponent(
    name = "freemocap_joint_angles",
    filename = "{tracker}_joint_angles.csv",
    relative_path = "validation/{tracker}/joint_angles",
    loader = load_csv,
    saver = save_csv
)

FREEMOCAP_GAIT_EVENTS = DataComponent(
    name = "freemocap_gait_events",
    filename = "{tracker}_gait_events.csv",
    relative_path = "validation/{tracker}/gait_events",
    loader = load_csv,
    saver = save_csv
)