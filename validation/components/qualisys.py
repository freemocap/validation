from validation.datatypes.data_component import DataComponent
from validation.utils.io_helpers import load_numpy, save_numpy, load_qualisys_timestamp_from_tsv, load_qualisys_tsv, load_csv, save_csv, save_parquet, return_path_only, empty_saver

#----
# The components in this section below are used for temporal/spatial alignment - which is present, but not run in this public release. They are left here for public availablity. 

QUALISYS_MARKERS = DataComponent(
    name="qualisys_markers",
    filename="qualisys_exported_markers.tsv",
    relative_path="qualisys",
    loader=load_qualisys_tsv,
)

QUALISYS_START_TIME =  DataComponent(
    name="qualisys_start_time",
    filename="qualisys_exported_markers.tsv",
    relative_path="qualisys",
    loader=load_qualisys_timestamp_from_tsv
)

QUALISYS_SYNCED_JOINT_CENTERS = DataComponent(
    name="qualisys_synced_joint_centers",
    filename = "qualisys_body_3d_xyz.npy",
    relative_path = "qualisys",
    loader = load_numpy,
    saver = save_numpy
)

QUALISYS_SYNCED_MARKER_DATA = DataComponent(
    name="qualisys_synced_marker_data",
    filename = "qualisys_synced_markers.csv",
    relative_path = "qualisys",
    loader = load_csv,
    saver = save_csv,
)

#-------

QUALISYS_COM = DataComponent(
    name = "qualisys_center_of_mass",
    filename = "qualisys_body_total_body_com.npy",
    relative_path = "qualisys",
    loader=load_numpy,
    saver = save_numpy
)

QUALISYS_PARQUET = DataComponent(
    name = "qualisys_parquet",
    filename = "freemocap_data_by_frame.parquet",
    relative_path = "qualisys/aligned_3d_data",
    loader = return_path_only,
    saver = empty_saver
)

QUALISYS_JOINT_ANGLES = DataComponent(
    name = "qualisys_joint_angles",
    filename = "qualisys_joint_angles.csv",
    relative_path = "qualisys/analysis_outputs/joint_angles",
    loader = load_csv,
    saver = save_csv
)

QUALISYS_GAIT_EVENTS = DataComponent(
    name = "qualisys_gait_events",
    filename = "qualisys_gait_events.csv",
    relative_path = "qualisys/analysis_outputs/gait_events",
    loader = load_csv,
    saver = save_csv
)