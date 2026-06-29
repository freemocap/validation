import numpy as np
import pandas as pd
from skellymodels.managers.human import Human
from skellymodels.models.trajectory import Trajectory
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

@dataclass
class XZCoordinateReferences:
    z: tuple[str, str]
    x: tuple[str, str]

@dataclass
class FootCoordinateReferences:
    y: tuple[str, str]

coordinate_systems = {
    'right_foot': FootCoordinateReferences(
        y = ('right_heel', 'right_foot_index')
    ),

    'left_foot': FootCoordinateReferences(
        y = ('left_heel', 'left_foot_index')
    ),

    'right_shank': XZCoordinateReferences(
        z = ('right_ankle', 'right_knee'),
        x = ('left_hip', 'right_hip')
    ),

    'left_shank': XZCoordinateReferences(
        z = ('left_ankle', 'left_knee'),
        x = ('left_hip', 'right_hip')
    ),

    'right_thigh': XZCoordinateReferences(
        z = ('right_knee', 'right_hip'),
        x = ('left_hip', 'right_hip')
    ),

    'left_thigh': XZCoordinateReferences(
        z = ('left_knee', 'left_hip'),
        x = ('left_hip', 'right_hip')
    ),

    'left_hip': XZCoordinateReferences(
        z = ('hips_center', 'neck_center'),
        x = ('left_hip', 'right_hip')
    ),
    'right_hip': XZCoordinateReferences(
        z = ('hips_center', 'neck_center'),
        x = ('left_hip', 'right_hip')
    ),
}

joint_angle_setup = {
    'right_knee': ['right_thigh', 'right_shank'],
    'right_ankle': ['right_shank', 'right_foot'],
    'left_knee': ['left_thigh', 'left_shank'],
    'left_ankle': ['left_shank', 'left_foot'],
    'left_hip': ['left_hip', 'left_thigh'],
    'right_hip': ['right_hip', 'right_thigh'],
}

def get_segment_rotation(joints:Trajectory, ref:XZCoordinateReferences):
    num_frames = joints.as_array.shape[0]
    x1 = joints.as_dict[ref.x[0]]
    x2 = joints.as_dict[ref.x[1]]
    z1 = joints.as_dict[ref.z[0]]
    z2 = joints.as_dict[ref.z[1]]

    x = x2 - x1
    z = z2 - z1

    x = norm(x)
    zhat = norm(z)

    yhat = norm(np.cross(zhat,x))
    xhat = norm(np.cross(yhat, zhat))

    R_segment = np.zeros((num_frames, 3, 3))
    R_segment[:,:,0] = xhat
    R_segment[:,:,1] = yhat
    R_segment[:,:,2] = zhat

    return R_segment

def subtract_neutral(angles:np.ndarray, neutral_frames:range) -> np.ndarray:
    neutral_mean = np.mean(angles[neutral_frames], axis=0)
    return angles - neutral_mean

def norm(v, eps=1e-12):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def get_foot_coordinate_system(joints:Trajectory, 
                               refs:FootCoordinateReferences,
                               x_ref: np.ndarray|None):
    num_frames = joints.as_array.shape[0]

    y1 = joints.as_dict[refs.y[0]]
    y2 = joints.as_dict[refs.y[1]]
    y_hat = norm((y2 - y1))
    

    # # Constrain ML to ground plane: #outdates as of 01/23/26 (causing issues in running when foot was nearly vertical)
    # # First compute an x candidate from the global up vector
    # up = np.array([0.0, 0.0, 1.0]) 
    # x_raw = np.cross(y_hat, up)
    # x_hat = norm(x_raw)

    x_projection = x_ref - np.sum(x_ref * y_hat, axis=1, keepdims=True) * y_hat
    x_hat = norm(x_projection)

    z_hat = norm(np.cross(x_hat, y_hat))
    x_hat = norm(np.cross(y_hat, z_hat))

    R_foot = np.empty((num_frames, 3, 3))
    R_foot[:, :, 0] = x_hat  # ML (shank-referenced, projected to this plane)
    R_foot[:, :, 1] = y_hat  # longitudinal
    R_foot[:, :, 2] = z_hat  # vertical of the foot

    return R_foot

def calculate_cardan_angles(R_proximal:np.ndarray,
                            R_distal: np.ndarray):
    
    num_frames = R_proximal.shape[0]
    R_rel = np.empty_like(R_proximal)
    for i in range(num_frames):
        R_rel[i] = R_proximal[i].T @ R_distal[i]
    
    r = R.from_matrix(R_rel)
    cardan_angles = r.as_euler('XYZ', degrees=True) 

    return cardan_angles

def calculate_angle(proximal_orientation: np.ndarray,
                    distal_orientation: np.ndarray):
    angle = calculate_cardan_angles(R_proximal=proximal_orientation,
                                    R_distal=distal_orientation)
    return angle


def split_angle_into_side(joint_name:str):
    components = joint_name.lower().split('_')
    side = None

    if 'left' in components:
        side = 'left'
        components.remove('left')
        joint_base = '_'.join(components)

    elif 'right' in components:
        side = "right"
        components.remove("right")
        joint_base = '_'.join(components).strip('_') #strip removes any leading or trailing underscores if one makes it in

    return side, joint_base or joint_name

def calculate_joint_angles(human: Human,
                           neutral_stance_frames: range|None,
                           use_rigid = True):
    
    COMPONENTS_BY_JOINT = {
    'hip':   ['flex_ext','abd_add','int_ext'],
    'knee':  ['flex_ext','abd_add','int_ext'],
    'ankle': ['dorsi_plantar','inv_ev','int_ext'],
    # fallback for unknowns:
    '_default': ['c1','c2','c3'],
    }

    if use_rigid:
        joints = human.body.rigid_xyz
    else:
        joints = human.body.xyz

    num_frames, num_markers,_ = joints.as_array.shape
    segment_orientations = {}
    for segment_name, refs in coordinate_systems.items():
        if not isinstance(refs, FootCoordinateReferences):
            segment_orientations[segment_name] = get_segment_rotation(joints, refs)


    for foot_name in ["right_foot", "left_foot"]: # use shank ML as foot x reference
        refs = coordinate_systems[foot_name]
        shank_name = "right_shank" if "right" in foot_name else "left_shank"
        x_ref = segment_orientations[shank_name][:,:,0]  

        R_foot = get_foot_coordinate_system(joints, refs, x_ref)
        segment_orientations[foot_name] = R_foot

    joint_angles = {}
    for angle_name, segments in joint_angle_setup.items():
        proximal_orientation = segment_orientations[segments[0]]
        distal_orientation = segment_orientations[segments[1]]
        angle = calculate_angle(proximal_orientation, distal_orientation)
        joint_angles[angle_name] = subtract_neutral(angle, neutral_stance_frames) if neutral_stance_frames is not None else angle


    dfs = []
    num_rows = num_frames*3
    for joint, angles in joint_angles.items():
        side, joint_name = split_angle_into_side(joint)
        df = pd.DataFrame({
            "frame": np.repeat(np.arange(num_frames),3),
            "side": side,
            "joint": joint_name,
            "angle": angles.ravel(),
            "component": COMPONENTS_BY_JOINT[joint_name] * num_frames
        })
        dfs.append(df)
    df_angles = pd.concat(dfs, ignore_index=True)
    return df_angles

