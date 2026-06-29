import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import random


def optimize_transformation_least_squares(transformation_matrix_guess, data_to_transform, reference_data):
    tx, ty, tz, rx, ry, rz, s = transformation_matrix_guess
    s = 1
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    transformed_data = s * rotation.apply(data_to_transform) + np.array([tx, ty, tz])
    residuals = reference_data - transformed_data
    return residuals.flatten()

def run_least_squares_optimization(data_to_transform, reference_data, initial_guess=[0,0,0,0,0,0,1]):
    result = least_squares(optimize_transformation_least_squares, initial_guess, args=(data_to_transform, reference_data), gtol=1e-10, verbose=1)
    return result.x

def apply_transformation(transformation_matrix, data):
    tx, ty, tz, rx, ry, rz, s = transformation_matrix
    s = 1
    rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    transformed_data = s * rotation.apply(data.reshape(-1, 3)) + np.array([tx, ty, tz])
    return transformed_data.reshape(data.shape)

def get_best_transformation_matrix_ransac(freemocap_data, qualisys_data, frames_to_sample=10, initial_guess=[0,0,0,0,0,0,1], max_iterations=10, inlier_threshold=70):
    if freemocap_data.shape[1] != qualisys_data.shape[1]:
        raise ValueError("The number of markers in freemocap_data and qualisys_data must be the same.")
    
    num_frames = freemocap_data.shape[0]
    all_frames = list(range(num_frames))
    
    best_inliers = []
    best_transformation_matrix = None

    for iteration in range(max_iterations):
        # Randomly sample frames
        sampled_frames = random.sample(all_frames, frames_to_sample)
        
        # Prepare data for least squares optimization
        sampled_freemocap = freemocap_data[sampled_frames, :, :]
        sampled_qualisys = qualisys_data[sampled_frames, :, :]
        
        # Flatten the data for optimization
        flattened_freemocap = sampled_freemocap.reshape(-1, 3)
        flattened_qualisys = sampled_qualisys.reshape(-1, 3)
        
        # Fit transformation matrix using least squares
        transformation_matrix = run_least_squares_optimization(
            data_to_transform=flattened_freemocap, 
            reference_data=flattened_qualisys,
            initial_guess=initial_guess
        )
        
        # Apply the transformation to the entire dataset
        transformed_freemocap_data = apply_transformation(transformation_matrix, freemocap_data)
        
        # Calculate the alignment error for each frame
        errors = np.linalg.norm(qualisys_data - transformed_freemocap_data, axis=2).mean(axis=1)
        
        # Determine inliers based on the error threshold
        inliers = np.where(errors < inlier_threshold)[0]
        
        # Update the best model if the current model has more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transformation_matrix = transformation_matrix
    
    # Apply the best transformation matrix to align the entire dataset
    if best_transformation_matrix is not None:
        # aligned_freemocap_data = apply_transformation(best_transformation_matrix, freemocap_data)
        return best_transformation_matrix
    else:
        raise ValueError("RANSAC failed to find a valid transformation.")
