from skellymodels.managers.human import Human
from validation.steps.spatial_alignment.config import SpatialAlignmentConfig
from validation.steps.spatial_alignment.core.alignment_utils import get_best_transformation_matrix_ransac, apply_transformation

from typing import List
import logging


def run_marker_check(freemocap_actor:Human,
                     qualisys_actor:Human,
                     markers_for_alignment: List[str]):
    """
    Validates the presence of alignment markers in both FreeMoCap and Qualisys skeleton models.

    This function checks whether all the markers specified for alignment are present in the marker lists
    of both the FreeMoCap and Qualisys skeleton models. If any markers are missing, it raises a ValueError
    with a descriptive message indicating which markers are missing in which model.
    """
    
    freemocap_markers = set(freemocap_actor.body.anatomical_structure.tracked_point_names)
    qualisys_markers = set(qualisys_actor.body.anatomical_structure.tracked_point_names)

    missing_in_freemocap = set(markers_for_alignment) - freemocap_markers
    missing_in_qualisys = set(markers_for_alignment) - qualisys_markers

    if missing_in_freemocap:
        raise ValueError(f"These markers for alignment were not found in FreeMoCap markers: {missing_in_freemocap}")

    if missing_in_qualisys:
        raise ValueError(f"These markers for alignment were not found in Qualisys markers: {missing_in_qualisys}")

def run_ransac_spatial_alignment(freemocap_actor:Human, 
                                 qualisys_actor:Human,
                                 config: SpatialAlignmentConfig,
                                 logger = None,
                                 ):
    
    if logger is None:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        logger = logging.getLogger(__name__)

    markers_for_alignment = config.markers_for_alignment

    run_marker_check(freemocap_actor=freemocap_actor,
                     qualisys_actor=qualisys_actor,
                     markers_for_alignment= config.markers_for_alignment)
    
    freemocap_indices = [freemocap_actor.body.xyz.landmark_names.index(marker) for marker in markers_for_alignment]
    qualisys_indices = [qualisys_actor.body.xyz.landmark_names.index(marker) for marker in markers_for_alignment]
    
    freemocap_data_for_alignment = freemocap_actor.body.xyz.as_array[:,freemocap_indices,:]
    qualisys_data_for_alignment = qualisys_actor.body.xyz.as_array[:, qualisys_indices, :]
    
    best_transformation_matrix = get_best_transformation_matrix_ransac(freemocap_data= freemocap_data_for_alignment,
                                                                       qualisys_data= qualisys_data_for_alignment,
                                                                       frames_to_sample= config.frames_to_sample,
                                                                       max_iterations=config.max_iterations,
                                                                       inlier_threshold=config.inlier_threshold)
    logger.info(f"Found best transformation matrix as {best_transformation_matrix}")

    aligned_freemocap_data = apply_transformation(transformation_matrix=best_transformation_matrix,
                                                  data = freemocap_actor.body.xyz.as_array)
    return aligned_freemocap_data, best_transformation_matrix
    f = 2
# def pld_run_ransac_spatial_alignment(alignment_config: SpatialAlignmentConfig):
#     """
#     Runs the RANSAC spatial alignment process using the provided configuration.

#     Parameters:
#     ----------
#     alignment_config : SpatialAlignmentConfig
#         The configuration for the alignment process.

#     Returns:    
#     -------
#     aligned_freemocap_skeleton_model : Skeleton
#         The aligned FreeMoCap data in a Skeleton model
#     best_transformation_matrix : np.ndarray
#         The best transformation matrix obtained from the RANSAC process.
#     """
#     freemocap_skeleton_model = alignment_config.freemocap_skeleton_function()
#     aligned_freemocap_skeleton_model = alignment_config.freemocap_skeleton_function()
#     qualisys_skeleton_model = alignment_config.qualisys_skeleton_function()

#     validate_marker_presence(
#         freemocap_skeleton_model=freemocap_skeleton_model,
#         qualisys_skeleton_model=qualisys_skeleton_model,
#         markers_for_alignment=alignment_config.markers_for_alignment
#     )

#     freemocap_data = np.load(alignment_config.path_to_freemocap_output_data)
#     freemocap_skeleton_model.integrate_freemocap_3d_data(freemocap_data)

#     qualisys_data = np.load(alignment_config.path_to_qualisys_output_data)
#     qualisys_skeleton_model.integrate_freemocap_3d_data(qualisys_data)

#     freemocap_data_handler = DataProcessor(
#         data=freemocap_skeleton_model.marker_data_as_numpy,
#         marker_list=freemocap_skeleton_model.marker_names,
#         markers_for_alignment=alignment_config.markers_for_alignment
#     )
#     qualisys_data_handler = DataProcessor(
#         data=qualisys_skeleton_model.marker_data_as_numpy,
#         marker_list=qualisys_skeleton_model.marker_names,
#         markers_for_alignment=alignment_config.markers_for_alignment
#     )

#     best_transformation_matrix = get_best_transformation_matrix_ransac(
#         freemocap_data=freemocap_data_handler.extracted_data_3d,
#         qualisys_data=qualisys_data_handler.extracted_data_3d,
#         frames_to_sample=alignment_config.frames_to_sample,
#         max_iterations=alignment_config.max_iterations,
#         inlier_threshold=alignment_config.inlier_threshold
#     )

#     print('Best transformation matrix: ', best_transformation_matrix)

#     aligned_freemocap_data = apply_transformation(
#         best_transformation_matrix, freemocap_skeleton_model.original_marker_data_as_numpy
#     )

#     aligned_freemocap_skeleton_model.integrate_freemocap_3d_data(aligned_freemocap_data)

#     return aligned_freemocap_skeleton_model, best_transformation_matrix