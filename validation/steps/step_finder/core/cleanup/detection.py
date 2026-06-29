from validation.steps.step_finder.core.models import GaitEvents, GaitResults
from validation.steps.step_finder.core.calculate_kinematics import FootKinematics
import numpy as np

def find_suspicious_events(foot_kinematics: FootKinematics, gait_events: GaitResults):
    left_hs_flagged = flag_events_for_removal(positions=foot_kinematics.left_heel_pos, events = gait_events.left_foot.heel_strikes)
    left_to_flagged = flag_events_for_removal(positions=foot_kinematics.left_toe_pos, events = gait_events.left_foot.toe_offs)
    right_hs_flagged = flag_events_for_removal(positions=foot_kinematics.right_heel_pos, events = gait_events.right_foot.heel_strikes)
    right_to_flagged = flag_events_for_removal(positions=foot_kinematics.right_toe_pos, events = gait_events.right_foot.toe_offs)

    return GaitResults(
        right_foot=GaitEvents(
            heel_strikes=right_hs_flagged,
            toe_offs=right_to_flagged,
        ),
        left_foot=GaitEvents(
            heel_strikes=left_hs_flagged,
            toe_offs=left_to_flagged,
        ),
    )


def flag_events_for_removal(positions, events):
    event_positions = positions[events]
    

    suspicious_events_clusters = interval_cluster(
        event_indices=events,
        median_threshold=0.7
    )
    

    ap_positions = event_positions[:,1]
    height_positions = event_positions[:,2]
    
    med_ap_position = np.median(ap_positions)
    mad_ap = np.median(np.abs(ap_positions - med_ap_position)) or 1e-8

    med_height_position = np.median(height_positions)
    mad_height = np.median(np.abs(height_positions - med_height_position)) or 1e-8

    flagged_for_removal = []
    for cluster in suspicious_events_clusters:
        cluster_scores = {}
        for event in cluster:
            # find this event's index in the full event list
            idx = np.where(events == event)[0][0]
            
            # local neighborhood of events
            local_slice = slice(max(0, idx - 5), idx + 6)
            local_ap = positions[events[local_slice], 1]
            local_height = positions[events[local_slice], 2]
            
            local_med_ap = np.median(local_ap)
            local_mad_ap = np.median(np.abs(local_ap - local_med_ap)) or 1e-8
            
            local_med_height = np.median(local_height)
            local_mad_height = np.median(np.abs(local_height - local_med_height)) or 1e-8
            
            z_ap = (positions[event][1] - local_med_ap) / local_mad_ap
            z_height = (positions[event][2] - local_med_height) / local_mad_height
            
            total_z = abs(z_ap) + abs(z_height)
            cluster_scores[event] = total_z
        print(f"Cluster: {cluster}")
        print(f"  Scores: {cluster_scores}")
        if cluster_scores:
            suspicious_event_index = max(cluster_scores, key=cluster_scores.get)
            # print(f"  Removing: {suspicious_event_index} (z={cluster_scores[suspicious_event_index]:.2f})")
            flagged_for_removal.append(suspicious_event_index)

    return np.array(flagged_for_removal, dtype=int)



def interval_cluster(event_indices:np.ndarray,
                     median_threshold: float = 0.6,):
                     
    median_interval = np.median(np.diff(event_indices))

    interval_threshold = median_interval*median_threshold

    intervals = np.diff(event_indices)
    # print(f"Median interval: {median_interval}, Threshold: {interval_threshold}")
    # print(f"Min interval: {intervals.min()}, Max interval: {intervals.max()}")

    clusters: list[np.ndarray] = []
    current_cluster: list[int] = []

    for i, event in enumerate(event_indices[1:], start=1):
        gap = event_indices[i] - event_indices[i - 1]

        if gap < interval_threshold:
            # short gap → these belong together
            if not current_cluster:
                # start cluster with previous event
                current_cluster.append(int(event_indices[i - 1]))
            current_cluster.append(int(event))
        else:
            # normal gap → close any active cluster
            if len(current_cluster) >= 2:
                clusters.append(np.asarray(current_cluster, dtype=int))
            current_cluster = []
    
    if len(current_cluster) >= 2:
        clusters.append(np.asarray(current_cluster, dtype=int))

    return clusters
