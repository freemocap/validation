from skellymodels.managers.human import Human
from validation.steps.temporal_alignment.core.lag_calculation import LagCalculatorComponent,LagCalculator
from validation.steps.temporal_alignment.core.qualisys_processing import QualisysMarkerData, QualisysJointCenterData, DataResampler

import pandas as pd
import numpy as np
from typing import Optional

class TemporalSyncManager:
    def __init__(self, freemocap_model: Human,
                 freemocap_timestamps: pd.DataFrame,
                 qualisys_marker_data: pd.DataFrame,
                 qualisys_unix_start_time:float,
                 joint_center_weights: dict,
                 start_frame: Optional[int] = None,
                 end_frame: Optional[int] = None):
        
        self.freemocap_model = freemocap_model
        self.freemocap_timestamps, self.framerate = self._get_timestamps(freemocap_timestamps)
        # self.freemocap_timestamps, self.framerate = self._get_prealpha_timestamps(freemocap_timestamps)
        self.qualisys_marker_data = qualisys_marker_data
        self.qualisys_unix_start_time =qualisys_unix_start_time
        self.joint_center_weights = joint_center_weights
        self.start_frame = start_frame or 0
        self.end_frame =   end_frame or len(self.freemocap_timestamps)

    def run(self):
        self._process_freemocap_data()
        self._process_qualisys_data()

        qualisys_component = self._create_qualisys_component(lag_in_seconds=0)
        initial_lag = self._calculate_lag(qualisys_component)

        corrected_qualisys_component = self._create_qualisys_component(lag_in_seconds=initial_lag)
        final_lag = self._calculate_lag(corrected_qualisys_component)
        synced_qualisys_markers = self._get_synced_qualisys_marker_data(lag_in_seconds=final_lag)
        print('Initial lag:', initial_lag)
        print('Final lag:', final_lag)

        assert qualisys_component.joint_center_array.shape[0] == self.freemocap_lag_component.joint_center_array.shape[0], f"Resampled qualisys data has {qualisys_component.joint_center_array.shape[0]} frames, but freemocap data has {self.freemocap_lag_component.joint_center_array.shape[0]} frames."

        return self.freemocap_lag_component, corrected_qualisys_component, qualisys_component, synced_qualisys_markers

    def _process_freemocap_data(self):
        freemocap_data = self.freemocap_model.body.xyz.as_array
        landmark_names = self.freemocap_model.body.xyz.landmark_names
        # origin_aligned_freemocap_data = run_skellyforge_rotation(raw_skeleton_data=freemocap_data,
        #                                                          landmark_names=landmark_names)
        self.freemocap_lag_component = LagCalculatorComponent(
            joint_center_array=freemocap_data,
            list_of_joint_center_names=landmark_names
        )

    def _process_qualisys_data(self):
        self.qualisys_marker_data_holder = QualisysMarkerData(
            marker_dataframe=self.qualisys_marker_data,
            unix_start_time=self.qualisys_unix_start_time
        )

        self.qualisys_joint_center_data_holder = QualisysJointCenterData(
            marker_data_holder=self.qualisys_marker_data_holder,
            weights=self.joint_center_weights
        )


    def _calculate_lag(self, qualisys_lag_component: LagCalculatorComponent):
        lag_corrector = LagCalculator(
            freemocap_component=self.freemocap_lag_component, 
            qualisys_component=qualisys_lag_component, 
            framerate=self.framerate,
            start_frame=self.start_frame,
            end_frame=self.end_frame)
        
        lag_corrector.run()
        print('Median lag:', lag_corrector.median_lag)
        print('Lag in seconds:', lag_corrector.get_lag_in_seconds())
        return lag_corrector.get_lag_in_seconds()
    f = 2
        
        
    def _extract_marker_data(self) -> pd.DataFrame:
        """Extract only marker data columns."""

        columns_of_interest = self.qualisys_marker_data.columns[
            ~self.qualisys_marker_data.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)
        ]
        return self.qualisys_marker_data[columns_of_interest]


    def _get_timestamps(self, freemocap_timestamps):
        timestamps = freemocap_timestamps['timestamp.utc.seconds']
        time_diff = np.diff(timestamps)
        framerate = 1 / np.nanmean(time_diff)
        print(f"Calculated FreeMoCap framerate: {framerate}")
        return timestamps, framerate

    def _get_prealpha_timestamps(self, freemocap_timestamps:pd.DataFrame):
        freemocap_timestamps.replace(-1, float('nan'), inplace=True)
        mean_timestamps = freemocap_timestamps.iloc[:, 2:].mean(axis=1, skipna=True)
        
        mean_timestamps.interpolate(method='linear', inplace=True) #interpolating because there are some frames where all the cameras are missing timestamps, leading to nans in the final list

        time_diff = np.diff(mean_timestamps)
        framerate = 1 / np.nanmean(time_diff)
        print(f"Calculated FreeMoCap framerate: {framerate}")
        return mean_timestamps, framerate


    def _create_qualisys_component(self, lag_in_seconds:float = 0) -> LagCalculatorComponent: 
        joint_center_names = list(self.joint_center_weights.keys())
        df = self.qualisys_joint_center_data_holder.as_dataframe_with_unix_timestamps(lag_seconds=lag_in_seconds)
        resampler = DataResampler(df, self.freemocap_timestamps)
        resampler.resample()
        self.resampled_qualisys_joint_center_data = resampler.as_dataframe
        return LagCalculatorComponent(
            joint_center_array=resampler.resampled_marker_array,
            list_of_joint_center_names=joint_center_names
        )

    def _get_synced_qualisys_marker_data(self, lag_in_seconds: float = 0) -> pd.DataFrame:
        """
        Returns a DataFrame with resampled Qualisys marker data aligned to FreeMoCap timestamps.
        
        Parameters:
            lag_in_seconds (float): Optional time offset to adjust timestamps.
        
        Returns:
            pd.DataFrame: DataFrame containing resampled Qualisys marker data.
        """
        df = self.qualisys_marker_data_holder.as_dataframe_with_unix_timestamps(lag_seconds=lag_in_seconds)
        resampler = DataResampler(df, self.freemocap_timestamps)
        resampler.resample()
        return resampler.as_dataframe

