from nicegui import ui
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

class SynchronizationVisualizer:
    def __init__(self, 
                 freemocap_component,
                 original_qualisys_component,
                 corrected_qualisys_component,):
        """
        Initialize the visualizer with motion capture components.
        
        Args:
            freemocap_component: LagCalculatorComponent containing FreeMoCap data
            original_qualisys_component: LagCalculatorComponent containing original Qualisys data
            corrected_qualisys_component: LagCalculatorComponent containing corrected Qualisys data
            framerate: The frame rate of the recordings
        """
        self.freemocap_component = freemocap_component
        self.original_qualisys_component = original_qualisys_component
        self.corrected_qualisys_component = corrected_qualisys_component
        
        # Get common joint names across all data sets
        self.common_joint_names = self._get_common_joint_names()
        
        # UI components
        self.joint_selector = None
        self.dimension_selector = None
        self.view_type_selector = None
        self.plot = None
        self.window_size_slider = None
        
        # Current state
        self.current_joint = None
        self.current_dimension = 0  # X dimension by default
        self.current_view = 'both'  # both, before, after
        self.window_size = 300      # default window size in frames
    
    def _get_common_joint_names(self) -> List[str]:
        """Get joint names common to all three components."""
        freemocap_joints = set(self.freemocap_component.list_of_joint_center_names)
        original_qualisys_joints = set(self.original_qualisys_component.list_of_joint_center_names)
        corrected_qualisys_joints = set(self.corrected_qualisys_component.list_of_joint_center_names)
        
        return list(freemocap_joints & original_qualisys_joints & corrected_qualisys_joints)
    
    def _get_joint_data(self, joint_name: str) -> Dict[str, np.ndarray]:
        """Get data for a specific joint from all components."""
        # Get indices for the joint in each component
        freemocap_idx = self.freemocap_component.list_of_joint_center_names.index(joint_name)
        original_qualisys_idx = self.original_qualisys_component.list_of_joint_center_names.index(joint_name)
        corrected_qualisys_idx = self.corrected_qualisys_component.list_of_joint_center_names.index(joint_name)
        
        # Extract the joint data
        freemocap_data = self.freemocap_component.joint_center_array[:, freemocap_idx, :]
        original_qualisys_data = self.original_qualisys_component.joint_center_array[:, original_qualisys_idx, :]
        corrected_qualisys_data = self.corrected_qualisys_component.joint_center_array[:, corrected_qualisys_idx, :]
        
        return {
            'freemocap': freemocap_data,
            'original_qualisys': original_qualisys_data,
            'corrected_qualisys': corrected_qualisys_data
        }
    
    def _normalize_signals(self, signals: Dict[str, np.ndarray], dimension: int) -> Dict[str, np.ndarray]:
        """Normalize signals for better visualization."""
        normalized = {}
        for key, signal in signals.items():
            # Extract the specified dimension
            dim_signal = signal[:, dimension]
            # Normalize to zero mean and unit variance
            normalized[key] = (dim_signal - np.mean(dim_signal)) / np.std(dim_signal)
        
        return normalized
    
    def _create_time_axis(self, length: int) -> np.ndarray:
        """Create a time axis in seconds based on frame count and framerate."""
        return np.arange(length)
    
    def update_plot(self):
        """Update all three dimension plots based on current selection."""
        if not self.current_joint:
            return
        
        # Get data for selected joint
        joint_data = self._get_joint_data(self.current_joint)
        
        # Get time axis
        min_length = min(
            joint_data['freemocap'].shape[0],
            joint_data['original_qualisys'].shape[0],
            joint_data['corrected_qualisys'].shape[0]
        )
        
        # Create full time axis before slicing
        full_time_axis = self._create_time_axis(min_length)
        
        # Instead of centering, start from the beginning to better show lag
        start_idx = 0
        end_idx = min(min_length, self.window_size)
        
        # If we have a lot of data, find an interesting segment
        if min_length > 500:
            # Look for high movement periods by computing velocity across all dimensions
            signal = joint_data['freemocap']
            velocity = np.sum(np.abs(np.diff(signal, axis=0)), axis=1)
            avg_velocity = np.convolve(velocity, np.ones(30)/30, mode='same')
            
            # Find a window with significant motion (skip first 100 frames)
            high_motion_indices = np.where(avg_velocity > np.percentile(avg_velocity, 75))[0]
            if len(high_motion_indices) > 0 and high_motion_indices[0] > 100:
                # Start at a point with high motion
                start_idx = max(0, high_motion_indices[0] - 50)
                end_idx = min(min_length, start_idx + self.window_size)
        
        time_axis = full_time_axis[start_idx:end_idx]
        
        # Update all three dimension plots
        dimension_labels = ['X', 'Y', 'Z']
        
        for dim in range(3):
            # Normalize signals for this dimension
            normalized_data = self._normalize_signals(joint_data, dim)
            
            # Create new plot
            fig = go.Figure()
            
            # Add traces based on view type
            if self.current_view in ['before', 'both']:
                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=normalized_data['freemocap'][start_idx:end_idx],
                    mode='lines',
                    name='FreeMoCap',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=normalized_data['original_qualisys'][start_idx:end_idx],
                    mode='lines',
                    name='Original Qualisys',
                    line=dict(color='red', width=2)
                ))
            
            if self.current_view in ['after', 'both']:
                if self.current_view == 'after':
                    fig.add_trace(go.Scatter(
                        x=time_axis,
                        y=normalized_data['freemocap'][start_idx:end_idx],
                        mode='lines',
                        name='FreeMoCap',
                        line=dict(color='blue', width=2)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=normalized_data['corrected_qualisys'][start_idx:end_idx],
                    mode='lines',
                    name='Corrected Qualisys',
                    line=dict(color='green', width=2)
                ))
            
            # Update layout with better styling
            fig.update_layout(
                title=f'{self.current_joint} - {dimension_labels[dim]} Dimension',
                xaxis_title='Time (seconds)' if dim == 2 else None,  # Only show on bottom plot
                yaxis_title='Normalized Position',
                showlegend=True if dim == 0 else False,  # Only show legend on top plot
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template='plotly_white',
                margin=dict(l=50, r=50, t=50, b=50 if dim == 2 else 20)
            )
            
            # Update the corresponding plot in the UI
            self.plots[dim].update_figure(fig)
    
    def on_joint_change(self, event):
        """Handle joint selection change."""
        self.current_joint = event.value
        self.update_plot()
    
    def on_view_change(self, event):
        """Handle view type change."""
        # Map UI selection to internal view types
        view_map = {
            'Both': 'both',
            'Before': 'before',
            'After': 'after'
        }
        self.current_view = view_map[event.value]
        self.update_plot()
    
    def on_window_change(self, event):
        """Handle window size change."""
        self.window_size = int(event.value)
        self.update_plot()
    
    def create_ui(self):
        """Create the UI components."""
        ui.markdown('# Temporal Synchronization Visualization')
        
        # Controls row - across the full width
        with ui.card().classes('w-full mb-4'):
            with ui.row().classes('items-end gap-4 flex-wrap'):
                # Joint selector
                with ui.column().classes('flex-grow'):
                    self.joint_selector = ui.select(
                        label='Joint',
                        options=self.common_joint_names,
                        on_change=self.on_joint_change
                    ).classes('w-full')
                
                # View type selector
                with ui.column().classes('flex-grow'):
                    view_options = ['Both', 'Before', 'After']
                    self.view_type_selector = ui.select(
                        label='View',
                        options=view_options,
                        value='Both',
                        on_change=self.on_view_change
                    ).classes('w-full')
                
                # Window size slider
                with ui.column().classes('flex-grow-2'):
                    ui.label('Window Size (frames):')
                    self.window_size_slider = ui.slider(
                        min=100, 
                        max=1000,
                        step=50,
                        value=self.window_size,
                        on_change=self.on_window_change
                    ).classes('w-full')
                    
                    # Display window size in seconds
                    ui.label().bind_text_from(
                        self.window_size_slider, 'value', 
                        lambda v: f'{int(v)} frames ({v} frames)'
                    )
        
        # Plot area - full width card with three plots (one for each dimension)
        with ui.card().classes('w-full'):
            # Create three plot containers for X, Y, Z dimensions
            self.plots = {
                0: ui.plotly({}).classes('w-full h-[250px]'),
                1: ui.plotly({}).classes('w-full h-[250px]'),
                2: ui.plotly({}).classes('w-full h-[250px]')
            }
            
            # Add some information text
            with ui.expansion('Visualization Help', icon='info').classes('w-full mt-4'):
                ui.markdown('''
                ## How to interpret this visualization
                
                - **Blue line**: FreeMoCap data trajectory
                - **Red line**: Original Qualisys data (before synchronization)
                - **Green line**: Corrected Qualisys data (after synchronization)
                
                If synchronization worked well, the green line (corrected) should align closely with the blue line (FreeMoCap),
                while the red line (original) will likely show a time offset.
                
                - Use the **Joint** selector to switch between different markers
                - Use the **View** selector to compare before/after synchronization
                - Adjust the **Window Size** to zoom in/out on the time axis
                
                The three plots show the X, Y, and Z dimensions simultaneously.
                ''')
                
        
        # Initialize with first joint
        if self.common_joint_names:
            self.current_joint = self.common_joint_names[0]
            self.joint_selector.set_value(self.current_joint)
            self.update_plot()