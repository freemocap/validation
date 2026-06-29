# validation/steps/temporal_alignment/ui_simple.py
from __future__ import annotations
from typing import Dict, Callable
import numpy as np
import plotly.graph_objects as go
from nicegui import ui

from validation.steps.temporal_alignment.core.temporal_synchronizer import TemporalSyncManager
from validation.steps.temporal_alignment.core.lag_calculation import LagCalculatorComponent

def _common_joint_names(a: LagCalculatorComponent, b: LagCalculatorComponent) -> list[str]:
    return sorted(list(set(a.list_of_joint_center_names) & set(b.list_of_joint_center_names)))

def _get_joint_arrays(
    fmc: LagCalculatorComponent,
    qls_orig: LagCalculatorComponent,
    qls_corr: LagCalculatorComponent,
    joint: str,
):
    fi = fmc.list_of_joint_center_names.index(joint)
    oi = qls_orig.list_of_joint_center_names.index(joint)
    ci = qls_corr.list_of_joint_center_names.index(joint)

    f = fmc.joint_center_array[:, fi, :]     # (T, 3)
    o = qls_orig.joint_center_array[:, oi, :]
    c = qls_corr.joint_center_array[:, ci, :]
    n = min(len(f), len(o), len(c))
    return f[:n], o[:n], c[:n], np.arange(n)

def _normalize(arr: np.ndarray) -> np.ndarray:
    # normalize each column to z-score (per-dimension)
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr,  axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (arr - mu) / sd

def _make_xyz_figure(f: np.ndarray, o: np.ndarray, c: np.ndarray, t: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    labels = ['X','Y','Z']
    colors = {'f':'blue','o':'red','c':'green'}
    for dim in range(3):
        fig.add_scatter(x=t, y=_normalize(f)[:, dim], name=f'FreeMoCap ({labels[dim]})', line=dict(color=colors['f']))
        fig.add_scatter(x=t, y=_normalize(o)[:, dim], name=f'Original Qualisys ({labels[dim]})', line=dict(color=colors['o']))
        fig.add_scatter(x=t, y=_normalize(c)[:, dim], name=f'Corrected Qualisys ({labels[dim]})', line=dict(color=colors['c']))
    fig.update_layout(
        title=title,
        xaxis_title='Frame',
        yaxis_title='Normalized position',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def run_simple_temporal_gui(manager: TemporalSyncManager) -> float:
    """
    Launches a minimal GUI for trajectory alignment.
    Returns the chosen lag (seconds) after the user clicks Accept.
    """
    # Prep base components
    manager._process_freemocap_data()
    manager._process_qualisys_data()
    fmc = manager.freemocap_lag_component
    qls_orig = manager._create_qualisys_component(lag_in_seconds=0.0)

    chosen = {'lag': 0.0, 'accepted': False}

    # Initial suggestion via your existing lag calculator (cross-corr/PCA internals)
    suggested_lag = manager._calculate_lag(qls_orig)  # seconds
    chosen['lag'] = float(suggested_lag)

    # First corrected preview
    qls_corr = manager._create_qualisys_component(lag_in_seconds=chosen['lag'])
    joints = _common_joint_names(fmc, qls_orig)
    current_joint = {'name': joints[0] if joints else None}

    # UI
    ui.markdown('# Temporal Alignment (Simple)')
    with ui.card().classes('w-full max-w-screen-xl mx-auto'):
        with ui.row().classes('items-end gap-4'):
            joint_select = ui.select(joints, value=current_joint['name'], label='Joint')
            step = 1.0 / getattr(manager, 'framerate', 100.0)
            slider = ui.slider(min=-2.0, max=2.0, step=step, value=chosen['lag']).props('label-always')
            ui.label().bind_text_from(slider, 'value', lambda v: f'Lag: {v:.3f} s')
            auto_btn = ui.button('Auto (cross-corr)')
            accept_btn = ui.button('Accept', color='primary')
            cancel_btn = ui.button('Cancel', color='negative')

        # One plot that stacks X/Y/Z; simple and fast
        plot = ui.plotly({}).classes('w-full h-[620px]')

    def refresh():
        nonlocal qls_corr
        qls_corr = manager._create_qualisys_component(lag_in_seconds=float(slider.value))
        f, o, c, t = _get_joint_arrays(fmc, qls_orig, qls_corr, current_joint['name'])
        fig = _make_xyz_figure(f, o, c, t, title=f'{current_joint["name"]} — lag={float(slider.value):.3f}s')
        plot.update(fig)

    # Interactions
    joint_select.on('update:model-value', lambda e: (current_joint.update({'name': e.value}), refresh()))
    slider.on('update:model-value', lambda e: refresh())

    def on_auto():
        # recompute lag suggestion using current original component
        lag = manager._calculate_lag(qls_orig)
        slider.set_value(float(lag))  # triggers refresh
    auto_btn.on('click', on_auto)

    def on_accept():
        chosen['lag'] = float(slider.value)
        chosen['accepted'] = True
        ui.close()
    def on_cancel():
        chosen['accepted'] = False
        ui.close()

    accept_btn.on('click', on_accept)
    cancel_btn.on('click', on_cancel)

    # First draw
    refresh()
    ui.run(reload=False)

    if not chosen['accepted']:
        raise RuntimeError('Temporal alignment cancelled.')
    return float(chosen['lag'])
