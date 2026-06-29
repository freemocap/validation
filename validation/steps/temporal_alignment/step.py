# inside validation/steps/temporal_alignment/step.py

from nicegui import ui, app
import numpy as np
import plotly.graph_objects as go
import pandas as pd 

from validation.pipeline.base import ValidationStep
from validation.steps.temporal_alignment.config import TemporalAlignmentConfig
from validation.steps.temporal_alignment.core.temporal_synchronizer import TemporalSyncManager
from validation.steps.temporal_alignment.core.lag_calculation import LagCalculator, LagMethod
from validation.utils.actor_utils import make_freemocap_actor_from_tracked_points
from validation.steps.temporal_alignment.core.postprocessing import process_and_filter_data

from validation.components import (
    FREEMOCAP_TIMESTAMPS,
    QUALISYS_MARKERS,
    QUALISYS_START_TIME,
    FREEMOCAP_PRE_SYNC_JOINT_CENTERS,
    QUALISYS_SYNCED_JOINT_CENTERS,
    QUALISYS_SYNCED_MARKER_DATA,
    FREEMOCAP_LAG,
    FREEMOCAP_PREALPHA_TIMESTAMPS
)
from validation.steps.temporal_alignment.components import REQUIRES, PRODUCES
import yaml

from pathlib import Path
class TemporalAlignmentStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = TemporalAlignmentConfig

    def calculate(self):
        self.logger.info("Starting temporal alignment (Trajectories + PCA + Kinetic)")

        # --- Load inputs ---
        freemocap_timestamps     = self.data[FREEMOCAP_TIMESTAMPS.name]
        qualisys_dataframe       = self.data[QUALISYS_MARKERS.name]
        qualisys_unix_start_time = self.data[QUALISYS_START_TIME.name]
        freemocap_joint_centers  = self.data[FREEMOCAP_PRE_SYNC_JOINT_CENTERS.name]

        freemocap_actor = make_freemocap_actor_from_tracked_points(
            freemocap_tracker=self.ctx.project_config.freemocap_tracker,
            tracked_points_data=freemocap_joint_centers,
        )

        qualisys_joint_weights_path = Path.cwd() / self.cfg.qualisys_joint_weights_file
        
        with open(qualisys_joint_weights_path, 'r') as f:
            joint_center_weights = yaml.safe_load(f)
    
        manager = TemporalSyncManager(
            freemocap_model=freemocap_actor,
            freemocap_timestamps=freemocap_timestamps,
            qualisys_marker_data=qualisys_dataframe,
            qualisys_unix_start_time=qualisys_unix_start_time,
            joint_center_weights=joint_center_weights,
            start_frame=self.cfg.start_frame,
            end_frame=self.cfg.end_frame,
        )

        # Prep components
        manager._process_freemocap_data()
        manager._process_qualisys_data()
        fmc  = manager.freemocap_lag_component
        qls0 = manager._create_qualisys_component(lag_in_seconds=0.0)

        fr = float(getattr(manager, "framerate", 30.0))
        def frames_to_seconds(frames: int | float) -> float:
            return float(frames) / fr
        
        if getattr(self.cfg, "lag_frames", None) is not None:
            self.logger.info(f"TemporalAlignmentStep: Using cfg.lag_frames={self.cfg.lag_frames}; skipping manual GUI choice.")
            final_frames = float(self.cfg.lag_frames)
            final_seconds = frames_to_seconds(final_frames)

            corrected_component = manager._create_qualisys_component(lag_in_seconds=final_seconds)
            synced_markers_df   = manager._get_synced_qualisys_marker_data(lag_in_seconds=final_seconds)

            self.freemocap_lag_component         = manager.freemocap_lag_component
            self.qualisys_original_lag_component = qls0
            self.qualisys_synced_lag_component   = corrected_component

            qualisys_postprocessed = process_and_filter_data(
                data_3d=corrected_component.joint_center_array,
                landmark_names=corrected_component.list_of_joint_center_names,
                cutoff_frequency=7,
                sampling_rate=30,
                filter_order=4,
            )

            lag_dataframe = pd.DataFrame({"lag_frames": [final_frames]})

            self.outputs[QUALISYS_SYNCED_JOINT_CENTERS.name] = qualisys_postprocessed
            self.outputs[QUALISYS_SYNCED_MARKER_DATA.name]   = synced_markers_df
            self.outputs[FREEMOCAP_LAG.name]                 = lag_dataframe
            return 

        # common joints for Trajectories
        joints = sorted(list(set(fmc.list_of_joint_center_names) & set(qls0.list_of_joint_center_names)))
        if not joints:
            raise RuntimeError("No common joints between FreeMoCap and Qualisys.")

        # Initial suggestions (ENERGY, PC1, KINETIC)
        calc_base = dict(freemocap_component=fmc, qualisys_component=qls0, framerate=fr,
                         start_frame=self.cfg.start_frame, end_frame=self.cfg.end_frame)
        def _safe_est(method):
            try:
                return LagCalculator(**calc_base).estimate(method).lag_frames
            except Exception:
                return 0
        suggested_frames_energy  = _safe_est(LagMethod.ENERGY)
        suggested_frames_pca     = _safe_est(LagMethod.PC1)
        suggested_frames_kinetic = _safe_est(LagMethod.KINETIC)

        # ---------- helpers ----------
        def z(v):
            mu = np.nanmean(v); sd = np.nanstd(v)
            if not np.isfinite(sd) or sd == 0: sd = 1.0
            return (v - mu) / sd

        def make_traj_axis_fig(jname: str, lag_frames: int, dim: int, title_suffix: str, xyrange=None):
            lag_s = frames_to_seconds(lag_frames)
            qls = manager._create_qualisys_component(lag_in_seconds=lag_s)
            fi = fmc.list_of_joint_center_names.index(jname)
            oi = qls0.list_of_joint_center_names.index(jname)
            ci = qls.list_of_joint_center_names.index(jname)
            F = fmc.joint_center_array[:, fi, dim]
            O = qls0.joint_center_array[:, oi, dim]
            C = qls.joint_center_array[:, ci, dim]
            n = min(len(F), len(O), len(C))
            t = np.arange(n) / fr
            fig = go.Figure()
            fig.add_scatter(x=t, y=z(F[:n]), name='FreeMoCap')
            # fig.add_scatter(x=t, y=z(O[:n]), name='Qualisys (orig)')
            fig.add_scatter(x=t, y=z(C[:n]), name='Qualisys (corr)')
            fig.update_layout(
                title=f'{jname} — {title_suffix}  (lag={lag_frames} fr ≈ {lag_s:.3f}s)',
                xaxis_title='Time (s)', yaxis_title='Normalized position',
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=40, r=40, t=60, b=40),
            )
            if xyrange and xyrange.get('x'): fig.update_xaxes(range=xyrange['x'])
            if xyrange and xyrange.get('y'): fig.update_yaxes(range=xyrange['y'])
            return fig
        
        def _xcorr_best_lag_float(a: np.ndarray, b: np.ndarray, *, fps: float, max_lag_s: float = 3.0, sign_align: bool = True) -> float:
            """Return sub-frame lag (float, in frames) that aligns b to a (positive => shift b forward)."""
            a = np.asarray(a, float); b = np.asarray(b, float)
            n = int(min(len(a), len(b)))
            if n < 3:
                return 0.0

            # demean
            a = a[:n] - np.nanmean(a[:n])
            b = b[:n] - np.nanmean(b[:n])

            # optional sign alignment for higher correlation
            if sign_align:
                r = np.corrcoef(np.nan_to_num(a), np.nan_to_num(b))[0, 1]
                if np.isfinite(r) and r < 0:
                    b = -b

            cc = np.correlate(np.nan_to_num(a), np.nan_to_num(b), mode='full')
            lags = np.arange(-n + 1, n)

            # limit search window
            max_lag = max(1, int(round(max_lag_s * fps)))
            mid = len(cc) // 2
            left = max(0, mid - max_lag)
            right = min(len(cc), mid + max_lag + 1)

            cc_win = cc[left:right]
            lags_win = lags[left:right]

            k = int(np.argmax(cc_win))
            k_global = left + k

            # 3-point quadratic interpolation around the peak for sub-frame resolution
            if 0 < k_global < len(cc) - 1:
                y1, y2, y3 = cc[k_global - 1], cc[k_global], cc[k_global + 1]
                denom = (y1 - 2.0 * y2 + y3)
                delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
            else:
                delta = 0.0

            return float(lags[k_global] + delta)
        
        def estimate_lag_from_trajectories_median(*, fps: float, joints: list[str], manager, max_lag_s: float = 3.0) -> float:
            """
            For each joint & axis {X,Y,Z}, compute a fractional-frame xcorr lag between:
            a = FreeMoCap series
            b = Qualisys (orig) series
            then return the median of all valid lags.
            """
            lags = []
            qls0_local = manager._create_qualisys_component(lag_in_seconds=0.0)  # orig (no shift)
            fmc_local  = manager.freemocap_lag_component

            for jname in joints:
                try:
                    fi = fmc_local.list_of_joint_center_names.index(jname)
                    oi = qls0_local.list_of_joint_center_names.index(jname)
                except ValueError:
                    continue

                # 3 axes
                for dim in (1, 2):
                    F = fmc_local.joint_center_array[:, fi, dim]
                    O = qls0_local.joint_center_array[:, oi, dim]

                    # z-score each 1D series for stability (matches how your plots normalize)
                    a = (F - np.nanmean(F)) / (np.nanstd(F) or 1.0)
                    b = (O - np.nanmean(O)) / (np.nanstd(O) or 1.0)

                    lag_frames = _xcorr_best_lag_float(a, b, fps=fps, max_lag_s=max_lag_s, sign_align=True)
                    if np.isfinite(lag_frames):
                        lags.append(lag_frames)

            if not lags:
                return 0.0
            return float(np.nanmedian(np.array(lags, dtype=float)))
        # --- PCA utilities (k = 1/2/3) ---
        def pc_series(tj3: np.ndarray, k: int) -> np.ndarray:
            T, J, C = tj3.shape
            X = tj3.reshape(T, J*C)
            mu = np.nanmean(X, axis=0, keepdims=True)
            sd = np.nanstd(X, axis=0, keepdims=True); sd[sd == 0] = 1.0
            Z = (X - mu) / sd
            U, S, _ = np.linalg.svd(np.nan_to_num(Z), full_matrices=False)
            if S.size < k: return np.zeros(Z.shape[0], dtype=float)
            s = U[:, k-1] * S[k-1]
            return z(s)

        def xcorr_best_lag(a: np.ndarray, b: np.ndarray, max_lag_s: float = 3.0, sign_align: bool = True) -> int:
            a = np.asarray(a, float); b = np.asarray(b, float)
            n = int(min(len(a), len(b)))
            if n < 3: return 0
            a = a[:n] - np.nanmean(a[:n]); b = b[:n] - np.nanmean(b[:n])
            if sign_align:
                r = np.corrcoef(np.nan_to_num(a), np.nan_to_num(b))[0,1]
                if np.isfinite(r) and r < 0: b = -b
            cc = np.correlate(np.nan_to_num(a), np.nan_to_num(b), mode='full')
            lags = np.arange(-n+1, n)
            max_lag = max(1, int(round(max_lag_s * fr)))
            mid = len(cc)//2
            sl = slice(max(0, mid-max_lag), min(len(cc), mid+max_lag+1))
            return int(lags[sl][np.argmax(cc[sl])])

        # --- Kinetic energy series for plotting (nonnegative; no flipping needed) ---
        def kinetic_series(tj3: np.ndarray) -> np.ndarray:
            dt = 1.0 / fr
            d = np.diff(tj3, axis=0) / dt              # (T-1, J, 3) velocities
            speed2 = np.sum(d*d, axis=2)               # (T-1, J)
            ke = 0.5 * np.sum(speed2, axis=1)          # all masses = 1 by default
            return z(ke)

        def make_pca_fig(lag_frames: int, pc_k: int, flip_mult: int, auto_flip: bool, xyrange=None):
            lag_s = frames_to_seconds(lag_frames)
            qls_corr = manager._create_qualisys_component(lag_in_seconds=lag_s)
            fmc_pc  = pc_series(fmc.joint_center_array,  pc_k)
            qls_pc0 = pc_series(qls0.joint_center_array, pc_k)
            qls_pcC = pc_series(qls_corr.joint_center_array, pc_k)
            if auto_flip:
                r = np.corrcoef(fmc_pc[:min(len(fmc_pc), len(qls_pc0))],
                                qls_pc0[:min(len(fmc_pc), len(qls_pc0))])[0,1]
                if np.isfinite(r) and r < 0:
                    qls_pc0 = -qls_pc0; qls_pcC = -qls_pcC
            if flip_mult == -1:
                qls_pc0 = -qls_pc0; qls_pcC = -qls_pcC
            n = min(len(fmc_pc), len(qls_pc0), len(qls_pcC))
            t = np.arange(n) / fr
            fig = go.Figure()
            fig.add_scatter(x=t, y=fmc_pc[:n],  name=f'FreeMoCap PC{pc_k}')
            fig.add_scatter(x=t, y=qls_pc0[:n], name=f'Qualisys PC{pc_k} (orig)')
            fig.add_scatter(x=t, y=qls_pcC[:n], name=f'Qualisys PC{pc_k} (corr)')
            fig.update_layout(
                title=f'PCA PC{pc_k} — lag={lag_frames} fr ≈ {lag_s:.3f}s',
                xaxis_title='Time (s)', yaxis_title=f'PC{pc_k} (z-scored)',
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=40, r=40, t=60, b=40),
            )
            if xyrange and xyrange.get('x'): fig.update_xaxes(range=xyrange['x'])
            if xyrange and xyrange.get('y'): fig.update_yaxes(range=xyrange['y'])
            return fig

        def make_ke_fig(lag_frames: int, xyrange=None):
            lag_s = frames_to_seconds(lag_frames)
            qls_corr = manager._create_qualisys_component(lag_in_seconds=lag_s)
            f_ke  = kinetic_series(fmc.joint_center_array)
            q0_ke = kinetic_series(qls0.joint_center_array)
            qc_ke = kinetic_series(qls_corr.joint_center_array)
            n = min(len(f_ke), len(q0_ke), len(qc_ke))
            t = np.arange(n) / fr
            fig = go.Figure()
            fig.add_scatter(x=t, y=f_ke[:n],  name='FreeMoCap KE')
            fig.add_scatter(x=t, y=q0_ke[:n], name='Qualisys KE (orig)')
            fig.add_scatter(x=t, y=qc_ke[:n], name='Qualisys KE (corr)')
            fig.update_layout(
                title=f'Kinetic Energy — lag={lag_frames} fr ≈ {lag_s:.3f}s',
                xaxis_title='Time (s)', yaxis_title='KE (z-scored)',
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=40, r=40, t=60, b=40),
            )
            if xyrange and xyrange.get('x'): fig.update_xaxes(range=xyrange['x'])
            if xyrange and xyrange.get('y'): fig.update_yaxes(range=xyrange['y'])
            return fig

        # ---------- UI ----------
        ui.markdown('# Temporal Alignment')

        with ui.card().classes('w-full max-w-screen-xl mx-auto'):
            # add 'kin' tab
            method = ui.toggle({'traj': 'TRAJECTORIES', 'pca': 'PCA', 'kin': 'KINETIC'}, value='traj')

            with ui.row().classes('items-end gap-4'):
                joint_select = ui.select(joints, value=joints[0], label='Joint')

                lag_input = ui.number(
                    label='Lag (frames)', value=float(suggested_frames_energy),
                    step=.1, min=-10000, max=10000, format='%.2f',
                ).classes('w-40')
                sec_label = ui.label(f'~ {frames_to_seconds(float(lag_input.value)):.3f} s').classes('text-gray-600')

                # PCA-only controls
                pc_select = ui.select([1, 2, 3], value=1, label='PC').classes('w-24')
                auto_flip = ui.switch('Auto-flip sign', value=True)
                flip_mult = {'val': 1}
                flip_btn = ui.button('Flip', on_click=lambda: flip_mult.update(val=-flip_mult['val']))

                # shared
                calc_btn   = ui.button('CALCULATE', color='secondary')
                auto_btn   = ui.button('AUTO')
                accept_btn = ui.button('ACCEPT', color='primary')
                cancel_btn = ui.button('CANCEL', color='negative')

            # zoom states
            axes_traj = {'x': {'x': None, 'y': None}, 'y': {'x': None, 'y': None}, 'z': {'x': None, 'y': None}}
            axes_pca  = {'x': None, 'y': None}
            axes_ke   = {'x': None, 'y': None}

            with ui.column().classes('w-full'):
                # trajectories
                plot_tx = ui.plotly(make_traj_axis_fig(joint_select.value, float(lag_input.value), 0, 'X', axes_traj['x']))\
                           .classes('w-full h-[300px]')
                plot_ty = ui.plotly(make_traj_axis_fig(joint_select.value, float(lag_input.value), 1, 'Y', axes_traj['y']))\
                           .classes('w-full h-[300px]')
                plot_tz = ui.plotly(make_traj_axis_fig(joint_select.value, float(lag_input.value), 2, 'Z', axes_traj['z']))\
                           .classes('w-full h-[300px]')
            def _relayout_traj(which: str):
                def _cb(e):
                    a = e.args or {}
                    xr = [a.get('xaxis.range[0]'), a.get('xaxis.range[1]')]
                    yr = [a.get('yaxis.range[0]'), a.get('yaxis.range[1]')]
                    if None not in xr: axes_traj[which]['x'] = xr
                    if None not in yr: axes_traj[which]['y'] = yr
                return _cb
            plot_tx.on('plotly_relayout', _relayout_traj('x'))
            plot_ty.on('plotly_relayout', _relayout_traj('y'))
            plot_tz.on('plotly_relayout', _relayout_traj('z'))

        # PCA / KE plots on demand
        pca_plot = {'obj': None}
        ke_plot  = {'obj': None}

        def show_traj():
            joint_select.enable()
            plot_tx.style('display:block'); plot_ty.style('display:block'); plot_tz.style('display:block')
            if pca_plot['obj'] is not None: pca_plot['obj'].style('display:none')
            if ke_plot['obj']  is not None: ke_plot['obj'].style('display:none')

        def show_pca():
            joint_select.disable()
            plot_tx.style('display:none'); plot_ty.style('display:none'); plot_tz.style('display:none')
            if pca_plot['obj'] is None:
                pca_plot['obj'] = ui.plotly(make_pca_fig(float(lag_input.value), int(pc_select.value),
                                                         flip_mult['val'], bool(auto_flip.value), axes_pca))\
                                    .classes('w-full h-[420px]')
                pca_plot['obj'].on('plotly_relayout', lambda e: (
                    axes_pca.update({
                        'x': [e.args.get('xaxis.range[0]'), e.args.get('xaxis.range[1]')],
                        'y': [e.args.get('yaxis.range[0]'), e.args.get('yaxis.range[1]')]
                    })
                ))
            else:
                pca_plot['obj'].style('display:block')
                pca_plot['obj'].update_figure(make_pca_fig(float(lag_input.value), int(pc_select.value),
                                                           flip_mult['val'], bool(auto_flip.value), axes_pca))
            if ke_plot['obj'] is not None: ke_plot['obj'].style('display:none')

        def show_ke():
            joint_select.disable()
            plot_tx.style('display:none'); plot_ty.style('display:none'); plot_tz.style('display:none')
            if pca_plot['obj'] is not None: pca_plot['obj'].style('display:none')
            if ke_plot['obj'] is None:
                ke_plot['obj'] = ui.plotly(make_ke_fig(float(lag_input.value), axes_ke)).classes('w-full h-[420px]')
                ke_plot['obj'].on('plotly_relayout', lambda e: (
                    axes_ke.update({
                        'x': [e.args.get('xaxis.range[0]'), e.args.get('xaxis.range[1]')],
                        'y': [e.args.get('yaxis.range[0]'), e.args.get('yaxis.range[1]')]
                    })
                ))
            else:
                ke_plot['obj'].style('display:block')
                ke_plot['obj'].update_figure(make_ke_fig(float(lag_input.value), axes_ke))

        def rebuild_panel():
            try:
                f = float(lag_input.value)
            except Exception:
                f = 0; lag_input.set_value(0.0)
            sec_label.text = f'~ {frames_to_seconds(float(f)):.3f} s'

            if method.value == 'traj':
                show_traj()
                j = joint_select.value
                plot_tx.update_figure(make_traj_axis_fig(j, f, 0, 'X', axes_traj['x']))
                plot_ty.update_figure(make_traj_axis_fig(j, f, 1, 'Y', axes_traj['y']))
                plot_tz.update_figure(make_traj_axis_fig(j, f, 2, 'Z', axes_traj['z']))
            elif method.value == 'pca':
                show_pca()
            else:  # 'kin'
                show_ke()

        # initial
        rebuild_panel()

        # interactions
        calc_btn.on('click', lambda: rebuild_panel())
        lag_input.on('keydown.enter', lambda e: rebuild_panel())
        joint_select.on('update:model-value', lambda e: (method.set_value('traj'), rebuild_panel()))
        method.on('update:model-value', lambda e: rebuild_panel())
        pc_select.on('update:model-value', lambda e: (method.set_value('pca'), rebuild_panel()))
        auto_flip.on('update:model-value', lambda e: (method.set_value('pca'), rebuild_panel()))
        flip_btn.on('click', lambda: (method.set_value('pca'), rebuild_panel()))
        
        def _auto_traj():
            f_med = estimate_lag_from_trajectories_median(fps=fr, joints=joints, manager=manager, max_lag_s=3.0)
            lag_input.set_value(float(f_med))
            rebuild_panel()
            return float(f_med)
        
        def on_auto():
            if method.value == 'traj':
                f_auto = _auto_traj()
                print(f'Auto TRAJ lag: {f_auto:.2f} frames')
            elif method.value == 'pca':
                # respect selected PC and flip settings
                k = int(pc_select.value)
                a = pc_series(fmc.joint_center_array,  k)
                b = pc_series(qls0.joint_center_array, k)
                # manual flip first; then optional auto sign in xcorr
                if flip_mult['val'] == -1: b = -b
                f_auto = xcorr_best_lag(a, b, max_lag_s=3.0, sign_align=bool(auto_flip.value))
            else:
                f_auto = _safe_est(LagMethod.KINETIC)
            lag_input.set_value(float(f_auto))
            rebuild_panel()
        auto_btn.on('click', on_auto)

        # accept / cancel
        accepted = {'ok': False}
        def do_accept():
            accepted['ok'] = True
            app.shutdown()
        def do_cancel():
            accepted['ok'] = False
            app.shutdown()
        accept_btn.on('click', do_accept)
        cancel_btn.on('click', do_cancel)

        # block UI
        ui.run(reload=False)
        if not accepted['ok']:
            raise RuntimeError('Temporal alignment cancelled by user.')

        # finalize
        final_frames = float(lag_input.value)
        final_seconds = frames_to_seconds(final_frames)

        corrected_component = manager._create_qualisys_component(lag_in_seconds=final_seconds)
        synced_markers_df   = manager._get_synced_qualisys_marker_data(lag_in_seconds=final_seconds)

        self.freemocap_lag_component         = manager.freemocap_lag_component
        self.qualisys_original_lag_component = qls0
        self.qualisys_synced_lag_component   = corrected_component

        qualisys_postprocessed = process_and_filter_data(data_3d = corrected_component.joint_center_array,
                                                         landmark_names=corrected_component.list_of_joint_center_names,
                                                         cutoff_frequency=7,
                                                         sampling_rate=30,
                                                         filter_order=4)

        lag_dataframe = pd.DataFrame({  
            'lag_frames': [final_frames],})
        self.outputs[QUALISYS_SYNCED_JOINT_CENTERS.name] = qualisys_postprocessed
        self.outputs[QUALISYS_SYNCED_MARKER_DATA.name]   = synced_markers_df
        self.outputs[FREEMOCAP_LAG.name]                 = lag_dataframe
