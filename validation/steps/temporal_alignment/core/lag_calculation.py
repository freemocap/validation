# validation/steps/temporal_alignment/core/lag_calculation.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np


# -----------------------------------------------------------------------------
# Minimal component
# -----------------------------------------------------------------------------
@dataclass
class LagCalculatorComponent:
    joint_center_array: np.ndarray          # (T, J, 3)
    list_of_joint_center_names: list[str]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
class LagMethod(str, Enum):
    ENERGY  = "energy"   # sum of squared frame-to-frame displacements (≈ motion energy)
    KINETIC = "kinetic"  # 0.5 * sum_j m_j * ||v_j||^2   (v in m/s; m_j optional)
    PC1     = "pc1"      # first principal component score time series


@dataclass
class LagResult:
    lag_frames: float
    lag_seconds: float
    peak_value: Optional[float]
    series_a: np.ndarray
    series_b: np.ndarray


class LagCalculator:
    """
    Unifies lag estimation across methods. Returns lag in FRAMES (primary).
    """

    def __init__(
        self,
        freemocap_component: LagCalculatorComponent,
        qualisys_component: LagCalculatorComponent,
        framerate: float,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        joint_masses: Optional[np.ndarray] = None,  # shape (J,), optional
    ) -> None:
        self.f = freemocap_component
        self.q = qualisys_component
        self.fps = float(framerate)
        self.start = 0 if start_frame is None else int(start_frame)
        self.end = None if end_frame is None else int(end_frame)
        self.masses = None
        if joint_masses is not None:
            m = np.asarray(joint_masses).reshape(-1)
            # only accept if matches J
            if m.size == self.f.joint_center_array.shape[1]:
                self.masses = m

        self._last_frames: int = 0

    # --------------------------- public entry point ---------------------------
    def estimate(
        self,
        method: LagMethod = LagMethod.ENERGY,
        *,
        max_lag_s: float = 3.0,
        sign_align: bool = True,
    ) -> LagResult:
        F = self._slice(self.f.joint_center_array)
        Q = self._slice(self.q.joint_center_array)

        if method == LagMethod.ENERGY:
            a = _energy_series(F)
            b = _energy_series(Q)
        elif method == LagMethod.KINETIC:
            a = _kinetic_series(F, fps=self.fps, masses=self.masses)
            b = _kinetic_series(Q, fps=self.fps, masses=self.masses)
        elif method == LagMethod.PC1:
            a = _pc1_series(F)
            b = _pc1_series(Q)
        else:
            raise ValueError(f"Unknown lag method: {method}")

        lag_frames, peak = _xcorr_lag(a, b, fps=self.fps, max_lag_s=max_lag_s, sign_align=sign_align)
        self._last_frames = float(lag_frames)
        return LagResult(
            lag_frames=self._last_frames,
            lag_seconds=self._last_frames / self.fps,
            peak_value=peak,
            series_a=a,
            series_b=b,
        )

    # ----------------------- compatibility helpers ---------------------------
    def run(self) -> int:
        return self.estimate(LagMethod.ENERGY).lag_frames

    @property
    def median_lag(self) -> int:
        return int(self._last_frames)

    def get_lag_in_seconds(self, lag: Optional[int] = None) -> float:
        frames = self._last_frames if lag is None else int(lag)
        return frames / self.fps

    # ---------------------------- internals ----------------------------------
    def _slice(self, arr: np.ndarray) -> np.ndarray:
        return arr[self.start:self.end]


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------
def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (x - mu) / sd


def _flatten_3d(tj3: np.ndarray) -> np.ndarray:
    T, J, C = tj3.shape
    assert C == 3, "Expected last dimension = 3"
    return tj3.reshape(T, J * C)


def _pc1_series(tj3: np.ndarray) -> np.ndarray:
    X = _flatten_3d(tj3)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    U, S, _ = np.linalg.svd(np.nan_to_num(Z), full_matrices=False)
    if S.size == 0:
        return np.zeros(Z.shape[0], dtype=float)
    s1 = U[:, 0] * S[0]
    return _zscore(s1)


def _energy_series(tj3: np.ndarray) -> np.ndarray:
    """Motion energy (unitless): sum_j ||Δpos_j||^2 per frame (length T-1)."""
    d = np.diff(tj3, axis=0)                 # (T-1, J, 3)
    v = np.linalg.norm(d, axis=2)            # (T-1, J)
    e = np.sum(np.square(v), axis=1)         # (T-1,)
    return _zscore(e)


def _kinetic_series(tj3: np.ndarray, *, fps: float, masses: Optional[np.ndarray]) -> np.ndarray:
    """
    Kinetic energy proxy per frame (length T-1):
      KE_t = 0.5 * sum_j m_j * ||v_j||^2,  v_j in m/s
    If masses is None, treat all joints equally (m_j = 1).
    Note: constant scale factors (0.5, unit conversions) don't affect xcorr.
    """
    dt = 1.0 / float(fps)
    d = np.diff(tj3, axis=0) / dt            # velocities ~ m/s -> (T-1, J, 3)
    speed2 = np.sum(d * d, axis=2)           # (T-1, J)
    if masses is None:
        ke = 0.5 * np.sum(speed2, axis=1)    # (T-1,)
    else:
        ke = 0.5 * speed2 @ masses           # (T-1,)
    return _zscore(ke)


def _xcorr_lag(
    a: np.ndarray,
    b: np.ndarray,
    *,
    fps: float,
    max_lag_s: float = 3.0,
    sign_align: bool = True,
) -> Tuple[int, float]:
    """Return lag in FRAMES; positive means b lags a (shift b forward)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(min(len(a), len(b)))
    if n < 3:
        return 0, float("nan")

    a = a[:n] - np.nanmean(a[:n])
    b = b[:n] - np.nanmean(b[:n])

    if sign_align:
        r = np.corrcoef(np.nan_to_num(a), np.nan_to_num(b))[0, 1]
        if np.isfinite(r) and r < 0:
            b = -b

    cc = np.correlate(np.nan_to_num(a), np.nan_to_num(b), mode="full")
    lags = np.arange(-n + 1, n)

    max_lag = max(1, int(round(float(max_lag_s) * float(fps))))
    mid = len(cc) // 2
    left = max(0, mid - max_lag)
    right = min(len(cc), mid + max_lag + 1)

    cc_win = cc[left:right]
    lags_win = lags[left:right]

    k = int(np.argmax(cc_win))
    return int(lags_win[k]), float(cc_win[k])
