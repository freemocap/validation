"""
SPM paired t-tests for joint angle waveforms (tracker vs Qualisys).

Key importable function:
  - run_spm_paired_ttests(df_trial_lr_mean, ...) -> (spm_clusters, spm_curves)

When run as a script, loads data from validation.db and exports CSVs.
"""

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import spm1d


def speed_key(cond: str) -> float:
    m = re.search(r"speed_(\d+)[_\.](\d+)", str(cond))
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m2 = re.search(r"speed_(\d+)", str(cond))
    if m2:
        return float(m2.group(1))
    return float("inf")


@dataclass
class SpmResultRow:
    condition: str
    joint: str
    component: str
    tracker: str
    n_trials: int
    two_tailed: bool
    alpha: float
    t_star: float
    n_clusters: int
    cluster_idx: int
    p_value: float
    start_node: int
    end_node: int
    start_gait_pct: float
    end_gait_pct: float
    extent_nodes: int


def _build_trial_waveform_matrix(
    df_trial_lr_mean: pd.DataFrame,
    condition: str,
    joint: str,
    component: str,
    tracker: str,
    reference: str = "qualisys",
    q_expected: int | None = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Y_ref: (J, Q)
      Y_trk: (J, Q)
      x:     (Q,) gait percent values (sorted)
    """
    sub = df_trial_lr_mean[
        (df_trial_lr_mean["condition"] == condition)
        & (df_trial_lr_mean["joint"] == joint)
        & (df_trial_lr_mean["component"] == component)
        & (df_trial_lr_mean["tracker"].isin([reference, tracker]))
    ].copy()

    if sub.empty:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,))

    piv = (
        sub.pivot_table(
            index=["participant_code", "trial_name", "percent_gait_cycle"],
            columns="tracker",
            values="trial_mean_angle",
            aggfunc="first",
        )
        .reset_index()
    )

    if reference not in piv.columns or tracker not in piv.columns:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,))
    piv = piv.dropna(subset=[reference, tracker])

    ref_wide = piv.pivot_table(
        index=["participant_code", "trial_name"],
        columns="percent_gait_cycle",
        values=reference,
        aggfunc="first",
    )
    trk_wide = piv.pivot_table(
        index=["participant_code", "trial_name"],
        columns="percent_gait_cycle",
        values=tracker,
        aggfunc="first",
    )

    common_trials = ref_wide.index.intersection(trk_wide.index)
    ref_wide = ref_wide.loc[common_trials]
    trk_wide = trk_wide.loc[common_trials]

    x = np.array(sorted(ref_wide.columns.astype(float)))
    ref_wide = ref_wide.reindex(columns=x)
    trk_wide = trk_wide.reindex(columns=x)

    good = (~ref_wide.isna().any(axis=1)) & (~trk_wide.isna().any(axis=1))
    ref_wide = ref_wide.loc[good]
    trk_wide = trk_wide.loc[good]

    if q_expected is not None and ref_wide.shape[1] != q_expected:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0,))

    return ref_wide.to_numpy(dtype=float), trk_wide.to_numpy(dtype=float), x


def run_spm_paired_ttests(
    df_trial_lr_mean: pd.DataFrame,
    trackers: list[str],
    reference: str = "qualisys",
    alpha: float = 0.05,
    two_tailed: bool = True,
    q_expected: int | None = 101,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run SPM{t} paired tests for each condition × joint × component × tracker vs reference.

    Returns:
      spm_clusters: DataFrame with cluster intervals and p-values
      spm_curves:   DataFrame with SPM{t} curves per condition/joint/tracker
    """
    results: list[SpmResultRow] = []
    curves = []

    conditions = sorted(df_trial_lr_mean["condition"].unique().tolist(), key=speed_key)
    targets = [("hip", "flex_ext"), ("knee", "flex_ext"), ("ankle", "dorsi_plantar")]

    for condition in conditions:
        for joint, component in targets:
            for tracker in trackers:
                if tracker == reference:
                    continue

                Y_ref, Y_trk, x = _build_trial_waveform_matrix(
                    df_trial_lr_mean=df_trial_lr_mean,
                    condition=condition,
                    joint=joint,
                    component=component,
                    tracker=tracker,
                    reference=reference,
                    q_expected=q_expected,
                )

                J, Q = Y_ref.shape if Y_ref.size else (0, 0)
                if J < 3 or Q == 0:
                    continue

                t = spm1d.stats.ttest_paired(Y_ref, Y_trk)
                ti = t.inference(alpha=alpha, two_tailed=two_tailed)

                t_star = float(ti.zstar)
                clusters = getattr(ti, "clusters", []) or []
                spm_t = np.asarray(t.z, dtype=float)

                curves.append(
                    pd.DataFrame(
                        {
                            "condition": condition,
                            "joint": joint,
                            "component": component,
                            "tracker": tracker,
                            "reference": reference,
                            "alpha": alpha,
                            "two_tailed": two_tailed,
                            "n_trials": J,
                            "t_star": t_star,
                            "percent_gait_cycle": x.astype(float),
                            "spm_t": spm_t,
                        }
                    )
                )

                if len(clusters) == 0:
                    results.append(
                        SpmResultRow(
                            condition=condition, joint=joint, component=component,
                            tracker=tracker, n_trials=J, two_tailed=two_tailed,
                            alpha=alpha, t_star=t_star, n_clusters=0,
                            cluster_idx=-1, p_value=np.nan,
                            start_node=-1, end_node=-1,
                            start_gait_pct=np.nan, end_gait_pct=np.nan,
                            extent_nodes=0,
                        )
                    )
                    continue

                for k, c in enumerate(clusters):
                    a, b = c.endpoints
                    start_node = max(0, min(Q - 1, int(np.floor(a))))
                    end_node = max(0, min(Q - 1, int(np.ceil(b))))

                    results.append(
                        SpmResultRow(
                            condition=condition, joint=joint, component=component,
                            tracker=tracker, n_trials=J, two_tailed=two_tailed,
                            alpha=alpha, t_star=t_star, n_clusters=len(clusters),
                            cluster_idx=k, p_value=float(c.P),
                            start_node=start_node, end_node=end_node,
                            start_gait_pct=float(x[start_node]),
                            end_gait_pct=float(x[end_node]),
                            extent_nodes=int(end_node - start_node + 1),
                        )
                    )

                print(
                    f"SPM: {condition} | {joint} {component} | {tracker} vs {reference}"
                    f"  n={J}, t*={t_star:.3f}, clusters={len(clusters)}"
                )
                for k, c in enumerate(clusters):
                    a, b = c.endpoints
                    sn = max(0, min(Q - 1, int(np.floor(a))))
                    en = max(0, min(Q - 1, int(np.ceil(b))))
                    print(f"   cluster {k}: {x[sn]:.1f}%–{x[en]:.1f}% GC, p={c.P:.4f}")

    out_clusters = pd.DataFrame([r.__dict__ for r in results])
    out_curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return out_clusters, out_curves


if __name__ == "__main__":
    from joint_angles_plots import load_joint_angle_data, compute_angle_summary

    root_dir = Path(r"D:\validation\gait\joint_angles")
    root_dir.mkdir(exist_ok=True, parents=True)

    REFERENCE_SYSTEM = "qualisys"
    SPM_TRACKERS = ["mediapipe", "rtmpose", "vitpose"]
    ALPHA = 0.05
    TWO_TAILED = True
    Q_EXPECTED = 100

    combined_df = load_joint_angle_data()
    _, df_trial_lr_mean = compute_angle_summary(combined_df)

    print("Unique trackers:", sorted(df_trial_lr_mean["tracker"].unique()))
    print("Unique joints:", sorted(df_trial_lr_mean["joint"].unique()))
    print("Unique components:", sorted(df_trial_lr_mean["component"].unique()))

    spm_clusters, spm_curves = run_spm_paired_ttests(
        df_trial_lr_mean=df_trial_lr_mean,
        trackers=SPM_TRACKERS,
        reference=REFERENCE_SYSTEM,
        alpha=ALPHA,
        two_tailed=TWO_TAILED,
        q_expected=Q_EXPECTED,
    )

    print("\nSPM clusters (head):")
    print(spm_clusters.head(20))

    spm_clusters.to_csv(root_dir / "spm_paired_ttest_clusters.csv", index=False)
    spm_curves.to_csv(root_dir / "spm_paired_ttest_curves.csv", index=False)
    print("Saved:", root_dir / "spm_paired_ttest_clusters.csv")
    print("Saved:", root_dir / "spm_paired_ttest_curves.csv")