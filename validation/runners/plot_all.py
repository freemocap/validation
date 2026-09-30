from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

from validation.paths import REPO_ROOT


@dataclass(frozen=True)
class PlotTask:
    name: str
    module: str
    section: str


PLOT_TASKS = [
    PlotTask(
        name="Balance metrics figure",
        module="plotting.balance.balance_metrics_full_fig",
        section="balance",
    ),
    PlotTask(
        name="Balance metrics table",
        module="plotting.balance.balance_metrics_table",
        section="balance",
    ),
    PlotTask(
        name="COM velocity violin",
        module="plotting.balance.com_violin_plots",
        section="balance",
    ),
    PlotTask(
        name="Path length agreement",
        module="plotting.balance.path_length_agreement_all_trackers",
        section="balance",
    ),
    PlotTask(
        name="Path length sensitivity",
        module="plotting.balance.path_length_sensitivity_all_trackers",
        section="balance",
    ),
    PlotTask(
        name="COM XY plane",
        module="plotting.balance.plot_xy_plane",
        section="balance",
    ),
    PlotTask(
        name="Balance sensitivity and agreement",
        module="plotting.balance.sensitivity_and_agreement_plots",
        section="balance",
    ),
    PlotTask(
        name="Gait Bland-Altman figure",
        module="plotting.gait.gait_parameters.gait_ba_figure",
        section="gait",
    ),
    PlotTask(
        name="Gait Bland-Altman tables",
        module="plotting.gait.gait_parameters.gait_ba_tables",
        section="gait",
    ),
    PlotTask(
        name="Joint angle RMSE",
        module="plotting.gait.joint_angles.joint_angle_rmse",
        section="gait",
    ),
    PlotTask(
        name="Joint angles and SPM",
        module="plotting.gait.joint_angles.normalized_angles_and_spm",
        section="gait",
    ),
    PlotTask(
        name="Joint trajectories",
        module="plotting.gait.joint_trajectories.joint_trajectories_xyz_plots",
        section="gait",
    ),
    PlotTask(
        name="Joint trajectory RMSE grid",
        module="plotting.gait.joint_trajectories.joint_trajectory_rmse_grid",
        section="gait",
    ),
    PlotTask(
        name="Joint trajectory RMSE tables",
        module="plotting.gait.joint_trajectories.joint_trajectory_rmse_tables",
        section="gait",
    ),
]


def run_plot_tasks(
    section: str = "all",
    show: bool = False,
) -> int:
    tasks = [
        task
        for task in PLOT_TASKS
        if section == "all"
        or task.section == section
    ]

    env = os.environ.copy()
    env["VALIDATION_SHOW_FIGURES"] = (
        "1" if show else "0"
    )

    failed = []

    print(
        f"Running {len(tasks)} plotting tasks"
    )
    print(
        f"Show figures: {show}"
    )

    for index, task in enumerate(
        tasks,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(tasks)}] "
            f"{task.name}"
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                task.module,
            ],
            cwd=REPO_ROOT,
            env=env,
        )

        if result.returncode == 0:
            print(
                f"[{index}/{len(tasks)}] PASSED"
            )
        else:
            print(
                f"[{index}/{len(tasks)}] FAILED"
            )
            failed.append(task)

    print()
    print("Plotting complete")
    print(
        f"Successful: "
        f"{len(tasks) - len(failed)}"
    )
    print(
        f"Failed:     {len(failed)}"
    )

    if failed:
        print()
        print("Failed tasks:")

        for task in failed:
            print(
                f"  - {task.name}"
            )

        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FreeMoCap validation "
            "figures and tables."
        )
    )

    parser.add_argument(
        "--section",
        choices=[
            "all",
            "balance",
            "gait",
        ],
        default="all",
        help=(
            "Generate all outputs, balance "
            "outputs, or gait outputs."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Display figures as they are "
            "generated."
        ),
    )

    args = parser.parse_args()

    raise SystemExit(
        run_plot_tasks(
            section=args.section,
            show=args.show,
        )
    )


if __name__ == "__main__":
    main()