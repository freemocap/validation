from __future__ import annotations

import argparse
from pathlib import Path

from database.index_recordings import (
    build_database,
    print_database_summary,
)
from validation.paths import (
    DATABASE_PATH,
    DATA_ROOT,
    DATASET_ROOT,
    REPO_ROOT,
)
from validation.download_data import download_dataset
from validation.runners.plot_all import run_plot_tasks
from validation.runners.run_batch import run_batch


DEFAULT_DATASET_ROOT = DATA_ROOT
DEFAULT_CONFIG_ROOT = REPO_ROOT / "configs"

def download_data_command(
    args: argparse.Namespace,
) -> int:
    download_dataset(
        destination=args.destination,
        force=args.force,
    )

    return 0

def run_command(args: argparse.Namespace) -> int:
    results = run_batch(
        config_root=args.config_root,
        dataset_root=args.dataset_root,
        trackers=args.trackers,
        start_at=args.start_at,
        use_rigid=args.use_rigid,
        config_pattern=args.config_pattern,
    )

    failed = [
        result
        for result in results
        if not result.succeeded
    ]

    return 1 if failed else 0


def build_db_command(args: argparse.Namespace) -> int:
    build_database(
        dataset_root=args.dataset_root,
        config_root=args.config_root,
        database_path=args.database_path,
        overwrite=args.overwrite,
        only_existing_artifacts=args.only_existing_artifacts,
    )

    print_database_summary(
        args.database_path
    )

    return 0


def plot_all_command(args: argparse.Namespace) -> int:
    return run_plot_tasks(
        section=args.section,
        show=args.show,
    )


def build_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(
        prog="validation",
        description=(
            "FreeMoCap validation and "
            "reproducibility tools."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    download_parser = subparsers.add_parser(
        "download-data",
        help=(
            "Download the validation dataset "
            "from Zenodo."
        ),
    )

    download_parser.add_argument(
        "--destination",
        type=Path,
        default=DATASET_ROOT,
        help=(
            "Dataset destination. Defaults to "
            "repo/freemocap_validation_dataset."
        ),
    )

    download_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace an existing dataset "
            "directory."
        ),
    )

    download_parser.set_defaults(
        func=download_data_command,
)

    # ---------------------------------------------------------
    # run
    # ---------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run the validation analysis pipeline.",
    )

    run_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Dataset directory. "
            "Defaults to repo/data."
        ),
    )

    run_parser.add_argument(
        "--config-root",
        type=Path,
        default=DEFAULT_CONFIG_ROOT,
        help=(
            "Trial config directory. "
            "Defaults to repo/configs."
        ),
    )

    run_parser.add_argument(
        "--tracker",
        action="append",
        dest="trackers",
        default=None,
        help=(
            "Optional tracker to run. "
            "Repeat for multiple trackers."
        ),
    )

    run_parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Pipeline step index to start at.",
    )

    run_parser.add_argument(
        "--use-rigid",
        action="store_true",
        help="Use rigid-body trajectories where supported.",
    )

    run_parser.add_argument(
        "--config-pattern",
        default="*.yaml",
        help="Config filename pattern.",
    )

    run_parser.set_defaults(
        func=run_command,
    )

    # ---------------------------------------------------------
    # build-db
    # ---------------------------------------------------------
    db_parser = subparsers.add_parser(
        "build-db",
        help="Build the validation artifact database.",
    )

    db_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Dataset directory. "
            "Defaults to repo/data."
        ),
    )

    db_parser.add_argument(
        "--config-root",
        type=Path,
        default=DEFAULT_CONFIG_ROOT,
        help=(
            "Trial config directory. "
            "Defaults to repo/configs."
        ),
    )

    db_parser.add_argument(
        "--database-path",
        type=Path,
        default=DATABASE_PATH,
        help=(
            "SQLite database path. "
            "Defaults to repo/validation.db."
        ),
    )

    db_parser.add_argument(
        "--only-existing-artifacts",
        action="store_true",
        help="Index only artifacts that exist.",
    )

    db_parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help=(
            "Do not clear the existing database "
            "before indexing."
        ),
    )

    db_parser.set_defaults(
        overwrite=True,
        func=build_db_command,
    )

    # ---------------------------------------------------------
    # plot-all
    # ---------------------------------------------------------
    plot_parser = subparsers.add_parser(
        "plot-all",
        help="Generate validation figures and tables.",
    )

    plot_parser.add_argument(
        "--section",
        choices=[
            "all",
            "balance",
            "gait",
        ],
        default="all",
        help="Generate all, balance, or gait outputs.",
    )

    plot_parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures as they are generated.",
    )

    plot_parser.set_defaults(
        func=plot_all_command,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raise SystemExit(
        args.func(args)
    )


if __name__ == "__main__":
    main()