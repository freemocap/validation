import sqlite3
from pathlib import Path

from database.db import (
    clear_database,
    connect_database,
    initialize_database,
)
from database.scanner import (
    ArtifactScan,
    TrialScan,
    discover_trial_configs,
    scan_trial_artifacts,
    scan_trial_config,
)


def insert_trial(
    connection: sqlite3.Connection,
    trial_scan: TrialScan,
) -> int:
    """
    Insert one trial row and return its database ID.
    """
    identity = trial_scan.identity

    cursor = connection.execute(
        """
        INSERT INTO trials (
            participant_code,
            trial_name,
            trial_path,
            trial_type,
            trial_number,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            identity.participant_code,
            identity.trial_name,
            identity.trial_path,
            identity.trial_type,
            identity.trial_number,
            None,
        ),
    )

    return int(cursor.lastrowid)


def insert_artifact(
    connection: sqlite3.Connection,
    trial_id: int,
    artifact: ArtifactScan,
) -> int:
    """
    Insert one artifact row and return its database ID.
    """
    cursor = connection.execute(
        """
        INSERT INTO artifacts (
            trial_id,
            category,
            condition,
            tracker,
            component_name,
            path,
            relative_path,
            file_exists,
            size_bytes,
            mtime_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trial_id,
            artifact.category,
            artifact.condition,
            artifact.tracker,
            artifact.component_name,
            str(artifact.path),
            artifact.relative_path,
            int(artifact.file_exists),
            artifact.size_bytes,
            artifact.mtime_utc,
        ),
    )

    return int(cursor.lastrowid)


def build_database(
    dataset_root: Path,
    config_root: Path,
    database_path: Path,
    overwrite: bool = True,
    only_existing_artifacts: bool = False,
) -> None:
    """
    Build the public validation database from trial configs and expected
    DataComponents.
    """
    dataset_root = dataset_root.expanduser().resolve()
    config_root = config_root.expanduser().resolve()
    database_path = database_path.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    if not config_root.exists():
        raise FileNotFoundError(
            f"Config root does not exist: {config_root}"
        )

    connection = connect_database(
        database_path
    )

    try:
        initialize_database(
            connection
        )

        if overwrite:
            clear_database(
                connection
            )

        config_paths = discover_trial_configs(
            config_root
        )

        print(
            f"Found {len(config_paths)} trial configs."
        )
        print(
            f"Dataset root: {dataset_root}"
        )
        print(
            f"Database path: {database_path}"
        )
        print(
            "Artifact mode: "
            + (
                "existing files only"
                if only_existing_artifacts
                else "all expected artifacts"
            )
        )
        print()

        total_artifacts = 0
        total_present = 0
        total_missing = 0

        for config_path in config_paths:
            trial_scan = scan_trial_config(
                config_path=config_path,
                dataset_root=dataset_root,
            )

            trial_id = insert_trial(
                connection=connection,
                trial_scan=trial_scan,
            )

            artifacts = scan_trial_artifacts(
                trial_scan=trial_scan,
                dataset_root=dataset_root,
                only_existing=only_existing_artifacts,
            )

            present_count = 0
            missing_count = 0

            for artifact in artifacts:
                insert_artifact(
                    connection=connection,
                    trial_id=trial_id,
                    artifact=artifact,
                )

                if artifact.file_exists:
                    present_count += 1
                else:
                    missing_count += 1

            total_artifacts += len(artifacts)
            total_present += present_count
            total_missing += missing_count

            print(
                f"Indexed {trial_scan.identity.trial_path}: "
                f"{len(artifacts)} artifacts "
                f"({present_count} present, "
                f"{missing_count} missing)"
            )

        connection.commit()

        trial_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM trials
            """
        ).fetchone()[0]

        artifact_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM artifacts
            """
        ).fetchone()[0]

        database_present = connection.execute(
            """
            SELECT COUNT(*)
            FROM artifacts
            WHERE file_exists = 1
            """
        ).fetchone()[0]

        database_missing = connection.execute(
            """
            SELECT COUNT(*)
            FROM artifacts
            WHERE file_exists = 0
            """
        ).fetchone()[0]

        print()
        print("Database build complete")
        print(f"Trials:            {trial_count}")
        print(f"Artifacts:         {artifact_count}")
        print(f"Present artifacts: {database_present}")
        print(f"Missing artifacts: {database_missing}")

        if artifact_count != total_artifacts:
            raise RuntimeError(
                "Inserted artifact count does not match "
                "the scanned artifact count."
            )

        if database_present != total_present:
            raise RuntimeError(
                "Present-artifact database count does not match "
                "the scanned count."
            )

        if database_missing != total_missing:
            raise RuntimeError(
                "Missing-artifact database count does not match "
                "the scanned count."
            )

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def print_database_summary(
    database_path: Path,
) -> None:
    """
    Print basic database counts grouped by tracker and category.
    """
    database_path = database_path.expanduser().resolve()

    connection = sqlite3.connect(
        database_path
    )

    try:
        rows = connection.execute(
            """
            SELECT
                tracker,
                category,
                COUNT(*) AS artifact_count,
                SUM(file_exists) AS present_count,
                COUNT(*) - SUM(file_exists) AS missing_count
            FROM artifacts
            GROUP BY
                tracker,
                category
            ORDER BY
                tracker,
                category
            """
        ).fetchall()

        print()
        print("Artifact summary")

        for (
            tracker,
            category,
            artifact_count,
            present_count,
            missing_count,
        ) in rows:
            print(
                f"{tracker:12s} "
                f"{category:30s} "
                f"total={artifact_count:4d} "
                f"present={present_count:4d} "
                f"missing={missing_count:4d}"
            )

    finally:
        connection.close()


if __name__ == "__main__":
    repo_root = (
        Path(__file__).resolve().parents[1]
    )

    # -----------------------------------------------------------------
    # USER SETTINGS
    # -----------------------------------------------------------------

    dataset_root = Path(
        r"D:\validation_public_release_v1\data"
    )

    config_root = (
        repo_root
        / "configs"
    )

    database_path = (
        repo_root
        / "validation_public_test.db"
    )

    overwrite_database = True

    # False preserves rows for expected-but-missing artifacts.
    # This is useful for completeness checks and matches the behavior
    # of the original database design more closely.
    only_existing_artifacts = False

    # -----------------------------------------------------------------

    build_database(
        dataset_root=dataset_root,
        config_root=config_root,
        database_path=database_path,
        overwrite=overwrite_database,
        only_existing_artifacts=only_existing_artifacts,
    )

    print_database_summary(
        database_path
    )