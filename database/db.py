import sqlite3
from pathlib import Path


TRIALS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY,

    participant_code TEXT NOT NULL,
    trial_name TEXT NOT NULL,
    trial_path TEXT NOT NULL UNIQUE,

    trial_type TEXT NOT NULL,
    trial_number INTEGER NOT NULL,

    notes TEXT,

    UNIQUE (
        participant_code,
        trial_name
    )
);
"""


ARTIFACTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY,

    trial_id INTEGER NOT NULL
        REFERENCES trials(id)
        ON DELETE CASCADE,

    category TEXT NOT NULL,
    condition TEXT NOT NULL DEFAULT '',
    tracker TEXT NOT NULL,
    component_name TEXT NOT NULL,

    path TEXT NOT NULL,
    relative_path TEXT NOT NULL,

    file_exists INTEGER NOT NULL,
    size_bytes INTEGER,
    mtime_utc REAL,

    UNIQUE (
        trial_id,
        category,
        condition,
        tracker,
        component_name
    )
);
"""


ARTIFACT_PATH_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_artifacts_path
ON artifacts(path);
"""


ARTIFACT_LOOKUP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_artifacts_lookup
ON artifacts(
    category,
    condition,
    tracker,
    component_name
);
"""


def connect_database(
    database_path: Path,
) -> sqlite3.Connection:
    """
    Open a SQLite database and enable foreign-key enforcement.
    """
    database_path = database_path.expanduser().resolve()

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


def initialize_database(
    connection: sqlite3.Connection,
) -> None:
    """
    Create database tables and indexes when they do not exist.
    """
    connection.execute(TRIALS_TABLE_SQL)
    connection.execute(ARTIFACTS_TABLE_SQL)
    connection.execute(ARTIFACT_PATH_INDEX_SQL)
    connection.execute(ARTIFACT_LOOKUP_INDEX_SQL)
    connection.commit()


def clear_database(
    connection: sqlite3.Connection,
) -> None:
    """
    Remove all indexed trials and artifacts.
    """
    connection.execute(
        "DELETE FROM artifacts"
    )
    connection.execute(
        "DELETE FROM trials"
    )
    connection.commit()