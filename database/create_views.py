import sqlite3
from pathlib import Path


VIEWS_SQL = """
DROP VIEW IF EXISTS v_artifacts;
DROP VIEW IF EXISTS v_completeness_trial_tracker;
DROP VIEW IF EXISTS v_missing_artifacts;


CREATE VIEW v_artifacts AS
SELECT
    a.id,
    t.participant_code,
    t.trial_name,
    t.trial_path,
    t.trial_type,
    t.trial_number,
    a.category,
    a.condition,
    a.tracker,
    a.component_name,
    a.path,
    a.relative_path,
    a.file_exists,
    a.size_bytes,
    a.mtime_utc
FROM artifacts a
JOIN trials t
    ON a.trial_id = t.id;


CREATE VIEW v_completeness_trial_tracker AS
SELECT
    t.participant_code,
    t.trial_name,
    t.trial_path,
    a.tracker,
    COUNT(*) AS total_artifacts,
    SUM(
        CASE
            WHEN a.file_exists = 1 THEN 1
            ELSE 0
        END
    ) AS present_artifacts,
    COUNT(*) - SUM(
        CASE
            WHEN a.file_exists = 1 THEN 1
            ELSE 0
        END
    ) AS missing_artifacts,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN a.file_exists = 1 THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        1
    ) AS pct_present
FROM artifacts a
JOIN trials t
    ON a.trial_id = t.id
GROUP BY
    t.participant_code,
    t.trial_name,
    t.trial_path,
    a.tracker;


CREATE VIEW v_missing_artifacts AS
SELECT
    t.participant_code,
    t.trial_name,
    t.trial_path,
    a.tracker,
    a.category,
    a.condition,
    a.component_name,
    a.relative_path,
    a.path
FROM artifacts a
JOIN trials t
    ON a.trial_id = t.id
WHERE a.file_exists = 0
ORDER BY
    t.participant_code,
    t.trial_name,
    a.tracker,
    a.category,
    a.condition,
    a.component_name;
"""


def create_views(
    connection: sqlite3.Connection,
) -> None:
    """
    Recreate convenience views for inspecting the validation database.
    """
    connection.executescript(VIEWS_SQL)
    connection.commit()