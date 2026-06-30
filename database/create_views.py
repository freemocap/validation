import sqlite3

VIEWS = """
-- paste the CREATE VIEW statements from earlier here
CREATE VIEW IF NOT EXISTS v_artifacts AS
SELECT
  a.id,
  t.participant_code,
  t.trial_name,
  t.trial_type,
  t.trial_number,
  t.session_date,
  a.category,
  a.condition,
  a.tracker,
  a.component_name,
  a.path,
  a.file_exists,
  a.size_bytes,
  a.mtime_utc
FROM artifacts a
JOIN trials t ON a.trial_id = t.id;

CREATE VIEW IF NOT EXISTS v_completeness_trial_tracker AS
SELECT
  t.participant_code,
  t.trial_name,
  a.tracker,
  COUNT(*)                         AS total_artifacts,
  SUM(CASE WHEN a.file_exists=1 THEN 1 ELSE 0 END) AS present_artifacts,
  ROUND(100.0 * SUM(CASE WHEN a.file_exists=1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_present
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
GROUP BY t.participant_code, t.trial_name, a.tracker;

CREATE VIEW IF NOT EXISTS v_missing_artifacts AS
SELECT
  t.participant_code, t.trial_name, a.tracker, a.category, a.condition, a.component_name, a.path
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE a.file_exists = 0
ORDER BY t.trial_name, a.tracker, a.category, a.condition, a.component_name;
"""

def create_views(db_path="validation.db"):
    conn = sqlite3.connect(db_path)
    conn.executescript(VIEWS)
    conn.commit()
    conn.close()
    print("✅ Views created or already existed.")

if __name__ == "__main__":
    create_views()