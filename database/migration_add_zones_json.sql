-- Adds per-zone analysis results (forehead / cheeks / nose / under-eye / chin)
-- to an EXISTING Skinglow database. New installs get the column from schema.sql.
-- Run once in phpMyAdmin (SQL tab) or: mysql -u root skinglow < migration_add_zones_json.sql
-- analyze.php / history.php keep working without it (zones are then not saved).

ALTER TABLE analysis_history
    ADD COLUMN zones_json JSON DEFAULT NULL AFTER detections_json;
