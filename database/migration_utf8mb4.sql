-- ============================================================
-- Skinglow - convert all tables to utf8mb4 so Thai text is stored
--
-- HOW TO RUN (phpMyAdmin on InfinityFree):
--   1. In the LEFT panel click your Skinglow database
--      (the one named like if0_XXXXXXXX_skinglow, NOT information_schema).
--   2. Open the SQL tab, paste this file, click Go.
--
-- Why: tables created without a charset take the server default (latin1).
-- Thai characters cannot be stored in latin1, so MySQL saved them as '?'.
-- Names already saved as '????' cannot be recovered - re-enter them after.
-- (ALTER DATABASE is not used: free hosting accounts are not allowed to run it.)
-- ============================================================

ALTER TABLE users            CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE analysis_history CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE products         CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE password_resets  CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE daily_insights   CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- check: the Collation column should now be utf8mb4_unicode_ci for every table
SHOW TABLE STATUS;
