-- ============================================================
-- Skinglow - convert all tables to utf8mb4 so Thai text is stored
-- Run once in phpMyAdmin (select the Skinglow database first, then the SQL tab).
--
-- Why: tables created without a charset take the server default, which on
-- InfinityFree is latin1. Thai characters cannot be stored in latin1, so MySQL
-- saves them as '?'. Names already saved as '????' cannot be recovered by
-- this script - re-enter them after running it.
-- ============================================================

-- 1) check the current charset (latin1_* = the problem)
SELECT TABLE_NAME, TABLE_COLLATION
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = DATABASE();

-- 2) make new tables default to utf8mb4
ALTER DATABASE CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 3) convert every existing table and its text columns
ALTER TABLE users            CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE analysis_history CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE products         CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE password_resets  CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
ALTER TABLE daily_insights   CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 4) check again (all should now be utf8mb4_unicode_ci)
SELECT TABLE_NAME, TABLE_COLLATION
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = DATABASE();
