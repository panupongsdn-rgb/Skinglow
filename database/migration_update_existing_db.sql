-- ============================================================
-- Skinglow: bring an EXISTING database up to date (safe to run more than once)
-- Run in phpMyAdmin -> select the database -> SQL tab -> Go
-- Works on XAMPP (MariaDB) and InfinityFree (MariaDB).
-- ============================================================

-- 1. per-zone analysis results (forehead / cheeks / nose / under-eye / chin)
ALTER TABLE analysis_history
    ADD COLUMN IF NOT EXISTS zones_json LONGTEXT DEFAULT NULL AFTER detections_json;

-- 2. forgot / reset password links
CREATE TABLE IF NOT EXISTS password_resets (
    id          INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id     INT UNSIGNED NOT NULL,
    token_hash  CHAR(64)     NOT NULL,
    expires_at  DATETIME     NOT NULL,
    used_at     DATETIME     DEFAULT NULL,
    created_at  DATETIME     NOT NULL,
    CONSTRAINT fk_reset_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY uq_reset_token (token_hash),
    KEY idx_reset_user_created (user_id, created_at)
) ENGINE=InnoDB;

-- 3. admin "insights" articles
CREATE TABLE IF NOT EXISTS daily_insights (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    title           VARCHAR(255) NOT NULL,
    category        VARCHAR(100) DEFAULT NULL,
    content         TEXT NOT NULL,
    image_url       VARCHAR(500) DEFAULT NULL,
    is_active       TINYINT(1) NOT NULL DEFAULT 1,
    publish_date    DATE DEFAULT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_daily_insights_date (publish_date),
    KEY idx_daily_insights_active (is_active)
) ENGINE=InnoDB;
