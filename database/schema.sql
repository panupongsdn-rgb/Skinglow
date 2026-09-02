-- ============================================================
-- Skinglow Database Schema
-- MySQL 8.0+ (utf8mb4)
-- ============================================================

CREATE DATABASE IF NOT EXISTS skinglow_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE skinglow_db;

-- ------------------------------------------------------------
-- Table: users
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    full_name       VARCHAR(150)    NOT NULL,
    email           VARCHAR(190)    NOT NULL UNIQUE,
    password_hash   VARCHAR(255)    NOT NULL,
    skin_type       ENUM('normal','oily','dry','combination','sensitive') DEFAULT NULL,
    avatar_path     VARCHAR(255)    DEFAULT NULL,
    is_active       TINYINT(1)      NOT NULL DEFAULT 1,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
                                     ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Table: analysis_history
-- Stores every skin-analysis run performed by a user
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analysis_history (
    id                  INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id             INT UNSIGNED NOT NULL,
    original_image_path VARCHAR(255) NOT NULL,
    processed_image_path VARCHAR(255) DEFAULT NULL, -- image with bounding boxes drawn
    skin_score          DECIMAL(5,2) DEFAULT NULL,   -- overall skin health score 0-100
    detections_json      JSON        DEFAULT NULL,   -- raw AI output: boxes/labels/confidence
    summary              TEXT        DEFAULT NULL,   -- human readable summary
    created_at           TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_analysis_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Table: products
-- Skincare products / ingredients catalogue
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    brand           VARCHAR(150) DEFAULT NULL,
    description     TEXT DEFAULT NULL,
    key_ingredients VARCHAR(255) DEFAULT NULL,   -- e.g. "Niacinamide, Salicylic Acid"
    target_issue    ENUM('acne','black_spot','eyebag','redness','oiliness','wrinkle','dryness','general')
                     NOT NULL DEFAULT 'general',
    price           DECIMAL(10,2) DEFAULT NULL,
    image_url       VARCHAR(255) DEFAULT NULL,
    purchase_url    VARCHAR(255) DEFAULT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE INDEX idx_products_target_issue ON products(target_issue);
CREATE INDEX idx_analysis_user ON analysis_history(user_id);

-- ------------------------------------------------------------
-- Seed data (sample)
-- ------------------------------------------------------------
INSERT INTO products (name, brand, description, key_ingredients, target_issue, price)
VALUES
('Salicylic Acid Acne Spot Gel', 'GlowLab', 'เจลจุดสิวช่วยลดการอักเสบและควบคุมความมัน', 'Salicylic Acid 2%, Tea Tree Oil', 'acne', 259.00),
('Niacinamide Serum 10%', 'GlowLab', 'เซรั่มลดรอยดำและกระชับรูขุมขน', 'Niacinamide, Zinc PCA', 'black_spot', 399.00),
('Caffeine Eye Cream', 'GlowLab', 'ครีมทารอบดวงตาลดถุงใต้ตาและอาการบวมคล้ำ', 'Caffeine, Peptides, Vitamin K', 'eyebag', 349.00),
('Centella Calming Cream', 'GreenSkin', 'ครีมช่วยลดรอยแดงและปลอบประโลมผิว', 'Centella Asiatica, Panthenol', 'redness', 320.00),
('Oil Control Mattifying Gel', 'PureDerm', 'เจลควบคุมความมันตลอดวัน', 'Zinc, Witch Hazel', 'oiliness', 280.00),
('Retinol Renewal Night Cream', 'GlowLab', 'ครีมกลางคืนลดริ้วรอยและกระตุ้นการผลัดเซลล์ผิว', 'Retinol 0.3%, Peptides', 'wrinkle', 590.00),
('Hyaluronic Acid Hydrating Serum', 'PureDerm', 'เซรั่มเติมความชุ่มชื้นเข้มข้น', 'Hyaluronic Acid, Ceramide', 'dryness', 350.00),
('Daily Broad-Spectrum Sunscreen SPF50', 'PureDerm', 'กันแดดเนื้อบางเบา ปกป้องผิวใสให้คงสภาพดีต่อเนื่อง', 'Zinc Oxide, Niacinamide', 'general', 320.00),
('Gentle Vitamin C Serum', 'GlowLab', 'เซรั่มบำรุงเบาๆ สำหรับผิวที่แข็งแรงอยู่แล้ว ช่วยคงความสดใส', 'Ascorbyl Glucoside', 'general', 380.00);
