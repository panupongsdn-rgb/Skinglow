<?php
declare(strict_types=1);

// Base paths
define('BASE_PATH', dirname(__DIR__));
define('UPLOAD_ORIGINAL_DIR', BASE_PATH . '/uploads/original/');
define('UPLOAD_PROCESSED_DIR', BASE_PATH . '/uploads/processed/');

// Public URL used to build links returned to the frontend
define('APP_BASE_URL', getenv('APP_BASE_URL') ?: 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/api/login.php');

// AI microservice endpoint (FastAPI / Python)
define('AI_SERVICE_URL', getenv('AI_SERVICE_URL') ?: 'https://skinglow-ai-service.onrender.com/analyze');

// Upload constraints
define('MAX_UPLOAD_SIZE', 8 * 1024 * 1024); // 8 MB
define('ALLOWED_MIME_TYPES', ['image/jpeg', 'image/png', 'image/webp']);

// JWT secret for auth (replace with a strong secret via env var in production)
define('JWT_SECRET', getenv('JWT_SECRET') ?: 'CHANGE_ME_TO_A_LONG_RANDOM_SECRET');

// CORS - restrict to your frontend origin(s) in production
header('Access-Control-Allow-Origin: https://skinglow-eck.pages.dev');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(204);
    exit;
}
