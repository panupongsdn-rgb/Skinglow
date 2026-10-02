<?php
declare(strict_types=1);

// Reads config/secrets.local.php (not in git). See secrets.local.example.php.
function localSecrets(): array
{
    static $cache = null;
    if ($cache === null) {
        $file = __DIR__ . '/secrets.local.php';
        $loaded = is_file($file) ? require $file : [];
        $cache = is_array($loaded) ? $loaded : [];
    }
    return $cache;
}

// Base paths
define('BASE_PATH', dirname(__DIR__));
define('UPLOAD_ORIGINAL_DIR', BASE_PATH . '/uploads/original/');
define('UPLOAD_PROCESSED_DIR', BASE_PATH . '/uploads/processed/');

// True when the site is opened on this computer (XAMPP: http://localhost:8080/...).
// Used to pick the local database / URLs so the same files work on XAMPP and InfinityFree.
function isLocalRequest(): bool
{
    if (PHP_SAPI === 'cli') {
        return true;
    }
    $host = strtolower(explode(':', $_SERVER['HTTP_HOST'] ?? '')[0]);
    return in_array($host, ['localhost', '127.0.0.1', '::1', '[::1]'], true);
}

// On localhost, build the base URL from the request (e.g. http://localhost:8080/skinglow/backend-php)
function localBaseUrl(): string
{
    $scheme = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off') ? 'https' : 'http';
    $script = str_replace('\\', '/', $_SERVER['SCRIPT_NAME'] ?? '');
    $pos = strpos($script, '/backend-php/');
    $root = $pos !== false ? substr($script, 0, $pos) . '/backend-php' : '/backend-php';
    return $scheme . '://' . ($_SERVER['HTTP_HOST'] ?? 'localhost') . $root;
}

// Public URL used to build links returned to the frontend
define('APP_BASE_URL', getenv('APP_BASE_URL')
    ?: (isLocalRequest() ? localBaseUrl() : 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php'));

// AI microservice endpoint (FastAPI / Python)
// Locally you can point to your own AI service with 'ai_service_url' in secrets.local.php
// (e.g. http://127.0.0.1:8000/analyze); otherwise the Render service is used everywhere.
define('AI_SERVICE_URL', getenv('AI_SERVICE_URL')
    ?: ((isLocalRequest() ? (localSecrets()['ai_service_url'] ?? '') : '')
        ?: 'https://skinglow-ai-service.onrender.com/analyze'));

// When a local AI service is configured but not running, analyze.php retries on Render.
define('AI_SERVICE_FALLBACK_URL',
    AI_SERVICE_URL !== 'https://skinglow-ai-service.onrender.com/analyze' && isLocalRequest()
        ? 'https://skinglow-ai-service.onrender.com/analyze' : '');

// Upload constraints
define('MAX_UPLOAD_SIZE', 8 * 1024 * 1024); // 8 MB
define('ALLOWED_MIME_TYPES', ['image/jpeg', 'image/png', 'image/webp']);

// ---------------------------------------------------------------
// Secrets (DB credentials + token-signing key) are NOT in this file —
// this repo is public. They come from environment variables, or from
// config/secrets.local.php, which exists only on the server / your own
// machine and is listed in .gitignore. See secrets.local.example.php.
// ---------------------------------------------------------------

// Token-signing key. Anyone who knows this can forge a login token for
// any user, including role=admin — so there is deliberately NO default.
$jwtSecret = getenv('JWT_SECRET') ?: (localSecrets()['jwt_secret'] ?? '');
if (strlen($jwtSecret) < 32) {
    http_response_code(500);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode([
        'success' => false,
        'message' => 'Server misconfigured: jwt_secret missing or too short in config/secrets.local.php',
    ]);
    exit;
}
define('JWT_SECRET', $jwtSecret);

// CORS - restrict to your frontend origin(s) in production
header('Access-Control-Allow-Origin: https://skinglow-eck.pages.dev');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(204);
    exit;
}
