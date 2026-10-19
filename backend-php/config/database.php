<?php
/**
 * Database connection (PDO, MySQL)
 *
 * Credentials are NOT stored in this file (this file is committed to a
 * public GitHub repo). They're read, in order of priority, from:
 *   1. Environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS)
 *   2. config/secrets.local.php (the 'db' section) — a file that exists
 *      ONLY on the server / your own machine and is listed in .gitignore.
 *      Copy secrets.local.example.php to secrets.local.php and fill in real
 *      values. Upload it via FileZilla; never commit it.
 *      On localhost the 'db_local' section is used instead of 'db' (if present).
 *   3. Local XAMPP defaults (127.0.0.1 / root / no password) as a last resort.
 */

declare(strict_types=1);

function getDbConnection(): PDO
{
    // localSecrets() is defined in config.php (loaded first by every endpoint)
    $secrets = function_exists('localSecrets') ? localSecrets() : [];
    // On localhost (XAMPP) use the 'db_local' section if it exists, so the same
    // secrets.local.php works on both XAMPP and InfinityFree.
    $onLocalhost = function_exists('isLocalRequest') && isLocalRequest();
    $local = ($onLocalhost && !empty($secrets['db_local'])) ? $secrets['db_local'] : ($secrets['db'] ?? []);

    $host   = getenv('DB_HOST') ?: ($local['host'] ?? '127.0.0.1');
    $port   = getenv('DB_PORT') ?: ($local['port'] ?? '3306');
    $dbName = getenv('DB_NAME') ?: ($local['name'] ?? 'skinglow_db');
    $user   = getenv('DB_USER') ?: ($local['user'] ?? 'root');
    $pass   = getenv('DB_PASS') ?: ($local['pass'] ?? '');

    $dsn = "mysql:host={$host};port={$port};dbname={$dbName};charset=utf8mb4";

    $options = [
        PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_EMULATE_PREPARES   => false, // real prepared statements -> SQL injection safe
    ];

    try {
        return new PDO($dsn, $user, $pass, $options);
    } catch (PDOException $e) {
        http_response_code(500);
        header('Content-Type: application/json');
        echo json_encode(['success' => false, 'message' => 'Database connection failed.']);
        exit;
    }
}
