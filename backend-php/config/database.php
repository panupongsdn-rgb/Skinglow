<?php
/**
 * Database connection (PDO, MySQL)
 * Use environment variables in production instead of hard-coded values.
 */

declare(strict_types=1);

function getDbConnection(): PDO
{
    $host = getenv('DB_HOST') ?: 'sql307.infinityfree.com';
    $port = getenv('DB_PORT') ?: '3306';
    $dbName = getenv('DB_NAME') ?: 'if0_42826508_skinglow_db';
    $user = getenv('DB_USER') ?: 'if0_42826508';
    $pass = getenv('DB_PASS') ?: 'lxQzFOjR1G7Y';

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
