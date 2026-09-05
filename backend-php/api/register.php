<?php
/**
 * POST /api/register.php
 * Body (JSON): { "full_name": "...", "email": "...", "password": "..." }
 */
declare(strict_types=1);

require_once __DIR__ . 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/config/config.php';
require_once __DIR__ . 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/config/database.php';
require_once __DIR__ . 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/src/Auth.php';

header('Content-Type: application/json; charset=utf-8');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['success' => false, 'message' => 'Method not allowed.']);
    exit;
}

$input = json_decode(file_get_contents('php://input'), true) ?? [];

$fullName = trim((string)($input['full_name'] ?? ''));
$email    = trim((string)($input['email'] ?? ''));
$password = (string)($input['password'] ?? '');

if ($fullName === '' || !filter_var($email, FILTER_VALIDATE_EMAIL) || strlen($password) < 8) {
    http_response_code(422);
    echo json_encode(['success' => false, 'message' => 'Invalid full_name, email, or password (min 8 chars).']);
    exit;
}

$pdo = getDbConnection();

$check = $pdo->prepare('SELECT id FROM users WHERE email = :email');
$check->execute([':email' => $email]);
if ($check->fetch()) {
    http_response_code(409);
    echo json_encode(['success' => false, 'message' => 'Email already registered.']);
    exit;
}

$hash = password_hash($password, PASSWORD_BCRYPT);

$stmt = $pdo->prepare(
    'INSERT INTO users (full_name, email, password_hash) VALUES (:full_name, :email, :password_hash)'
);
$stmt->execute([
    ':full_name'     => $fullName,
    ':email'         => $email,
    ':password_hash' => $hash,
]);

$newUserId = (int) $pdo->lastInsertId();
$token = Auth::issueToken($newUserId);

echo json_encode([
    'success' => true,
    'data' => [
        'token' => $token,
        'user' => ['id' => $newUserId, 'full_name' => $fullName, 'email' => $email],
    ],
]);
