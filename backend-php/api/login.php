<?php
/**
 * POST /api/login.php
 * Body (JSON): { "email": "...", "password": "..." }
 * Returns a signed token the frontend stores and sends as:
 *   Authorization: Bearer <token>
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
$email    = trim((string)($input['email'] ?? ''));
$password = (string)($input['password'] ?? '');

$pdo = getDbConnection();
$stmt = $pdo->prepare('SELECT id, full_name, email, password_hash FROM users WHERE email = :email AND is_active = 1');
$stmt->execute([':email' => $email]);
$user = $stmt->fetch();

if (!$user || !password_verify($password, $user['password_hash'])) {
    http_response_code(401);
    echo json_encode(['success' => false, 'message' => 'Invalid email or password.']);
    exit;
}

$token = Auth::issueToken((int) $user['id']);

echo json_encode([
    'success' => true,
    'data' => [
        'token' => $token,
        'user'  => ['id' => $user['id'], 'full_name' => $user['full_name'], 'email' => $user['email']],
    ],
]);
