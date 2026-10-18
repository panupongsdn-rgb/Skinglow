<?php
/**
 * POST /api/change_password.php   body: {"current_password": "...", "new_password": "..."}
 * Requires: Authorization: Bearer <token>. The current password must be
 * correct; the new one needs at least 8 characters (same rule as register).
 * Any unused password-reset links of this user are cancelled.
 */
declare(strict_types=1);

header('Access-Control-Allow-Origin: https://skinglow-eck.pages.dev');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(204);
    exit;
}

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../config/database.php';
require_once __DIR__ . '/../src/Auth.php';

header('Content-Type: application/json; charset=utf-8');

function jsonOut(int $code, array $body): void
{
    http_response_code($code);
    echo json_encode($body, JSON_UNESCAPED_UNICODE);
    exit;
}

$userId = Auth::requireAuth();
$pdo = getDbConnection();

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    jsonOut(405, ['success' => false, 'message' => 'Method not allowed.']);
}

$input = json_decode((string) file_get_contents('php://input'), true) ?: [];
$current = (string) ($input['current_password'] ?? '');
$new = (string) ($input['new_password'] ?? '');

if (strlen($new) < 8) {
    jsonOut(422, ['success' => false, 'message' => 'รหัสผ่านใหม่ต้องมีอย่างน้อย 8 ตัวอักษร']);
}

$stmt = $pdo->prepare('SELECT password_hash FROM users WHERE id = :id AND is_active = 1');
$stmt->execute([':id' => $userId]);
$hash = $stmt->fetchColumn();
if (!$hash || !password_verify($current, (string) $hash)) {
    jsonOut(400, ['success' => false, 'message' => 'รหัสผ่านปัจจุบันไม่ถูกต้อง']);
}
if (password_verify($new, (string) $hash)) {
    jsonOut(422, ['success' => false, 'message' => 'รหัสผ่านใหม่ต้องไม่ซ้ำกับรหัสผ่านเดิม']);
}

$pdo->prepare('UPDATE users SET password_hash = :hash WHERE id = :id')
    ->execute([':hash' => password_hash($new, PASSWORD_BCRYPT), ':id' => $userId]);

try {
    $pdo->prepare('UPDATE password_resets SET used_at = UTC_TIMESTAMP() WHERE user_id = :id AND used_at IS NULL')
        ->execute([':id' => $userId]);
} catch (PDOException $e) {
    // password_resets table missing on an old database: nothing to cancel
}

jsonOut(200, ['success' => true, 'message' => 'เปลี่ยนรหัสผ่านเรียบร้อยแล้ว']);
