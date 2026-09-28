<?php
/**
 * POST /api/reset_password.php
 * Body (JSON): { "token": "<from the emailed link>", "password": "<new password>" }
 *
 * Sets a new password if the token is valid, unused and unexpired, then
 * invalidates every other outstanding reset link for that account.
 */

declare(strict_types=1);

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../config/database.php';

header('Content-Type: application/json; charset=utf-8');

function fail(int $status, string $message): void
{
    http_response_code($status);
    echo json_encode(['success' => false, 'message' => $message], JSON_UNESCAPED_UNICODE);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    fail(405, 'Method not allowed.');
}

$input = json_decode(file_get_contents('php://input'), true) ?? [];
$token = trim((string) ($input['token'] ?? ''));
$password = (string) ($input['password'] ?? '');

$invalidLink = 'ลิงก์นี้ไม่ถูกต้อง หมดอายุ หรือถูกใช้ไปแล้ว กรุณาขอลิงก์ใหม่อีกครั้ง';

if (!preg_match('/^[a-f0-9]{64}$/', $token)) {
    fail(400, $invalidLink);
}
if (strlen($password) < 8) {   // same rule as register.php
    fail(422, 'รหัสผ่านต้องมีอย่างน้อย 8 ตัวอักษร');
}

$pdo = getDbConnection();

$stmt = $pdo->prepare(
    'SELECT pr.id, pr.user_id, pr.expires_at
     FROM password_resets pr
     JOIN users u ON u.id = pr.user_id AND u.is_active = 1
     WHERE pr.token_hash = :hash AND pr.used_at IS NULL'
);
$stmt->execute([':hash' => hash('sha256', $token)]);
$reset = $stmt->fetch();

// expires_at is a UTC string written by PHP — compare in PHP, not SQL, so
// the MySQL server's timezone can't shift the expiry
if (!$reset || strtotime($reset['expires_at'] . ' UTC') < time()) {
    fail(400, $invalidLink);
}

$nowUtc = gmdate('Y-m-d H:i:s');

$pdo->beginTransaction();
try {
    $pdo->prepare('UPDATE users SET password_hash = :hash WHERE id = :uid')
        ->execute([':hash' => password_hash($password, PASSWORD_BCRYPT), ':uid' => $reset['user_id']]);

    // this link and every other outstanding link for the account are now spent
    $pdo->prepare('UPDATE password_resets SET used_at = :now WHERE user_id = :uid AND used_at IS NULL')
        ->execute([':now' => $nowUtc, ':uid' => $reset['user_id']]);

    $pdo->commit();
} catch (Throwable $e) {
    $pdo->rollBack();
    error_log('[reset_password] ' . $e->getMessage());
    fail(500, 'เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง');
}

echo json_encode([
    'success' => true,
    'message' => 'ตั้งรหัสผ่านใหม่เรียบร้อยแล้ว กรุณาเข้าสู่ระบบด้วยรหัสผ่านใหม่',
], JSON_UNESCAPED_UNICODE);
