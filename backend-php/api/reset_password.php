<?php
/**
 * POST /api/reset_password.php
 * Body (JSON): { "token": "...", "check": true } -> { success, valid }  (page checks the link on load;
 *   POST rather than GET so the token never lands in server access logs)
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

$invalidLink = 'ลิงก์นี้ไม่ถูกต้อง หมดอายุ หรือถูกใช้ไปแล้ว กรุณาขอลิงก์ใหม่อีกครั้ง';

/** The reset row for a token if it is unused, unexpired and its account active; null otherwise. */
function findValidReset(PDO $pdo, string $token): ?array
{
    if (!preg_match('/^[a-f0-9]{64}$/', $token)) {
        return null;
    }
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
        return null;
    }
    return $reset;
}

$input = json_decode(file_get_contents('php://input'), true) ?? [];
$token = trim((string) ($input['token'] ?? ''));
$password = (string) ($input['password'] ?? '');

if (!empty($input['check'])) {
    // lets reset_password.html refuse a used/expired link before showing the form
    $valid = findValidReset(getDbConnection(), $token) !== null;
    header('Cache-Control: no-store');
    echo json_encode(['success' => true, 'valid' => $valid, 'message' => $valid ? '' : $invalidLink],
        JSON_UNESCAPED_UNICODE);
    exit;
}

if (!preg_match('/^[a-f0-9]{64}$/', $token)) {
    fail(400, $invalidLink);
}
if (strlen($password) < 8) {   // same rule as register.php
    fail(422, 'รหัสผ่านต้องมีอย่างน้อย 8 ตัวอักษร');
}

$pdo = getDbConnection();
$reset = findValidReset($pdo, $token);
if (!$reset) {
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
