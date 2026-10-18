<?php
/**
 * POST /api/delete_account.php   body: {"password": "..."}
 * Requires: Authorization: Bearer <token>.
 *
 * Permanently deletes the current user's account: every analysis image file
 * (uploads/original + uploads/processed), then the users row, which removes
 * analysis_history and password_resets rows through ON DELETE CASCADE.
 * The password is asked again so a borrowed, logged-in device cannot wipe an
 * account. Admin accounts are refused so the site always keeps an admin.
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
$password = (string) ($input['password'] ?? '');

$stmt = $pdo->prepare('SELECT password_hash, role FROM users WHERE id = :id');
$stmt->execute([':id' => $userId]);
$user = $stmt->fetch();
if (!$user) {
    jsonOut(404, ['success' => false, 'message' => 'ไม่พบบัญชีผู้ใช้']);
}
if (!password_verify($password, (string) $user['password_hash'])) {
    jsonOut(400, ['success' => false, 'message' => 'รหัสผ่านไม่ถูกต้อง']);
}
if (($user['role'] ?? '') === 'admin') {
    jsonOut(403, ['success' => false, 'message' => 'บัญชีผู้ดูแลระบบลบด้วยตนเองไม่ได้']);
}

// collect the image files before the rows disappear
$files = [];
$rows = $pdo->prepare('SELECT original_image_path, processed_image_path FROM analysis_history WHERE user_id = :id');
$rows->execute([':id' => $userId]);
foreach ($rows->fetchAll() as $r) {
    $files[] = UPLOAD_ORIGINAL_DIR . basename((string) ($r['original_image_path'] ?? ''));
    $files[] = UPLOAD_PROCESSED_DIR . basename((string) ($r['processed_image_path'] ?? ''));
}

$pdo->beginTransaction();
$pdo->prepare('DELETE FROM analysis_history WHERE user_id = :id')->execute([':id' => $userId]);
$pdo->prepare('DELETE FROM users WHERE id = :id')->execute([':id' => $userId]);
$pdo->commit();

foreach ($files as $path) {
    if (is_file($path)) {
        @unlink($path);
    }
}

jsonOut(200, ['success' => true, 'message' => 'ลบบัญชีและข้อมูลทั้งหมดเรียบร้อยแล้ว']);
