<?php
/**
 * GET  /api/profile.php   -> the current user's profile
 * POST /api/profile.php   body: {"full_name": "...", "skin_type": "oily"|...|""}
 * Requires: Authorization: Bearer <token>. Only the token's own user is read
 * or changed; email and role cannot be changed here.
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

const SKIN_TYPES = ['normal', 'oily', 'dry', 'combination', 'sensitive'];

function loadProfile(PDO $pdo, int $userId): ?array
{
    $stmt = $pdo->prepare('SELECT id, full_name, email, role, skin_type, created_at FROM users WHERE id = :id AND is_active = 1');
    $stmt->execute([':id' => $userId]);
    $row = $stmt->fetch();
    if (!$row) {
        return null;
    }
    $row['id'] = (int) $row['id'];
    return $row;
}

if ($_SERVER['REQUEST_METHOD'] === 'GET') {
    $profile = loadProfile($pdo, $userId);
    if (!$profile) {
        jsonOut(404, ['success' => false, 'message' => 'ไม่พบบัญชีผู้ใช้']);
    }
    jsonOut(200, ['success' => true, 'data' => $profile]);
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    jsonOut(405, ['success' => false, 'message' => 'Method not allowed.']);
}

$input = json_decode((string) file_get_contents('php://input'), true) ?: [];
$fullName = trim((string) ($input['full_name'] ?? ''));
$skinType = trim((string) ($input['skin_type'] ?? ''));

if ($fullName === '' || mb_strlen($fullName) > 150) {
    jsonOut(422, ['success' => false, 'message' => 'กรุณากรอกชื่อ-นามสกุล (ไม่เกิน 150 ตัวอักษร)']);
}
if ($skinType !== '' && !in_array($skinType, SKIN_TYPES, true)) {
    jsonOut(422, ['success' => false, 'message' => 'ประเภทผิวไม่ถูกต้อง']);
}

$stmt = $pdo->prepare('UPDATE users SET full_name = :name, skin_type = :skin WHERE id = :id');
$stmt->execute([':name' => $fullName, ':skin' => $skinType === '' ? null : $skinType, ':id' => $userId]);

jsonOut(200, ['success' => true, 'message' => 'บันทึกข้อมูลเรียบร้อยแล้ว', 'data' => loadProfile($pdo, $userId)]);
