<?php
/**
 * GET /api/admin/user_analyses.php?user_id=X
 * Requires: Authorization: Bearer <token> for a user with role = 'admin'
 *
 * Same response shape as the regular history.php, but lets an admin view
 * ANY user's history (by user_id query param) rather than only their own.
 */
declare(strict_types=1);

require_once __DIR__ . '/../../config/config.php';
require_once __DIR__ . '/../../config/database.php';
require_once __DIR__ . '/../../src/Auth.php';

header('Content-Type: application/json; charset=utf-8');

Auth::requireAdmin();

$targetUserId = filter_input(INPUT_GET, 'user_id', FILTER_VALIDATE_INT);
if (!$targetUserId) {
    http_response_code(422);
    echo json_encode(['success' => false, 'message' => 'user_id is required and must be an integer.']);
    exit;
}

$pdo = getDbConnection();

$userStmt = $pdo->prepare('SELECT id, full_name, email FROM users WHERE id = :id');
$userStmt->execute([':id' => $targetUserId]);
$user = $userStmt->fetch();
if (!$user) {
    http_response_code(404);
    echo json_encode(['success' => false, 'message' => 'ไม่พบผู้ใช้นี้']);
    exit;
}

$stmt = $pdo->prepare(
    'SELECT id, original_image_path, processed_image_path, skin_score, detections_json, summary, created_at
     FROM analysis_history
     WHERE user_id = :user_id
     ORDER BY created_at DESC
     LIMIT 50'
);
$stmt->execute([':user_id' => $targetUserId]);
$rows = $stmt->fetchAll();

$data = array_map(function ($row) {
    return [
        'id'              => (int) $row['id'],
        'skin_score'      => $row['skin_score'] !== null ? (float) $row['skin_score'] : null,
        'summary'         => $row['summary'],
        'detections'      => json_decode($row['detections_json'] ?? '[]', true) ?: [],
        'original_image'  => APP_BASE_URL . '/' . ltrim($row['original_image_path'], '/'),
        'processed_image' => APP_BASE_URL . '/' . ltrim($row['processed_image_path'], '/'),
        'created_at'      => $row['created_at'],
    ];
}, $rows);

echo json_encode(['success' => true, 'user' => $user, 'data' => $data], JSON_UNESCAPED_UNICODE);
