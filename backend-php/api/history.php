<?php
/**
 * GET /api/history.php
 * Requires: Authorization: Bearer <token>
 * Returns this user's own analysis history only — never another user's,
 * since $userId comes from the verified token, not a client-supplied value.
 */
declare(strict_types=1);

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../config/database.php';
require_once __DIR__ . '/../src/Auth.php';

header('Content-Type: application/json; charset=utf-8');

$userId = Auth::requireAuth();

$pdo = getDbConnection();
$stmt = $pdo->prepare(
    'SELECT id, original_image_path, processed_image_path, skin_score, detections_json, summary, created_at
     FROM analysis_history
     WHERE user_id = :user_id
     ORDER BY created_at DESC
     LIMIT 50'
);
$stmt->execute([':user_id' => $userId]);
$rows = $stmt->fetchAll();

// Build full URLs (DB stores relative paths) and decode detections for the
// frontend, same shape analyze.php already returns for a fresh scan.
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

echo json_encode(['success' => true, 'data' => $data], JSON_UNESCAPED_UNICODE);
