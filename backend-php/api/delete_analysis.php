<?php
/**
 * POST /api/delete_analysis.php   body: {"id": <analysis id>}
 * Requires: Authorization: Bearer <token>
 *
 * Deletes one of the CURRENT user's own analyses: the analysis_history row
 * and its two image files (uploads/original and uploads/processed).
 * The row is matched on both id AND user_id from the verified token, so a
 * user can never delete another user's analysis by guessing an id.
 * POST is used instead of DELETE because some free hosts block DELETE.
 */

declare(strict_types=1);

header('Access-Control-Allow-Origin: https://skinglow-eck.pages.dev');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(204);
    exit;
}

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../config/database.php';
require_once __DIR__ . '/../src/Auth.php';

header('Content-Type: application/json; charset=utf-8');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['success' => false, 'message' => 'Method not allowed.']);
    exit;
}

$userId = Auth::requireAuth();

$input = json_decode((string) file_get_contents('php://input'), true);
$analysisId = filter_var($input['id'] ?? null, FILTER_VALIDATE_INT);
if (!$analysisId) {
    http_response_code(422);
    echo json_encode(['success' => false, 'message' => 'id is required and must be an integer.']);
    exit;
}

$pdo = getDbConnection();

$stmt = $pdo->prepare(
    'SELECT original_image_path, processed_image_path
     FROM analysis_history
     WHERE id = :id AND user_id = :user_id'
);
$stmt->execute([':id' => $analysisId, ':user_id' => $userId]);
$row = $stmt->fetch();

if (!$row) {
    http_response_code(404);
    echo json_encode(['success' => false, 'message' => 'ไม่พบผลวิเคราะห์นี้ในประวัติของคุณ'], JSON_UNESCAPED_UNICODE);
    exit;
}

$del = $pdo->prepare('DELETE FROM analysis_history WHERE id = :id AND user_id = :user_id');
$del->execute([':id' => $analysisId, ':user_id' => $userId]);

// Remove the image files. basename() keeps the delete inside the upload
// folders even if a stored path were ever tampered with.
$files = [
    UPLOAD_ORIGINAL_DIR . basename((string) ($row['original_image_path'] ?? '')),
    UPLOAD_PROCESSED_DIR . basename((string) ($row['processed_image_path'] ?? '')),
];
foreach ($files as $path) {
    if (is_file($path)) {
        @unlink($path);
    }
}

echo json_encode(['success' => true, 'message' => 'ลบผลวิเคราะห์เรียบร้อยแล้ว'], JSON_UNESCAPED_UNICODE);
