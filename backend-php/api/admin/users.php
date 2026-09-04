<?php
/**
 * GET    /api/admin/users.php            — list all users + their scan count
 * DELETE /api/admin/users.php?id=X        — delete a user (cascades to their
 *                                            analysis_history via FK)
 * Requires: Authorization: Bearer <token> for a user with role = 'admin'
 */
declare(strict_types=1);

require_once __DIR__ . '/../../config/config.php';
require_once __DIR__ . '/../../config/database.php';
require_once __DIR__ . '/../../src/Auth.php';

header('Content-Type: application/json; charset=utf-8');

$adminId = Auth::requireAdmin();
$pdo = getDbConnection();

if ($_SERVER['REQUEST_METHOD'] === 'GET') {
    $stmt = $pdo->query(
        'SELECT u.id, u.full_name, u.email, u.role, u.is_active, u.created_at,
                COUNT(a.id) AS scan_count
         FROM users u
         LEFT JOIN analysis_history a ON a.user_id = u.id
         GROUP BY u.id, u.full_name, u.email, u.role, u.is_active, u.created_at
         ORDER BY u.created_at DESC'
    );
    echo json_encode(['success' => true, 'data' => $stmt->fetchAll()], JSON_UNESCAPED_UNICODE);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'DELETE') {
    $targetId = filter_input(INPUT_GET, 'id', FILTER_VALIDATE_INT);
    if (!$targetId) {
        http_response_code(422);
        echo json_encode(['success' => false, 'message' => 'id is required and must be an integer.']);
        exit;
    }
    if ($targetId === $adminId) {
        http_response_code(400);
        echo json_encode(['success' => false, 'message' => 'ไม่สามารถลบบัญชีของตัวเองได้']);
        exit;
    }

    $stmt = $pdo->prepare('DELETE FROM users WHERE id = :id');
    $stmt->execute([':id' => $targetId]);

    if ($stmt->rowCount() === 0) {
        http_response_code(404);
        echo json_encode(['success' => false, 'message' => 'ไม่พบผู้ใช้นี้']);
        exit;
    }

    echo json_encode(['success' => true, 'message' => 'ลบผู้ใช้เรียบร้อยแล้ว']);
    exit;
}

http_response_code(405);
echo json_encode(['success' => false, 'message' => 'Method not allowed.']);
