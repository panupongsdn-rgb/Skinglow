<?php
/**
 * POST /api/forgot_password.php
 * Body (JSON): { "email": "..." }
 *
 * Emails a one-time password-reset link. The response is IDENTICAL whether
 * or not the email belongs to an account, so this endpoint can't be used to
 * discover which emails are registered.
 *
 * Security properties:
 *  - token: 32 random bytes; only its SHA-256 hash is stored in the DB
 *  - valid for RESET_TOKEN_MINUTES, single use
 *  - requesting a new link invalidates any older unused links
 *  - max RESET_MAX_REQUESTS per account per RESET_WINDOW_MINUTES
 */

declare(strict_types=1);

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../config/database.php';
require_once __DIR__ . '/../src/Mailer.php';

header('Content-Type: application/json; charset=utf-8');

const RESET_TOKEN_MINUTES = 30;
const RESET_MAX_REQUESTS  = 3;
const RESET_WINDOW_MINUTES = 15;

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['success' => false, 'message' => 'Method not allowed.']);
    exit;
}

$input = json_decode(file_get_contents('php://input'), true) ?? [];
$email = trim((string) ($input['email'] ?? ''));

if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
    http_response_code(422);
    echo json_encode(['success' => false, 'message' => 'กรุณากรอกอีเมลให้ถูกต้อง'], JSON_UNESCAPED_UNICODE);
    exit;
}

// Same answer for every valid-looking email — never reveal whether it's registered.
$genericResponse = [
    'success' => true,
    'message' => 'หากอีเมลนี้มีบัญชีอยู่ในระบบ เราได้ส่งลิงก์สำหรับตั้งรหัสผ่านใหม่ไปแล้ว กรุณาตรวจสอบกล่องจดหมาย (รวมถึงโฟลเดอร์สแปม)',
];

$pdo = getDbConnection();

$stmt = $pdo->prepare('SELECT id, full_name, email FROM users WHERE email = :email AND is_active = 1');
$stmt->execute([':email' => $email]);
$user = $stmt->fetch();

if ($user) {
    $now = time();

    // rate limit per account (timestamps are UTC strings written by PHP, so
    // comparisons don't depend on the MySQL server's timezone setting)
    $countStmt = $pdo->prepare(
        'SELECT COUNT(*) FROM password_resets WHERE user_id = :uid AND created_at > :since'
    );
    $countStmt->execute([
        ':uid'   => $user['id'],
        ':since' => gmdate('Y-m-d H:i:s', $now - RESET_WINDOW_MINUTES * 60),
    ]);
    $recentRequests = (int) $countStmt->fetchColumn();

    if ($recentRequests < RESET_MAX_REQUESTS) {
        $token = bin2hex(random_bytes(32));
        $nowUtc = gmdate('Y-m-d H:i:s', $now);

        // any older, still-unused links for this account stop working
        $pdo->prepare('UPDATE password_resets SET used_at = :now WHERE user_id = :uid AND used_at IS NULL')
            ->execute([':now' => $nowUtc, ':uid' => $user['id']]);

        $pdo->prepare(
            'INSERT INTO password_resets (user_id, token_hash, expires_at, created_at)
             VALUES (:uid, :hash, :expires, :created)'
        )->execute([
            ':uid'     => $user['id'],
            ':hash'    => hash('sha256', $token),
            ':expires' => gmdate('Y-m-d H:i:s', $now + RESET_TOKEN_MINUTES * 60),
            ':created' => $nowUtc,
        ]);

        // https://.../Skinglow/backend-php  ->  https://.../Skinglow/frontend
        $frontendBase = defined('FRONTEND_BASE_URL')
            ? rtrim(FRONTEND_BASE_URL, '/')
            : preg_replace('#/backend-php/?$#', '/frontend', rtrim(APP_BASE_URL, '/'));
        $resetLink = $frontendBase . '/reset_password.html?token=' . $token;

        if (!Mailer::sendPasswordReset($user['email'], $user['full_name'], $resetLink, RESET_TOKEN_MINUTES)) {
            error_log('[forgot_password] could not send reset email to user_id=' . $user['id']);
        }
    } else {
        error_log('[forgot_password] rate limit hit for user_id=' . $user['id']);
    }
}

echo json_encode($genericResponse, JSON_UNESCAPED_UNICODE);
