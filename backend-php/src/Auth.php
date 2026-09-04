<?php
declare(strict_types=1);

/**
 * Minimal HMAC-signed token helper, shared by login.php (issues tokens) and
 * any endpoint that needs to know who's calling (verifies tokens).
 *
 * Token format: base64(json({sub, role, exp})) . hmac_sha256(that, JWT_SECRET)
 * Not a standards-compliant JWT — fine for a thesis project, swap for
 * firebase/php-jwt if this ever needs to interoperate with other systems.
 */
final class Auth
{
    public static function issueToken(int $userId, string $role = 'user'): string
    {
        $payload = base64_encode(json_encode([
            'sub'  => $userId,
            'role' => $role,
            'exp'  => time() + 3600 * 24,
        ]));
        $signature = hash_hmac('sha256', $payload, JWT_SECRET);
        return $payload . '.' . $signature;
    }

    /**
     * Returns ['sub' => user_id, 'role' => 'user'|'admin'] for a valid,
     * unexpired token, or null otherwise. Prefer this over verifyToken()
     * when you need the role, not just the id.
     */
    public static function verifyTokenFull(?string $token): ?array
    {
        if (!$token || !str_contains($token, '.')) {
            return null;
        }

        [$payload, $signature] = explode('.', $token, 2);
        $expected = hash_hmac('sha256', $payload, JWT_SECRET);

        // hash_equals for timing-safe comparison
        if (!hash_equals($expected, $signature)) {
            return null;
        }

        $data = json_decode(base64_decode($payload), true);
        if (!is_array($data) || !isset($data['sub'], $data['exp'])) {
            return null;
        }
        if (time() > (int) $data['exp']) {
            return null; // expired
        }

        return [
            'sub'  => (int) $data['sub'],
            'role' => $data['role'] ?? 'user', // tokens issued before roles existed default to 'user'
        ];
    }

    /** Returns just the user_id encoded in a valid, unexpired token, or null otherwise. */
    public static function verifyToken(?string $token): ?int
    {
        $data = self::verifyTokenFull($token);
        return $data['sub'] ?? null;
    }

    private static function bearerToken(): ?string
    {
        // Some Apache configs strip the Authorization header from $_SERVER
        // and only expose it as REDIRECT_HTTP_AUTHORIZATION (seen with
        // certain mod_php/mod_fcgid setups) — check both.
        $header = $_SERVER['HTTP_AUTHORIZATION']
            ?? $_SERVER['REDIRECT_HTTP_AUTHORIZATION']
            ?? '';

        if (!$header && function_exists('getallheaders')) {
            foreach (getallheaders() as $name => $value) {
                if (strcasecmp($name, 'Authorization') === 0) {
                    $header = $value;
                    break;
                }
            }
        }

        if (preg_match('/Bearer\s+(.+)/i', $header, $matches)) {
            return trim($matches[1]);
        }
        return null;
    }

    private static function denyJson(int $status, string $message): never
    {
        http_response_code($status);
        header('Content-Type: application/json; charset=utf-8');
        echo json_encode(['success' => false, 'message' => $message]);
        exit;
    }

    /**
     * Call at the top of any endpoint that requires a logged-in user.
     * Returns the authenticated user_id, or halts the request with a 401
     * JSON response if the token is missing/invalid/expired.
     */
    public static function requireAuth(): int
    {
        $data = self::verifyTokenFull(self::bearerToken());

        if ($data === null) {
            self::denyJson(401, 'กรุณาเข้าสู่ระบบก่อนใช้งาน');
        }

        return $data['sub'];
    }

    /**
     * Call at the top of any endpoint that requires an admin. Returns the
     * admin's own user_id, or halts with 401 (not logged in) / 403 (logged
     * in but not an admin) JSON response.
     */
    public static function requireAdmin(): int
    {
        $data = self::verifyTokenFull(self::bearerToken());

        if ($data === null) {
            self::denyJson(401, 'กรุณาเข้าสู่ระบบก่อนใช้งาน');
        }
        if ($data['role'] !== 'admin') {
            self::denyJson(403, 'ต้องใช้สิทธิ์ผู้ดูแลระบบสำหรับส่วนนี้');
        }

        return $data['sub'];
    }
}
