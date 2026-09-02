<?php
declare(strict_types=1);

/**
 * Minimal HMAC-signed token helper, shared by login.php (issues tokens) and
 * any endpoint that needs to know who's calling (verifies tokens).
 *
 * Token format: base64(json({sub, exp})) . hmac_sha256(that, JWT_SECRET)
 * Not a standards-compliant JWT — fine for a thesis project, swap for
 * firebase/php-jwt if this ever needs to interoperate with other systems.
 */
final class Auth
{
    public static function issueToken(int $userId): string
    {
        $payload = base64_encode(json_encode(['sub' => $userId, 'exp' => time() + 3600 * 24]));
        $signature = hash_hmac('sha256', $payload, JWT_SECRET);
        return $payload . '.' . $signature;
    }

    /** Returns the user_id encoded in a valid, unexpired token, or null otherwise. */
    public static function verifyToken(?string $token): ?int
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

        return (int) $data['sub'];
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

    /**
     * Call at the top of any endpoint that requires a logged-in user.
     * Returns the authenticated user_id, or halts the request with a 401
     * JSON response if the token is missing/invalid/expired.
     */
    public static function requireAuth(): int
    {
        $userId = self::verifyToken(self::bearerToken());

        if ($userId === null) {
            http_response_code(401);
            header('Content-Type: application/json; charset=utf-8');
            echo json_encode(['success' => false, 'message' => 'กรุณาเข้าสู่ระบบก่อนใช้งาน']);
            exit;
        }

        return $userId;
    }
}
