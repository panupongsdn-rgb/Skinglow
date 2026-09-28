<?php
declare(strict_types=1);

require_once __DIR__ . '/../lib/PHPMailer/Exception.php';
require_once __DIR__ . '/../lib/PHPMailer/PHPMailer.php';
require_once __DIR__ . '/../lib/PHPMailer/SMTP.php';

use PHPMailer\PHPMailer\PHPMailer;
use PHPMailer\PHPMailer\Exception as MailException;

/**
 * Sends email through an external SMTP server (e.g. Gmail).
 *
 * WHY NOT PHP mail(): InfinityFree rejects almost all mail sent via PHP's
 * built-in mail() function to prevent spam. Sending through an external
 * SMTP server (Gmail + an App Password) is the approach InfinityFree's own
 * documentation points to.
 *
 * SETTINGS come from config/secrets.local.php ('smtp' section) or env vars
 * (SMTP_HOST, SMTP_PORT, SMTP_SECURE, SMTP_USER, SMTP_PASS, SMTP_FROM,
 * SMTP_FROM_NAME). Never commit real SMTP credentials to the repo.
 */
final class Mailer
{
    /** Reason for the most recent failed send (no secrets) — shown by tools/mail_test.php */
    public static string $lastError = '';

    private static function settings(): array
    {
        $local = function_exists('localSecrets') ? (localSecrets()['smtp'] ?? []) : [];
        $get = fn(string $env, string $key, $default = '') => getenv($env) ?: ($local[$key] ?? $default);

        return [
            'host'      => $get('SMTP_HOST', 'host', 'smtp.gmail.com'),
            'port'      => (int) $get('SMTP_PORT', 'port', 587),
            'secure'    => $get('SMTP_SECURE', 'secure', 'tls'),   // 'tls' (587), 'ssl' (465) or '' (none, local testing only)
            'user'      => $get('SMTP_USER', 'user'),
            'pass'      => $get('SMTP_PASS', 'pass'),
            'from'      => $get('SMTP_FROM', 'from'),
            'from_name' => $get('SMTP_FROM_NAME', 'from_name', 'Skinglow'),
        ];
    }

    public static function isConfigured(): bool
    {
        $s = self::settings();
        return $s['host'] !== '' && $s['from'] !== '';
    }

    /** Returns true if the SMTP server accepted the message. Never throws. */
    public static function send(string $toEmail, string $toName, string $subject, string $html, string $text): bool
    {
        $s = self::settings();
        if (!self::isConfigured()) {
            self::$lastError = 'SMTP not configured';
            error_log('[Mailer] SMTP not configured — add an smtp section to config/secrets.local.php');
            return false;
        }

        $mail = new PHPMailer(true);
        try {
            $mail->isSMTP();
            $mail->Host = $s['host'];
            $mail->Port = $s['port'];
            $mail->SMTPAuth = $s['user'] !== '';
            if ($mail->SMTPAuth) {
                $mail->Username = $s['user'];
                $mail->Password = $s['pass'];
            }
            if ($s['secure'] === 'ssl') {
                $mail->SMTPSecure = PHPMailer::ENCRYPTION_SMTPS;
            } elseif ($s['secure'] === 'tls') {
                $mail->SMTPSecure = PHPMailer::ENCRYPTION_STARTTLS;
            } else {
                $mail->SMTPSecure = '';
                $mail->SMTPAutoTLS = false;
            }
            $mail->Timeout = 15;
            $mail->CharSet = 'UTF-8';

            $mail->setFrom($s['from'], $s['from_name']);
            $mail->addAddress($toEmail, $toName);
            $mail->Subject = $subject;
            $mail->isHTML(true);
            $mail->Body = $html;
            $mail->AltBody = $text;

            return $mail->send();
        } catch (MailException $e) {
            self::$lastError = $mail->ErrorInfo;
            error_log('[Mailer] send failed: ' . $mail->ErrorInfo);
            return false;
        }
    }

    public static function sendPasswordReset(string $toEmail, string $toName, string $resetLink, int $validMinutes): bool
    {
        $safeName = htmlspecialchars($toName, ENT_QUOTES, 'UTF-8');
        $safeLink = htmlspecialchars($resetLink, ENT_QUOTES, 'UTF-8');

        $html = <<<HTML
<div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;color:#241A33">
  <h2 style="color:#3B2A54">Skinglow — ตั้งรหัสผ่านใหม่</h2>
  <p>สวัสดีคุณ {$safeName}</p>
  <p>เราได้รับคำขอตั้งรหัสผ่านใหม่สำหรับบัญชีของคุณ กดปุ่มด้านล่างเพื่อตั้งรหัสผ่านใหม่
     (ลิงก์ใช้ได้ {$validMinutes} นาที และใช้ได้เพียงครั้งเดียว)</p>
  <p style="text-align:center;margin:28px 0">
    <a href="{$safeLink}" style="background:#B48EE0;color:#fff;padding:12px 24px;border-radius:999px;text-decoration:none;font-weight:bold">ตั้งรหัสผ่านใหม่</a>
  </p>
  <p style="font-size:13px;color:#7A6C8E">ถ้าปุ่มกดไม่ได้ ให้คัดลอกลิงก์นี้ไปเปิดในเบราว์เซอร์:<br>{$safeLink}</p>
  <p style="font-size:13px;color:#7A6C8E">ถ้าคุณไม่ได้ขอตั้งรหัสผ่านใหม่ ไม่ต้องทำอะไร — รหัสผ่านเดิมยังใช้ได้ตามปกติ</p>
</div>
HTML;

        $text = "Skinglow — ตั้งรหัสผ่านใหม่\n\nสวัสดีคุณ {$toName}\n\n"
              . "เปิดลิงก์นี้เพื่อตั้งรหัสผ่านใหม่ (ใช้ได้ {$validMinutes} นาที ครั้งเดียว):\n{$resetLink}\n\n"
              . "ถ้าคุณไม่ได้ขอตั้งรหัสผ่านใหม่ ไม่ต้องทำอะไร";

        return self::send($toEmail, $toName, 'Skinglow — ตั้งรหัสผ่านใหม่', $html, $text);
    }
}
