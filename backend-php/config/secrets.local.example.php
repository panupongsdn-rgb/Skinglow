<?php
/**
 * TEMPLATE — safe to commit (contains no real secrets).
 *
 * 1. Copy this file to secrets.local.php (same folder).
 * 2. Fill in real values.
 * 3. Upload secrets.local.php to the server via FileZilla.
 * 4. NEVER commit secrets.local.php — it's listed in .gitignore.
 */
return [
    // Long random string (at least 32 characters). Used to sign login tokens —
    // anyone who knows it can forge an admin login. Changing it logs everyone out.
    // Generate one with:  php -r "echo bin2hex(random_bytes(32));"
    'jwt_secret' => 'PUT_A_LONG_RANDOM_STRING_HERE',

    // InfinityFree: Control Panel -> MySQL Databases
    // Local XAMPP: host 127.0.0.1, user root, empty password, your local port
    'db' => [
        'host' => 'sqlXXX.infinityfree.com',
        'port' => '3306',
        'name' => 'if0_XXXXXXXX_skinglow_db',
        'user' => 'if0_XXXXXXXX',
        'pass' => 'PUT_YOUR_NEW_PASSWORD_HERE',
    ],

    // Used instead of 'db' when the site is opened on localhost (XAMPP).
    // Check the port in phpMyAdmin ("Server: localhost:3307").
    'db_local' => [
        'host' => '127.0.0.1',
        'port' => '3306',
        'name' => 'skinglow_db',
        'user' => 'root',
        'pass' => '',
    ],

    // Optional, localhost only: your own AI service instead of Render
    // 'ai_service_url' => 'http://127.0.0.1:8000/analyze',

    // Gmail SMTP for "forgot password" (use a Google App Password)
    'smtp' => [
        'host'      => 'smtp.gmail.com',
        'port'      => 587,
        'secure'    => 'tls',
        'user'      => 'you@gmail.com',
        'pass'      => 'GOOGLE_APP_PASSWORD',
        'from'      => 'you@gmail.com',
        'from_name' => 'SkinGlow',
    ],
];
