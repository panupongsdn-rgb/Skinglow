<?php
/**
 * POST /api/analyze.php
 * Requires: Authorization: Bearer <token> (from login.php/register.php)
 * Accepts a multipart/form-data image upload, forwards it to the Python
 * AI service, draws bounding boxes on the result, stores everything in
 * MySQL, and returns a JSON response to the frontend.
 *
 * Expected form fields:
 *   - image     (file, required)
 */

declare(strict_types=1);

require_once __DIR__ . 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/config/config.php';
require_once __DIR__ . 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/config/database.php';
require_once __DIR__ . 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/src/BoundingBoxDrawer.php';
require_once __DIR__ . 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/src/Auth.php';

header('Content-Type: application/json; charset=utf-8');

function respond(int $status, array $payload): void
{
    http_response_code($status);
    echo json_encode($payload, JSON_UNESCAPED_UNICODE);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    respond(405, ['success' => false, 'message' => 'Method not allowed.']);
}

// ---------------------------------------------------------------
// 1. Validate input
// ---------------------------------------------------------------
$userId = Auth::requireAuth(); // halts with 401 JSON if not logged in

if (!isset($_FILES['image']) || $_FILES['image']['error'] !== UPLOAD_ERR_OK) {
    respond(422, ['success' => false, 'message' => 'Image upload failed or missing.']);
}

$file = $_FILES['image'];

if ($file['size'] > MAX_UPLOAD_SIZE) {
    respond(413, ['success' => false, 'message' => 'Image exceeds maximum allowed size (8MB).']);
}

// Verify real MIME type (never trust the client-supplied one)
$finfo = finfo_open(FILEINFO_MIME_TYPE);
$actualMime = finfo_file($finfo, $file['tmp_name']);
finfo_close($finfo);

if (!in_array($actualMime, ALLOWED_MIME_TYPES, true)) {
    respond(415, ['success' => false, 'message' => 'Unsupported image type. Allowed: JPEG, PNG, WEBP.']);
}

// ---------------------------------------------------------------
// 2. Save original upload with a safe, unique filename
// ---------------------------------------------------------------
$extMap = ['image/jpeg' => 'jpg', 'image/png' => 'png', 'image/webp' => 'webp'];
$ext = $extMap[$actualMime];
$uniqueName = bin2hex(random_bytes(16)) . '.' . $ext;

if (!is_dir(UPLOAD_ORIGINAL_DIR)) {
    mkdir(UPLOAD_ORIGINAL_DIR, 0755, true);
}
if (!is_dir(UPLOAD_PROCESSED_DIR)) {
    mkdir(UPLOAD_PROCESSED_DIR, 0755, true);
}

$originalPath = UPLOAD_ORIGINAL_DIR . $uniqueName;

if (!move_uploaded_file($file['tmp_name'], $originalPath)) {
    respond(500, ['success' => false, 'message' => 'Failed to store uploaded image.']);
}

// ---------------------------------------------------------------
// 3. Forward image to the Python AI service via cURL
// ---------------------------------------------------------------
$ch = curl_init(AI_SERVICE_URL);
curl_setopt_array($ch, [
    CURLOPT_POST           => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_TIMEOUT        => 30,
    CURLOPT_POSTFIELDS     => [
        'file' => new CURLFile($originalPath, $actualMime, $uniqueName),
    ],
]);

$aiResponseRaw = curl_exec($ch);
$curlError = curl_error($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($aiResponseRaw === false || $httpCode !== 200) {
    respond(502, [
        'success' => false,
        'message' => 'AI service request failed.',
        'detail'  => $curlError ?: "AI service returned HTTP {$httpCode}",
    ]);
}

$aiResult = json_decode($aiResponseRaw, true);
if (!is_array($aiResult) || !isset($aiResult['detections'])) {
    respond(502, ['success' => false, 'message' => 'Malformed response from AI service.']);
}

$detections = $aiResult['detections'];   // e.g. [{box, label, confidence}, ...]
$skinScore  = $aiResult['skin_score'] ?? null;
$skinStatus = $aiResult['skin_status'] ?? 'issues_found'; // 'clear' | 'issues_found' | 'no_face_detected'
$faceDetected = $aiResult['face_detected'] ?? true;

if ($skinStatus === 'no_face_detected') {
    respond(422, [
        'success' => false,
        'message' => 'ไม่พบใบหน้าในภาพ กรุณาถ่ายภาพใบหน้าให้ชัดเจนแล้วลองใหม่อีกครั้ง',
    ]);
}

// ---------------------------------------------------------------
// 4. Draw bounding boxes onto a copy of the image (GD library)
// ---------------------------------------------------------------
$processedPath = UPLOAD_PROCESSED_DIR . $uniqueName;
$drawSuccess = BoundingBoxDrawer::draw($originalPath, $detections, $processedPath);

if (!$drawSuccess) {
    // Not fatal - fall back to the original image if drawing failed
    $processedPath = $originalPath;
}

// ---------------------------------------------------------------
// 5. Build a short human-readable summary + persist to MySQL
// ---------------------------------------------------------------
$issueCounts = [];
foreach ($detections as $d) {
    $label = $d['label'] ?? 'unknown';
    $issueCounts[$label] = ($issueCounts[$label] ?? 0) + 1;
}
$summaryParts = [];
foreach ($issueCounts as $label => $count) {
    $summaryParts[] = "{$label} x{$count}";
}
$summary = $summaryParts
    ? implode(', ', $summaryParts)
    : ($skinStatus === 'clear' ? 'ผิวคุณดูใสสะอาด ไม่พบปัญหาที่ชัดเจน 🎉' : 'ไม่พบปัญหาผิวที่ชัดเจน');

$pdo = getDbConnection();

$stmt = $pdo->prepare(
    'INSERT INTO analysis_history
        (user_id, original_image_path, processed_image_path, skin_score, detections_json, summary)
     VALUES (:user_id, :original_path, :processed_path, :skin_score, :detections_json, :summary)'
);

$stmt->execute([
    ':user_id'         => $userId,
    ':original_path'   => 'uploads/original/' . $uniqueName,
    ':processed_path'  => 'uploads/processed/' . basename($processedPath),
    ':skin_score'      => $skinScore,
    ':detections_json' => json_encode($detections, JSON_UNESCAPED_UNICODE),
    ':summary'         => $summary,
]);

$analysisId = (int) $pdo->lastInsertId();

// ---------------------------------------------------------------
// 6. Fetch recommended products for the detected issues
// ---------------------------------------------------------------
$recommendedProducts = [];
$issueLabels = array_keys($issueCounts);

if ($issueLabels) {
    $placeholders = implode(',', array_fill(0, count($issueLabels), '?'));
    $productStmt = $pdo->prepare(
        "SELECT id, name, brand, description, key_ingredients, target_issue, price, image_url, purchase_url
         FROM products
         WHERE target_issue IN ($placeholders)
         ORDER BY FIELD(target_issue, $placeholders)"
    );
    // bind twice: once for WHERE, once for ORDER BY FIELD
    $productStmt->execute(array_merge($issueLabels, $issueLabels));
    $recommendedProducts = $productStmt->fetchAll();
} elseif ($skinStatus === 'clear') {
    // no specific issue to target — suggest general maintenance products instead
    $productStmt = $pdo->prepare(
        "SELECT id, name, brand, description, key_ingredients, target_issue, price, image_url, purchase_url
         FROM products WHERE target_issue = 'general' LIMIT 3"
    );
    $productStmt->execute();
    $recommendedProducts = $productStmt->fetchAll();
}

// ---------------------------------------------------------------
// 7. Respond to frontend
// ---------------------------------------------------------------
respond(200, [
    'success' => true,
    'data' => [
        'analysis_id'      => $analysisId,
        'skin_score'       => $skinScore,
        'skin_status'      => $skinStatus,
        'summary'          => $summary,
        'detections'       => $detections,
        'original_image'   => APP_BASE_URL . '/uploads/original/' . $uniqueName,
        'processed_image'  => APP_BASE_URL . '/uploads/processed/' . basename($processedPath),
        'recommended_products' => $recommendedProducts,
    ],
]);
