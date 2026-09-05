<?php
/**
 * GET    /api/admin/products.php           — list all products
 * POST   /api/admin/products.php           — create a product (JSON body)
 * PUT    /api/admin/products.php?id=X      — update a product (JSON body)
 * DELETE /api/admin/products.php?id=X      — delete a product
 * Requires: Authorization: Bearer <token> for a user with role = 'admin'
 *
 * This is the "edit basic site content" surface — the product catalogue is
 * the one piece of the app's content that's data-driven rather than baked
 * into the frontend HTML, so it's what an admin can safely edit without
 * touching code.
 */
declare(strict_types=1);

require_once __DIR__ . 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/config/config.php';
require_once __DIR__ . 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/config/database.php';
require_once __DIR__ . 'https://skinglowjourney.infinityfree.io/Skinglow/backend-php/src/Auth.php';

header('Content-Type: application/json; charset=utf-8');

Auth::requireAdmin();
$pdo = getDbConnection();

const VALID_ISSUES = ['acne', 'black_spot', 'eyebag', 'redness', 'oiliness', 'wrinkle', 'dryness', 'general'];

function fail(int $status, string $message): never
{
    http_response_code($status);
    echo json_encode(['success' => false, 'message' => $message]);
    exit;
}

function readJsonBody(): array
{
    $input = json_decode(file_get_contents('php://input'), true);
    return is_array($input) ? $input : [];
}

function validateProductInput(array $input, bool $partial = false): array
{
    $errors = [];
    $clean = [];

    foreach (['name', 'brand', 'description', 'key_ingredients', 'image_url', 'purchase_url'] as $field) {
        if (isset($input[$field])) {
            $clean[$field] = trim((string) $input[$field]);
        }
    }

    if (isset($input['target_issue'])) {
        if (!in_array($input['target_issue'], VALID_ISSUES, true)) {
            $errors[] = 'target_issue must be one of: ' . implode(', ', VALID_ISSUES);
        } else {
            $clean['target_issue'] = $input['target_issue'];
        }
    } elseif (!$partial) {
        $clean['target_issue'] = 'general';
    }

    if (isset($input['price'])) {
        if (!is_numeric($input['price']) || (float) $input['price'] < 0) {
            $errors[] = 'price must be a non-negative number';
        } else {
            $clean['price'] = (float) $input['price'];
        }
    }

    if (!$partial && empty($clean['name'])) {
        $errors[] = 'name is required';
    }

    return [$clean, $errors];
}

$method = $_SERVER['REQUEST_METHOD'];

if ($method === 'GET') {
    $stmt = $pdo->query('SELECT * FROM products ORDER BY created_at DESC');
    echo json_encode(['success' => true, 'data' => $stmt->fetchAll()], JSON_UNESCAPED_UNICODE);
    exit;
}

if ($method === 'POST') {
    [$clean, $errors] = validateProductInput(readJsonBody(), partial: false);
    if ($errors) {
        fail(422, implode('; ', $errors));
    }

    $columns = array_keys($clean);
    $placeholders = array_map(fn($c) => ":$c", $columns);
    $stmt = $pdo->prepare(
        'INSERT INTO products (' . implode(', ', $columns) . ') VALUES (' . implode(', ', $placeholders) . ')'
    );
    $stmt->execute(array_combine($placeholders, array_values($clean)));

    echo json_encode(['success' => true, 'data' => ['id' => (int) $pdo->lastInsertId()]]);
    exit;
}

if ($method === 'PUT') {
    $id = filter_input(INPUT_GET, 'id', FILTER_VALIDATE_INT);
    if (!$id) {
        fail(422, 'id is required and must be an integer.');
    }

    [$clean, $errors] = validateProductInput(readJsonBody(), partial: true);
    if ($errors) {
        fail(422, implode('; ', $errors));
    }
    if (!$clean) {
        fail(422, 'No valid fields to update.');
    }

    $setClauses = array_map(fn($c) => "$c = :$c", array_keys($clean));
    $stmt = $pdo->prepare('UPDATE products SET ' . implode(', ', $setClauses) . ' WHERE id = :id');
    $params = [];
    foreach ($clean as $k => $v) {
        $params[":$k"] = $v;
    }
    $params[':id'] = $id;
    $stmt->execute($params);

    if ($stmt->rowCount() === 0) {
        // could mean "not found" or "no actual change" — check existence explicitly
        $exists = $pdo->prepare('SELECT 1 FROM products WHERE id = :id');
        $exists->execute([':id' => $id]);
        if (!$exists->fetch()) {
            fail(404, 'ไม่พบสินค้านี้');
        }
    }

    echo json_encode(['success' => true, 'message' => 'อัปเดตสินค้าเรียบร้อยแล้ว']);
    exit;
}

if ($method === 'DELETE') {
    $id = filter_input(INPUT_GET, 'id', FILTER_VALIDATE_INT);
    if (!$id) {
        fail(422, 'id is required and must be an integer.');
    }

    $stmt = $pdo->prepare('DELETE FROM products WHERE id = :id');
    $stmt->execute([':id' => $id]);

    if ($stmt->rowCount() === 0) {
        fail(404, 'ไม่พบสินค้านี้');
    }

    echo json_encode(['success' => true, 'message' => 'ลบสินค้าเรียบร้อยแล้ว']);
    exit;
}

fail(405, 'Method not allowed.');
