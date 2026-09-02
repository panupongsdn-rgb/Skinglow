<?php
declare(strict_types=1);

/**
 * Draws bounding boxes + labels on a face image using the GD library.
 * Input format for detections (as returned by the AI service):
 * [
 *   ["box" => [x_min, y_min, x_max, y_max], "label" => "acne", "confidence" => 0.85],
 *   ...
 * ]
 */
final class BoundingBoxDrawer
{
    /** @var array<string,array{int,int,int}> label => RGB color */
    private static array $colorMap = [
        'acne'       => [255, 71, 87],
        'black_spot' => [47, 158, 143],
        'eyebag'     => [155, 122, 89],
        'redness'    => [231, 76, 60],
        'oiliness'   => [241, 169, 62],
        'wrinkle'    => [52, 152, 219],
        'default'    => [46, 204, 113],
    ];

    /**
     * @param string $sourcePath      Absolute path to the original uploaded image
     * @param array  $detections      Decoded AI detection results
     * @param string $destinationPath Absolute path to save the annotated image
     */
    public static function draw(string $sourcePath, array $detections, string $destinationPath): bool
    {
        $mime = mime_content_type($sourcePath);

        $image = match ($mime) {
            'image/jpeg' => imagecreatefromjpeg($sourcePath),
            'image/png'  => imagecreatefrompng($sourcePath),
            'image/webp' => imagecreatefromwebp($sourcePath),
            default      => false,
        };

        if ($image === false) {
            return false;
        }

        foreach ($detections as $detection) {
            $box   = $detection['box'] ?? null;
            $label = $detection['label'] ?? 'default';
            $conf  = $detection['confidence'] ?? null;

            if (!$box || count($box) !== 4) {
                continue;
            }

            [$xMin, $yMin, $xMax, $yMax] = $box;
            [$r, $g, $b] = self::$colorMap[$label] ?? self::$colorMap['default'];
            $color = imagecolorallocate($image, $r, $g, $b);

            // Draw rectangle (thickness 3px by drawing multiple offsets)
            for ($t = 0; $t < 3; $t++) {
                imagerectangle($image, $xMin - $t, $yMin - $t, $xMax + $t, $yMax + $t, $color);
            }

            // Label background + text
            $text = $conf !== null ? sprintf('%s %.0f%%', $label, $conf * 100) : $label;
            $textBoxWidth = imagefontwidth(4) * strlen($text) + 8;
            imagefilledrectangle($image, $xMin, max(0, $yMin - 18), $xMin + $textBoxWidth, $yMin, $color);

            $white = imagecolorallocate($image, 255, 255, 255);
            imagestring($image, 4, $xMin + 4, max(0, $yMin - 18), $text, $white);
        }

        $saved = match ($mime) {
            'image/jpeg' => imagejpeg($image, $destinationPath, 90),
            'image/png'  => imagepng($image, $destinationPath),
            'image/webp' => imagewebp($image, $destinationPath, 90),
            default      => false,
        };

        imagedestroy($image);

        return (bool) $saved;
    }
}
