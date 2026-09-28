<?php
declare(strict_types=1);

/**
 * Draws the analysed face zones (thin outlines) and the detection bounding
 * boxes + labels on a face image using the GD library.
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
     * @param array  $zones           Optional per-zone results; each may carry
     *                                "polygons" => [[[x, y], ...], ...]
     */
    public static function draw(string $sourcePath, array $detections, string $destinationPath, array $zones = []): bool
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

        self::drawZones($image, $zones);

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

    /** Zone outlines: lavender, 2px, drawn under the detection boxes. */
    private static function drawZones($image, array $zones): void
    {
        if (!$zones) {
            return;
        }
        $width = imagesx($image);
        $thickness = max(2, (int) round($width / 500));
        imagesetthickness($image, $thickness);
        $color = imagecolorallocatealpha($image, 179, 136, 235, 20);
        foreach ($zones as $zone) {
            foreach (($zone['polygons'] ?? []) as $polygon) {
                if (!is_array($polygon) || count($polygon) < 3) {
                    continue;
                }
                // imagesetthickness() is ignored for polygons by some GD
                // builds, so draw the outline a few times, 1px apart
                for ($t = 0; $t < $thickness; $t++) {
                    $points = [];
                    foreach ($polygon as $pt) {
                        $points[] = (int) ($pt[0] ?? 0) + $t;
                        $points[] = (int) ($pt[1] ?? 0);
                    }
                    imagepolygon($image, $points, $color);
                }
            }
        }
        imagesetthickness($image, 1);
    }
}
