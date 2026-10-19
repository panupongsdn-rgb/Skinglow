<?php
declare(strict_types=1);

/**
 * Draws the analysed face zones (thin outlines) and the detection bounding
 * boxes + labels on a face image using the GD library. Detections with
 * "source" => "zone" are whole-zone boxes added by ZoneDetections (thinner line).
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

        $labelSlots = []; // box key => number of labels already drawn on that box
        $placed = [];     // label rectangles already drawn [x1, y1, x2, y2]
        $font = imagesx($image) >= 900 ? 5 : 4;
        $lineH = imagefontheight($font) + 4;

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

            [$xMin, $yMin, $xMax, $yMax] = array_map('intval', [$xMin, $yMin, $xMax, $yMax]);
            $key = "$xMin,$yMin,$xMax,$yMax";
            $slot = $labelSlots[$key] ?? 0;
            $labelSlots[$key] = $slot + 1;

            // Rectangle: 3px for exact detector boxes, 2px for whole-zone boxes;
            // drawn once per box even when several problems share it
            if ($slot === 0) {
                $thick = (($detection['source'] ?? '') === 'zone') ? 2 : 3;
                for ($t = 0; $t < $thick; $t++) {
                    imagerectangle($image, $xMin - $t, $yMin - $t, $xMax + $t, $yMax + $t, $color);
                }
            }

            // Label: above the box (inside it when there is no room), stacked
            // downwards when one box carries several problems
            $text = $conf !== null ? sprintf('%s %.0f%%', $label, $conf * 100) : $label;
            $textBoxWidth = imagefontwidth($font) * strlen($text) + 8;
            $top = ($yMin - $lineH >= 0) ? $yMin - $lineH + $slot * $lineH : $yMin + $slot * $lineH;
            if ($slot > 0 && $yMin - $lineH >= 0) {
                $top = $yMin + ($slot - 1) * $lineH; // first label sits above, the rest inside
            }
            // Keep the label inside the image and off labels already drawn
            $left = max(0, min($xMin, imagesx($image) - $textBoxWidth - 1));
            for ($try = 0; $try < 8 && self::overlaps($placed, $left, $top, $textBoxWidth, $lineH); $try++) {
                $top += $lineH;
            }
            $top = max(0, min($top, imagesy($image) - $lineH - 1));
            $placed[] = [$left, $top, $left + $textBoxWidth, $top + $lineH];
            imagefilledrectangle($image, $left, $top, $left + $textBoxWidth, $top + $lineH, $color);
            $white = imagecolorallocate($image, 255, 255, 255);
            imagestring($image, $font, $left + 4, $top + 2, $text, $white);
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

    /** @param array<int,array{int,int,int,int}> $placed */
    private static function overlaps(array $placed, int $x, int $y, int $w, int $h): bool
    {
        foreach ($placed as [$x1, $y1, $x2, $y2]) {
            if ($x < $x2 && $x + $w > $x1 && $y < $y2 && $y + $h > $y1) {
                return true;
            }
        }
        return false;
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
