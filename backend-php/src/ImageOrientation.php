<?php
declare(strict_types=1);

/**
 * Normalizes JPEG orientation using GD, mirroring what
 * PIL.ImageOps.exif_transpose() does on the Python AI service side
 * (ai-service/main.py). Both sides must agree on "upright" pixels, or
 * detection box coordinates computed by one service land in the wrong
 * place when drawn by the other.
 *
 * WHY THIS MATTERS: browsers auto-rotate a raw <img src="photo.jpg">
 * using its EXIF orientation tag, so an unrotated phone photo still
 * *looks* correct when previewed directly. GD does not do this — it
 * reads raw pixel data and ignores the orientation tag entirely. Worse,
 * when GD re-saves a file (e.g. after drawing bounding boxes), the
 * output JPEG has no EXIF orientation tag at all, so there's nothing
 * left for the browser to auto-correct with either — the sideways
 * pixels are all that's left. This is why only the *processed* image
 * (with boxes drawn) appeared landscape, while the original preview
 * looked fine.
 *
 * Only JPEG carries EXIF orientation in any standard way; PNG/WEBP
 * uploads are left untouched (silently skipped, not an error).
 */
final class ImageOrientation
{
    /**
     * Rotates the file in place if its EXIF orientation tag requires it.
     * Returns true if a rotation was applied, false otherwise (already
     * upright, not a JPEG, no EXIF data, or GD/exif unavailable).
     */
    public static function normalize(string $path, string $mimeType): bool
    {
        if ($mimeType !== 'image/jpeg' || !function_exists('exif_read_data')) {
            return false;
        }

        $exif = @exif_read_data($path);
        $orientation = $exif['Orientation'] ?? 1;

        if (!in_array($orientation, [3, 6, 8], true)) {
            // 1 = already upright. Orientations 2/4/5/7 involve a mirror
            // flip in addition to rotation — vanishingly rare from real
            // camera captures (only 1/3/6/8 occur in practice), so they're
            // intentionally left unhandled rather than adding untested
            // complexity for a case that won't occur in real usage.
            return false;
        }

        $image = imagecreatefromjpeg($path);
        if ($image === false) {
            return false;
        }

        $rotated = match ($orientation) {
            3 => imagerotate($image, 180, 0),
            6 => imagerotate($image, -90, 0),
            8 => imagerotate($image, 90, 0),
            default => $image,
        };
        imagedestroy($image);

        if ($rotated === false) {
            return false;
        }

        $saved = imagejpeg($rotated, $path, 92);
        imagedestroy($rotated);

        return $saved;
    }
}
