<?php
declare(strict_types=1);

/**
 * Makes the boxes drawn on the photo agree with the per-zone results.
 *
 * The per-zone classifier decides WHAT each zone has (forehead: dark spots,
 * chin: acne, ...) but not exactly where inside the zone. The YOLO detector
 * gives exact boxes but misses or mislabels a lot. So:
 *   1. keep a YOLO box only if the zone it sits in was also given that
 *      problem by the classifier (drops e.g. a "redness" box on a forehead
 *      the classifier says has dark spots);
 *   2. for every zone problem that has no YOLO box, add a box around that
 *      zone (one per outline: under-eye gets one box per eye), marked
 *      source = "zone", confidence = the classifier's probability.
 * Without zone-classifier results the detections are returned unchanged.
 */
final class ZoneDetections
{
    /**
     * @param array $detections [{box:[x1,y1,x2,y2], label, confidence}, ...]
     * @param array $zones      [{zone, name_th, box, polygons, issues:[{label, label_th, probability}]}, ...]
     * @return array            detections to draw / count
     */
    public static function reconcile(array $detections, array $zones): array
    {
        $hasIssues = false;
        foreach ($zones as $z) {
            if (!empty($z['issues'])) {
                $hasIssues = true;
                break;
            }
        }
        if (!$zones) {
            return $detections;
        }

        $kept = [];
        $covered = []; // "zone|label" => true
        foreach ($detections as $d) {
            $box = $d['box'] ?? null;
            if (!is_array($box) || count($box) !== 4) {
                continue;
            }
            $cx = ($box[0] + $box[2]) / 2;
            $cy = ($box[1] + $box[3]) / 2;
            $zone = self::zoneAt($zones, $cx, $cy);
            if ($zone === null) {
                continue; // outside every zone (hair, background, neck)
            }
            foreach (($zone['issues'] ?? []) as $issue) {
                if (($issue['label'] ?? null) === ($d['label'] ?? null)) {
                    $d['zone'] = $zone['zone'] ?? null;
                    $d['source'] = 'detector';
                    $kept[] = $d;
                    $covered[($zone['zone'] ?? '') . '|' . $d['label']] = true;
                    break;
                }
            }
        }
        if (!$hasIssues) {
            return $kept;
        }

        foreach ($zones as $z) {
            foreach (($z['issues'] ?? []) as $issue) {
                $label = $issue['label'] ?? null;
                if ($label === null || isset($covered[($z['zone'] ?? '') . '|' . $label])) {
                    continue;
                }
                foreach (self::zoneBoxes($z) as $box) {
                    $kept[] = [
                        'box'        => $box,
                        'label'      => $label,
                        'confidence' => (float) ($issue['probability'] ?? 0),
                        'zone'       => $z['zone'] ?? null,
                        'source'     => 'zone',
                    ];
                }
            }
        }
        return $kept;
    }

    /** One box per outline of the zone (under-eye has two), else the zone box. */
    private static function zoneBoxes(array $zone): array
    {
        $boxes = [];
        foreach (($zone['polygons'] ?? []) as $poly) {
            if (!is_array($poly) || count($poly) < 3) {
                continue;
            }
            $xs = array_map(fn($p) => (int) ($p[0] ?? 0), $poly);
            $ys = array_map(fn($p) => (int) ($p[1] ?? 0), $poly);
            $boxes[] = [min($xs), min($ys), max($xs), max($ys)];
        }
        if (!$boxes && is_array($zone['box'] ?? null) && count($zone['box']) === 4) {
            $boxes[] = array_map('intval', $zone['box']);
        }
        return $boxes;
    }

    private static function zoneAt(array $zones, float $x, float $y): ?array
    {
        foreach ($zones as $z) {
            foreach (($z['polygons'] ?? []) as $poly) {
                if (is_array($poly) && count($poly) >= 3 && self::inPolygon($poly, $x, $y)) {
                    return $z;
                }
            }
        }
        return null;
    }

    private static function inPolygon(array $poly, float $x, float $y): bool
    {
        $inside = false;
        $n = count($poly);
        for ($i = 0, $j = $n - 1; $i < $n; $j = $i++) {
            $xi = (float) $poly[$i][0]; $yi = (float) $poly[$i][1];
            $xj = (float) $poly[$j][0]; $yj = (float) $poly[$j][1];
            if ((($yi > $y) !== ($yj > $y)) && ($x < ($xj - $xi) * ($y - $yi) / (($yj - $yi) ?: 1e-9) + $xi)) {
                $inside = !$inside;
            }
        }
        return $inside;
    }
}
