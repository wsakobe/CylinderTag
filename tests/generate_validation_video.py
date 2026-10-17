"""Render independent known-ID cylindrical scenes and marker-free controls.
Requires numpy and OpenCV Python. No CAD or detector implementation imports.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def clip_polygon(poly, bound, keep_greater):
    out = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        ina = a[0] >= bound if keep_greater else a[0] <= bound
        inb = b[0] >= bound if keep_greater else b[0] <= bound
        if ina:
            out.append(a)
        if ina != inb:
            t = (bound - a[0]) / (b[0] - a[0])
            out.append((bound, a[1] + t * (b[1] - a[1])))
    return out


def render(codes, yaw, distance, mirror, roll):
    image = np.full((480, 640, 3), 175, np.uint8)
    radius, height, focal = 15., 40., 800.
    limit = math.acos(radius / distance)

    def project(poly):
        points = []
        for a, b in zip(poly, poly[1:] + poly[:1]):
            steps = max(1, int(abs(b[0] - a[0]) / .02) + 1)
            for t in np.linspace(0, 1, steps, endpoint=False):
                angle, y = np.array(a) * (1 - t) + np.array(b) * t
                z = distance - radius * math.cos(angle)
                points.append((320 + focal * radius * math.sin(angle) / z,
                               240 + focal * (y - height / 2) / z))
        return np.round(points).astype(np.int32)

    cv2.fillPoly(image, [project([(-limit, -2), (limit, -2), (limit, height + 2), (-limit, height + 2)])], (240, 240, 240))
    values = (1.47, 1.54, 1.61, 1.68, 1.68, 1.61, 1.54, 1.47)

    def center(symbol):
        t = math.sqrt(1 + 4 * (.11 - .2 * values[symbol]))
        return height * (1 + (t if symbol >= 4 else -t)) / 2

    for j, code in enumerate(codes):
        a, b = center(code // 8), center(code % 8)
        for wrap in (-2, -1, 0, 1, 2):
            left = 2 * math.pi * (j / len(codes) + wrap) + yaw
            right = left + 2 * math.pi / len(codes) / 1.5
            for poly in [[(left, 0), (right, 0), (right, b - 4), (left, a - 4)],
                         [(left, a + 4), (right, b + 4), (right, height), (left, height)]]:
                poly = clip_polygon(poly, -limit, True)
                if poly:
                    poly = clip_polygon(poly, limit, False)
                if len(poly) >= 3:
                    cv2.fillPoly(image, [project(poly)], (22, 22, 22))
    if mirror:
        image = cv2.flip(image, 0)
    image = cv2.warpAffine(image, cv2.getRotationMatrix2D((320, 240), roll, 1), (640, 480), borderValue=(175,)*3)
    image = cv2.GaussianBlur(image, (3, 3), .55)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--canonical-only', action='store_true',
                        help='Render every marker in dictionary orientation for like-for-like detector comparison')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    lines = (root / 'CTag_2f12c.marker').read_text().splitlines()
    dictionary = [list(map(int, x.split())) for x in lines[1:] if x.strip()]
    writer = cv2.VideoWriter(str(args.output / 'validation.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 480))
    if not writer.isOpened():
        raise RuntimeError('Cannot open validation writer')
    labels = []
    rng = np.random.default_rng(731)
    for marker_id, codes in enumerate(dictionary):
        for variant in range(4):
            mirror = bool(variant % 2) and not args.canonical_only
            image = render(codes, (marker_id * .271 + variant * .39) % (2 * math.pi),
                           160 if variant < 2 else 230, mirror, -12 if variant % 2 else 9)
            noise = rng.normal(0, 1.2, image.shape)
            image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
            writer.write(image)
            labels.append({'frame': len(labels), 'id': marker_id, 'mirrored': mirror, 'kind': 'synthetic_cylinder', 'variant': variant})
    for i in range(36):
        image = np.full((480, 640, 3), 180, np.uint8)
        if i % 3 == 1:
            for _ in range(25):
                x, y = rng.integers([10, 10], [600, 440])
                w, h = rng.integers([5, 5], [35, 80])
                cv2.rectangle(image, (x, y), (x+w, y+h), (30,)*3, -1)
        elif i % 3 == 2:
            for x in range(220, 420, 20):
                cv2.rectangle(image, (x, 120), (x+12, 360), (25,)*3, -1)
        writer.write(image)
        labels.append({'frame': len(labels), 'id': None, 'kind': 'negative_control'})
    writer.release()
    (args.output / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
    print(f'Generated {len(labels)} labeled frames ({len(dictionary)} IDs, four variants, 36 negatives)')


if __name__ == '__main__':
    main()
