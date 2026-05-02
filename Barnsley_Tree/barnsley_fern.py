from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path


WIDTH = 900
HEIGHT = 1200
DEFAULT_POINTS = 120_000
DEFAULT_FPS = 30
DEFAULT_SECONDS = 8
DEFAULT_MODEL_REPORT = Path("barnsley_models.md")


class NearestCentroidModel:
    def __init__(self) -> None:
        self.centroids: dict[int, tuple[float, float]] = {}

    def fit(self, samples: list[tuple[float, float, int]]) -> None:
        totals: dict[int, list[float]] = {}
        counts: dict[int, int] = {}

        for x, y, label in samples:
            if label not in totals:
                totals[label] = [0.0, 0.0]
                counts[label] = 0
            totals[label][0] += x
            totals[label][1] += y
            counts[label] += 1

        self.centroids = {
            label: (totals[label][0] / counts[label], totals[label][1] / counts[label])
            for label in totals
        }

    def predict_one(self, x: float, y: float) -> int:
        best_label = -1
        best_distance = float("inf")
        for label, (centroid_x, centroid_y) in self.centroids.items():
            distance = (x - centroid_x) ** 2 + (y - centroid_y) ** 2
            if distance < best_distance:
                best_distance = distance
                best_label = label
        return best_label


class GaussianNaiveBayesModel:
    def __init__(self) -> None:
        self.means: dict[int, tuple[float, float]] = {}
        self.variances: dict[int, tuple[float, float]] = {}
        self.priors: dict[int, float] = {}

    def fit(self, samples: list[tuple[float, float, int]]) -> None:
        grouped: dict[int, list[tuple[float, float]]] = {}
        for x, y, label in samples:
            grouped.setdefault(label, []).append((x, y))

        total = max(1, len(samples))
        for label, rows in grouped.items():
            count = len(rows)
            xs = [x for x, _ in rows]
            ys = [y for _, y in rows]
            mean_x = sum(xs) / count
            mean_y = sum(ys) / count
            var_x = sum((x - mean_x) ** 2 for x in xs) / count + 1e-6
            var_y = sum((y - mean_y) ** 2 for y in ys) / count + 1e-6
            self.means[label] = (mean_x, mean_y)
            self.variances[label] = (var_x, var_y)
            self.priors[label] = count / total

    def predict_one(self, x: float, y: float) -> int:
        best_label = -1
        best_score = float("-inf")
        for label in self.means:
            mean_x, mean_y = self.means[label]
            var_x, var_y = self.variances[label]
            score = math.log(self.priors[label])
            score += -0.5 * (math.log(2 * math.pi * var_x) + ((x - mean_x) ** 2) / var_x)
            score += -0.5 * (math.log(2 * math.pi * var_y) + ((y - mean_y) ** 2) / var_y)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label


class KNearestNeighborsModel:
    def __init__(self, k: int = 7) -> None:
        self.k = k
        self.samples: list[tuple[float, float, int]] = []

    def fit(self, samples: list[tuple[float, float, int]]) -> None:
        self.samples = list(samples)

    def predict_one(self, x: float, y: float) -> int:
        scored = sorted(
            ((x - sample_x) ** 2 + (y - sample_y) ** 2, label)
            for sample_x, sample_y, label in self.samples
        )[: self.k]

        votes: dict[int, float] = {}
        for distance, label in scored:
            votes[label] = votes.get(label, 0.0) + 1.0 / (distance + 1e-9)
        return max(votes, key=votes.get)


def barnsley_points(count: int, seed: int) -> list[tuple[float, float]]:
    return [(x, y) for x, y, _ in barnsley_dataset(count, seed)]


def barnsley_dataset(count: int, seed: int) -> list[tuple[float, float, int]]:
    rng = random.Random(seed)
    x = 0.0
    y = 0.0
    points: list[tuple[float, float, int]] = []

    for _ in range(count):
        r = rng.random()
        if r < 0.01:
            x, y = 0.0, 0.16 * y
            label = 0
        elif r < 0.86:
            x, y = 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6
            label = 1
        elif r < 0.93:
            x, y = 0.2 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6
            label = 2
        else:
            x, y = -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44
            label = 3
        points.append((x, y, label))

    return points


def project_points(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[int, int, int]]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    padding = 50
    scale_x = (width - 2 * padding) / (max_x - min_x)
    scale_y = (height - 2 * padding) / (max_y - min_y)
    scale = min(scale_x, scale_y)

    projected: list[tuple[int, int, int]] = []
    total = max(1, len(points) - 1)
    for index, (x, y) in enumerate(points):
        px = int(round(padding + (x - min_x) * scale))
        py = int(round(height - padding - (y - min_y) * scale))
        projected.append((px, py, index))

    return projected


def to_svg(points: list[tuple[float, float]], width: int, height: int) -> str:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    padding = 50
    scale_x = (width - 2 * padding) / (max_x - min_x)
    scale_y = (height - 2 * padding) / (max_y - min_y)
    scale = min(scale_x, scale_y)

    def project(x: float, y: float) -> tuple[float, float]:
        px = padding + (x - min_x) * scale
        py = height - padding - (y - min_y) * scale
        return px, py

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs>',
        '<radialGradient id="bg" cx="50%" cy="35%" r="70%">',
        '<stop offset="0%" stop-color="#173222"/>',
        '<stop offset="100%" stop-color="#07150d"/>',
        '</radialGradient>',
        '<filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feGaussianBlur stdDeviation="1.2" result="blur"/>',
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>',
        '</filter>',
        '</defs>',
        '<rect width="100%" height="100%" fill="url(#bg)"/>',
        '<g filter="url(#glow)">',
    ]

    total = max(1, len(points) - 1)
    for index, (x, y) in enumerate(points):
        px, py = project(x, y)
        t = index / total
        radius = 0.42 + 0.3 * (1 - t)
        opacity = 0.06 + 0.72 * t**0.45
        green = 130 + int(70 * t)
        parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="hsl({green} 68% 58%)" opacity="{opacity:.3f}"/>'
        )

    parts.extend(['</g>', '</svg>'])
    return "\n".join(parts)


def make_frame_buffer(width: int, height: int) -> bytearray:
    buffer = bytearray(width * height * 3)
    background = (7, 21, 13)
    for offset in range(0, len(buffer), 3):
        buffer[offset : offset + 3] = bytes(background)
    return buffer


def plot_pixel(buffer: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return

    offset = (y * width + x) * 3
    buffer[offset : offset + 3] = bytes(color)


def plot_brush(buffer: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1)):
        plot_pixel(buffer, width, height, x + dx, y + dy, color)


def write_video(points: list[tuple[float, float]], output: Path, width: int, height: int, fps: int, seconds: int) -> None:
    projected = project_points(points, width, height)
    frame_count = max(1, fps * seconds)
    points_per_frame = max(1, math.ceil(len(projected) / frame_count))

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    assert ffmpeg.stdin is not None
    try:
        buffer = make_frame_buffer(width, height)
        total = max(1, len(projected) - 1)

        for frame_index in range(frame_count):
            end = min(len(projected), (frame_index + 1) * points_per_frame)
            start = frame_index * points_per_frame
            for x, y, point_index in projected[start:end]:
                t = point_index / total
                green = 130 + int(70 * t)
                color = (40 + int(20 * t), green, 60 + int(20 * t))
                plot_brush(buffer, width, height, x, y, color)

            ffmpeg.stdin.write(buffer)

        ffmpeg.stdin.close()
        stderr = ffmpeg.stderr.read().decode("utf-8", errors="replace") if ffmpeg.stderr else ""
        return_code = ffmpeg.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr.strip()}")
    finally:
        if ffmpeg.stdin and not ffmpeg.stdin.closed:
            ffmpeg.stdin.close()


def split_dataset(samples: list[tuple[float, float, int]], test_ratio: float, seed: int) -> tuple[list[tuple[float, float, int]], list[tuple[float, float, int]]]:
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    test_size = max(1, int(len(shuffled) * test_ratio))
    return shuffled[test_size:], shuffled[:test_size]


def sample_dataset(samples: list[tuple[float, float, int]], limit: int, seed: int) -> list[tuple[float, float, int]]:
    if len(samples) <= limit:
        return list(samples)

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    return [samples[index] for index in indices[:limit]]


def accuracy(model: object, samples: list[tuple[float, float, int]]) -> float:
    correct = 0
    for x, y, label in samples:
        if model.predict_one(x, y) == label:
            correct += 1
    return correct / max(1, len(samples))


def write_model_report(output: Path, samples: list[tuple[float, float, int]], seed: int) -> None:
    reduced_samples = sample_dataset(samples, limit=8_000, seed=seed)
    train_samples, test_samples = split_dataset(reduced_samples, test_ratio=0.2, seed=seed)

    models: list[tuple[str, object]] = [
        ("Nearest centroid", NearestCentroidModel()),
        ("Gaussian naive Bayes", GaussianNaiveBayesModel()),
        ("K-nearest neighbors (k=7)", KNearestNeighborsModel(k=7)),
    ]

    lines = [
        "# Barnsley Fern Models",
        "",
        "These models learn to predict which affine transform generated each fern point.",
        "",
        "| Model | Train accuracy | Test accuracy |",
        "| --- | ---: | ---: |",
    ]

    for name, model in models:
        model.fit(train_samples)
        train_accuracy = accuracy(model, train_samples)
        test_accuracy = accuracy(model, test_samples)
        lines.append(f"| {name} | {train_accuracy:.3f} | {test_accuracy:.3f} |")

    lines.extend([
        "",
        "The labels come directly from the four Barnsley fern transforms, so this is a clean supervised learning setup.",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Barnsley fern SVG demo.")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help="number of points to plot")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--output", type=Path, default=Path("barnsley_fern.svg"), help="output SVG path")
    parser.add_argument("--video-output", type=Path, default=Path("barnsley_fern.mp4"), help="output MP4 path")
    parser.add_argument("--model-report", type=Path, default=DEFAULT_MODEL_REPORT, help="output markdown report for the ML models")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="video frames per second")
    parser.add_argument("--seconds", type=int, default=DEFAULT_SECONDS, help="video duration in seconds")
    args = parser.parse_args()

    dataset = barnsley_dataset(args.points, args.seed)
    points = [(x, y) for x, y, _ in dataset]
    svg = to_svg(points, WIDTH, HEIGHT)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output} with {len(points)} points")

    write_video(points, args.video_output, WIDTH, HEIGHT, args.fps, args.seconds)
    print(f"wrote {args.video_output} at {args.fps} fps for {args.seconds} seconds")

    write_model_report(args.model_report, dataset, args.seed + 99)
    print(f"wrote {args.model_report} with 3 trained models")


if __name__ == "__main__":
    main()