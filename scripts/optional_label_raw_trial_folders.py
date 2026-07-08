from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# Change these two paths if your dataset/output location is different.
SESSION_DIR = Path(r"/kaggle/input/datasets/shishensaiweng/kaggle-flat-90trials-images-and-labels")
OUTPUT_DIR = Path(r"/kaggle/working")

PER_TRIAL_DIR = OUTPUT_DIR / "per_trial_csv"
METRICS_DIR = OUTPUT_DIR / "per_trial_metrics"


LABELS = {
    -1: "ignore_first_frame",
    0: "stable",
    1: "incipient_slip",
    2: "translational_slip",
}


@dataclass
class TrialSummary:
    trial_folder: str
    frame_count: int
    stable_range: str
    incipient_range: str
    translational_range: str
    incipient_start: int | None
    translational_start: int | None
    note: str


def load_gray(path: Path) -> np.ndarray:
    """Load one tactile image as grayscale float array."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def build_contact_mask(arr: np.ndarray) -> np.ndarray:
    """
    Keep the dark circular contact area and suppress the white background.
    This avoids tracking glare outside the tactile imprint.
    """
    rough = arr < 225
    ys, xs = np.where(rough)
    if len(xs) < 50:
        return np.ones(arr.shape, dtype=bool)

    cx = float(xs.mean())
    cy = float(ys.mean())
    radii = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r = float(np.percentile(radii, 88))

    yy, xx = np.indices(arr.shape)
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < (r * 0.88) ** 2
    return mask


def marker_centers(arr: np.ndarray, mask: np.ndarray, max_points: int = 180) -> np.ndarray:
    """
    Detect bright tactile markers as local maxima.

    Returns an array with shape [N, 2], columns are x, y.
    """
    vals = arr[mask]
    if vals.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    threshold = np.percentile(vals, 96)
    candidate = (arr >= threshold) & mask

    padded = np.pad(arr, 2, mode="edge")
    local_max = np.ones(arr.shape, dtype=bool)
    for dy in range(5):
        for dx in range(5):
            if dy == 2 and dx == 2:
                continue
            local_max &= arr >= padded[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]

    ys, xs = np.where(candidate & local_max)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    intensities = arr[ys, xs]
    order = np.argsort(intensities)[::-1]

    selected: list[tuple[float, float]] = []
    min_sep = 5.0
    for idx in order:
        x = float(xs[idx])
        y = float(ys[idx])
        if all((x - px) ** 2 + (y - py) ** 2 >= min_sep**2 for px, py in selected):
            selected.append((x, y))
        if len(selected) >= max_points:
            break

    return np.asarray(selected, dtype=np.float32)


def match_points(
    ref_points: np.ndarray,
    cur_points: np.ndarray,
    max_distance: float = 18.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy nearest-neighbor matching from reference markers to current markers.
    This is sufficient for small frame-to-frame marker motion.
    """
    if len(ref_points) == 0 or len(cur_points) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(ref_points):
        d = np.sqrt(((cur_points - p) ** 2).sum(axis=1))
        j = int(np.argmin(d))
        if float(d[j]) <= max_distance:
            pairs.append((float(d[j]), i, j))

    pairs.sort(key=lambda item: item[0])
    used_ref: set[int] = set()
    used_cur: set[int] = set()
    matched_ref: list[np.ndarray] = []
    matched_cur: list[np.ndarray] = []

    for _, i, j in pairs:
        if i in used_ref or j in used_cur:
            continue
        used_ref.add(i)
        used_cur.add(j)
        matched_ref.append(ref_points[i])
        matched_cur.append(cur_points[j])

    if not matched_ref:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    return np.asarray(matched_ref, dtype=np.float32), np.asarray(matched_cur, dtype=np.float32)


def fit_similarity_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a 2D rigid/similarity transform:
        dst ~= scale * R @ src + t

    This approximates the object imprint moving as a 2D rigid body.
    Local slip is then measured as residual motion beyond this transform.
    """
    if len(src) < 3:
        return np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean

    covariance = src0.T @ dst0 / len(src)
    u, s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    variance = np.mean(np.sum(src0**2, axis=1))
    scale = float(np.sum(s) / variance) if variance > 1e-8 else 1.0

    matrix = (scale * rotation).astype(np.float32)
    translation = (dst_mean - matrix @ src_mean).astype(np.float32)
    return matrix, translation


def predict_points(src: np.ndarray, matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return src @ matrix.T + translation


def central_region(points: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    """
    Split markers into central and peripheral regions.

    Incipient slip often begins locally/peripherally while the center is still
    close to sticking. Translational slip tends to move the whole marker field.
    """
    h, w = mask_shape
    cx = w / 2.0
    cy = h / 2.0
    dist = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    radius = max(1.0, np.percentile(dist, 70))
    return dist <= radius * 0.58


def direction_consistency(displacements: np.ndarray) -> float:
    """
    Value near 1 means all markers move in almost the same direction.
    This is a useful sign of gross/translational slip.
    """
    if len(displacements) < 3:
        return 0.0

    mags = np.linalg.norm(displacements, axis=1)
    valid = mags > 1e-6
    if valid.sum() < 3:
        return 0.0

    unit = displacements[valid] / mags[valid, None]
    mean_vector = unit.mean(axis=0)
    return float(np.linalg.norm(mean_vector))


def robust_baseline(values: list[float], n: int = 4) -> tuple[float, float]:
    """
    Use early frames after frame_01 to estimate natural noise.
    Returns median and robust standard deviation.
    """
    sample = np.asarray(values[:n], dtype=np.float32)
    if sample.size == 0:
        return 0.0, 1.0
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    robust_std = 1.4826 * mad
    return med, max(robust_std, 0.15)


def first_sustained_frame(
    frame_numbers: list[int],
    values: list[float],
    threshold: float,
    min_consecutive: int = 2,
    start_at: int = 3,
) -> int | None:
    for i, frame_number in enumerate(frame_numbers):
        if frame_number < start_at:
            continue
        window = values[i : i + min_consecutive]
        if len(window) < min_consecutive:
            return None
        if all(v >= threshold for v in window):
            return frame_number
    return None


def range_text(start: int | None, end: int | None) -> str:
    if start is None or end is None or start > end:
        return ""
    if start == end:
        return f"frame_{start:02d}"
    return f"frame_{start:02d}-frame_{end:02d}"


def analyze_trial(trial_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], TrialSummary]:
    frame_paths = sorted(trial_dir.glob("frame_*.png"))
    if len(frame_paths) < 3:
        raise ValueError(f"Not enough frames in {trial_dir}")

    ref_path = trial_dir / "frame_02.png"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference frame: {ref_path}")

    ref_img = load_gray(ref_path)
    mask = build_contact_mask(ref_img)
    ref_markers = marker_centers(ref_img, mask)

    metrics: list[dict[str, object]] = []
    frame_numbers: list[int] = []
    global_motion_values: list[float] = []
    residual_ratio_values: list[float] = []
    peripheral_residual_values: list[float] = []
    consistency_values: list[float] = []

    for path in frame_paths:
        frame_number = int(path.stem.split("_")[1])
        img = load_gray(path)
        cur_markers = marker_centers(img, mask)
        matched_ref, matched_cur = match_points(ref_markers, cur_markers)

        if len(matched_ref) >= 3:
            displacement = matched_cur - matched_ref
            global_motion = float(np.median(np.linalg.norm(displacement, axis=1)))
            consistency = direction_consistency(displacement)

            center_mask = central_region(matched_ref, ref_img.shape)
            if center_mask.sum() >= 3:
                fit_src = matched_ref[center_mask]
                fit_dst = matched_cur[center_mask]
            else:
                fit_src = matched_ref
                fit_dst = matched_cur

            matrix, translation = fit_similarity_transform(fit_src, fit_dst)
            predicted = predict_points(matched_ref, matrix, translation)
            residual = np.linalg.norm(matched_cur - predicted, axis=1)

            central_residual = float(np.median(residual[center_mask])) if center_mask.any() else float(np.median(residual))
            peripheral_mask = ~center_mask
            peripheral_residual = (
                float(np.median(residual[peripheral_mask])) if peripheral_mask.any() else float(np.median(residual))
            )
            rigid_residual = float(np.median(residual))
            residual_ratio = float(peripheral_residual / max(central_residual, 0.15))
        else:
            global_motion = math.nan
            consistency = math.nan
            central_residual = math.nan
            peripheral_residual = math.nan
            rigid_residual = math.nan
            residual_ratio = math.nan

        row = {
            "trial_folder": trial_dir.name,
            "frame_id": path.stem,
            "frame_number": frame_number,
            "matched_markers": int(len(matched_ref)),
            "global_motion_px": global_motion,
            "direction_consistency": consistency,
            "rigid_residual_px": rigid_residual,
            "central_residual_px": central_residual,
            "peripheral_residual_px": peripheral_residual,
            "peripheral_to_central_residual_ratio": residual_ratio,
        }
        metrics.append(row)

        if frame_number >= 2 and not math.isnan(global_motion):
            frame_numbers.append(frame_number)
            global_motion_values.append(global_motion)
            residual_ratio_values.append(residual_ratio)
            peripheral_residual_values.append(peripheral_residual)
            consistency_values.append(consistency)

    # Adaptive thresholds from early stable-looking frames.
    # These thresholds are intentionally conservative to avoid labeling noise as slip.
    gm_base, gm_noise = robust_baseline(global_motion_values, n=4)
    pr_base, pr_noise = robust_baseline(peripheral_residual_values, n=4)
    rr_base, rr_noise = robust_baseline(residual_ratio_values, n=4)

    incipient_threshold = max(pr_base + 3.0 * pr_noise, 0.65)
    residual_ratio_threshold = max(rr_base + 2.5 * rr_noise, 1.55)
    translational_motion_threshold = max(gm_base + 5.0 * gm_noise, 3.0)
    translational_consistency_threshold = 0.72

    incipient_candidates = [
        max(pr, 0.0) * max(rr / residual_ratio_threshold, 0.0)
        for pr, rr in zip(peripheral_residual_values, residual_ratio_values)
    ]
    incipient_score_threshold = incipient_threshold

    incipient_start = first_sustained_frame(
        frame_numbers,
        incipient_candidates,
        incipient_score_threshold,
        min_consecutive=2,
        start_at=3,
    )

    translational_flags = [
        gm >= translational_motion_threshold and dc >= translational_consistency_threshold
        for gm, dc in zip(global_motion_values, consistency_values)
    ]
    translational_values = [1.0 if flag else 0.0 for flag in translational_flags]
    translational_start = first_sustained_frame(
        frame_numbers,
        translational_values,
        threshold=1.0,
        min_consecutive=2,
        start_at=3,
    )

    # Fallbacks for unusual trials:
    # If rigid residual did not clearly show local slip, use the first obvious
    # global motion increase as incipient and the later consistent motion as translational.
    if incipient_start is None:
        incipient_start = first_sustained_frame(
            frame_numbers,
            global_motion_values,
            gm_base + 3.0 * gm_noise,
            min_consecutive=2,
            start_at=3,
        )

    if translational_start is None:
        translational_start = first_sustained_frame(
            frame_numbers,
            global_motion_values,
            translational_motion_threshold,
            min_consecutive=2,
            start_at=3,
        )

    # Ensure incipient is earlier than translational. If they coincide, mark the
    # previous frame as incipient when possible.
    if translational_start is not None and incipient_start is not None:
        if incipient_start >= translational_start and translational_start > 3:
            incipient_start = translational_start - 1

    last_frame = max(int(p.stem.split("_")[1]) for p in frame_paths)
    labels: list[dict[str, object]] = []
    for path in frame_paths:
        frame_number = int(path.stem.split("_")[1])
        if frame_number == 1:
            phase = -1
        elif translational_start is not None and frame_number >= translational_start:
            phase = 2
        elif incipient_start is not None and frame_number >= incipient_start:
            phase = 1
        else:
            phase = 0

        labels.append(
            {
                "trial_folder": trial_dir.name,
                "frame_id": path.stem,
                "phase": phase,
                "phase_name": LABELS[phase],
            }
        )

    stable_end = None
    incipient_end = None
    if incipient_start is not None:
        stable_end = incipient_start - 1
    elif translational_start is not None:
        stable_end = translational_start - 1
    else:
        stable_end = last_frame

    if incipient_start is not None:
        incipient_end = (translational_start - 1) if translational_start is not None else last_frame

    summary = TrialSummary(
        trial_folder=trial_dir.name,
        frame_count=len(frame_paths),
        stable_range=range_text(2, stable_end),
        incipient_range=range_text(incipient_start, incipient_end),
        translational_range=range_text(translational_start, last_frame),
        incipient_start=incipient_start,
        translational_start=translational_start,
        note=(
            "frame_01 ignored; labels use marker motion, central rigid-body fit, "
            "peripheral residual, and global direction consistency"
        ),
    )

    return labels, metrics, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    trial_dirs = sorted([p for p in SESSION_DIR.iterdir() if p.is_dir() and p.name.startswith("trial_")])
    if not trial_dirs:
        raise FileNotFoundError(f"No trial folders found in {SESSION_DIR}")

    all_labels: list[dict[str, object]] = []
    all_training_labels: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for index, trial_dir in enumerate(trial_dirs, start=1):
        print(f"[{index:03d}/{len(trial_dirs):03d}] analyzing {trial_dir.name}")
        labels, metrics, summary = analyze_trial(trial_dir)

        write_csv(PER_TRIAL_DIR / f"{trial_dir.name}_labels.csv", labels)
        write_csv(METRICS_DIR / f"{trial_dir.name}_metrics.csv", metrics)

        all_labels.extend(labels)
        all_metrics.extend(metrics)
        all_training_labels.extend([row for row in labels if int(row["phase"]) >= 0])
        summaries.append(summary.__dict__)

    # Add trial-level split: first 72 trials train, remaining trials validation.
    train_trials = {p.name for p in trial_dirs[:72]}
    for row in all_training_labels:
        row["split"] = "train" if row["trial_folder"] in train_trials else "val"

    write_csv(OUTPUT_DIR / "all_90_trials_frame_labels_including_ignore.csv", all_labels)
    write_csv(OUTPUT_DIR / "all_90_trials_training_labels_excluding_frame01.csv", all_training_labels)
    write_csv(OUTPUT_DIR / "all_90_trials_motion_metrics.csv", all_metrics)
    write_csv(OUTPUT_DIR / "trial_phase_summary.csv", summaries)

    config = {
        "session_dir": str(SESSION_DIR),
        "output_dir": str(OUTPUT_DIR),
        "method": "marker motion + central rigid-body fit + peripheral residual + direction consistency",
        "labels": LABELS,
        "frame_01_policy": "excluded from training and labeled -1 because it is often contact/capture settling",
        "split": "first 72 trial folders train, remaining 18 validation",
    }
    with (OUTPUT_DIR / "labeling_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Per-trial CSV folder: {PER_TRIAL_DIR}")
    print(f"Training label CSV: {OUTPUT_DIR / 'all_90_trials_training_labels_excluding_frame01.csv'}")
    print(f"Summary CSV: {OUTPUT_DIR / 'trial_phase_summary.csv'}")
    print(f"Metrics CSV: {OUTPUT_DIR / 'all_90_trials_motion_metrics.csv'}")


if __name__ == "__main__":
    main()
