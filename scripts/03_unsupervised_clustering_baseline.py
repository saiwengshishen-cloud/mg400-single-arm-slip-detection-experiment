from __future__ import annotations

import itertools
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Kaggle paths
# ============================================================

DATA_DIR = Path(
    "/kaggle/input/datasets/shishensaiweng/kaggle-flat-90trials-images-and-labels"
)
GENERATED_LABEL_CSV = Path(
    "/kaggle/working/generated_rigid_residual_labels_from_images_only/"
    "generated_training_labels_excluding_frame01.csv"
)
ANNOTATION_CSV = GENERATED_LABEL_CSV if GENERATED_LABEL_CSV.exists() else DATA_DIR / "flat_training_labels.csv"
OUTPUT_DIR = Path("/kaggle/working")

N_CLUSTERS = 3
RANDOM_STATE = 42

# This controls whether frame number is used as one feature.
# For a fair image/motion-based comparison, keep it False.
USE_FRAME_NUMBER_FEATURE = False


PHASE_NAMES = {
    0: "stable",
    1: "incipient_slip",
    2: "translational_slip",
}


# ============================================================
# 2. Image and marker feature functions
# ============================================================

def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def contact_mask(arr: np.ndarray) -> np.ndarray:
    """
    Keep the tactile contact imprint and suppress the white background.
    """
    rough = arr < 225
    ys, xs = np.where(rough)
    if len(xs) < 50:
        return np.ones(arr.shape, dtype=bool)

    cx = float(xs.mean())
    cy = float(ys.mean())
    radii = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    radius = float(np.percentile(radii, 88))

    yy, xx = np.indices(arr.shape)
    return ((xx - cx) ** 2 + (yy - cy) ** 2) < (radius * 0.88) ** 2


def marker_centers(arr: np.ndarray, mask: np.ndarray, max_points: int = 180) -> np.ndarray:
    """
    Detect bright marker centers using local maxima.
    Returns [N, 2] with columns x, y.
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

    selected = []
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
    """
    if len(ref_points) == 0 or len(cur_points) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    candidates = []
    for i, p in enumerate(ref_points):
        d = np.sqrt(((cur_points - p) ** 2).sum(axis=1))
        j = int(np.argmin(d))
        if float(d[j]) <= max_distance:
            candidates.append((float(d[j]), i, j))

    candidates.sort(key=lambda x: x[0])
    used_ref = set()
    used_cur = set()
    matched_ref = []
    matched_cur = []

    for _, i, j in candidates:
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

    The residual after this fit is treated as local slip evidence.
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


def direction_consistency(displacements: np.ndarray) -> float:
    """
    Near 1 means most markers move in the same direction.
    This usually corresponds to gross/translational slip.
    """
    if len(displacements) < 3:
        return np.nan

    mags = np.linalg.norm(displacements, axis=1)
    valid = mags > 1e-6
    if valid.sum() < 3:
        return np.nan

    unit = displacements[valid] / mags[valid, None]
    return float(np.linalg.norm(unit.mean(axis=0)))


def central_marker_mask(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    cx = w / 2.0
    cy = h / 2.0
    dist = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    radius = max(1.0, np.percentile(dist, 70))
    return dist <= radius * 0.58


def extract_features_for_trial(trial_df: pd.DataFrame) -> list[dict[str, float | str | int]]:
    """
    For one trial, use frame_02 as reference and compute unsupervised features.
    """
    trial_df = trial_df.copy()
    trial_df["frame_number"] = trial_df["frame_id"].str.extract(r"(\d+)").astype(int)
    trial_df = trial_df.sort_values("frame_number")

    reference_rows = trial_df[trial_df["frame_number"] == 2]
    if len(reference_rows) == 0:
        reference_rows = trial_df.iloc[[0]]

    ref_row = reference_rows.iloc[0]
    ref_img = load_gray(DATA_DIR / ref_row["image_file"])
    mask = contact_mask(ref_img)
    ref_markers = marker_centers(ref_img, mask)

    rows = []
    ref_vals = ref_img[mask]

    for _, row in trial_df.iterrows():
        image_path = DATA_DIR / row["image_file"]
        img = load_gray(image_path)

        cur_vals = img[mask]
        diff = cur_vals - ref_vals
        abs_diff = np.abs(diff)

        cur_markers = marker_centers(img, mask)
        matched_ref, matched_cur = match_points(ref_markers, cur_markers)

        if len(matched_ref) >= 3:
            displacement = matched_cur - matched_ref
            displacement_mag = np.linalg.norm(displacement, axis=1)
            global_motion_median = float(np.median(displacement_mag))
            global_motion_mean = float(np.mean(displacement_mag))
            global_motion_p90 = float(np.percentile(displacement_mag, 90))
            direction_score = direction_consistency(displacement)

            center_mask = central_marker_mask(matched_ref, ref_img.shape)
            if center_mask.sum() >= 3:
                fit_src = matched_ref[center_mask]
                fit_dst = matched_cur[center_mask]
            else:
                fit_src = matched_ref
                fit_dst = matched_cur

            matrix, translation = fit_similarity_transform(fit_src, fit_dst)
            predicted = matched_ref @ matrix.T + translation
            residual = np.linalg.norm(matched_cur - predicted, axis=1)

            central_residual = (
                float(np.median(residual[center_mask]))
                if center_mask.any()
                else float(np.median(residual))
            )
            peripheral_mask = ~center_mask
            peripheral_residual = (
                float(np.median(residual[peripheral_mask]))
                if peripheral_mask.any()
                else float(np.median(residual))
            )
            rigid_residual = float(np.median(residual))
            residual_ratio = float(peripheral_residual / max(central_residual, 0.15))
        else:
            global_motion_median = np.nan
            global_motion_mean = np.nan
            global_motion_p90 = np.nan
            direction_score = np.nan
            rigid_residual = np.nan
            central_residual = np.nan
            peripheral_residual = np.nan
            residual_ratio = np.nan

        feature_row = {
            "image_file": row["image_file"],
            "trial_folder": row["trial_folder"],
            "frame_id": row["frame_id"],
            "frame_number": int(row["frame_number"]),
            "split": row.get("split", "unknown"),
            "phase": int(row["phase"]) if "phase" in row else -999,
            "phase_name": row.get("phase_name", ""),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "std_abs_diff": float(np.std(abs_diff)),
            "p90_abs_diff": float(np.percentile(abs_diff, 90)),
            "p99_abs_diff": float(np.percentile(abs_diff, 99)),
            "mean_signed_diff": float(np.mean(diff)),
            "image_intensity_mean": float(np.mean(cur_vals)),
            "image_intensity_std": float(np.std(cur_vals)),
            "matched_markers": int(len(matched_ref)),
            "global_motion_median_px": global_motion_median,
            "global_motion_mean_px": global_motion_mean,
            "global_motion_p90_px": global_motion_p90,
            "direction_consistency": direction_score,
            "rigid_residual_px": rigid_residual,
            "central_residual_px": central_residual,
            "peripheral_residual_px": peripheral_residual,
            "peripheral_to_central_residual_ratio": residual_ratio,
        }

        if USE_FRAME_NUMBER_FEATURE:
            feature_row["frame_number_norm"] = float(row["frame_number"]) / 40.0

        rows.append(feature_row)

    return rows


# ============================================================
# 3. Cluster-label alignment for evaluation only
# ============================================================

def best_cluster_to_phase_mapping(y_true: np.ndarray, y_cluster: np.ndarray) -> dict[int, int]:
    """
    Cluster IDs are arbitrary. This finds the best mapping from cluster id
    to phase id only for evaluation/printing.
    """
    cluster_ids = sorted(np.unique(y_cluster).tolist())
    phase_ids = sorted(np.unique(y_true).tolist())

    best_score = -1
    best_mapping = {}

    for perm in itertools.permutations(phase_ids, len(cluster_ids)):
        mapping = dict(zip(cluster_ids, perm))
        mapped = np.asarray([mapping[c] for c in y_cluster])
        score = int((mapped == y_true).sum())
        if score > best_score:
            best_score = score
            best_mapping = mapping

    return best_mapping


def evaluate_clusters(name: str, df: pd.DataFrame, cluster_col: str, phase_col: str = "phase") -> None:
    """
    Evaluate cluster results against labels.
    Important: labels are not used for clustering, only for this evaluation.
    """
    valid_df = df[df[phase_col].isin([0, 1, 2])].copy()
    if len(valid_df) == 0:
        print(f"\n{name}: no labels available for evaluation.")
        return

    y_true = valid_df[phase_col].to_numpy(dtype=int)
    y_cluster = valid_df[cluster_col].to_numpy(dtype=int)
    mapping = best_cluster_to_phase_mapping(y_true, y_cluster)
    y_pred = np.asarray([mapping[c] for c in y_cluster])

    acc = float((y_pred == y_true).mean())
    ari = adjusted_rand_score(y_true, y_cluster)
    nmi = normalized_mutual_info_score(y_true, y_cluster)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print(f"\n{name}")
    print("Cluster -> phase mapping:", mapping)
    print("Accuracy after best mapping:", acc)
    print("ARI:", ari)
    print("NMI:", nmi)
    print("Confusion matrix rows=true phase, cols=mapped cluster phase:")
    print(cm)


def add_mapped_phase_columns(df: pd.DataFrame, cluster_col: str, prefix: str) -> pd.DataFrame:
    y_true = df["phase"].to_numpy(dtype=int)
    y_cluster = df[cluster_col].to_numpy(dtype=int)
    mapping = best_cluster_to_phase_mapping(y_true, y_cluster)

    mapped_phase = np.asarray([mapping[c] for c in y_cluster])
    df[f"{prefix}_mapped_phase"] = mapped_phase
    df[f"{prefix}_mapped_phase_name"] = [PHASE_NAMES[int(x)] for x in mapped_phase]
    return df


# ============================================================
# 4. Main unsupervised pipeline
# ============================================================

print("DATA_DIR exists:", DATA_DIR.exists())
print("ANNOTATION_CSV exists:", ANNOTATION_CSV.exists())

df = pd.read_csv(ANNOTATION_CSV)
print(df.head())
print("Rows:", len(df))
print("Split counts:")
print(df["split"].value_counts())
print("Phase counts:")
print(df["phase"].value_counts().sort_index())

all_feature_rows = []
for trial_name, trial_df in df.groupby("trial_folder", sort=True):
    print("Extracting features:", trial_name)
    all_feature_rows.extend(extract_features_for_trial(trial_df))

features_df = pd.DataFrame(all_feature_rows)
features_path = OUTPUT_DIR / "unsupervised_motion_features.csv"
features_df.to_csv(features_path, index=False)
print("Saved features to:", features_path)

feature_cols = [
    "mean_abs_diff",
    "std_abs_diff",
    "p90_abs_diff",
    "p99_abs_diff",
    "mean_signed_diff",
    "image_intensity_mean",
    "image_intensity_std",
    "matched_markers",
    "global_motion_median_px",
    "global_motion_mean_px",
    "global_motion_p90_px",
    "direction_consistency",
    "rigid_residual_px",
    "central_residual_px",
    "peripheral_residual_px",
    "peripheral_to_central_residual_ratio",
]

if USE_FRAME_NUMBER_FEATURE:
    feature_cols.append("frame_number_norm")

# Fill occasional failed marker measurements with column medians.
X_raw = features_df[feature_cols].copy()
X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

train_mask = features_df["split"].eq("train").to_numpy()
val_mask = features_df["split"].eq("val").to_numpy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_raw.loc[train_mask])
X_all_scaled = scaler.transform(X_raw)

pca_components = min(8, X_train_scaled.shape[1])
pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_all_pca = pca.transform(X_all_scaled)

print("PCA explained variance ratio:")
print(pca.explained_variance_ratio_)
print("PCA total explained variance:", float(pca.explained_variance_ratio_.sum()))

# KMeans clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=50)
kmeans.fit(X_train_pca)
features_df["kmeans_cluster"] = kmeans.predict(X_all_pca)

# Gaussian Mixture clustering
gmm = GaussianMixture(
    n_components=N_CLUSTERS,
    covariance_type="full",
    random_state=RANDOM_STATE,
    n_init=10,
)
gmm.fit(X_train_pca)
features_df["gmm_cluster"] = gmm.predict(X_all_pca)

# Add PCA coordinates for plotting.
features_df["pca_1"] = X_all_pca[:, 0]
features_df["pca_2"] = X_all_pca[:, 1]

# Evaluation: labels are used here only as external validation.
evaluate_clusters("KMeans train", features_df[train_mask], "kmeans_cluster")
evaluate_clusters("KMeans val", features_df[val_mask], "kmeans_cluster")
evaluate_clusters("GMM train", features_df[train_mask], "gmm_cluster")
evaluate_clusters("GMM val", features_df[val_mask], "gmm_cluster")

features_df = add_mapped_phase_columns(features_df, "kmeans_cluster", "kmeans")
features_df = add_mapped_phase_columns(features_df, "gmm_cluster", "gmm")

result_path = OUTPUT_DIR / "unsupervised_slip_clustering_results.csv"
features_df.to_csv(result_path, index=False)
print("Saved clustering results to:", result_path)

joblib.dump(
    {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "gmm": gmm,
        "phase_names": PHASE_NAMES,
        "use_frame_number_feature": USE_FRAME_NUMBER_FEATURE,
    },
    OUTPUT_DIR / "unsupervised_slip_clustering_models.joblib",
)
print("Saved clustering models to:", OUTPUT_DIR / "unsupervised_slip_clustering_models.joblib")


# ============================================================
# 5. Simple PCA visualization
# ============================================================

def save_scatter(color_col: str, title: str, filename: str) -> None:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        features_df["pca_1"],
        features_df["pca_2"],
        c=features_df[color_col],
        s=12,
        alpha=0.75,
        cmap="viridis",
    )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.show()


save_scatter("phase", "True labels, used only for evaluation", "pca_true_phase.png")
save_scatter("kmeans_cluster", "KMeans clusters", "pca_kmeans_clusters.png")
save_scatter("gmm_cluster", "GMM clusters", "pca_gmm_clusters.png")

print("\nDone.")
print("Main output files:")
print(OUTPUT_DIR / "unsupervised_motion_features.csv")
print(OUTPUT_DIR / "unsupervised_slip_clustering_results.csv")
print(OUTPUT_DIR / "unsupervised_slip_clustering_models.joblib")
print(OUTPUT_DIR / "pca_true_phase.png")
print(OUTPUT_DIR / "pca_kmeans_clusters.png")
print(OUTPUT_DIR / "pca_gmm_clusters.png")
