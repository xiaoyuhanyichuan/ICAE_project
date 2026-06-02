from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeIndex:
    mode: str
    frame: pd.DataFrame
    representative_days: list[pd.Timestamp]
    day_weights: pd.Series


def _prepare_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    if "timestamp_hour_utc" not in hourly.columns:
        raise ValueError("hourly data must include timestamp_hour_utc")
    frame = hourly.copy()
    frame["timestamp_hour_utc"] = pd.to_datetime(frame["timestamp_hour_utc"], utc=True)
    frame = frame.sort_values("timestamp_hour_utc").reset_index(drop=True)
    return frame


def _available_features(hourly: pd.DataFrame, requested: list[str]) -> list[str]:
    if not requested:
        raise ValueError(
            "No typical-day feature columns were requested."
        )
    missing = [col for col in requested if col not in hourly.columns]
    if missing:
        raise ValueError(
            "Missing requested typical-day feature columns: "
            f"{missing}"
        )
    return requested


def _complete_day_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    frame["_day"] = frame["timestamp_hour_utc"].dt.floor("D")
    complete_days = []
    for day, day_frame in frame.groupby("_day", sort=True):
        timestamps = pd.DatetimeIndex(day_frame["timestamp_hour_utc"]).sort_values()
        expected = pd.date_range(day, periods=24, freq="h")
        if len(day_frame) == 24 and timestamps.equals(expected):
            complete_days.append(day)
    if not complete_days:
        raise ValueError("No complete 24-hour days available for typical-day generation.")
    return frame[frame["_day"].isin(complete_days)].copy()


def _daily_feature_matrix(
    hourly: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.Index, np.ndarray]:
    available = _available_features(hourly, feature_columns)
    complete = _complete_day_frame(hourly)
    days = []
    vectors = []
    for day, day_frame in complete.groupby("_day", sort=True):
        day_frame = day_frame.sort_values("timestamp_hour_utc")
        vector_parts = []
        for col in available:
            values = pd.to_numeric(day_frame[col], errors="coerce").to_numpy(dtype=float)
            vector_parts.append(values)
        vector = np.concatenate(vector_parts)
        if not np.isfinite(vector).all():
            raise ValueError("Typical-day feature columns must be numeric and finite.")
        days.append(day)
        vectors.append(vector)
    matrix = np.vstack(vectors)
    return pd.Index(days), _standardize(matrix)


def _standardize(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    safe_scales = np.where(scales == 0.0, 1.0, scales)
    return (matrix - means) / safe_scales


def _squared_distances(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    delta = matrix[:, None, :] - centroids[None, :, :]
    return np.sum(delta * delta, axis=2)


def _repair_empty_clusters(
    matrix: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    empty_clusters: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not empty_clusters:
        return labels, centroids
    counts = np.bincount(labels, minlength=len(centroids))
    for cluster_id in empty_clusters:
        distances = _squared_distances(matrix, centroids)
        nearest_distance = distances.min(axis=1)
        order = np.lexsort((np.arange(len(matrix)), -nearest_distance))
        chosen = next(idx for idx in order if counts[labels[int(idx)]] > 1)
        chosen = int(chosen)
        counts[labels[chosen]] -= 1
        labels[chosen] = cluster_id
        counts[cluster_id] += 1
        centroids[cluster_id] = matrix[chosen]
    return labels, centroids


def _centroids_from_labels(matrix: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    updated = centroids.copy()
    for cluster_id in range(len(centroids)):
        members = matrix[labels == cluster_id]
        if len(members) == 0:
            raise ValueError("Internal k-means error: empty cluster after repair.")
        updated[cluster_id] = members.mean(axis=0)
    return updated


def _kmeans(
    matrix: np.ndarray,
    k: int,
    random_seed: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    if k < 1:
        raise ValueError("typical_days.k must be at least 1")
    if len(matrix) < k:
        raise ValueError(f"Need at least {k} complete days, found {len(matrix)}.")
    rng = np.random.default_rng(random_seed)
    centroids = matrix[rng.choice(len(matrix), size=k, replace=False)].copy()
    labels = np.full(len(matrix), -1, dtype=int)
    for _ in range(max_iter):
        new_labels = _squared_distances(matrix, centroids).argmin(axis=1)
        labels = new_labels
        updated = centroids.copy()
        empty_clusters = []
        for cluster_id in range(k):
            members = matrix[labels == cluster_id]
            if len(members) == 0:
                empty_clusters.append(cluster_id)
            else:
                updated[cluster_id] = members.mean(axis=0)
        repaired_labels, repaired_centroids = _repair_empty_clusters(
            matrix,
            labels,
            updated,
            empty_clusters,
        )
        repaired_centroids = _centroids_from_labels(matrix, repaired_labels, repaired_centroids)
        if np.array_equal(labels, repaired_labels) and np.allclose(centroids, repaired_centroids):
            labels = repaired_labels
            centroids = repaired_centroids
            break
        labels = repaired_labels
        centroids = repaired_centroids
    return labels, centroids


def _select_representative_days(
    days: pd.Index,
    matrix: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> tuple[list[pd.Timestamp], pd.Series]:
    selected_by_cluster = {}
    weights_by_day = {}
    for cluster_id in range(len(centroids)):
        member_positions = np.flatnonzero(labels == cluster_id)
        if len(member_positions) == 0:
            continue
        distances = _squared_distances(matrix[member_positions], centroids[[cluster_id]])[:, 0]
        nearest_position = int(member_positions[int(distances.argmin())])
        selected_day = pd.Timestamp(days[nearest_position])
        selected_by_cluster[cluster_id] = selected_day
        weights_by_day[selected_day] = int(len(member_positions))
    representative_days = sorted(selected_by_cluster.values())
    day_weights = pd.Series(
        [weights_by_day[day] for day in representative_days],
        index=pd.Index(representative_days, name="representative_day"),
        name="day_weight",
        dtype="int64",
    )
    return representative_days, day_weights


def _append_peak_day(
    representative_days: list[pd.Timestamp],
    day_weights: pd.Series,
    days: pd.Index,
    labels: np.ndarray,
    hourly: pd.DataFrame,
    peak_column: str,
) -> tuple[list[pd.Timestamp], pd.Series]:
    if peak_column not in hourly.columns:
        raise ValueError(f"Configured peak_column {peak_column!r} is missing from hourly data.")
    complete = _complete_day_frame(hourly)
    daily_peak = complete.groupby("_day", sort=True)[peak_column].max()
    peak_day = pd.Timestamp(daily_peak.astype(float).idxmax())
    if peak_day in set(representative_days):
        return representative_days, day_weights

    try:
        day_position = list(pd.Index(days)).index(peak_day)
    except ValueError as exc:
        raise ValueError("Peak day was not found among complete typical-day candidates.") from exc

    cluster_id = int(labels[day_position])
    cluster_days = [pd.Timestamp(days[idx]) for idx in np.flatnonzero(labels == cluster_id)]
    original_rep = next((day for day in representative_days if day in cluster_days), None)
    if original_rep is None:
        raise ValueError("Internal typical-day error: peak day cluster has no representative.")

    updated_weights = day_weights.copy()
    updated_weights.loc[original_rep] = int(updated_weights.loc[original_rep]) - 1
    updated_weights.loc[peak_day] = 1
    updated_weights = updated_weights[updated_weights > 0]
    representative_days = sorted(pd.Timestamp(day) for day in updated_weights.index)
    updated_weights = updated_weights.reindex(representative_days).astype("int64")
    updated_weights.index = pd.Index(representative_days, name="representative_day")
    updated_weights.name = "day_weight"
    return representative_days, updated_weights


def _full_timeseries(hourly: pd.DataFrame) -> TimeIndex:
    days = sorted(hourly["timestamp_hour_utc"].dt.floor("D").unique().tolist())
    day_weights = pd.Series(
        [1] * len(days),
        index=pd.Index(days, name="representative_day"),
        name="day_weight",
        dtype="int64",
    )
    return TimeIndex(
        mode="full_timeseries",
        frame=hourly.copy(),
        representative_days=[pd.Timestamp(day) for day in days],
        day_weights=day_weights,
    )


def build_time_index(hourly: pd.DataFrame, typical_config: dict) -> TimeIndex:
    frame = _prepare_hourly(hourly)
    if not typical_config.get("enabled", True):
        return _full_timeseries(frame)

    feature_columns = list(typical_config.get("feature_columns", []))
    days, matrix = _daily_feature_matrix(frame, feature_columns)
    labels, centroids = _kmeans(
        matrix,
        k=int(typical_config["k"]),
        random_seed=int(typical_config.get("random_seed", 0)),
    )
    representative_days, day_weights = _select_representative_days(
        days,
        matrix,
        labels,
        centroids,
    )
    if bool(typical_config.get("append_peak_day", False)):
        representative_days, day_weights = _append_peak_day(
            representative_days,
            day_weights,
            days,
            labels,
            frame,
            str(typical_config.get("peak_column", "it_load_kw")),
        )
    selected = frame[frame["timestamp_hour_utc"].dt.floor("D").isin(representative_days)].copy()
    selected["_representative_day"] = selected["timestamp_hour_utc"].dt.floor("D")
    day_to_id = {day: idx for idx, day in enumerate(representative_days)}
    selected["representative_day_id"] = selected["_representative_day"].map(day_to_id).astype(int)
    selected["day_weight"] = selected["_representative_day"].map(day_weights).astype(int)
    selected = selected.drop(columns=["_representative_day"]).reset_index(drop=True)
    return TimeIndex(
        mode="typical_days",
        frame=selected,
        representative_days=representative_days,
        day_weights=day_weights,
    )
