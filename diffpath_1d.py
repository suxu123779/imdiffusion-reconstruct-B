import json
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import ensure_outputs_can_be_written


DEFAULT_KDE_BANDWIDTHS = (0.05, 0.1, 0.2, 0.5, 1.0)
DEFAULT_KDE_BANDWIDTHS_6D = (0.2, 0.5, 1.0, 2.0, 5.0)
DEFAULT_GMM_COMPONENTS_6D = (2, 4, 8, 16)
DEFAULT_GMM_COVARIANCE_TYPES_6D = ("diag", "full")
DIFFPATH_6D_FEATURE_NAMES = (
    "eps_sum",
    "eps_sum_sq",
    "eps_sum_cb",
    "deps_dt",
    "deps_dt_sq",
    "deps_dt_cb",
)


def set_all_seeds(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_for_save(base_seed, save_id):
    suffix = ""
    for character in reversed(str(save_id)):
        if not character.isdigit():
            break
        suffix = character + suffix
    return int(base_seed) + (int(suffix) if suffix else 0)


class DiffPathNormalWindowDataset(Dataset):
    def __init__(self, train_path, window_length=100, stride=50):
        with open(train_path, "rb") as f:
            data = np.asarray(pickle.load(f), dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(
                f"normal training data must be [time, feature], got {data.shape}"
            )
        if len(data) <= int(window_length):
            raise ValueError(
                f"normal training data length {len(data)} must exceed "
                f"window length {window_length}"
            )

        self.data = torch.from_numpy(data) * 20.0
        self.window_length = int(window_length)
        self.begin_indexes = list(
            range(0, len(self.data) - self.window_length, int(stride))
        )
        if not self.begin_indexes:
            raise ValueError(f"no normal windows could be built from {train_path}")

    def __len__(self):
        return len(self.begin_indexes)

    def __getitem__(self, item):
        begin = self.begin_indexes[item]
        observed_data = self.data[begin : begin + self.window_length]
        observed_mask = torch.ones_like(observed_data)
        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": torch.zeros_like(observed_data),
            "timepoints": np.arange(self.window_length),
            "strategy_type": 0,
        }


def build_normal_loader(
    train_path,
    batch_size=24,
    window_length=100,
    split=4,
):
    left = int(window_length) // int(split)
    stride = int(window_length) - 2 * left
    if stride <= 0:
        raise ValueError(
            f"invalid DiffPath window stride {stride} for "
            f"window_length={window_length}, split={split}"
        )
    dataset = DiffPathNormalWindowDataset(
        train_path,
        window_length=window_length,
        stride=stride,
    )
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False)


def backproject_windows_to_time(window_scores, split):
    if window_scores.dim() != 2:
        raise ValueError(
            f"window scores must be [num_windows, length], got "
            f"{list(window_scores.shape)}"
        )
    num_windows, length = window_scores.shape
    if num_windows <= 0:
        raise ValueError("window scores are empty")

    left = length // int(split)
    right = length - left
    stride = right - left
    score_parts = [window_scores[0, :left].detach().cpu().reshape(-1)]
    score_parts.append(
        window_scores[:, left:right].detach().cpu().reshape(-1)
    )
    scores = torch.cat(score_parts, dim=0).float()

    index_parts = [torch.arange(0, left, dtype=torch.long)]
    index_parts.extend(
        torch.arange(
            window_index * stride + left,
            window_index * stride + right,
            dtype=torch.long,
        )
        for window_index in range(num_windows)
    )
    valid_indices = torch.cat(index_parts, dim=0)
    if scores.numel() != valid_indices.numel():
        raise RuntimeError(
            f"DiffPath score/index length mismatch: "
            f"{scores.numel()} vs {valid_indices.numel()}"
        )
    return (
        scores.numpy().astype(np.float32),
        valid_indices.numpy().astype(np.int64),
    )


def backproject_feature_windows_to_time(window_features, split):
    if window_features.dim() != 3:
        raise ValueError(
            f"window features must be [num_windows, dim, length], got "
            f"{list(window_features.shape)}"
        )
    num_windows, feature_dim, length = window_features.shape
    if num_windows <= 0:
        raise ValueError("window features are empty")

    left = length // int(split)
    right = length - left
    stride = right - left
    feature_parts = [
        window_features[0, :, :left]
        .detach()
        .cpu()
        .transpose(0, 1)
        .reshape(-1, feature_dim)
    ]
    feature_parts.append(
        window_features[:, :, left:right]
        .detach()
        .cpu()
        .permute(0, 2, 1)
        .reshape(-1, feature_dim)
    )
    features = torch.cat(feature_parts, dim=0).float()

    index_parts = [torch.arange(0, left, dtype=torch.long)]
    index_parts.extend(
        torch.arange(
            window_index * stride + left,
            window_index * stride + right,
            dtype=torch.long,
        )
        for window_index in range(num_windows)
    )
    valid_indices = torch.cat(index_parts, dim=0)
    if features.shape[0] != valid_indices.numel():
        raise RuntimeError(
            f"DiffPath feature/index length mismatch: "
            f"{features.shape[0]} vs {valid_indices.numel()}"
        )
    return (
        features.numpy().astype(np.float32),
        valid_indices.numpy().astype(np.int64),
    )


def collect_diffpath_scores(
    model,
    loader,
    num_path_steps,
    split,
    desc,
):
    recon_windows = []
    diffpath_windows = []
    diffpath_6d_windows = []
    resolved_timesteps = None
    model.eval()

    with torch.no_grad():
        with tqdm(loader, desc=desc, mininterval=5.0, maxinterval=50.0) as it:
            for batch in it:
                output = model.diffpath_evaluate(
                    batch,
                    n_samples=1,
                    num_path_steps=num_path_steps,
                    return_moments=True,
                )
                samples, observed_data, _, _, _, diffpath_result = output
                reconstruction = samples.median(dim=1).values
                recon_score = torch.abs(
                    reconstruction - observed_data
                ).sum(dim=1)
                diffpath_score = diffpath_result[
                    "diffpath_1d_statistic"
                ]
                diffpath_6d = diffpath_result[
                    "diffpath_6d_moment_sums"
                ]

                if recon_score.shape != diffpath_score.shape:
                    raise RuntimeError(
                        f"reconstruction and DiffPath window shapes differ: "
                        f"{list(recon_score.shape)} vs "
                        f"{list(diffpath_score.shape)}"
                    )
                if (
                    diffpath_6d.dim() != 3
                    or diffpath_6d.shape[0] != recon_score.shape[0]
                    or diffpath_6d.shape[2] != recon_score.shape[1]
                    or diffpath_6d.shape[1] != len(DIFFPATH_6D_FEATURE_NAMES)
                ):
                    raise RuntimeError(
                        f"DiffPath-6D window shape is invalid: "
                        f"{list(diffpath_6d.shape)}"
                    )
                if resolved_timesteps is None:
                    resolved_timesteps = [
                        int(step)
                        for step in diffpath_result["diffpath_timesteps"]
                    ]
                recon_windows.append(recon_score.detach().cpu().float())
                diffpath_windows.append(
                    diffpath_score.detach().cpu().float()
                )
                diffpath_6d_windows.append(
                    diffpath_6d.detach().cpu().float()
                )

    if not recon_windows:
        raise ValueError(f"no windows were collected for {desc}")
    recon_time, recon_indices = backproject_windows_to_time(
        torch.cat(recon_windows, dim=0),
        split,
    )
    diffpath_time, diffpath_indices = backproject_windows_to_time(
        torch.cat(diffpath_windows, dim=0),
        split,
    )
    diffpath_6d_time, diffpath_6d_indices = (
        backproject_feature_windows_to_time(
            torch.cat(diffpath_6d_windows, dim=0),
            split,
        )
    )
    if not np.array_equal(recon_indices, diffpath_indices):
        raise RuntimeError(
            "reconstruction and DiffPath valid indices are different"
        )
    if not np.array_equal(recon_indices, diffpath_6d_indices):
        raise RuntimeError(
            "reconstruction and DiffPath-6D valid indices are different"
        )
    if not np.isfinite(recon_time).all():
        raise RuntimeError("reconstruction scores contain NaN/Inf")
    if not np.isfinite(diffpath_time).all():
        raise RuntimeError("DiffPath statistics contain NaN/Inf")
    if not np.isfinite(diffpath_6d_time).all():
        raise RuntimeError("DiffPath-6D features contain NaN/Inf")
    return (
        recon_time,
        diffpath_time,
        diffpath_6d_time,
        recon_indices,
        resolved_timesteps,
    )


def collect_reconstruction_scores(
    model,
    loader,
    split,
    desc,
    n_samples=1,
):
    recon_sum_windows = []
    recon_max_windows = []
    model.eval()

    with torch.no_grad():
        with tqdm(loader, desc=desc, mininterval=5.0, maxinterval=50.0) as it:
            for batch in it:
                (
                    observed_data,
                    observed_mask,
                    observed_tp,
                    _gt_mask,
                    _for_pattern_mask,
                    _cut_length,
                    strategy_type,
                ) = model.process_data(batch)
                if model.task_mode != "reconstruction":
                    raise RuntimeError(
                        "DiffPath reconstruction scoring requires "
                        "reconstruction checkpoints"
                    )
                cond_mask = torch.zeros_like(observed_mask)
                strategy_type = torch.zeros_like(strategy_type)
                side_info = model.get_side_info(observed_tp, cond_mask)
                samples = model.reconstruct(
                    observed_data,
                    cond_mask,
                    side_info,
                    int(n_samples),
                    strategy_type,
                )
                reconstruction = samples.median(dim=1).values
                recon_error = torch.abs(
                    reconstruction - observed_data
                )
                recon_sum_score = recon_error.sum(dim=1)
                recon_max_score = recon_error.max(dim=1).values
                recon_sum_windows.append(
                    recon_sum_score.detach().cpu().float()
                )
                recon_max_windows.append(
                    recon_max_score.detach().cpu().float()
                )

    if not recon_sum_windows:
        raise ValueError(f"no reconstruction windows were collected for {desc}")
    recon_sum_time, recon_indices = backproject_windows_to_time(
        torch.cat(recon_sum_windows, dim=0),
        split,
    )
    recon_max_time, recon_max_indices = backproject_windows_to_time(
        torch.cat(recon_max_windows, dim=0),
        split,
    )
    if not np.array_equal(recon_indices, recon_max_indices):
        raise RuntimeError("reconstruction sum/max valid indices differ")
    if not np.isfinite(recon_sum_time).all():
        raise RuntimeError("reconstruction sum scores contain NaN/Inf")
    if not np.isfinite(recon_max_time).all():
        raise RuntimeError("reconstruction max scores contain NaN/Inf")
    return (
        recon_sum_time,
        recon_max_time,
        recon_indices,
    )


def collect_diffpath_statistics(
    model,
    loader,
    num_path_steps,
    split,
    desc,
):
    diffpath_windows = []
    diffpath_6d_windows = []
    resolved_timesteps = None
    model.eval()

    with torch.no_grad():
        with tqdm(loader, desc=desc, mininterval=5.0, maxinterval=50.0) as it:
            for batch in it:
                (
                    observed_data,
                    observed_mask,
                    observed_tp,
                    _gt_mask,
                    _for_pattern_mask,
                    _cut_length,
                    strategy_type,
                ) = model.process_data(batch)
                if model.task_mode != "reconstruction":
                    raise RuntimeError(
                        "DiffPath statistics require reconstruction checkpoints"
                    )
                cond_mask = torch.zeros_like(observed_mask)
                strategy_type = torch.zeros_like(strategy_type)
                side_info = model.get_side_info(observed_tp, cond_mask)
                diffpath_result = model.compute_diffpath_1d(
                    observed_data,
                    cond_mask,
                    side_info,
                    strategy_type,
                    num_path_steps=num_path_steps,
                    return_moments=True,
                )
                diffpath_score = diffpath_result[
                    "diffpath_1d_statistic"
                ]
                diffpath_6d = diffpath_result[
                    "diffpath_6d_moment_sums"
                ]
                if (
                    diffpath_6d.dim() != 3
                    or diffpath_6d.shape[0] != diffpath_score.shape[0]
                    or diffpath_6d.shape[2] != diffpath_score.shape[1]
                    or diffpath_6d.shape[1] != len(DIFFPATH_6D_FEATURE_NAMES)
                ):
                    raise RuntimeError(
                        f"DiffPath-6D window shape is invalid: "
                        f"{list(diffpath_6d.shape)}"
                    )
                if resolved_timesteps is None:
                    resolved_timesteps = [
                        int(step)
                        for step in diffpath_result["diffpath_timesteps"]
                    ]
                diffpath_windows.append(
                    diffpath_score.detach().cpu().float()
                )
                diffpath_6d_windows.append(
                    diffpath_6d.detach().cpu().float()
                )

    if not diffpath_windows:
        raise ValueError(f"no DiffPath windows were collected for {desc}")
    diffpath_time, diffpath_indices = backproject_windows_to_time(
        torch.cat(diffpath_windows, dim=0),
        split,
    )
    diffpath_6d_time, diffpath_6d_indices = (
        backproject_feature_windows_to_time(
            torch.cat(diffpath_6d_windows, dim=0),
            split,
        )
    )
    if not np.array_equal(diffpath_indices, diffpath_6d_indices):
        raise RuntimeError("DiffPath 1D/6D valid indices are different")
    if not np.isfinite(diffpath_time).all():
        raise RuntimeError("DiffPath statistics contain NaN/Inf")
    if not np.isfinite(diffpath_6d_time).all():
        raise RuntimeError("DiffPath-6D features contain NaN/Inf")
    return (
        diffpath_time,
        diffpath_6d_time,
        diffpath_indices,
        resolved_timesteps,
    )


def robust_norm_stats(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot normalize empty DiffPath values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    std = float(np.std(values))
    if mad >= float(eps):
        scale = float(1.4826 * mad)
        source = "mad"
    elif std >= float(eps):
        scale = std
        source = "std"
    else:
        scale = float(eps)
        source = "eps"
    return {
        "median": median,
        "mad": mad,
        "std": std,
        "scale": scale,
        "scale_source": source,
    }


def robust_z(values, median, scale, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    return (values - float(median)) / (float(scale) + float(eps))


def robust_norm_stats_by_column(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"column normalization expects [points, dim], got {values.shape}"
        )
    if values.shape[0] == 0:
        raise ValueError("cannot normalize empty DiffPath-6D values")
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median.reshape(1, -1)), axis=0)
    std = np.std(values, axis=0)
    scale = np.where(
        mad >= float(eps),
        1.4826 * mad,
        np.where(std >= float(eps), std, float(eps)),
    )
    source = np.where(
        mad >= float(eps),
        "mad",
        np.where(std >= float(eps), "std", "eps"),
    )
    return {
        "median": median.astype(np.float64),
        "mad": mad.astype(np.float64),
        "std": std.astype(np.float64),
        "scale": scale.astype(np.float64),
        "scale_source": source,
    }


def robust_z_by_column(values, median, scale, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    median = np.asarray(median, dtype=np.float64).reshape(1, -1)
    scale = np.asarray(scale, dtype=np.float64).reshape(1, -1)
    return (values - median) / (scale + float(eps))


def empirical_cdf(values, sorted_reference):
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(sorted_reference, dtype=np.float64).reshape(-1)
    if reference.size == 0:
        raise ValueError("empirical CDF reference is empty")
    ranks = np.searchsorted(reference, values, side="right")
    return (ranks.astype(np.float64) / float(reference.size)).astype(
        np.float32
    )


def parse_bandwidths(value):
    if isinstance(value, str):
        bandwidths = [
            float(item)
            for item in value.replace(",", " ").split()
            if item
        ]
    else:
        bandwidths = [float(item) for item in value]
    if not bandwidths or any(item <= 0 for item in bandwidths):
        raise ValueError(
            f"KDE bandwidths must be positive, got {bandwidths}"
        )
    return bandwidths


def parse_gmm_components(value):
    if isinstance(value, str):
        components = [
            int(item)
            for item in value.replace(",", " ").split()
            if item
        ]
    else:
        components = [int(item) for item in value]
    if not components or any(item <= 0 for item in components):
        raise ValueError(
            f"GMM component counts must be positive, got {components}"
        )
    unique = []
    seen = set()
    for item in components:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def parse_gmm_covariance_types(value):
    if isinstance(value, str):
        covariance_types = [
            item.strip()
            for item in value.replace(",", " ").split()
            if item.strip()
        ]
    else:
        covariance_types = [str(item) for item in value]
    allowed = {"diag", "full"}
    invalid = [
        item
        for item in covariance_types
        if item not in allowed
    ]
    if invalid or not covariance_types:
        raise ValueError(
            "GMM covariance types must be drawn from "
            f"{sorted(allowed)}, got {covariance_types}"
        )
    unique = []
    seen = set()
    for item in covariance_types:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _kernel_density_class():
    try:
        from sklearn.neighbors import KernelDensity
    except ImportError as exc:
        raise ImportError(
            "DiffPath requires scikit-learn for Gaussian KDE"
        ) from exc
    return KernelDensity


def _gmm_classes():
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "DiffPath-6D GMM requires scikit-learn"
        ) from exc
    return GaussianMixture, StandardScaler


def _as_kde_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(
            f"KDE values must be [points] or [points, dim], got {values.shape}"
        )
    return values


def kde_score_samples(kde, values, chunk_size=20000):
    values = _as_kde_matrix(values)
    chunks = []
    for begin in range(0, len(values), int(chunk_size)):
        chunks.append(
            kde.score_samples(values[begin : begin + int(chunk_size)])
        )
    return np.concatenate(chunks, axis=0)


def gmm_score_samples(gmm, values, chunk_size=20000):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"GMM values must be [points, dim], got {values.shape}")
    chunks = []
    for begin in range(0, len(values), int(chunk_size)):
        chunks.append(
            gmm.score_samples(values[begin : begin + int(chunk_size)])
        )
    return np.concatenate(chunks, axis=0)


def fit_diffpath_calibrator(
    normal_recon,
    normal_diffpath,
    timesteps,
    bandwidths=DEFAULT_KDE_BANDWIDTHS,
    seed=0,
    normal_recon_max=None,
    eps=1e-8,
):
    KernelDensity = _kernel_density_class()
    normal_recon = np.asarray(normal_recon, dtype=np.float64).reshape(-1)
    if normal_recon_max is None:
        normal_recon_max = normal_recon
    normal_recon_max = np.asarray(
        normal_recon_max,
        dtype=np.float64,
    ).reshape(-1)
    normal_diffpath = np.asarray(
        normal_diffpath,
        dtype=np.float64,
    ).reshape(-1)
    if (
        len(normal_recon) != len(normal_diffpath)
        or len(normal_recon_max) != len(normal_diffpath)
    ):
        raise ValueError(
            "normal reconstruction and DiffPath lengths are different"
        )
    if len(normal_diffpath) < 10:
        raise ValueError(
            f"at least 10 normal points are required, got "
            f"{len(normal_diffpath)}"
        )

    norm_stats = robust_norm_stats(normal_diffpath, eps=eps)
    normalized = robust_z(
        normal_diffpath,
        norm_stats["median"],
        norm_stats["scale"],
        eps=eps,
    )
    bandwidths = parse_bandwidths(bandwidths)

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(normalized))
    validation_count = max(int(round(0.1 * len(order))), 1)
    validation_count = min(validation_count, len(order) - 1)
    validation_values = normalized[order[:validation_count]]
    fit_values = normalized[order[validation_count:]]

    best_bandwidth = None
    best_likelihood = -np.inf
    bandwidth_likelihoods = []
    for bandwidth in bandwidths:
        candidate = KernelDensity(
            kernel="gaussian",
            bandwidth=float(bandwidth),
        ).fit(fit_values.reshape(-1, 1))
        mean_likelihood = float(
            kde_score_samples(candidate, validation_values).mean()
        )
        bandwidth_likelihoods.append(mean_likelihood)
        if (
            mean_likelihood > best_likelihood
            or (
                np.isclose(mean_likelihood, best_likelihood)
                and (
                    best_bandwidth is None
                    or float(bandwidth) < best_bandwidth
                )
            )
        ):
            best_likelihood = mean_likelihood
            best_bandwidth = float(bandwidth)

    final_kde = KernelDensity(
        kernel="gaussian",
        bandwidth=best_bandwidth,
    ).fit(normalized.reshape(-1, 1))
    normal_path_raw_score = -kde_score_samples(final_kde, normalized)

    return {
        "diffpath_timesteps": np.asarray(timesteps, dtype=np.int64),
        "path_scale": np.asarray(len(timesteps), dtype=np.float32),
        "normal_recon_sorted": np.sort(normal_recon).astype(np.float32),
        "normal_recon_sum_abs_sorted": np.sort(normal_recon).astype(
            np.float32
        ),
        "normal_recon_max_abs_sorted": np.sort(normal_recon_max).astype(
            np.float32
        ),
        "normal_diffpath_statistic_sorted": np.sort(
            normal_diffpath
        ).astype(np.float32),
        "normal_diffpath_raw_score_sorted": np.sort(
            normal_path_raw_score
        ).astype(np.float32),
        "kde_fit_values": normalized.astype(np.float32),
        "kde_bandwidth": np.asarray(
            best_bandwidth,
            dtype=np.float32,
        ),
        "kde_bandwidth_candidates": np.asarray(
            bandwidths,
            dtype=np.float32,
        ),
        "kde_validation_mean_log_likelihood": np.asarray(
            bandwidth_likelihoods,
            dtype=np.float32,
        ),
        "diffpath_median": np.asarray(
            norm_stats["median"],
            dtype=np.float32,
        ),
        "diffpath_scale": np.asarray(
            norm_stats["scale"],
            dtype=np.float32,
        ),
        "diffpath_mad": np.asarray(
            norm_stats["mad"],
            dtype=np.float32,
        ),
        "diffpath_std": np.asarray(
            norm_stats["std"],
            dtype=np.float32,
        ),
        "diffpath_scale_source": np.asarray(
            norm_stats["scale_source"]
        ),
        "normal_point_count": np.asarray(
            len(normal_diffpath),
            dtype=np.int64,
        ),
        "calibration_seed": np.asarray(int(seed), dtype=np.int64),
    }


def fit_diffpath_6d_calibrator(
    normal_recon,
    normal_diffpath_6d,
    timesteps,
    bandwidths=DEFAULT_KDE_BANDWIDTHS_6D,
    seed=0,
    eps=1e-8,
):
    KernelDensity = _kernel_density_class()
    normal_recon = np.asarray(normal_recon, dtype=np.float64).reshape(-1)
    normal_diffpath_6d = np.asarray(
        normal_diffpath_6d,
        dtype=np.float64,
    )
    if normal_diffpath_6d.ndim != 2:
        raise ValueError(
            f"DiffPath-6D normal features must be [points, 6], got "
            f"{normal_diffpath_6d.shape}"
        )
    if normal_diffpath_6d.shape[1] != len(DIFFPATH_6D_FEATURE_NAMES):
        raise ValueError(
            f"DiffPath-6D expected {len(DIFFPATH_6D_FEATURE_NAMES)} "
            f"features, got {normal_diffpath_6d.shape[1]}"
        )
    if len(normal_recon) != len(normal_diffpath_6d):
        raise ValueError(
            "normal reconstruction and DiffPath-6D lengths are different"
        )
    if len(normal_diffpath_6d) < 10:
        raise ValueError(
            f"at least 10 normal points are required, got "
            f"{len(normal_diffpath_6d)}"
        )

    norm_stats = robust_norm_stats_by_column(normal_diffpath_6d, eps=eps)
    normalized = robust_z_by_column(
        normal_diffpath_6d,
        norm_stats["median"],
        norm_stats["scale"],
        eps=eps,
    )
    bandwidths = parse_bandwidths(bandwidths)

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(normalized))
    validation_count = max(int(round(0.1 * len(order))), 1)
    validation_count = min(validation_count, len(order) - 1)
    validation_values = normalized[order[:validation_count]]
    fit_values = normalized[order[validation_count:]]

    best_bandwidth = None
    best_likelihood = -np.inf
    bandwidth_likelihoods = []
    for bandwidth in bandwidths:
        candidate = KernelDensity(
            kernel="gaussian",
            bandwidth=float(bandwidth),
        ).fit(fit_values)
        mean_likelihood = float(
            kde_score_samples(candidate, validation_values).mean()
        )
        bandwidth_likelihoods.append(mean_likelihood)
        if (
            mean_likelihood > best_likelihood
            or (
                np.isclose(mean_likelihood, best_likelihood)
                and (
                    best_bandwidth is None
                    or float(bandwidth) < best_bandwidth
                )
            )
        ):
            best_likelihood = mean_likelihood
            best_bandwidth = float(bandwidth)

    final_kde = KernelDensity(
        kernel="gaussian",
        bandwidth=best_bandwidth,
    ).fit(normalized)
    normal_path_raw_score = -kde_score_samples(final_kde, normalized)

    return {
        "diffpath_timesteps": np.asarray(timesteps, dtype=np.int64),
        "path_scale": np.asarray(len(timesteps), dtype=np.float32),
        "feature_names": np.asarray(DIFFPATH_6D_FEATURE_NAMES),
        "normal_recon_sorted": np.sort(normal_recon).astype(np.float32),
        "normal_diffpath_6d_raw_score_sorted": np.sort(
            normal_path_raw_score
        ).astype(np.float32),
        "kde_fit_values": normalized.astype(np.float32),
        "kde_bandwidth": np.asarray(
            best_bandwidth,
            dtype=np.float32,
        ),
        "kde_bandwidth_candidates": np.asarray(
            bandwidths,
            dtype=np.float32,
        ),
        "kde_validation_mean_log_likelihood": np.asarray(
            bandwidth_likelihoods,
            dtype=np.float32,
        ),
        "diffpath_median": norm_stats["median"].astype(np.float32),
        "diffpath_scale": norm_stats["scale"].astype(np.float32),
        "diffpath_mad": norm_stats["mad"].astype(np.float32),
        "diffpath_std": norm_stats["std"].astype(np.float32),
        "diffpath_scale_source": np.asarray(norm_stats["scale_source"]),
        "normal_point_count": np.asarray(
            len(normal_diffpath_6d),
            dtype=np.int64,
        ),
        "calibration_seed": np.asarray(int(seed), dtype=np.int64),
    }


def _validate_diffpath_6d_inputs(normal_recon, normal_diffpath_6d):
    normal_recon = np.asarray(normal_recon, dtype=np.float64).reshape(-1)
    normal_diffpath_6d = np.asarray(
        normal_diffpath_6d,
        dtype=np.float64,
    )
    if normal_diffpath_6d.ndim != 2:
        raise ValueError(
            f"DiffPath-6D normal features must be [points, 6], got "
            f"{normal_diffpath_6d.shape}"
        )
    if normal_diffpath_6d.shape[1] != len(DIFFPATH_6D_FEATURE_NAMES):
        raise ValueError(
            f"DiffPath-6D expected {len(DIFFPATH_6D_FEATURE_NAMES)} "
            f"features, got {normal_diffpath_6d.shape[1]}"
        )
    if len(normal_recon) != len(normal_diffpath_6d):
        raise ValueError(
            "normal reconstruction and DiffPath-6D lengths are different"
        )
    if len(normal_diffpath_6d) < 10:
        raise ValueError(
            f"at least 10 normal points are required, got "
            f"{len(normal_diffpath_6d)}"
        )
    if not np.isfinite(normal_diffpath_6d).all():
        raise ValueError("DiffPath-6D normal features contain NaN/Inf")
    return normal_recon, normal_diffpath_6d


def fit_diffpath_6d_gmm_calibrator(
    normal_recon,
    normal_diffpath_6d,
    timesteps,
    n_components=DEFAULT_GMM_COMPONENTS_6D,
    covariance_types=DEFAULT_GMM_COVARIANCE_TYPES_6D,
    seed=0,
):
    GaussianMixture, StandardScaler = _gmm_classes()
    normal_recon, normal_diffpath_6d = _validate_diffpath_6d_inputs(
        normal_recon,
        normal_diffpath_6d,
    )
    component_candidates = parse_gmm_components(n_components)
    covariance_candidates = parse_gmm_covariance_types(covariance_types)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(normal_diffpath_6d)
    if not np.isfinite(scaled).all():
        raise RuntimeError("standardized DiffPath-6D features contain NaN/Inf")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(scaled))
    validation_count = max(int(round(0.1 * len(order))), 1)
    validation_count = min(validation_count, len(order) - 1)
    validation_values = scaled[order[:validation_count]]
    fit_values = scaled[order[validation_count:]]

    best_params = None
    best_likelihood = -np.inf
    tried_components = []
    tried_covariance_types = []
    validation_likelihoods = []
    candidate_converged = []
    candidate_n_iter = []
    for covariance_type in covariance_candidates:
        for component_count in component_candidates:
            if component_count > len(fit_values):
                continue
            candidate = GaussianMixture(
                n_components=int(component_count),
                covariance_type=str(covariance_type),
                random_state=int(seed),
                reg_covar=1e-6,
                max_iter=200,
                n_init=3,
            )
            candidate.fit(fit_values)
            mean_likelihood = float(
                gmm_score_samples(candidate, validation_values).mean()
            )
            tried_components.append(int(component_count))
            tried_covariance_types.append(str(covariance_type))
            validation_likelihoods.append(mean_likelihood)
            candidate_converged.append(bool(candidate.converged_))
            candidate_n_iter.append(int(candidate.n_iter_))
            candidate_key = (
                mean_likelihood,
                -int(component_count),
                1 if covariance_type == "diag" else 0,
            )
            if best_params is None or candidate_key > best_params["key"]:
                best_params = {
                    "n_components": int(component_count),
                    "covariance_type": str(covariance_type),
                    "key": candidate_key,
                }
                best_likelihood = mean_likelihood

    if best_params is None:
        raise ValueError(
            "no valid DiffPath-6D GMM candidate could be fitted; "
            f"fit points={len(fit_values)}, candidates={component_candidates}"
        )

    final_gmm = GaussianMixture(
        n_components=best_params["n_components"],
        covariance_type=best_params["covariance_type"],
        random_state=int(seed),
        reg_covar=1e-6,
        max_iter=200,
        n_init=3,
    )
    final_gmm.fit(scaled)
    normal_path_raw_score = -gmm_score_samples(final_gmm, scaled)

    return {
        "diffpath_timesteps": np.asarray(timesteps, dtype=np.int64),
        "path_scale": np.asarray(len(timesteps), dtype=np.float32),
        "feature_names": np.asarray(DIFFPATH_6D_FEATURE_NAMES),
        "normal_recon_sorted": np.sort(normal_recon).astype(np.float32),
        "normal_diffpath_6d_gmm_raw_score_sorted": np.sort(
            normal_path_raw_score
        ).astype(np.float32),
        "standard_scaler_mean": scaler.mean_.astype(np.float32),
        "standard_scaler_scale": scaler.scale_.astype(np.float32),
        "standard_scaler_var": scaler.var_.astype(np.float32),
        "gmm_n_components": np.asarray(
            best_params["n_components"],
            dtype=np.int64,
        ),
        "gmm_covariance_type": np.asarray(best_params["covariance_type"]),
        "gmm_weights": final_gmm.weights_.astype(np.float32),
        "gmm_means": final_gmm.means_.astype(np.float32),
        "gmm_covariances": final_gmm.covariances_.astype(np.float32),
        "gmm_precisions_cholesky": final_gmm.precisions_cholesky_.astype(
            np.float32
        ),
        "gmm_validation_best_mean_log_likelihood": np.asarray(
            best_likelihood,
            dtype=np.float32,
        ),
        "gmm_candidate_n_components": np.asarray(
            tried_components,
            dtype=np.int64,
        ),
        "gmm_candidate_covariance_type": np.asarray(tried_covariance_types),
        "gmm_validation_mean_log_likelihood": np.asarray(
            validation_likelihoods,
            dtype=np.float32,
        ),
        "gmm_candidate_converged": np.asarray(
            candidate_converged,
            dtype=np.bool_,
        ),
        "gmm_candidate_n_iter": np.asarray(candidate_n_iter, dtype=np.int64),
        "gmm_converged": np.asarray(bool(final_gmm.converged_)),
        "gmm_n_iter": np.asarray(int(final_gmm.n_iter_), dtype=np.int64),
        "gmm_lower_bound": np.asarray(final_gmm.lower_bound_, dtype=np.float32),
        "normal_point_count": np.asarray(
            len(normal_diffpath_6d),
            dtype=np.int64,
        ),
        "calibration_seed": np.asarray(int(seed), dtype=np.int64),
    }


def save_calibrator(path, calibrator, metadata):
    payload = dict(calibrator)
    payload.update(
        {
            key: np.asarray(value)
            for key, value in metadata.items()
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)


def load_calibrator(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"DiffPath calibrator not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def apply_diffpath_calibrator(
    calibrator,
    recon_score,
    diffpath_statistic,
    eps=1e-8,
):
    KernelDensity = _kernel_density_class()
    recon_score = np.asarray(recon_score, dtype=np.float64).reshape(-1)
    diffpath_statistic = np.asarray(
        diffpath_statistic,
        dtype=np.float64,
    ).reshape(-1)
    normalized = robust_z(
        diffpath_statistic,
        float(calibrator["diffpath_median"]),
        float(calibrator["diffpath_scale"]),
        eps=eps,
    )
    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=float(calibrator["kde_bandwidth"]),
    ).fit(
        np.asarray(
            calibrator["kde_fit_values"],
            dtype=np.float64,
        ).reshape(-1, 1)
    )
    diffpath_raw_score = -kde_score_samples(kde, normalized)
    recon_cdf = empirical_cdf(
        recon_score,
        calibrator["normal_recon_sorted"],
    )
    diffpath_cdf = empirical_cdf(
        diffpath_raw_score,
        calibrator["normal_diffpath_raw_score_sorted"],
    )
    return (
        diffpath_raw_score.astype(np.float32),
        recon_cdf,
        diffpath_cdf,
    )


def apply_diffpath_6d_calibrator(
    calibrator,
    recon_score,
    diffpath_6d_features,
    eps=1e-8,
):
    KernelDensity = _kernel_density_class()
    recon_score = np.asarray(recon_score, dtype=np.float64).reshape(-1)
    diffpath_6d_features = np.asarray(
        diffpath_6d_features,
        dtype=np.float64,
    )
    normalized = robust_z_by_column(
        diffpath_6d_features,
        calibrator["diffpath_median"],
        calibrator["diffpath_scale"],
        eps=eps,
    )
    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=float(calibrator["kde_bandwidth"]),
    ).fit(
        np.asarray(
            calibrator["kde_fit_values"],
            dtype=np.float64,
        )
    )
    diffpath_raw_score = -kde_score_samples(kde, normalized)
    recon_cdf = empirical_cdf(
        recon_score,
        calibrator["normal_recon_sorted"],
    )
    diffpath_cdf = empirical_cdf(
        diffpath_raw_score,
        calibrator["normal_diffpath_6d_raw_score_sorted"],
    )
    return (
        diffpath_raw_score.astype(np.float32),
        recon_cdf,
        diffpath_cdf,
    )


def _restore_gmm_from_calibrator(calibrator):
    GaussianMixture, _StandardScaler = _gmm_classes()
    covariance_type = str(np.asarray(calibrator["gmm_covariance_type"]).item())
    component_count = int(np.asarray(calibrator["gmm_n_components"]).item())
    gmm = GaussianMixture(
        n_components=component_count,
        covariance_type=covariance_type,
    )
    gmm.weights_ = np.asarray(calibrator["gmm_weights"], dtype=np.float64)
    gmm.means_ = np.asarray(calibrator["gmm_means"], dtype=np.float64)
    gmm.covariances_ = np.asarray(
        calibrator["gmm_covariances"],
        dtype=np.float64,
    )
    gmm.precisions_cholesky_ = np.asarray(
        calibrator["gmm_precisions_cholesky"],
        dtype=np.float64,
    )
    gmm.converged_ = bool(np.asarray(calibrator["gmm_converged"]).item())
    gmm.n_iter_ = int(np.asarray(calibrator["gmm_n_iter"]).item())
    gmm.lower_bound_ = float(np.asarray(calibrator["gmm_lower_bound"]).item())
    gmm.n_features_in_ = int(gmm.means_.shape[1])
    return gmm


def apply_diffpath_6d_gmm_calibrator(
    calibrator,
    recon_score,
    diffpath_6d_features,
    eps=1e-8,
):
    recon_score = np.asarray(recon_score, dtype=np.float64).reshape(-1)
    diffpath_6d_features = np.asarray(
        diffpath_6d_features,
        dtype=np.float64,
    )
    scaler_mean = np.asarray(
        calibrator["standard_scaler_mean"],
        dtype=np.float64,
    ).reshape(1, -1)
    scaler_scale = np.asarray(
        calibrator["standard_scaler_scale"],
        dtype=np.float64,
    ).reshape(1, -1)
    scaler_scale = np.where(scaler_scale > float(eps), scaler_scale, 1.0)
    scaled = (diffpath_6d_features - scaler_mean) / scaler_scale
    if not np.isfinite(scaled).all():
        raise RuntimeError("standardized DiffPath-6D test features contain NaN/Inf")

    gmm = _restore_gmm_from_calibrator(calibrator)
    diffpath_raw_score = -gmm_score_samples(gmm, scaled)
    recon_cdf = empirical_cdf(
        recon_score,
        calibrator["normal_recon_sorted"],
    )
    diffpath_cdf = empirical_cdf(
        diffpath_raw_score,
        calibrator["normal_diffpath_6d_gmm_raw_score_sorted"],
    )
    return (
        diffpath_raw_score.astype(np.float32),
        recon_cdf,
        diffpath_cdf,
    )


def _score_output_paths(output_root, result_tag, save_id):
    ensemble_dir = os.path.join(output_root, result_tag, "ensemble")
    return {
        "ensemble_dir": ensemble_dir,
        "score_npz": os.path.join(
            ensemble_dir,
            f"diffpath_1d_scores_{save_id}.npz",
        ),
        "metadata": os.path.join(
            ensemble_dir,
            f"diffpath_1d_metadata_{save_id}.json",
        ),
        "recon": os.path.join(
            ensemble_dir,
            f"diffpath_final_recon_score_ensemble_{save_id}.pt",
        ),
        "recon_max": os.path.join(
            ensemble_dir,
            f"diffpath_final_recon_score_max_abs_ensemble_{save_id}.pt",
        ),
        "statistic": os.path.join(
            ensemble_dir,
            f"diffpath_1d_statistic_ensemble_{save_id}.pt",
        ),
        "raw": os.path.join(
            ensemble_dir,
            f"diffpath_1d_raw_ensemble_{save_id}.pt",
        ),
        "diffpath_cdf": os.path.join(
            ensemble_dir,
            f"diffpath_1d_cdf_ensemble_{save_id}.pt",
        ),
        "features_6d": os.path.join(
            ensemble_dir,
            f"diffpath_6d_features_ensemble_{save_id}.pt",
        ),
        "raw_6d": os.path.join(
            ensemble_dir,
            f"diffpath_6d_raw_ensemble_{save_id}.pt",
        ),
        "raw_6d_gmm": os.path.join(
            ensemble_dir,
            f"diffpath_6d_gmm_raw_ensemble_{save_id}.pt",
        ),
        "diffpath_6d_cdf": os.path.join(
            ensemble_dir,
            f"diffpath_6d_cdf_ensemble_{save_id}.pt",
        ),
        "diffpath_6d_gmm_cdf": os.path.join(
            ensemble_dir,
            f"diffpath_6d_gmm_cdf_ensemble_{save_id}.pt",
        ),
        "recon_cdf": os.path.join(
            ensemble_dir,
            f"recon_cdf_ensemble_{save_id}.pt",
        ),
        "recon_max_cdf": os.path.join(
            ensemble_dir,
            f"recon_max_abs_cdf_ensemble_{save_id}.pt",
        ),
    }


def run_diffpath_pathb(
    model,
    normal_loader,
    test_loader,
    label_path,
    output_root,
    result_tag,
    base_dataset,
    save_id,
    num_path_steps=10,
    split=4,
    seed=0,
    bandwidths=DEFAULT_KDE_BANDWIDTHS,
    bandwidths_6d=DEFAULT_KDE_BANDWIDTHS_6D,
    gmm_components_6d=DEFAULT_GMM_COMPONENTS_6D,
    gmm_covariance_types_6d=DEFAULT_GMM_COVARIANCE_TYPES_6D,
    enable_6d_kde=True,
    recompute_calibrator=False,
    overwrite=False,
    recon_normal_loader=None,
    recon_test_loader=None,
    eps=1e-8,
):
    recon_normal_loader = recon_normal_loader or normal_loader
    recon_test_loader = recon_test_loader or test_loader
    prototype_dir = os.path.join(
        output_root,
        "_prototypes",
        base_dataset,
    )
    calibrator_path = os.path.join(
        prototype_dir,
        f"diffpath_1d_steps{int(num_path_steps)}_seed{int(seed)}_"
        f"{save_id}.npz",
    )
    calibrator_6d_path = os.path.join(
        prototype_dir,
        f"diffpath_6d_steps{int(num_path_steps)}_seed{int(seed)}_"
        f"{save_id}.npz",
    )
    calibrator_6d_gmm_path = os.path.join(
        prototype_dir,
        f"diffpath_6d_gmm_steps{int(num_path_steps)}_seed{int(seed)}_"
        f"{save_id}.npz",
    )

    need_1d_calibrator = (
        recompute_calibrator or not os.path.exists(calibrator_path)
    )
    enable_6d_kde = bool(enable_6d_kde)
    need_6d_calibrator = enable_6d_kde and (
        recompute_calibrator or not os.path.exists(calibrator_6d_path)
    )
    need_6d_gmm_calibrator = (
        recompute_calibrator or not os.path.exists(calibrator_6d_gmm_path)
    )

    if not need_1d_calibrator:
        calibrator = load_calibrator(calibrator_path)
        print(f"[DiffPath] loaded calibrator: {calibrator_path}")
    else:
        calibrator = None
    if enable_6d_kde and not need_6d_calibrator:
        calibrator_6d = load_calibrator(calibrator_6d_path)
        print(f"[DiffPath] loaded 6D calibrator: {calibrator_6d_path}")
    else:
        calibrator_6d = None
    if not need_6d_gmm_calibrator:
        calibrator_6d_gmm = load_calibrator(calibrator_6d_gmm_path)
        print(
            "[DiffPath] loaded 6D GMM calibrator: "
            f"{calibrator_6d_gmm_path}"
        )
    else:
        calibrator_6d_gmm = None

    if need_1d_calibrator or need_6d_calibrator or need_6d_gmm_calibrator:
        set_all_seeds(seed)
        (
            normal_recon_sum,
            normal_recon_max,
            normal_recon_indices,
        ) = collect_reconstruction_scores(
            model,
            recon_normal_loader,
            split,
            desc=f"Reconstruction normal calibration {save_id}",
        )
        (
            normal_diffpath,
            normal_diffpath_6d,
            normal_diffpath_indices,
            timesteps,
        ) = collect_diffpath_statistics(
            model,
            normal_loader,
            num_path_steps,
            split,
            desc=f"DiffPath normal calibration {save_id}",
        )
        if not np.array_equal(normal_recon_indices, normal_diffpath_indices):
            raise RuntimeError(
                "normal reconstruction and DiffPath valid indices differ"
        )
        if need_1d_calibrator:
            calibrator = fit_diffpath_calibrator(
                normal_recon_sum,
                normal_diffpath,
                timesteps,
                bandwidths=bandwidths,
                seed=seed,
                normal_recon_max=normal_recon_max,
                eps=eps,
            )
            save_calibrator(
                calibrator_path,
                calibrator,
                {
                    "score_type": "diffpath_1d_epsilon_derivative_kde",
                    "base_dataset": base_dataset,
                    "save": save_id,
                },
            )
            calibrator = load_calibrator(calibrator_path)
            print(f"[DiffPath] saved calibrator: {calibrator_path}")
        if enable_6d_kde and need_6d_calibrator:
            calibrator_6d = fit_diffpath_6d_calibrator(
                normal_recon_sum,
                normal_diffpath_6d,
                timesteps,
                bandwidths=bandwidths_6d,
                seed=seed,
                eps=eps,
            )
            save_calibrator(
                calibrator_6d_path,
                calibrator_6d,
                {
                    "score_type": "diffpath_6d_epsilon_derivative_kde",
                    "base_dataset": base_dataset,
                    "save": save_id,
                },
            )
            calibrator_6d = load_calibrator(calibrator_6d_path)
            print(f"[DiffPath] saved 6D calibrator: {calibrator_6d_path}")
        if need_6d_gmm_calibrator:
            calibrator_6d_gmm = fit_diffpath_6d_gmm_calibrator(
                normal_recon_sum,
                normal_diffpath_6d,
                timesteps,
                n_components=gmm_components_6d,
                covariance_types=gmm_covariance_types_6d,
                seed=seed,
            )
            save_calibrator(
                calibrator_6d_gmm_path,
                calibrator_6d_gmm,
                {
                    "score_type": (
                        "diffpath_6d_epsilon_derivative_standard_"
                        "scaler_gmm"
                    ),
                    "base_dataset": base_dataset,
                    "save": save_id,
                },
            )
            calibrator_6d_gmm = load_calibrator(calibrator_6d_gmm_path)
            print(
                "[DiffPath] saved 6D GMM calibrator: "
                f"{calibrator_6d_gmm_path}"
            )

    expected_steps = model.resolve_diffpath_timesteps(num_path_steps)
    stored_steps = [
        int(step)
        for step in calibrator["diffpath_timesteps"].reshape(-1)
    ]
    if stored_steps != expected_steps:
        raise ValueError(
            f"calibrator timesteps {stored_steps} do not match model "
            f"timesteps {expected_steps}"
        )
    if "normal_recon_max_abs_sorted" not in calibrator:
        raise ValueError(
            "DiffPath calibrator does not contain max-abs reconstruction "
            "calibration. Use a new OUTPUT_ROOT or pass "
            "--diffpath_recompute_calibrator."
        )
    if enable_6d_kde:
        stored_6d_steps = [
            int(step)
            for step in calibrator_6d["diffpath_timesteps"].reshape(-1)
        ]
        if stored_6d_steps != expected_steps:
            raise ValueError(
                f"6D calibrator timesteps {stored_6d_steps} do not match "
                f"model timesteps {expected_steps}"
            )
    stored_6d_gmm_steps = [
        int(step)
        for step in calibrator_6d_gmm["diffpath_timesteps"].reshape(-1)
    ]
    if stored_6d_gmm_steps != expected_steps:
        raise ValueError(
            f"6D GMM calibrator timesteps {stored_6d_gmm_steps} do not "
            f"match model timesteps {expected_steps}"
        )

    output_paths = _score_output_paths(
        output_root,
        result_tag,
        save_id,
    )
    writable_outputs = [
            output_paths["score_npz"],
            output_paths["metadata"],
            output_paths["recon"],
            output_paths["recon_max"],
            output_paths["statistic"],
            output_paths["raw"],
            output_paths["diffpath_cdf"],
            output_paths["features_6d"],
            output_paths["raw_6d_gmm"],
            output_paths["diffpath_6d_gmm_cdf"],
            output_paths["recon_cdf"],
            output_paths["recon_max_cdf"],
    ]
    if enable_6d_kde:
        writable_outputs.extend(
            [
                output_paths["raw_6d"],
                output_paths["diffpath_6d_cdf"],
            ]
        )
    ensure_outputs_can_be_written(writable_outputs, overwrite=overwrite)
    os.makedirs(output_paths["ensemble_dir"], exist_ok=True)

    set_all_seeds(int(seed) + 100000)
    (
        recon_sum_score,
        recon_max_score,
        recon_indices,
    ) = collect_reconstruction_scores(
        model,
        recon_test_loader,
        split,
        desc=f"Reconstruction test score {result_tag} {save_id}",
    )
    (
        diffpath_statistic,
        diffpath_6d_features,
        valid_indices,
        timesteps,
    ) = collect_diffpath_statistics(
        model,
        test_loader,
        num_path_steps,
        split,
        desc=f"DiffPath test score {result_tag} {save_id}",
    )
    if not np.array_equal(recon_indices, valid_indices):
        raise RuntimeError(
            "test reconstruction and DiffPath valid indices differ"
        )
    (
        diffpath_raw_score,
        recon_cdf,
        diffpath_cdf,
    ) = apply_diffpath_calibrator(
        calibrator,
        recon_sum_score,
        diffpath_statistic,
        eps=eps,
    )
    recon_max_cdf = empirical_cdf(
        recon_max_score,
        calibrator["normal_recon_max_abs_sorted"],
    )
    diffpath_6d_raw_score = None
    diffpath_6d_cdf = None
    if enable_6d_kde:
        (
            diffpath_6d_raw_score,
            recon_cdf_6d,
            diffpath_6d_cdf,
        ) = apply_diffpath_6d_calibrator(
            calibrator_6d,
            recon_sum_score,
            diffpath_6d_features,
            eps=eps,
        )
        if not np.allclose(recon_cdf, recon_cdf_6d, atol=1e-7):
            raise RuntimeError("1D and 6D reconstruction ECDF values differ")
    (
        diffpath_6d_gmm_raw_score,
        recon_cdf_6d_gmm,
        diffpath_6d_gmm_cdf,
    ) = apply_diffpath_6d_gmm_calibrator(
        calibrator_6d_gmm,
        recon_sum_score,
        diffpath_6d_features,
        eps=eps,
    )
    if not np.allclose(recon_cdf, recon_cdf_6d_gmm, atol=1e-7):
        raise RuntimeError("1D and 6D GMM reconstruction ECDF values differ")
    with open(label_path, "rb") as f:
        labels = np.asarray(pickle.load(f), dtype=np.int64).reshape(-1)
    if valid_indices.size == 0 or valid_indices.max() >= len(labels):
        raise RuntimeError(
            f"DiffPath valid indices exceed labels: "
            f"max={valid_indices.max() if valid_indices.size else 'empty'}, "
            f"label_len={len(labels)}"
        )
    labels_aligned = labels[valid_indices]

    for name, values in {
        "recon_sum_score": recon_sum_score,
        "recon_max_score": recon_max_score,
        "diffpath_statistic": diffpath_statistic,
        "diffpath_raw_score": diffpath_raw_score,
        "diffpath_6d_gmm_raw_score": diffpath_6d_gmm_raw_score,
        "recon_cdf": recon_cdf,
        "recon_max_cdf": recon_max_cdf,
        "diffpath_cdf": diffpath_cdf,
        "diffpath_6d_gmm_cdf": diffpath_6d_gmm_cdf,
    }.items():
        if len(values) != len(valid_indices):
            raise RuntimeError(
                f"{name} length {len(values)} != valid index length "
                f"{len(valid_indices)}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError(f"{name} contains NaN/Inf")
    if enable_6d_kde:
        for name, values in {
            "diffpath_6d_raw_score": diffpath_6d_raw_score,
            "diffpath_6d_cdf": diffpath_6d_cdf,
        }.items():
            if len(values) != len(valid_indices):
                raise RuntimeError(
                    f"{name} length {len(values)} != valid index length "
                    f"{len(valid_indices)}"
                )
            if not np.isfinite(values).all():
                raise RuntimeError(f"{name} contains NaN/Inf")
    if diffpath_6d_features.shape != (
        len(valid_indices),
        len(DIFFPATH_6D_FEATURE_NAMES),
    ):
        raise RuntimeError(
            f"DiffPath-6D feature shape {diffpath_6d_features.shape} "
            f"does not match valid length {len(valid_indices)}"
        )
    if not np.isfinite(diffpath_6d_features).all():
        raise RuntimeError("diffpath_6d_features contains NaN/Inf")

    score_payload = {
        "dataset": np.asarray(result_tag),
        "base_dataset": np.asarray(base_dataset),
        "save": np.asarray(save_id),
        "labels": labels.astype(np.int64),
        "labels_aligned": labels_aligned.astype(np.int64),
        "valid_indices": valid_indices.astype(np.int64),
        "final_recon_score": recon_sum_score.astype(np.float32),
        "final_recon_score_sum_abs": recon_sum_score.astype(np.float32),
        "final_recon_score_max_abs": recon_max_score.astype(np.float32),
        "diffpath_1d_statistic": diffpath_statistic.astype(np.float32),
        "diffpath_1d_raw_score": diffpath_raw_score.astype(np.float32),
        "diffpath_6d_features": diffpath_6d_features.astype(np.float32),
        "diffpath_6d_gmm_raw_score": diffpath_6d_gmm_raw_score.astype(
            np.float32
        ),
        "recon_cdf": recon_cdf.astype(np.float32),
        "recon_sum_abs_cdf": recon_cdf.astype(np.float32),
        "recon_max_abs_cdf": recon_max_cdf.astype(np.float32),
        "diffpath_1d_cdf": diffpath_cdf.astype(np.float32),
        "diffpath_6d_gmm_cdf": diffpath_6d_gmm_cdf.astype(np.float32),
        "diffpath_6d_feature_names": np.asarray(
            DIFFPATH_6D_FEATURE_NAMES
        ),
        "diffpath_timesteps": np.asarray(timesteps, dtype=np.int64),
        "calibrator_path": np.asarray(calibrator_path),
        "calibrator_6d_gmm_path": np.asarray(calibrator_6d_gmm_path),
        "diffpath_6d_kde_enabled": np.asarray(enable_6d_kde),
        "score_len": np.asarray(len(valid_indices), dtype=np.int64),
        "raw_label_len": np.asarray(len(labels), dtype=np.int64),
        "raw_label_sum": np.asarray(labels.sum(), dtype=np.int64),
        "aligned_label_sum": np.asarray(
            labels_aligned.sum(),
            dtype=np.int64,
        ),
    }
    if enable_6d_kde:
        score_payload.update(
            {
                "diffpath_6d_raw_score": diffpath_6d_raw_score.astype(
                    np.float32
                ),
                "diffpath_6d_kde_raw_score": diffpath_6d_raw_score.astype(
                    np.float32
                ),
                "diffpath_6d_cdf": diffpath_6d_cdf.astype(np.float32),
                "diffpath_6d_kde_cdf": diffpath_6d_cdf.astype(np.float32),
                "calibrator_6d_path": np.asarray(calibrator_6d_path),
            }
        )
    np.savez_compressed(output_paths["score_npz"], **score_payload)
    torch.save(
        torch.from_numpy(recon_sum_score.astype(np.float32)),
        output_paths["recon"],
    )
    torch.save(
        torch.from_numpy(recon_max_score.astype(np.float32)),
        output_paths["recon_max"],
    )
    torch.save(
        torch.from_numpy(diffpath_statistic.astype(np.float32)),
        output_paths["statistic"],
    )
    torch.save(
        torch.from_numpy(diffpath_raw_score.astype(np.float32)),
        output_paths["raw"],
    )
    torch.save(
        torch.from_numpy(diffpath_cdf.astype(np.float32)),
        output_paths["diffpath_cdf"],
    )
    torch.save(
        torch.from_numpy(diffpath_6d_features.astype(np.float32)),
        output_paths["features_6d"],
    )
    torch.save(
        torch.from_numpy(diffpath_6d_gmm_raw_score.astype(np.float32)),
        output_paths["raw_6d_gmm"],
    )
    torch.save(
        torch.from_numpy(diffpath_6d_gmm_cdf.astype(np.float32)),
        output_paths["diffpath_6d_gmm_cdf"],
    )
    torch.save(
        torch.from_numpy(recon_cdf.astype(np.float32)),
        output_paths["recon_cdf"],
    )
    torch.save(
        torch.from_numpy(recon_max_cdf.astype(np.float32)),
        output_paths["recon_max_cdf"],
    )
    if enable_6d_kde:
        torch.save(
            torch.from_numpy(diffpath_6d_raw_score.astype(np.float32)),
            output_paths["raw_6d"],
        )
        torch.save(
            torch.from_numpy(diffpath_6d_cdf.astype(np.float32)),
            output_paths["diffpath_6d_cdf"],
        )

    metadata = {
        "score_type": "strict_diffpath_1d_and_6d",
        "dataset": result_tag,
        "base_dataset": base_dataset,
        "save": save_id,
        "num_path_steps": int(num_path_steps),
        "diffpath_timesteps": timesteps,
        "path_scale": float(len(timesteps)),
        "diffpath_formula": (
            "sqrt(sum_over_steps_and_features("
            "(num_path_steps * delta_epsilon)^2))"
        ),
        "diffpath_6d_formula": (
            "per-time [sum(eps), sum(eps^2), sum(eps^3), "
            "sum(num_path_steps*delta_eps), "
            "sum((num_path_steps*delta_eps)^2), "
            "sum((num_path_steps*delta_eps)^3)] over features and path"
        ),
        "kde_bandwidth": float(calibrator["kde_bandwidth"]),
        "diffpath_6d_kde_enabled": enable_6d_kde,
        "kde_6d_bandwidth": (
            float(calibrator_6d["kde_bandwidth"])
            if enable_6d_kde
            else None
        ),
        "gmm_6d_n_components": int(calibrator_6d_gmm["gmm_n_components"]),
        "gmm_6d_covariance_type": str(
            np.asarray(calibrator_6d_gmm["gmm_covariance_type"]).item()
        ),
        "gmm_6d_components_searched": [
            int(item)
            for item in calibrator_6d_gmm[
                "gmm_candidate_n_components"
            ].reshape(-1)
        ],
        "gmm_6d_covariance_types_searched": [
            str(item)
            for item in calibrator_6d_gmm[
                "gmm_candidate_covariance_type"
            ].reshape(-1)
        ],
        "calibrator_path": calibrator_path,
        "calibrator_6d_path": calibrator_6d_path if enable_6d_kde else None,
        "calibrator_6d_gmm_path": calibrator_6d_gmm_path,
        "diffpath_6d_feature_names": list(DIFFPATH_6D_FEATURE_NAMES),
        "normal_point_count": int(calibrator["normal_point_count"]),
        "score_len": int(len(valid_indices)),
        "raw_label_len": int(len(labels)),
        "raw_label_sum": int(labels.sum()),
        "aligned_label_sum": int(labels_aligned.sum()),
        "uses_attention": False,
        "uses_hidden_features": False,
        "uses_multistep_voting": False,
    }
    with open(output_paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[DiffPath] saved score bundle: {output_paths['score_npz']}")
    return output_paths
