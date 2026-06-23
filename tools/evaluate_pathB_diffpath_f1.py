import argparse
import csv
import glob
import json
import os

import numpy as np
import torch


METHOD_ORDER = (
    "final_reconstruction",
    "final_reconstruction_sum_abs",
    "final_reconstruction_max_abs",
    "diffpath_1d",
    "diffpath_6d",
    "diffpath_6d_gmm",
    "fused_reconstruction_diffpath_1d",
    "fused_sum_abs_reconstruction_diffpath_1d",
    "fused_max_abs_reconstruction_diffpath_1d",
    "fused_reconstruction_diffpath_6d",
    "fused_sum_abs_reconstruction_diffpath_6d",
    "fused_max_abs_reconstruction_diffpath_6d",
    "fused_reconstruction_diffpath_6d_gmm",
    "fused_sum_abs_reconstruction_diffpath_6d_gmm",
    "fused_max_abs_reconstruction_diffpath_6d_gmm",
)


def parse_alpha_values(text):
    text = str(text).strip()
    if not text:
        raise ValueError("alpha values cannot be empty")
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        if len(parts) != 3:
            raise ValueError("alpha range must be start:end:step")
        start, end, step = parts
        if step <= 0:
            raise ValueError("alpha step must be positive")
        values = []
        current = start
        while current <= end + step * 1e-6:
            values.append(float(current))
            current += step
    else:
        values = [
            float(part)
            for part in text.replace(",", " ").split()
            if part
        ]

    unique = []
    seen = set()
    for value in values:
        if value < -1e-8 or value > 1.0 + 1e-8:
            raise ValueError(f"alpha must be in [0,1], got {value}")
        value = min(max(value, 0.0), 1.0)
        key = round(value, 10)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def point_adjust(prediction, labels):
    prediction = np.asarray(prediction, dtype=np.int64).copy()
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    index = 0
    while index < len(labels):
        if labels[index] != 1:
            index += 1
            continue
        end = index + 1
        while end < len(labels) and labels[end] == 1:
            end += 1
        if prediction[index:end].any():
            prediction[index:end] = 1
        index = end
    return prediction


def scores_for_point_adjustment(scores, labels):
    adjusted_scores = np.asarray(scores, dtype=np.float64).copy()
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    index = 0
    while index < len(labels):
        if labels[index] != 1:
            index += 1
            continue
        end = index + 1
        while end < len(labels) and labels[end] == 1:
            end += 1
        adjusted_scores[index:end] = np.max(adjusted_scores[index:end])
        index = end
    return adjusted_scores


def precision_recall_f1(prediction, labels):
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    tp = int(np.sum((prediction == 1) & (labels == 1)))
    fp = int(np.sum((prediction == 1) & (labels == 0)))
    fn = int(np.sum((prediction == 0) & (labels == 1)))
    precision = tp / float(tp + fp) if tp + fp > 0 else 0.0
    recall = tp / float(tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


def _better_threshold(candidate, best):
    if best is None:
        return True
    candidate_key = (
        candidate["f1"],
        candidate["precision"],
        candidate["threshold"],
    )
    best_key = (
        best["f1"],
        best["precision"],
        best["threshold"],
    )
    return candidate_key > best_key


def best_point_adjusted_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError(
            f"label/score length mismatch: {len(labels)} vs {len(scores)}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN/Inf")
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError("point-adjusted F1 requires positive labels")

    adjusted_scores = scores_for_point_adjustment(scores, labels)
    order = np.argsort(-adjusted_scores, kind="mergesort")
    sorted_scores = adjusted_scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    cumulative_fp = np.cumsum(sorted_labels == 0)

    group_ends = np.flatnonzero(
        np.r_[
            sorted_scores[:-1] != sorted_scores[1:],
            True,
        ]
    )
    best = None
    for group_end in group_ends:
        tp = int(cumulative_tp[group_end])
        fp = int(cumulative_fp[group_end])
        fn = positive_count - tp
        precision = tp / float(tp + fp) if tp + fp > 0 else 0.0
        recall = tp / float(tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        candidate = {
            "threshold": float(sorted_scores[group_end]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if _better_threshold(candidate, best):
            best = candidate

    raw_prediction = (scores >= best["threshold"]).astype(np.int64)
    adjusted_prediction = point_adjust(raw_prediction, labels)
    raw_precision, raw_recall, raw_f1 = precision_recall_f1(
        raw_prediction,
        labels,
    )
    adjusted_precision, adjusted_recall, adjusted_f1 = (
        precision_recall_f1(adjusted_prediction, labels)
    )
    best.update(
        {
            "precision": adjusted_precision,
            "recall": adjusted_recall,
            "f1": adjusted_f1,
            "raw_precision": raw_precision,
            "raw_recall": raw_recall,
            "raw_f1": raw_f1,
            "raw_prediction_count": int(raw_prediction.sum()),
            "adjusted_prediction_count": int(
                adjusted_prediction.sum()
            ),
        }
    )
    return best


def best_fusion(labels, recon_cdf, diffpath_cdf, alphas):
    best = None
    for alpha in alphas:
        fused = (
            float(alpha) * np.asarray(recon_cdf, dtype=np.float64)
            + (1.0 - float(alpha))
            * np.asarray(diffpath_cdf, dtype=np.float64)
        )
        result = best_point_adjusted_threshold(labels, fused)
        result["alpha"] = float(alpha)
        result["scores"] = fused.astype(np.float32)
        if best is None:
            best = result
            continue
        candidate_key = (
            result["f1"],
            result["precision"],
            result["threshold"],
            result["alpha"],
        )
        best_key = (
            best["f1"],
            best["precision"],
            best["threshold"],
            best["alpha"],
        )
        if candidate_key > best_key:
            best = result
    return best


def infer_save_id(path):
    name = os.path.basename(path)
    prefix = "diffpath_1d_scores_"
    suffix = ".npz"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return os.path.splitext(name)[0]


def load_score_file(path):
    with np.load(path, allow_pickle=False) as npz:
        payload = {
            key: np.asarray(npz[key])
            for key in npz.files
        }
    required = [
        "labels_aligned",
        "valid_indices",
        "final_recon_score",
        "diffpath_1d_raw_score",
        "recon_cdf",
        "diffpath_1d_cdf",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")

    labels = payload["labels_aligned"].astype(np.int64).reshape(-1)
    valid_indices = payload["valid_indices"].astype(
        np.int64
    ).reshape(-1)
    recon_sum_raw = payload.get(
        "final_recon_score_sum_abs",
        payload["final_recon_score"],
    ).astype(
        np.float64
    ).reshape(-1)
    recon_max_raw = None
    if "final_recon_score_max_abs" in payload:
        recon_max_raw = payload["final_recon_score_max_abs"].astype(
            np.float64
        ).reshape(-1)
    diffpath_raw = payload["diffpath_1d_raw_score"].astype(
        np.float64
    ).reshape(-1)
    recon_sum_cdf = payload.get(
        "recon_sum_abs_cdf",
        payload["recon_cdf"],
    ).astype(np.float64).reshape(-1)
    recon_max_cdf = None
    if "recon_max_abs_cdf" in payload:
        recon_max_cdf = payload["recon_max_abs_cdf"].astype(
            np.float64
        ).reshape(-1)
    if (recon_max_raw is None) != (recon_max_cdf is None):
        raise ValueError(
            f"{path} must contain both raw and CDF max-abs reconstruction scores"
        )
    diffpath_cdf = payload["diffpath_1d_cdf"].astype(
        np.float64
    ).reshape(-1)
    diffpath_6d_raw = None
    if "diffpath_6d_kde_raw_score" in payload:
        diffpath_6d_raw = payload["diffpath_6d_kde_raw_score"].astype(
            np.float64
        ).reshape(-1)
    elif "diffpath_6d_raw_score" in payload:
        diffpath_6d_raw = payload["diffpath_6d_raw_score"].astype(
            np.float64
        ).reshape(-1)
    diffpath_6d_cdf = None
    if "diffpath_6d_kde_cdf" in payload:
        diffpath_6d_cdf = payload["diffpath_6d_kde_cdf"].astype(
            np.float64
        ).reshape(-1)
    elif "diffpath_6d_cdf" in payload:
        diffpath_6d_cdf = payload["diffpath_6d_cdf"].astype(
            np.float64
        ).reshape(-1)
    if (diffpath_6d_raw is None) != (diffpath_6d_cdf is None):
        raise ValueError(
            f"{path} must contain both raw and CDF DiffPath-6D KDE scores"
        )
    diffpath_6d_gmm_raw = None
    if "diffpath_6d_gmm_raw_score" in payload:
        diffpath_6d_gmm_raw = payload[
            "diffpath_6d_gmm_raw_score"
        ].astype(np.float64).reshape(-1)
    diffpath_6d_gmm_cdf = None
    if "diffpath_6d_gmm_cdf" in payload:
        diffpath_6d_gmm_cdf = payload["diffpath_6d_gmm_cdf"].astype(
            np.float64
        ).reshape(-1)
    if (diffpath_6d_gmm_raw is None) != (diffpath_6d_gmm_cdf is None):
        raise ValueError(
            f"{path} must contain both raw and CDF DiffPath-6D-GMM scores"
        )
    score_lengths = {
        "labels": labels,
        "valid_indices": valid_indices,
        "final_recon_score": recon_sum_raw,
        "diffpath_1d_raw_score": diffpath_raw,
        "recon_cdf": recon_sum_cdf,
        "recon_sum_abs_cdf": recon_sum_cdf,
        "diffpath_1d_cdf": diffpath_cdf,
    }
    if recon_max_raw is not None:
        score_lengths["final_recon_score_max_abs"] = recon_max_raw
        score_lengths["recon_max_abs_cdf"] = recon_max_cdf
    if diffpath_6d_raw is not None:
        score_lengths["diffpath_6d_raw_score"] = diffpath_6d_raw
        score_lengths["diffpath_6d_cdf"] = diffpath_6d_cdf
    if diffpath_6d_gmm_raw is not None:
        score_lengths["diffpath_6d_gmm_raw_score"] = (
            diffpath_6d_gmm_raw
        )
        score_lengths["diffpath_6d_gmm_cdf"] = diffpath_6d_gmm_cdf
    bad_lengths = {
        name: len(values)
        for name, values in score_lengths.items()
        if len(values) != len(labels)
    }
    if bad_lengths:
        lengths = {
            name: len(values)
            for name, values in score_lengths.items()
        }
        raise ValueError(f"{path} aligned lengths differ: {lengths}")
    loaded = {
        "path": path,
        "payload": payload,
        "save": infer_save_id(path),
        "labels": labels,
        "valid_indices": valid_indices,
        "recon_raw": recon_sum_raw,
        "recon_sum_raw": recon_sum_raw,
        "diffpath_raw": diffpath_raw,
        "recon_cdf": recon_sum_cdf,
        "recon_sum_cdf": recon_sum_cdf,
        "diffpath_cdf": diffpath_cdf,
    }
    if recon_max_raw is not None:
        loaded["recon_max_raw"] = recon_max_raw
        loaded["recon_max_cdf"] = recon_max_cdf
    if diffpath_6d_raw is not None:
        loaded["diffpath_6d_raw"] = diffpath_6d_raw
        loaded["diffpath_6d_cdf"] = diffpath_6d_cdf
    if diffpath_6d_gmm_raw is not None:
        loaded["diffpath_6d_gmm_raw"] = diffpath_6d_gmm_raw
        loaded["diffpath_6d_gmm_cdf"] = diffpath_6d_gmm_cdf
    return loaded


def result_row(
    dataset,
    save,
    method,
    result,
    label_count,
    row_type="per_save",
):
    return {
        "dataset": dataset,
        "row_type": row_type,
        "save": save,
        "method": method,
        "alpha": result.get("alpha", ""),
        "threshold": result.get("threshold", ""),
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
        "raw_precision": result.get("raw_precision", ""),
        "raw_recall": result.get("raw_recall", ""),
        "raw_f1": result.get("raw_f1", ""),
        "raw_prediction_count": result.get(
            "raw_prediction_count",
            "",
        ),
        "adjusted_prediction_count": result.get(
            "adjusted_prediction_count",
            "",
        ),
        "label_count": int(label_count),
        "std_ddof": "",
    }


def summary_rows(dataset, per_save_rows):
    rows = []
    for method in METHOD_ORDER:
        method_rows = [
            row
            for row in per_save_rows
            if row["method"] == method
        ]
        if not method_rows:
            continue
        for row_type, reducer in (
            ("mean", np.mean),
            ("std", None),
        ):
            summary = {
                "dataset": dataset,
                "row_type": row_type,
                "save": "",
                "method": method,
                "alpha": "",
                "threshold": "",
                "raw_precision": "",
                "raw_recall": "",
                "raw_f1": "",
                "raw_prediction_count": "",
                "adjusted_prediction_count": "",
                "label_count": method_rows[0]["label_count"],
                "std_ddof": 1 if row_type == "std" else "",
            }
            for metric in ("precision", "recall", "f1"):
                values = np.asarray(
                    [row[metric] for row in method_rows],
                    dtype=np.float64,
                )
                if row_type == "mean":
                    summary[metric] = float(reducer(values))
                else:
                    summary[metric] = (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    )
            rows.append(summary)
    return rows


def save_best_fusion(loaded, dataset, result, suffix, formula):
    ensemble_dir = os.path.dirname(loaded["path"])
    save_id = loaded["save"]
    score_path = os.path.join(
        ensemble_dir,
        f"diffpath_{suffix}_fused_best_ensemble_{save_id}.pt",
    )
    metadata_path = os.path.join(
        ensemble_dir,
        f"diffpath_{suffix}_fused_best_{save_id}.json",
    )
    torch.save(
        torch.from_numpy(result["scores"]),
        score_path,
    )
    metadata = {
        "dataset": dataset,
        "save": save_id,
        "alpha": result["alpha"],
        "threshold": result["threshold"],
        "point_adjusted_precision": result["precision"],
        "point_adjusted_recall": result["recall"],
        "point_adjusted_f1": result["f1"],
        "raw_precision": result["raw_precision"],
        "raw_recall": result["raw_recall"],
        "raw_f1": result["raw_f1"],
        "evaluation_protocol": (
            "per-save test-set alpha and threshold search with "
            "point adjustment"
        ),
        "fusion_formula": formula,
        "source_score_file": loaded["path"],
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def evaluate_dataset(root, dataset, alphas, saves=None):
    ensemble_dir = os.path.join(root, dataset, "ensemble")
    paths = sorted(
        glob.glob(
            os.path.join(
                ensemble_dir,
                "diffpath_1d_scores_*.npz",
            )
        )
    )
    if saves:
        requested_saves = list(saves)
        paths_by_save = {
            infer_save_id(path): path
            for path in paths
        }
        missing = [
            save_id
            for save_id in requested_saves
            if save_id not in paths_by_save
        ]
        if missing:
            raise FileNotFoundError(
                f"{dataset}: missing DiffPath score files for "
                f"{missing} under {ensemble_dir}"
            )
        paths = [
            paths_by_save[save_id]
            for save_id in requested_saves
        ]
    if not paths:
        raise FileNotFoundError(
            f"no DiffPath score files found under {ensemble_dir}"
        )
    if saves and len(saves) != 3:
        raise ValueError(
            "the repeated-experiment protocol requires exactly 3 saves"
        )
    if not saves and len(paths) != 3:
        print(
            f"[WARN] {dataset}: expected 3 repeated runs, found "
            f"{len(paths)}"
        )

    per_save_rows = []
    reference_indices = None
    reference_labels = None
    for path in paths:
        loaded = load_score_file(path)
        labels = loaded["labels"]
        if reference_indices is None:
            reference_indices = loaded["valid_indices"]
            reference_labels = labels
        elif (
            not np.array_equal(
                loaded["valid_indices"],
                reference_indices,
            )
            or not np.array_equal(labels, reference_labels)
        ):
            raise ValueError(
                f"{dataset}: repeated runs do not use identical aligned "
                "test points and labels"
            )
        recon_sum_result = best_point_adjusted_threshold(
            labels,
            loaded["recon_sum_raw"],
        )
        recon_max_result = None
        if "recon_max_raw" in loaded:
            recon_max_result = best_point_adjusted_threshold(
                labels,
                loaded["recon_max_raw"],
            )
        diffpath_result = best_point_adjusted_threshold(
            labels,
            loaded["diffpath_raw"],
        )
        fused_1d_sum_result = best_fusion(
            labels,
            loaded["recon_sum_cdf"],
            loaded["diffpath_cdf"],
            alphas,
        )
        save_best_fusion(
            loaded,
            dataset,
            fused_1d_sum_result,
            suffix="sum_abs_1d",
            formula=(
                "alpha * recon_sum_abs_cdf + "
                "(1-alpha) * diffpath_1d_cdf"
            ),
        )
        fused_1d_max_result = None
        if "recon_max_cdf" in loaded:
            fused_1d_max_result = best_fusion(
                labels,
                loaded["recon_max_cdf"],
                loaded["diffpath_cdf"],
                alphas,
            )
            save_best_fusion(
                loaded,
                dataset,
                fused_1d_max_result,
                suffix="max_abs_1d",
                formula=(
                    "alpha * recon_max_abs_cdf + "
                    "(1-alpha) * diffpath_1d_cdf"
                ),
            )
        diffpath_6d_result = None
        fused_6d_sum_result = None
        fused_6d_max_result = None
        if "diffpath_6d_cdf" in loaded:
            diffpath_6d_result = best_point_adjusted_threshold(
                labels,
                loaded["diffpath_6d_raw"],
            )
            fused_6d_sum_result = best_fusion(
                labels,
                loaded["recon_sum_cdf"],
                loaded["diffpath_6d_cdf"],
                alphas,
            )
            save_best_fusion(
                loaded,
                dataset,
                fused_6d_sum_result,
                suffix="sum_abs_6d",
                formula=(
                    "alpha * recon_sum_abs_cdf + "
                    "(1-alpha) * diffpath_6d_cdf"
                ),
            )
            if "recon_max_cdf" in loaded:
                fused_6d_max_result = best_fusion(
                    labels,
                    loaded["recon_max_cdf"],
                    loaded["diffpath_6d_cdf"],
                    alphas,
                )
                save_best_fusion(
                    loaded,
                    dataset,
                    fused_6d_max_result,
                    suffix="max_abs_6d",
                    formula=(
                        "alpha * recon_max_abs_cdf + "
                        "(1-alpha) * diffpath_6d_cdf"
                    ),
                )
        diffpath_6d_gmm_result = None
        fused_6d_gmm_sum_result = None
        fused_6d_gmm_max_result = None
        if "diffpath_6d_gmm_cdf" in loaded:
            diffpath_6d_gmm_result = best_point_adjusted_threshold(
                labels,
                loaded["diffpath_6d_gmm_raw"],
            )
            fused_6d_gmm_sum_result = best_fusion(
                labels,
                loaded["recon_sum_cdf"],
                loaded["diffpath_6d_gmm_cdf"],
                alphas,
            )
            save_best_fusion(
                loaded,
                dataset,
                fused_6d_gmm_sum_result,
                suffix="sum_abs_6d_gmm",
                formula=(
                    "alpha * recon_sum_abs_cdf + "
                    "(1-alpha) * diffpath_6d_gmm_cdf"
                ),
            )
            if "recon_max_cdf" in loaded:
                fused_6d_gmm_max_result = best_fusion(
                    labels,
                    loaded["recon_max_cdf"],
                    loaded["diffpath_6d_gmm_cdf"],
                    alphas,
                )
                save_best_fusion(
                    loaded,
                    dataset,
                    fused_6d_gmm_max_result,
                    suffix="max_abs_6d_gmm",
                    formula=(
                        "alpha * recon_max_abs_cdf + "
                        "(1-alpha) * diffpath_6d_gmm_cdf"
                    ),
                )

        save_rows = [
            result_row(
                dataset,
                loaded["save"],
                "final_reconstruction",
                recon_sum_result,
                labels.sum(),
            ),
            result_row(
                dataset,
                loaded["save"],
                "final_reconstruction_sum_abs",
                recon_sum_result,
                labels.sum(),
            ),
            result_row(
                dataset,
                loaded["save"],
                "diffpath_1d",
                diffpath_result,
                labels.sum(),
            ),
            result_row(
                dataset,
                loaded["save"],
                "fused_reconstruction_diffpath_1d",
                fused_1d_sum_result,
                labels.sum(),
            ),
            result_row(
                dataset,
                loaded["save"],
                "fused_sum_abs_reconstruction_diffpath_1d",
                fused_1d_sum_result,
                labels.sum(),
            ),
        ]
        if recon_max_result is not None:
            save_rows.extend(
                [
                    result_row(
                        dataset,
                        loaded["save"],
                        "final_reconstruction_max_abs",
                        recon_max_result,
                        labels.sum(),
                    ),
                    result_row(
                        dataset,
                        loaded["save"],
                        "fused_max_abs_reconstruction_diffpath_1d",
                        fused_1d_max_result,
                        labels.sum(),
                    ),
                ]
            )
        if diffpath_6d_result is not None:
            save_rows.extend(
                [
                    result_row(
                        dataset,
                        loaded["save"],
                        "diffpath_6d",
                        diffpath_6d_result,
                        labels.sum(),
                    ),
                    result_row(
                        dataset,
                        loaded["save"],
                        "fused_reconstruction_diffpath_6d",
                        fused_6d_sum_result,
                        labels.sum(),
                    ),
                    result_row(
                        dataset,
                        loaded["save"],
                        "fused_sum_abs_reconstruction_diffpath_6d",
                        fused_6d_sum_result,
                        labels.sum(),
                    ),
                ]
            )
            if fused_6d_max_result is not None:
                save_rows.append(
                    result_row(
                        dataset,
                        loaded["save"],
                        "fused_max_abs_reconstruction_diffpath_6d",
                        fused_6d_max_result,
                        labels.sum(),
                    )
                )
        if diffpath_6d_gmm_result is not None:
            save_rows.extend(
                [
                    result_row(
                        dataset,
                        loaded["save"],
                        "diffpath_6d_gmm",
                        diffpath_6d_gmm_result,
                        labels.sum(),
                    ),
                    result_row(
                        dataset,
                        loaded["save"],
                        "fused_reconstruction_diffpath_6d_gmm",
                        fused_6d_gmm_sum_result,
                        labels.sum(),
                    ),
                    result_row(
                        dataset,
                        loaded["save"],
                        "fused_sum_abs_reconstruction_diffpath_6d_gmm",
                        fused_6d_gmm_sum_result,
                        labels.sum(),
                    ),
                ]
            )
            if fused_6d_gmm_max_result is not None:
                save_rows.append(
                    result_row(
                        dataset,
                        loaded["save"],
                        "fused_max_abs_reconstruction_diffpath_6d_gmm",
                        fused_6d_gmm_max_result,
                        labels.sum(),
                    )
                )
        per_save_rows.extend(save_rows)

        message = (
            f"[DiffPath-F1] {dataset} {loaded['save']}: "
            f"recon_sum_raw_f1={recon_sum_result['f1']:.6f}, "
            f"diffpath_1d_raw_f1={diffpath_result['f1']:.6f}, "
            f"fused_sum_1d_f1={fused_1d_sum_result['f1']:.6f}, "
            f"alpha_sum_1d={fused_1d_sum_result['alpha']:.2f}"
        )
        if recon_max_result is not None:
            message += (
                f", recon_max_raw_f1={recon_max_result['f1']:.6f}, "
                f"fused_max_1d_f1={fused_1d_max_result['f1']:.6f}, "
                f"alpha_max_1d={fused_1d_max_result['alpha']:.2f}"
            )
        if diffpath_6d_result is not None:
            message += (
                f", diffpath_6d_raw_f1={diffpath_6d_result['f1']:.6f}, "
                f"fused_sum_6d_f1={fused_6d_sum_result['f1']:.6f}, "
                f"alpha_sum_6d={fused_6d_sum_result['alpha']:.2f}"
            )
            if fused_6d_max_result is not None:
                message += (
                    f", fused_max_6d_f1={fused_6d_max_result['f1']:.6f}, "
                    f"alpha_max_6d={fused_6d_max_result['alpha']:.2f}"
                )
        if diffpath_6d_gmm_result is not None:
            message += (
                f", diffpath_6d_gmm_raw_f1={diffpath_6d_gmm_result['f1']:.6f}, "
                f"fused_sum_6d_gmm_f1={fused_6d_gmm_sum_result['f1']:.6f}, "
                f"alpha_sum_6d_gmm={fused_6d_gmm_sum_result['alpha']:.2f}"
            )
            if fused_6d_gmm_max_result is not None:
                message += (
                    f", fused_max_6d_gmm_f1={fused_6d_gmm_max_result['f1']:.6f}, "
                    f"alpha_max_6d_gmm={fused_6d_gmm_max_result['alpha']:.2f}"
                )
        print(message)
    return per_save_rows + summary_rows(dataset, per_save_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pathB_output_root",
        type=str,
        default="pathB_result",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SMAP"],
    )
    parser.add_argument(
        "--alpha_values",
        type=str,
        default="0:1:0.05",
    )
    parser.add_argument(
        "--saves",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--out",
        type=str,
        default="pathB_diffpath_f1_summary.csv",
    )
    args = parser.parse_args()

    alphas = parse_alpha_values(args.alpha_values)
    rows = []
    for dataset in args.datasets:
        rows.extend(
            evaluate_dataset(
                args.pathB_output_root,
                dataset,
                alphas,
                saves=args.saves,
            )
        )

    fieldnames = [
        "dataset",
        "row_type",
        "save",
        "method",
        "alpha",
        "threshold",
        "precision",
        "recall",
        "f1",
        "raw_precision",
        "raw_recall",
        "raw_f1",
        "raw_prediction_count",
        "adjusted_prediction_count",
        "label_count",
        "std_ddof",
    ]
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DiffPath-F1] saved summary: {args.out}")


if __name__ == "__main__":
    main()
