import argparse
import csv
import glob
import os

import numpy as np


DEFAULT_VARIANTS = [
    "SMAP_MVE_d01",
    "SMAP_MVE_d02",
    "SMAP_MVE_d03",
    "SMAP_MVE_d05",
    "SMAP_MVE_d08",
    "SMAP_MVE_d10",
]

HDSAC_METHODS = [
    "hdsac_ch_mean_step_mean",
    "hdsac_ch_mean_step_median",
    "hdsac_ch_mean_step_max",
    "hdsac_ch_max_step_mean",
    "hdsac_ch_max_step_median",
    "hdsac_ch_max_step_max",
    "hdsac_ch_top3_step_mean",
    "hdsac_ch_top3_step_median",
    "hdsac_ch_top3_step_max",
    "hdsac_ch_top3_step_top2mean",
]

DEFAULT_METHODS = [
    "final_recon_score",
] + HDSAC_METHODS
DEFAULT_METHODS = DEFAULT_METHODS + [f"fused_{method}" for method in HDSAC_METHODS]


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def roc_auc_score(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    pos_rank_sum = ranks[pos].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_recall_scores(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos_count = int(labels.sum())
    neg_count = int((labels == 0).sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan"), float("nan")

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1).astype(np.float64)
    fp = np.cumsum(sorted_labels == 0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(pos_count)

    recall_with_origin = np.concatenate([[0.0], recall])
    precision_with_origin = np.concatenate([[1.0], precision])
    auc_pr = float(np.trapz(precision_with_origin, recall_with_origin))
    ap = float(np.sum((recall_with_origin[1:] - recall_with_origin[:-1]) * precision))
    return auc_pr, ap


def infer_save_from_path(path, variant):
    name = os.path.basename(path)
    prefix = f"{variant}_"
    suffix = "_scores.npz"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return os.path.splitext(name)[0]


def find_score_files(root, base_dataset, variant):
    search_path = os.path.join(root, base_dataset, f"{variant}_*_scores.npz")
    return sorted(glob.glob(search_path))


def scalar_from_npz(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def parse_alpha_values(text):
    text = str(text).strip()
    if not text:
        raise ValueError("alpha values cannot be empty")
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        if len(parts) != 3:
            raise ValueError("--alpha_values range must be start:end:step")
        start, end, step = parts
        if step <= 0:
            raise ValueError("--alpha_values step must be positive")
        values = []
        current = start
        # Include the end point with a small tolerance for floating point steps.
        while current <= end + (step * 1e-6):
            values.append(current)
            current += step
    else:
        values = [float(part) for part in text.split(",") if part.strip()]
    clipped = []
    for value in values:
        if value < -1e-8 or value > 1.0 + 1e-8:
            raise ValueError(f"alpha must be in [0, 1], got {value}")
        clipped.append(min(max(value, 0.0), 1.0))
    # Keep stable order but remove duplicates after rounding.
    seen = set()
    unique = []
    for value in clipped:
        key = round(value, 10)
        if key not in seen:
            seen.add(key)
            unique.append(float(value))
    return unique


def metric_value(metrics, metric_name):
    if metric_name == "auc_roc":
        return metrics[0]
    if metric_name == "auc_pr":
        return metrics[1]
    if metric_name == "ap":
        return metrics[2]
    raise ValueError(f"unknown alpha metric: {metric_name}")


def compute_metrics(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    label_sum = int(labels.sum())
    label_len = int(len(labels))
    if label_sum == 0 or label_sum == label_len:
        return float("nan"), float("nan"), float("nan")
    auc_roc = roc_auc_score(labels, scores)
    auc_pr, ap = precision_recall_scores(labels, scores)
    return auc_roc, auc_pr, ap


def make_row(
    variant,
    save_id,
    method,
    metrics,
    labels,
    score_len,
    raw_label_len,
    valid_indices_len,
    raw_label_sum,
    dropped_points,
    row_type,
    alpha="",
    alpha_metric="",
):
    return [
        variant,
        save_id,
        method,
        metrics[0],
        metrics[1],
        metrics[2],
        int(np.asarray(labels).sum()),
        int(score_len),
        int(len(labels)),
        int(raw_label_len),
        int(valid_indices_len),
        int(raw_label_sum),
        int(dropped_points),
        row_type,
        alpha,
        alpha_metric,
    ]


def load_score_file(path, variant):
    with np.load(path, allow_pickle=False) as npz:
        payload = {key: np.asarray(npz[key]) for key in npz.files}
    raw_labels = np.asarray(payload["labels"], dtype=np.int64).reshape(-1)
    if "valid_indices" in payload:
        valid_indices = np.asarray(payload["valid_indices"], dtype=np.int64).reshape(-1)
    else:
        first_score_key = next(
            (key for key in payload if key in DEFAULT_METHODS),
            None,
        )
        if first_score_key is None:
            raise RuntimeError(f"{path} has no known score method and no valid_indices")
        print(f"[WARN] {path} has no valid_indices; falling back to contiguous score region")
        valid_indices = np.arange(len(np.asarray(payload[first_score_key]).reshape(-1)), dtype=np.int64)
    if len(valid_indices) == 0:
        raise RuntimeError(f"{path} has empty valid_indices")
    if valid_indices.max() >= len(raw_labels):
        raise RuntimeError(
            f"{path} valid_indices exceed raw labels: max={valid_indices.max()}, raw_label_len={len(raw_labels)}"
        )
    labels = raw_labels[valid_indices]
    raw_label_sum = int(raw_labels.sum())
    aligned_label_sum = int(labels.sum())
    raw_label_len = int(len(raw_labels))
    valid_indices_len = int(len(valid_indices))
    dropped_points = int(raw_label_len - valid_indices_len)
    save_id = infer_save_from_path(path, variant)
    if aligned_label_sum < raw_label_sum:
        print(
            f"[WARN] {variant} {save_id}: aligned_label_sum < raw_label_sum "
            f"({aligned_label_sum} < {raw_label_sum}); anomaly points were dropped from valid-region evaluation"
        )
    return {
        "path": path,
        "payload": payload,
        "raw_labels": raw_labels,
        "labels": labels,
        "valid_indices": valid_indices,
        "save_id": save_id,
        "raw_label_sum": raw_label_sum,
        "aligned_label_sum": aligned_label_sum,
        "raw_label_len": raw_label_len,
        "valid_indices_len": valid_indices_len,
        "dropped_points": dropped_points,
    }


def evaluate_loaded_file(loaded, variant, methods):
    payload = loaded["payload"]
    labels = loaded["labels"]
    save_id = loaded["save_id"]
    rows = []
    for method in methods:
        if method not in payload:
            print(f"[WARN] {loaded['path']} missing method {method}; skipping")
            continue
        scores = np.asarray(payload[method], dtype=np.float64).reshape(-1)
        score_len = int(len(scores))
        label_len = int(len(labels))
        if score_len != loaded["valid_indices_len"]:
            raise RuntimeError(
                f"{variant} {save_id} {method}: score_len={score_len} "
                f"!= valid_indices_len={loaded['valid_indices_len']}"
            )
        if score_len != label_len:
            raise RuntimeError(
                f"{variant} {save_id} {method}: score_len={score_len} "
                f"!= aligned label_len={label_len}"
            )

        metrics = compute_metrics(labels, scores)
        print(
            f"[HDSAC-AUC] {variant} {save_id} {method}: "
            f"auc_roc={metrics[0]}, auc_pr={metrics[1]}, ap={metrics[2]}, "
            f"raw_label_len={loaded['raw_label_len']}, score_len={score_len}, "
            f"valid_indices_len={loaded['valid_indices_len']}, raw_label_sum={loaded['raw_label_sum']}, "
            f"aligned_label_sum={loaded['aligned_label_sum']}, dropped_points={loaded['dropped_points']}"
        )
        rows.append(
            make_row(
                variant,
                save_id,
                method,
                metrics,
                labels,
                score_len,
                loaded["raw_label_len"],
                loaded["valid_indices_len"],
                loaded["raw_label_sum"],
                loaded["dropped_points"],
                "per_save",
            )
        )
    return rows


def assert_same_alignment(loaded_files, variant):
    if not loaded_files:
        raise RuntimeError(f"{variant}: no files to align")
    first = loaded_files[0]
    for current in loaded_files[1:]:
        if not np.array_equal(first["valid_indices"], current["valid_indices"]):
            raise RuntimeError(
                f"{variant}: valid_indices differ between {first['save_id']} and {current['save_id']}; "
                "cannot compute save_mean ensemble"
            )
        if not np.array_equal(first["labels"], current["labels"]):
            raise RuntimeError(
                f"{variant}: aligned labels differ between {first['save_id']} and {current['save_id']}"
            )


def evaluate_score_mean(loaded_files, variant, methods):
    assert_same_alignment(loaded_files, variant)
    if len(loaded_files) != 3:
        print(f"[WARN] {variant}: score_mean is averaging {len(loaded_files)} saves, not exactly 3")
    ref = loaded_files[0]
    labels = ref["labels"]
    rows = []
    for method in methods:
        missing = [loaded["save_id"] for loaded in loaded_files if method not in loaded["payload"]]
        if missing:
            print(f"[WARN] {variant} score_mean missing method {method} in saves {missing}; skipping")
            continue
        stacked = np.stack(
            [
                np.asarray(loaded["payload"][method], dtype=np.float64).reshape(-1)
                for loaded in loaded_files
            ],
            axis=0,
        )
        scores = stacked.mean(axis=0)
        if len(scores) != len(labels):
            raise RuntimeError(f"{variant} score_mean {method}: score_len={len(scores)} != label_len={len(labels)}")
        metrics = compute_metrics(labels, scores)
        print(
            f"[HDSAC-AUC] {variant} score_mean {method}: "
            f"auc_roc={metrics[0]}, auc_pr={metrics[1]}, ap={metrics[2]}, saves={len(loaded_files)}"
        )
        rows.append(
            make_row(
                variant,
                "score_mean",
                method,
                metrics,
                labels,
                len(scores),
                ref["raw_label_len"],
                ref["valid_indices_len"],
                ref["raw_label_sum"],
                ref["dropped_points"],
                "score_mean",
            )
        )
    return rows


def nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(values[finite].mean())


def evaluate_save_metric_mean(loaded_files, variant, methods):
    assert_same_alignment(loaded_files, variant)
    if len(loaded_files) != 3:
        print(f"[WARN] {variant}: save_metric_mean is averaging {len(loaded_files)} saves, not exactly 3")
    ref = loaded_files[0]
    labels = ref["labels"]
    rows = []
    for method in methods:
        missing = [loaded["save_id"] for loaded in loaded_files if method not in loaded["payload"]]
        if missing:
            print(f"[WARN] {variant} save_metric_mean missing method {method} in saves {missing}; skipping")
            continue
        metrics_by_save = []
        for loaded in loaded_files:
            scores = np.asarray(loaded["payload"][method], dtype=np.float64).reshape(-1)
            metrics_by_save.append(compute_metrics(loaded["labels"], scores))
        metrics = (
            nanmean([item[0] for item in metrics_by_save]),
            nanmean([item[1] for item in metrics_by_save]),
            nanmean([item[2] for item in metrics_by_save]),
        )
        print(
            f"[HDSAC-AUC] {variant} save_metric_mean {method}: "
            f"auc_roc={metrics[0]}, auc_pr={metrics[1]}, ap={metrics[2]}, saves={len(loaded_files)}"
        )
        rows.append(
            make_row(
                variant,
                "save_metric_mean",
                method,
                metrics,
                labels,
                ref["valid_indices_len"],
                ref["raw_label_len"],
                ref["valid_indices_len"],
                ref["raw_label_sum"],
                ref["dropped_points"],
                "save_metric_mean",
            )
        )
    return rows


def best_alpha_for_scores(labels, z_recon, z_hdsac, alphas, alpha_metric):
    best = None
    for alpha in alphas:
        scores = (float(alpha) * z_recon) + ((1.0 - float(alpha)) * z_hdsac)
        metrics = compute_metrics(labels, scores)
        value = metric_value(metrics, alpha_metric)
        if np.isnan(value):
            continue
        if best is None or value > best["value"]:
            best = {
                "alpha": float(alpha),
                "scores": scores,
                "metrics": metrics,
                "value": float(value),
            }
    return best


def evaluate_alpha_search_per_save(loaded_files, variant, hdsac_methods, alphas, alpha_metric):
    rows = []
    for loaded in loaded_files:
        payload = loaded["payload"]
        labels = loaded["labels"]
        save_id = loaded["save_id"]
        if "z_final_recon_score" not in payload:
            print(f"[WARN] {variant} {save_id}: missing z_final_recon_score; skipping alpha search")
            continue
        z_recon = np.asarray(payload["z_final_recon_score"], dtype=np.float64).reshape(-1)
        for method in hdsac_methods:
            z_key = f"z_{method}"
            if z_key not in payload:
                print(f"[WARN] {variant} {save_id}: missing {z_key}; skipping alpha search for {method}")
                continue
            z_hdsac = np.asarray(payload[z_key], dtype=np.float64).reshape(-1)
            best = best_alpha_for_scores(labels, z_recon, z_hdsac, alphas, alpha_metric)
            if best is None:
                continue
            print(
                f"[HDSAC-AUC] {variant} {save_id} alpha_search_{method}: "
                f"best_alpha={best['alpha']}, best_{alpha_metric}={best['value']}, "
                f"auc_roc={best['metrics'][0]}, auc_pr={best['metrics'][1]}, ap={best['metrics'][2]}"
            )
            rows.append(
                make_row(
                    variant,
                    save_id,
                    f"fused_alpha_search_{method}",
                    best["metrics"],
                    labels,
                    len(best["scores"]),
                    loaded["raw_label_len"],
                    loaded["valid_indices_len"],
                    loaded["raw_label_sum"],
                    loaded["dropped_points"],
                    "alpha_search_per_save",
                    best["alpha"],
                    alpha_metric,
                )
            )
    return rows


def evaluate_alpha_search_save_metric_mean(loaded_files, variant, hdsac_methods, alphas, alpha_metric):
    assert_same_alignment(loaded_files, variant)
    ref = loaded_files[0]
    labels = ref["labels"]
    rows = []
    for method in hdsac_methods:
        if any("z_final_recon_score" not in loaded["payload"] for loaded in loaded_files):
            print(f"[WARN] {variant}: missing z_final_recon_score; skipping save_metric_mean alpha search")
            return rows
        z_key = f"z_{method}"
        missing = [loaded["save_id"] for loaded in loaded_files if z_key not in loaded["payload"]]
        if missing:
            print(f"[WARN] {variant} save_metric_mean: missing {z_key} in saves {missing}; skipping")
            continue
        best = None
        for alpha in alphas:
            metrics_by_save = []
            for loaded in loaded_files:
                z_recon = np.asarray(loaded["payload"]["z_final_recon_score"], dtype=np.float64).reshape(-1)
                z_hdsac = np.asarray(loaded["payload"][z_key], dtype=np.float64).reshape(-1)
                # Fusion is done at score level for each save. Metrics are averaged only after
                # each save has produced its own fused-score AUC/AP.
                fused_scores = (float(alpha) * z_recon) + ((1.0 - float(alpha)) * z_hdsac)
                metrics_by_save.append(compute_metrics(loaded["labels"], fused_scores))
            metrics = (
                nanmean([item[0] for item in metrics_by_save]),
                nanmean([item[1] for item in metrics_by_save]),
                nanmean([item[2] for item in metrics_by_save]),
            )
            value = metric_value(metrics, alpha_metric)
            if np.isnan(value):
                continue
            if best is None or value > best["value"]:
                best = {
                    "alpha": float(alpha),
                    "metrics": metrics,
                    "value": float(value),
                }
        if best is None:
            continue
        print(
            f"[HDSAC-AUC] {variant} save_metric_mean alpha_search_{method}: "
            f"best_alpha={best['alpha']}, best_{alpha_metric}={best['value']}, "
            f"auc_roc={best['metrics'][0]}, auc_pr={best['metrics'][1]}, ap={best['metrics'][2]}"
        )
        rows.append(
            make_row(
                variant,
                "save_metric_mean",
                f"fused_alpha_search_{method}",
                best["metrics"],
                labels,
                ref["valid_indices_len"],
                ref["raw_label_len"],
                ref["valid_indices_len"],
                ref["raw_label_sum"],
                ref["dropped_points"],
                "alpha_search_save_metric_mean",
                best["alpha"],
                alpha_metric,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathB_output_root", type=str, default="pathB_result_hdsac_debug")
    parser.add_argument("--base_dataset", type=str, default="SMAP")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--hdsac_methods", nargs="*", default=HDSAC_METHODS)
    parser.add_argument("--fusion_base_methods", nargs="*", default=None)
    parser.add_argument("--out", type=str, default="pathB_hdsac_auc.csv")
    parser.add_argument(
        "--alpha_values",
        type=str,
        default="0:1:0.05",
        help="Alpha grid for fusion search. Use comma list or start:end:step.",
    )
    parser.add_argument("--alpha_metric", choices=["auc_roc", "auc_pr", "ap"], default="auc_roc")
    parser.add_argument(
        "--include_score_mean",
        action="store_true",
        help="Also evaluate score-level save averaging. Off by default; main ensemble is save_metric_mean.",
    )
    parser.add_argument("--no_save_metric_mean", action="store_true")
    parser.add_argument("--no_alpha_search", action="store_true")
    args = parser.parse_args()

    alphas = parse_alpha_values(args.alpha_values)
    fusion_base_methods = args.fusion_base_methods if args.fusion_base_methods is not None else args.hdsac_methods
    rows = []
    for variant in args.variants:
        paths = find_score_files(args.pathB_output_root, args.base_dataset, variant)
        if not paths:
            raise FileNotFoundError(
                f"no HDSAC score files found for {variant} under "
                f"{os.path.join(args.pathB_output_root, args.base_dataset)}"
            )
        loaded_files = [load_score_file(path, variant) for path in paths]
        for path in paths:
            loaded = next(loaded for loaded in loaded_files if loaded["path"] == path)
            rows.extend(evaluate_loaded_file(loaded, variant, args.methods))
        if args.include_score_mean:
            rows.extend(evaluate_score_mean(loaded_files, variant, args.methods))
        if not args.no_save_metric_mean:
            rows.extend(evaluate_save_metric_mean(loaded_files, variant, args.methods))
        if not args.no_alpha_search:
            rows.extend(
                evaluate_alpha_search_per_save(
                    loaded_files,
                    variant,
                    fusion_base_methods,
                    alphas,
                    args.alpha_metric,
                )
            )
            rows.extend(
                evaluate_alpha_search_save_metric_mean(
                    loaded_files,
                    variant,
                    fusion_base_methods,
                    alphas,
                    args.alpha_metric,
                )
            )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant",
                "save",
                "method",
                "auc_roc",
                "auc_pr",
                "ap",
                "label_sum",
                "score_len",
                "label_len",
                "raw_label_len",
                "valid_indices_len",
                "raw_label_sum",
                "dropped_points",
                "row_type",
                "alpha",
                "alpha_metric",
            ]
        )
        writer.writerows(rows)
    print(f"[HDSAC-AUC] saved summary to {args.out}")


if __name__ == "__main__":
    main()
