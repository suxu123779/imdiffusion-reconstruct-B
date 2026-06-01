import argparse
import csv
import glob
import os
import pickle
import re

import numpy as np
import torch


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_meta(path):
    if not path or not os.path.exists(path):
        return {}
    return load_pickle(path)


def tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float().reshape(-1).numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def average_ranks(values):
    values = np.asarray(values)
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


def percentile_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return (average_ranks(values) - 1.0) / (len(values) - 1.0)


def roc_auc_score(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    positive = labels == 1
    negative = labels == 0
    n_pos = int(np.sum(positive))
    n_neg = int(np.sum(negative))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = average_ranks(scores)
    pos_rank_sum = np.sum(ranks[positive])
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def infer_run_id(path, score_name):
    name = os.path.basename(path)
    prefix = f"{score_name}_"
    suffix = ".pt"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]
    return os.path.splitext(name)[0]


def load_score_runs(ensemble_dir, score_name):
    paths = sorted(glob.glob(os.path.join(ensemble_dir, f"{score_name}_*.pt")))
    if len(paths) == 0:
        raise FileNotFoundError(f"no {score_name}_*.pt files found in {ensemble_dir}")

    runs = {}
    for path in paths:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            value = value[score_name]
        runs[infer_run_id(path, score_name)] = tensor_to_numpy(value)
    return runs


def mad(values):
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values)
    deviation = np.median(np.abs(values - center))
    return center, max(float(deviation), 1e-12)


def clean_calibrate(variant_scores, clean_scores):
    clean_center, clean_mad = mad(clean_scores)
    return np.abs(np.asarray(variant_scores, dtype=np.float64) - clean_center) / clean_mad


def ensemble_scores(score_runs, mode):
    min_len = min(len(score) for score in score_runs)
    aligned = [np.asarray(score[:min_len], dtype=np.float64) for score in score_runs]

    if mode == "rank":
        aligned = [percentile_ranks(score) for score in aligned]
    elif mode == "zscore":
        aligned = [
            (score - np.mean(score)) / max(float(np.std(score)), 1e-12)
            for score in aligned
        ]
    elif mode == "mean":
        pass
    else:
        raise ValueError(f"unknown ensemble mode: {mode}")

    return np.mean(np.stack(aligned, axis=0), axis=0)


def infer_delta(variant, meta):
    variant_meta = meta.get(variant, {})
    if isinstance(variant_meta, dict) and "delta" in variant_meta:
        return variant_meta["delta"]

    match = re.search(r"_d(\d+)$", variant)
    if match:
        return int(match.group(1)) / 10.0
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathB_output_root", type=str, default="pathB_result")
    parser.add_argument("--data_root", type=str, default="data/Machine")
    parser.add_argument("--clean_variant", type=str, default="SMAP_MVE_clean")
    parser.add_argument("--variant_prefix", type=str, default="SMAP_MVE")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--score_name", type=str, default="pathB_feature_score_ensemble")
    parser.add_argument("--ensemble", choices=["rank", "zscore", "mean"], default="rank")
    parser.add_argument("--meta_path", type=str, default="data/Machine/SMAP_MVE_meta.pkl")
    parser.add_argument("--out", type=str, default="pathB_clean_calibrated_auc_summary.csv")
    args = parser.parse_args()

    meta = load_meta(args.meta_path)
    variants = args.variants
    if not variants:
        variants = [f"{args.variant_prefix}_d{index:02d}" for index in range(args.start, args.end + 1)]

    clean_dir = os.path.join(args.pathB_output_root, args.clean_variant, "ensemble")
    clean_runs = load_score_runs(clean_dir, args.score_name)

    rows = []
    for variant in variants:
        variant_dir = os.path.join(args.pathB_output_root, variant, "ensemble")
        variant_runs = load_score_runs(variant_dir, args.score_name)

        common_run_ids = sorted(set(clean_runs) & set(variant_runs))
        if len(common_run_ids) == 0:
            raise ValueError(f"no common run ids between {args.clean_variant} and {variant}")

        calibrated_runs = []
        run_auc_values = []
        label = np.asarray(load_pickle(os.path.join(args.data_root, f"{variant}_test_label.pkl"))).astype(np.int64).reshape(-1)

        for run_id in common_run_ids:
            calibrated = clean_calibrate(variant_runs[run_id], clean_runs[run_id])
            calibrated_runs.append(calibrated)
            n = min(len(label), len(calibrated))
            run_auc_values.append(roc_auc_score(label[:n], calibrated[:n]))

        ensemble = ensemble_scores(calibrated_runs, args.ensemble)
        n = min(len(label), len(ensemble))
        if n != len(label) or n != len(ensemble):
            print(f"[WARN] {variant} length mismatch: labels={len(label)}, score={len(ensemble)}; using {n}")
        auc_calibrated = roc_auc_score(label[:n], ensemble[:n])

        rows.append(
            [
                variant,
                infer_delta(variant, meta),
                args.score_name,
                args.ensemble,
                len(common_run_ids),
                auc_calibrated,
                float(np.mean(run_auc_values)),
                float(np.std(run_auc_values)),
                ";".join(f"{run_id}:{auc_value:.6f}" for run_id, auc_value in zip(common_run_ids, run_auc_values)),
            ]
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant",
                "delta",
                "score_name",
                "ensemble",
                "run_count",
                "auc_clean_calibrated",
                "mean_single_run_auc",
                "std_single_run_auc",
                "single_run_auc",
            ]
        )
        writer.writerows(rows)

    print(f"saved clean-calibrated AUC summary to {args.out}")


if __name__ == "__main__":
    main()
