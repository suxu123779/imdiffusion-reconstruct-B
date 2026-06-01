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


def tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float().reshape(-1).numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def load_score_runs(ensemble_dir, score_name):
    paths = sorted(glob.glob(os.path.join(ensemble_dir, f"{score_name}*.pt")))
    if len(paths) == 0:
        raise FileNotFoundError(f"no {score_name}*.pt files found in {ensemble_dir}")

    score_runs = []
    for path in paths:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            value = value[score_name]
        score_runs.append(tensor_to_numpy(value))

    min_len = min(len(score) for score in score_runs)
    if len({len(score) for score in score_runs}) != 1:
        print(f"[WARN] {score_name} run lengths differ in {ensemble_dir}; truncating to {min_len}")
    return np.mean(np.stack([score[:min_len] for score in score_runs], axis=0), axis=0)


def infer_delta(variant, meta):
    variant_meta = meta.get(variant, {})
    if isinstance(variant_meta, dict) and "delta" in variant_meta:
        return variant_meta["delta"]

    match = re.search(r"_d(\d+)$", variant)
    if match:
        return f"d{match.group(1)}"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathB_output_root", type=str, default="pathB_result")
    parser.add_argument("--data_root", type=str, default="data/Machine")
    parser.add_argument("--variant_prefix", type=str, default="SMAP_MVE")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--meta_path", type=str, default="data/Machine/SMAP_MVE_meta.pkl")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--pathB_score_name", type=str, default="pathB_feature_score_ensemble")
    parser.add_argument("--out", type=str, default="pathB_auc_summary.csv")
    args = parser.parse_args()

    meta = load_meta(args.meta_path)
    rows = []
    if args.variants:
        variants = args.variants
    else:
        variants = [f"{args.variant_prefix}_d{index:02d}" for index in range(args.start, args.end + 1)]

    for variant in variants:
        ensemble_dir = os.path.join(args.pathB_output_root, variant, "ensemble")
        label_path = os.path.join(args.data_root, f"{variant}_test_label.pkl")

        labels = np.asarray(load_pickle(label_path)).astype(np.int64).reshape(-1)
        final_recon_score = load_score_runs(ensemble_dir, "final_recon_score_ensemble")
        pathB_feature_score = load_score_runs(ensemble_dir, args.pathB_score_name)

        length = min(len(labels), len(final_recon_score), len(pathB_feature_score))
        if length != len(labels) or length != len(final_recon_score) or length != len(pathB_feature_score):
            print(
                f"[WARN] {variant} length mismatch: labels={len(labels)}, "
                f"final={len(final_recon_score)}, pathB={len(pathB_feature_score)}; using {length}"
            )

        labels = labels[:length]
        final_recon_score = final_recon_score[:length]
        pathB_feature_score = pathB_feature_score[:length]

        auc_final = roc_auc_score(labels, final_recon_score)
        auc_pathB = roc_auc_score(labels, pathB_feature_score)
        rows.append(
            [
                variant,
                infer_delta(variant, meta),
                args.pathB_score_name,
                auc_final,
                auc_pathB,
                auc_pathB - auc_final,
            ]
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "delta", "pathB_score_name", "auc_final_recon", "auc_pathB_feature", "auc_gain"])
        writer.writerows(rows)

    print(f"saved AUC summary to {args.out}")


if __name__ == "__main__":
    main()
