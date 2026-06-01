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


def load_score_runs(ensemble_dir, score_prefix):
    paths = sorted(glob.glob(os.path.join(ensemble_dir, f"{score_prefix}_*.pt")))
    if len(paths) == 0:
        raise FileNotFoundError(f"no {score_prefix}_*.pt files found in {ensemble_dir}")

    score_runs = []
    for path in paths:
        score_runs.append(tensor_to_numpy(torch.load(path, map_location="cpu")))
    min_len = min(len(score) for score in score_runs)
    return np.mean(np.stack([score[:min_len] for score in score_runs], axis=0), axis=0)


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
    parser.add_argument("--variant_prefix", type=str, default="SMAP_MVE")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--steps", type=str, default="49,45,40,35,30,25,20,15,10,5,0")
    parser.add_argument("--aggregations", nargs="*", default=["proto_top3", "proto_max"])
    parser.add_argument("--meta_path", type=str, default="data/Machine/SMAP_MVE_meta.pkl")
    parser.add_argument("--out", type=str, default="pathB_timestep_sweep_auc_summary.csv")
    args = parser.parse_args()

    meta = load_meta(args.meta_path)
    variants = args.variants or [f"{args.variant_prefix}_d{index:02d}" for index in range(1, 11)]
    steps = [int(step) for step in args.steps.replace(",", " ").split()]

    rows = []
    for variant in variants:
        ensemble_dir = os.path.join(args.pathB_output_root, variant, "ensemble")
        labels = np.asarray(load_pickle(os.path.join(args.data_root, f"{variant}_test_label.pkl"))).astype(np.int64).reshape(-1)
        final_recon_score = load_score_runs(ensemble_dir, "final_recon_score_ensemble")

        for aggregation in args.aggregations:
            if aggregation == "top3":
                base_name = "pathB_feature_score_top3"
            elif aggregation == "max":
                base_name = "pathB_feature_score_max"
            elif aggregation == "global":
                base_name = "pathB_feature_score"
            elif aggregation == "proto_top3":
                base_name = "pathB_proto_score_top3"
            elif aggregation == "proto_max":
                base_name = "pathB_proto_score_max"
            else:
                raise ValueError(f"unknown aggregation: {aggregation}")

            for step in steps:
                score_name = f"{base_name}_t{step}_ensemble"
                pathB_score = load_score_runs(ensemble_dir, score_name)
                length = min(len(labels), len(final_recon_score), len(pathB_score))
                auc_final = roc_auc_score(labels[:length], final_recon_score[:length])
                auc_pathB = roc_auc_score(labels[:length], pathB_score[:length])
                rows.append([
                    variant,
                    infer_delta(variant, meta),
                    aggregation,
                    step,
                    score_name,
                    auc_final,
                    auc_pathB,
                    auc_pathB - auc_final,
                ])

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant",
            "delta",
            "aggregation",
            "compare_step",
            "pathB_score_name",
            "auc_final_recon",
            "auc_pathB_feature",
            "auc_gain",
        ])
        writer.writerows(rows)

    print(f"saved timestep sweep AUC summary to {args.out}")


if __name__ == "__main__":
    main()
