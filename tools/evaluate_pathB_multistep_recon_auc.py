import argparse
import csv
import glob
import os
import pickle
import re

import numpy as np
import torch


DEFAULT_VARIANTS = [
    "SMAP_MVE_d01",
    "SMAP_MVE_d02",
    "SMAP_MVE_d03",
    "SMAP_MVE_d05",
    "SMAP_MVE_d08",
    "SMAP_MVE_d10",
]

DEFAULT_SCORE_NAMES = [
    "pathB_multistep_recon_sumabs_mean",
    "pathB_multistep_recon_sumabs_max",
    "pathB_multistep_recon_sumabs_late_mean",
    "pathB_multistep_recon_maxabs_mean",
    "pathB_multistep_recon_maxabs_max",
    "pathB_multistep_recon_maxabs_late_mean",
]


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
    prefix = f"{score_name}_ensemble_"
    suffix = ".pt"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix):-len(suffix)]
    return os.path.splitext(name)[0]


def load_score_run_dict(ensemble_dir, score_name):
    paths = sorted(glob.glob(os.path.join(ensemble_dir, f"{score_name}_ensemble_*.pt")))
    if len(paths) == 0:
        raise FileNotFoundError(f"no {score_name}_ensemble_*.pt files found in {ensemble_dir}")

    score_runs = {}
    for path in paths:
        score_runs[infer_run_id(path, score_name)] = tensor_to_numpy(torch.load(path, map_location="cpu"))
    return score_runs


def mean_score_runs(score_runs, score_name):
    score_runs = list(score_runs.values()) if isinstance(score_runs, dict) else list(score_runs)
    min_len = min(len(score) for score in score_runs)
    if len({len(score) for score in score_runs}) != 1:
        print(f"[WARN] {score_name} run lengths differ; truncating to {min_len}")
    return np.mean(np.stack([score[:min_len] for score in score_runs], axis=0), axis=0)


def infer_delta(variant, meta):
    variant_meta = meta.get(variant, {})
    if isinstance(variant_meta, dict) and "delta" in variant_meta:
        return variant_meta["delta"]

    match = re.search(r"_d(\d+)$", variant)
    if match:
        return int(match.group(1)) / 10.0
    return ""


def topk_label_hits(labels, scores, k):
    if len(scores) == 0:
        return 0
    k = min(int(k), len(scores))
    order = np.argsort(scores)[::-1][:k]
    return int(np.sum(labels[order]))


def summarize_score(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    if len(positive_scores) == 0:
        mean_pos = median_pos = float("nan")
    else:
        mean_pos = float(np.mean(positive_scores))
        median_pos = float(np.median(positive_scores))
    if len(negative_scores) == 0:
        mean_neg = median_neg = float("nan")
    else:
        mean_neg = float(np.mean(negative_scores))
        median_neg = float(np.median(negative_scores))

    return {
        "mean_score_pos": mean_pos,
        "mean_score_neg": mean_neg,
        "median_score_pos": median_pos,
        "median_score_neg": median_neg,
        "top100_label_hits": topk_label_hits(labels, scores, 100),
        "top200_label_hits": topk_label_hits(labels, scores, 200),
        "top500_label_hits": topk_label_hits(labels, scores, 500),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathB_output_root", type=str, default="pathB_result_Bpp")
    parser.add_argument("--data_root", type=str, default="data/Machine")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--score_names", nargs="*", default=DEFAULT_SCORE_NAMES)
    parser.add_argument("--meta_path", type=str, default="data/Machine/SMAP_MVE_meta.pkl")
    parser.add_argument("--out", type=str, default="pathB_Bpp_multistep_recon_auc_summary.csv")
    parser.add_argument("--per_save_out", type=str, default="")
    args = parser.parse_args()

    meta = load_meta(args.meta_path)
    rows = []
    per_save_rows = []
    for variant in args.variants:
        if variant.endswith("_clean"):
            print(f"[WARN] skipping clean variant for formal AUC: {variant}")
            continue

        ensemble_dir = os.path.join(args.pathB_output_root, variant, "ensemble")
        label_path = os.path.join(args.data_root, f"{variant}_test_label.pkl")
        labels = np.asarray(load_pickle(label_path)).astype(np.int64).reshape(-1)
        final_runs = load_score_run_dict(ensemble_dir, "final_recon_score")
        final_raw_ensemble = mean_score_runs(final_runs, "final_recon_score")
        final_rank_runs = {
            run_id: percentile_ranks(score)
            for run_id, score in final_runs.items()
        }
        final_rank_ensemble = mean_score_runs(final_rank_runs, "final_recon_score_rank")

        for score_name in args.score_names:
            pathB_runs = load_score_run_dict(ensemble_dir, score_name)
            pathB_score = mean_score_runs(pathB_runs, score_name)
            used_len = min(len(labels), len(final_raw_ensemble), len(final_rank_ensemble), len(pathB_score))
            label_used = labels[:used_len]
            final_raw_used = final_raw_ensemble[:used_len]
            final_rank_used = final_rank_ensemble[:used_len]
            score_used = pathB_score[:used_len]
            label_sum_used = int(np.sum(label_used))

            if used_len != len(labels) or used_len != len(pathB_score):
                print(
                    f"[WARN] {variant} {score_name} length mismatch: "
                    f"labels={len(labels)}, final_raw={len(final_raw_ensemble)}, "
                    f"final_rank={len(final_rank_ensemble)}, "
                    f"score={len(pathB_score)}; using {used_len}"
                )
            if label_sum_used == 0:
                print(f"[WARN] {variant} has no positive labels in used range; AUC will be nan")

            auc_final_raw = roc_auc_score(label_used, final_raw_used)
            auc_final_rank = roc_auc_score(label_used, final_rank_used)
            auc_pathB = roc_auc_score(label_used, score_used)
            auc_reverse = roc_auc_score(label_used, -score_used)
            summary = summarize_score(label_used, score_used)

            common_run_ids = sorted(set(final_runs) & set(pathB_runs))
            for run_id in common_run_ids:
                per_save_used_len = min(len(labels), len(final_runs[run_id]), len(pathB_runs[run_id]))
                per_save_labels = labels[:per_save_used_len]
                auc_final_save = roc_auc_score(per_save_labels, final_runs[run_id][:per_save_used_len])
                auc_pathB_save = roc_auc_score(per_save_labels, pathB_runs[run_id][:per_save_used_len])
                per_save_rows.append(
                    [
                        variant,
                        infer_delta(variant, meta),
                        score_name,
                        run_id,
                        auc_final_save,
                        auc_pathB_save,
                        auc_pathB_save - auc_final_save
                        if not np.isnan(auc_pathB_save) and not np.isnan(auc_final_save)
                        else float("nan"),
                        len(pathB_runs[run_id]),
                        len(labels),
                        per_save_used_len,
                        int(np.sum(per_save_labels)),
                    ]
                )

            rows.append(
                [
                    variant,
                    infer_delta(variant, meta),
                    score_name,
                    auc_final_raw,
                    auc_final_rank,
                    auc_pathB,
                    auc_pathB - auc_final_raw
                    if not np.isnan(auc_pathB) and not np.isnan(auc_final_raw)
                    else float("nan"),
                    auc_pathB - auc_final_rank
                    if not np.isnan(auc_pathB) and not np.isnan(auc_final_rank)
                    else float("nan"),
                    len(pathB_score),
                    len(labels),
                    used_len,
                    label_sum_used,
                    auc_reverse,
                    max(auc_pathB, auc_reverse) if not np.isnan(auc_pathB) and not np.isnan(auc_reverse) else float("nan"),
                    summary["mean_score_pos"],
                    summary["mean_score_neg"],
                    summary["median_score_pos"],
                    summary["median_score_neg"],
                    summary["top100_label_hits"],
                    summary["top200_label_hits"],
                    summary["top500_label_hits"],
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
                "auc_final_recon_raw_ensemble",
                "auc_final_recon_rank_ensemble",
                "auc_pathB_score",
                "gain_vs_final_raw",
                "gain_vs_final_rank",
                "score_len",
                "label_len",
                "used_len",
                "label_sum_used",
                "auc_pathB_reverse",
                "auc_pathB_best_direction",
                "mean_score_pos",
                "mean_score_neg",
                "median_score_pos",
                "median_score_neg",
                "top100_label_hits",
                "top200_label_hits",
                "top500_label_hits",
            ]
        )
        writer.writerows(rows)

    if args.per_save_out:
        per_save_out = args.per_save_out
    else:
        out_root, out_ext = os.path.splitext(args.out)
        per_save_out = f"{out_root}_per_save{out_ext or '.csv'}"
    per_save_dir = os.path.dirname(per_save_out)
    if per_save_dir:
        os.makedirs(per_save_dir, exist_ok=True)
    with open(per_save_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant",
                "delta",
                "score_name",
                "save_id",
                "auc_final_recon_save",
                "auc_pathB_save",
                "auc_gain_save",
                "score_len",
                "label_len",
                "used_len",
                "label_sum_used",
            ]
        )
        writer.writerows(per_save_rows)

    print(f"saved PathB++ multistep recon AUC summary to {args.out}")
    print(f"saved PathB++ per-save AUC diagnostics to {per_save_out}")


if __name__ == "__main__":
    main()
