import argparse
import glob
import json
import os
import re
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ensemble_proper_reconstruction import merge


SCORE_NAMES = [
    "pathB_multistep_recon_sumabs_mean",
    "pathB_multistep_recon_sumabs_max",
    "pathB_multistep_recon_sumabs_late_mean",
    "pathB_multistep_recon_maxabs_mean",
    "pathB_multistep_recon_maxabs_max",
    "pathB_multistep_recon_maxabs_late_mean",
]


def ensure_outputs_can_be_written(paths, overwrite=False):
    existing_paths = [path for path in paths if path and os.path.exists(path)]
    if existing_paths and not overwrite:
        raise FileExistsError(
            "output already exists. Pass --overwrite to replace:\n"
            + "\n".join(existing_paths)
        )


def parse_steps(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item) for item in str(value).replace(",", " ").split() if item]


def percentile_rank_1d(x):
    # Average-rank percentile in [0, 1]; higher x receives higher rank.
    x = x.detach().float().reshape(-1)
    n = x.numel()
    if n <= 1:
        return torch.zeros_like(x)

    order = torch.argsort(x)
    sorted_x = x[order]
    ranks = torch.empty(n, dtype=torch.float32, device=x.device)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        average_rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = average_rank
        i = j

    return ranks / float(n - 1)


def find_variant_pkl(window_result_root, save_id, diffusion_step, variant):
    base = os.path.join(window_result_root, save_id, str(diffusion_step))
    candidates = sorted(
        glob.glob(os.path.join(base, f"{variant}_unconditional*", "*.pk"))
    )
    if len(candidates) == 0:
        candidates = sorted(
            path
            for path in glob.glob(os.path.join(base, "*", "*.pk"))
            if variant in path
        )
    if len(candidates) == 0:
        raise FileNotFoundError(f"no pkl found for {variant} {save_id} under {base}")
    if len(candidates) > 1:
        print(f"[WARN] multiple pkl files for {variant} {save_id}; using {candidates[0]}")
    return candidates[0]


def resolve_available_steps(all_gen_middle, requested_steps):
    max_step = all_gen_middle.shape[0] - 1
    available = []
    missing = []
    for step in requested_steps:
        if 0 <= step <= max_step:
            if step not in available:
                available.append(step)
        else:
            missing.append(step)
    if len(available) == 0:
        raise ValueError(
            f"none of requested steps exist; requested={requested_steps}, available range=0..{max_step}"
        )
    return available, missing


def compute_step_rank_scores(all_gen_middle, all_target, steps):
    rank_by_type = {"sumabs": {}, "maxabs": {}}
    for step in steps:
        residual_abs = torch.abs(all_gen_middle[step].float() - all_target.float())
        residual_sumabs = residual_abs.sum(dim=-1)
        residual_maxabs = residual_abs.max(dim=-1).values
        rank_by_type["sumabs"][step] = percentile_rank_1d(residual_sumabs)
        rank_by_type["maxabs"][step] = percentile_rank_1d(residual_maxabs)
    return rank_by_type


def fuse_rank_scores(rank_by_step, selected_steps, late_steps):
    selected_stack = torch.stack([rank_by_step[step] for step in selected_steps], dim=0)
    existing_late_steps = [step for step in late_steps if step in rank_by_step]
    if len(existing_late_steps) == 0:
        raise ValueError(f"none of late steps exist; late_steps={late_steps}")
    late_stack = torch.stack([rank_by_step[step] for step in existing_late_steps], dim=0)
    return {
        "mean": selected_stack.mean(dim=0),
        "max": selected_stack.max(dim=0).values,
        "late_mean": late_stack.mean(dim=0),
    }, existing_late_steps


def compute_multistep_scores(pkl_path, variant, selected_steps, late_steps):
    all_gen_middle, label, all_target = merge(pkl_path, variant, load_label=True)
    all_gen_middle = all_gen_middle.float()
    all_target = all_target.float()

    available_steps, missing_steps = resolve_available_steps(all_gen_middle, selected_steps)
    rank_by_type = compute_step_rank_scores(all_gen_middle, all_target, available_steps)

    scores = {}
    used_late_steps = None
    for residual_type in ["sumabs", "maxabs"]:
        fused, existing_late_steps = fuse_rank_scores(
            rank_by_type[residual_type],
            available_steps,
            late_steps,
        )
        used_late_steps = existing_late_steps
        scores[f"pathB_multistep_recon_{residual_type}_mean"] = fused["mean"]
        scores[f"pathB_multistep_recon_{residual_type}_max"] = fused["max"]
        scores[f"pathB_multistep_recon_{residual_type}_late_mean"] = fused["late_mean"]

    final_recon_score = torch.abs(all_gen_middle[0] - all_target).sum(dim=-1)
    metadata = {
        "pathB_score_type": "multi_step_recon_residual_rank_fusion",
        "selected_steps": selected_steps,
        "used_selected_steps": available_steps,
        "missing_selected_steps": missing_steps,
        "late_steps": late_steps,
        "used_late_steps": used_late_steps,
        "residual_types": ["sumabs", "maxabs"],
        "fusion_types": ["mean", "max", "late_mean"],
        "rank_normalization": "percentile_rank_per_step",
        "final_step_index": 0,
        "note": "diagnostic score for PathB++; not a new trained model",
        "source_pkl_path": pkl_path,
        "diffusion_steps": int(all_gen_middle.shape[0]),
        "score_len": int(final_recon_score.numel()),
        "label_len": None if label is None else int(label.numel()),
    }
    return final_recon_score, scores, metadata


def infer_save_ids(window_result_root, requested_saves):
    if requested_saves:
        return requested_saves
    save_ids = sorted(
        item
        for item in os.listdir(window_result_root)
        if re.match(r"save\d+$", item)
    )
    if len(save_ids) == 0:
        raise FileNotFoundError(f"no save* directories found under {window_result_root}")
    return save_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_result_root", type=str, default="window_result")
    parser.add_argument("--pathB_output_root", type=str, default="pathB_result_Bpp")
    parser.add_argument("--diffusion_step", type=int, default=50)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "SMAP_MVE_d01",
            "SMAP_MVE_d02",
            "SMAP_MVE_d03",
            "SMAP_MVE_d05",
            "SMAP_MVE_d08",
            "SMAP_MVE_d10",
        ],
    )
    parser.add_argument("--saves", nargs="*", default=None)
    parser.add_argument("--selected_steps", type=str, default="49,45,40,35,30,25,20,15,10,5,0")
    parser.add_argument("--late_steps", type=str, default="20,15,10,5,0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected_steps = parse_steps(args.selected_steps)
    late_steps = parse_steps(args.late_steps)
    save_ids = infer_save_ids(args.window_result_root, args.saves)

    for variant in args.variants:
        ensemble_dir = os.path.join(args.pathB_output_root, variant, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        for save_id in save_ids:
            pkl_path = find_variant_pkl(
                args.window_result_root,
                save_id,
                args.diffusion_step,
                variant,
            )
            output_paths = [
                os.path.join(ensemble_dir, f"final_recon_score_ensemble_{save_id}.pt"),
                os.path.join(ensemble_dir, f"metadata_{save_id}.json"),
            ]
            output_paths.extend(
                os.path.join(ensemble_dir, f"{score_name}_ensemble_{save_id}.pt")
                for score_name in SCORE_NAMES
            )
            ensure_outputs_can_be_written(output_paths, overwrite=args.overwrite)

            final_recon_score, scores, metadata = compute_multistep_scores(
                pkl_path,
                variant,
                selected_steps,
                late_steps,
            )
            torch.save(final_recon_score.detach().cpu(), output_paths[0])
            for score_name in SCORE_NAMES:
                score = scores[score_name].detach().cpu()
                if not torch.isfinite(score).all():
                    raise ValueError(f"{score_name} contains nan/inf for {variant} {save_id}")
                if score.numel() > 0 and (score.min() < -1e-6 or score.max() > 1.0 + 1e-6):
                    raise ValueError(f"{score_name} rank-fused score outside [0,1] for {variant} {save_id}")
                torch.save(
                    score,
                    os.path.join(ensemble_dir, f"{score_name}_ensemble_{save_id}.pt"),
                )

            metadata.update(
                {
                    "variant": variant,
                    "run_id": save_id,
                    "output_dir": ensemble_dir,
                    "saved_score_names": SCORE_NAMES,
                }
            )
            with open(os.path.join(ensemble_dir, f"metadata_{save_id}.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"saved PathB++ multistep recon scores for {variant} {save_id} to {ensemble_dir}")


if __name__ == "__main__":
    main()
