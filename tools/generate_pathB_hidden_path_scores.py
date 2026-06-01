import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import get_dataloader
from main_model import CSDI_Physio
from utils import ensure_outputs_can_be_written


DEFAULT_COMPARE_STEPS = "30,27,24,21,18,15,12,9,6,3"
DEFAULT_LATE_STEPS = "15,12,9,6,3"
Z_TYPES = ["zpos", "zabs"]
VAR_AGGS = ["top3", "kmax"]
STEP_AGGS = ["mean", "max", "late_mean"]
SCORE_NAMES = [
    f"hidden_path_{var_agg}_{step_agg}_{z_type}"
    for z_type in Z_TYPES
    for var_agg in VAR_AGGS
    for step_agg in STEP_AGGS
]


def parse_steps(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item) for item in str(value).replace(",", " ").split() if item]


def unique_keep_order(values):
    result = []
    for value in values:
        value = int(value)
        if value not in result:
            result.append(value)
    return result


def ensure_subset_outputs_can_be_written(paths, overwrite=False):
    ensure_outputs_can_be_written(paths, overwrite=overwrite)


def infer_save_ids(train_result_root, requested_saves):
    if requested_saves:
        return requested_saves
    save_ids = sorted(
        item
        for item in os.listdir(train_result_root)
        if re.match(r"save\d+$", item)
    )
    if len(save_ids) == 0:
        raise FileNotFoundError(f"no save* directories found under {train_result_root}")
    return save_ids


def feature_dim_for_dataset(dataset):
    if dataset == "SMD":
        return 38
    if dataset == "PSM":
        return 25
    if dataset == "MSL":
        return 55
    if dataset == "SMAP" or dataset.startswith("SMAP_MVE"):
        return 25
    if dataset == "GCP":
        return 19
    if dataset == "SWaT":
        return 45
    if dataset == "CODERED":
        return 48
    raise ValueError(f"Unknown dataset {dataset}")


def find_model_dir(train_result_root, save_id, model_dataset, diffusion_step=None):
    save_root = os.path.join(train_result_root, save_id)
    if not os.path.isdir(save_root):
        raise FileNotFoundError(f"missing save directory: {save_root}")

    def collect_matches(dataset_name):
        matches = []
        for subset_name in sorted(os.listdir(save_root)):
            subset_path = os.path.join(save_root, subset_name)
            if not os.path.isdir(subset_path):
                continue
            data_id = subset_name.split("_unconditional")[0]
            if data_id != dataset_name:
                continue
            if diffusion_step is not None:
                try:
                    subset_diffusion_step = int(subset_name.split("diffusion_step:")[-1])
                except ValueError:
                    continue
                if subset_diffusion_step != int(diffusion_step):
                    continue
            matches.append((subset_name, subset_path))
        return matches

    lookup_datasets = [model_dataset]
    if str(model_dataset).startswith("SMAP_MVE"):
        lookup_datasets.append("SMAP")

    matches = []
    matched_dataset = None
    for lookup_dataset in lookup_datasets:
        matches = collect_matches(lookup_dataset)
        if matches:
            matched_dataset = lookup_dataset
            break

    if len(matches) == 0:
        available = [
            item
            for item in sorted(os.listdir(save_root))
            if os.path.isdir(os.path.join(save_root, item))
        ]
        available_text = "\n".join(available[:30])
        raise FileNotFoundError(
            f"No model directory matched {model_dataset} under {save_root}.\n"
            f"Tried lookup datasets: {lookup_datasets}\n"
            f"Available model directories:\n{available_text}"
        )
    if matched_dataset != model_dataset:
        print(
            f"[INFO] no exact model directory for {model_dataset} under {save_root}; "
            f"using {matched_dataset} model directory instead"
        )
    if len(matches) > 1:
        print(
            f"[WARN] multiple model directories matched {matched_dataset} {save_id}; "
            f"using {matches[0][0]}"
        )
    return matches[0]


def load_model_for_save(args, save_id):
    subset_name, base_folder = find_model_dir(
        args.train_result_root,
        save_id,
        args.model_dataset,
        args.diffusion_step,
    )
    config_path = os.path.join(base_folder, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            run_config = json.load(f)
    else:
        with open(os.path.join("config", args.config), "r") as f:
            run_config = yaml.safe_load(f)

    unconditional = "unconditional:True" in subset_name
    diffusion_step = int(subset_name.split("diffusion_step:")[-1])
    run_config["model"]["is_unconditional"] = unconditional
    run_config["model"]["test_missing_ratio"] = args.testmissingratio
    run_config["diffusion"]["num_steps"] = diffusion_step
    run_config["train"]["epochs"] = args.epochs

    model = CSDI_Physio(
        run_config,
        args.device,
        target_dim=feature_dim_for_dataset(args.model_dataset),
        ratio=args.ratio,
    ).to(args.device)
    model.load_state_dict(
        torch.load(os.path.join(base_folder, "best-model.pth"), map_location=args.device)
    )
    model.eval()
    return model, subset_name, diffusion_step


def get_variant_loader(variant, batch_size, split):
    train_path = f"data/Machine/{variant}_train.pkl"
    test_path = f"data/Machine/{variant}_test.pkl"
    label_path = f"data/Machine/{variant}_test_label.pkl"
    missing = [path for path in [train_path, test_path, label_path] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("missing data files:\n" + "\n".join(missing))
    _, _, test_loader, _ = get_dataloader(
        train_path,
        test_path,
        label_path,
        batch_size=batch_size,
        window_split=2,
        split=split,
    )
    return test_loader


def hidden_path_raw_by_step(hidden_by_step, compare_steps, reference_step):
    h_ref = hidden_by_step[int(reference_step)]
    h_ref_feature = h_ref.permute(0, 3, 2, 1)  # (B,L,K,C)
    raw_by_step = {}
    for step in compare_steps:
        h_step = hidden_by_step[int(step)].permute(0, 3, 2, 1)
        raw_by_step[int(step)] = 1.0 - F.cosine_similarity(
            h_step,
            h_ref_feature,
            dim=-1,
            eps=1e-8,
        )
    return raw_by_step


def crop_batch_scores(raw_by_step, final_recon_score, batch_no, split):
    any_score = next(iter(raw_by_step.values()))
    L = any_score.shape[1]
    if batch_no == 1:
        head_raw_by_step = {
            step: score[0, 0 : L // split].detach().cpu()
            for step, score in raw_by_step.items()
        }
        head_final = final_recon_score[0, 0 : L // split].detach().cpu()
    else:
        head_raw_by_step = None
        head_final = None

    window_raw_by_step = {
        step: score[:, L // split : L - L // split].detach().cpu()
        for step, score in raw_by_step.items()
    }
    window_final = final_recon_score[:, L // split : L - L // split].detach().cpu()
    return head_raw_by_step, head_final, window_raw_by_step, window_final


def collect_hidden_path_raw(model, loader, args, compare_steps, capture_steps, reference_step, desc):
    raw_chunks_by_step = {int(step): [] for step in compare_steps}
    head_raw_by_step = None
    final_chunks = []
    head_final = None

    with torch.no_grad():
        with tqdm(loader, desc=desc, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                output = model.get_middle_evaluate(
                    batch,
                    n_samples=1,
                    return_pathB=True,
                    pathB_compare_steps=capture_steps,
                    pathB_mode="hidden",
                    return_pathB_hidden=True,
                )
                _, _, _, _, _, _, pathB_result = output
                hidden_by_step = pathB_result["hidden_by_step"]
                final_recon_score = pathB_result["final_recon_score"]
                raw_by_step = hidden_path_raw_by_step(
                    hidden_by_step,
                    compare_steps,
                    reference_step,
                )
                (
                    batch_head_raw_by_step,
                    batch_head_final,
                    window_raw_by_step,
                    window_final,
                ) = crop_batch_scores(raw_by_step, final_recon_score, batch_no, args.split)

                if batch_no == 1:
                    head_raw_by_step = batch_head_raw_by_step
                    head_final = batch_head_final
                for step in compare_steps:
                    raw_chunks_by_step[int(step)].append(window_raw_by_step[int(step)])
                final_chunks.append(window_final)

    raw_series_by_step = {}
    for step in compare_steps:
        parts = [head_raw_by_step[int(step)].reshape(-1, head_raw_by_step[int(step)].shape[-1])]
        parts.extend(chunk.reshape(-1, chunk.shape[-1]) for chunk in raw_chunks_by_step[int(step)])
        raw_series_by_step[int(step)] = torch.cat(parts, dim=0).float()

    final_parts = [head_final.reshape(-1)]
    final_parts.extend(chunk.reshape(-1) for chunk in final_chunks)
    final_recon_score = torch.cat(final_parts, dim=0).float()
    return raw_series_by_step, final_recon_score


def build_normal_stats(raw_series_by_step, compare_steps, eps):
    stats = {
        "median_by_step": {},
        "mad_by_step": {},
        "std_by_step": {},
        "scale_by_step": {},
    }
    for step in compare_steps:
        raw = raw_series_by_step[int(step)].float()
        median = raw.median(dim=0).values
        mad = torch.abs(raw - median.reshape(1, -1)).median(dim=0).values
        std = raw.std(dim=0, unbiased=False)
        scale = torch.where(
            mad >= eps,
            mad,
            torch.where(std >= eps, std, torch.full_like(std, float(eps))),
        )
        stats["median_by_step"][int(step)] = median.cpu()
        stats["mad_by_step"][int(step)] = mad.cpu()
        stats["std_by_step"][int(step)] = std.cpu()
        stats["scale_by_step"][int(step)] = scale.cpu()
    return stats


def stack_step_scores(score_by_step, steps):
    return torch.stack([score_by_step[int(step)] for step in steps], dim=0)


def fuse_step_scores(score_by_step, compare_steps, late_steps):
    missing_late = [step for step in late_steps if int(step) not in score_by_step]
    if missing_late:
        raise ValueError(f"late steps missing from score map: {missing_late}")
    compare_stack = stack_step_scores(score_by_step, compare_steps)
    late_stack = stack_step_scores(score_by_step, late_steps)
    return {
        "mean": compare_stack.mean(dim=0),
        "max": compare_stack.max(dim=0).values,
        "late_mean": late_stack.mean(dim=0),
    }


def compute_hidden_scores(raw_series_by_step, normal_stats, compare_steps, late_steps, eps, topk=3):
    score_maps = {}
    for z_type in Z_TYPES:
        score_maps[z_type] = {"top3": {}, "kmax": {}}

    for step in compare_steps:
        raw = raw_series_by_step[int(step)].float()
        median = normal_stats["median_by_step"][int(step)].to(raw.dtype).reshape(1, -1)
        scale = normal_stats["scale_by_step"][int(step)].to(raw.dtype).reshape(1, -1)
        scale = torch.clamp(scale, min=float(eps))
        centered = (raw - median) / scale
        z_values = {
            "zpos": torch.clamp(centered, min=0.0),
            "zabs": torch.abs(centered),
        }
        for z_type, z_score in z_values.items():
            k = min(int(topk), z_score.shape[-1])
            score_maps[z_type]["top3"][int(step)] = z_score.topk(k=k, dim=-1).values.mean(dim=-1)
            score_maps[z_type]["kmax"][int(step)] = z_score.max(dim=-1).values

    scores = {}
    for z_type in Z_TYPES:
        for var_agg in VAR_AGGS:
            fused = fuse_step_scores(score_maps[z_type][var_agg], compare_steps, late_steps)
            for step_agg, score in fused.items():
                scores[f"hidden_path_{var_agg}_{step_agg}_{z_type}"] = score.float()
    return scores


def stats_to_serializable(normal_stats, compare_steps):
    return {
        key: {
            int(step): list(value[int(step)].shape)
            for step in compare_steps
        }
        for key, value in normal_stats.items()
    }


def save_normal_stats(path, normal_stats, metadata):
    payload = {
        **normal_stats,
        **metadata,
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--diffusion_step", type=int, default=None)
    parser.add_argument("--train_result_root", type=str, default="train_result")
    parser.add_argument("--pathB_output_root", type=str, default="pathB_result_hidden_path")
    parser.add_argument("--model_dataset", type=str, default="SMAP_MVE_clean")
    parser.add_argument("--normal_variant", type=str, default="SMAP_MVE_clean")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "SMAP_MVE_clean",
            "SMAP_MVE_d01",
            "SMAP_MVE_d02",
            "SMAP_MVE_d03",
            "SMAP_MVE_d05",
            "SMAP_MVE_d08",
            "SMAP_MVE_d10",
        ],
    )
    parser.add_argument("--saves", nargs="*", default=None)
    parser.add_argument("--compare_steps", type=str, default=DEFAULT_COMPARE_STEPS)
    parser.add_argument("--late_steps", type=str, default=DEFAULT_LATE_STEPS)
    parser.add_argument("--reference_step", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    compare_steps = unique_keep_order(parse_steps(args.compare_steps))
    late_steps = unique_keep_order(parse_steps(args.late_steps))
    if args.reference_step in compare_steps:
        raise ValueError("--reference_step must not be included in --compare_steps")
    missing_late = [step for step in late_steps if step not in compare_steps]
    if missing_late:
        raise ValueError(f"late_steps must be a subset of compare_steps; missing={missing_late}")
    capture_steps = unique_keep_order(compare_steps + [args.reference_step])
    save_ids = infer_save_ids(args.train_result_root, args.saves)

    stats_root = os.path.join(
        args.pathB_output_root,
        "_hidden_path_stats",
        args.normal_variant,
    )
    os.makedirs(stats_root, exist_ok=True)

    for save_id in save_ids:
        model, subset_name, model_diffusion_step = load_model_for_save(args, save_id)
        normal_loader = get_variant_loader(args.normal_variant, args.batch_size, args.split)
        stats_path = os.path.join(stats_root, f"hidden_path_stats_{save_id}.pt")
        ensure_subset_outputs_can_be_written([stats_path], overwrite=args.overwrite)

        normal_raw_by_step, _ = collect_hidden_path_raw(
            model,
            normal_loader,
            args,
            compare_steps,
            capture_steps,
            args.reference_step,
            desc=f"normal hidden path {save_id}",
        )
        normal_stats = build_normal_stats(normal_raw_by_step, compare_steps, args.eps)
        stats_metadata = {
            "pathB_score_type": "hidden_self_reference_path_deviation",
            "normal_variant": args.normal_variant,
            "model_dataset": args.model_dataset,
            "model_subset_name": subset_name,
            "run_id": save_id,
            "compare_steps": compare_steps,
            "late_steps": late_steps,
            "reference_step": int(args.reference_step),
            "hidden_capture_steps": capture_steps,
            "calibration": "clean_median_mad_with_std_fallback",
            "scale_rule": "mad_if_mad_ge_eps_else_std_if_std_ge_eps_else_eps",
            "eps": float(args.eps),
            "z_types": ["z_pos", "z_abs"],
            "rank_normalization": "evaluation_only",
            "model_diffusion_step": int(model_diffusion_step),
            "normal_stat_shapes": stats_to_serializable(normal_stats, compare_steps),
        }
        save_normal_stats(stats_path, normal_stats, stats_metadata)
        print(f"saved hidden path normal stats for {save_id} to {stats_path}")

        for variant in args.variants:
            variant_loader = get_variant_loader(variant, args.batch_size, args.split)
            ensemble_dir = os.path.join(args.pathB_output_root, variant, "ensemble")
            os.makedirs(ensemble_dir, exist_ok=True)
            output_paths = [
                os.path.join(ensemble_dir, f"final_recon_score_ensemble_{save_id}.pt"),
                os.path.join(ensemble_dir, f"metadata_{save_id}.json"),
            ]
            output_paths.extend(
                os.path.join(ensemble_dir, f"{score_name}_ensemble_{save_id}.pt")
                for score_name in SCORE_NAMES
            )
            ensure_subset_outputs_can_be_written(output_paths, overwrite=args.overwrite)

            raw_by_step, final_recon_score = collect_hidden_path_raw(
                model,
                variant_loader,
                args,
                compare_steps,
                capture_steps,
                args.reference_step,
                desc=f"hidden path {variant} {save_id}",
            )
            scores = compute_hidden_scores(
                raw_by_step,
                normal_stats,
                compare_steps,
                late_steps,
                args.eps,
                topk=args.topk,
            )
            torch.save(final_recon_score.detach().cpu(), output_paths[0])
            for score_name in SCORE_NAMES:
                score = scores[score_name].detach().cpu()
                if not torch.isfinite(score).all():
                    raise ValueError(f"{score_name} contains nan/inf for {variant} {save_id}")
                if score.numel() > 0 and score.min() < -1e-6:
                    raise ValueError(f"{score_name} contains negative values for {variant} {save_id}")
                torch.save(score, os.path.join(ensemble_dir, f"{score_name}_ensemble_{save_id}.pt"))

            metadata = {
                **stats_metadata,
                "variant": variant,
                "output_dir": ensemble_dir,
                "normal_stats_path": stats_path,
                "saved_score_names": SCORE_NAMES,
                "score_shape": {
                    "final_recon_score_ensemble": list(final_recon_score.shape),
                    **{
                        f"{score_name}_ensemble": list(scores[score_name].shape)
                        for score_name in SCORE_NAMES
                    },
                },
            }
            with open(os.path.join(ensemble_dir, f"metadata_{save_id}.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"saved hidden path scores for {variant} {save_id} to {ensemble_dir}")


if __name__ == "__main__":
    main()
