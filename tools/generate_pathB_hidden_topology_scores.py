import argparse
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from generate_pathB_hidden_path_scores import (
    crop_batch_scores,
    get_variant_loader,
    infer_save_ids,
    load_model_for_save,
    parse_steps,
    unique_keep_order,
)
from utils import ensure_outputs_can_be_written


DEFAULT_STEPS = "20,15,10,5,0"
SCORE_NAMES = [
    "pathB_hidden_topology_l2_offdiag_late_mean",
    "pathB_hidden_topology_l2_offdiag_late_max",
    "pathB_hidden_topology_l2_all_late_mean",
    "pathB_hidden_topology_l2_all_late_max",
]


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
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j

    return ranks / float(n - 1)


def hidden_to_topology(hidden):
    hidden_permuted = hidden.permute(0, 3, 2, 1).contiguous()  # (B,L,K,C)
    hidden_norm = F.normalize(hidden_permuted, p=2, dim=-1, eps=1e-8)
    topology = torch.matmul(hidden_norm, hidden_norm.transpose(-1, -2))
    return hidden_permuted, topology


def maybe_print_shape_debug(prefix, step, hidden, hidden_permuted, topology, printed):
    if printed:
        return True
    print(f"[DEBUG] {prefix} step={step}")
    print(f"[DEBUG] h_t original shape {list(hidden.shape)}")
    print(f"[DEBUG] h_t permuted shape {list(hidden_permuted.shape)}")
    print(f"[DEBUG] G_t shape {list(topology.shape)}")
    return True


def collect_topology_prototype(model, loader, args, selected_steps):
    sum_by_step = {int(step): None for step in selected_steps}
    count_by_step = {int(step): 0 for step in selected_steps}
    hidden_shape_info = None
    printed_debug = False

    with torch.no_grad():
        with tqdm(loader, desc="hidden topology prototype", mininterval=5.0, maxinterval=50.0) as it:
            for batch in it:
                output = model.get_middle_evaluate(
                    batch,
                    n_samples=1,
                    return_pathB=True,
                    pathB_compare_steps=selected_steps,
                    pathB_mode="hidden",
                    return_pathB_hidden=True,
                )
                _, _, _, _, _, _, pathB_result = output
                hidden_by_step = pathB_result["hidden_by_step"]
                for step in selected_steps:
                    hidden = hidden_by_step[int(step)]
                    hidden_permuted, topology = hidden_to_topology(hidden)
                    printed_debug = maybe_print_shape_debug(
                        "prototype",
                        step,
                        hidden,
                        hidden_permuted,
                        topology,
                        printed_debug,
                    )
                    if hidden_shape_info is None:
                        _, channels, features, _ = hidden.shape
                        hidden_shape_info = {"K": int(features), "C": int(channels)}
                    topo_sum = topology.sum(dim=(0, 1)).detach().cpu()
                    if sum_by_step[int(step)] is None:
                        sum_by_step[int(step)] = topo_sum
                    else:
                        sum_by_step[int(step)] += topo_sum
                    count_by_step[int(step)] += int(topology.shape[0] * topology.shape[1])

    prototype_by_step = {}
    for step in selected_steps:
        step = int(step)
        if count_by_step[step] == 0:
            raise ValueError(f"no topology samples collected for step {step}")
        prototype_by_step[step] = sum_by_step[step] / float(count_by_step[step])

    return prototype_by_step, hidden_shape_info


def load_or_build_prototype(model, loader, args, save_id, selected_steps, subset_name, diffusion_step):
    proto_root = os.path.join(
        args.pathB_output_root,
        "_topology_prototypes",
        args.proto_dataset,
    )
    os.makedirs(proto_root, exist_ok=True)
    proto_path = os.path.join(proto_root, f"hidden_topology_prototype_{save_id}.pt")
    if os.path.exists(proto_path) and not args.proto_recompute:
        print(f"[INFO] loading hidden topology prototype from {proto_path}")
        payload = torch.load(proto_path, map_location="cpu")
        cached_steps = [int(step) for step in payload.get("selected_steps", [])]
        if cached_steps == [int(step) for step in selected_steps]:
            return payload, proto_path
        if not args.overwrite:
            raise ValueError(
                f"cached prototype steps {cached_steps} do not match requested steps {selected_steps}. "
                f"Pass --overwrite or --proto_recompute to rebuild {proto_path}."
            )
        print(
            f"[WARN] cached prototype steps {cached_steps} do not match requested steps {selected_steps}; "
            "rebuilding because --overwrite was passed"
        )

    ensure_outputs_can_be_written([proto_path], overwrite=args.overwrite)
    prototype_by_step, hidden_shape_info = collect_topology_prototype(
        model,
        loader,
        args,
        selected_steps,
    )
    payload = {
        "prototype_topology_by_step": {
            int(step): prototype_by_step[int(step)].detach().cpu()
            for step in selected_steps
        },
        "proto_dataset": args.proto_dataset,
        "save_id": save_id,
        "selected_steps": selected_steps,
        "hidden_shape_info": hidden_shape_info,
        "score_type": "hidden_topology_cosine_matrix",
        "model_dataset": args.model_dataset,
        "model_subset_name": subset_name,
        "model_diffusion_step": int(diffusion_step),
    }
    torch.save(payload, proto_path)
    print(f"saved hidden topology prototype for {save_id} to {proto_path}")
    return payload, proto_path


def compute_topology_step_scores(hidden_by_step, prototype_payload, selected_steps, debug=False):
    score_by_type = {"l2_offdiag": {}, "l2_all": {}}
    printed_debug = False
    for step in selected_steps:
        step = int(step)
        hidden = hidden_by_step[step]
        hidden_permuted, topology = hidden_to_topology(hidden)
        if debug:
            printed_debug = maybe_print_shape_debug(
                "score",
                step,
                hidden,
                hidden_permuted,
                topology,
                printed_debug,
            )
        prototype = prototype_payload["prototype_topology_by_step"][step].to(topology.device).to(topology.dtype)
        if debug:
            print(f"[DEBUG] P_topo_t shape {list(prototype.shape)} for step={step}")
        diff = topology - prototype.reshape(1, 1, prototype.shape[0], prototype.shape[1])
        score_all = torch.sqrt(torch.clamp(torch.sum(diff * diff, dim=(-1, -2)), min=0.0))
        mask = ~torch.eye(prototype.shape[0], dtype=torch.bool, device=topology.device)
        score_offdiag = torch.sqrt(torch.clamp(torch.sum(diff[:, :, mask] ** 2, dim=-1), min=0.0))
        if debug:
            print(f"[DEBUG] topo_l2_offdiag_t before merge shape {list(score_offdiag.shape)} for step={step}")
        score_by_type["l2_offdiag"][step] = score_offdiag
        score_by_type["l2_all"][step] = score_all
    return score_by_type


def collect_topology_scores(model, loader, args, selected_steps, prototype_payload, desc):
    chunks_by_type = {
        score_type: {int(step): [] for step in selected_steps}
        for score_type in ["l2_offdiag", "l2_all"]
    }
    head_by_type = None
    final_chunks = []
    head_final = None

    with torch.no_grad():
        with tqdm(loader, desc=desc, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                output = model.get_middle_evaluate(
                    batch,
                    n_samples=1,
                    return_pathB=True,
                    pathB_compare_steps=selected_steps,
                    pathB_mode="hidden",
                    return_pathB_hidden=True,
                )
                _, _, _, _, _, _, pathB_result = output
                step_scores = compute_topology_step_scores(
                    pathB_result["hidden_by_step"],
                    prototype_payload,
                    selected_steps,
                    debug=(batch_no == 1),
                )
                final_recon_score = pathB_result["final_recon_score"]

                batch_head_by_type = {}
                batch_window_by_type = {}
                batch_head_final = None
                batch_window_final = None
                for score_type, score_by_step in step_scores.items():
                    (
                        batch_head_scores,
                        current_head_final,
                        batch_window_scores,
                        current_window_final,
                    ) = crop_batch_scores(score_by_step, final_recon_score, batch_no, args.split)
                    batch_head_by_type[score_type] = batch_head_scores
                    batch_window_by_type[score_type] = batch_window_scores
                    batch_head_final = current_head_final
                    batch_window_final = current_window_final

                if batch_no == 1:
                    head_by_type = batch_head_by_type
                    head_final = batch_head_final
                for score_type in chunks_by_type:
                    for step in selected_steps:
                        chunks_by_type[score_type][int(step)].append(batch_window_by_type[score_type][int(step)])
                final_chunks.append(batch_window_final)

    score_series_by_type = {"l2_offdiag": {}, "l2_all": {}}
    for score_type in score_series_by_type:
        for step in selected_steps:
            step = int(step)
            parts = [head_by_type[score_type][step].reshape(-1)]
            parts.extend(chunk.reshape(-1) for chunk in chunks_by_type[score_type][step])
            series = torch.cat(parts, dim=0).float()
            print(f"[DEBUG] {score_type}_t{step} after merge shape {list(series.shape)}")
            score_series_by_type[score_type][step] = series

    final_parts = [head_final.reshape(-1)]
    final_parts.extend(chunk.reshape(-1) for chunk in final_chunks)
    final_recon_score = torch.cat(final_parts, dim=0).float()
    return score_series_by_type, final_recon_score


def fuse_rank_scores(score_series_by_step, selected_steps, late_steps):
    rank_by_step = {}
    for step in selected_steps:
        step = int(step)
        rank = percentile_rank_1d(score_series_by_step[step])
        rank_by_step[step] = rank
        print(
            f"[DEBUG] rank step={step} min/max="
            f"{float(rank.min()) if rank.numel() else 'nan'}/"
            f"{float(rank.max()) if rank.numel() else 'nan'}"
        )

    missing_late = [int(step) for step in late_steps if int(step) not in rank_by_step]
    if missing_late:
        raise ValueError(f"late steps missing from selected scores: {missing_late}")
    late_stack = torch.stack([rank_by_step[int(step)] for step in late_steps], dim=0)
    return {
        "late_mean": late_stack.mean(dim=0),
        "late_max": late_stack.max(dim=0).values,
    }


def compute_fused_scores(score_series_by_type, selected_steps, late_steps):
    scores = {}
    for score_type, score_series_by_step in score_series_by_type.items():
        fused = fuse_rank_scores(score_series_by_step, selected_steps, late_steps)
        for agg_name, score in fused.items():
            scores[f"pathB_hidden_topology_{score_type}_{agg_name}"] = score.float()
    return scores


def save_final_if_missing(path, final_recon_score):
    if os.path.exists(path):
        print(f"[INFO] keeping existing final_recon_score file: {path}")
        return
    torch.save(final_recon_score.detach().cpu(), path)
    print(f"saved final_recon_score to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--diffusion_step", type=int, default=None)
    parser.add_argument("--train_result_root", type=str, default="train_result")
    parser.add_argument("--pathB_output_root", type=str, default="pathB_result_Bpp")
    parser.add_argument("--model_dataset", type=str, default="SMAP")
    parser.add_argument("--proto_dataset", type=str, default="SMAP_MVE_clean")
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
    parser.add_argument("--selected_steps", type=str, default=DEFAULT_STEPS)
    parser.add_argument("--late_steps", type=str, default=DEFAULT_STEPS)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--proto_recompute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected_steps = unique_keep_order(parse_steps(args.selected_steps))
    late_steps = unique_keep_order(parse_steps(args.late_steps))
    missing_late = [step for step in late_steps if step not in selected_steps]
    if missing_late:
        raise ValueError(f"late_steps must be a subset of selected_steps; missing={missing_late}")

    save_ids = infer_save_ids(args.train_result_root, args.saves)
    for save_id in save_ids:
        model, subset_name, diffusion_step = load_model_for_save(args, save_id)
        proto_loader = get_variant_loader(args.proto_dataset, args.batch_size, args.split)
        prototype_payload, proto_path = load_or_build_prototype(
            model,
            proto_loader,
            args,
            save_id,
            selected_steps,
            subset_name,
            diffusion_step,
        )

        for variant in args.variants:
            variant_loader = get_variant_loader(variant, args.batch_size, args.split)
            ensemble_dir = os.path.join(args.pathB_output_root, variant, "ensemble")
            os.makedirs(ensemble_dir, exist_ok=True)
            output_paths = [
                os.path.join(ensemble_dir, f"{score_name}_ensemble_{save_id}.pt")
                for score_name in SCORE_NAMES
            ]
            output_paths.append(os.path.join(ensemble_dir, f"metadata_hidden_topology_{save_id}.json"))
            ensure_outputs_can_be_written(output_paths, overwrite=args.overwrite)

            score_series_by_type, final_recon_score = collect_topology_scores(
                model,
                variant_loader,
                args,
                selected_steps,
                prototype_payload,
                desc=f"hidden topology {variant} {save_id}",
            )
            scores = compute_fused_scores(score_series_by_type, selected_steps, late_steps)

            for score_name in SCORE_NAMES:
                score = scores[score_name].detach().cpu()
                if not torch.isfinite(score).all():
                    raise ValueError(f"{score_name} contains nan/inf for {variant} {save_id}")
                if score.numel() > 0 and (score.min() < -1e-6 or score.max() > 1.0 + 1e-6):
                    raise ValueError(f"{score_name} rank score outside [0,1] for {variant} {save_id}")
                torch.save(score, os.path.join(ensemble_dir, f"{score_name}_ensemble_{save_id}.pt"))

            final_path = os.path.join(ensemble_dir, f"final_recon_score_ensemble_{save_id}.pt")
            save_final_if_missing(final_path, final_recon_score)

            metadata = {
                "pathB_score_type": "hidden_topology_rank_fusion",
                "score_type": "hidden_topology_cosine_matrix",
                "variant": variant,
                "proto_dataset": args.proto_dataset,
                "save_id": save_id,
                "model_dataset": args.model_dataset,
                "model_subset_name": subset_name,
                "prototype_cache_path": proto_path,
                "selected_steps": selected_steps,
                "late_steps": late_steps,
                "rank_normalization": "percentile_rank_per_step",
                "score_names": SCORE_NAMES,
                "priority_note": "l2_offdiag is primary; l2_all is diagnostic",
                "score_shape": {
                    "final_recon_score_ensemble": list(final_recon_score.shape),
                    **{f"{score_name}_ensemble": list(scores[score_name].shape) for score_name in SCORE_NAMES},
                },
            }
            with open(os.path.join(ensemble_dir, f"metadata_hidden_topology_{save_id}.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"saved hidden topology scores for {variant} {save_id} to {ensemble_dir}")


if __name__ == "__main__":
    main()
