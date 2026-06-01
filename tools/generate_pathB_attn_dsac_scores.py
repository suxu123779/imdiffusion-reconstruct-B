import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from generate_pathB_hidden_path_scores import (
    infer_save_ids,
    load_model_for_save,
    parse_steps,
    unique_keep_order,
)
from generate_pathB_hdsac_scores import (
    backproject_windows_to_valid_time,
    build_validation_loader,
    build_variant_loader,
    check_dataset_label_sum,
    descriptor_mahalanobis,
    empirical_cdf_scores,
    load_labels,
    require_no_mve_name,
    robust_norm_stats,
    robust_z,
    selected_steps_key,
    set_all_seeds,
)
from utils import ensure_outputs_can_be_written


DEFAULT_SELECTED_STEPS = "40,25,10,0"
ATTN_DSAC_METHODS = [
    "attn_dsac_ch_mean_step_mean",
    "attn_dsac_ch_mean_step_median",
    "attn_dsac_ch_mean_step_max",
    "attn_dsac_ch_max_step_mean",
    "attn_dsac_ch_max_step_median",
    "attn_dsac_ch_max_step_max",
    "attn_dsac_ch_top3_step_mean",
    "attn_dsac_ch_top3_step_median",
    "attn_dsac_ch_top3_step_max",
    "attn_dsac_ch_top3_step_top2mean",
]
CALIBRATION_SCORE_NAMES = ["final_recon_score"] + ATTN_DSAC_METHODS


def default_prototype_path(output_root, base_dataset, selected_steps, seed, save_id):
    filename = f"attn_dsac_prototype_steps_{selected_steps_key(selected_steps)}_seed{int(seed)}_{save_id}.npz"
    return os.path.join(output_root, base_dataset, filename)


def check_captured_steps(attention_by_step, selected_steps):
    captured = sorted(int(step) for step in attention_by_step)
    requested = [int(step) for step in selected_steps]
    print(f"[ATTN-DSAC] selected steps requested: {requested}")
    print(f"[ATTN-DSAC] selected steps actually captured: {captured}")
    missing = [step for step in requested if step not in attention_by_step]
    if missing:
        raise RuntimeError(f"selected steps were not all captured: missing={missing}, captured={captured}")


def attention_to_descriptor(attention, top_r, eps, print_debug=False, step=None):
    # Expected shape after model capture: [B,L,K,K].
    # If a future capture keeps source/head dim [B,L,A,K,K], average it here.
    if attention.dim() == 5:
        attention = attention.mean(dim=2)
    if attention.dim() != 4:
        raise ValueError(f"attention must have shape [B,L,K,K], got {list(attention.shape)}")
    if print_debug:
        print(f"[ATTN-DSAC] step={step} real feature attention shape [B,L,K,K]: {list(attention.shape)}")

    affinity = attention.float()
    row_sum = affinity.sum(dim=-1, keepdim=True)
    affinity = affinity / torch.clamp(row_sum, min=float(eps))
    k_dim = affinity.shape[-1]

    entropy = -(affinity * torch.log(affinity + float(eps))).sum(dim=-1)
    if k_dim > 1:
        entropy = entropy / np.log(float(k_dim))
    self_strength = torch.diagonal(affinity, dim1=-2, dim2=-1)
    topk = min(int(top_r), k_dim)
    top_mass = affinity.topk(k=topk, dim=-1).values.sum(dim=-1)
    phi = torch.cat(
        [
            affinity,
            entropy.unsqueeze(-1),
            self_strength.unsqueeze(-1),
            top_mass.unsqueeze(-1),
        ],
        dim=-1,
    )
    if print_debug:
        print(f"[ATTN-DSAC] step={step} phi shape [B,L,K,K+3]: {list(phi.shape)}")
    return phi


def collect_pathb_result(model, batch, selected_steps):
    output = model.get_middle_evaluate(
        batch,
        n_samples=1,
        return_pathB=True,
        pathB_compare_steps=selected_steps,
        pathB_mode="attention",
        return_pathB_attention=True,
    )
    _, _, _, _, _, _, pathB_result = output
    attention_by_step = pathB_result["feature_attention_by_step"]
    check_captured_steps(attention_by_step, selected_steps)
    return pathB_result


def collect_attention_by_step(model, batch, selected_steps):
    return collect_pathb_result(model, batch, selected_steps)["feature_attention_by_step"]


def compute_descriptor_stats(model, loader, selected_steps, args):
    sum_by_step = {}
    sumsq_by_step = {}
    count_by_step = {}
    k_dim = None
    d_phi = None
    printed = False

    with torch.no_grad():
        with tqdm(loader, desc="ATTN-DSAC prototype pass1 stats", mininterval=5.0, maxinterval=50.0) as it:
            for batch in it:
                attention_by_step = collect_attention_by_step(model, batch, selected_steps)
                for step in selected_steps:
                    step = int(step)
                    phi = attention_to_descriptor(
                        attention_by_step[step],
                        top_r=args.top_r_descriptor,
                        eps=args.eps,
                        print_debug=not printed,
                        step=step,
                    )
                    printed = True
                    _, _, channels, feature_dim = phi.shape
                    if k_dim is None:
                        k_dim = int(channels)
                        d_phi = int(feature_dim)
                    phi_flat = phi.reshape(-1, channels, feature_dim).detach().cpu().double()
                    if step not in sum_by_step:
                        sum_by_step[step] = torch.zeros(channels, feature_dim, dtype=torch.float64)
                        sumsq_by_step[step] = torch.zeros(channels, feature_dim, dtype=torch.float64)
                        count_by_step[step] = 0
                    sum_by_step[step] += phi_flat.sum(dim=0)
                    sumsq_by_step[step] += (phi_flat * phi_flat).sum(dim=0)
                    count_by_step[step] += int(phi_flat.shape[0])

    mu = []
    var = []
    for step in selected_steps:
        step = int(step)
        if count_by_step.get(step, 0) == 0:
            raise ValueError(f"no validation descriptors collected for step {step}")
        count = float(count_by_step[step])
        step_mu = sum_by_step[step] / count
        step_var = torch.clamp((sumsq_by_step[step] / count) - (step_mu * step_mu), min=0.0)
        mu.append(step_mu.float())
        var.append(step_var.float())
    mu = torch.stack(mu, dim=0)
    var = torch.stack(var, dim=0)
    print(f"[ATTN-DSAC] mu shape [S,K,D_phi]: {list(mu.shape)}")
    print(f"[ATTN-DSAC] var shape [S,K,D_phi]: {list(var.shape)}")
    return mu, var, int(k_dim), int(d_phi)


def collect_validation_distances(model, loader, selected_steps, args, mu, var):
    distances_by_step = {int(step): [] for step in selected_steps}
    printed = False

    with torch.no_grad():
        with tqdm(loader, desc="ATTN-DSAC prototype pass2 val distances", mininterval=5.0, maxinterval=50.0) as it:
            for batch in it:
                attention_by_step = collect_attention_by_step(model, batch, selected_steps)
                for step_index, step in enumerate(selected_steps):
                    step = int(step)
                    phi = attention_to_descriptor(
                        attention_by_step[step],
                        top_r=args.top_r_descriptor,
                        eps=args.eps,
                        print_debug=False,
                        step=step,
                    )
                    dist = descriptor_mahalanobis(phi, mu[step_index], var[step_index], args.ridge)
                    if not printed:
                        print(f"[ATTN-DSAC] validation distance shape [B,L,K]: {list(dist.shape)}")
                        printed = True
                    distances_by_step[step].append(dist.reshape(-1, dist.shape[-1]).detach().cpu().float())

    sorted_steps = []
    for step in selected_steps:
        step = int(step)
        dist = torch.cat(distances_by_step[step], dim=0)
        sorted_dist = torch.sort(dist, dim=0).values.transpose(0, 1).contiguous()
        print(f"[ATTN-DSAC] val_dist_sorted step={step} shape [K,N]: {list(sorted_dist.shape)}")
        sorted_steps.append(sorted_dist)
    return torch.stack(sorted_steps, dim=0)


def channel_aggregate(q, topk):
    return {
        "ch_mean": q.mean(dim=-1),
        "ch_max": q.max(dim=-1).values,
        "ch_top3": q.topk(k=min(int(topk), q.shape[-1]), dim=-1).values.mean(dim=-1),
    }


def step_top2mean(values):
    k = min(2, values.shape[0])
    return values.topk(k=k, dim=0).values.mean(dim=0)


def aggregate_q_by_method(q_by_step, selected_steps, topk):
    channel_step_scores = {}
    for step in selected_steps:
        channel_step_scores[int(step)] = channel_aggregate(q_by_step[int(step)], topk)

    methods = {}
    for channel_name in ["ch_mean", "ch_max", "ch_top3"]:
        stack = torch.stack(
            [channel_step_scores[int(step)][channel_name] for step in selected_steps],
            dim=0,
        )
        methods[f"attn_dsac_{channel_name}_step_mean"] = stack.mean(dim=0)
        methods[f"attn_dsac_{channel_name}_step_median"] = stack.median(dim=0).values
        methods[f"attn_dsac_{channel_name}_step_max"] = stack.max(dim=0).values
        if channel_name == "ch_top3":
            methods[f"attn_dsac_{channel_name}_step_top2mean"] = step_top2mean(stack)
    return {name: methods[name] for name in ATTN_DSAC_METHODS}


def compute_attn_dsac_window_methods(attention_by_step, selected_steps, mu, var, val_dist_sorted, eps, ridge, topk, top_r, print_debug=False):
    q_by_step = {}
    printed = False
    for step_index, step in enumerate(selected_steps):
        step = int(step)
        phi = attention_to_descriptor(
            attention_by_step[step],
            top_r=top_r,
            eps=eps,
            print_debug=print_debug and not printed,
            step=step,
        )
        printed = True
        dist = descriptor_mahalanobis(phi, mu[step_index], var[step_index], ridge)
        q = empirical_cdf_scores(dist, val_dist_sorted[step_index])
        q_by_step[step] = q
        if print_debug:
            print(
                f"[ATTN-DSAC] step={step} q shape [num_windows,L,K]: {list(q.shape)} "
                f"min/max={float(q.min())}/{float(q.max())}"
            )
    return aggregate_q_by_method(q_by_step, selected_steps, topk)


def collect_calibration_norm_stats(model, loader, selected_steps, args, mu, var, val_dist_sorted):
    window_chunks = {name: [] for name in CALIBRATION_SCORE_NAMES}
    with torch.no_grad():
        with tqdm(loader, desc="ATTN-DSAC calibration robust stats", mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                pathB_result = collect_pathb_result(model, batch, selected_steps)
                batch_methods = compute_attn_dsac_window_methods(
                    pathB_result["feature_attention_by_step"],
                    selected_steps,
                    mu,
                    var,
                    val_dist_sorted,
                    eps=args.eps,
                    ridge=args.ridge,
                    topk=args.channel_topk,
                    top_r=args.top_r_descriptor,
                    print_debug=(batch_no == 1),
                )
                window_chunks["final_recon_score"].append(pathB_result["final_recon_score"].detach().cpu().float())
                for method in ATTN_DSAC_METHODS:
                    window_chunks[method].append(batch_methods[method].detach().cpu().float())

    medians = []
    scales = []
    mads = []
    stds = []
    scale_sources = []
    for score_name in CALIBRATION_SCORE_NAMES:
        window_scores = torch.cat(window_chunks[score_name], dim=0)
        score_time, valid_indices = backproject_windows_to_valid_time(window_scores, args.split)
        stats = robust_norm_stats(score_time, args.eps)
        medians.append(stats["median"])
        scales.append(stats["scale"])
        mads.append(stats["mad"])
        stds.append(stats["std"])
        scale_sources.append(stats["scale_source"])
        print(
            f"[ATTN-DSAC] calibration {score_name}: n={len(score_time)}, "
            f"valid_indices_len={len(valid_indices)}, median={stats['median']}, "
            f"scale={stats['scale']} ({stats['scale_source']}), mad={stats['mad']}, std={stats['std']}"
        )
    return {
        "calibration_score_names": np.asarray(CALIBRATION_SCORE_NAMES),
        "calibration_median": np.asarray(medians, dtype=np.float32),
        "calibration_scale": np.asarray(scales, dtype=np.float32),
        "calibration_mad": np.asarray(mads, dtype=np.float32),
        "calibration_std": np.asarray(stds, dtype=np.float32),
        "calibration_scale_source": np.asarray(scale_sources),
        "calibration_rule": np.asarray("robust_z_median_1.4826mad_with_std_fallback"),
    }


def run_build_prototype(args, save_id, selected_steps):
    require_no_mve_name(args.base_dataset, "--base_dataset")
    require_no_mve_name(args.model_dataset, "--model_dataset")
    if args.dataset:
        require_no_mve_name(args.dataset, "--dataset")
    print("[ATTN-DSAC] mode=build_prototype")
    print(f"[ATTN-DSAC] base_dataset={args.base_dataset}; model_dataset={args.model_dataset}; save={save_id}")

    validation_loader_1 = build_validation_loader(args.base_dataset, args.batch_size, args.split, args.val_ratio, args.seed)
    validation_loader_2 = build_validation_loader(args.base_dataset, args.batch_size, args.split, args.val_ratio, args.seed)
    calibration_loader = build_validation_loader(args.base_dataset, args.batch_size, args.split, args.val_ratio, args.seed)
    model, subset_name, model_diffusion_step = load_model_for_save(args, save_id)
    mu, var, k_dim, d_phi = compute_descriptor_stats(model, validation_loader_1, selected_steps, args)
    val_dist_sorted = collect_validation_distances(model, validation_loader_2, selected_steps, args, mu, var)
    calibration_payload = collect_calibration_norm_stats(
        model,
        calibration_loader,
        selected_steps,
        args,
        mu,
        var,
        val_dist_sorted,
    )

    prototype_path = default_prototype_path(args.output_root, args.base_dataset, selected_steps, args.seed, save_id)
    ensure_outputs_can_be_written([prototype_path], overwrite=args.overwrite)
    os.makedirs(os.path.dirname(prototype_path), exist_ok=True)
    np.savez_compressed(
        prototype_path,
        selected_steps=np.asarray(selected_steps, dtype=np.int64),
        mu=mu.numpy().astype(np.float32),
        var=var.numpy().astype(np.float32),
        val_dist_sorted=val_dist_sorted.numpy().astype(np.float32),
        C=np.asarray(k_dim, dtype=np.int64),
        D_phi=np.asarray(d_phi, dtype=np.int64),
        ridge=np.asarray(float(args.ridge), dtype=np.float32),
        eps=np.asarray(float(args.eps), dtype=np.float32),
        top_r_descriptor=np.asarray(int(args.top_r_descriptor), dtype=np.int64),
        val_ratio=np.asarray(float(args.val_ratio), dtype=np.float32),
        seed=np.asarray(int(args.seed), dtype=np.int64),
        save_name=np.asarray(save_id),
        base_dataset=np.asarray(args.base_dataset),
        model_dataset=np.asarray(args.model_dataset),
        model_subset_name=np.asarray(subset_name),
        model_diffusion_step=np.asarray(int(model_diffusion_step), dtype=np.int64),
        score_type=np.asarray("real_feature_attention_dsac_validation_prototype"),
        attention_source=np.asarray("feature_layer_true_attention_mean_over_blocks_heads"),
        **calibration_payload,
    )
    print(f"[ATTN-DSAC] saved prototype: {prototype_path}")


def load_prototype(path, selected_steps):
    if not path:
        raise ValueError("--prototype_path is required in score mode")
    if not os.path.exists(path):
        raise FileNotFoundError(f"prototype not found: {path}")
    payload = np.load(path, allow_pickle=False)
    proto_steps = [int(step) for step in payload["selected_steps"].tolist()]
    if proto_steps != [int(step) for step in selected_steps]:
        raise ValueError(f"prototype selected_steps={proto_steps} != requested selected_steps={selected_steps}")
    return payload


def load_calibration_stats_from_prototype(prototype):
    required = ["calibration_score_names", "calibration_median", "calibration_scale"]
    missing = [key for key in required if key not in prototype]
    if missing:
        raise ValueError(
            "prototype is missing fusion calibration stats "
            f"{missing}. Rebuild prototype with build_prototype mode."
        )
    names = [str(name) for name in prototype["calibration_score_names"].tolist()]
    medians = prototype["calibration_median"].astype(np.float32)
    scales = prototype["calibration_scale"].astype(np.float32)
    return {
        name: {"median": float(medians[index]), "scale": float(scales[index])}
        for index, name in enumerate(names)
    }


def run_score(args, save_id, selected_steps):
    if not args.dataset:
        raise ValueError("--dataset is required in score mode")
    print("[ATTN-DSAC] mode=score")
    print(f"[ATTN-DSAC] dataset={args.dataset}; base_dataset={args.base_dataset}; save={save_id}")
    prototype = load_prototype(args.prototype_path, selected_steps)
    mu = torch.from_numpy(prototype["mu"].astype(np.float32))
    var = torch.from_numpy(prototype["var"].astype(np.float32))
    val_dist_sorted = prototype["val_dist_sorted"].astype(np.float32)
    calibration_stats = load_calibration_stats_from_prototype(prototype)
    print(f"[ATTN-DSAC] loaded prototype mu shape: {list(mu.shape)}")
    print(f"[ATTN-DSAC] loaded prototype var shape: {list(var.shape)}")
    print(f"[ATTN-DSAC] loaded prototype val_dist_sorted shape [S,K,N]: {list(val_dist_sorted.shape)}")

    labels = load_labels(args.dataset)
    label_sum = check_dataset_label_sum(args.dataset, labels)
    label_len = int(labels.shape[0])

    model, _, _ = load_model_for_save(args, save_id)
    loader = build_variant_loader(args.dataset, args.batch_size, args.split)
    window_chunks_by_method = {method: [] for method in ATTN_DSAC_METHODS}
    final_recon_chunks = []

    with torch.no_grad():
        with tqdm(loader, desc=f"ATTN-DSAC score {args.dataset} {save_id}", mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                pathB_result = collect_pathb_result(model, batch, selected_steps)
                batch_methods = compute_attn_dsac_window_methods(
                    pathB_result["feature_attention_by_step"],
                    selected_steps,
                    mu,
                    var,
                    val_dist_sorted,
                    eps=float(prototype["eps"]),
                    ridge=float(prototype["ridge"]),
                    topk=args.channel_topk,
                    top_r=int(prototype["top_r_descriptor"]),
                    print_debug=(batch_no == 1),
                )
                final_recon_chunks.append(pathB_result["final_recon_score"].detach().cpu().float())
                for method, score_window in batch_methods.items():
                    window_chunks_by_method[method].append(score_window.detach().cpu().float())

    output = {
        "labels": labels.astype(np.int64),
        "selected_steps": np.asarray(selected_steps, dtype=np.int64),
        "dataset": np.asarray(args.dataset),
        "base_dataset": np.asarray(args.base_dataset),
        "prototype_path": np.asarray(args.prototype_path),
        "raw_label_sum": np.asarray(label_sum, dtype=np.int64),
        "label_sum": np.asarray(label_sum, dtype=np.int64),
        "raw_label_len": np.asarray(label_len, dtype=np.int64),
        "label_len": np.asarray(label_len, dtype=np.int64),
        "save": np.asarray(save_id),
        "fusion_alpha": np.asarray(float(args.fusion_alpha), dtype=np.float32),
        "fusion_formula": np.asarray("fused = alpha * z_recon + (1-alpha) * z_attn_dsac"),
        "fusion_normalization": np.asarray("validation_robust_z_median_1.4826mad_with_std_fallback"),
        "score_type": np.asarray("real_feature_attention_dsac"),
    }

    final_recon_windows = torch.cat(final_recon_chunks, dim=0)
    final_recon_score, final_valid_indices = backproject_windows_to_valid_time(final_recon_windows, args.split)
    if final_valid_indices.size == 0:
        raise RuntimeError("final_recon_score produced empty valid_indices")
    if final_valid_indices.max() >= label_len:
        raise RuntimeError(
            f"final_recon_score valid index exceeds label range: max={final_valid_indices.max()}, "
            f"label_len={label_len}"
        )
    final_recon_stats = calibration_stats["final_recon_score"]
    z_recon = robust_z(final_recon_score, final_recon_stats["median"], final_recon_stats["scale"], args.eps)
    output["final_recon_score"] = final_recon_score
    output["z_final_recon_score"] = z_recon
    output["final_recon_score_score_len"] = np.asarray(len(final_recon_score), dtype=np.int64)
    output["final_recon_calibration_median"] = np.asarray(final_recon_stats["median"], dtype=np.float32)
    output["final_recon_calibration_scale"] = np.asarray(final_recon_stats["scale"], dtype=np.float32)

    valid_indices = None
    for method in ATTN_DSAC_METHODS:
        window_scores = torch.cat(window_chunks_by_method[method], dim=0)
        score_time, method_valid_indices = backproject_windows_to_valid_time(window_scores, args.split)
        if not np.isfinite(score_time).all():
            raise RuntimeError(f"{method} contains nan/inf")
        if method_valid_indices.size == 0:
            raise RuntimeError(f"{method} produced empty valid_indices")
        if method_valid_indices.max() >= label_len:
            raise RuntimeError(
                f"{method} valid index exceeds label range: max={method_valid_indices.max()}, label_len={label_len}"
            )
        if valid_indices is None:
            valid_indices = method_valid_indices
            if not np.array_equal(valid_indices, final_valid_indices):
                raise RuntimeError(f"{method} valid_indices differ from final_recon_score valid_indices")
            labels_aligned = labels[valid_indices]
            aligned_label_sum = int(labels_aligned.sum())
            dropped_points = int(label_len - len(valid_indices))
            print(f"[ATTN-DSAC] raw_label_len={label_len}")
            print(f"[ATTN-DSAC] score_len={len(score_time)}")
            print(f"[ATTN-DSAC] valid_indices_len={len(valid_indices)}")
            print(f"[ATTN-DSAC] raw_label_sum={label_sum}")
            print(f"[ATTN-DSAC] aligned_label_sum={aligned_label_sum}")
            print(f"[ATTN-DSAC] dropped_points={dropped_points}")
            if aligned_label_sum < label_sum:
                print(
                    f"[WARN] aligned_label_sum < raw_label_sum for {args.dataset}: "
                    f"{aligned_label_sum} < {label_sum}. Some injected anomalies are outside scored valid region."
                )
            output["valid_indices"] = valid_indices.astype(np.int64)
            output["labels_aligned"] = labels_aligned.astype(np.int64)
            output["aligned_label_sum"] = np.asarray(aligned_label_sum, dtype=np.int64)
            output["valid_indices_len"] = np.asarray(len(valid_indices), dtype=np.int64)
            output["dropped_points"] = np.asarray(dropped_points, dtype=np.int64)
        elif not np.array_equal(valid_indices, method_valid_indices):
            raise RuntimeError(f"{method} valid_indices differ from previous methods")

        output[method] = score_time
        stats = calibration_stats[method]
        z_attn = robust_z(score_time, stats["median"], stats["scale"], args.eps)
        fused_score = (float(args.fusion_alpha) * z_recon) + ((1.0 - float(args.fusion_alpha)) * z_attn)
        fused_name = f"fused_{method}"
        output[f"z_{method}"] = z_attn
        output[fused_name] = fused_score.astype(np.float32)
        output[f"{method}_score_len"] = np.asarray(len(score_time), dtype=np.int64)
        output[f"{fused_name}_score_len"] = np.asarray(len(fused_score), dtype=np.int64)
        print(
            f"[ATTN-DSAC] {method}: score_len={len(score_time)}, raw_label_len={label_len}, "
            f"fused={fused_name}, alpha={args.fusion_alpha}"
        )

    output["score_len"] = np.asarray(len(output[ATTN_DSAC_METHODS[0]]), dtype=np.int64)
    if int(output["score_len"]) != int(output["valid_indices_len"]):
        raise RuntimeError(
            f"score_len != valid_indices_len: {int(output['score_len'])} vs {int(output['valid_indices_len'])}"
        )

    output_dir = os.path.join(args.output_root, args.base_dataset)
    os.makedirs(output_dir, exist_ok=True)
    score_path = os.path.join(output_dir, f"{args.dataset}_{save_id}_scores.npz")
    ensure_outputs_can_be_written([score_path], overwrite=args.overwrite)
    np.savez_compressed(score_path, **output)
    print(f"[ATTN-DSAC] saved scores: {score_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build_prototype", "score"], required=True)
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--diffusion_step", type=int, default=None)
    parser.add_argument("--train_result_root", type=str, default="train_result")
    parser.add_argument("--base_dataset", type=str, default="SMAP")
    parser.add_argument("--model_dataset", type=str, default="SMAP")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--selected_steps", type=str, default=DEFAULT_SELECTED_STEPS)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--saves", nargs="*", default=None)
    parser.add_argument("--output_root", type=str, default="pathB_result_attn_dsac")
    parser.add_argument("--prototype_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--top_r_descriptor", type=int, default=3)
    parser.add_argument("--channel_topk", type=int, default=3)
    parser.add_argument("--fusion_alpha", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    selected_steps = unique_keep_order(parse_steps(args.selected_steps))
    save_ids = infer_save_ids(args.train_result_root, args.saves)
    if len(save_ids) != 1:
        print(f"[ATTN-DSAC] multiple saves requested: {save_ids}")
    for save_id in save_ids:
        if args.mode == "build_prototype":
            run_build_prototype(args, save_id, selected_steps)
        else:
            run_score(args, save_id, selected_steps)


if __name__ == "__main__":
    main()
