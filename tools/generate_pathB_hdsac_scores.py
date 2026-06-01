import argparse
import os
import pickle
import random
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from dataset import TrainData, get_dataloader
from generate_pathB_hidden_path_scores import (
    infer_save_ids,
    load_model_for_save,
    parse_steps,
    unique_keep_order,
)
from utils import ensure_outputs_can_be_written


DEFAULT_SELECTED_STEPS = "40,25,10,0"
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
FUSED_METHODS = [f"fused_{method}" for method in HDSAC_METHODS]
CALIBRATION_SCORE_NAMES = ["final_recon_score"] + HDSAC_METHODS


def set_all_seeds(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def require_no_mve_name(name, arg_name):
    if "MVE" in str(name):
        raise ValueError(
            f"{arg_name}={name} is forbidden in build_prototype. "
            "HDSAC prototype must come only from base_dataset training validation split."
        )


def data_paths(dataset):
    return (
        f"data/Machine/{dataset}_train.pkl",
        f"data/Machine/{dataset}_test.pkl",
        f"data/Machine/{dataset}_test_label.pkl",
    )


def ensure_paths_exist(paths):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("missing data files:\n" + "\n".join(missing))


def build_validation_loader(base_dataset, batch_size, split, val_ratio, seed):
    train_path, test_path, _ = data_paths(base_dataset)
    ensure_paths_exist([train_path, test_path])
    full_train = TrainData(train_path, test_path, split=split)
    val_len = int(float(val_ratio) * len(full_train))
    if val_len <= 0:
        raise ValueError(f"validation split is empty: len={len(full_train)}, val_ratio={val_ratio}")
    train_len = len(full_train) - val_len
    generator = torch.Generator().manual_seed(int(seed))
    _, valid_data = random_split(full_train, [train_len, val_len], generator=generator)
    print(
        f"[HDSAC] validation split from {base_dataset}: "
        f"train_windows={len(full_train)}, val_windows={len(valid_data)}, "
        f"val_ratio={val_ratio}, seed={seed}"
    )
    return DataLoader(valid_data, batch_size=batch_size, shuffle=False)


def build_variant_loader(dataset, batch_size, split):
    train_path, test_path, label_path = data_paths(dataset)
    ensure_paths_exist([train_path, test_path, label_path])
    _, _, test_loader, _ = get_dataloader(
        train_path,
        test_path,
        label_path,
        batch_size=batch_size,
        window_split=2,
        split=split,
    )
    return test_loader


def load_labels(dataset):
    _, _, label_path = data_paths(dataset)
    ensure_paths_exist([label_path])
    with open(label_path, "rb") as f:
        labels = pickle.load(f)
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def selected_steps_key(selected_steps):
    return "_".join(str(int(step)) for step in selected_steps)


def default_prototype_path(output_root, base_dataset, selected_steps, seed, save_id):
    filename = f"prototype_steps_{selected_steps_key(selected_steps)}_seed{int(seed)}_{save_id}.npz"
    return os.path.join(output_root, base_dataset, filename)


def check_captured_steps(hidden_by_step, selected_steps):
    captured = sorted(int(step) for step in hidden_by_step)
    requested = [int(step) for step in selected_steps]
    print(f"[HDSAC] selected steps requested: {requested}")
    print(f"[HDSAC] selected steps actually captured: {captured}")
    missing = [step for step in requested if step not in hidden_by_step]
    if missing:
        raise RuntimeError(f"selected steps were not all captured: missing={missing}, captured={captured}")


def hidden_to_descriptor(hidden, tau, top_r, eps, print_debug=False, step=None):
    if hidden.dim() != 4:
        raise ValueError(f"hidden must have shape [B,D_h,C,L], got {list(hidden.shape)}")
    if print_debug:
        print(f"[HDSAC] step={step} hidden shape [B,D_h,C,L]: {list(hidden.shape)}")

    z = hidden.permute(0, 3, 2, 1).contiguous()  # [B,L,C,D_h]
    if print_debug:
        print(f"[HDSAC] step={step} hidden permuted shape [B,L,C,D_h]: {list(z.shape)}")

    z_norm = z / (torch.linalg.norm(z, dim=-1, keepdim=True) + float(eps))
    logits = torch.matmul(z_norm, z_norm.transpose(-1, -2)) / float(tau)
    affinity = torch.softmax(logits, dim=-1)
    c_dim = affinity.shape[-1]

    entropy = -(affinity * torch.log(affinity + float(eps))).sum(dim=-1)
    if c_dim > 1:
        entropy = entropy / np.log(float(c_dim))
    self_strength = torch.diagonal(affinity, dim1=-2, dim2=-1)
    topk = min(int(top_r), c_dim)
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
        print(f"[HDSAC] step={step} topology A shape [B,L,C,C]: {list(affinity.shape)}")
        print(f"[HDSAC] step={step} phi shape [B,L,C,C+3]: {list(phi.shape)}")
    return phi


def collect_pathb_result(model, batch, selected_steps):
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
    check_captured_steps(hidden_by_step, selected_steps)
    return pathB_result


def collect_hidden_by_step(model, batch, selected_steps):
    pathB_result = collect_pathb_result(model, batch, selected_steps)
    hidden_by_step = pathB_result["hidden_by_step"]
    return hidden_by_step


def compute_descriptor_stats(model, loader, selected_steps, args):
    sum_by_step = {}
    sumsq_by_step = {}
    count_by_step = {}
    c_dim = None
    d_phi = None
    printed = False

    with torch.no_grad():
        with tqdm(loader, desc="HDSAC prototype pass1 stats", mininterval=5.0, maxinterval=50.0) as it:
            for batch in it:
                hidden_by_step = collect_hidden_by_step(model, batch, selected_steps)
                for step in selected_steps:
                    step = int(step)
                    phi = hidden_to_descriptor(
                        hidden_by_step[step],
                        tau=args.tau,
                        top_r=args.top_r_descriptor,
                        eps=args.eps,
                        print_debug=not printed,
                        step=step,
                    )
                    printed = True
                    b_size, length, channels, feature_dim = phi.shape
                    if c_dim is None:
                        c_dim = int(channels)
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
    print(f"[HDSAC] mu shape [S,C,D_phi]: {list(mu.shape)}")
    print(f"[HDSAC] var shape [S,C,D_phi]: {list(var.shape)}")
    return mu, var, int(c_dim), int(d_phi)


def descriptor_mahalanobis(phi, mu_step, var_step, ridge):
    # phi: [B,L,C,D_phi], mu/var: [C,D_phi] -> [B,L,C]
    diff = phi - mu_step.reshape(1, 1, mu_step.shape[0], mu_step.shape[1]).to(phi.device, phi.dtype)
    denom = var_step.reshape(1, 1, var_step.shape[0], var_step.shape[1]).to(phi.device, phi.dtype) + float(ridge)
    return torch.sqrt(torch.clamp(torch.sum((diff * diff) / denom, dim=-1), min=0.0))


def robust_norm_stats(values, eps):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot compute robust normalization stats from empty values")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    std = float(np.std(values))
    if mad >= float(eps):
        scale = float(1.4826 * mad)
        scale_source = "mad"
    elif std >= float(eps):
        scale = std
        scale_source = "std"
    else:
        scale = float(eps)
        scale_source = "eps"
    return {
        "median": median,
        "mad": mad,
        "std": std,
        "scale": scale,
        "scale_source": scale_source,
    }


def robust_z(values, median, scale, eps):
    values = np.asarray(values, dtype=np.float32)
    return ((values - float(median)) / (float(scale) + float(eps))).astype(np.float32)


def collect_validation_distances(model, loader, selected_steps, args, mu, var):
    distances_by_step = {int(step): [] for step in selected_steps}
    printed = False

    with torch.no_grad():
        with tqdm(loader, desc="HDSAC prototype pass2 val distances", mininterval=5.0, maxinterval=50.0) as it:
            for batch in it:
                hidden_by_step = collect_hidden_by_step(model, batch, selected_steps)
                for step_index, step in enumerate(selected_steps):
                    step = int(step)
                    phi = hidden_to_descriptor(
                        hidden_by_step[step],
                        tau=args.tau,
                        top_r=args.top_r_descriptor,
                        eps=args.eps,
                        print_debug=False,
                        step=step,
                    )
                    dist = descriptor_mahalanobis(phi, mu[step_index], var[step_index], args.ridge)
                    if not printed:
                        print(f"[HDSAC] validation distance shape [B,L,C]: {list(dist.shape)}")
                        printed = True
                    distances_by_step[step].append(dist.reshape(-1, dist.shape[-1]).detach().cpu().float())

    sorted_steps = []
    for step in selected_steps:
        step = int(step)
        dist = torch.cat(distances_by_step[step], dim=0)  # [N_time,C]
        sorted_dist = torch.sort(dist, dim=0).values.transpose(0, 1).contiguous()  # [C,N_time]
        print(f"[HDSAC] val_dist_sorted step={step} shape [C,N]: {list(sorted_dist.shape)}")
        sorted_steps.append(sorted_dist)
    return torch.stack(sorted_steps, dim=0)  # [S,C,N_time]


def run_build_prototype(args, save_id, selected_steps):
    require_no_mve_name(args.base_dataset, "--base_dataset")
    require_no_mve_name(args.model_dataset, "--model_dataset")
    if args.dataset:
        require_no_mve_name(args.dataset, "--dataset")
    print("[HDSAC] mode=build_prototype")
    print(f"[HDSAC] base_dataset={args.base_dataset}; model_dataset={args.model_dataset}; save={save_id}")

    validation_loader_1 = build_validation_loader(
        args.base_dataset,
        args.batch_size,
        args.split,
        args.val_ratio,
        args.seed,
    )
    validation_loader_2 = build_validation_loader(
        args.base_dataset,
        args.batch_size,
        args.split,
        args.val_ratio,
        args.seed,
    )
    calibration_loader = build_validation_loader(
        args.base_dataset,
        args.batch_size,
        args.split,
        args.val_ratio,
        args.seed,
    )
    model, subset_name, model_diffusion_step = load_model_for_save(args, save_id)
    mu, var, c_dim, d_phi = compute_descriptor_stats(model, validation_loader_1, selected_steps, args)
    val_dist_sorted = collect_validation_distances(
        model,
        validation_loader_2,
        selected_steps,
        args,
        mu,
        var,
    )
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
        C=np.asarray(c_dim, dtype=np.int64),
        D_phi=np.asarray(d_phi, dtype=np.int64),
        tau=np.asarray(float(args.tau), dtype=np.float32),
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
        score_type=np.asarray("hidden_topology_dsac_validation_prototype"),
        **calibration_payload,
    )
    print(f"[HDSAC] saved prototype: {prototype_path}")


def empirical_cdf_scores(dist, sorted_dist):
    # dist: [B,L,C], sorted_dist: [C,N] -> q [B,L,C]
    dist_cpu = dist.detach().cpu().float()
    if torch.is_tensor(sorted_dist):
        sorted_dist_np = sorted_dist.detach().cpu().float().numpy()
    else:
        sorted_dist_np = np.asarray(sorted_dist, dtype=np.float32)
    q_channels = []
    for channel in range(dist_cpu.shape[-1]):
        flat = dist_cpu[..., channel].reshape(-1).numpy()
        reference = sorted_dist_np[channel]
        rank = np.searchsorted(reference, flat, side="right")
        q = rank.astype(np.float32) / float(len(reference))
        q_channels.append(torch.from_numpy(q.reshape(dist_cpu.shape[0], dist_cpu.shape[1])))
    return torch.stack(q_channels, dim=-1)


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
        methods[f"hdsac_{channel_name}_step_mean"] = stack.mean(dim=0)
        methods[f"hdsac_{channel_name}_step_median"] = stack.median(dim=0).values
        methods[f"hdsac_{channel_name}_step_max"] = stack.max(dim=0).values
        if channel_name == "ch_top3":
            methods[f"hdsac_{channel_name}_step_top2mean"] = step_top2mean(stack)
    return {name: methods[name] for name in HDSAC_METHODS}


def compute_hdsac_window_methods(hidden_by_step, selected_steps, mu, var, val_dist_sorted, tau, top_r, eps, ridge, topk, print_debug=False):
    q_by_step = {}
    printed = False
    for step_index, step in enumerate(selected_steps):
        step = int(step)
        phi = hidden_to_descriptor(
            hidden_by_step[step],
            tau=tau,
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
                f"[HDSAC] step={step} q shape [num_windows,L,C]: {list(q.shape)} "
                f"min/max={float(q.min())}/{float(q.max())}"
            )
    return aggregate_q_by_method(q_by_step, selected_steps, topk)


def backproject_windows_to_valid_time(window_scores, split):
    if window_scores.dim() != 2:
        raise ValueError(f"window_scores must be [num_windows,L], got {list(window_scores.shape)}")
    num_windows, length = window_scores.shape
    if num_windows <= 0:
        raise ValueError("window_scores is empty")
    left = length // split
    right = length - left
    stride = right - left
    parts = [window_scores[0, :left].detach().cpu().reshape(-1)]
    parts.append(window_scores[:, left:right].detach().cpu().reshape(-1))
    score_time = torch.cat(parts, dim=0).float()

    index_parts = [torch.arange(0, left, dtype=torch.long)]
    middle_indices = [
        torch.arange(window_index * stride + left, window_index * stride + right, dtype=torch.long)
        for window_index in range(num_windows)
    ]
    index_parts.extend(middle_indices)
    valid_indices = torch.cat(index_parts, dim=0)
    if score_time.numel() != valid_indices.numel():
        raise RuntimeError(
            f"score_len != valid_indices_len after backprojection: "
            f"{score_time.numel()} vs {valid_indices.numel()}"
        )
    return score_time.numpy().astype(np.float32), valid_indices.numpy().astype(np.int64)


def collect_calibration_norm_stats(model, loader, selected_steps, args, mu, var, val_dist_sorted):
    window_chunks = {name: [] for name in CALIBRATION_SCORE_NAMES}
    with torch.no_grad():
        with tqdm(loader, desc="HDSAC calibration robust stats", mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                pathB_result = collect_pathb_result(model, batch, selected_steps)
                batch_methods = compute_hdsac_window_methods(
                    pathB_result["hidden_by_step"],
                    selected_steps,
                    mu,
                    var,
                    val_dist_sorted,
                    tau=args.tau,
                    top_r=args.top_r_descriptor,
                    eps=args.eps,
                    ridge=args.ridge,
                    topk=args.channel_topk,
                    print_debug=(batch_no == 1),
                )
                final_recon_score = pathB_result["final_recon_score"].detach().cpu().float()
                window_chunks["final_recon_score"].append(final_recon_score)
                for method in HDSAC_METHODS:
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
            f"[HDSAC] calibration {score_name}: n={len(score_time)}, "
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


def check_dataset_label_sum(dataset, labels):
    label_sum = int(labels.sum())
    print(f"[HDSAC] dataset={dataset} label_sum={label_sum}")
    if re.match(r"^SMAP_MVE_d\d+$", dataset) and label_sum != 100:
        raise RuntimeError(f"{dataset} label_sum must be 100, got {label_sum}")
    return label_sum


def load_calibration_stats_from_prototype(prototype):
    required = [
        "calibration_score_names",
        "calibration_median",
        "calibration_scale",
    ]
    missing = [key for key in required if key not in prototype]
    if missing:
        raise ValueError(
            "prototype is missing fusion calibration stats "
            f"{missing}. Rebuild prototype with the updated build_prototype mode."
        )
    names = [str(name) for name in prototype["calibration_score_names"].tolist()]
    medians = prototype["calibration_median"].astype(np.float32)
    scales = prototype["calibration_scale"].astype(np.float32)
    return {
        name: {
            "median": float(medians[index]),
            "scale": float(scales[index]),
        }
        for index, name in enumerate(names)
    }


def run_score(args, save_id, selected_steps):
    if not args.dataset:
        raise ValueError("--dataset is required in score mode")
    print("[HDSAC] mode=score")
    print(f"[HDSAC] dataset={args.dataset}; base_dataset={args.base_dataset}; save={save_id}")
    prototype = load_prototype(args.prototype_path, selected_steps)
    mu = torch.from_numpy(prototype["mu"].astype(np.float32))
    var = torch.from_numpy(prototype["var"].astype(np.float32))
    val_dist_sorted = prototype["val_dist_sorted"].astype(np.float32)
    calibration_stats = load_calibration_stats_from_prototype(prototype)
    c_dim = int(prototype["C"])
    d_phi = int(prototype["D_phi"])
    print(f"[HDSAC] loaded prototype mu shape: {list(mu.shape)}")
    print(f"[HDSAC] loaded prototype var shape: {list(var.shape)}")
    print(f"[HDSAC] loaded prototype val_dist_sorted shape [S,C,N]: {list(val_dist_sorted.shape)}")
    print(f"[HDSAC] loaded prototype C={c_dim}, D_phi={d_phi}")

    labels = load_labels(args.dataset)
    label_sum = check_dataset_label_sum(args.dataset, labels)
    label_len = int(labels.shape[0])

    model, _, _ = load_model_for_save(args, save_id)
    loader = build_variant_loader(args.dataset, args.batch_size, args.split)
    window_chunks_by_method = {method: [] for method in HDSAC_METHODS}
    final_recon_chunks = []

    with torch.no_grad():
        with tqdm(loader, desc=f"HDSAC score {args.dataset} {save_id}", mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                pathB_result = collect_pathb_result(model, batch, selected_steps)
                batch_methods = compute_hdsac_window_methods(
                    pathB_result["hidden_by_step"],
                    selected_steps,
                    mu,
                    var,
                    val_dist_sorted,
                    tau=float(prototype["tau"]),
                    top_r=int(prototype["top_r_descriptor"]),
                    eps=float(prototype["eps"]),
                    ridge=float(prototype["ridge"]),
                    topk=args.channel_topk,
                    print_debug=(batch_no == 1),
                )
                final_recon_chunks.append(pathB_result["final_recon_score"].detach().cpu().float())
                for method, score_window in batch_methods.items():
                    window_chunks_by_method[method].append(score_window.detach().cpu().float())

    valid_indices = None
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
        "fusion_formula": np.asarray("fused = alpha * z_recon + (1-alpha) * z_hdsac"),
        "fusion_normalization": np.asarray("validation_robust_z_median_1.4826mad_with_std_fallback"),
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
    z_recon = robust_z(
        final_recon_score,
        final_recon_stats["median"],
        final_recon_stats["scale"],
        args.eps,
    )
    output["final_recon_score"] = final_recon_score
    output["z_final_recon_score"] = z_recon
    output["final_recon_score_score_len"] = np.asarray(len(final_recon_score), dtype=np.int64)
    output["final_recon_calibration_median"] = np.asarray(final_recon_stats["median"], dtype=np.float32)
    output["final_recon_calibration_scale"] = np.asarray(final_recon_stats["scale"], dtype=np.float32)

    for method in HDSAC_METHODS:
        window_scores = torch.cat(window_chunks_by_method[method], dim=0)
        score_time, method_valid_indices = backproject_windows_to_valid_time(window_scores, args.split)
        if not np.isfinite(score_time).all():
            raise RuntimeError(f"{method} contains nan/inf")
        if method_valid_indices.size == 0:
            raise RuntimeError(f"{method} produced empty valid_indices")
        if method_valid_indices.max() >= label_len:
            raise RuntimeError(
                f"{method} valid index exceeds label range: max={method_valid_indices.max()}, "
                f"label_len={label_len}"
            )
        if valid_indices is None:
            valid_indices = method_valid_indices
            if not np.array_equal(valid_indices, final_valid_indices):
                raise RuntimeError(f"{method} valid_indices differ from final_recon_score valid_indices")
            labels_aligned = labels[valid_indices]
            aligned_label_sum = int(labels_aligned.sum())
            dropped_points = int(label_len - len(valid_indices))
            print(f"[HDSAC] raw_label_len={label_len}")
            print(f"[HDSAC] score_len={len(score_time)}")
            print(f"[HDSAC] valid_indices_len={len(valid_indices)}")
            print(f"[HDSAC] raw_label_sum={label_sum}")
            print(f"[HDSAC] aligned_label_sum={aligned_label_sum}")
            print(f"[HDSAC] dropped_points={dropped_points}")
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
        hdsac_stats = calibration_stats[method]
        z_hdsac = robust_z(score_time, hdsac_stats["median"], hdsac_stats["scale"], args.eps)
        fused_score = (float(args.fusion_alpha) * z_recon) + ((1.0 - float(args.fusion_alpha)) * z_hdsac)
        fused_name = f"fused_{method}"
        output[f"z_{method}"] = z_hdsac
        output[fused_name] = fused_score.astype(np.float32)
        output[f"{fused_name}_score_len"] = np.asarray(len(fused_score), dtype=np.int64)
        output[f"{method}_score_len"] = np.asarray(len(score_time), dtype=np.int64)
        print(
            f"[HDSAC] {method}: score_len={len(score_time)}, raw_label_len={label_len}, "
            f"fused={fused_name}, alpha={args.fusion_alpha}"
        )

    output["score_len"] = np.asarray(len(output[HDSAC_METHODS[0]]), dtype=np.int64)
    if int(output["score_len"]) != int(output["valid_indices_len"]):
        raise RuntimeError(
            f"score_len != valid_indices_len: {int(output['score_len'])} vs {int(output['valid_indices_len'])}"
        )

    output_dir = os.path.join(args.output_root, args.base_dataset)
    os.makedirs(output_dir, exist_ok=True)
    score_path = os.path.join(output_dir, f"{args.dataset}_{save_id}_scores.npz")
    ensure_outputs_can_be_written([score_path], overwrite=args.overwrite)
    np.savez_compressed(score_path, **output)
    print(f"[HDSAC] saved scores: {score_path}")


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
    parser.add_argument("--output_root", type=str, default="pathB_result_hdsac_debug")
    parser.add_argument("--prototype_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--split", type=int, default=4)
    parser.add_argument("--tau", type=float, default=1.0)
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
        print(f"[HDSAC] multiple saves requested: {save_ids}")
    for save_id in save_ids:
        if args.mode == "build_prototype":
            run_build_prototype(args, save_id, selected_steps)
        else:
            run_score(args, save_id, selected_steps)


if __name__ == "__main__":
    main()
