import argparse
import csv
import glob
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ensemble_proper_reconstruction import (
    compute_final_step_residual,
    get_threshold_from_validation,
    resolve_validation_threshold_path,
)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_pkl_paths(args):
    paths = []
    if args.pkl_path:
        paths.append(args.pkl_path)
    if args.pkl_dir:
        paths.extend(glob.glob(os.path.join(args.pkl_dir, "*.pk")))
    if args.pkl_glob:
        paths.extend(glob.glob(args.pkl_glob))

    paths = sorted(set(paths))
    if len(paths) == 0:
        raise ValueError("No .pk files found. Pass --pkl_path, --pkl_dir, or --pkl_glob.")
    return paths


def infer_run_id(pkl_path):
    parts = pkl_path.split(os.sep)
    for part in parts:
        if part.startswith("save"):
            return part
    return os.path.basename(os.path.dirname(pkl_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, default="")
    parser.add_argument("--pkl_dir", type=str, default="")
    parser.add_argument("--pkl_glob", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="SMAP")
    parser.add_argument("--meta_path", type=str, default="data/Machine/SMAP_MVE_meta.pkl")
    parser.add_argument("--variant", type=str, default="", help="Example: SMAP_MVE_d01. Optional; inferred from dataset_name when empty.")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--validation_threshold_root", type=str, default="validation_threshold")
    parser.add_argument("--threshold_path", type=str, default="")
    parser.add_argument("--compute_abs", action="store_true", default=True)
    parser.add_argument("--no_compute_abs", dest="compute_abs", action="store_false")
    parser.add_argument("--compute_sum", action="store_true", default=True)
    parser.add_argument("--no_compute_sum", dest="compute_sum", action="store_false")
    args = parser.parse_args()

    meta = load_pkl(args.meta_path)
    chosen_times = meta["chosen_times"]
    chosen_dims = meta["chosen_dims"]
    variant = args.variant or args.dataset_name
    source_start = int(meta.get("source_test_range", (0, 0))[0])

    variant_meta = meta.get(variant, {})
    delta = variant_meta.get("delta", "")
    pkl_paths = collect_pkl_paths(args)

    if args.out:
        out_path = args.out
    else:
        os.makedirs("debug_scores", exist_ok=True)
        suffix = "all_runs" if len(pkl_paths) > 1 else "single_run"
        out_path = f"debug_scores/{variant}_injected_reconstruction_scores_{suffix}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "pkl_path",
            "injection_order",
            "time_index",
            "source_test_index",
            "feature_dim",
            "residual",
            "threshold",
            "prediction",
            "final_step_index",
            "delta",
        ])

        for pkl_path in pkl_paths:
            residual, _label, final_step_index = compute_final_step_residual(
                pkl_path,
                args.dataset_name,
                compute_abs=args.compute_abs,
                compute_sum=args.compute_sum,
                load_label=False,
            )

            threshold = None
            threshold_path = args.threshold_path
            if not threshold_path:
                try:
                    threshold_path = resolve_validation_threshold_path(
                        pkl_path,
                        validation_threshold_root=args.validation_threshold_root,
                    )
                except FileNotFoundError:
                    threshold_path = ""

            if threshold_path:
                threshold, *_ = get_threshold_from_validation(
                    threshold_path,
                    compute_abs=args.compute_abs,
                    compute_sum=args.compute_sum,
                )

            run_id = infer_run_id(pkl_path)
            for order, (time_index, feature_dim) in enumerate(zip(chosen_times, chosen_dims)):
                time_index = int(time_index)
                feature_dim = int(feature_dim)
                source_test_index = source_start + time_index
                score = residual[time_index].item()
                prediction = "" if threshold is None else int(score >= threshold)
                writer.writerow([
                    run_id,
                    pkl_path,
                    order,
                    time_index,
                    source_test_index,
                    feature_dim,
                    score,
                    "" if threshold is None else threshold,
                    prediction,
                    final_step_index,
                    delta,
                ])

    print(f"saved injected reconstruction scores to {out_path}")
    print(f"exported {len(pkl_paths)} pkl file(s)")


if __name__ == "__main__":
    main()
