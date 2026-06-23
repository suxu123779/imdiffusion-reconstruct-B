import argparse
import torch

import json
import yaml
import os

from main_model import CSDI_Physio
from dataset import get_dataloader
from diffpath_1d import (
    build_normal_loader,
    run_diffpath_pathb,
    seed_for_save,
)
from utils import (
    train,
    window_trick_evaluate_middle,
    reconstruction_window_trick_evaluate_middle,
    reconstruction_validation_threshold,
    ensure_outputs_can_be_written,
)

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:3', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=30)
parser.add_argument("--ratio",type=float,default=0.7)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--diffusion_step",type=int,default=50)
parser.add_argument("--machine_number",type=int,default=1)
parser.add_argument("--file",type=str)
parser.add_argument('--dataset',type=str,default="SMD")
parser.add_argument("--model_dataset", type=str, default="")
parser.add_argument("--validation_threshold_root", type=str, default="validation_threshold")
parser.add_argument("--validation_threshold_ratio", type=float, default=0.02)
parser.add_argument("--pathB_output_root", type=str, default="pathB_result")
parser.add_argument("--result_tag", type=str, default="")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--pathB_mid_step", type=int, default=None)
parser.add_argument("--pathB_compare_steps", type=str, default="49,45,40,35,30,25,20,15,10,5,0")
parser.add_argument("--pathB_mode", choices=["self", "proto", "both", "diffpath"], default="self")
parser.add_argument("--pathB_proto_dataset", type=str, default="")
parser.add_argument("--pathB_proto_recompute", action="store_true")
parser.add_argument("--saves", nargs="+", default=None)
parser.add_argument("--diffpath_num_steps", type=int, default=10)
parser.add_argument("--reconstruction_batch_size", type=int, default=24)
parser.add_argument("--diffpath_batch_size", type=int, default=128)
parser.add_argument(
    "--diffpath_kde_bandwidths",
    type=str,
    default="0.05,0.1,0.2,0.5,1.0",
)
parser.add_argument(
    "--diffpath_6d_kde_bandwidths",
    type=str,
    default="0.2,0.5,1.0,2.0,5.0",
)
parser.add_argument(
    "--diffpath_6d_gmm_components",
    type=str,
    default="2,4,8,16",
)
parser.add_argument(
    "--diffpath_6d_gmm_covariance_types",
    type=str,
    default="diag,full",
)
parser.add_argument("--diffpath_disable_6d_kde", action="store_true")
parser.add_argument("--diffpath_recompute_calibrator", action="store_true")
args = parser.parse_args()
result_tag = args.result_tag or args.dataset
model_dataset = args.model_dataset or args.dataset
pathB_proto_dataset = args.pathB_proto_dataset or model_dataset


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)



# 由于是分开进行预测，
machine_number = args.machine_number

train_data_path_list = []
test_data_path_list = []
label_data_path_list = []


try:
    os.mkdir("window_result")
except:
    pass

if not os.path.isdir("train_result"):
    raise FileNotFoundError("train_result directory not found. Put save0/save1/save2 under train_result before inference.")

matched_model_count = 0
available_model_dirs = []
requested_saves = set(args.saves) if args.saves else None
for iteration in sorted(os.listdir("train_result")):
    if requested_saves is not None and iteration not in requested_saves:
        continue

    try:
        os.mkdir(f"window_result/{iteration}")
    except:
        pass

    for subset_name in os.listdir(f"train_result/{iteration}/"):
        available_model_dirs.append(f"{iteration}/{subset_name}")

        data_id = subset_name.split("_unconditional")[0]
        if data_id != model_dataset:
            continue
        matched_model_count += 1

        if "unconditional:True" in subset_name:
            unconditional = True
        else:
            unconditional = False

        split = 4
        diffusion_step = int(subset_name.split("diffusion_step:")[-1])

        train_data_path_list = []
        test_data_path_list = []
        label_data_path_list = []


        data_file = f"{args.dataset}_train.pkl"
        train_data_path_list.append("data/Machine/" + data_file)
        test_data_path_list.append("data/Machine/" + data_file.replace("_train.pkl","_test.pkl"))
        label_data_path_list.append("data/Machine/" + data_file.replace("_train.pkl","_test_label.pkl"))


        # epoch = file.split("-")[0]
        train_data_path = train_data_path_list[0]
        test_data_path = test_data_path_list[0]
        label_data_path = label_data_path_list[0]
        eval_batch_size = (
            args.diffpath_batch_size
            if args.pathB_mode == "diffpath"
            else args.reconstruction_batch_size
        )
        train_loader, valid_loader, train_error_loader_list, test_loader_list = get_dataloader(
            train_data_path,
            test_data_path,
            label_data_path,
            batch_size=eval_batch_size,
            window_split=2,
            split=split
        )
        recon_train_error_loader_list = train_error_loader_list
        if (
            args.pathB_mode == "diffpath"
            and args.reconstruction_batch_size != eval_batch_size
        ):
            (
                _recon_train_loader,
                _recon_valid_loader,
                recon_train_error_loader_list,
                _recon_test_loader_list,
            ) = get_dataloader(
                train_data_path,
                test_data_path,
                label_data_path,
                batch_size=args.reconstruction_batch_size,
                window_split=2,
                split=split,
            )
        pathB_proto_loader = None
        if args.pathB_mode in ("proto", "both"):
            proto_train_data_path = f"data/Machine/{pathB_proto_dataset}_train.pkl"
            proto_test_data_path = f"data/Machine/{pathB_proto_dataset}_test.pkl"
            proto_label_data_path = f"data/Machine/{pathB_proto_dataset}_test_label.pkl"
            missing_proto_files = [
                path
                for path in [proto_train_data_path, proto_test_data_path, proto_label_data_path]
                if not os.path.exists(path)
            ]
            if missing_proto_files:
                raise FileNotFoundError(
                    "pathB proto mode requires normal prototype data files:\n"
                    + "\n".join(missing_proto_files)
                )
            _, pathB_proto_loader, _, _ = get_dataloader(
                proto_train_data_path,
                proto_test_data_path,
                proto_label_data_path,
                batch_size=24,
                window_split=2,
                split=split,
            )
        base_folder = f"train_result/{iteration}/{subset_name}"
        config_path = f"{base_folder}/config.json"
        if os.path.exists(config_path):
            # RECON_CHANGE: 推理需要和训练时的 task_mode 保持一致，因此优先读取训练目录里的 config.json。
            with open(config_path, "r") as f:
                run_config = json.load(f)
        else:
            run_config = json.loads(json.dumps(config))

        run_config["model"]["is_unconditional"] = unconditional
        run_config["model"]["test_missing_ratio"] = args.testmissingratio
        run_config["diffusion"]["num_steps"] = diffusion_step
        run_config["train"]["epochs"] = args.epochs
        print(json.dumps(run_config, indent=4))

        task_mode = run_config["model"].get("task_mode", "imputation")

        if args.dataset == "SMD" or args.dataset.startswith("machine-"):
            feature_dim = 38
        elif args.dataset == "PSM":
            feature_dim = 25
        elif args.dataset == "MSL":
            feature_dim = 55
        elif args.dataset == "SMAP" or args.dataset.startswith("SMAP_MVE"):
            feature_dim = 25
        elif args.dataset == "GCP":
            feature_dim = 19
        elif args.dataset == "SWaT":
            feature_dim = 45
        elif args.dataset == "CODERED":
            feature_dim = 48
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")

        model = CSDI_Physio(run_config, args.device, target_dim=feature_dim, ratio=args.ratio).to(args.device)

        model.load_state_dict(torch.load(f"{base_folder}/best-model.pth",map_location=args.device))

        print("base folder is ")
        print(base_folder)

        try:
            os.mkdir(f"window_result/{iteration}/{diffusion_step}")
        except:
            pass

        os.makedirs(f"window_result/{iteration}/{diffusion_step}", exist_ok=True)#改过代码
        if data_id == args.dataset:
            output_subset_name = subset_name
        else:
            output_subset_name = subset_name.replace(f"{data_id}_", f"{args.dataset}_", 1)

        target_folder = f"window_result/{iteration}/{diffusion_step}/{output_subset_name}"
        os.makedirs(target_folder, exist_ok=True)
        validation_threshold_folder = f"{args.validation_threshold_root}/{iteration}/{diffusion_step}/{output_subset_name}"
        os.makedirs(validation_threshold_folder, exist_ok=True)

        for temp_i in range(0,1):
            if task_mode == "reconstruction":
                if args.pathB_mode == "diffpath":
                    diffpath_seed = seed_for_save(args.seed, iteration)
                    normal_train_path = (
                        f"data/Machine/{pathB_proto_dataset}_train.pkl"
                    )
                    if not os.path.exists(normal_train_path):
                        raise FileNotFoundError(
                            "DiffPath normal training data not found: "
                            f"{normal_train_path}"
                        )
                    normal_loader = build_normal_loader(
                        normal_train_path,
                        batch_size=args.diffpath_batch_size,
                        split=split,
                    )
                    recon_normal_loader = build_normal_loader(
                        normal_train_path,
                        batch_size=args.reconstruction_batch_size,
                        split=split,
                    )
                    run_diffpath_pathb(
                        model,
                        normal_loader,
                        train_error_loader_list,
                        label_data_path,
                        output_root=args.pathB_output_root,
                        result_tag=result_tag,
                        base_dataset=pathB_proto_dataset,
                        save_id=iteration,
                        num_path_steps=args.diffpath_num_steps,
                        split=split,
                        seed=diffpath_seed,
                        bandwidths=args.diffpath_kde_bandwidths,
                        bandwidths_6d=args.diffpath_6d_kde_bandwidths,
                        gmm_components_6d=args.diffpath_6d_gmm_components,
                        gmm_covariance_types_6d=(
                            args.diffpath_6d_gmm_covariance_types
                        ),
                        enable_6d_kde=not args.diffpath_disable_6d_kde,
                        recompute_calibrator=(
                            args.diffpath_recompute_calibrator
                        ),
                        overwrite=args.overwrite,
                        recon_normal_loader=recon_normal_loader,
                        recon_test_loader=recon_train_error_loader_list,
                    )
                    continue

                threshold_filename = f"{0}-generated_outputs_nsample1{str(temp_i)}_stop_number_-1_threshold.json"
                ensure_outputs_can_be_written(
                    [
                        f"{target_folder}/{0}-generated_outputs_nsample1{str(temp_i)}_stop_number_-1.pk",
                        os.path.join(validation_threshold_folder, threshold_filename),
                    ],
                    overwrite=args.overwrite,
                )
                # RECON_CHANGE: reconstruction 推理忽略第二个 strategy loader，仅使用单个 loader 输出完整重构结果。
                reconstruction_window_trick_evaluate_middle(
                    model,
                    train_error_loader_list,
                    nsample=1,
                    scaler=1,
                    foldername=target_folder,
                    epoch_number=0,
                    name=str(temp_i),
                    split=split,
                    pathB_output_root=args.pathB_output_root,
                    result_tag=result_tag,
                    run_id=iteration,
                    overwrite=args.overwrite,
                    pathB_mid_step=args.pathB_mid_step,
                    pathB_compare_steps=args.pathB_compare_steps,
                    pathB_mode=args.pathB_mode,
                    pathB_proto_loader=pathB_proto_loader,
                    pathB_proto_dataset=pathB_proto_dataset,
                    pathB_proto_recompute=args.pathB_proto_recompute,
                )
                compute_abs = True
                compute_sum = True
                if args.dataset == "PSM":
                    compute_sum = False
                if args.dataset == "SMD" or args.dataset == "GCP":
                    compute_sum = False

                # RECON_CHANGE: validation residual chooses threshold without saving a second reconstruction pkl.
                reconstruction_validation_threshold(
                    model,
                    valid_loader,
                    topk_ratio=args.validation_threshold_ratio,
                    compute_abs=compute_abs,
                    compute_sum=compute_sum,
                    nsample=1,
                    foldername=validation_threshold_folder,
                    filename=threshold_filename,
                    split=split,
                )
            else:
                window_trick_evaluate_middle(model, train_error_loader_list, test_loader_list, nsample=1, scaler=1,
                                  foldername=target_folder,
                                  epoch_number=0, name=str(temp_i),split=split)

if matched_model_count == 0:
    available_text = "\n".join(available_model_dirs[:50])
    raise FileNotFoundError(
        f"No train_result model directory matched --model_dataset {model_dataset}.\n"
        "Expected a directory like train_result/save0/"
        f"{model_dataset}_unconditional:..._diffusion_step:50\n"
        "Use --model_dataset only if the checkpoint prefix differs from --dataset.\n"
        f"Available model directories:\n{available_text}"
    )
