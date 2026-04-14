import argparse
import torch

import json
import yaml
import os

from main_model import CSDI_Physio
from dataset import get_dataloader
from utils import train,  window_trick_evaluate_middle, reconstruction_window_trick_evaluate_middle, reconstruction_validation_threshold

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
parser.add_argument("--validation_threshold_root", type=str, default="validation_threshold")
parser.add_argument("--validation_threshold_ratio", type=float, default=0.02)
args = parser.parse_args()


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

for iteration in os.listdir("train_result"):

    try:
        os.mkdir(f"window_result/{iteration}")
    except:
        pass

    for subset_name in os.listdir(f"train_result/{iteration}/"):

        data_id = subset_name.split("_unconditional")[0]

        if "unconditional:True" in subset_name:
            unconditional = True
        else:
            unconditional = False

        split = 4
        diffusion_step = int(subset_name.split("diffusion_step:")[-1])

        train_data_path_list = []
        test_data_path_list = []
        label_data_path_list = []


        data_file = f"{data_id}_train.pkl"
        train_data_path_list.append("data/Machine/" + data_file)
        test_data_path_list.append("data/Machine/" + data_file.replace("_train.pkl","_test.pkl"))
        label_data_path_list.append("data/Machine/" + data_file.replace("_train.pkl","_test_label.pkl"))


        # epoch = file.split("-")[0]
        train_data_path = train_data_path_list[0]
        test_data_path = test_data_path_list[0]
        label_data_path = label_data_path_list[0]
        train_loader, valid_loader, train_error_loader_list, test_loader_list = get_dataloader(
            train_data_path,
            test_data_path,
            label_data_path,
            batch_size=24,
            window_split=2,
            split=split
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

        if args.dataset == "SMD":
            feature_dim = 38
        elif args.dataset == "PSM":
            feature_dim = 25
        elif args.dataset == "MSL":
            feature_dim = 55
        elif args.dataset == "SMAP":
            feature_dim = 25
        elif args.dataset == "GCP":
            feature_dim = 19
        elif args.dataset == "SWaT":
            feature_dim = 45

        model = CSDI_Physio(run_config, args.device, target_dim=feature_dim, ratio=args.ratio).to(args.device)

        model.load_state_dict(torch.load(f"{base_folder}/best-model.pth",map_location=args.device))

        print("base folder is ")
        print(base_folder)

        try:
            os.mkdir(f"window_result/{iteration}/{diffusion_step}")
        except:
            pass

        os.makedirs(f"window_result/{iteration}/{diffusion_step}", exist_ok=True)#改过代码
        target_folder = f"window_result/{iteration}/{diffusion_step}/{subset_name}"
        os.makedirs(target_folder, exist_ok=True)
        validation_threshold_folder = f"{args.validation_threshold_root}/{iteration}/{diffusion_step}/{subset_name}"
        os.makedirs(validation_threshold_folder, exist_ok=True)

        for temp_i in range(0,1):
            if task_mode == "reconstruction":
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
                    filename=f"{0}-generated_outputs_nsample1{str(temp_i)}_stop_number_-1_threshold.json",
                    split=split,
                )
            else:
                window_trick_evaluate_middle(model, train_error_loader_list, test_loader_list, nsample=1, scaler=1,
                                  foldername=target_folder,
                                  epoch_number=0, name=str(temp_i),split=split)
