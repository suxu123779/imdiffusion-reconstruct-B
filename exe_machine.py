import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Physio
from dataset import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device ')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)

parser.add_argument("--modelfolder", type=str, default="")

parser.add_argument("--ratio",type=float,default=0.7)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--dataset",type=str,default="SMD")
parser.add_argument("--task_mode", type=str, default="reconstruction")
args = parser.parse_args()




train_data_path_list = []
test_data_path_list = []
label_data_path_list = []

if args.dataset == "SMD":
    data_set_number = (
        [f"1-{i}" for i in range(1, 9)] +
        [f"2-{i}" for i in range(1, 10)] +
        [f"3-{i}" for i in range(1, 12)]
    )


    for data_set_id in data_set_number:
            file = f"machine-{data_set_id}_train.pkl"
            train_data_path_list.append("data/Machine/" + file)
            test_data_path_list.append("data/Machine/" + file.replace("_train.pkl","_test.pkl"))
            label_data_path_list.append("data/Machine/" + file.replace("_train.pkl","_test_label.pkl"))
elif args.dataset == "GCP":
    data_set_number = [f"service{i}" for i in range(0,30)]
    for data_set_id in data_set_number:
            file = f"{data_set_id}_train.pkl"
            train_data_path_list.append("data/Machine/" + file)
            test_data_path_list.append("data/Machine/" + file.replace("_train.pkl","_test.pkl"))
            label_data_path_list.append("data/Machine/" + file.replace("_train.pkl","_test_label.pkl"))
else: # for dataset with only one subset
    data_set_number = [args.dataset]
    for data_set_id in data_set_number:
        file = f"{data_set_id}_train.pkl"
        train_data_path_list.append("data/Machine/" + file)
        test_data_path_list.append("data/Machine/" + file.replace("_train.pkl", "_test.pkl"))
        label_data_path_list.append("data/Machine/" + file.replace("_train.pkl", "_test_label.pkl"))

diffusion_step_list = [50]

unconditional_list = [True]

split_list = [10]



try:
    os.mkdir("train_result")
except:
    pass


for training_epoch in range(0,3):
    print(f"begin to train for training_epoch {training_epoch} ...")
    try:
        os.mkdir(f"train_result/save{training_epoch}")
    except:
        pass
    for diffusion_step in diffusion_step_list:
        for unconditional in unconditional_list:
            for split in split_list:


                for i, train_data_path in enumerate(train_data_path_list):
                    path = "config/" + args.config
                    with open(path, "r") as f:
                        config = yaml.safe_load(f)

                    config["model"]["is_unconditional"] = unconditional
                    # RECON_CHANGE: 显式把训练模式写入 config，让模型前向从插补式切换为重构式。
                    config["model"]["task_mode"] = args.task_mode

                    config["diffusion"]["num_steps"] = diffusion_step
                    print(json.dumps(config, indent=4))

                    foldername = f"./train_result/save{training_epoch}/" + f"{train_data_path.replace('_train.pkl', '').replace('data/Machine/', '')}" + "_unconditional:" + str(
                        unconditional) + "_task:" + str(args.task_mode) + "_split:" + str(
                        split) + "_diffusion_step:" + str(diffusion_step) + "/"
                    print('model folder:', foldername)
                    os.makedirs(foldername, exist_ok=True)
                    with open(foldername + "config.json", "w") as f:
                        json.dump(config, f, indent=4)

                    test_data_path = test_data_path_list[i]
                    label_data_path = label_data_path_list[i]

                    train_loader, valid_loader, test_loader1, test_loader2 = get_dataloader(
                        train_data_path,
                        test_data_path,
                        label_data_path,
                        batch_size=12,
                        split=split
                    )
                    print("train path is")
                    print(train_data_path)
                    print(test_data_path)
                    print(label_data_path)

                    if args.dataset == "SMD":
                        feature_dim = 38
                    elif args.dataset == "SMAP" or args.dataset == "PSM":
                        feature_dim = 25
                    elif args.dataset == "MSL":
                        feature_dim = 55
                    elif args.dataset == "SWAT":
                        feature_dim = 51

                    elif args.dataset == "GCP":
                        feature_dim = 19

                    elif args.dataset == "CODERED":
                        feature_dim = 48

                    model = CSDI_Physio(config, args.device,target_dim=feature_dim,ratio = args.ratio).to(args.device)

                    train(
                        model,
                        config["train"],
                        train_loader,
                        valid_loader=valid_loader,
                        foldername=foldername,
                        test_loader1=test_loader1,
                        test_loader2=test_loader2
                    )
