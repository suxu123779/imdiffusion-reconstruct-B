import os
import pickle
import torch
import csv
from tqdm import tqdm
import argparse
from pathlib import Path
import json

RESULT_HEADER = [
    'point_p',
    'point_r',
    'point_f1',
    'add',
    'validation_topk_ratio',
    'validation_score_threshold',
    'final_step_index',
    'mean_test_last_step_score',
    'mean_validation_last_step_score',
    'validation_topk_count',
    'validation_residual_count',
    'test_prediction_count',
]

def compute_f(prediction,label):
    label = label[:len(prediction)]
    TP = torch.sum(prediction * label)
    TN = torch.sum((1 - prediction) * (1 - label))
    FP = torch.sum(prediction * (1 - label))
    FN = torch.sum((1 - prediction) * label)
    precise = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)

    f = 2 * precise * recall / (precise + recall + 0.00001)
    return  precise, recall, f


def compute_add(prediction, labels):
    labels = labels[:len(prediction)]

    now_anomaly_flag = False  # 当前点是否是异常点
    find_anomaly_flag = False  # 当前是否有找到这段异常点

    latency_list = []
    latency = 0

    for i, label in enumerate(labels):
        if not label:
            if now_anomaly_flag:  # 上一个点是异常点
                latency_list.append(latency)
                now_anomaly_flag = False
                find_anomaly_flag = False
                latency = 0
            else:
                pass
        else:
            now_anomaly_flag = True
            if prediction[i]:
                find_anomaly_flag = True

            if not find_anomaly_flag:
                latency += 1
            else:
                pass

    if latency > 0:
        latency_list.append(latency)

    return latency_list

def merge(pkl_path,data_id,machine_number = "",load_label=True):
    fr = open(pkl_path, "rb")
    result = pickle.load(fr)

    all_gen = result[0]

    all_target = result[1]
    head = result[5]
    head_target = result[6]
    head_middle = result[-1]  # [50,10,38]
    all_gen_middle = result[-2]  # [573, 50, 80, 38]

    head = torch.cat(
        [head]
    )
    head_target = torch.cat(
        [head_target]
    )
    # print(f"shape of head middle is {head_middle.shape}")
    # print(f"shape of all_gen_middle[0, :, 0:15, :] is {all_gen_middle[0, :, 0:15, :].shape}")
    head_middle = torch.cat(
        [head_middle], dim=1
    )

    all_gen = all_gen[:, :, :]
    all_target = all_target[:, : , :]
    all_gen_middle = all_gen_middle[:, :,:, :].permute(1, 0, 2, 3)  # [diffusion step, batch number, window length, feature number]


    diffusion_steps = all_gen_middle.shape[0]
    feature_number = all_gen_middle.shape[-1]

    all_gen = torch.Tensor(all_gen).reshape(-1, feature_number)
    all_target = torch.Tensor(all_target).reshape(-1, feature_number)
    all_gen_middle = all_gen_middle.reshape(diffusion_steps, -1, feature_number)


    head = torch.Tensor(head).squeeze()
    head_target = torch.Tensor(head_target).squeeze()
    head_middle = head_middle.squeeze()

    all_gen = torch.cat([head, all_gen], dim=0)
    all_target = torch.cat([head_target, all_target], dim=0)
    # print(f'shape of head middle is {head_middle.shape}')
    # print(f"shape of all gen is {all_gen_middle.shape}")
    all_gen_middle = torch.cat([head_middle, all_gen_middle], dim=1)

    label = None
    if load_label:
        if data_id == "SMD" or data_id == "GCP":
            print(f"machine number is {machine_number}")
            label = pickle.load(
                open(f"data/Machine/{machine_number}_test_label.pkl", "rb")
            )
            origin_data = pickle.load(
                open(f"data/Machine/{machine_number}_test.pkl", "rb")
            )
        else:
            label = pickle.load(
                open(f"data/Machine/{data_id}_test_label.pkl", "rb")
            )
            origin_data = pickle.load(
                open(f"data/Machine/{data_id}_test.pkl", "rb")
            )

        print(f"check equal is {torch.all(all_target == torch.Tensor(origin_data)[:all_target.shape[0]] * 20)}")
        label = torch.Tensor(label)
    # print(f"all gen shape is {all_gen.shape}")

    # print(f"all gen is middle shape is {all_gen_middle.shape}")
    # print(f"all target shape is {all_target.shape}")
    return all_gen_middle, label, all_target

def compute_average_residual(prediction, all_target):
    residual = torch.sum(
        (prediction - all_target) ** 2, dim=-1
    )
    average_residual = torch.sum(residual) / len(residual)

    return average_residual.item()

def compute_residual(prediction, all_target,compute_abs=True,compute_sum=True):
    print(f"compute abs is {compute_abs} and compute sum is {compute_sum}")
    if compute_sum and compute_abs:
        residual = torch.sum(
            torch.abs(prediction - all_target), dim=-1
        )
    elif compute_sum and not compute_abs:
        residual = torch.sum(
            (prediction - all_target) ** 2, dim=-1
        )
    elif not compute_sum and compute_abs:
        residual, _ = torch.max(
            torch.abs(prediction - all_target), dim=-1
        )
    elif not compute_sum and not compute_abs:
        residual, _ = torch.max(
            (prediction - all_target) ** 2, dim=-1
        )
    return residual


def compute_final_step_residual(pkl_path,data_id,compute_abs=True,compute_sum=True,machine_number="",load_label=True):
    if data_id == "SMD" or data_id == "GCP":
        all_gen_middle, label, all_target = merge(
            pkl_path,
            data_id,
            machine_number=machine_number,
            load_label=load_label,
        )
    else:
        all_gen_middle, label, all_target = merge(
            pkl_path,
            data_id,
            load_label=load_label,
        )

    # RECON_CHANGE: use final diffusion step only
    final_step_index = 0
    final_reconstruction = all_gen_middle[final_step_index]
    residual = compute_residual(final_reconstruction, all_target, compute_abs,compute_sum)
    if len(residual) == 0:
        raise ValueError(f"empty residual for {pkl_path}")
    return residual, label, final_step_index


def resolve_validation_threshold_path(test_pkl_path, validation_threshold_root="validation_threshold"):
    test_path = Path(test_pkl_path)
    candidates = []

    if validation_threshold_root:
        validation_root = Path(validation_threshold_root)
        if validation_root.suffix == ".json":
            candidates.append(validation_root)
        else:
            parts = test_path.parts
            if "window_result" in parts:
                window_result_index = parts.index("window_result")
                relative_path = Path(*parts[window_result_index + 1:])
                candidates.append(validation_root / relative_path.with_name(relative_path.stem + "_threshold.json"))
            candidates.append(validation_root / (test_path.stem + "_threshold.json"))

    candidates.append(test_path.with_name(test_path.stem + "_threshold.json"))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    candidate_text = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "validation threshold file not found. Expected one of:\n"
        f"{candidate_text}\n"
        "Run reconstruction inference first to generate validation_threshold, or pass --validation_threshold_root."
    )


def get_threshold_from_validation(threshold_path,last_step_threshold=0.02,compute_abs=None,compute_sum=None):
    with open(threshold_path, "r") as f:
        threshold_info = json.load(f)

    threshold = threshold_info["threshold"]
    final_step_index = threshold_info.get("final_step_index", 0)
    anomaly_number = threshold_info.get("validation_topk_count", 0)
    validation_residual_count = threshold_info.get("validation_residual_count", 0)
    mean_validation_score = threshold_info.get("mean_validation_last_step_score", 0.0)
    validation_topk_ratio = threshold_info.get("topk_ratio", last_step_threshold)
    if compute_abs is not None and "compute_abs" in threshold_info and threshold_info["compute_abs"] != compute_abs:
        raise ValueError(
            f"validation threshold compute_abs {threshold_info['compute_abs']} != test compute_abs {compute_abs}"
        )
    if compute_sum is not None and "compute_sum" in threshold_info and threshold_info["compute_sum"] != compute_sum:
        raise ValueError(
            f"validation threshold compute_sum {threshold_info['compute_sum']} != test compute_sum {compute_sum}"
        )
    print(
        f"validation threshold file is {threshold_path}, final reverse diffusion step index is t={final_step_index}, "
        f"top-k ratio is {validation_topk_ratio}, validation threshold is {threshold}"
    )
    return threshold, anomaly_number, validation_residual_count, mean_validation_score, final_step_index, validation_topk_ratio


def ensemble_proper_reconstruction(pkl_path, data_id, ensemble_strategy_list = [],last_step_threshold = 0.02,compute_abs=True,compute_sum=True,machine_number="",validation_threshold_path=None,validation_threshold_root="validation_threshold"):
    residual, label, final_step_index = compute_final_step_residual(
        pkl_path,
        data_id,
        compute_abs,
        compute_sum,
        machine_number=machine_number,
        load_label=True,
    )
    if validation_threshold_path is None:
        validation_threshold_path = resolve_validation_threshold_path(
            pkl_path,
            validation_threshold_root=validation_threshold_root,
        )
    threshold, validation_anomaly_number, validation_residual_count, mean_validation_score, validation_final_step_index, validation_topk_ratio = get_threshold_from_validation(
        validation_threshold_path,
        last_step_threshold,
        compute_abs=compute_abs,
        compute_sum=compute_sum,
    )
    if validation_final_step_index != final_step_index:
        raise ValueError(
            f"validation final step index {validation_final_step_index} != test final step index {final_step_index}"
        )

    # RECON_CHANGE: remove segment adjustment / voting
    prediction = (residual >= threshold).float()

    add_list = compute_add(prediction, label)
    if len(add_list) == 0:
        add_value = 0.0
    else:
        add_value = sum(add_list) / len(add_list)

    # RECON_CHANGE: strict point-wise evaluation
    p,r,f = compute_f(prediction,label)
    mean_last_step_score = torch.mean(residual).item()
    prediction_count = torch.sum(prediction).item()
    print(f"final reverse diffusion step index is t={final_step_index}, validation top-k ratio is {validation_topk_ratio}, fixed threshold is {threshold}, test prediction count is {prediction_count}")
    print(f"point-wise f update and its value is {f.item(),p.item(),r.item()}")

    return [p.item(),r.item(),f.item(),add_value, validation_topk_ratio, threshold, final_step_index, mean_last_step_score, mean_validation_score, validation_anomaly_number, validation_residual_count, prediction_count],\
           [1.0], 0.0, \
           [1.0], 0.0


def ensemble(pkl_path, data_id, ensemble_strategy_list = [],last_step_threshold = 0.02,compute_abs=True,compute_sum=True,machine_number="",validation_threshold_path=None,validation_threshold_root="validation_threshold"):
    return ensemble_proper_reconstruction(
        pkl_path,
        data_id,
        ensemble_strategy_list,
        last_step_threshold,
        compute_abs,
        compute_sum,
        machine_number=machine_number,
        validation_threshold_path=validation_threshold_path,
        validation_threshold_root=validation_threshold_root,
    )



def compute_one_strategy(data_id,strategy_name,ensemble_strategy_list,csv_writer,last_step_threshold=0.02,validation_threshold_root="validation_threshold"):
    print(f"reconstruction eval for {data_id} in {strategy_name} ...")
    csv_writer.writerow([data_id,strategy_name])
    # for default setting
    compute_abs = True
    compute_sum = True

    if data_id == "MSL":
        compute_abs = True
        compute_sum = True
    if data_id == "PSM":
        compute_abs = True
        compute_sum = False
    if data_id == "SMAP":
        compute_abs = True
        compute_sum = True
    if data_id == "SWaT":
        compute_abs = True
        compute_sum = True
    if data_id == "CODERED":
        # >>> MOD 2: CODERED 作为单数据集，先用和 SWaT 类似的默认设置（L1 sum）
        compute_abs = True
        compute_sum = True

    if data_id == "SMD" or data_id == "GCP":
        compute_abs = True
        compute_sum = False

    if data_id == "SMD":
        machine_number_list = [f"machine-1-{i}" for i in range(1, 9)]
        machine_number_list += [f"machine-2-{i}" for i in range(1,10)]
        machine_number_list += [f"machine-3-{i}" for i in range(1,12)]
        # machine_number_list = [f"machine-1-5"]
        for machine_number in machine_number_list:

            iter_result_list = []
            pkl_path_list = []
            for save_file in os.listdir(f"window_result/"):
                if "save" not in save_file:
                    continue
                for data_file in os.listdir(f"window_result/{save_file}/50/"):
                    if machine_number +"_" not in data_file or "unconditional" not in data_file:
                        continue
                    base_path = f"window_result/{save_file}/50/{data_file}"
                    # print(base_path)
                    for pkl_path in os.listdir(base_path):
                        if ".pk" in pkl_path:
                            pkl_path_list.append(
                                f"{base_path}/{pkl_path}"
                            )
            print(f'length of pkl path list is {len(pkl_path_list)}')
            # print(pkl_path_list)
            for item in pkl_path_list:
                print(item)
            print(f"now top-k ratio for {machine_number} is {last_step_threshold}")
            for pkl_path in pkl_path_list:
                result, _same_list, _same_std, _same_anomaly_list, _same_anomaly_std = ensemble(pkl_path, data_id,
                                                                                            ensemble_strategy_list,
                                                                                            last_step_threshold,
                                                                                            compute_abs, compute_sum,machine_number=machine_number,
                                                                                            validation_threshold_root=validation_threshold_root)
                result = list(result)
                iter_result_list.append(result)
                # csv_writer.writerow([compute_abs, compute_sum] + result)
                # csv_writer.writerow([])
            iter_result_tensor = torch.Tensor(iter_result_list)
            average = iter_result_tensor.mean(0).tolist()
            f_std = torch.std(iter_result_tensor[:, 0])
            csv_writer.writerow([f"average for {machine_number}"] + average)

    elif data_id == "GCP":
        machine_number_list = [f"service{i}" for i in range(0, 30)]
        for save_file in os.listdir(f"window_result/"):
            if "save" not in save_file:
                continue


            iter_result_list = []
            for machine_number in machine_number_list:
                pkl_path_list = []
                for data_file in os.listdir(f"window_result/{save_file}/50/"):
                    if machine_number + "_" not in data_file or "unconditional" not in data_file:
                        continue
                    base_path = f"window_result/{save_file}/50/{data_file}"
                    # print(base_path)
                    for pkl_path in os.listdir(base_path):
                        if ".pk" in pkl_path:
                            pkl_path_list.append(
                                f"{base_path}/{pkl_path}"
                            )
                print(f'length of pkl path list is {len(pkl_path_list)}')
                print(f"now top-k ratio for {machine_number} is {last_step_threshold}")
                for pkl_path in pkl_path_list:
                    print(f"now machine number is {machine_number}")
                    result, _same_list, _same_std, _same_anomaly_list, _same_anomaly_std = ensemble(pkl_path, data_id,
                                                                                                ensemble_strategy_list,
                                                                                                last_step_threshold,
                                                                                                compute_abs, compute_sum,
                                                                                                machine_number=machine_number,
                                                                                                validation_threshold_root=validation_threshold_root)
                    result = list(result)

                    iter_result_list.append(result)

            iter_result_tensor = torch.Tensor(iter_result_list)
            average = iter_result_tensor.mean(0).tolist()
            csv_writer.writerow([f"average for {save_file}"] + average)


    else:

        iter_result_list = []
        pkl_path_list = []
        for save_file in os.listdir(f"window_result/"):
            if "save" not in save_file:
                continue
            for data_file in os.listdir(f"window_result/{save_file}/50/"):
                if data_id not in data_file or "unconditional" not in data_file:
                    continue
                base_path = f"window_result/{save_file}/50/{data_file}"
                # print(base_path)
                for pkl_path in os.listdir(base_path):
                    if ".pk" in pkl_path:
                        pkl_path_list.append(
                            f"{base_path}/{pkl_path}"
                        )
        print(f'length of pkl path list is {len(pkl_path_list)}')
        for pkl_path in pkl_path_list:
            result, _same_list, _same_std, _same_anomaly_list, _same_anomaly_std = ensemble(pkl_path,data_id,ensemble_strategy_list,last_step_threshold,compute_abs,compute_sum,validation_threshold_root=validation_threshold_root)
            result = list(result)
            iter_result_list.append(result)
            csv_writer.writerow([compute_abs,compute_sum] + result)
            csv_writer.writerow([])
        iter_result_tensor = torch.Tensor(iter_result_list)
        average = iter_result_tensor.mean(0).tolist()
        f_std = torch.std(iter_result_tensor[:,0])
        csv_writer.writerow(['average'])
        csv_writer.writerow(
            RESULT_HEADER
        )
        csv_writer.writerow(average )
        csv_writer.writerow(['std'])
        csv_writer.writerow(
            RESULT_HEADER
        )
        csv_writer.writerow(
            iter_result_tensor.std(0).tolist()
        )


def compute_one_data(data_id,validation_threshold_root="validation_threshold"):
    strategy_dict = {
        "final-step-reconstruction": [],
    }
    os.makedirs("ensemble_residual",exist_ok=True)

    csv_writer = csv.writer(open(f"ensemble_residual/{data_id}.csv","w"))
    for key in strategy_dict.keys():
        strategy_name = key
        compute_one_strategy(data_id,strategy_name,
                             strategy_dict[strategy_name],
                             csv_writer,
                             validation_threshold_root=validation_threshold_root)

if __name__ =="__main__":
    # compute_one_data("PSM")
    # compute_one_data("MSL")
    # compute_one_data("SMD")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="SMD")
    parser.add_argument("--validation_threshold_root", type=str, default="validation_threshold")

    args = parser.parse_args()
    compute_one_data(args.dataset_name,validation_threshold_root=args.validation_threshold_root)
