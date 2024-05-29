import argparse
import h5py
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from timeit import default_timer

import metrics
from utils import *

METRICS = ['MSE', 'RMSE', 'L2RE', 'MaxError', 'NMSE', 'MAE']

def test_loop(test_loader, model, args, metric_names=METRICS, test_type="accumulate"):
    assert test_type in ["frame", "accumulate"]
    res_dict = {}
    for name in metric_names:
        res_dict[name] = []
    cost_time = []

    model.eval()
    prev_case_id = -1
    preds = torch.Tensor().to(device)
    targets = torch.Tensor().to(device)

    t1 = default_timer()
    for x, y, mask, case_params, grid, case_id in test_loader:
        if prev_case_id != case_id:
            if prev_case_id != -1: # compute metric here
                t2 = default_timer()
                cost_time.append(t2-t1)
                for name in metric_names:
                    metric_fn = getattr(metrics, name)
                    res_dict[name].append(metric_fn(preds, targets))
                t1 = default_timer()
            preds = torch.Tensor().to(device)
            targets = torch.Tensor().to(device)

        x = x.to(device) if test_type == "frame" or prev_case_id != case_id else preds[:, -1]
        y = y.to(device) # y: target tensor (The latter time step) [b, x1, ..., xd, v]
        grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
        mask = mask.to(device) # mask [b, x1, ..., xd, 1]
        case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]
        y = y * mask
        
        with torch.no_grad():
            pred = model(x, case_params, mask, grid) # [bs, h, w, c] (mpnn: [bs, h, w, 1])

        # update
        preds = torch.cat([preds, pred.unsqueeze(1)], dim=1) # [bs, t, h, w, c] (mpnn: [bs, t, h, w, 1])
        if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
            y = y[..., args["model"]["var_id"]].unsqueeze(-1) # [bs, t, h, w, 1]
        targets = torch.cat([targets, y.unsqueeze(1)], dim=1) # [bs, t, h, w, c] (mpnn: [bs, t, h, w, 1])

        prev_case_id = case_id
    
    t2 = default_timer()
    cost_time.append(t2-t1)

    # compute metrics for last case
    for name in metric_names:
        metric_fn = getattr(metrics, name)
        res_dict[name].append(metric_fn(preds, targets))
    
    # post process
    for name in metric_names:
        res_list = res_dict[name]
        if name == "MaxError":
            res = torch.stack(res_list, dim=0)
            res, _ = torch.max(res, dim=0)
        else:
            res = torch.cat(res_list, dim=0)
            res = torch.mean(res, dim=0)
        res_dict[name] = res

    print(f"Total test time: {sum(cost_time):.4f}s, average {sum(cost_time) / len(cost_time):.4f}s per case.")

    return res_dict


def main(args):
    assert not args["if_training"]
    setup_seed(args["seed"])

    # get test data
    _, _, test_data = get_dataset(args)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    # load model from checkpoint
    checkpoint = torch.load(args["model_path"])
    model = get_model(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of model parameters to train:", total_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Load model from {args['model_path']}")
    print(f"Best epoch: {checkpoint['epoch']}")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # test
    print("Start testing.")
    res_dict = test_loop(test_loader, model, args, test_type=cmd_args.test_type)
    for k in res_dict:
        print(f"{k}: {res_dict[k]}")

    # save results
    if not cmd_args.save_result:
        return

    if not os.path.exists(cmd_args.output_dir):
        os.makedirs(cmd_args.output_dir)

    result_file_path = os.path.join(cmd_args.output_dir, f"{args['flow_name']}.hdf5")
    print("save result to", result_file_path)
    with h5py.File(result_file_path, "a") as f:
        # create group
        group_name = os.path.join(args["dataset"]["case_name"], args["model_name"])
        if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
            if args['flow_name'] == "NSCH":
                if args["model"]["var_id"] == 0:
                    group_name = os.path.join(group_name, "f")
                elif args["model"]["var_id"] == 1:
                    group_name =  os.path.join(group_name, "u")
                elif args["model"]["var_id"] == 2:
                    group_name =  os.path.join(group_name, "v")
                else:
                    raise NotImplementedError
            elif args['flow_name'] == "tube":
                if args["model"]["var_id"] == 0:
                    group_name = os.path.join(group_name, "u")
                elif args["model"]["var_id"] == 1:
                    group_name =  os.path.join(group_name, "v")
                else:
                    raise NotImplementedError
            elif args['flow_name'] == "hills":
                if args["model"]["var_id"] == 0:
                    group_name = os.path.join(group_name, "u")
                elif args["model"]["var_id"] == 1:
                    group_name =  os.path.join(group_name, "v")
                elif args["model"]["var_id"] == 2:
                    group_name =  os.path.join(group_name, "w")
                elif args["model"]["var_id"] == 3:
                    group_name =  os.path.join(group_name, "p")
                else:
                    raise NotImplementedError
            else:
                assert args['flow_name'] in ["cavity", "TGV", "cylinder"]
                if args["model"]["var_id"] == 0:
                    group_name = os.path.join(group_name, "u")
                elif args["model"]["var_id"] == 1:
                    group_name =  os.path.join(group_name, "v")
                elif args["model"]["var_id"] == 2:
                    group_name =  os.path.join(group_name, "p")
                else:
                    raise NotImplementedError
        # write result
        try:
            group = f[group_name]
        except:
            group = f.create_group(group_name)

        for k in res_dict:
            if k in group.keys():
                group.__delitem__(k)
            group.create_dataset(k, data=res_dict[k].cpu().numpy())


if __name__ == "__main__":
    # specific device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # can be accessed globally

    # parse args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file.")
    parser.add_argument("--output_dir", type=str, default="result", help="Path to save test results. (default: result)")
    parser.add_argument("-c", "--case_name", type=str, help="Case name.")
    parser.add_argument("--model_path", type=str, help="Checkpoint path to test.")
    parser.add_argument("--test_type", type=str, default="accumulate", help="Checkpoint path to test.")
    parser.add_argument("--save_result", action="store_true", help="Save result if declared.")
    cmd_args = parser.parse_args() # can be accessed globally

    # read default args from config file
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)

    # update args using command args
    if cmd_args.model_path:
        args["model_path"] = cmd_args.model_path
    if cmd_args.case_name:
        args["dataset"]["case_name"] = cmd_args.case_name
    print(args)
    
    main(args)