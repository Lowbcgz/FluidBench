import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from timeit import default_timer

import metrics
from utils import *

METRICS = ['MSE', 'RMSE', 'L2RE', 'MaxError', 'NMSE', 'MAE']

def train_loop(train_loader, model, optimizer, loss_fn, args):
    model.train()
    t1 = default_timer()
    train_loss = 0
    train_l_inf = 0
    step = 0
    for x, y, mask, case_params, grid, _ in train_loader:
        # preprocess data
        step += 1
        batch_size = x.shape[0]
        x = x.to(device) # x: input tensor (The previous time step) [b, x1, ..., xd, v]
        y = y.to(device) # y: target tensor (The latter time step) [b, x1, ..., xd, v]
        grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
        mask = mask.to(device) # mask [b, x1, ..., xd, 1]
        case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]
        y = y * mask

        # forward
        assert hasattr(train_loader.dataset, "multi_step_size")
        if train_loader.dataset.multi_step_size > 1:
            preds = []
            total_loss = 0
            for i in range(train_loader.dataset.multi_step_size):
                loss, pred, info = model.one_forward_step(x, case_params, mask[:, i], grid, y[:, i].clone(), loss_fn)
                preds.append(pred)
                total_loss += loss
                x = pred
            preds = torch.stack(preds, dim=1)
        else:
            total_loss, preds, info = model.one_forward_step(x, case_params, mask, grid, y.clone(), loss_fn)

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # record
        if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
            y = y[..., args["model"]["var_id"]].unsqueeze(-1)
        train_loss += total_loss.item()
        train_l_inf = max(train_l_inf, torch.max((torch.abs(preds.reshape(batch_size, -1) - y.reshape(batch_size, -1)))))

    train_loss /= step
    t2 = default_timer()
    return train_loss, train_l_inf, t2 - t1


def val_loop(val_loader, model, loss_fn, args, metric_names=METRICS):
    model.eval()
    val_l2 = 0
    val_l_inf = 0
    step = 0
    res_dict = {}
    for name in metric_names:
        res_dict[name] = []
    
    with torch.no_grad():
        for x, y, mask, case_params, grid, _ in val_loader:
            step += 1
            batch_size = x.shape[0]
            x = x.to(device) # x: input tensor (The previous time step) [b, x1, ..., xd, v]
            y = y.to(device) # y: target tensor (The latter time step) [b, x1, ..., xd, v]
            grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
            mask = mask.to(device) # mask [b, x1, ..., xd, 1]
            case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]
            y = y * mask

            # infer
            assert hasattr(val_loader.dataset,"multi_step_size")
            if val_loader.dataset.multi_step_size > 1:
                preds = []
                for i in range(val_loader.dataset.multi_step_size):
                    pred = model(x, case_params, mask[:, i], grid)
                    preds.append(pred)
                    x = pred
                preds = torch.stack(preds, dim=1)
            else:
                preds = model(x, case_params, mask, grid)

            # compute metric
            if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
                y = y[..., args["model"]["var_id"]].unsqueeze(-1)
            val_l2 += loss_fn(preds.reshape(batch_size, -1), y.reshape(batch_size, -1)).item()
            val_l_inf = max(val_l_inf, torch.max((torch.abs(preds.reshape(batch_size, -1) - y.reshape(batch_size, -1)))))
            for name in metric_names:
                metric_fn = getattr(metrics, name)
                res_dict[name].append(metric_fn(preds, y))

    for name in metric_names:
        res_list = res_dict[name]
        if name == "MaxError":
            res = torch.stack(res_list, dim=0)
            res, _ = torch.max(res, dim=0)
        else:
            res = torch.cat(res_list, dim=0)
            res = torch.mean(res, dim=0)
        res_dict[name] = res
    
    # print metrics
    for k in res_dict:
        print(f"{k}: {res_dict[k]}")

    val_l2 /= step

    return val_l2, val_l_inf


def test_loop(test_loader, model, args, metric_names=METRICS):
    """compute accumulate loss
    """
    res_dict = {}
    for name in metric_names:
        res_dict[name] = []
    model.eval()
    prev_case_id = -1
    preds = torch.Tensor().to(device)
    targets = torch.Tensor().to(device)
    for x, y, mask, case_params, grid, case_id in test_loader:
        if prev_case_id != case_id:
            if prev_case_id != -1: # compute metric here
                for name in metric_names:
                    metric_fn = getattr(metrics, name)
                    res_dict[name].append(metric_fn(preds, targets))
            preds = torch.Tensor().to(device)
            targets = torch.Tensor().to(device)

        x = x.to(device) # x: input tensor (The previous time step) [b, x1, ..., xd, v]
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

    return res_dict


def main(args):
    # init
    setup_seed(args["seed"])
    checkpoint = torch.load(args["model_path"]) if not args["if_training"] or args["continue_training"] else None
    saved_model_name = get_model_name(args)
    saved_dir = os.path.join(args["saved_dir"], args["flow_name"], args["dataset"]["case_name"])
    # check path existence
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    # visualize
    if args["if_training"] and args["tensorboard"]:
        log_path = os.path.join(args["output_dir"], 
                                args["model_name"], 
                                args["flow_name"] + '_' + args['dataset']['case_name'], 
                                saved_model_name)
        writer = SummaryWriter(log_path)

    # data
    train_data, val_data, test_data = get_dataset(args)
    train_loader = DataLoader(train_data, shuffle=True, **args["dataloader"])
    val_loader = DataLoader(val_data, shuffle=False, **args["dataloader"])
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    # model
    model = get_model(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of model parameters to train:", total_params)
    if not args["if_training"]:
        print(f"Test mode, load checkpoint from {args['model_path']}")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Best epoch: {checkpoint['epoch']}")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        print("Start testing")
        # TODO: add a write function
        results = test_loop(test_loader, model, args)
        for k in results:
            print(f"{k}: {results[k]}")
        print("Done")
        return
    if args["continue_training"]:
        print(f"Continue training, load checkpoint from {args['model_path']}")
        model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # optimizer
    optim_args = args["optimizer"]
    optim_name = optim_args.pop("name")
    # if continue training, resume optimizer and scheduler from checkpoint
    if args["continue_training"]:
        optimizer = getattr(torch.optim, optim_name)(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        optimizer = getattr(torch.optim, optim_name)(model.parameters(), **optim_args)
    
    # scheduler
    start_epoch = 0
    min_val_loss = torch.inf
    if args["continue_training"]:
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['loss']
    sched_args = args["scheduler"]
    sched_name = sched_args.pop("name")
    scheduler = getattr(torch.optim.lr_scheduler, sched_name)(optimizer, last_epoch=start_epoch-1, **sched_args)

    # loss function
    loss_fn = nn.MSELoss(reduction="mean")

    # train
    print(f"Start training from epoch {start_epoch}")
    total_time = 0
    for epoch in range(start_epoch, args["epochs"]):
        train_loss, train_l_inf, time = train_loop(train_loader, model, optimizer, loss_fn, args)
        scheduler.step()
        total_time += time
        if args["tensorboard"]:
            writer.add_scalar('Train Loss', train_loss, epoch)
        print(f"[Epoch {epoch}] train_loss: {train_loss}, train_l_inf: {train_l_inf}, time_spend: {time:.3f}s")
        saved_path = os.path.join(saved_dir, saved_model_name)
        model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        torch.save({"epoch": epoch + 1, 
                    "loss": min_val_loss,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict()}, saved_path + "-latest.pt")
        if (epoch + 1) % args["save_period"] == 0 or (epoch + 1) == args["epochs"]:
            print("====================validate====================")
            val_l2_full, val_l_inf = val_loop(val_loader, model, loss_fn, args)
            if args["tensorboard"]:
                writer.add_scalar('Val Loss', val_l2_full, epoch)
            print(f"[Epoch {epoch}] val_l2_full: {val_l2_full} val_l_inf: {val_l_inf}")
            print("================================================")
            if val_l2_full < min_val_loss:
                min_val_loss = val_l2_full
                torch.save({"epoch": epoch + 1, 
                            "loss": min_val_loss,
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optimizer.state_dict()
                            }, saved_path + "-best.pt")
    print(f"Total train time: {total_time:.4f}s. Train one epoch every {total_time / (args['epochs'] - start_epoch):.4f}s on average.")
    print("Done.")


if __name__ == "__main__":
    # specific device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # can be accessed globally

    # parse args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file.")
    parser.add_argument("-c", "--case_name", type=str, help="Case name.")
    parser.add_argument("--lr", type=float, help="learning rate.")
    parser.add_argument("-bs", "--batch_size", type=int, help="Batch size.")
    parser.add_argument("-wd", "--weight_decay", type=float, help="Weight decay.")
    parser.add_argument("--epochs", type=int, help="The number of training epochs.")
    cmd_args = parser.parse_args()

    # read default args from config file
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)

    # update args using command args
    if cmd_args.case_name:
        args["dataset"]["case_name"] = cmd_args.case_name
    if cmd_args.epochs:
        args["epochs"] = cmd_args.epochs
    if cmd_args.batch_size:
        args["dataloader"]["batch_size"] = cmd_args.batch_size
    for k in args["optimizer"]:
        if hasattr(cmd_args, k) and getattr(cmd_args, k) is not None:
            args["optimizer"][k] = getattr(cmd_args, k)
    print(args)
    
    main(args)