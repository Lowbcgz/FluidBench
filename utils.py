import numpy as np
import torch
import random

from dataset import TubeDataset, NSCHDataset, PDEDarcyDataset, CavityDataset, TGVDataset, IRCylinderDataset, IRHillsDataset
from model import MPNN, GNOT, MPNNIrregular

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def get_dataset(args):
    dataset_args = args["dataset"].copy()
    if args["flow_name"] == "tube":
        train_dataset = TubeDataset(filename="tube_train.hdf5", **dataset_args)
        val_dataset = TubeDataset(filename="tube_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = TubeDataset(filename="tube_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = TubeDataset(filename="tube_test.hdf5", **dataset_args)
    elif args["flow_name"] == "NSCH":
        train_dataset = NSCHDataset(filename="train.hdf5", **dataset_args)
        val_dataset = NSCHDataset(filename="val.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = NSCHDataset(filename="test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = NSCHDataset(filename="test.hdf5", **dataset_args)
    elif args["flow_name"] == "Darcy":
        if dataset_args["case_name"] == "PDEBench":
            dataset_args.pop("case_name")
            train_dataset = PDEDarcyDataset(filename="2D_DarcyFlow_beta1.0_Train.hdf5", split="train", **dataset_args)
            val_dataset = PDEDarcyDataset(filename="2D_DarcyFlow_beta1.0_Train.hdf5", split="val", **dataset_args)
            test_dataset = PDEDarcyDataset(filename="2D_DarcyFlow_beta1.0_Train.hdf5", split="test", **dataset_args)
        elif dataset_args["case_name"] == "darcy":
            dataset_args.pop("case_name")
            train_dataset = PDEDarcyDataset(filename="darcy.hdf5", split="train", **dataset_args)
            val_dataset = PDEDarcyDataset(filename="darcy.hdf5", split="val", **dataset_args)
            test_dataset = PDEDarcyDataset(filename="darcy.hdf5", split="test", **dataset_args)
        else:
            raise NotImplementedError
    elif args["flow_name"] == "cavity":
        train_dataset = CavityDataset(filename="cavity_train.hdf5", **dataset_args)
        val_dataset = CavityDataset(filename="cavity_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = CavityDataset(filename="cavity_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = CavityDataset(filename="cavity_test.hdf5", **dataset_args)
    elif args["flow_name"] == "TGV":
        train_dataset = TGVDataset(filename="train.hdf5", **dataset_args)
        val_dataset = TGVDataset(filename="val.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = TGVDataset(filename="test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = TGVDataset(filename="cavity_test.hdf5", **dataset_args)
    elif args["flow_name"] == "cylinder":
        train_dataset = IRCylinderDataset(filename="cylinder_train.hdf5", **dataset_args)
        val_dataset = IRCylinderDataset(filename="cylinder_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = IRCylinderDataset(filename="cylinder_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = IRCylinderDataset(filename="cylinder_test.hdf5", **dataset_args)
    elif args["flow_name"] == "hills":
        train_dataset = IRHillsDataset(filename="hills_train.hdf5", **dataset_args)
        val_dataset = IRHillsDataset(filename="hills_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = IRHillsDataset(filename="hills_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = IRHillsDataset(filename="hills_test.hdf5", **dataset_args)
    else:
        raise NotImplementedError
    return train_dataset, val_dataset, test_dataset


def get_model(args):
    if args["model_name"] == "mpnn":
        model = MPNN(**args["model"])
    elif args["model_name"] == "mpnn_irregular":
        model = MPNNIrregular(**args["model"])
    elif args["model_name"] == "gnot":
        model = GNOT(**args["model"])
    else:
        raise NotImplementedError
    return model


def get_model_name(args):
    suffix = (f"_lr{args['optimizer']['lr']}" + 
              f"_bs{args['dataloader']['batch_size']}" + 
              f"_wd{args['optimizer']['weight_decay']}" +
              f"_ep{args['epochs']}")
    if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
        model_name = (f"{args['model_name']}" +
                      f"_k{args['model']['neighbors']}" +
                      f"_layer{args['model']['hidden_layers']}" +
                      f"_dim{args['model']['hidden_features']}" +
                      f"_v{args['model']['var_id']}")
        return model_name + suffix
    elif args["model_name"] == "gnot":
        model_name = (f"{args['model_name']}" +
                      f"_layer{args['model']['n_layers']}" +
                      f"_dim{args['model']['n_hidden']}" +
                      f"_head{args['model']['n_head']}")
        return model_name + suffix
    else:
        raise NotImplementedError