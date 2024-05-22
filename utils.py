import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from model import FNO2d, LSM_2d, AutoDeepONet, UNO2d, KNO2d, UNet2d
from dataset import *

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

def get_dataset(args):
    dataset_args = args["dataset"]
    if(args["flow_name"] == "tube"):
        train_data = TubeDataset(filename=args['flow_name'] + '_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        val_data = TubeDataset(filename=args['flow_name'] + '_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        test_data = TubeDataset(filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc']
                                )
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = TubeDataset(filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        else:
            test_ms_data = None
    elif args["flow_name"] == "cavity":
        train_data = CavityDataset(
                                filename=args['flow_name'] + '_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        val_data = CavityDataset(
                                filename=args['flow_name'] + '_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        test_data = CavityDataset(
                                filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                )
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = CavityDataset(
                                    filename=args['flow_name'] + '_test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    stable_state_diff = dataset_args['stable_state_diff'],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size']
                                )
        else:
            test_ms_data = None
    elif args["flow_name"] == "NSCH":
        train_data = NSCHDataset(
                                filename='train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        val_data = NSCHDataset(
                                filename='val.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        test_data = NSCHDataset(
                                filename='test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                )
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = NSCHDataset(
                                    filename='test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    stable_state_diff = dataset_args['stable_state_diff'],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size']
                                    )
        else:
            test_ms_data = None
    elif args["flow_name"] == "TGV":
        train_data = TGVDataset(
                                filename='train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        val_data = TGVDataset(
                                filename='val.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        test_data = TGVDataset(
                                filename='test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                )
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = TGVDataset(
                                    filename='test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    stable_state_diff = dataset_args['stable_state_diff'],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size']
                                    )
        else:
            test_ms_data = None
    elif args['flow_name'] == 'Darcy':
        if dataset_args["case_name"] == "PDEBench":
            train_data = PDEDarcyDataset(split="train",
                                         saved_folder= dataset_args["saved_folder"], 
                                         reduced_batch=dataset_args["reduced_batch"],
                                        reduced_resolution=dataset_args["reduced_resolution"])
            val_data = PDEDarcyDataset(split="val", saved_folder= dataset_args["saved_folder"], 
                                         reduced_batch=dataset_args["reduced_batch"],
                                        reduced_resolution=dataset_args["reduced_resolution"])
            test_data = PDEDarcyDataset(split="test", saved_folder= dataset_args["saved_folder"], 
                                         reduced_batch=dataset_args["reduced_batch"],
                                        reduced_resolution=dataset_args["reduced_resolution"])
        else:
            train_data = DarcyDataset(
                                    filename=args['flow_name'] + '_train.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    )
            val_data = DarcyDataset(
                                    filename=args['flow_name'] + '_dev.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    )
            test_data = DarcyDataset(
                                    filename=args['flow_name'] + '_test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    )
        test_ms_data = None
    else:
        raise ValueError("Invalid flow name.")
    print("#Train data: ", len(train_data))
    print("#Validation data: ", len(val_data))
    print("#Test data: ", len(test_data))
    return train_data, val_data, test_data, test_ms_data

def get_dataloader(train_data, val_data, test_data, test_ms_data, args):
    dataloader_args = args["dataloader"]
    if dataloader_args['num_workers'] > 0:
        train_loader = DataLoader(train_data, shuffle=True, multiprocessing_context = 'spawn', generator=torch.Generator(device = 'cpu'), 
                                batch_size=dataloader_args['train_batch_size'], 
                                num_workers= dataloader_args['num_workers'], pin_memory=dataloader_args['pin_memory'])
        val_loader = DataLoader(val_data, shuffle=False, multiprocessing_context = 'spawn', generator=torch.Generator(device = 'cpu'), 
                                batch_size=dataloader_args['val_batch_size'],
                                num_workers= dataloader_args['num_workers'], pin_memory=dataloader_args['pin_memory'])
    else:
        train_loader = DataLoader(train_data, shuffle=True,
                                batch_size=dataloader_args['train_batch_size'],
                                num_workers= 0, pin_memory=dataloader_args['pin_memory'])
        val_loader = DataLoader(val_data, shuffle=False,
                                batch_size=dataloader_args['val_batch_size'],
                                num_workers= 0, pin_memory=dataloader_args['pin_memory'])
    
    test_loader = DataLoader(test_data, shuffle=False, drop_last=True,
                            batch_size=dataloader_args['test_batch_size'],
                            num_workers= 0, pin_memory=dataloader_args['pin_memory'])
    if test_ms_data is not None:
        test_ms_loader = DataLoader(test_ms_data, shuffle=False, drop_last=True,
                                batch_size=dataloader_args['test_batch_size'],
                                num_workers=0, pin_memory=dataloader_args['pin_memory'])
    else:
        test_ms_loader = None
    
    return train_loader, val_loader, test_loader, test_ms_loader

def get_model(spatial_dim, n_case_params, args):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."
    model_name = args["model_name"]
    model_args = args["model"]
    if args['flow_name'] in ["Darcy"]:   # time irrelevant
        if spatial_dim == 2:
            if model_name == "FNO": 
                model = FNO2d(inputs_channel=model_args['inputs_channel'],
                                outputs_channel=model_args['outputs_channel'],
                        width = model_args['width'],
                        modes1 = model_args['modes'],
                        modes2 = model_args['modes'],
                        n_case_params = n_case_params)
            elif model_name == "LSM":
                model = LSM_2d(inputs_channel=model_args['inputs_channel'],
                        outputs_channel=model_args['outputs_channel'],
                        d_model = model_args['width'],
                        num_token=model_args['num_token'], 
                        num_basis=model_args['num_basis'], 
                        patch_size=model_args['patch_size'],
                        padding=model_args['padding'],
                        n_case_params = n_case_params)
            elif model_name == "AutoDeepOnet":
                model = AutoDeepONet(
                        branch_dim=model_args['branch_dim'],
                        trunk_dim =2, # (x,y)
                        inputs_channel=model_args['inputs_channel'],
                        out_channel=model_args['outputs_channel'],
                        width=model_args["deeponet_width"],
                        trunk_depth=model_args["trunk_depth"],
                        branch_depth=model_args["branch_depth"],
                        act_name=model_args["act_fn"],
                        )
            elif model_name == 'UNO':
                model = UNO2d(in_channels=model_args["in_channels"],
                          out_channels = model_args["out_channels"],
                          width=model_args["width"],
                          n_case_params = n_case_params)
            elif model_name == 'UNet':
                model = UNet2d(in_channels=model_args['in_channels'],
                        out_channels=model_args['out_channels'],
                        init_features=model_args['init_features'],
                        n_case_params = n_case_params)
        else:
            #TODO
            pass
    else:
        if spatial_dim == 1:
            pass
        elif spatial_dim == 2:
            if model_name == "FNO":
                model = FNO2d(inputs_channel=model_args['inputs_channel'],
                              outputs_channel=model_args['outputs_channel'],
                      width = model_args['width'],
                      modes1 = model_args['modes'],
                      modes2 = model_args['modes'],
                      n_case_params = n_case_params)
            elif model_name == "LSM":
                model = LSM_2d(inputs_channel=model_args['inputs_channel'],
                        outputs_channel=model_args['outputs_channel'],
                      d_model = model_args['width'],
                      num_token=model_args['num_token'], 
                      num_basis=model_args['num_basis'], 
                      patch_size=model_args['patch_size'],
                      padding=model_args['padding'],
                      n_case_params = n_case_params)
            elif model_name == 'UNO':
                model = UNO2d(in_channels=model_args["in_channels"],
                          out_channels = model_args["out_channels"],
                          width=model_args["width"],
                          n_case_params = n_case_params)
            elif model_name == 'KNO':
                model = KNO2d(n_case_params=n_case_params, **model_args)
            elif model_name == 'UNet':
                model = UNet2d(in_channels=model_args['in_channels'],
                        out_channels=model_args['out_channels'],
                        init_features=model_args['init_features'],
                        n_case_params = n_case_params)

        elif spatial_dim == 3:
            pass
    return model