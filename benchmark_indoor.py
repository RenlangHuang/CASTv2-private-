import os
#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import json
import argparse
import numpy as np
from munch import munchify, Munch
from torch.utils.data import DataLoader

from data.indoor_data import IndoorTestDataset
from engine.trainer import EpochBasedTrainer

from models.models.cast import CAST
from engine.evaluator import Evaluator


class Tester(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        val_dataset = IndoorTestDataset(cfg.data.root, _args.benchmark, cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, True)
        self.val_loader = DataLoader(val_dataset, 1, num_workers=cfg.data.num_workers, shuffle=False, pin_memory=True)
        self.val_dataset = val_dataset
        
        self.model = CAST(cfg.model).cuda()
        self.evaluator = Evaluator(cfg.eval).cuda()
    
    def step(self, data_dict):
        output_dict = self.model(*data_dict[:3])
        output_dict['covariance'] = data_dict[-1][0]
        #est_poses.append(output_dict['refined_transform'].detach().cpu().numpy())
        return self.evaluator(output_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./config/3dmatch.json')
    parser.add_argument("--split", default='test', choices=['train', 'val', 'test'])
    parser.add_argument("--benchmark", default='3DMatch', choices=['3DMatch', '3DLoMatch'])
    parser.add_argument("--load_pretrained", default='cast-epoch-08-7500', type=str)

    _args = parser.parse_args()
    with open(_args.config, 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
    
    tester = Tester(args)
    tester.set_eval_mode()
    results_dict = {}
    
    tester.load_snapshot(_args.load_pretrained)
    result_list = tester.validate_epoch()
    RRs = result_list.meter_dict['RR'].records
    splits = {}
    
    for data_dict, recall in zip(tester.val_dataset.dataset, RRs):
        scene = data_dict['points1'].split('/')[-3]
        if scene not in splits.keys():
            splits[scene] = []
        splits[scene].append(recall)

    print("Registration Recalls:")
    splits = {k:np.array(v).mean() for k,v in splits.items()}
    for k, v in splits.items(): print(k, v)
    print("Average Registration Recall:", np.array([v for v in splits.values()]).mean())
        
    result_dict = result_list.summary()
    result_dict['ARR'] = np.array([v for v in splits.values()]).mean()
    #np.save(f"./data/{_args.benchmark}_cast.npy", np.stack(est_poses, axis=0))
