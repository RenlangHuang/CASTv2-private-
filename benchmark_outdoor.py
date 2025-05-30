#import os
#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import json
import argparse
import numpy as np
from munch import munchify, Munch
from torch.utils.data import DataLoader

from data.kitti_data import KittiDataset
from data.mulran_data import MulRanDataset
from data.kitti360_data import Kitti360Dataset
from data.nuscenes_data import NuscenesDataset
from engine.trainer import EpochBasedTrainer

from models.models.cast import CAST
from engine.evaluator import Evaluator


class Tester(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = CAST(cfg.model).cuda()
        self.evaluator = Evaluator(cfg.eval).cuda()
        
        self.model.ransac_nodes = False
        self.model.ransac_filter = 0.6
        self.model.ransac = True
        self.TE_threshold = 0.5
        self.RE_threshold = 5.0

        if cfg.dataset == 'kitti360_reloc':
            self.dataset = Kitti360Dataset(cfg.data.root, Kitti360Dataset.benchmark.reloc, cfg.data.voxel_size, cfg.data.npoints)
        elif cfg.dataset == 'kitti360_loop':
            self.dataset = Kitti360Dataset(cfg.data.root, Kitti360Dataset.benchmark.loop, cfg.data.voxel_size, cfg.data.npoints)
        elif cfg.dataset == 'mulran_reloc':
            self.dataset = MulRanDataset(cfg.data.root, MulRanDataset.benchmark.reloc, cfg.data.voxel_size, cfg.data.npoints)
        elif cfg.dataset == 'mulran_loop':
            self.dataset = MulRanDataset(cfg.data.root, MulRanDataset.benchmark.loop, cfg.data.voxel_size, cfg.data.npoints)
        else:
            raise NotImplementedError
        
        self.benchmark = cfg.dataset
        self.loader = DataLoader(self.dataset, 1, num_workers=cfg.data.num_workers, shuffle=False, pin_memory=True)

        self.load_snapshot(cfg.load_pretrained)
        self.set_eval_mode()
    
    def run(self):
        est_poses = []
        metrics = {"RR": [], "TE": [], "RE": []}
        print("---------Start validation---------")
        for iteration, data_dict in enumerate(self.loader):
            data_dict = self.to_cuda(data_dict)
            result_dict = self.model(*data_dict)
            pose = result_dict['refined_transform']
            rte, rre = self.evaluator.transform_error(data_dict[2][0],pose)
            rte, rre = rte.item(), rre.item()
            rr = rte < self.TE_threshold and rre < self.RE_threshold
            print("[%d/%d]"%(iteration, len(self.loader)), end=' ')
            print(f"RR: {rr}, TE: %.4f, RE: %.4f"%(rte, rre))
            est_poses.append(pose.cpu().numpy())
            metrics['RR'].append(float(rr))
            if rr:
                metrics['RE'].append(rre)
                metrics['TE'].append(rte)
        
        #np.save("./data/results/" + self.benchmark + "_cast.npy", np.stack(est_poses,axis=0))
        rr = np.array(metrics['RR']).mean()

        print("Summary: RR: %.4f"%np.array(metrics['RR']).mean(), end='; ')
        print("TE: %.4f"%np.array(metrics['TE']).mean(), end='; ')
        print("RE: %.4f"%np.array(metrics['RE']).mean())
    
    def evaluate(self, est_poses: np.ndarray):
        poses: np.ndarray = self.dataset.dataset["poses"]
        err_rot = np.transpose(poses[:, :3, :3], (0, 2, 1)) @ est_poses[:, :3, :3]
        rre = (err_rot[:, 0, 0] + err_rot[:, 1, 1] + err_rot[:, 2, 2] - 1.) * 0.5
        rre = 180.0 / np.pi * np.arccos(np.clip(rre, -1., 1.))
        rte = np.linalg.norm(poses[:, :3, 3] - est_poses[:, :3, 3], axis=1)
        return rre, rte


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./config/kitti.json', type=str)
    parser.add_argument("--dataset", choices=['kitti360_reloc','kitti360_loop','mulran_reloc','mulran_loop'])
    parser.add_argument("--load_pretrained", default='cast-epoch-40-best', type=str)
    parser.add_argument("--root", default=None, type=str)
    
    _args = parser.parse_args()
    with open(_args.config, 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
        args.dataset = _args.dataset
        if _args.root is not None:
            args.data.root = _args.root
        args.load_pretrained = _args.load_pretrained
    
    tester = Tester(args)
    tester.run()
