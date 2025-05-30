import json
import torch
import argparse
import numpy as np
import open3d as o3d
from munch import munchify

from data.kitti_data import KittiDataset
from data.nuscenes_data import NuscenesDataset
from engine.trainer import EpochBasedTrainer

from models.models.cast import CAST


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='kitti', choices=['kitti', 'nuscenes'])
parser.add_argument("--split", default='train', choices=['train', 'val', 'test'])
parser.add_argument("--mode", default='corr', choices=['corr', 'reg'])
parser.add_argument("--load_pretrained", default='cast-epoch-40', type=str)
#parser.add_argument("--load_pretrained", default='cast-epoch-04-24000', type=str)
parser.add_argument("--id", default=208, type=int)  # 355 01 begin

_args = parser.parse_args()


class Engine(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.dataset == 'kitti':
            self.train_dataset = KittiDataset(cfg.data.root, 'train', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list)
            self.val_dataset = KittiDataset(cfg.data.root, 'val', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list)
            self.test_dataset = KittiDataset(cfg.data.root, 'test', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list)
        elif cfg.dataset == 'nuscenes':
            self.train_dataset = NuscenesDataset(cfg.data.root, 'train', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list)
            self.val_dataset = NuscenesDataset(cfg.data.root, 'val', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list)
            self.test_dataset = NuscenesDataset(cfg.data.root, 'test', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list)
        
        self.model = CAST(cfg.model).cuda()



if __name__ == "__main__":
    with open('./config/' + _args.dataset + '.json', 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
    
    tester = Engine(args)
    tester.set_eval_mode()
    tester.load_snapshot(_args.load_pretrained)

    if _args.split == 'train':
        data = tester.train_dataset[_args.id]
    elif _args.split == 'val':
        data = tester.val_dataset[_args.id]
    else:
        data = tester.test_dataset[_args.id]
    gt_trans = data[2].numpy()

    custom_yellow = np.asarray([[221., 184., 34.]]) / 255.0
    custom_blue = np.asarray([[9., 151., 247.]]) / 255.0
    custom_green = np.asarray([[17., 238., 194.]]) / 255.0
    custom_red = np.asarray([[204., 51., 51.]]) / 255.0

    ref_cloud = o3d.geometry.PointCloud()
    ref_cloud.points = o3d.utility.Vector3dVector(data[0].numpy())

    src_cloud = o3d.geometry.PointCloud()
    src_cloud.points = o3d.utility.Vector3dVector(data[1].numpy())
    

    ref_cloud.paint_uniform_color(custom_blue.T)
    src_cloud.paint_uniform_color(custom_yellow.T)
    
    data = [v.cuda().unsqueeze(0) for v in data]
    with torch.no_grad():
        output_dict = tester.model(*data)
        trans = output_dict['refined_transform'].cpu().numpy()
        TE = np.linalg.norm(trans[:3,3] - gt_trans[:3,3])
        RE = (np.trace(gt_trans[:3, :3].T @ trans[:3, :3]) - 1.) * 0.5
        RE = 180.0 / np.pi * np.arccos(np.clip(RE, -1., 1.))
        print(TE, RE)
        
        corres_xyz = output_dict['corres'].cpu().numpy()
        corr_weight = output_dict['corr_confidence'].cpu().numpy()
        corres_xyz = corres_xyz[corr_weight > np.max(corr_weight)* 0.5]  # 0.3

    if _args.mode == 'reg':
        src_cloud.transform(trans)
    elif _args.mode == 'corr':
        lines = list()
        points = np.reshape(corres_xyz + np.array([[0,0,0,0,0,-15.]]), [-1,3])
        lines = np.arange(points.shape[0], dtype=np.int32).reshape([-1,2])
        colors = corres_xyz[:, 3:] @ gt_trans[:3, :3].T + gt_trans[:3, 3:].T - corres_xyz[:, :3]
        colors = np.asarray(np.linalg.norm(colors, axis=-1) < 0.6, dtype=np.float64).reshape([-1, 1])
        colors = colors * custom_green + (1. - colors) * custom_red
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        src_cloud.translate(np.array([[0., 0., -15.]]).T)


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    view_option: o3d.visualization.ViewControl = vis.get_view_control()
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	
    render_option.background_color = np.array([0, 0, 0])
    render_option.background_color = np.array([1, 1, 1])
    render_option.point_size = 3.0
    vis.add_geometry(ref_cloud)
    vis.add_geometry(src_cloud)

    if _args.mode == 'corr':
        vis.add_geometry(line_set)
        if _args.dataset == 'kitti':
            view_option.set_front([-0.1, -1., 0.7])
            view_option.set_up([0., 0., 1.])
        else:
            view_option.set_front([-1., -0.1, 0.6])
            view_option.set_up([1., 0., 0.])
    
    view_option.set_zoom(0.4)
    vis.run()
