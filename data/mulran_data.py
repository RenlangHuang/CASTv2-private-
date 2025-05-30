import os
import glob
import torch
import numpy as np
import open3d as o3d
from enum import Enum
from tqdm import tqdm
import MinkowskiEngine as ME
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


class MulRanDataset(Dataset):
    sequences = ["DCC01","DCC02","DCC03","KAIST01","KAIST02","KAIST03","Riverside01","Riverside02","Riverside03"]
    class benchmark(Enum): reloc = 0; loop = 1

    def __init__(self, root, data_list:"MulRanDataset.benchmark", voxel_size=0.3, npoints=30000, refine_iters=200):
        super(MulRanDataset, self).__init__()
        self.root = root
        self.npoints = npoints
        self.voxel_size = voxel_size
        self.refine_iters = refine_iters
        self.dataset = self.make_dataset(data_list)
    
    def make_dataset(self, data_list:"MulRanDataset.benchmark"):
        _indices, _poses = list(), list()
        if data_list == MulRanDataset.benchmark.reloc:
            for i, s in enumerate(MulRanDataset.sequences):
                try:
                    data = np.genfromtxt(f"./data/mulran_list/{s}_reloc.txt", dtype=np.float128)
                    indices, poses = data[:,:2].astype(np.int64), data[:,2:].reshape([-1,3,4]).astype(np.float64)
                    pad = np.array([[[0.,0.,0.,1.]]]).repeat(poses.shape[0], axis=0)
                    poses = np.concatenate([poses, pad], axis=1)
                except IOError:
                    indices, poses = self.generate_relocalization_pairs(s)
                    print("Generate %d pairs for sequence %s"%(indices.shape[0], s))
                    file = open(f"./data/mulran_list/{s}_reloc.txt", 'w')
                    for pair, pose in zip(indices, poses):
                        pose = pose[:3].reshape([-1])
                        item = '%ld %ld '%(pair[0], pair[1])
                        item = item + ' '.join(str(k) for k in pose) + '\n'
                        file.write(item)
                    file.close()
                
                indices = np.concatenate([np.ones_like(indices[:,:1])*i, indices], axis=1)
                _indices.append(indices); _poses.append(poses)
        
        elif data_list == MulRanDataset.benchmark.loop:
            for i, s in enumerate(MulRanDataset.sequences):
                try:
                    data = np.genfromtxt(f"./data/mulran_list/{s}_loop.txt", dtype=np.float128)
                    indices, poses = data[:,:2].astype(np.int64), data[:,2:].reshape([-1,3,4]).astype(np.float64)
                    pad = np.array([[[0.,0.,0.,1.]]]).repeat(poses.shape[0], axis=0)
                    poses = np.concatenate([poses, pad], axis=1)
                except IOError:
                    indices, poses = self.generate_loop_closing_pairs(s)
                    print("Generate %d pairs for sequence %s"%(indices.shape[0], s))
                    file = open(f"./data/mulran_list/{s}_loop.txt", 'w')
                    for pair, pose in zip(indices, poses):
                        pose = pose[:3].reshape([-1])
                        item = '%ld %ld '%(pair[0], pair[1])
                        item = item + ' '.join(str(k) for k in pose) + '\n'
                        file.write(item)
                    file.close()
                
                indices = np.concatenate([np.ones_like(indices[:,:1])*i, indices], axis=1)
                _indices.append(indices); _poses.append(poses)
        
        _poses = np.concatenate(_poses, axis=0).astype(np.float32)
        _indices = np.concatenate(_indices, axis=0).astype(np.int64)
        return {"indices": _indices, "poses": _poses}
    
    def load_gt_pose(self, seq):
        timestamp = glob.glob(self.root + "/" + seq + f"/sensor_data/Ouster/*.bin")
        timestamp = np.array([int(fname.split('/')[-1][:-4]) for fname in timestamp]); timestamp.sort()
        poses = np.loadtxt(self.root + "/" + seq + "/global_pose.csv", float, delimiter=',', encoding='utf-8')
        gt_timestamp, poses = poses[:, 0].astype(timestamp.dtype), poses[:, 1:].reshape([-1, 3, 4])
        poses = np.concatenate([poses,np.repeat(np.array([[[0,0,0,1.]]]),poses.shape[0],axis=0)],axis=1)
        timestamp = timestamp[timestamp > gt_timestamp[0]]

        calibLidar2Base = np.eye(4)
        calibLidar2Base[:3,3] = np.array([1.7042, -0.021, 1.8047])
        calibLidar2Base[:3,:3] = Rotation.from_euler('xyz', [0.0001, 0.0003, 179.6654], degrees=True).as_matrix()
        calibLidar2Base = np.expand_dims(calibLidar2Base, axis=0)
        poses = poses @ np.linalg.inv(calibLidar2Base)

        '''timestamp = torch.from_numpy(timestamp)
        gt_timestamp = torch.from_numpy(gt_timestamp)
        diff = (timestamp.unsqueeze(1) - gt_timestamp.unsqueeze(0)).abs()
        diff, index = diff.min(dim=1); valid = diff.lt(1e8)
        timestamp = timestamp[valid].numpy()
        poses = poses[index[valid].numpy()]
        
        '''
        curr_time = 0
        aligned_time = []
        aligned_poses = []
        print(timestamp.shape)
        for t in timestamp:
            while t > gt_timestamp[curr_time]:
                curr_time = curr_time + 1
                if curr_time >= len(gt_timestamp): break
            if curr_time >= len(gt_timestamp): break
            if gt_timestamp[curr_time] - t < 1e8 and t - gt_timestamp[curr_time-1] < 1e8:
                last_rot = Rotation.from_matrix(poses[curr_time-1,:3,:3]).as_rotvec()
                next_rot = Rotation.from_matrix(poses[curr_time,:3,:3]).as_rotvec()
                interp_rot = (t - gt_timestamp[curr_time-1]) / (gt_timestamp[curr_time] - gt_timestamp[curr_time-1])
                interp_trans = (1. - interp_rot) * poses[curr_time-1,:3,3] + interp_rot * poses[curr_time,:3,3]
                interp_rot = Rotation.from_rotvec((1. - interp_rot) * last_rot + interp_rot * next_rot).as_matrix()
                transform = np.concatenate([interp_rot, np.expand_dims(interp_trans, axis=1)], axis=1)
                aligned_poses.append(transform)
                aligned_time.append(t)
        
        poses = np.stack(aligned_poses, axis=0)
        poses = np.concatenate([poses, np.array([[[0.,0.,0.,1.]]]).repeat(poses.shape[0],axis=0)], axis=1)
        timestamp = np.array(aligned_time)
        print(timestamp.shape, poses.shape)
        return poses, timestamp
    
    def _make_point_cloud(self, seq, index):
        fname = self.root + seq + f"/sensor_data/Ouster/{index}.bin"
        pts = np.fromfile(fname, dtype=np.float32, count=-1).reshape([-1,4])[:, :3]
        pts = pts[np.linalg.norm(pts, axis=1)>3.5]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd.voxel_down_sample(voxel_size=0.05)
    
    def generate_relocalization_pairs(self, seq, distance=10.):
        poses, indices = self.load_gt_pose(seq)
        pdist = torch.from_numpy(poses)
        pdist = torch.cdist(pdist[:,:3,3], pdist[:,:3,3])
        more_than_th = (pdist.gt(distance) & pdist.lt(distance * 2.)).numpy()
        indices = indices.tolist()

        pairs = []
        rela_pose = []
        curr_time = 0
        pbar = tqdm(total=len(indices))
        while curr_time < len(indices):
            next_time = np.where(more_than_th[curr_time][curr_time:curr_time + 100])[0]
            if len(next_time) == 0:
                curr_time += 1
                pbar.update(1)
            else:
                next_time = next_time[0] + curr_time
                ref_pcd = self._make_point_cloud(seq, indices[curr_time])
                src_pcd = self._make_point_cloud(seq, indices[next_time])
                transformation = np.linalg.inv(poses[curr_time]) @ poses[next_time]
                if self.refine_iters > 0: # refine the transformation matrix via ICP
                    transformation = o3d.pipelines.registration.registration_icp(
                        src_pcd, ref_pcd, 0.2, transformation, 
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.refine_iters)
                    ).transformation
                # check if the groundtruth transformation is reliable
                information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    src_pcd, ref_pcd, 0.2, transformation
                )
                num_src_pts = np.asarray(src_pcd.points).shape[0]
                num_ref_pts = np.asarray(ref_pcd.points).shape[0]
                if information[5,5] / min(num_src_pts, num_ref_pts) > 0.3:
                    pairs.append((indices[curr_time], indices[next_time]))
                    rela_pose.append(transformation)
                pbar.update(next_time - curr_time + 1)
                curr_time = next_time + 1
                
                '''custom_yellow = np.asarray([[221., 184., 34.]]) / 255.0
                custom_blue = np.asarray([[9., 151., 247.]]) / 255.0
                ref_pcd.paint_uniform_color(custom_blue.T)
                src_pcd.paint_uniform_color(custom_yellow.T)
                src_pcd.transform(transformation)
                o3d.visualization.draw_geometries([ref_pcd, src_pcd])'''
        
        pairs = np.array(pairs, dtype=np.int64)
        rela_pose = np.array(rela_pose)
        return pairs, rela_pose

    def generate_loop_closing_pairs(self, seq, distance=10.):
        poses, indices = self.load_gt_pose(seq)
        tdist = torch.from_numpy(indices).double() / 1e8
        tdist = tdist.unsqueeze(1) - tdist.unsqueeze(0)
        pdist = torch.from_numpy(poses)
        pdist = torch.cdist(pdist[:,:3,3], pdist[:,:3,3])
        pdist.masked_fill_(tdist.lt(100), 1e8)
        has_loop = pdist.lt(distance)
        pdist.masked_fill_(~has_loop, 0.)
        has_loop = has_loop.any(dim=1)
        loop_id = pdist.max(dim=1).indices

        pairs = []
        rela_pose = []
        last_time = 0
        for i in tqdm(range(len(indices))):
            lid = loop_id[i].item()
            if not has_loop[i].item() or indices[i] - last_time < 5e8: continue
            ref_pcd = self._make_point_cloud(seq, indices[lid])
            src_pcd = self._make_point_cloud(seq, indices[i])
            transformation = np.linalg.inv(poses[lid]) @ poses[i]
            if self.refine_iters > 0: # refine the transformation matrix via ICP
                transformation = o3d.pipelines.registration.registration_icp(
                    src_pcd, ref_pcd, 0.2, transformation, 
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.refine_iters)
                ).transformation
            # check if the groundtruth transformation is reliable
            information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                src_pcd, ref_pcd, 0.2, transformation
            )
            num_src_pts = np.asarray(src_pcd.points).shape[0]
            num_ref_pts = np.asarray(ref_pcd.points).shape[0]
            if information[5,5] / min(num_src_pts, num_ref_pts) > 0.3:
                rela_pose.append(transformation)
                pairs.append((indices[lid], indices[i]))
                last_time = indices[i]

                '''custom_yellow = np.asarray([[221., 184., 34.]]) / 255.0
                custom_blue = np.asarray([[9., 151., 247.]]) / 255.0
                ref_pcd.paint_uniform_color(custom_blue.T)
                src_pcd.paint_uniform_color(custom_yellow.T)
                src_pcd.transform(transformation)
                o3d.visualization.draw_geometries([ref_pcd, src_pcd])'''
        
        pairs = np.array(pairs, dtype=np.int64)
        rela_pose = np.array(rela_pose)
        return pairs, rela_pose
    
    def _read_point_cloud(self, seq, index):
        fname = self.root + f"/{MulRanDataset.sequences[seq]}/sensor_data/Ouster/{index}.bin"
        scan = np.fromfile(fname, dtype=np.float32, count=-1).reshape([-1,4])[:, :3]
        scan = scan[ME.utils.sparse_quantize(scan / self.voxel_size, return_index=True)[1]]
        if self.npoints is None:
            return scan.astype('float32')
        
        dist = np.linalg.norm(scan, ord=2, axis=1)
        N = scan.shape[0]
        if N >= self.npoints:
            sample_idx = np.argsort(dist)[:self.npoints]
            scan = scan[sample_idx, :].astype('float32')
            dist = dist[sample_idx]
        scan = scan[dist > 3.5]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scan)
        pcd = pcd.remove_radius_outlier(nb_points=2, radius=0.9)[0]
        scan = np.asarray(pcd.points, np.float32)
        return scan
    
    def __getitem__(self, index):
        data_dict = self.dataset["indices"][index]
        src_points = self._read_point_cloud(data_dict[0], data_dict[1])
        dst_points = self._read_point_cloud(data_dict[0], data_dict[2])
        Tr = self.dataset["poses"][index]
        
        src_points = torch.from_numpy(src_points)
        dst_points = torch.from_numpy(dst_points)
        Tr = torch.from_numpy(Tr)
        return src_points, dst_points, Tr
    
    def __len__(self):
        return self.dataset["poses"].shape[0]


if __name__=="__main__":
    root = "/media/jacko/My Passport/MulRan/"
    ds = MulRanDataset(root, MulRanDataset.benchmark.reloc)
    ds = MulRanDataset(root, MulRanDataset.benchmark.loop)

