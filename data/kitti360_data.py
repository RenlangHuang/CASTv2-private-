import glob
import torch
import numpy as np
import open3d as o3d
from enum import Enum
from tqdm import tqdm
import MinkowskiEngine as ME
from torch.utils.data import Dataset


class Kitti360Dataset(Dataset):
    loop_sequences = [0, 2, 4, 5, 6, 9]
    reloc_sequences = [0, 2, 3, 4, 5, 6, 7, 9, 10]
    class benchmark(Enum): reloc = 0; loop = 1

    def __init__(self, root, data_list:"Kitti360Dataset.benchmark", voxel_size=0.3, npoints=30000, refine_iters=200):
        super(Kitti360Dataset, self).__init__()
        self.root = root
        self.npoints = npoints
        self.voxel_size = voxel_size
        self.refine_iters = refine_iters
        self.dataset = self.make_dataset(data_list)
    
    def make_dataset(self, data_list:"Kitti360Dataset.benchmark"):
        _indices, _poses = list(), list()
        if data_list == Kitti360Dataset.benchmark.reloc:
            for s in Kitti360Dataset.reloc_sequences:
                try:
                    data = np.genfromtxt("./data/kitti360_list/%04d_reloc.txt"%s)
                    indices, poses = data[:,:2].astype(np.int64), data[:,2:].reshape([-1,3,4])
                    pad = np.array([[[0.,0.,0.,1.]]]).repeat(poses.shape[0], axis=0)
                    poses = np.concatenate([poses, pad], axis=1)
                except IOError:
                    indices, poses = self.generate_relocalization_pairs(s)
                    print("Generate %d pairs for sequence %04d"%(indices.shape[0], s))
                    file = open('./data/kitti360_list/%04d_reloc.txt'%s, 'w')
                    for pair, pose in zip(indices, poses):
                        pose = pose[:3].reshape([-1])
                        item = '%d %d '%(pair[0], pair[1])
                        item = item + ' '.join(str(k) for k in pose) + '\n'
                        file.write(item)
                    file.close()
                
                indices = np.concatenate([np.ones_like(indices[:,:1])*s, indices], axis=1)
                _indices.append(indices); _poses.append(poses)
        
        elif data_list == Kitti360Dataset.benchmark.loop:
            for s in Kitti360Dataset.loop_sequences:
                try:
                    data = np.genfromtxt("./data/kitti360_list/%04d_loop.txt"%s)
                    indices, poses = data[:,:2].astype(np.int64), data[:,2:].reshape([-1,3,4])
                    pad = np.array([[[0.,0.,0.,1.]]]).repeat(poses.shape[0], axis=0)
                    poses = np.concatenate([poses, pad], axis=1)
                except IOError:
                    indices, poses = self.generate_loop_closing_pairs(s)
                    print("Generate %d pairs for sequence %04d"%(indices.shape[0], s))
                    file = open('./data/kitti360_list/%04d_loop.txt'%s, 'w')
                    for pair, pose in zip(indices, poses):
                        pose = pose[:3].reshape([-1])
                        item = '%d %d '%(pair[0], pair[1])
                        item = item + ' '.join(str(k) for k in pose) + '\n'
                        file.write(item)
                    file.close()
                
                indices = np.concatenate([np.ones_like(indices[:,:1])*s, indices], axis=1)
                _indices.append(indices); _poses.append(poses)
        
        _poses = np.concatenate(_poses, axis=0).astype(np.float32)
        _indices = np.concatenate(_indices, axis=0).astype(np.int64)
        return {"indices": _indices, "poses": _poses}
    

    def load_gt_pose(self, seq):
        cam0_to_velo = np.genfromtxt(self.root + "/calibration/calib_cam_to_velo.txt").reshape([3,4])
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])
        data_path = self.root + "/data_poses/2013_05_28_drive_%04d_sync/cam0_to_world.txt"%seq
        poses = np.genfromtxt(data_path, dtype=np.float64)
        indices, poses = poses[:,0].astype(np.int64), poses[:,1:].reshape([-1, 4, 4])
        poses = poses @ np.expand_dims(np.linalg.inv(cam0_to_velo), axis=0)
        
        fnames = glob.glob(self.root + "/data_3d_raw/2013_05_28_drive_%04d_sync/velodyne_points/data/*.bin"%seq)
        fnames = {int(f.split('/')[-1][:-4]) for f in fnames}
        indices = np.array([(i,j) for i, j in enumerate(indices) if j in fnames])
        indices, poses = indices[:, 1], poses[indices[:, 0]]
        return poses, indices
    
    def _make_point_cloud(self, seq, index):
        fname = self.root + "/data_3d_raw/2013_05_28_drive_%04d_sync/velodyne_points/data/%010d.bin"%(seq, index)
        pts = np.fromfile(fname, dtype=np.float32, count=-1).reshape([-1,4])[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd.voxel_down_sample(voxel_size=0.05)
    
    def generate_relocalization_pairs(self, seq, distance=10.):
        poses, indices = self.load_gt_pose(seq)
        pdist = torch.from_numpy(poses)
        pdist = torch.cdist(pdist[:,:3,3], pdist[:,:3,3])
        more_than_th = pdist.gt(distance).numpy()
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
                # check if the groundtruth transformation is reliable
                information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    src_pcd, ref_pcd, 0.2, transformation
                )
                num_src_pts = np.asarray(src_pcd.points).shape[0]
                num_ref_pts = np.asarray(ref_pcd.points).shape[0]
                if information[5,5] / min(num_src_pts, num_ref_pts) > 0.3:
                    if self.refine_iters > 0: # refine the transformation matrix via ICP
                        transformation = o3d.pipelines.registration.registration_icp(
                            src_pcd, ref_pcd, 0.2, transformation, 
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.refine_iters))
                        rela_pose.append(np.array(transformation.transformation))
                    else: rela_pose.append(transformation)
                    pairs.append((curr_time, next_time))
                pbar.update(next_time - curr_time + 1)
                curr_time = next_time + 1
        
        pairs = np.array(pairs)
        indices = np.array(indices)
        indices = np.stack([indices[pairs[:,0]], indices[pairs[:,1]]], axis=1)
        rela_pose = np.array(rela_pose)
        return indices, rela_pose

    def generate_loop_closing_pairs(self, seq, distance=10.):
        poses, indices = self.load_gt_pose(seq)
        tdist = torch.from_numpy(indices)
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
        last_time = -100
        for i in tqdm(range(len(indices))):
            lid = loop_id[i].item()
            if not has_loop[i].item() or indices[i] - last_time < 5: continue
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
        
        pairs = np.array(pairs)
        rela_pose = np.array(rela_pose)
        return pairs, rela_pose
    
    def _read_point_cloud(self, seq, index):
        fname = self.root + "/data_3d_raw/2013_05_28_drive_%04d_sync/velodyne_points/data/%010d.bin"%(seq, index)
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
        scan = scan[np.logical_and(dist > 3., scan[:, 2] > -3.5)]
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


def demo(seq, ref_id, src_id, pose):
    root = "/media/jacko/My Passport/KITTI-360/"
    src_fn = root + "/data_3d_raw/2013_05_28_drive_%04d_sync/velodyne_points/data/%010d.bin"%(seq,src_id)
    ref_fn = root + "/data_3d_raw/2013_05_28_drive_%04d_sync/velodyne_points/data/%010d.bin"%(seq,ref_id)
    xyz1 = np.fromfile(src_fn, dtype=np.float32, count=-1).reshape([-1,4])[:, :3]
    xyz2 = np.fromfile(ref_fn, dtype=np.float32, count=-1).reshape([-1,4])[:, :3]
    
    ply1 = o3d.geometry.PointCloud()
    ply2 = o3d.geometry.PointCloud()
    ply1.points = o3d.utility.Vector3dVector(xyz1)
    ply2.points = o3d.utility.Vector3dVector(xyz2)
    #ply1 = ply1.voxel_down_sample(voxel_size=0.05)
    #ply2 = ply2.voxel_down_sample(voxel_size=0.05)

    custom_yellow = np.asarray([[221., 184., 34.]]) / 255.0
    custom_blue = np.asarray([[9., 151., 247.]]) / 255.0
    ply1.paint_uniform_color(custom_blue.T)
    ply2.paint_uniform_color(custom_yellow.T)
    ply1.transform(pose)

    ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5,origin=[0,0,0])
    o3d.visualization.draw_geometries([ply1, ply2, ax])


if __name__=="__main__":
    root = "/media/jacko/My Passport/KITTI-360/"
    ds = Kitti360Dataset(root, Kitti360Dataset.benchmark.reloc,refine_iters=200)
    ds = Kitti360Dataset(root, Kitti360Dataset.benchmark.loop,refine_iters=200)
