import numpy as np
import open3d as o3d
import torch



def registration_ransac_based_on_correspondence(
    ref_corres_xyz: torch.Tensor,
    src_corres_xyz: torch.Tensor,
    corr_weight: torch.Tensor = None,
    inlier_threshold = 0.05,
    topk = 250,
    ransac_iters = 50000,
    ransac_n = 4
):
    if corr_weight is None:
        indices = torch.arange(ref_corres_xyz.shape[0], device=ref_corres_xyz.device)[:topk]
    else:
        indices = torch.argsort(corr_weight, descending=True)[:topk]
    
    ref_points = ref_corres_xyz[indices]
    src_points = src_corres_xyz[indices]
    indices = np.arange(indices.shape[0])
    correspondences = np.stack([indices, indices], axis=1)
    correspondences = o3d.utility.Vector2iVector(correspondences)
        
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points.detach().cpu().numpy())
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_points.detach().cpu().numpy())

    transform = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd, ref_pcd, correspondences, inlier_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(inlier_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iters, 0.999)
    ).transformation

    return torch.FloatTensor(np.array(transform)).to(ref_corres_xyz.device)

