import torch
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F
from models.utils import apply_transform


class SpotMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(SpotMatchingLoss, self).__init__()
        self.positive_overlap = cfg.positive_overlap
    
    def forward(self, output_dict):
        coarse_matching_scores = output_dict['coarse_matching_scores']
        gt_node_corr_indices = output_dict['gt_patch_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_patch_corr_overlaps']
        
        with torch.no_grad():
            overlaps = torch.zeros_like(coarse_matching_scores)
            overlaps[gt_node_corr_indices[:, 0], gt_node_corr_indices[:, 1]] = gt_node_corr_overlaps
            pos_masks = torch.gt(overlaps, self.positive_overlap)

            row_mask = torch.zeros_like(overlaps, dtype=torch.bool)
            idx = overlaps.max(dim=1, keepdim=True)[1]
            row_mask.scatter_(1, idx, True)
            col_mask = torch.zeros_like(overlaps, dtype=torch.bool)
            idx = overlaps.max(dim=0, keepdim=True)[1]
            col_mask.scatter_(0, idx, True)
            pos_masks = overlaps * (pos_masks & row_mask & col_mask).float()
        
        if 'spot_matching_scores' in output_dict.keys():
            matching_scores = output_dict['spot_matching_scores']
            loss = -torch.log(matching_scores + 1e-8) * pos_masks.unsqueeze(0)
            loss = torch.sum(loss) / pos_masks.sum() / matching_scores.shape[0]
        
        coarse_loss = -torch.log(coarse_matching_scores + 1e-8) * pos_masks
        coarse_loss = torch.sum(coarse_loss) / pos_masks.sum()

        if 'ref_patch_overlap' in output_dict.keys():
            gt_ref_patch_overlap = 1. - pos_masks.sum(-1).gt(0).float()
            gt_src_patch_overlap = 1. - pos_masks.sum(-2).gt(0).float()
            gt_ref_patch_overlap = gt_ref_patch_overlap / (gt_ref_patch_overlap.sum() + 1e-8)
            gt_src_patch_overlap = gt_src_patch_overlap / (gt_src_patch_overlap.sum() + 1e-8)
            loss_ref_ov = -torch.log(1. - output_dict['ref_patch_overlap'] + 1e-8) * gt_ref_patch_overlap
            loss_src_ov = -torch.log(1. - output_dict['src_patch_overlap'] + 1e-8) * gt_src_patch_overlap
            #coarse_loss = coarse_loss + loss_ref_ov.mean() + loss_src_ov.mean()
            coarse_loss = coarse_loss + loss_ref_ov.sum() + loss_src_ov.sum()
            #loss = loss + loss_ref_ov.mean() + loss_src_ov.mean()
        
        if 'spot_matching_scores' in output_dict.keys():
            return loss, coarse_loss
        else: return coarse_loss



class KeypointMatchingLoss(nn.Module):
    """
    Modified from source codes of:
     - REGTR https://github.com/yewzijian/RegTR.
    """
    def __init__(self, positive_threshold, negative_threshold):
        super(KeypointMatchingLoss, self).__init__()
        self.r_p = positive_threshold
        self.r_n = negative_threshold
    
    def cal_loss(self, src_xyz, tgt_grouped_xyz, tgt_corres, match_logits, match_score, transform):
        #tgt_grouped_xyz = apply_transform(tgt_grouped_xyz, transform)
        #tgt_corres = apply_transform(tgt_corres, transform)
        src_xyz = apply_transform(src_xyz, transform)

        with torch.no_grad():
            dist_keypts:torch.Tensor = torch.norm(src_xyz.unsqueeze(1) - tgt_grouped_xyz, dim=-1)
            dist1, idx1 = torch.topk(dist_keypts, k=1, dim=-1, largest=False)
            mask = dist1[..., 0] < self.r_p  # Only consider points with correspondences
            ignore = dist_keypts < self.r_n  # Ignore all the points within a certain boundary
            ignore.scatter_(-1, idx1, 0)     # except the positive
            mask_id = mask.nonzero().squeeze()
        
        match_logits:torch.Tensor = match_logits - 1e4 * ignore.float()
        loss_feat = match_logits.logsumexp(dim=-1) - match_logits.gather(-1, idx1).squeeze(-1)
        loss_feat = loss_feat.index_select(0, mask_id).mean()
        if loss_feat.isnan(): loss_feat = 0.
        
        dist_keypts:torch.Tensor = torch.norm(src_xyz - tgt_corres, dim=-1)
        loss_corr = dist_keypts.index_select(0, mask_id).mean()
        if loss_corr.isnan(): loss_corr = 0.

        label = dist_keypts.lt(self.r_p)
        weight = torch.logical_not(label.logical_xor(dist_keypts.lt(self.r_n)))
        loss_ov = F.binary_cross_entropy(match_score, label.float(), weight.float())

        return loss_feat, loss_ov, loss_corr

    def forward(self, output_dict: Dict[str, torch.Tensor]):
        return self.cal_loss(
            output_dict['corres'][:, 3:],
            output_dict['ref_knn_points'],
            output_dict['corres'][:, :3],
            output_dict['match_logits'],
            output_dict['corr_confidence'],
            output_dict['gt_transform']
        )