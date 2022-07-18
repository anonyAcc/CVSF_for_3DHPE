import torch 
import numpy as np


def loss_weighted_rep_no_scale(p2d, p3d, confs):
    # the weighted reprojection loss as defined in Equation 5

    # normalize by scale
    scale_p2d = torch.sqrt(p2d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p2d_scaled = p2d[:, 0:32]/scale_p2d

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p3d_scaled = p3d[:, 0:32]/scale_p3d

    loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 16).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

    return loss

loss3d = 0    

def multiview3D(loss3d, pred_poses2, all_cams):
    pred_poses2_re = pred_poses2.reshape(-1,len(all_cams),48)
    for c_cnt in range(len(all_cams)-1):
        ## view consistency
        # get all cameras and active cameras
        ac = np.array(range(len(all_cams)))[c_cnt+1:]

        loss = (pred_poses2_re[:,c_cnt].repeat(len(ac), 1, 1).permute(1,0,2) - pred_poses2_re[:,ac]).abs()
        loss = loss.sum() / (pred_poses2_re[:,ac].shape[1])
        loss3d  += loss 
    loss3d /= (pred_poses2_re[:,ac].shape[0]*pred_poses2_re[:,ac].shape[2])

    return loss3d