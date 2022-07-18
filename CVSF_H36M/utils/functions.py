import torch 
import numpy as np


def scaled_normalized2d(pose):    # 
    scale_norm = torch.sqrt(pose[:, 0:32].square().sum(axis=1, keepdim=True) / 32)                    # 3d pose도 scaling 필요 
    p3d_scaled = pose[:, 0:32]/scale_norm
    return p3d_scaled.reshape(-1, 2, 16).cpu().detach().numpy(), scale_norm.cpu().detach().numpy()

def scaled_normalized3d(pose):    # 
    scale_norm = torch.sqrt(pose[:, 0:48].square().sum(axis=1, keepdim=True) / 48)
    scaled_pose = pose[:, 0:48]/scale_norm
    return scaled_pose.reshape(-1, 3, 16).cpu().detach().numpy(), scale_norm.cpu().detach().numpy()

# def regular_normalized3d(poseset):
#     pose_norm_list = []

#     for i in range(len(poseset)):
#         root_joints = poseset[i].T[:, [0]]                                     
#         pose_norm = np.linalg.norm((poseset[i].T - root_joints).reshape(-1, 48), ord=2, axis=1, keepdims=True)  
#         poseset[i] = (poseset[i].T - root_joints).T                
#         poseset[i] /= pose_norm
#         pose_norm_list.append(pose_norm)

#     return poseset, np.array(pose_norm_list)

def regular_normalized3d(poseset):
    pose_norm_list = []

    for i in range(len(poseset)):
        root_joints = poseset[i][0] 
        poseset[i] = (poseset[i] - root_joints)                                     
        pose_norm = np.linalg.norm(poseset[i].T.reshape(-1, 48), ord=2, axis=1, keepdims=True)  
                   
        poseset[i] /= pose_norm
        pose_norm_list.append(pose_norm)

    return poseset, np.array(pose_norm_list)

def get_denormalized_pose(inp_poses,rot_poses,joints_2d,norm_2d):
    vis_input_2d_poses, scale_norm = scaled_normalized2d(inp_poses)
    vis_pred_2d_poses,_ = scaled_normalized2d(rot_poses) 

    cnt = 0
    for b in range(joints_2d['cam0'].shape[0]):
        for rj_idx, rj in enumerate(joints_2d):
            vis_input_2d_poses[cnt][0] =  (vis_input_2d_poses[cnt][0] * norm_2d[rj][b].cpu().detach().numpy()[0] * scale_norm[0]) + joints_2d[rj][b].cpu().detach().numpy()[0][0] 
            vis_input_2d_poses[cnt][1] = (vis_input_2d_poses[cnt][1] * norm_2d[rj][b].cpu().detach().numpy()[0] * scale_norm[0]) + joints_2d[rj][b].cpu().detach().numpy()[1][0] 

            vis_pred_2d_poses[cnt][0] =  (vis_pred_2d_poses[cnt][0] * norm_2d[rj][b].cpu().detach().numpy()[0] * scale_norm[0]) + joints_2d[rj][b].cpu().detach().numpy()[0][0] 
            vis_pred_2d_poses[cnt][1] = (vis_pred_2d_poses[cnt][1] * norm_2d[rj][b].cpu().detach().numpy()[0] * scale_norm[0]) + joints_2d[rj][b].cpu().detach().numpy()[1][0] 
            cnt += 1
    return vis_input_2d_poses, vis_pred_2d_poses

# denormalize pose 
def denormalize_pose(all_cams,poses,norm_2d,root):
    poses = poses.reshape(-1,16,2)
    denorm_pose = (poses * norm_2d.reshape(-1,1,1).cpu().detach().numpy())  # + inp_root
    denorm_pose = denorm_pose + root.permute(0,2,1).cpu().detach().numpy()
    return denorm_pose 

def each_joints_jdrs(half_head,mypose,gtpose,joints_jdr_lists):
    joints_distances = np.sqrt(np.sum((mypose - gtpose)**2, axis=1))          # 
    each_joints_jdrs =[1 if jd < half_head else 0 for jd in joints_distances]
    for j,jdr in enumerate(each_joints_jdrs):
        joints_jdr_lists[j].append(jdr)

def reprojection(p2d, p3d):
    # the weighted reprojection loss as defined in Equation 5

    # normalize by scale
    scale_p2d = torch.sqrt(p2d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p2d_scaled = p2d[:, 0:32]/scale_p2d

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p3d_scaled = p3d[:, 0:32]/scale_p3d
    
    reprojection_2d = p3d_scaled * scale_p2d

    return reprojection_2d


from pytorch3d.transforms import so3_exponential_map as rodrigues

def vaild_function(valid_loader,all_cams,cam_names,model_skel_morph,model):
    mpjpes = []
    pmpjpes = []
    pcks = []
    for vaild_iter, vaild_sample in enumerate(valid_loader):
        poses_2d = {key:vaild_sample[key] for key in all_cams}
        poses_3d_gt = {key:vaild_sample[key+'_3dgt'] for key in all_cams} 
        inp_poses = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), 32)).cuda()
        inp_confidences = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), 16)).cuda()
        poses_3dgt = torch.zeros((poses_3d_gt['cam0'].shape[0] * len(all_cams), 48)).cuda()

        # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
        cnt = 0
        for b in range(poses_2d['cam0'].shape[0]):
            for c_idx, cam in enumerate(poses_2d):
                inp_poses[cnt] = poses_2d[cam][b]
                inp_confidences[cnt] = vaild_sample['confidences'][cam_names[c_idx]][b]   # np.ones(annot2d_cam0[i].shape[0])
                poses_3dgt[cnt] = poses_3d_gt[cam][b]
                cnt += 1

        inp_poses = model_skel_morph(inp_poses)
        pred = model(inp_poses, inp_confidences)
        pred_poses = pred[0]
        pred_cam_angles = pred[1]
        pred_rot = rodrigues(pred_cam_angles)      
        rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, 16)).reshape(-1, 48)  
        pred_3dpose = rot_poses.reshape(-1, 3, 16).cpu().detach().numpy()
        pred_3dpose = np.transpose(pred_3dpose,(0,2,1))
        poses_3dgt = poses_3dgt.reshape(-1, 16, 3).cpu().detach().numpy()
        poses_3dgt, pose_norm = regular_normalized3d(poses_3dgt)  
        pred_3dpose, _ = regular_normalized3d(pred_3dpose) 
        poses_3dgt = poses_3dgt * 3500
        pred_3dpose = pred_3dpose * 3500
        mpjpe = sum([np.mean(np.sqrt(np.sum((pred_3dpose[i]- poses_3dgt[i])**2, axis=1))) for i in range(len(all_cams))]) / len(all_cams)
        mpjpes.append(mpjpe)
        # print('mpjpe:',mpjpe)

        pmpjpe = []
        for v in range(len(all_cams)):
            mpjpe = np.mean(np.sqrt(np.sum((pred_3dpose[v] - poses_3dgt[v])**2, axis=1)))
            pmpjpe.append(mpjpe)
        pmpjpe = min(pmpjpe)
        #print('pmpjpe:',pmpjpe)
        pmpjpes.append(pmpjpe)

        diff = np.sqrt(np.square(poses_3dgt - pred_3dpose).sum(axis=2))
        pck = 100 * len(np.where(diff < 150)[0]) / (diff.shape[0] * diff.shape[1])
        #print('pck:',round(pck))
        pcks.append(pck)
            

    # print('---'*10)
    # # print(len(mpjpes))
    print('MPJE', np.mean(mpjpes))
    print('PMPJE', np.mean(pmpjpes))
    print('PCK',np.mean(pcks))

    return np.mean(mpjpes), np.mean(pmpjpes), np.mean(pcks)