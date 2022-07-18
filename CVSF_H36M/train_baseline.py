import torch
from torch import absolute
import torch.nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils import data
import torch.optim as optim
import model_confidences

from utils.data import H36MDataset
from utils.print_losses import print_losses
from utils.functions import *
from utils.losses import *
from utils.vis import peep_skeleton

from types import SimpleNamespace
from pytorch3d.transforms import so3_exponential_map as rodrigues
from numpy.random import default_rng

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter('runs')


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

config = SimpleNamespace()

config.learning_rate = 0.0001
config.BATCH_SIZE = 64 #256 # 256
config.N_epochs = 100   # 100

# weights for the different losses
config.weight_rep = 1
config.weight_view = 1
config.weight_camera = 0.1


# config.load_model = 'models/model_lifter_ski.pt'
config.save_model = 'models/model_lifter_baseline.pt'


data_folder = '../data/'

config.datafile = data_folder + 'alphapose_2d3dgt_img_h36m.pickle' 
config.morph_network = 'models/model_skeleton_morph_S1.pt'

config.gt_version=False
config.vis = False
config.save_img = 'experiments/training/'
if not os.path.isdir(config.save_img):
    os.makedirs(config.save_img)

cam_names = ['54138969', '55011271', '58860488', '60457274']
all_cams = ['cam0', 'cam1', 'cam2', 'cam3']


def train():
    # loading the H36M dataset                                 
    my_dataset = H36MDataset(config.datafile, normalize_2d=True, subjects=[5, 6, 7, 8])
    train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    # load the skeleton morphing model
    # for another joint detector it needs to be retrained 
    model_skel_morph = torch.load(config.morph_network )
    model_skel_morph.eval()

    # loading the lifting network
    _model = model_confidences.Lifter().cuda()
    model = torch.nn.DataParallel(_model).to(device) 
    #model = torch.load(config.load_model )
    #model.train()
    params = list(model.parameters())

    optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    losses = SimpleNamespace()
    losses_mean = SimpleNamespace()


    if config.vis:
        plt.ion()
        fig = plt.figure(figsize=(16,8))
    # 학습시작 
    for epoch in range(config.N_epochs):
        # iterlation
        for i, sample in enumerate(train_loader):
            poses_2d = {key:sample[key] for key in all_cams}
            joints_2d = {key:sample[key+'_joint'] for key in all_cams} 
            norm_2d = {key:sample[key+'_norm'] for key in all_cams}

            poses_2d_gt = {key:sample[key+'_2dgt'] for key in all_cams}
            joints_2d_gt = {key:sample[key+'_joint_gt'] for key in all_cams} 
            norm_2d_gt = {key:sample[key+'_norm_gt'] for key in all_cams}

            poses_3d_gt = {key:sample[key+'_3dgt'] for key in all_cams} 

            images = {key:sample[key+'_img'] for key in all_cams} 

            inp_poses = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), 32)).cuda()
            inp_confidences = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), 16)).cuda()
            poses_2dgt = torch.zeros((poses_2d_gt['cam0'].shape[0] * len(all_cams), 32)).cuda()
            poses_3dgt = torch.zeros((poses_3d_gt['cam0'].shape[0] * len(all_cams), 48)).cuda()

            inp_root = torch.zeros((joints_2d['cam0'].shape[0] * len(all_cams), 2,1)).cuda()
            inp_norm_2d = torch.zeros((norm_2d['cam0'].shape[0] * len(all_cams), 1)).cuda()
            gt_root = torch.zeros((joints_2d_gt['cam0'].shape[0] * len(all_cams), 2,1)).cuda()
            gt_norm_2d = torch.zeros((norm_2d_gt['cam0'].shape[0] * len(all_cams), 1)).cuda()

            # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
            cnt = 0
            for b in range(poses_2d['cam0'].shape[0]):
                for c_idx, cam in enumerate(poses_2d):
                    inp_poses[cnt] = poses_2d[cam][b]
                    inp_confidences[cnt] = sample['confidences'][cam_names[c_idx]][b]
                    poses_2dgt[cnt] = poses_2d_gt[cam][b]
                    poses_3dgt[cnt] = poses_3d_gt[cam][b]
                    
                    inp_root[cnt] = joints_2d[cam][b]
                    inp_norm_2d[cnt] = norm_2d[cam][b]
                    gt_root[cnt] = joints_2d_gt[cam][b]
                    gt_norm_2d[cnt] = norm_2d_gt[cam][b]
                    cnt += 1

# ---------------------------------------------- 1 Stage Lifting -----------------------------------------------
            inp_poses = model_skel_morph(inp_poses)

             # predict 3d poses
            pred = model(inp_poses, inp_confidences)
            pred_poses = pred[0]
            pred_cam_angles = pred[1]
            pred_rot = rodrigues(pred_cam_angles)    

            cam_location = np.array([0,0,-0.5])
            rot_all_view_cam_location = pred_rot.matmul(torch.from_numpy(cam_location).float().cuda())
            rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, 16)).reshape(-1, 48)          
            relative_rotations_array = []

            pred_poses_rs = pred_poses.reshape((-1, len(all_cams), 48))
            pred_rot_rs = pred_rot.reshape(-1, len(all_cams), 3, 3)                 # ([b*4, 3, 3]) --> ([b, 4, 3, 3])
            confidences_rs = inp_confidences.reshape(-1, len(all_cams), 16)
            inp_poses_rs = inp_poses.reshape(-1, len(all_cams), 32)
            rot_poses_rs = rot_poses.reshape(-1, len(all_cams), 48)

# ----------------------------------------------------- loss ------------------------------------------------------------------

            # reprojection loss
            losses.rep = loss_weighted_rep_no_scale(inp_poses, rot_poses, inp_confidences)
       
            # view and camera consistency are computed in the same loop
            losses.view = 0
            losses.camera = 0
            
            for c_cnt in range(len(all_cams)):
                ## view consistency
                # get all cameras and active cameras
                ac = np.array(range(len(all_cams)))
                coi = np.delete(ac, c_cnt)                     

                # view consistency             
                projected_to_other_cameras = pred_rot_rs[:, coi].matmul(pred_poses_rs.reshape(-1, len(all_cams), 3, 16)[:, c_cnt:c_cnt+1].repeat(1, len(all_cams)-1, 1, 1)).reshape(-1, len(all_cams)-1, 48)
                losses.view += loss_weighted_rep_no_scale(inp_poses.reshape(-1, len(all_cams), 32)[:, coi].reshape(-1, 32),
                                                        projected_to_other_cameras.reshape(-1, 48),
                                                        inp_confidences.reshape(-1, len(all_cams), 16)[:, coi].reshape(-1, 16))

                ## camera consistency
                relative_rotations = pred_rot_rs[:, coi].matmul(pred_rot_rs[:, [c_cnt]].permute(0, 1, 3, 2))    # ([b, 3(나머지3개view), 3, 3]) * ([b, 1(하나view), 3, 3]) = ([b, 3, 3, 3])
                relative_rotations_array.append(relative_rotations)
                # only shuffle in between subjects
                rng = default_rng()
                for subject in sample['subjects'].unique():                 # 5,6,7,8
                    # only shuffle if enough subjects are available
                    if (sample['subjects'] == subject).sum() > 1:
                        shuffle_subjects = (sample['subjects'] == subject)
                        num_shuffle_subjects = shuffle_subjects.sum()           # batch 데이터에서 subject 5의 개수 
                        rand_perm = rng.choice(num_shuffle_subjects.cpu().numpy(), size=num_shuffle_subjects.cpu().numpy(), replace=False)
                        samp_relative_rotations = relative_rotations[shuffle_subjects]
                        samp_rot_poses_rs = rot_poses_rs[shuffle_subjects]
                        samp_inp_poses = inp_poses_rs[shuffle_subjects][:, coi].reshape(-1, 32)
                        samp_inp_confidences = confidences_rs[shuffle_subjects][:, coi].reshape(-1, 16)

                        random_shuffled_relative_projections = samp_relative_rotations[rand_perm].matmul(samp_rot_poses_rs.reshape(-1, len(all_cams), 3, 16)[:, c_cnt:c_cnt+1].repeat(1, len(all_cams)-1, 1, 1)).reshape(-1, len(all_cams)-1, 48)

                        losses.camera += loss_weighted_rep_no_scale(samp_inp_poses,
                                                                    random_shuffled_relative_projections.reshape(-1, 48),
                                                                    samp_inp_confidences)

               


            # get combined loss
            losses.loss = config.weight_rep * losses.rep + \
                        config.weight_view * losses.view + \
                        config.weight_camera * losses.camera

            optimizer.zero_grad()
            losses.loss.backward()

            optimizer.step()

            for key, value in losses.__dict__.items():
                if key not in losses_mean.__dict__.keys():
                    losses_mean.__dict__[key] = []

                losses_mean.__dict__[key].append(value.item())

            # print progress every 100 iterations

            if not i % 100: 
                if config.vis:
                    peep_skeleton(config.save_img, fig, inp_poses, rot_poses, relative_rotations_array, norm_2d_gt, joints_2d_gt, epoch, i)
                print_losses(config.N_epochs, epoch, i, len(my_dataset) / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not(i % 1000))
                # this line is important for logging!
                losses_mean = SimpleNamespace()

        # 텐서보드에 저장   # 100 epoch, 1epoch당 저장 
        writer.add_scalar('total loss', losses.loss, epoch)
        writer.add_scalar('reproject loss', losses.rep, epoch)
        writer.add_scalar('view loss', losses.view, epoch)
        writer.add_scalar('camera loss', losses.camera, epoch)

        # epoch한번 돌때마다 모델 저장 
        # save the new trained model every epoch
        torch.save(model, config.save_model)
        # if epoch % 10 == 0:
        #     torch.save({
        #             'model': model.state_dict(),
        #             'optimizer': optimizer.state_dict()
        #         }, config.checkpoint + '/ski_checkpoint_{0:3d}.tar'.format(epoch))
        scheduler.step()

    print('done')

if __name__ == '__main__':
    train()
    print('done')
    print(config.save_model)







