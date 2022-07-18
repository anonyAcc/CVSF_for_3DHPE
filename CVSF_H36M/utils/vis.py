import numpy as np
import cv2
# from scipy import ndimage

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 
from utils.functions import *
from utils.skeletonmap import *

color_box = [(255, 0, 0),(0, 255, 0),(0, 0, 255)]
skeleton_color_box = [(0,0,255),(255,0,0),(0,150,0)]
red_skels = [0,1,2,12,13,14]
blue_skels = [3,4,5,9,10,11]
black_skels = [6,7,8]

def show3Dposecam(vis_3d_poses, ax, radius=40, lcolor='red', cam_view=0,data_type='h36m',angle=(10,-60)):            # channels : (17,3)
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6
    
    vals = vis_3d_poses[cam_view].T
    for ind, (i,j) in enumerate(JOINTMAP):
        x, z, y = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]    # connections와 짝을 맞춰서 2개씩 
        ax.plot(x, y, -z, lw=2, c=lcolor)

    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = vals[root_joint_numer, 0], vals[root_joint_numer, 1], vals[root_joint_numer, 2]    # root joint를 기준
        
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.view_init(angle[0], angle[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def show3Dpose_with_annot(annot, vals, ax, radius=40, data_type='h36m', lcolor='red', rcolor='#0000ff',angles=(10,-60)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    for ind, (i,j) in enumerate(JOINTMAP):
        x, z, y = [np.array([annot[i, c], annot[j, c]]) for c in range(3)]
        ax.plot(x, y, -z, lw=2, c='black')
        # x, y, z = [np.array([annot[i, c], annot[j, c]]) for c in range(3)]
        # ax.plot(x, -y, z, lw=2, c='black')

    # x,y,z = annot[root_joint_numer]
    # ax.scatter(x, -y, z, marker='o', s=15, c='purple')
    x, z, y = annot[root_joint_numer]
    ax.scatter(x, y, -z, marker='o', s=15, c='purple')

    for ind, (i,j) in enumerate(JOINTMAP):
        if ind in [0,1,2,6,7,8]:          # 오른쪽 
            color = 'b'    
        elif ind in [3,4,5,9,10,11]:       # 왼쪽 
            color = 'r'
        else:
            color = 'g'        # 중앙

        x, z, y = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, -z, lw=2, c=color)
    
    x, z, y = vals[root_joint_numer]
    ax.scatter(x, y, -z, marker='o', s=15, c='purple')
    RADIUS = radius  # space around the subject

    xroot, yroot, zroot = vals[root_joint_numer, 0], vals[root_joint_numer, 1], vals[root_joint_numer, 2]

    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.view_init(angles[0], angles[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def show2Dpose(vis_2d_poses,vis_pred_2d_poses, ax, data_type='h36m', cam_view=0, colorpred='#dc143c',image_size=(1280,1000)):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    image = np.zeros((image_size[0], image_size[1], 3), np.uint8)

    annot2d_keypoints = vis_2d_poses[cam_view].T
    pred2d_keypoints = vis_pred_2d_poses[cam_view].T
    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for j in range(len(JOINTMAP)):
        child = tuple(np.array(annot2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(annot2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        color = (144, 243, 34)

        cv2.circle(image, parent, 8, (255, 255, 255), -1)
        cv2.line(image, child, parent, color, 3) 

        childpred = tuple(np.array(pred2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parentpred = tuple(np.array(pred2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        # colorpred = (255, 0, 0)

        cv2.circle(image, parentpred, 8, (255, 0, 0), -1)
        cv2.line(image, childpred, parentpred, colorpred, 3) 

    cv2.circle(image, tuple(np.array(pred2d_keypoints[0][:2]).astype(int)), 8, (0, 0, 0), -1)
    plt.imshow(image)


def show2Dposeimg(annot2d_keypoints, img, ax, data_type='h36m', thin=2,color=(0,212,255),skeleton_color_box = [(0,0,255),(127,0,0),(0,150,0)]):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6
    # skeleton_color_box = [(255,255,255),(255,255,255),(255,255,255)]
    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for j in range(len(JOINTMAP)):
        if j in red_skels:
            lcolor = skeleton_color_box[0]
        elif j in blue_skels:
            lcolor = skeleton_color_box[1]
        else:
            lcolor = skeleton_color_box[2]

        child = tuple(np.array(annot2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(annot2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))

        cv2.line(img, child, parent, lcolor, thin) 
        # cv2.circle(img, parent, thin+1, color, -1)

    for jo in range(len(annot2d_keypoints)):
        joint = tuple(np.array(annot2d_keypoints[jo]).astype(int))
        cv2.circle(img, joint, thin, color, -1)
        

    # cv2.circle(img, tuple(annot2d_keypoints[0].astype(int)), thin+1, color, -1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return img


def show2Dposeimg_gt(inp_2d_keypoints, gt_2d_keypoints, conf, img, ax, data_type='h36m', thin=2,color=(0,0,0),lcolor=(0, 127, 255),th=0.8):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    # x=np.linspace(0,img.shape[0],img.shape[0]).astype(int)

    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for j in range(len(JOINTMAP)):
        child_gt = tuple(np.array(gt_2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent_gt = tuple(np.array(gt_2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))

        # cv2.circle(img, parent_gt, thin+1, (255,0,127), -1)
        cv2.line(img, child_gt, parent_gt, (255,0,0), thin+1) 

        child = tuple(np.array(inp_2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(inp_2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        # if conf[JOINTMAP[j][1]] >= th:
        #     cv2.circle(img, parent, thin+1, lcolor, -1)
        # else:
        #     cv2.circle(img, parent, thin+2, (0,255,0), -1)
        cv2.line(img, child, parent, lcolor, thin) 

    for jo in range(len(gt_2d_keypoints)):
        joint = tuple(np.array(gt_2d_keypoints[jo]).astype(int))
        cv2.circle(img, joint, thin+1, (255,0,127), -1)
        joint2 = tuple(np.array(inp_2d_keypoints[jo]).astype(int))
        if conf[jo] >= th:
            cv2.circle(img, joint2, thin+1, lcolor, -1)
        else:
            cv2.circle(img, joint2, thin+2, (0,255,0), -1)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img





def peep_skeleton(save_img, fig,inp_poses, rot_poses, relative_rotations_array, norm_2d, joints_2d, epoch, i):
    colors = ['#dc143c', '#8212c5', '#cdc013','#1d18ea','#13d0c2','#e415cc','#5ad214','#b61e11']
    colors_num = [(220,20,60), (130,20,200), (205,192,20),(30,30,245),(20,200,200),(230,20,200),(90,210,20),(180,30,20)]
    vis_input_2d_poses,_ = scaled_normalized2d(inp_poses)   
    vis_pred_2d_poses,_ = scaled_normalized2d(rot_poses) 
    pred_3dpose = rot_poses.reshape(-1, len(relative_rotations_array), 3, 16).cpu().detach().numpy()[0]
    pred_3dpose, _ = regular_normalized3d(pred_3dpose)

    
    view_3Dposes = np.zeros((len(relative_rotations_array),3,16))
    cams = np.zeros((len(relative_rotations_array),3))
    view_3Dposes[0] = pred_3dpose[0] 
    cams[0] = np.array([0,0,-1]) # np.array([0,0,-1/(34)**0.5])
    for view in range(1,len(relative_rotations_array)):
        view_3Dposes[view] = relative_rotations_array[view][0][0].cpu().detach().numpy() @ pred_3dpose[view] 
        cams[view] = relative_rotations_array[0][0][view-1].cpu().detach().numpy() @ cams[0]



    # 2d annot denormalization
    cnt = 0
    for b in range(joints_2d['cam0'].shape[0]):
        for rj_idx, rj in enumerate(joints_2d):
            # vis_input_2d_poses[cnt][0] =  (vis_input_2d_poses[cnt][0] * norm_2d[rj][b].cpu().detach().numpy()[0])/4 + joints_2d[rj][b].cpu().detach().numpy()[0][0] 
            # vis_input_2d_poses[cnt][1] = (vis_input_2d_poses[cnt][1] * norm_2d[rj][b].cpu().detach().numpy()[0])/4 + joints_2d[rj][b].cpu().detach().numpy()[1][0] 
            vis_input_2d_poses[cnt][0] =  (vis_input_2d_poses[cnt][0] * 90) + 250
            vis_input_2d_poses[cnt][1] = (vis_input_2d_poses[cnt][1] * 90) + 250
            # vis_pred_2d_poses[cnt][0] =  (vis_pred_2d_poses[cnt][0] * norm_2d[rj][b].cpu().detach().numpy()[0])/4 + joints_2d[rj][b].cpu().detach().numpy()[0][0] 
            # vis_pred_2d_poses[cnt][1] = (vis_pred_2d_poses[cnt][1] * norm_2d[rj][b].cpu().detach().numpy()[0])/4 + joints_2d[rj][b].cpu().detach().numpy()[1][0] 
            vis_pred_2d_poses[cnt][0] =  (vis_pred_2d_poses[cnt][0] * 90) + 250
            vis_pred_2d_poses[cnt][1] = (vis_pred_2d_poses[cnt][1] * 90) + 250 
            cnt += 1

    # 계획 : input 2d pose 시각화 및 회전 등등 
    ax1 = fig.add_subplot(1,2,1,projection='3d',aspect='auto')
    for v in range(len(pred_3dpose)):
        show3Dposecam(view_3Dposes*2, ax1, radius=1, lcolor=colors[v], cam_view=v, data_type='h36m',angle=(20,-60))
        ax1.scatter(cams[v][0], cams[v][2], -cams[v][1], c=colors[v], marker='o', s=15)

        ax2 = fig.add_subplot(4,4,3+ int(v/2)*4+(v%2))
        show2Dpose(vis_input_2d_poses,vis_pred_2d_poses, ax2,data_type='h36m',cam_view=v,colorpred=colors_num[v],image_size=(500,500))
        ax2.axis('off')

    plt.draw()
    plt.savefig(save_img + '%03d_%05d.png'% (epoch,i))
    plt.pause(0.1)
    plt.show()
    fig.clear()


def draw_skeleton(annot2d_keypoints, img, ax, data_type='h36m', thin=2,color=(0,212,255),skeleton_color_box = [(0,0,255),(255,0,0),(0,150,0)]):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6
    # skeleton_color_box = [(255,255,255),(255,255,255),(255,255,255)]
    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for j in range(len(JOINTMAP)):
        if j in red_skels:
            lcolor = skeleton_color_box[0]
        elif j in blue_skels:
            lcolor = skeleton_color_box[1]
        else:
            lcolor = skeleton_color_box[2]

        child = tuple(np.array(annot2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(annot2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))

        cv2.line(img, child, parent, lcolor, thin) 
        # cv2.circle(img, parent, thin+1, color, -1)

    for jo in range(len(annot2d_keypoints)):
        joint = tuple(np.array(annot2d_keypoints[jo]).astype(int))
        cv2.circle(img, joint, thin+1, color, -1)

    # cv2.circle(img, tuple(annot2d_keypoints[0].astype(int)), thin+1, color, -1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return img

def draw_skeleton_gt_conf(inp_2d_keypoints, gt_2d_keypoints, conf, img, ax, data_type='h36m', thin=1,color=(0,0,255),lcolor=(0, 127, 255),th=0.8):
    if data_type == 'h36m':
        JOINTMAP = H36M_JOINTMAP
        root_joint_numer = 0
    else:
        JOINTMAP = MPII_JOINTMAP
        root_joint_numer = 6

    # x=np.linspace(0,img.shape[0],img.shape[0]).astype(int)

    # 빈 이미지(검정 바탕)에 스켈레톤 그리기 
    for j in range(len(JOINTMAP)):
        child_gt = tuple(np.array(gt_2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent_gt = tuple(np.array(gt_2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))

        cv2.circle(img, parent_gt, thin+1, (255,0,0), -1)
        cv2.line(img, child_gt, parent_gt, (255,127,0), thin) 

        child = tuple(np.array(inp_2d_keypoints[JOINTMAP[j][0]][:2]).astype(int))
        parent = tuple(np.array(inp_2d_keypoints[JOINTMAP[j][1]][:2]).astype(int))
        if conf[JOINTMAP[j][1]] >= th:
            cv2.circle(img, parent, thin+1, color, -1)
        else:
            cv2.circle(img, parent, thin+1, (0,255,0), -1)
        cv2.line(img, child, parent, lcolor, thin) 

    cv2.circle(img, tuple(inp_2d_keypoints[0].astype(int)), thin+1, (255,127,127), -1)


    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


