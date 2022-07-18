import numpy as np
from tenacity import retry
import torch


def heatmap_to_coord(heatmaps):
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1))  # (26,3072)
    idx = np.argmax(heatmaps_reshaped, 1)       # 이거는 confidence가 가장 높은 joint의 위치 26개
    maxvals = np.max(heatmaps_reshaped, 1)      # confidence 값이고 

    maxvals = maxvals.reshape((num_joints, 1))  # (26,1)
    idx = idx.reshape((num_joints, 1))          # 

    preds = np.tile(idx, (1, 2)).astype(np.float32)

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = np.floor((preds[:, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds #, maxvals

def joints_heatmaps_fusion(all_cams, poses, size, sigma, joint_weights=np.array([[1.]*16])):
    poses = poses.reshape(-1,len(all_cams),32) # (b*view(4),viewpose(4),32)
    sigma = int(sigma * size / 128)
    weights = joint_weights.reshape(-1,16)
    win_size = 2 * sigma + 1
    x, y = np.meshgrid(np.linspace(-sigma, sigma, num=win_size, endpoint=True), np.linspace(-sigma, sigma, num=win_size, endpoint=True)) # joint의 heatmap을 만들크기 (31,31)
    dst = np.sqrt(x * x + y * y)
    mu = 0.000
    gauss = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))     

    fusion_heatmap_list = []
    # refine_poses_list = []
    for i in range(len(poses)):             # 4 
        pose = poses[i].reshape(-1,2,16)
        pose = np.transpose(pose,(0,2,1))
        heatmaps = np.zeros((pose.shape[0], pose.shape[1], size + 2 * sigma, size + 2 * sigma))
        output_heatmaps = np.zeros((pose.shape[0], pose.shape[1], size, size ))
        # joint_conf = weights[i]
        for view in range(len(pose)):               # view 4개 
                                            # 자기자신의 heatmap은 만들지말고 다른 view의 joint heatmap만 만들어 
            for j, [X,Y] in enumerate(pose[view]):  # joint 16개
                X,Y = int(X), int(Y)
                if X <= 0 or X >= size or Y <= 0 or Y >= size:
                    weights[view][j] = 0
                    continue
                heatmaps[view, j, Y: Y + win_size, X: X + win_size] = gauss * weights[view][j]    # joint 위치에서 x,y로 31만큼의 정사각형 위치에 값을 gaussian weight를 넣어줌

            output_heatmaps[view] = heatmaps[view, :, sigma:-sigma, sigma:-sigma]

        fusion_pose = np.sum(output_heatmaps, axis=0)
        fusion_heatmap_list.append(fusion_pose)
        # refine_pose = heatmap_to_coord(fusion_pose)
        # refine_poses_list.append(refine_pose)
    
    # refine_poses = np.array(refine_poses_list)

    # return refine_poses, fusion_heatmap_list
    return fusion_heatmap_list

def epipolarline_heatmap_tensor(all_cams, except_joints, BATCH_SIZE, size, relative_rotations_list, ray_3d, conf, joint_th, sampler=12, sigma=2, joints_lists = [15]):
    # all_cams = ['cam0', 'cam1', 'cam2', 'cam3']
    # rot_ray_list = [[[[],[],[],[]]]*BATCH_SIZE][0]
    rot_ray_list = list([[list()]*len(all_cams)]*BATCH_SIZE)
    # power=4
    sigma = int(sigma * size / 64)

    lines_heatmap = np.zeros((BATCH_SIZE,len(all_cams),len(all_cams)-1,len(joints_lists),size + 2 * sigma, size + 2 * sigma))
    lines_heatmap_output = np.zeros((BATCH_SIZE,len(all_cams),len(all_cams)-1,len(joints_lists),size,size))
    win_size = 2 * sigma + 1

    # dst=np.linspace(-sigma, sigma, num=win_size, endpoint=True)
    # mu = 0.000
    # gauss = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))[win_size//2:]

    x, y = np.meshgrid(np.linspace(-sigma, sigma, num=win_size, endpoint=True), np.linspace(-sigma, sigma, num=win_size, endpoint=True)) # joint의 heatmap을 만들크기 (31,31)
    dst = np.sqrt(x * x + y * y)
    mu = 0.000
    gauss = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))  


    for b in range(BATCH_SIZE):                                     # batch별               # 32*4*3*16
        list_fusion_joints_view = []
        list_fusion_joints_views =  [[] for i in range(len(all_cams))] 
        for jo in range(len(conf[b][0])):           # 16개 joint별    
            if jo in except_joints:   
                continue
            for v in range(len(all_cams)):   # 4개의 view별
                if conf[b][v][jo] < joint_th:               # th 미만인 joint만 refine 하기위해
                    list_fusion_joints_views[v].append(jo)   # 16개중 확인안할꺼 빼고, conf작은거 빼고 
                    list_fusion_joints_view.append(jo)
        list_fusion_joints_view = list(set(list_fusion_joints_view))
        # list_fusion_joints_view가 [[2, 15], [12], [6], []]라면 
        for c_cnt in range(len(all_cams)):                          # 4개 view씩
            ac = np.array(range(len(all_cams)))
            coi = np.delete(ac, c_cnt) 
            rot_rays = []
            for i in range(len(all_cams)-1):
                rot_to0_ray = (relative_rotations_list[b][coi[i]][int(np.where(np.delete(ac, coi[i])==c_cnt)[0])] @ ray_3d[b][coi[i]]).permute(0,2,1)  
                rot_rays.append(rot_to0_ray[list_fusion_joints_view])

            rot_ray_list[b][c_cnt] = rot_rays

            for k, ray in enumerate(rot_ray_list[b][c_cnt]):          # 1,2,3 -> 0 len(rot_ray_list[b][c_cnt]) : 3   
                for i in range(len(ray[:,:,:2])):                       # joint 16개
                    ray2d = (ray[:,:,:2][i].T).cpu().detach().numpy()   # 3번째 z는 뺌
                    xy_set = ray2d * size + size // 2                            # 32개
                    vec = (ray2d[1][0]-ray2d[1][1]) / (ray2d[0][0]-ray2d[0][1])
                    c = xy_set[1][0] - (vec * xy_set[0][0])

                    x = np.linspace(0,size,size).astype(int)
                    y = vec * x + c
                    ndx = (x>size*0.2) & (x<size*0.8) & (y>size*0.1) & (y<size*0.9)
                    # ndx = (x>0) & (x<size) & (y>0) & (y<size)
                    
                    if len(x[ndx]) > 1 :
                        xx = np.linspace(x[ndx][0],x[ndx][-1],num=sampler)  # 
                        yy = vec * xx + c

                        for xxx,yyy in zip(xx.astype(int), yy.astype(int)):
                            lines_heatmap[b][c_cnt][k][list_fusion_joints_view[i]][yyy : yyy+win_size, xxx : xxx+win_size] = (conf[:,:,joints_lists][b][coi[k]][list_fusion_joints_view[i]])*gauss   # 10*gauss # (conf[:,:,joints_lists][0][coi[k]][i])**power * 22 

                lines_heatmap_output[b][c_cnt][k] = lines_heatmap[b][c_cnt][k][:,sigma:-sigma, sigma:-sigma]
            
            for j in range(16):     # joint별 epipolar 3개가 0인게 2개이면 그 joint별 epipolar를 참고하지않는다
                count = 0
                for v in range(len(all_cams)-1):
                    if np.max(lines_heatmap_output[b][c_cnt][:,j,:,:][v]) == 0:
                        count += 1
                if count == len(all_cams)-2:
                    lines_heatmap_output[b][c_cnt][:,j,:,:] = np.zeros((len(all_cams)-1,size,size))


    return  np.sum(lines_heatmap_output,axis=2), list_fusion_joints_views



def epipolarline_heatmap_tensor2(all_cams, except_joints, BATCH_SIZE, size, relative_rotations_list, ray_3d, joints_heatmaps, refine_inp_poses, conf, joint_th, sampler=12, sigma=2, joints_lists = [15]):
    # all_cams = ['cam0', 'cam1', 'cam2', 'cam3']
    # rot_ray_list = [[[[],[],[],[]]]*BATCH_SIZE][0]
    rot_ray_list = list([[list()]*len(all_cams)]*BATCH_SIZE)
    fusion_sum_list = []
    # power=4
    sigma = int(sigma * size / 64)

    lines_heatmap = np.zeros((BATCH_SIZE,len(all_cams),len(all_cams)-1,len(joints_lists),size + 2 * sigma, size + 2 * sigma))
    lines_heatmap_output = np.zeros((BATCH_SIZE,len(all_cams),len(all_cams)-1,len(joints_lists),size,size))
    win_size = 2 * sigma + 1

    # dst=np.linspace(-sigma, sigma, num=win_size, endpoint=True)
    # mu = 0.000
    # gauss = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))[win_size//2:]

    x, y = np.meshgrid(np.linspace(-sigma, sigma, num=win_size, endpoint=True), np.linspace(-sigma, sigma, num=win_size, endpoint=True)) # joint의 heatmap을 만들크기 (31,31)
    dst = np.sqrt(x * x + y * y)
    mu = 0.000
    gauss = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))  


    for b in range(BATCH_SIZE):                                     # batch별               # 32*4*3*16
        list_fusion_joints_view = []
        list_fusion_joints_views =  [[] for i in range(len(all_cams))] 
        for jo in range(len(conf[b][0])):           # 16개 joint별    
            if jo in except_joints:   
                continue
            for v in range(len(all_cams)):   # 4개의 view별
                if conf[b][v][jo] < joint_th:               # th 미만인 joint만 refine 하기위해
                    list_fusion_joints_views[v].append(jo)   # 16개중 확인안할꺼 빼고, conf작은거 빼고 
                    list_fusion_joints_view.append(jo)
        list_fusion_joints_view = list(set(list_fusion_joints_view))
        # list_fusion_joints_view가 [[2, 15], [12], [6], []]라면 
        for c_cnt in range(len(all_cams)):                          # 4개 view씩
            ac = np.array(range(len(all_cams)))
            coi = np.delete(ac, c_cnt) 
            rot_rays = []
            for i in range(len(all_cams)-1):
                rot_to0_ray = (relative_rotations_list[b][coi[i]][int(np.where(np.delete(ac, coi[i])==c_cnt)[0])] @ ray_3d[b][coi[i]]).permute(0,2,1)  
                rot_rays.append(rot_to0_ray[list_fusion_joints_view])

            rot_ray_list[b][c_cnt] = rot_rays

            for k, ray in enumerate(rot_ray_list[b][c_cnt]):          # 1,2,3 -> 0 len(rot_ray_list[b][c_cnt]) : 3   
                for i in range(len(ray[:,:,:2])):                       # joint 16개
                    ray2d = (ray[:,:,:2][i].T).cpu().detach().numpy()   # 3번째 z는 뺌
                    xy_set = ray2d * size + size // 2                            # 32개
                    vec = (ray2d[1][0]-ray2d[1][1]) / (ray2d[0][0]-ray2d[0][1])
                    c = xy_set[1][0] - (vec * xy_set[0][0])

                    x = np.linspace(0,size,size).astype(int)
                    y = vec * x + c
                    ndx = (x>size*0.2) & (x<size*0.8) & (y>size*0.1) & (y<size*0.9)
                    # ndx = (x>0) & (x<size) & (y>0) & (y<size)
                    
                    if len(x[ndx]) > 1 :
                        xx = np.linspace(x[ndx][0],x[ndx][-1],num=sampler)  # 
                        yy = vec * xx + c

                        for xxx,yyy in zip(xx.astype(int), yy.astype(int)):
                            lines_heatmap[b][c_cnt][k][list_fusion_joints_view[i]][yyy : yyy+win_size, xxx : xxx+win_size] = (conf[:,:,joints_lists][b][coi[k]][list_fusion_joints_view[i]])*gauss   # 10*gauss # (conf[:,:,joints_lists][0][coi[k]][i])**power * 22 

                lines_heatmap_output[b][c_cnt][k] = lines_heatmap[b][c_cnt][k][:,sigma:-sigma, sigma:-sigma]
            
            for j in range(16):     # joint별 epipolar 3개가 0인게 2개이면 그 joint별 epipolar를 참고하지않는다
                count = 0
                for v in range(len(all_cams)-1):
                    if np.max(lines_heatmap_output[b][c_cnt][:,j,:,:][v]) == 0:
                        count += 1
                if count == len(all_cams)-2:
                    lines_heatmap_output[b][c_cnt][:,j,:,:] = np.zeros((len(all_cams)-1,size,size))

            # lines_heatmap_output[b][c_cnt]    np.sum(lines_heatmap_output[b][c_cnt], axis=0)
            # 뷰별로 
            # np.sum(lines_heatmap_output,axis=2)
            ep_heatmaps = np.sum(lines_heatmap_output[b][c_cnt], axis=0)
            # view_heatmap = joints_heatmaps[b][c_cnt,list_fusion_joints_view[c_cnt],:,:] + ep_heatmaps[list_fusion_joints_view[c_cnt],:,:]
            view_heatmap = joints_heatmaps[b][c_cnt,:,:,:] + ep_heatmaps[:,:,:]
            fusion_sum_list.append(joints_heatmaps[b][c_cnt,:,:,:] + ep_heatmaps[:,:,:])
            if len(view_heatmap) > 0 and len(view_heatmap.shape) !=len(all_cams)-2:    
                view_refine_joints = heatmap_to_coord(view_heatmap)
                cnt = 0
                for i in list_fusion_joints_view[c_cnt]:
                    refine_inp_poses[b][c_cnt][i] = view_refine_joints[cnt] # torch.Tensor(view_refine_joints[cnt]).to(device)   # refine되는곳
                    cnt += 1

    return  np.array(fusion_sum_list), refine_inp_poses

    # return  np.sum(lines_heatmap_output,axis=2), list_fusion_joints_views



def epipolarline_heatmap(all_cams, except_joints, BATCH_SIZE, size, relative_rotations_list, ray_3d, joints_heatmaps, refine_inp_poses, 
                                conf, joint_th, sigma=2):
    # all_cams = ['cam0', 'cam1', 'cam2', 'cam3']
    joints_lists = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
    rot_ray_list = [[list([[]]*len(all_cams))]*BATCH_SIZE][0] # [[[[],[],[],[]]]*BATCH_SIZE][0] # list_fusion_joints_view = list([[]]*len(all_cams))
    fusion_sum_list = []
    sigma = int(sigma * size / 64)
    conf = conf.reshape(-1,len(all_cams),16)
    lines_heatmap = np.zeros((BATCH_SIZE,len(all_cams),len(all_cams)-1,len(joints_lists),size + 2 * sigma, size + 2 * sigma))
    lines_heatmap_output = np.zeros((BATCH_SIZE,len(all_cams),len(all_cams)-1,len(joints_lists),size,size))
    win_size = 2 * sigma + 1

    # dst=np.linspace(-sigma, sigma, num=win_size, endpoint=True)
    # mu = 0.000
    # gauss = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))[win_size//2:]

    # x, y = np.meshgrid(np.linspace(-sigma, sigma, num=win_size, endpoint=True), np.linspace(-sigma, sigma, num=win_size, endpoint=True)) # joint의 heatmap을 만들크기 (31,31)
    # dst = np.sqrt(x * x + y * y)
    dst=np.linspace(-sigma, sigma, num=win_size, endpoint=True)
    mu = 0.000
    # gauss = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))[win_size//2:]
    gauss = [1,0.8007,0.6065,0.4072]

    for b in range(BATCH_SIZE):                                     # batch별               # 32*4*3*16
         # view별 joint의 conf가 th보다 낮으면 view별로 refine할 joint를 리스트에 넣는다
        list_fusion_joints_view = list([[]]*len(all_cams))
        list_fusion_joints = []
        for jo in range(len(conf[b][0])):           # 16개 joint별    
            if jo in except_joints:   
                continue
            for v in range(len(all_cams)):   # 4개의 view별
                if conf[b][v][jo] < joint_th:               # th 미만인 joint만 refine 하기위해
                    list_fusion_joints_view[v].append(jo)   # 16개중 확인안할꺼 빼고, conf작은거 빼고 
                    list_fusion_joints.append(jo)

        list_fusion_joints = list(set(list_fusion_joints))
        # list_fusion_joints_view가 [[2, 15], [12], [6], []]라면 
        for c_cnt in range(len(all_cams)):                          # 4개 view씩
            ac = np.array(range(len(all_cams)))
            coi = np.delete(ac, c_cnt) 
            rotation_ray_list = []
            for i in range(len(all_cams)-1):
                rot_Ato0_ray = (relative_rotations_list[b][coi[i]][int(np.where(np.delete(ac, coi[i])==c_cnt)[0])] @ ray_3d[b][coi[i]]).permute(0,2,1)  
                rotation_ray_list.append(rot_Ato0_ray[list_fusion_joints])
            # rot_1to0_ray = (relative_rotations_list[b][coi[0]][int(np.where(np.delete(ac, coi[0])==c_cnt)[0])] @ ray_3d[b][coi[0]]).permute(0,2,1)  
            # rot_2to0_ray = (relative_rotations_list[b][coi[1]][int(np.where(np.delete(ac, coi[1])==c_cnt)[0])] @ ray_3d[b][coi[1]]).permute(0,2,1) 
            # rot_3to0_ray = (relative_rotations_list[b][coi[2]][int(np.where(np.delete(ac, coi[2])==c_cnt)[0])] @ ray_3d[b][coi[2]]).permute(0,2,1) 
            # b별 카메라별 3개씩의 epipolar line이 있다.
            #rot_ray_list[b][c_cnt] = [rot_1to0_ray[list_fusion_joints_view[coi[0]]],rot_2to0_ray[list_fusion_joints_view[coi[1]]],rot_3to0_ray[list_fusion_joints_view[coi[2]]]]
            # rot_ray_list[b][c_cnt] = [rot_1to0_ray[list_fusion_joints],rot_2to0_ray[list_fusion_joints],rot_3to0_ray[list_fusion_joints]]
            rot_ray_list[b][c_cnt] = rotation_ray_list
     
            for k, ray in enumerate(rot_ray_list[b][c_cnt]):          # 1,2,3 -> 0 len(rot_ray_list[b][c_cnt]) : 3   
                for i in range(len(ray[:,:,:2])):                       # joint 16개
                    ray2d = (ray[:,:,:2][i].T).cpu().detach().numpy()   # 3번째 z는 뺌
                    xy_set = ray2d * size + size // 2                            # 32개
                    vec = (ray2d[1][0]-ray2d[1][1]) / (ray2d[0][0]-ray2d[0][1])
                    c = xy_set[1][0] - (vec * xy_set[0][0])

                    x = np.linspace(0,size,size).astype(int)
                    y = vec * x + c
                    ndx = (x>0) & (x<size) & (y>0) & (y<size)
                    # ndx = (x>0) & (x<size) & (y>0) & (y<size)
                    
                    if len(x[ndx]) > 1 :
                        xx = np.linspace(x[ndx][0],x[ndx][-1],num=size)  # 
                        yy = vec * xx + c
                        for g in reversed(range(len(gauss))):
                            for xxx,yyy in zip(xx.astype(int), yy.astype(int)):
                                lines_heatmap[b][c_cnt][k][list_fusion_joints[i]][yyy-g:yyy+g, xxx-g:xxx+g] = (conf[:,:,joints_lists][b][coi[k]][list_fusion_joints[i]])*gauss[g]  

                lines_heatmap_output[b][c_cnt][k] = lines_heatmap[b][c_cnt][k][:,sigma:-sigma, sigma:-sigma]
            
            for j in range(16):     # joint별 epipolar 3개가 0인게 2개이면 그 joint별 epipolar를 참고하지않는다
                count = 0
                for v in range(len(all_cams)-1):
                    if np.max(lines_heatmap_output[b][c_cnt][:,j,:,:][v]) == 0:
                        count += 1
                if count == len(all_cams)-2:
                    lines_heatmap_output[b][c_cnt][:,j,:,:] = np.zeros((len(all_cams)-1,size,size))

            # lines_heatmap_output[b][c_cnt]    np.sum(lines_heatmap_output[b][c_cnt], axis=0)
            # 뷰별로 
            # np.sum(lines_heatmap_output,axis=2)
            ep_heatmaps = np.sum(lines_heatmap_output[b][c_cnt], axis=0)
            view_heatmap = joints_heatmaps[b][c_cnt,list_fusion_joints_view[c_cnt],:,:] + ep_heatmaps[list_fusion_joints_view[c_cnt],:,:]
            fusion_sum_list.append(joints_heatmaps[b][c_cnt,:,:,:] + ep_heatmaps[:,:,:])
            if len(view_heatmap) > 0 and len(view_heatmap.shape) !=len(all_cams)-2:    
                view_refine_joints = heatmap_to_coord(view_heatmap)
                cnt = 0
                for i in list_fusion_joints_view[c_cnt]:
                    refine_inp_poses[b][c_cnt][i] = view_refine_joints[cnt] # torch.Tensor(view_refine_joints[cnt]).to(device)   # refine되는곳
                    cnt += 1

    return  np.array(fusion_sum_list), refine_inp_poses




def refine_module(all_cams, except_joints, BATCH_SIZE, size, relative_rotations_list, ray_3d, joints_heatmaps, refine_inp_poses, conf, joint_th, sampler=12, sigma=2, joints_lists = [15]):
    # epipolar line heatmap generator 
    rot_ray_list = list([[list()]*len(all_cams)]*BATCH_SIZE)
    fusion_sum_list = []
    # power=4
    sigma = int(sigma * size / 64)

    lines_heatmap = np.zeros((BATCH_SIZE,len(all_cams),len(all_cams)-1,len(joints_lists),size + 2 * sigma, size + 2 * sigma))
    lines_heatmap_output = np.zeros((BATCH_SIZE,len(all_cams),len(all_cams)-1,len(joints_lists),size,size))
    win_size = 2 * sigma + 1

    # dst=np.linspace(-sigma, sigma, num=win_size, endpoint=True)
    # mu = 0.000
    # gauss = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))[win_size//2:]

    x, y = np.meshgrid(np.linspace(-sigma, sigma, num=win_size, endpoint=True), np.linspace(-sigma, sigma, num=win_size, endpoint=True)) # joint의 heatmap을 만들크기 (31,31)
    dst = np.sqrt(x * x + y * y)
    mu = 0.000
    gauss = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))  


    for b in range(BATCH_SIZE):                                     # batch별               # 32*4*3*16
        list_fusion_joints_view = []
        list_fusion_joints_views =  [[] for i in range(len(all_cams))] 
        for jo in range(len(conf[b][0])):           # 16개 joint별    
            if jo in except_joints:   
                continue
            for v in range(len(all_cams)):   # 4개의 view별
                if conf[b][v][jo] < joint_th:               # th 미만인 joint만 refine 하기위해
                    list_fusion_joints_views[v].append(jo)   # 16개중 확인안할꺼 빼고, conf작은거 빼고 
                    list_fusion_joints_view.append(jo)
        list_fusion_joints_view = list(set(list_fusion_joints_view))
        # list_fusion_joints_view가 [[2, 15], [12], [6], []]라면 
        for c_cnt in range(len(all_cams)):                          # 4개 view씩
            ac = np.array(range(len(all_cams)))
            coi = np.delete(ac, c_cnt) 
            rot_rays = []
            for i in range(len(all_cams)-1):
                rot_to0_ray = (relative_rotations_list[b][coi[i]][int(np.where(np.delete(ac, coi[i])==c_cnt)[0])] @ ray_3d[b][coi[i]]).permute(0,2,1)  
                rot_rays.append(rot_to0_ray[list_fusion_joints_view])

            rot_ray_list[b][c_cnt] = rot_rays

            for k, ray in enumerate(rot_ray_list[b][c_cnt]):          # 1,2,3 -> 0 len(rot_ray_list[b][c_cnt]) : 3   
                for i in range(len(ray[:,:,:2])):                       # joint 16개
                    ray2d = (ray[:,:,:2][i].T).cpu().detach().numpy()   # 3번째 z는 뺌
                    xy_set = ray2d * size + size // 2                            # 32개
                    vec = (ray2d[1][0]-ray2d[1][1]) / (ray2d[0][0]-ray2d[0][1])
                    c = xy_set[1][0] - (vec * xy_set[0][0])

                    x = np.linspace(0,size,size).astype(int)
                    y = vec * x + c
                    ndx = (x>size*0.2) & (x<size*0.8) & (y>size*0.1) & (y<size*0.9)
                    # ndx = (x>0) & (x<size) & (y>0) & (y<size)
                    
                    if len(x[ndx]) > 1 :
                        xx = np.linspace(x[ndx][0],x[ndx][-1],num=sampler)  # 
                        yy = vec * xx + c

                        for xxx,yyy in zip(xx.astype(int), yy.astype(int)):
                            lines_heatmap[b][c_cnt][k][list_fusion_joints_view[i]][yyy : yyy+win_size, xxx : xxx+win_size] = (conf[:,:,joints_lists][b][coi[k]][list_fusion_joints_view[i]])*gauss   # 10*gauss # (conf[:,:,joints_lists][0][coi[k]][i])**power * 22 

                lines_heatmap_output[b][c_cnt][k] = lines_heatmap[b][c_cnt][k][:,sigma:-sigma, sigma:-sigma]
            
            for j in range(16):     # joint별 epipolar 3개가 0인게 2개이면 그 joint별 epipolar를 참고하지않는다
                count = 0
                for v in range(len(all_cams)-1):
                    if np.max(lines_heatmap_output[b][c_cnt][:,j,:,:][v]) == 0:
                        count += 1
                if count == len(all_cams)-2:
                    lines_heatmap_output[b][c_cnt][:,j,:,:] = np.zeros((len(all_cams)-1,size,size))

            # lines_heatmap_output[b][c_cnt]    np.sum(lines_heatmap_output[b][c_cnt], axis=0)
            # 뷰별로 
            # np.sum(lines_heatmap_output,axis=2)
            ep_heatmaps = np.sum(lines_heatmap_output[b][c_cnt], axis=0)
            view_heatmap = joints_heatmaps[b][c_cnt,list_fusion_joints_views[c_cnt],:,:] + ep_heatmaps[list_fusion_joints_views[c_cnt],:,:]
            fusion_sum_list.append(joints_heatmaps[b][c_cnt,:,:,:] + ep_heatmaps[:,:,:])
            # view_heatmap = joints_heatmaps[b][c_cnt,:,:,:] + ep_heatmaps[:,:,:]
            # fusion_sum_list.append(view_heatmap)
            if len(view_heatmap) > 0 and len(view_heatmap.shape) !=len(all_cams)-2:    
                view_refine_joints = heatmap_to_coord(view_heatmap)
                cnt = 0
                for i in list_fusion_joints_views[c_cnt]:
                    refine_inp_poses[b][c_cnt][i] = view_refine_joints[cnt] # torch.Tensor(view_refine_joints[cnt]).to(device)   # refine되는곳
                    cnt += 1

    return  np.array(fusion_sum_list), refine_inp_poses

    # return  np.sum(lines_heatmap_output,axis=2), list_fusion_joints_views



def CSM_test(BATCH_SIZE,all_cams,inp_poses,conf,pred_poses,pred_rot,image_size,except_joints,joint_sigma,joint_th,ep_sigma):
    pred_rot_rs = pred_rot.reshape(-1, len(all_cams), 3, 3)      
    pred_poses_rs = pred_poses.reshape((-1, len(all_cams), 48))    
    device = inp_poses.device
    batch = 0
    # normalize by scale
    scale_p2d = torch.sqrt(inp_poses[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    # p2d_scaled = inp_poses[:, 0:32]/scale_p2d

    relative_rotations_tensor = torch.empty((len(all_cams),BATCH_SIZE,len(all_cams)-1,3,3)).to(device) # torch.zeros_like()도 가능
    relative_rotations_list = []
    cposes_to_views = []
    for c_cnt in range(len(all_cams)):
        ac = np.array(range(len(all_cams)))
        coi = np.delete(ac, c_cnt)     
        relative_rotations = pred_rot_rs[:, coi].matmul(pred_rot_rs[:, [c_cnt]].permute(0, 1, 3, 2))    # ([b, 3(나머지3개view), 3, 3]) * ([b, 1(하나view), 3, 3]) = ([b, 3, 3, 3])
        relative_rotations_list.append(relative_rotations.cpu().detach().numpy())
        relative_rotations_tensor[c_cnt] = relative_rotations

        # 하나의 canonpose로 각 view로 회전
        canonpose_to_views = pred_rot_rs.matmul(pred_poses_rs.reshape(-1, len(all_cams), 3, 16)[:, c_cnt:c_cnt+1].repeat(1, len(all_cams), 1, 1)).reshape(-1, len(all_cams), 48)
        canonpose_to_views = canonpose_to_views.reshape(-1,48)
        scale_p3d = torch.sqrt(canonpose_to_views[:,0:32].square().sum(axis=1, keepdim=True) / 32)
        reproject_pose = canonpose_to_views[:,0:32]/scale_p3d
        reproject_pose = reproject_pose * scale_p2d
        cposes_to_views.append(reproject_pose.cpu().detach().numpy())

    cposes_to_views = np.transpose(np.array(cposes_to_views), (1,0,2))
    relative_rotations_list = np.transpose(np.array(relative_rotations_list),(1,0,2,3,4))   # (b,4,3,3,3)
    relative_rotations_tensor = relative_rotations_tensor.permute(1,0,2,3,4)    # (b,4,3,3,3)

    # skeleton normalize된 input pose을 homogeneous 좌표계로 만들어준다. [u,v,1]
    inp_skel_norm_pose = inp_poses.reshape(-1,len(all_cams),2,16).cpu().detach().numpy()
    inp_skel_norm_pose = inp_skel_norm_pose*image_size[0]+image_size[0]//2
    inp_skel_norm_pose = np.transpose((inp_skel_norm_pose),(0,1,3,2))         # 이거 나중에 다 torch tensor로 진행해야함


    cposes_to_views = cposes_to_views*image_size[0]+image_size[0]//2

    # Step 1
    # 각 view에서 추정한 3D pose를 각 view별 2D reprojections들을 합침 
    # refine_poses, joints_heatmaps = joints_heatmaps_fusion(all_cams, cposes_to_views, size=image_size[0], sigma=joint_sigma, joint_weights=conf) 
    joints_heatmaps = joints_heatmaps_fusion(all_cams, cposes_to_views, size=image_size[0], sigma=joint_sigma, joint_weights=conf) 
    
    joints_heatmaps = np.array(joints_heatmaps) 

    joints_heatmaps = joints_heatmaps.reshape(-1,len(all_cams),16,image_size[0],image_size[0])
    
    
    
    # Step 2
    p2_3d = inp_poses.reshape(-1,2,16).permute(0,2,1)
    p2_3d_start = torch.cat((p2_3d, torch.tensor([[-3]*len(p2_3d[0])]*len(p2_3d)).reshape(-1,16,1).to(device)),dim=2)
    p2_3d_end = torch.cat((p2_3d, torch.tensor([[3]*len(p2_3d[0])]*len(p2_3d)).reshape(-1,16,1).to(device)),dim=2)
    ray_sample = torch.cat((p2_3d_start, p2_3d_end),dim=2).reshape(-1,len(all_cams),16,2,3).permute(0,1,2,4,3)    # (2, 4, 16, 3, 2)

    # 확인할 joint, 지금은 모든 joints를 epipolar line을 만들어서 heatmap을 만들고 epipolar heatmap에서 conf에 따라 골라 fusion하지만
    # 처음부터 conf가 th이상인 conf만 epipolar line을 만들어 epipolar line heatmap을 선택적으로 만들자  
    joints_lists = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
    
    # Step 3 
    ep_heatmaps,list_fusion_joints_views = epipolarline_heatmap_tensor(all_cams, except_joints, BATCH_SIZE, image_size[0], relative_rotations_tensor, ray_sample, conf, joint_th, sampler=16, sigma=ep_sigma, joints_lists=joints_lists)  # (b, 4, 3, 16, 1000,1000)


    ep_heatmap = ep_heatmaps[batch] # np.sum(ep_heatmaps[0],axis=1)              # (1, 4, 3, 16, 536, 536) --> (4, 16, 536, 536)
    joints_heatmap = joints_heatmaps[batch]
    total_heatmap = joints_heatmap + ep_heatmap         # (4, 16, 536, 536) --> (4, 16, 536, 536)



        
    # view별 joint의 conf가 th보다 낮으면 view별로 refine할 joint를 리스트에 넣는다
    refine_skel_norm_pose = inp_skel_norm_pose.copy()

    for m in range(len(all_cams)):      
        view_heatmap = joints_heatmap[m,list_fusion_joints_views[m],:,:] + ep_heatmap[m,list_fusion_joints_views[m],:,:]
        if len(view_heatmap) > 0:                                
            view_refine_joints = heatmap_to_coord(view_heatmap)
            cnt = 0
            for i in list_fusion_joints_views[m]:
                refine_skel_norm_pose[batch][m][i] = view_refine_joints[cnt]  
                cnt += 1

    return inp_skel_norm_pose, refine_skel_norm_pose, total_heatmap




def CSM(BATCH_SIZE,all_cams,inp_poses,conf,pred_poses,pred_rot,hs,except_joints,joint_sigma,joint_th,ep_sigma):
    pred_rot_rs = pred_rot.reshape(-1, len(all_cams), 3, 3)      
    pred_poses_rs = pred_poses.reshape((-1, len(all_cams), 48))    
    device = inp_poses.device
    batch = 0
    # normalize by scale
    scale_p2d = torch.sqrt(inp_poses[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    # p2d_scaled = inp_poses[:, 0:32]/scale_p2d

    relative_rotations_tensor = torch.empty((len(all_cams),BATCH_SIZE,len(all_cams)-1,3,3)).to(device) # torch.zeros_like()도 가능
    relative_rotations_list = []
    cposes_to_views = []
    for c_cnt in range(len(all_cams)):
        ac = np.array(range(len(all_cams)))
        coi = np.delete(ac, c_cnt)     
        relative_rotations = pred_rot_rs[:, coi].matmul(pred_rot_rs[:, [c_cnt]].permute(0, 1, 3, 2))    # ([b, 3(나머지3개view), 3, 3]) * ([b, 1(하나view), 3, 3]) = ([b, 3, 3, 3])
        relative_rotations_list.append(relative_rotations.cpu().detach().numpy())
        relative_rotations_tensor[c_cnt] = relative_rotations

        # 하나의 canonpose로 각 view로 회전
        canonpose_to_views = pred_rot_rs.matmul(pred_poses_rs.reshape(-1, len(all_cams), 3, 16)[:, c_cnt:c_cnt+1].repeat(1, len(all_cams), 1, 1)).reshape(-1, len(all_cams), 48)
        canonpose_to_views = canonpose_to_views.reshape(-1,48)
        scale_p3d = torch.sqrt(canonpose_to_views[:,0:32].square().sum(axis=1, keepdim=True) / 32)
        reproject_pose = canonpose_to_views[:,0:32]/scale_p3d
        reproject_pose = reproject_pose * scale_p2d
        cposes_to_views.append(reproject_pose.cpu().detach().numpy())

    
    relative_rotations_list = np.transpose(np.array(relative_rotations_list),(1,0,2,3,4))   # (b,4,3,3,3)
    relative_rotations_tensor = relative_rotations_tensor.permute(1,0,2,3,4)    # (b,4,3,3,3)

    # skeleton normalize된 input pose을 homogeneous 좌표계로 만들어준다. [u,v,1]
    inp_skel_norm_pose = inp_poses.reshape(-1,len(all_cams),2,16).permute(0,1,3,2).cpu().detach().numpy()
    inp_skel_norm_pose = inp_skel_norm_pose*hs+hs//2

    cposes_to_views = np.transpose(np.array(cposes_to_views), (1,0,2))
    cposes_to_views = cposes_to_views*hs+hs//2
    
    # Step 1 
    joints_heatmaps = joints_heatmaps_fusion(all_cams, cposes_to_views, size=hs, sigma=joint_sigma, joint_weights=conf) 
    joints_heatmaps = np.array(joints_heatmaps) 
    joints_heatmaps = joints_heatmaps.reshape(-1,len(all_cams),16,hs,hs)


    # Step 2
    p2_3d = inp_poses.reshape(-1,2,16).permute(0,2,1)
    p2_3d_start = torch.cat((p2_3d, torch.tensor([[-3]*len(p2_3d[0])]*len(p2_3d)).reshape(-1,16,1).to(device)),dim=2)
    p2_3d_end = torch.cat((p2_3d, torch.tensor([[3]*len(p2_3d[0])]*len(p2_3d)).reshape(-1,16,1).to(device)),dim=2)
    ray_sample = torch.cat((p2_3d_start, p2_3d_end),dim=2).reshape(-1,len(all_cams),16,2,3).permute(0,1,2,4,3)    # (2, 4, 16, 3, 2)

    # Step 3
    refine_inp_poses = inp_skel_norm_pose.copy()
    joints_lists = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
    ep_heatmaps,refine_inp_poses = refine_module(all_cams, except_joints, BATCH_SIZE, hs, relative_rotations_tensor, ray_sample, 
                                                                        joints_heatmaps, refine_inp_poses, conf, joint_th, sampler=16, sigma=ep_sigma, joints_lists=joints_lists)  # (b, 4, 3, 16, 1000,1000)

    # Step4
    refined_poses = np.transpose(refine_inp_poses.reshape(-1,16,2),(0,2,1)).reshape(-1,32)
    refined_poses = (refined_poses - hs//2) / hs
    refined_poses = torch.tensor(refined_poses).to(device)
 

    return inp_skel_norm_pose, refine_inp_poses, refined_poses, ep_heatmaps


# =========================================================================

def generate_heatmaps_tensor(pose, size, sigma, joint_weights=np.array([[1.]*16])):    # pose : (4,16,2)
    pose = pose.reshape(-1,16,2)                    # (b*4,16,2)
    joint_weights = joint_weights.reshape(-1,16)    # (b*4,16)

    sigma = int(sigma * size / 128)
    # (b*4,16,)
    heatmaps = np.zeros((pose.shape[0], pose.shape[1], size + 2 * sigma, size + 2 * sigma))
    output_heatmaps = np.zeros((pose.shape[0], pose.shape[1], size, size ))
    # joint_weights[:,[2,3,5,6]] *= 1.0
    weights = joint_weights 
    win_size = 2 * sigma + 1

    x, y = np.meshgrid(np.linspace(-sigma, sigma, num=win_size, endpoint=True), np.linspace(-sigma, sigma, num=win_size, endpoint=True)) # joint의 heatmap을 만들크기 (31,31)
    dst = np.sqrt(x * x + y * y)
    mu = 0.000
    gauss = (np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2))))     
          
    for v in range(len(pose)):                  # v는 batch * views 수
        for j, [X, Y] in enumerate(pose[v]):    # i는 joints 수
            X, Y = int(X), int(Y)   
            if X <= 0 or X >= size or Y <= 0 or Y >= size:
                weights[v][j] = 0
                continue
            
            #power = 15 if weights[j][i] > 0.85 else 2
            power=1
            heatmaps[v, j, Y: Y + win_size, X: X + win_size] = gauss**power * weights[v][j]    # joint 위치에서 x,y로 31만큼의 정사각형 위치에 값을 gaussian weight를 넣어줌

        output_heatmaps[v] = heatmaps[v, :, sigma:-sigma, sigma:-sigma]

    return output_heatmaps



def refine_epipolarline_heatmap(before_mpjpe, after_mpjpe, poses_gt, except_joints, inp_poses,inp_confidences, pred_poses, pred_rot, all_cams, hs=96, jsigma=4, epsigma=2, joint_th=0.9):
    
    # except_joints = [0,1,4,7,8,9,10,13] 
    pred_rot_rs = pred_rot.reshape(-1, len(all_cams), 3, 3)
    pred_poses_rs = pred_poses.reshape((-1, len(all_cams), 48)) 
    inp_poses_rs = inp_poses.reshape(-1, len(all_cams), 32)
    device = inp_poses.device
    relative_rotations_tensor = torch.empty((len(all_cams),len(pred_rot_rs),len(all_cams)-1,3,3)).to(device)
    #relative_rotations_list = []

    scale_p2d = torch.sqrt(inp_poses[:, 0:32].square().sum(axis=1, keepdim=True) / 32)
    p2d_scaled = inp_poses[:, 0:32]/scale_p2d
    cposes_to_views = []
    for c_cnt in range(len(all_cams)):
        ac = np.array(range(len(all_cams)))
        coi = np.delete(ac,c_cnt)
        relative_rotations = pred_rot_rs[:, coi].matmul(pred_rot_rs[:, [c_cnt]].permute(0, 1, 3, 2))    # ([b, 3(나머지3개view), 3, 3]) * ([b, 1(하나view), 3, 3]) = ([b, 3, 3, 3])
        #relative_rotations_list.append(relative_rotations.cpu().detach().numpy())
        relative_rotations_tensor[c_cnt] = relative_rotations
        canonpose_to_views = pred_rot_rs.matmul(pred_poses_rs.reshape(-1, len(all_cams), 3, 16)[:, c_cnt:c_cnt+1].repeat(1, len(all_cams), 1, 1)).reshape(-1, len(all_cams), 48)
        canonpose_to_views = canonpose_to_views.reshape(-1,48)
        scale_p3d = torch.sqrt(canonpose_to_views[:,0:32].square().sum(axis=1, keepdim=True) / 32)
        reproject_pose = canonpose_to_views[:,0:32]/scale_p3d
        reproject_pose = reproject_pose * scale_p2d
        cposes_to_views.append(reproject_pose.cpu().detach().numpy())

    relative_rotations_tensor = relative_rotations_tensor.permute(1,0,2,3,4) 
    
    # input pose heatmap 만들기 위해 input pose를 heatmap size로 키운다. (2, 4, 2, 16)
    refine_inp_poses = inp_poses_rs.reshape(-1, len(all_cams), 2, 16).permute(0,1,3,2).cpu().detach().numpy()
    refine_inp_poses = refine_inp_poses*hs + hs // 2
    # conf = inp_confidences.reshape(-1,len(all_cams),16).cpu().detach().numpy()
    conf = inp_confidences.cpu().detach().numpy()
    gt_conf = np.ones(conf.shape)
    cposes_to_views = np.transpose(np.array(cposes_to_views), (1,0,2))  # 이거자체가 추정한 3d를 2d로 reproject했기때문에 자체 bias가 껴있음
    cposes_to_views = cposes_to_views*hs+hs//2
    fusion_heatmap_list = joints_heatmaps_fusion(all_cams,cposes_to_views, size=hs, sigma=jsigma, joint_weights=conf) 


    # refine전 mpjpe
    gt_skel_norm_pose = poses_gt.reshape(-1,len(all_cams),2,16).cpu().detach().numpy()
    gt_skel_norm_pose = gt_skel_norm_pose*hs+hs//2
    gt_skel_norm_pose = np.transpose((gt_skel_norm_pose),(0,1,3,2))

    before_refine_pose = refine_inp_poses.reshape(-1,16,2)
    before_refine_pose_gt = gt_skel_norm_pose.reshape(-1,16,2)
    mpjpe = sum([np.mean(np.sqrt(np.sum(((before_refine_pose[vn] * int(1000/hs)) - (before_refine_pose_gt[vn] * int(1000/hs)))**2, axis=1))) for vn in range(len(before_refine_pose))]) / len(before_refine_pose)
    before_mpjpe.append(mpjpe)
    # print(round(mpjpe,4),'/',round(sum(before_mpjpe)/len(before_mpjpe),4))

    '''
    step1 : normalize된 input_pose를 heatmap크기로 키우고 joint heatmap을 만든다.
    step2 : normalize된 input_pose를 회전해서 epipolar line을 만들기위해 homogeneous 좌표계로 바꾸고 ray화 한다.
    step3 : 각 view별 각 joint별 refine이 필요한 joint를 선별하고 다른 views들의 joint를 epipolar로 만들어 heatmap에 찍는다.
    step4 : 원래 input pose와 epipolar line과 합쳐서 가장 높은 구역 softmax로 joint가 refine된다
    '''

    # step1 
    gt_joints_heatmaps = generate_heatmaps_tensor(gt_skel_norm_pose,hs,jsigma,gt_conf)
    joints_heatmaps = generate_heatmaps_tensor(refine_inp_poses,hs,jsigma,conf)     # (128, 16, 96, 96)
    refine_heatmaps = joints_heatmaps.copy()

    for v in range(len(conf)):
        for jo in range(len(conf[v])):
            if jo in except_joints:
                continue
            if conf[v][jo] < joint_th:
                refine_heatmaps[v][jo] =  fusion_heatmap_list[v][jo]

    refine_heatmaps = refine_heatmaps.reshape(-1,len(all_cams),16,hs,hs)            # (32, 4, 16, 96, 96)

    # step2
    p2_3d = inp_poses.reshape(-1,2,16).permute(0,2,1)
    p2_3d_start = torch.cat((p2_3d, torch.tensor([[-3]*len(p2_3d[0])]*len(p2_3d)).reshape(-1,16,1).to(device)),dim=2)
    p2_3d_end = torch.cat((p2_3d, torch.tensor([[3]*len(p2_3d[0])]*len(p2_3d)).reshape(-1,16,1).to(device)),dim=2)
    ray_sample = torch.cat((p2_3d_start, p2_3d_end),dim=2).reshape(-1,len(all_cams),16,2,3).permute(0,1,2,4,3)    # (32, 4, 16, 3, 2)

    # step3
    ep_heatmaps,refine_inp_poses = epipolarline_heatmap(all_cams, except_joints, len(ray_sample), hs, relative_rotations_tensor, ray_sample, 
                                                                refine_heatmaps, refine_inp_poses, conf, joint_th, sigma=epsigma)




    after_refine_pose = refine_inp_poses.reshape(-1,16,2)
    mpjpe2 = sum([np.mean(np.sqrt(np.sum(((after_refine_pose[mp]*int(1000/hs)) - (before_refine_pose_gt[mp]*int(1000/hs)))**2, axis=1))) for mp in range(len(before_refine_pose))]) / len(before_refine_pose)
    after_mpjpe.append(mpjpe2)
    #print(round(mpjpe2,4),'/',round(sum(after_mpjpe)/len(after_mpjpe),4))
    #print()

    # step4
    refine_inp_poses = np.transpose(refine_inp_poses.reshape(-1,16,2),(0,2,1)).reshape(-1,32)
    refine_inp_poses = (refine_inp_poses - hs//2) / hs
    refine_inp_poses = torch.tensor(refine_inp_poses).to(device)
# --------------------------------------------------모델에 다시 넣는다-----------------------------------------------------------------         
    # 재조정된 confidence는 어떻게 정의하지..?

    return refine_inp_poses, before_mpjpe, after_mpjpe, gt_joints_heatmaps, joints_heatmaps, ep_heatmaps