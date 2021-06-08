import torch
import numpy as np
from tensor_operations.geometric import pts3d_transform
import pytorch3d.transforms as t3d



def dist_transls(transl1, transl2):

    dist = torch.norm((transl1 - transl2), dim=-1)

    return dist


def angle_rots(rot1, rot2):

    rot = torch.matmul(rot1.permute(0, 2, 1), rot2)
    #print("rot", rot)
    for i in range(len(rot)):
        trace = torch.trace(rot[i, :3, :3])
        if trace > 3.1 or trace <= -1:
            print('trace', torch.trace(rot[i, :3, :3]))
            print(rot[i, :3, :3])

    try:
        rot_logs = t3d.so3_log_map(rot[:, :3, :3])
    except:
        print('failed')
        import time
        time.sleep(2)
    angle = torch.norm(rot_logs, dim=1)

    return angle


def dist_angle_transfs(transf1, transf2):

    rot1 = transf1[:, :3, :3]
    transl1 = transf1[:, :3, 3]

    rot2 = transf2[:, :3, :3]
    transl2 = transf2[:, :3, 3]

    angle = angle_rots(rot1, rot2)
    dist = dist_transls(transl1, transl2)

    return dist, angle

def calc_pointset_registration(pts1, pts2):
    # mask_valid = (pts1[2] < 25.) * (pts2[2] < 25.)
    # pts1 = pts1[:, mask_valid].detach()
    # pts2 = pts2[:, mask_valid].detach()
    device = pts1.device
    dtype = pts1.dtype

    global_transform = torch.eye(4, dtype=pts1.dtype, device=pts1.device)
    transform = torch.eye(4, dtype=pts1.dtype, device=pts1.device)
    pts1_ftf = pts1.clone()
    for i in range(1):

        # pts1, pts2: 3 x N
        centroid_pts1 = torch.mean(pts1_ftf, dim=1)
        centroid_pts2 = torch.mean(pts2, dim=1)

        pts1_norm = pts1_ftf - centroid_pts1.unsqueeze(
            1
        ) # / torch.norm(pts1, dim=0).unsqueeze(0)**2
        pts2_norm = pts2 - centroid_pts2.unsqueeze(
            1
        ) # / torch.norm(pts1, dim=0).unsqueeze(0)**2

        U, S, V = torch.svd(torch.matmul(pts2_norm, pts1_norm.T))

        #M = torch.diag(torch.Tensor([1., 1., 1.]).to(device))
        M = torch.diag(torch.Tensor([1., 1., torch.det(U) * torch.det(V)]).to(device))

        rot = torch.matmul(U, torch.matmul(M, V.T))

        rot = rot.detach().cpu().numpy()

        transl = centroid_pts2.detach().cpu().numpy() - np.dot(
            rot, centroid_pts1.detach().cpu().numpy()
        )

        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = transl

        transform = torch.from_numpy(transform.astype(np.float32)).to(pts1.device)

        global_transform = torch.matmul(transform, global_transform)
        pts1_ftf = (
            pts3d_transform(
                pts1.permute(1, 0).unsqueeze(0), global_transform.unsqueeze(0)
            )
            .squeeze(0)
            .permute(1, 0)
        )

        epe = torch.norm(pts1_ftf - pts2, dim=0)
        N = pts1.shape[1]
        epe_ind = torch.argsort(epe)
        # pts1_ftf = pts1_ftf[:, epe_ind[:int(0.95 * N)]]
        # pts1 = pts1[:, epe_ind[:int(0.95 * N)]]
        # pts2 = pts2[:, epe_ind[:int(0.95 * N)]]

    return global_transform

def filter_sim_se3(objects_se3s, objects_se3s_devs, dist_thresh, angle_thresh):
    K = len(objects_se3s)

    objects_se3s_dists, objects_se3s_angles = dist_angle_transfs(objects_se3s.repeat(K, 1, 1),
                                                                   objects_se3s.repeat_interleave(K, dim=0))
    objects_se3s_dists = objects_se3s_dists.reshape(K, K)
    objects_se3s_angles = objects_se3s_angles.reshape(K, K)

    #print('dists', objects_se3s_dists)
    #print('angles', objects_se3s_angles)

    objects_se3s_sim = (objects_se3s_dists < dist_thresh) * (objects_se3s_angles < angle_thresh)

    min_devs_ids = objects_se3s_devs.argsort()
    core_se3s_filtered = []
    unselected = torch.zeros_like(min_devs_ids) == 0

    for id in min_devs_ids:
        if unselected[id]:
            core_se3s_filtered.append(objects_se3s[id])
            unselected[objects_se3s_sim[id]] = False

    objects_se3s = core_se3s_filtered
    objects_se3s = torch.stack(objects_se3s)

    return objects_se3s

def register_objects(objects_masks, pts1, pts2, thresh=0.1):
    K = len(objects_masks)
    objects_se3s = []
    objects_se3s_devs = []

    for k in range(K):
        core_mask = objects_masks[k]
        se3 = calc_pointset_registration(pts1[:, core_mask],
                                         pts2[:, core_mask])

        pts1_ftf = pts3d_transform(pts1[None, ], se3[None,])
        dev = torch.norm(pts1_ftf - pts2[None, ], dim=1, keepdim=True)[0, 0, core_mask]
        dev_mean = dev.mean()
        if dev_mean < thresh:
            objects_se3s_devs.append(dev_mean)
            objects_se3s.append(se3)
        # ops_vis.visualize_img(core_mask[None,])

    objects_se3s = torch.stack(objects_se3s)
    objects_se3s_devs = torch.stack(objects_se3s_devs)

    return objects_se3s, objects_se3s_devs