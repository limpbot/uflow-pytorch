import torch
import unittest
from util import helpers
from .forwardwarp_package.forward_warp import forward_warp

import os
from datasets.datasets import KittiDataset


def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)


def _elementwise_l1(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=1, dim=1, keepdim=True)


def _SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.nn.AvgPool2d(3, 1)(x)
    mu_y = torch.nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = torch.nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = torch.nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

    return torch.nn.functional.pad(SSIM_img, pad=(1, 1, 1, 1), mode="constant", value=0)


def reconstructPts(coord, pts):
    grid = coord.transpose(1, 2).transpose(2, 3)
    pts_warp = torch.nn.functional.grid_sample(pts, grid)

    mask = torch.ones_like(pts, requires_grad=False)
    mask = torch.nn.functional.grid_sample(mask, grid)
    mask = (mask >= 1.0).float()
    return pts_warp * mask


def pts2pixel(pts, intrinsics):
    b, _, h, w = pts.size()
    proj_pts = torch.matmul(intrinsics, pts.view(b, 3, -1))
    pixels_mat = proj_pts.div(proj_pts[:, 2:3, :] + 1e-8)[:, 0:2, :]

    return pixels_mat.view(b, 2, h, w)


def pts2pixel_ms(intrinsic, pts, output_sf, disp_size):

    # +sceneflow and reprojection
    sf_s = torch.nn.functional.interpolate(
        output_sf, disp_size, mode="bilinear", align_corners=True
    )
    pts_tform = pts + sf_s
    coord = pts2pixel(pts_tform, intrinsic)

    norm_coord = coord
    """
    norm_coord_w = coord[:, 0:1, :, :] / (disp_size[1] - 1) * 2 - 1
    norm_coord_h = coord[:, 1:2, :, :] / (disp_size[0] - 1) * 2 - 1
    norm_coord = torch.cat((norm_coord_w, norm_coord_h), dim=1)
    """
    return sf_s, pts_tform, norm_coord


def disp2depth_kitti(pred_disp, k_value):

    pred_depth = (
        k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (pred_disp + 1e-8)
    )
    pred_depth = torch.clamp(pred_depth, 1e-3, 80)

    return pred_depth


def projectSceneFlow2Flow(intrinsic, sceneflow, disp):

    _, _, h, w = disp.size()

    output_depth = disp2depth_kitti(disp, intrinsic[:, 0, 0])
    pts, pixelgrid = pixel2pts(intrinsic, output_depth)

    sf_s = torch.nn.functional.interpolate(
        sceneflow, [h, w], mode="bilinear", align_corners=True
    )
    pts_tform = pts + sf_s
    coord = pts2pixel(pts_tform, intrinsic)
    flow = coord - pixelgrid[:, 0:2, :, :]

    return flow


def get_pixelgrid(b, h, w):
    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)

    ones = torch.ones_like(grid_h)
    pixelgrid = (
        torch.cat((grid_h, grid_v, ones), dim=1).float().requires_grad_(False).cuda()
    )

    return pixelgrid


def pixel2pts(intrinsics, depth):
    b, _, h, w = depth.size()

    pixelgrid = get_pixelgrid(b, h, w)

    depth_mat = depth.view(b, 1, -1)
    pixel_mat = pixelgrid.view(b, 3, -1)
    pts_mat = (
        torch.matmul(torch.inverse(intrinsics.cpu()).cuda(), pixel_mat) * depth_mat
    )
    pts = pts_mat.view(b, -1, h, w)

    return pts, pixelgrid


def _adaptive_disocc_detection(flow):
    # init mask
    (
        b,
        _,
        h,
        w,
    ) = flow.size()
    mask = (
        torch.ones(b, 1, h, w, dtype=flow.dtype, device=flow.device)
        .float()
        .requires_grad_(False)
    )
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1)
    disocc_map = disocc > 0.5

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(
            b, 1, h, w, dtype=torch.bool, device=flow.device
        ).requires_grad_(False)

    return disocc_map


def calc_rec_loss(pts1, pts2, occ_map_f, occ_map_b, sf_f, sf_b):

    _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
    _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp])

    pts2_warp = reconstructPts(
        coord1, pts2
    )  # interpolate grid of pts2 at grid-coords of pts1
    pts1_warp = reconstructPts(coord2, pts1)

    pts1_tf = pts1 + sf_f
    pts2_tf = pts2 + sf_b

    ## Point reconstruction Loss
    pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
    pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
    pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (
        pts_norm1 + 1e-8
    )
    pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (
        pts_norm2 + 1e-8
    )

    loss_pts1 = pts_diff1[occ_map_f].mean()
    loss_pts2 = pts_diff2[occ_map_b].mean()

    # is that required? this happens automatically because it is not part of the loss?
    pts_diff1[~occ_map_f].detach_()
    pts_diff2[~occ_map_b].detach_()
    loss_pts = loss_pts1 + loss_pts2


def warp3d(x, sceneflow, disp, k1, input_size):
    _, _, h_x, w_x = x.size()
    disp = interpolate2d_as(disp, x) * w_x

    local_scale = torch.zeros_like(input_size)
    local_scale[:, 0] = h_x
    local_scale[:, 1] = w_x

    pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
    _, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

    grid = coord1.transpose(1, 2).transpose(2, 3)
    x_warp = torch.nn.functional.grid_sample(x, grid)

    mask = torch.ones_like(x, requires_grad=False)
    mask = torch.nn.functional.grid_sample(mask, grid)
    mask = (mask >= 1.0).float()

    return x_warp * mask


def _gradient_x(img, margin=0):
    # img = tf.pad(img, (0, 1, 0, 0), mode="replicate")
    shift = 1 + margin
    gx = img[:, :, :, :-shift] - img[:, :, :, shift:]  # NCHW
    return gx


def _gradient_y(img, margin=0):
    # img = tf.pad(img, (0, 0, 0, 1), mode="replicate")
    shift = 1 + margin
    gy = img[:, :, :-shift, :] - img[:, :, shift:, :]  # NCHW
    return gy


def _gradient_x_2nd(img):
    img_l = torch.nn.functional.pad(img, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]
    img_r = torch.nn.functional.pad(img, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
    gx = img_l + img_r - 2 * img
    gx = gx[:, :, :, 1:-1]
    # gx = _gradient_x(_gradient_x(img))
    return gx


def _gradient_y_2nd(img):
    img_t = torch.nn.functional.pad(img, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :]
    img_b = torch.nn.functional.pad(img, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :]
    gy = img_t + img_b - 2 * img
    gy = gy[:, :, 1:-1, :]
    # gy = _gradient_y(_gradient_y(img))
    return gy


def _smoothness_motion_2nd(sf, img, beta=1):
    sf_grad_x = _gradient_x_2nd(sf)
    sf_grad_y = _gradient_y_2nd(sf)

    img_grad_x = _gradient_x(img, margin=1)
    img_grad_y = _gradient_y(img, margin=1)
    weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x), 1, keepdim=True) * beta)
    weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y), 1, keepdim=True) * beta)

    smoothness_x = sf_grad_x * weights_x
    smoothness_y = sf_grad_y * weights_y

    return smoothness_x.abs().mean() + smoothness_y.abs().mean()


class Tests(unittest.TestCase):
    def __init__(self, args):
        super().__init__(args)
        self.datasets_dir = "/media/driveD/datasets"
        self.val_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                self.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                self.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                self.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            calibs_dir=os.path.join(
                self.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            disps0_dir=os.path.join(
                self.datasets_dir, "KITTI_flow/training/disp_occ_0"
            ),
            disps1_dir=os.path.join(
                self.datasets_dir, "KITTI_flow/training/disp_occ_1"
            ),
            return_disp=True,
            preload=False,
            dev="cpu",
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, shuffle=False
        )

    def test_calc_photo_loss(self):

        B, C, H, W = (6, 5, 1640, 640)

        img1 = torch.rand(size=(B, C, H, W))
        img2 = torch.rand(size=(B, C, H, W))

        mask = torch.rand(size=(B, 1, H, W))
        mask = mask > 0.5

        loss_photo_pxl_smsf = (
            _elementwise_l1(img1, img2) * (1.0 - 0.85) + _SSIM(img1, img2) * 0.85
        ).mean(dim=1, keepdim=True)
        loss_photo_smsf = (loss_photo_pxl_smsf[mask]).mean()

        # loss_photo_our = helpers.calc_photo_loss(img1, img2, type='self-mono-sf', masks_flow_valid=mask)

        l1_per_pixel = helpers.calc_l1_per_pixel(img1, img2)
        dssim_per_pixel = helpers.calc_ddsim_per_pixel(img1, img2)
        dssim_per_pixel = torch.nn.functional.pad(
            dssim_per_pixel, (1, 1, 1, 1), mode="constant", value=0
        )
        loss_per_pixel = (0.85 * dssim_per_pixel + 0.15 * l1_per_pixel).mean(
            dim=1, keepdim=True
        )

        loss_photo_pxl_diff = torch.abs(loss_photo_pxl_smsf - loss_per_pixel)

        print(torch.max(loss_photo_pxl_diff))

        loss_per_pixel = loss_per_pixel * mask
        loss = torch.sum(loss_per_pixel) / (torch.sum(mask) + 1e-8)

        loss_smsf = loss_photo_pxl_smsf[mask].mean()

        print(loss, loss_smsf)

    def test_forward_warp(self):
        B, C, H, W = (2, 3, 640, 640)

        src = torch.randn(size=(B, C, 3, 3)) * 10
        src = torch.nn.functional.interpolate(src, size=(H, W))
        flow = torch.rand(size=(B, 2, 3, 3)) * 100.0
        flow[:, 0] += 100.0
        flow[:, 1] += 50.0
        flow = torch.nn.functional.interpolate(flow, size=(H, W))
        # flow: the optical flow with shape [B, H, W, 2] (different to grid_sample, it's range is from [-W, -H] to [W, H])
        src_warped_smsf = forward_warp()(
            src.cuda(), -flow.permute([0, 2, 3, 1]).cuda()
        ).cpu()

        src_warped_ours = helpers.warp(src, flow)

        src_warped_diff = src_warped_smsf - src_warped_ours
        src_warped_diff_rel = src_warped_diff / (
            (torch.abs(src_warped_smsf) + torch.abs(src_warped_ours)) / 2.0 + 0.001
        )

        helpers.visualize_img(
            torch.cat((src_warped_ours[0], src_warped_smsf[0]), dim=2)
        )
        print("error", torch.mean(torch.abs(src_warped_diff)))
        print("error_rel", torch.mean(torch.abs(src_warped_diff_rel)))

        ### results
        # note 1: no flip required dx dy order same as ours
        # note 2: somehow it is negative warping compared to ours
        # note 3: warping slightly deviates | align_corners= True/False does not make a clear impact
        pass

    def test_occlusion_map(self):
        B, C, H, W = (2, 3, 640, 640)

        flow_f = torch.rand(size=(B, 2, 3, 3)) * 100.0
        flow_f[:, 0] += 100.0
        flow_f[:, 1] += 50.0
        flow_f = torch.nn.functional.interpolate(flow_f, size=(H, W))

        flow_b = torch.rand(size=(B, 2, 3, 3)) * 1.0
        flow_b[:, 0] -= 10.0
        flow_b[:, 1] -= 5.0
        flow_b = torch.nn.functional.interpolate(flow_b, size=(H, W))

        occ_map_b_smsf = _adaptive_disocc_detection(flow_f.cuda()).cpu().detach()
        occ_map_f_smsf = _adaptive_disocc_detection(flow_b.cuda()).cpu().detach()

        occ_map_ours = helpers.calc_masks_non_occlusion(flow_f, flow_b)
        occ_map_f_ours = occ_map_ours[:B]
        occ_map_b_ours = occ_map_ours[B:]
        helpers.visualize_img(
            torch.cat((occ_map_b_ours[0] * 1.0, occ_map_b_smsf[0] * 1.0), dim=2)
        )

        ### results
        # note1: they are actually forward warping a grid filled with ones where we do kind of the same but with a kernel of 1
        # note2: they return only 1 if less than half of the occ_map only would be 1
        # note3: return 1 if weight > 0.5 else return 0.

        # occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        # occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1
        # note: occ map is multiplication of occ_flow * occ_disp

    def test_calc_projection_sflow(self):
        # sflow :
        # disp :

        for batch_id, (
            imgpairs_left_fwd,
            gt_flow_noc_uv_fwd,
            gt_flow_noc_valid_fwd,
            gt_flow_occ_uv_fwd,
            gt_flow_occ_valid_fwd,
            gt_disps_left_fwd,
            gt_disps_masks_left_fwd,
            gt_disps_left_bwd,
            gt_disps_masks_left_bwd,
            proj_mats_left_fwd,
            reproj_mats_left_fwd,
        ) in enumerate(self.val_dataloader):

            _, _, gt_H, gt_W = gt_flow_noc_uv_fwd.shape
            _, _, H, W = imgpairs_left_fwd.shape
            sx = gt_W / W
            sy = gt_H / H

            proj_mats = proj_mats_left_fwd
            reproj_mats = reproj_mats_left_fwd

            proj_mats, reproj_mats = helpers.rescale_intrinsics(
                proj_mats, reproj_mats, sx, sy
            )

            intrinsics = torch.cat(
                (proj_mats, torch.Tensor([0.0, 0.0, 1.0]).unsqueeze(0).unsqueeze(1)),
                dim=1,
            )
            pts_win1_tp1 = helpers.disp2xyz(gt_disps_left_fwd, proj_mats, reproj_mats)
            pts_win1_tp2 = helpers.disp_oflow2xyz(
                gt_disps_left_bwd, gt_flow_occ_uv_fwd, proj_mats, reproj_mats
            )
            sflow = pts_win1_tp2 - pts_win1_tp1
            disp = gt_disps_left_fwd

            # helpers.visualize_point_cloud(pts1[0])

            oflow_smsf = projectSceneFlow2Flow(
                intrinsics.cuda(), sflow.cuda(), disp.cuda()
            ).cpu()

            oflow_ours = helpers.sflow2oflow(sflow, disp, proj_mats, reproj_mats)
            helpers.visualize_flow(
                torch.cat((oflow_ours[0], oflow_smsf[0], gt_flow_occ_uv_fwd[0]), dim=1)
            )

            helpers.visualize_flow(gt_flow_noc_uv_fwd[0])

            oflow_diff = oflow_smsf - oflow_ours

            print("error oflow:", torch.mean(torch.abs(oflow_diff)))

            ### results
            # note1: sceneflow = (camera + objection) motion

    def test_calc_rec_loss(self):
        B, C, H, W = (2, 3, 640, 640)

        for batch_id, (
            imgpairs_left_fwd,
            gt_flow_noc_uv_fwd,
            gt_flow_noc_valid_fwd,
            gt_flow_occ_uv_fwd,
            gt_flow_occ_valid_fwd,
            gt_disps_left_fwd,
            gt_disps_masks_left_fwd,
            gt_disps_left_bwd,
            gt_disps_masks_left_bwd,
            proj_mats_left_fwd,
            reproj_mats_left_fwd,
        ) in enumerate(self.val_dataloader):

            _, _, gt_H, gt_W = gt_flow_noc_uv_fwd.shape
            _, _, H, W = imgpairs_left_fwd.shape
            sx = gt_W / W
            sy = gt_H / H

            proj_mats = proj_mats_left_fwd
            reproj_mats = reproj_mats_left_fwd

            proj_mats, reproj_mats = helpers.rescale_intrinsics(
                proj_mats, reproj_mats, sx, sy
            )

            intrinsics = torch.cat(
                (proj_mats, torch.Tensor([0.0, 0.0, 1.0]).unsqueeze(0).unsqueeze(1)),
                dim=1,
            )

            intrinsics = torch.cat(
                (proj_mats, torch.Tensor([0.0, 0.0, 1.0]).unsqueeze(0).unsqueeze(1)),
                dim=1,
            )
            pts_win1_tp1 = helpers.disp2xyz(gt_disps_left_fwd, proj_mats, reproj_mats)
            pts_win1_tp2 = helpers.disp_oflow2xyz(
                gt_disps_left_bwd, gt_flow_occ_uv_fwd, proj_mats, reproj_mats
            )
            sflow = pts_win1_tp2 - pts_win1_tp1

            disp = gt_disps_left_fwd

            occ_map_f = torch.rand(size=(1, 1, gt_H, gt_W)) > 0.5

            # self-mono-sf rec loss
            k1_scale = 1.0
            sf_f = sflow
            h_dp = gt_H
            w_dp = gt_W

            # _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts_win1_tp1, sf_f, [h_dp, w_dp])

            # pts_win2_tp2_bwrpd = reconstructPts(coord1, pts_win2_tp2) # interpolate grid of pts2 at grid-coords of pts1
            # pts_win1_tp1_ftf = pts_win1_tp1 + sf_f

            ## Point reconstruction Loss
            pts_norm1 = torch.norm(pts_win1_tp1, p=2, dim=1, keepdim=True)
            pts_diff1 = _elementwise_epe(pts_win1_tp1, pts_win1_tp2).mean(
                dim=1, keepdim=True
            ) / (pts_norm1 + 1e-8)

            # print('norm1', torch.sum(pts_norm1))
            # print('epe1', torch.sum(_elementwise_epe(pts_win1_tp1, pts_win1_tp2)))

            loss_pts1 = pts_diff1[occ_map_f].mean()

            # is that required? this happens automatically because it is not part of the loss?
            pts_diff1[~occ_map_f].detach_()
            loss_pts_v1 = loss_pts1

            # pts_win1_tp1 = pts_win1_tp1.repeat([2, 1, 1, 1])
            # pts_win1_tp2 = pts_win1_tp2.repeat([2, 1, 1, 1])
            # occ_map_f = occ_map_f.repeat([2, 1, 1, 1])

            loss_pts_v2 = helpers.calc_reconstruction_loss(
                pts_win1_tp1, pts_win1_tp1, pts_win1_tp2, occ_map_f
            )

            print("l1", loss_pts_v1, "l2", loss_pts_v2)

    def test_warp3d(self):
        pass

    def test_reproj(self):
        B, C, H, W = (2, 3, 640, 640)

        for batch_id, (
            imgpairs_left_fwd,
            gt_flow_noc_uv_fwd,
            gt_flow_noc_valid_fwd,
            gt_flow_occ_uv_fwd,
            gt_flow_occ_valid_fwd,
            gt_disps_left_fwd,
            gt_disps_masks_left_fwd,
            gt_disps_left_bwd,
            gt_disps_masks_left_bwd,
            proj_mats_left_fwd,
            reproj_mats_left_fwd,
        ) in enumerate(self.val_dataloader):
            _, _, gt_H, gt_W = gt_flow_noc_uv_fwd.shape
            _, _, H, W = imgpairs_left_fwd.shape
            sx = gt_W / W
            sy = gt_H / H

            proj_mats = proj_mats_left_fwd
            reproj_mats = reproj_mats_left_fwd

            proj_mats, reproj_mats = helpers.rescale_intrinsics(
                proj_mats, reproj_mats, sx, sy
            )

            intrinsics = torch.cat(
                (proj_mats, torch.Tensor([0.0, 0.0, 1.0]).unsqueeze(0).unsqueeze(1)),
                dim=1,
            )
            pts_win1_tp1 = helpers.disp2xyz(gt_disps_left_fwd, proj_mats, reproj_mats)
            pts_win1_tp1[:, 2] = pts_win1_tp1[:, 2] * 0.0 - 0e-8

            pts_win1_tp2 = helpers.disp_oflow2xyz(
                gt_disps_left_bwd, gt_flow_occ_uv_fwd, proj_mats, reproj_mats
            )
            sflow = pts_win1_tp2 - pts_win1_tp1
            disp = gt_disps_left_fwd

            _, pts1_tf, coord1_smsf = pts2pixel_ms(
                intrinsics, pts_win1_tp1, sflow * 0.0, [gt_H, gt_W]
            )

            coord1_our = helpers.xyz2uv(pts_win1_tp1, proj_mats=proj_mats)

            pass

    def test_disp_smoothness_loss(self):
        B = 2
        C = 5
        H = 280
        W = 640
        img = torch.rand(size=(B, 3, H, W))
        disp = torch.rand(size=(B, 3, H, W))

        ii = 0
        loss_smooth_smsf = _smoothness_motion_2nd(disp, img, beta=10.0).mean() / (
            2 ** ii
        )

        disp = disp * W

        # caus gx+gy / 2.   =>     2. *
        # in code cause stride=2     =>  beta=20
        # in code cause disp1, disp2 => 2. *
        loss_smooth = (
                2.0
                * helpers.calc_smoothness_loss(flow=disp, img1=img, edge_weight=10, order=2)
                / W
        )

        print(loss_smooth, loss_smooth_smsf)


if __name__ == "__main__":
    unittest.main()
