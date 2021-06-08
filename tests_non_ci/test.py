import unittest
import os
import torch
import tensorflow as tf
from torchvision.transforms import ToPILImage
import cv2
import numpy as np


from util.helpers import (
    rgb_to_grayscale,
    neighbors_to_channels,
    census_transform,
    calc_census_loss,
    warp,
    calc_masks_non_occlusion,
    visualize_img,
)

from coach import Coach
from datasets.datasets import KittiDataset

from util import helpers

from usflow import options

parser = options.setup_comon_options()

preliminary_args = [
    "-s",
    "../config/config_setup_0.yaml",
    "-c",
    "../config/config_coach_uflow_0.yaml",
]
args = parser.parse_args(preliminary_args)


class Tests(unittest.TestCase):
    def test_network(self):
        uflow = UFlow()
        print(uflow.dev)
        uflow.cuda()
        self.assertTrue(uflow)

        x_in = torch.rand((12, 2 * 3, 512, 512))
        x_out = uflow(x_in.cuda())

        self.assertTrue(x_out is not None)

    def test_rgb_to_grayscale(self):

        B = 12
        H = 512
        W = 512
        x_in = torch.rand((B, 3, H, W))

        x_out = rgb_to_grayscale(x_in.cuda())

        self.assertTrue(x_out.size() == torch.Size([B, 1, H, W]))

    def test_neighbors_to_channels(self):
        B = 12
        C = 3
        H = 512
        W = 512
        x_in = torch.rand((B, C, H, W)).cuda()

        P = 3

        x_out = neighbors_to_channels(x_in, P)

        for i in range(C):
            self.assertTrue(torch.equal(x_in[:, i], x_out[:, 4 + i * 9]))

    def test_census_transform(self):
        B = 12
        C = 1
        H = 512
        W = 512

        x_in = torch.rand((B, C, H, W)).cuda()

        x_out = census_transform(x_in, patch_size=7)

        self.assertTrue(x_out.size() == torch.Size([B, 49, H, W]))

    def test_census_loss(self):
        B = 12
        C = 3
        H = 512
        W = 512

        x1_in = torch.rand((B, C, H, W)).cuda()
        x2_in = torch.rand((B, C, H, W)).cuda()

        x_out = calc_census_loss(x1_in, x2_in, patch_size=7)

        self.assertTrue(x_out.size() == torch.Size([]))

    def test_kitti_dataset(self):
        kitti_dataset = KittiDataset(
            "../datasets/KITTI_flow_multiview/training/image_2",
            max_num_imgs=10,
            preload=True,
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=True
        )

        for i, imgpair in enumerate(kitti_dataloader):
            print(i, imgpair.size())

            img1 = imgpair[0, :3, :, :]
            img2 = imgpair[0, 3:, :, :]

            imgs = torch.cat((img1, img2), dim=2).permute(1, 2, 0)
            # Hx2*Wx3

            cv2.imshow("abc", imgs.cpu().numpy()[:, :, ::-1])
            cv2.waitKey(0)

    def test_visualize_flow(self):

        kitti_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            calibs_dir=os.path.join(
                "..",
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            disps_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/disp_noc_0"
            ),
            return_disp=True,
            preload=False,
            dev="cpu",
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=False
        )

        for batch_id, (
            imgpairs_left_fwd,
            gt_flow_noc_uv_fwd,
            gt_flow_noc_valid_fwd,
            gt_flow_occ_uv_fwd,
            gt_flow_occ_valid_fwd,
            gt_disps_left_img1,
            gt_disps_masks_left_fwd,
            proj_mats_left_fwd,
            reproj_mats_left_fwd,
        ) in enumerate(kitti_dataloader):

            helpers.visualize_flow(gt_flow_occ_uv_fwd[0], draw_arrows=True)

            img = helpers.flow2rgb(gt_flow_occ_uv_fwd[0])

            start, end = helpers.flow2startendpoints(gt_flow_occ_uv_fwd[0])

            img_arrows = helpers.draw_arrows_in_rgb(img, start, end)

            # helpers.visualize_img(img_arrows)

    def test_warp3d(self):

        kitti_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            calibs_dir=os.path.join(
                "..",
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            disps_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/disp_noc_0"
            ),
            return_disp=True,
            preload=False,
            dev="cpu",
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=False
        )

        num_batches = len(kitti_dataloader)

        metrics_acc = {}

        for batch_id, (
            imgpairs_left_fwd,
            gt_flow_noc_uv_fwd,
            gt_flow_noc_valid_fwd,
            gt_flow_occ_uv_fwd,
            gt_flow_occ_valid_fwd,
            gt_disps_left_img1,
            gt_disps_masks_left_fwd,
            proj_mats_left_fwd,
            reproj_mats_left_fwd,
        ) in enumerate(kitti_dataloader):

            B, _, gt_H, gt_W = gt_disps_left_img1.shape

            fx = proj_mats_left_fwd[:, 0, 0]
            print("rel_fx =", fx / 640.0)

            _, _, H, W = imgpairs_left_fwd.shape

            sx = gt_W / W
            sy = gt_H / H

            imgpairs_left_fwd = torch.nn.functional.interpolate(
                imgpairs_left_fwd,
                size=(gt_H, gt_W),
                mode="bilinear",
                align_corners=False,
            )

            # gt_disps_left_img1 = torch.nn.functional.interpolate(gt_disps_left_img1, size=(H, W), mode='bilinear', align_corners=False)
            # gt_disps_left_img1 = gt_disps_left_img1 / sx

            # gt_flow_occ_uv_fwd = torch.nn.functional.interpolate(gt_flow_occ_uv_fwd, size=(H, W), mode='bilinear',
            #                                          align_corners=False)
            # gt_flow_occ_uv_fwd[:, 0] = gt_flow_occ_uv_fwd[:, 0] / sx
            # gt_flow_occ_uv_fwd[:, 1] = gt_flow_occ_uv_fwd[:, 1] / sy

            proj_mats_left_fwd[:, 0] = proj_mats_left_fwd[:, 0] * sx
            proj_mats_left_fwd[:, 1] = proj_mats_left_fwd[:, 1] * sy

            reproj_mats_left_fwd[:, :, 0] = reproj_mats_left_fwd[:, :, 0] / sx
            reproj_mats_left_fwd[:, :, 1] = reproj_mats_left_fwd[:, :, 1] / sy

            """
            reproj_mats_left_fwd
            tensor([[[ 1.3771,  0.0000, -0.8448],
                     [ 0.0000,  0.4158, -0.2396],
                     [ 0.0000,  0.0000,  1.0000]]], device='cuda:0')
            proj_mats_left_fwd
            tensor([[[ 371.8069,    0.0000,  314.1046],
                     [   0.0000, 1231.4243,  295.0042]]], device='cuda:0')
            """

            gt_depths_left_img1 = helpers.disp2depth(
                gt_disps_left_img1, proj_mats_left_fwd[:, 0, 0]
            )
            gt_points3d_left_img1 = helpers.depth2xyz(
                gt_depths_left_img1, reproj_mats_left_fwd
            )
            sflow = torch.rand(size=(B, 3, gt_H, gt_W))
            sflow[:, 0] = sflow[:, 0] * 0.0 - 0.01
            sflow[:, 1] = sflow[:, 1] * 0.0 - 0.01
            sflow[:, 2] = sflow[:, 2] * 0.0  # - 2.0

            gt_points3d_left_img1_fwdwrpd = gt_points3d_left_img1 + sflow

            # helpers.visualize_point_cloud(gt_points3d_left_img1_fwdwrpd[0], mask=gt_disps_masks_left_fwd[0])

            gt_pxlcoords_left_img1_fwdwrpd = helpers.xyz2uv(
                gt_points3d_left_img1_fwdwrpd, proj_mats_left_fwd
            )

            oflow = helpers.pxlcoords2flow(gt_pxlcoords_left_img1_fwdwrpd)

            imgpairs_left_fwd_wrpd3d = helpers.interpolate2d(
                imgpairs_left_fwd, gt_pxlcoords_left_img1_fwdwrpd
            )

            helpers.visualize_flow(oflow[0], draw_arrows=True)

            imgpairs_left_fwd_wrpd2d = helpers.warp(imgpairs_left_fwd[:, :], oflow)
            imgpairs_left_fwd_wrpd3d = helpers.warp3d(
                imgpairs_left_fwd,
                sflow,
                gt_disps_left_img1,
                proj_mats_left_fwd,
                reproj_mats_left_fwd,
            )

            helpers.visualize_img(
                torch.cat(
                    (imgpairs_left_fwd_wrpd2d[0, :3], imgpairs_left_fwd_wrpd3d[0, :3]),
                    dim=1,
                )
            )
            print(torch.mean(imgpairs_left_fwd_wrpd2d - imgpairs_left_fwd_wrpd3d))

    def test_warping(self):
        print("asdfasdfasdf")
        kitti_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            preload=False,
            dev="cpu",
            max_num_imgs=args.val_dataset_max_num_imgs,
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=False
        )

        pil_transform = ToPILImage()
        for i, (
            imgpairs_forward,
            gt_flow_noc_uv_forward,
            gt_flow_noc_valid_forward,
            gt_flow_occ_uv_forward,
            gt_flow_occ_valid_forward,
        ) in enumerate(kitti_dataloader):

            imgpair = imgpairs_forward
            gt_flow_uv = gt_flow_noc_uv_forward
            gt_flow_valid = gt_flow_noc_valid_forward

            img1 = imgpair[:, :3, :, :]
            img2 = imgpair[:, 3:, :, :]

            B, C, H, W = gt_flow_uv.size()
            dev = gt_flow_uv.device
            dtype = gt_flow_uv.dtype

            # gt_flow_uv = 100.0 * torch.ones((B, C, H, W), device=dev, dtype=dtype)
            # gt_flow_uv[:, 0, :, :] = 0.0

            orig_height = gt_flow_uv.size(2)
            orig_width = gt_flow_uv.size(3)

            img1 = torch.nn.functional.interpolate(
                img1,
                size=(orig_height, orig_width),
                mode="bilinear",
                align_corners=False,
            )

            img2 = torch.nn.functional.interpolate(
                img2,
                size=(orig_height, orig_width),
                mode="bilinear",
                align_corners=False,
            )
            print("img1.size()", img1.size())
            print("gt_flow_uv.size()", gt_flow_uv.size())

            # mask = gt_flow_noc_valid_forward[0]
            # gt_flow_uv[:, 1] = gt_flow_uv[:, 1] + 100.
            # gt_flow_uv = gt_flow_uv.unsqueeze(1)

            tf_coords = uflow_helpers.flow_to_warp(
                self.torch_to_tf(torch.flip(gt_flow_uv, [1]))
            )
            # tf_coords = uflow_helpers.flow_to_warp(self.torch_to_tf(gt_flow_uv))
            # TODO: tensorflow check grid sample (tensorflow implementation)

            tf_img2_warped = uflow_helpers.resample(
                source=self.torch_to_tf(img2), coords=tf_coords
            )
            # tf_img2_warped = uflow_helpers.grid_sample_2d(self.torch_to_tf(img2), tf_coords)
            tf_img2_warped = self.tf_to_torch(tf_img2_warped, dev=img2.device)

            tf_coords = self.tf_to_torch(tf_coords, dev=img2.device)
            tf_coords = torch.flip(tf_coords, [1])

            coords = helpers.oflow2pxlcoords(gt_flow_uv)

            img2_warped, masks_flow_inside = warp(
                img2, gt_flow_uv, return_masks_flow_inside=True
            )

            img2s_warped = torch.nn.functional.interpolate(
                torch.cat((tf_img2_warped[0], img2_warped[0]), dim=2).unsqueeze(0),
                size=(480, 640),
            )
            helpers.visualize_img(img2s_warped[0])

            helpers.visualize_img(img2_warped[0] - tf_img2_warped[0])
            """
            grid = helpers.flow2coords(gt_flow_uv)
            grid[:, 0] = grid[:, 0] / orig_width * 2.0 - 1.
            grid[:, 1] = grid[:, 1] / orig_height * 2.0 - 1.
            mask = torch.ones_like(img2_warped, requires_grad=False)
            mask = torch.nn.functional.grid_sample(mask, grid.permute(0, 2, 3, 1))
            mask = (mask >= 1.0).float()

            #img2_warped_v1 = warp(img2, gt_flow_uv)
            #img2_warped_v2 = warp(img2, gt_flow_uv) * mask
            #helpers.visualize_img(img2_warped_v1[0] - img2_warped_v2[0])

            #imgs = torch.cat((img1[0], img2[0]), dim=1)
            imgs_warped = torch.cat((img1[0], img2_warped[0] * gt_flow_valid[0]), dim=1)
            imgs_warped = torch.cat((img1[0], img2_warped[0] * gt_flow_valid[0]), dim=1)

            helpers.visualize_flow(gt_flow_uv[0])
            helpers.visualize_img(imgs_warped)
            helpers.visualize_img(masks_flow_inside[0])

            #img = torch.cat((imgs, imgs_warped), dim=2)
            #img = imgs_warped
            #pil_transform(img.cpu()).show()
            #input('press key to continue')
            """

        """'
        mask = torch.ones_like(img, requires_grad=False)
        mask = tf.grid_sample(mask, grid)
        mask = (mask >= 1.0).float()
        return img_warp * mask
        """

    def test_coach(self):

        coach = Coach(model_tag=None)
        #'2020_08_12_v64'
        coach.run()

    def test_flow2coords(self):
        uflow = UFlow()
        uflow.cuda()

        model_dir = os.path.join("models", "2020_08_25_v2")
        uflow.load_state_dict(
            torch.load(os.path.join(model_dir, args.model_state_dict_name))
        )

        uflow.eval()

        kitti_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            preload=False,
            max_num_imgs=1,
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=False
        )

        for i, (
            imgpairs_forward,
            gt_flow_noc_uv_forward,
            gt_flow_noc_valid_forward,
            gt_flow_occ_uv_forward,
            gt_flow_occ_valid_forward,
        ) in enumerate(kitti_dataloader):
            print(i, ":", len(kitti_dataloader))

            flow = gt_flow_noc_uv_forward
            orig_flow = torch.flip(flow, [1])

            orig_coords = self.tf_to_torch(
                uflow_helpers.flow_to_warp(self.torch_to_tf(orig_flow)), dev=flow.device
            )
            orig_coords = torch.flip(orig_coords, [1])
            our_coords = helpers.flow2coords(
                self.tf_to_torch(self.torch_to_tf(flow), dev=flow.device)
            )
            diff_coords = orig_coords - our_coords
            helpers.visualize_hist(diff_coords)
            pass

    def torch_to_tf(self, x, device="cpu"):

        x = tf.convert_to_tensor(x.permute(0, 2, 3, 1).cpu().detach().numpy())
        return x

    def tf_to_torch(self, x, dev=None):
        x = torch.from_numpy(x.numpy()).permute(0, 3, 1, 2)
        if dev is not None:
            x = x.to(device=dev)

        return x

    def test_non_occlusion_masks(self):
        mask_type = "wang"
        uflow = UFlow()
        uflow.cuda()

        model_dir = os.path.join("..", "models", "2020_10_26_v5")
        uflow.load_state_dict(
            torch.load(os.path.join(model_dir, args.model_state_dict_name))
        )

        uflow.eval()

        kitti_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            preload=False,
            max_num_imgs=5,
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=2, shuffle=False
        )

        for i, (
            imgpairs_forward,
            gt_flow_noc_uv_forward,
            gt_flow_noc_valid_forward,
            gt_flow_occ_uv_forward,
            gt_flow_occ_valid_forward,
        ) in enumerate(kitti_dataloader):
            print(i, ":", len(kitti_dataloader))

            img1s_forward = imgpairs_forward[:, :3, :, :]
            img2s_forward = imgpairs_forward[:, 3:, :, :]

            imgpairs_backward = torch.cat((img2s_forward, img1s_forward), dim=1)

            imgpairs = torch.cat((imgpairs_forward, imgpairs_backward), dim=0)

            flow_uv = uflow(imgpairs)
            # flow_uv = torch.zeros(flow_uv.size(), device=uflow.dev)

            height = flow_uv.size(2)
            width = flow_uv.size(3)
            orig_height = gt_flow_noc_uv_forward.size(2)
            orig_width = gt_flow_noc_uv_forward.size(3)
            num_imgpairs_forward = imgpairs_forward.size(0)

            flow_uv_up = (
                torch.nn.functional.interpolate(
                    flow_uv, scale_factor=4.0, mode="bilinear", align_corners=False
                )
                * 4.0
            )

            flow_uv_orig = torch.nn.functional.interpolate(
                flow_uv,
                size=(orig_height, orig_width),
                mode="bilinear",
                align_corners=False,
            )
            flow_uv_orig[:, 0] = flow_uv_orig[:, 0] * orig_width / width
            flow_uv_orig[:, 1] = flow_uv_orig[:, 1] * orig_height / height

            flow_uv_up_forward = flow_uv_up[:num_imgpairs_forward]
            flow_uv_up_backward = flow_uv_up[num_imgpairs_forward:]

            flow_uv_orig_forward = flow_uv_orig[:num_imgpairs_forward]
            flow_uv_orig_backward = flow_uv_orig[num_imgpairs_forward:]

            if mask_type == "wang":
                # flow_uv_up_backward
                # flow_uv_up_backward = torch.nn.functional.interpolate(flow_uv_up_backward, size=(int(width/4.), int(height/4.)), mode='bilinear',
                #                                               align_corners=False) / 4.
                # flow_uv_up_backward[:, 1] = 0.
                our_non_occlusion_mask = helpers.calc_masks_non_occlusion(
                    flow_uv_up_forward,
                    flow_uv_up_backward,
                    return_forward_and_backward=False,
                )

                orig_flow_backward = torch.flip(flow_uv_up_backward, [1])
                range_map = uflow_helpers.compute_range_map(
                    self.torch_to_tf(orig_flow_backward),
                    downsampling_factor=1,
                    reduce_downsampling_bias=False,
                    resize_output=False,
                )
                # Invert so that low values correspond to probable occlusions,
                # range [0, 1].
                orig_non_occlusion_mask = 1.0 - (
                    1.0 - tf.clip_by_value(range_map, 0.0, 1.0)
                )
                orig_non_occlusion_mask = self.tf_to_torch(
                    orig_non_occlusion_mask, dev=imgpairs.device
                )

                non_occlusion_masks = torch.cat(
                    (orig_non_occlusion_mask, our_non_occlusion_mask), dim=3
                )

                visualize_img(non_occlusion_masks[0])

            elif mask_type == "brox":
                orig_flow_forward = self.torch_to_tf(
                    torch.flip(flow_uv_up_forward, [1])
                )
                orig_flow_backward = self.torch_to_tf(
                    torch.flip(flow_uv_up_backward, [1])
                )
                flow_ji_in_i = uflow_helpers.resample(
                    orig_flow_backward, uflow_helpers.flow_to_warp(orig_flow_forward)
                )
                fb_sq_diff = tf.reduce_sum(
                    input_tensor=(orig_flow_forward + flow_ji_in_i) ** 2,
                    axis=-1,
                    keepdims=True,
                )
                fb_sum_sq = tf.reduce_sum(
                    input_tensor=(orig_flow_forward ** 2 + flow_ji_in_i ** 2),
                    axis=-1,
                    keepdims=True,
                )
                orig_non_occlusion_mask = tf.cast(
                    fb_sq_diff <= 0.01 * fb_sum_sq + 0.5, tf.float32
                )
                orig_non_occlusion_mask = self.tf_to_torch(
                    orig_non_occlusion_mask, dev=flow_uv_up_forward.device
                )
                our_non_occlusion_mask = calc_masks_non_occlusion(
                    flow_uv_up_forward,
                    flow_uv_up_backward,
                    return_masks_flow_inside=False,
                    mask_type="brox",
                )
                non_occlusion_masks = torch.cat(
                    (orig_non_occlusion_mask, our_non_occlusion_mask), dim=3
                )
                visualize_img(non_occlusion_masks[0])

    def test_dataloader(self):
        kitti_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            imgs_right_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/image_3"
            ),
            return_left_and_right=True,
            flows_noc_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            disps_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/disp_occ_0"
            ),
            return_disp=True,
            calibs_dir=os.path.join(
                "..",
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            preload=False,
            max_num_imgs=100,
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=False
        )

        flow_avgs = []
        for i, (
            imgpairs_left_forward,
            imgpairs_right_forward,
            gt_flow_noc_uv_forward,
            gt_flow_noc_valid_forward,
            gt_flow_occ_uv_forward,
            gt_flow_occ_valid_forward,
            gt_disp,
            gt_disp_mask,
            projection_matrix,
            reprojection_matrix,
        ) in enumerate(kitti_dataloader):

            helpers.visualize_imgpair(imgpairs_left_forward[0])

            from photo_transform import PhotoTransform

            photo_transform = PhotoTransform()
            imgpairs_left_forward[0] = photo_transform(imgpairs_left_forward[0])

            helpers.visualize_imgpair(imgpairs_left_forward[0])

            print(torch.mean(torch.abs(gt_flow_occ_uv_forward)))
            flow_avgs.append(torch.mean(torch.abs(gt_flow_occ_uv_forward)).item())
            # flow_rgb = helpers.flow2rgb(gt_flow_occ_uv_forward[0])
            # helpers.visualize_img(flow_rgb)
            # helpers.visualize_img(imgpairs_left_forward[0, :3])
            # helpers.visualize_img(gt_flow_occ_uv_forward[0, 0].unsqueeze(0) / 100.)
            # helpers.visualize_img(gt_flow_occ_uv_forward[0, 1].unsqueeze(0) / 100.)
            self.assertTrue(imgpairs_left_forward is not None)
            self.assertTrue(imgpairs_right_forward is not None)
            self.assertTrue(gt_flow_noc_uv_forward is not None)
            self.assertTrue(gt_flow_noc_valid_forward is not None)
            self.assertTrue(gt_flow_occ_uv_forward is not None)
            self.assertTrue(gt_flow_occ_valid_forward is not None)
            self.assertTrue(gt_disp is not None)
            self.assertTrue(gt_disp_mask is not None)
            self.assertTrue(projection_matrix is not None)
            self.assertTrue(reprojection_matrix is not None)

            """
            imgs_left_right_1 = torch.cat((imgpairs_left_forward[:, :3], imgpairs_right_forward[:, :3]), dim=3)
            imgs_left_right_2 = torch.cat((imgpairs_left_forward[:, 3:], imgpairs_right_forward[:, 3:]), dim=3)
            imgs = torch.cat((imgs_left_right_1, imgs_left_right_2), dim=2)
            imgs = torch.nn.functional.interpolate(imgs, size=(480, 640))
            helpers.visualize_img(imgs[0])
            """

            # B, _, H, W = gt_disp.shape
            # dtype = gt_disp.dtype
            # dev = gt_disp.device

            # gt_depth = helpers.disp2depth(gt_disp, projection_matrix[0][0][0])
            # xyz = helpers.depth2xyz(gt_depth[0], reprojection_matrix[0])

            """
            flow = torch.zeros(size=(B, 2, H, W), dtype=dtype, device=dev)
            xy = helpers.flow2coords(flow)[0]
            z = helpers.disp2depth(gt_disp[0], fx=projection_matrix[0][0][0])
            xyz = torch.cat((xy, z), dim=0)
            """
        print("flow-avg", sum(flow_avgs) / len(flow_avgs))

        # imgs_left = torch.nn.functional.interpolate(imgpairs_left_forward[:, :3], size=(H, W))

        # helpers.visualize_point_cloud(xyz, rgb=imgs_left[0], mask=gt_disp_mask[0])

    def test_transformation2flow(self):

        kitti_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            imgs_right_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/image_3"
            ),
            return_left_and_right=True,
            flows_noc_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            disps_dir=os.path.join(args.datasets_dir, "KITTI_flow/training/disp_occ_0"),
            return_disp=True,
            calibs_dir=os.path.join(
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            preload=False,
            max_num_imgs=10,
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=False
        )

        for i, (
            imgpairs_left_forward,
            imgpairs_right_forward,
            gt_flow_noc_uv_forward,
            gt_flow_noc_valid_forward,
            gt_flow_occ_uv_forward,
            gt_flow_occ_valid_forward,
            gt_disp,
            gt_disp_mask,
            projection_matrix,
            reprojection_matrix,
        ) in enumerate(kitti_dataloader):

            gt_depth = helpers.disp2depth(gt_disp, projection_matrix[0][0][0])
            xyz = helpers.depth2xyz(gt_depth[0], reprojection_matrix[0])
            transf = helpers.get_stereo_points_transf(xyz.dtype, xyz.device)
            flow = helpers.transformation2flow(xyz, transf, projection_matrix[0])

            _, _, H, W = gt_depth.shape

            img_left = torch.nn.functional.interpolate(
                imgpairs_left_forward[:, :3], size=(H, W)
            )[0]
            img_right = torch.nn.functional.interpolate(
                imgpairs_right_forward[:, :3], size=(H, W)
            )[0]
            img_right_warped = helpers.warp(img_right.unsqueeze(0), flow.unsqueeze(0))[
                0
            ]
            img_right_warped[~gt_disp_mask[0].repeat(3, 1, 1)] = 0.0

            imgs_orig = torch.cat((img_left, img_right), dim=2)
            imgs_warp = torch.cat((img_left, img_right_warped), dim=2)

            img_left[~gt_disp_mask[0].repeat(3, 1, 1)] = 0.0
            img_diff = img_left - img_right_warped
            img_diff = torch.mean(img_diff, dim=0, keepdim=True).repeat(3, 1, 1)
            img_diff = torch.cat((img_left, img_diff), dim=2)

            imgs = torch.cat((imgs_orig, imgs_warp, img_diff), dim=1)

            imgs = torch.nn.functional.interpolate(imgs.unsqueeze(0), size=(H, W))
            helpers.visualize_img(imgs[0])
            # helpers.visualize_flow(flow)

    def test_read_calibs(self):

        """
        Keys:
        dict_keys(['corner_dist',
        'S_00', 'K_00', 'D_00', 'R_00', 'T_00', 'S_rect_00', 'R_rect_00', 'P_rect_00',
        'S_01', 'K_01', 'D_01', 'R_01', 'T_01', 'S_rect_01', 'R_rect_01', 'P_rect_01',
        'S_02', 'K_02', 'D_02', 'R_02', 'T_02', 'S_rect_02', 'R_rect_02', 'P_rect_02',
        'S_03', 'K_03', 'D_03', 'R_03', 'T_03', 'S_rect_03', 'R_rect_03', 'P_rect_03'])

        Focal Lengths in x direction: fx:
        width_to_focal = dict()
        width_to_focal[1242] = 721.5377
        width_to_focal[1241] = 718.856
        width_to_focal[1224] = 707.0493
        width_to_focal[1238] = 718.3351

        training: {1242.0: array([721.5377]), 1224.0: array([707.0493]), 1238.0: array([718.3351]), 1241.0: array([718.856])}
        testing: {1242.0: array([721.5377]), 1224.0: array([707.0493]), 1238.0: array([718.3351]), 1226.0: array([707.0912]), 1241.0: array([718.856])}
        """

        dict_fx = {}
        dict_fy = {}

        dict_cx = {}
        dict_cy = {}

        calib_dirs = [
            os.path.join(
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/testing/calib_cam_to_cam",
            ),
            os.path.join(
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/testing/calib_cam_to_cam",
            ),
        ]

        for calib_dir in calib_dirs:
            filenames = os.listdir(calib_dir)
            for filename in filenames:
                data = helpers.read_calib_file(os.path.join(calib_dir, filename))
                # indices 0, 1, 2, 3  = left-gray, right-gray, left-rgb, right-rgb

                width = data["S_rect_02"][0]
                height = data["S_rect_02"][1]
                fx = data["P_rect_02"][0]
                fy = data["P_rect_02"][5]

                cx = data["P_rect_02"][2]
                cy = data["P_rect_02"][6]

                if width not in dict_fx.keys():
                    dict_fx[width] = [fx]
                else:
                    dict_fx[width].append(fx)

                if height not in dict_fy.keys():
                    dict_fy[height] = [fy]
                else:
                    dict_fy[height].append(fy)

                if width not in dict_cx.keys():
                    dict_cx[width] = [cx]
                else:
                    dict_cx[width].append(cx)

                if height not in dict_cy.keys():
                    dict_cy[height] = [cy]
                else:
                    dict_cy[height].append(cy)

                width_02 = width
                height_02 = height
                fx_02 = fx
                fy_02 = fy
                cx_02 = cx
                cy_02 = cy

                width = data["S_rect_03"][0]
                height = data["S_rect_03"][1]

                fx = data["P_rect_03"][0]
                fy = data["P_rect_03"][5]

                cx = data["P_rect_03"][2]
                cy = data["P_rect_03"][6]

                if (
                    cx != cx_02
                    or cy != cy_02
                    or fx != fx_02
                    or fy != fy_02
                    or width != width_02
                    or height != height_02
                ):

                    print("ERROR: difference in rgb-cameras")

                if width not in dict_fx.keys():
                    dict_fx[width] = [fx]
                else:
                    dict_fx[width].append(fx)

                if height not in dict_fy.keys():
                    dict_fy[height] = [fy]
                else:
                    dict_fy[height].append(fy)

                if width not in dict_cx.keys():
                    dict_cx[width] = [cx]
                else:
                    dict_cx[width].append(cx)

                if height not in dict_cy.keys():
                    dict_cy[height] = [cy]
                else:
                    dict_cy[height].append(cy)

                # print('S_rect_02:', np.array(data['S_rect_02']))
                # print('P_rect_02:', np.array(data['P_rect_02'])[0])

                # print('S_rect_03:', np.array(data['S_rect_03']))
                # print('P_rect_03:', np.array(data['P_rect_03'])[0])

                # for key, val in data.items():
                #    print(key, ': ' , val)
                # print(data['K_03'][0])

        for key in dict_fx.keys():
            dict_fx[key] = np.unique(np.array(dict_fx[key]))
        for key in dict_fy.keys():
            dict_fy[key] = np.unique(np.array(dict_fy[key]))
        for key in dict_cx.keys():
            dict_cx[key] = np.unique(np.array(dict_cx[key]))
        for key in dict_cy.keys():
            dict_cy[key] = np.unique(np.array(dict_cy[key]))

        print("fx", dict_fx)
        print("fy", dict_fy)
        print("cx", dict_cx)
        print("cy", dict_cy)
        pass

    def test_normalization_features(self):
        x = torch.randn((2, 3, 480, 640)) * 10 + 10
        # x = np.array([[0.123, 1.123], [2.123, 3.123]], dtype=np.float32)

        mean1, var1 = tf.nn.moments(tf.convert_to_tensor(x), axes=[-3, -2, -1])
        print(var1)

        var2, mean2 = torch.var_mean(x, dim=[1, 2, 3], unbiased=False)
        print(var2)

        print("np", np.var(x.numpy(), axis=(1, 2, 3)))

        features1 = torch.randn((2, 3, 2, 2)) * 10 + 10
        features2 = torch.randn((2, 3, 2, 2)) * 10 + 10
        max_displacement = 4

        torch_features1 = features1.clone()
        torch_features2 = features2.clone()

        # torch_features1 = helpers.normalize_features(torch_features1)
        # torch_features2 = helpers.normalize_features(torch_features2)

        torch_features1, torch_features2 = helpers.normalize_features(
            torch_features1, torch_features2
        )
        # torch_features1_diff = torch_features1_normalized - torch_features1
        # torch_features2_diff = torch_features2_normalized - torch_features2
        # print(torch.max(torch.abs(torch_features1_diff)))
        # print(torch.max(torch.abs(torch_features2_diff)))

        # torch_features1, torch_features2 = helpers.normalize_features(torch_features1, torch_features2)

        features1 = self.torch_to_tf(features1)
        features2 = self.torch_to_tf(features2)

        def normalize_features(
            feature_list,
            normalize,
            center,
            moments_across_channels,
            moments_across_images,
        ):
            """Normalizes feature tensors (e.g., before computing the cost volume).

            Args:
              feature_list: list of tf.tensors, each with dimensions [b, h, w, c]
              normalize: bool flag, divide features by their standard deviation
              center: bool flag, subtract feature mean
              moments_across_channels: bool flag, compute mean and std across channels
              moments_across_images: bool flag, compute mean and std across images

            Returns:
              list, normalized feature_list
            """

            # Compute feature statistics.
            import collections

            statistics = collections.defaultdict(list)
            axes = [-3, -2, -1] if moments_across_channels else [-3, -2]
            for feature_image in feature_list:
                mean, variance = tf.nn.moments(
                    x=feature_image, axes=axes, keepdims=True
                )
                statistics["mean"].append(mean)
                statistics["var"].append(variance)

            print("vars", statistics["var"])
            if moments_across_images:
                statistics["mean"] = [
                    tf.reduce_mean(input_tensor=statistics["mean"])
                ] * len(feature_list)
                statistics["var"] = [
                    tf.reduce_mean(input_tensor=statistics["var"])
                ] * len(feature_list)

            statistics["std"] = [tf.sqrt(v + 1e-16) for v in statistics["var"]]

            # Center and normalize features.

            if center:
                feature_list = [
                    f - mean for f, mean in zip(feature_list, statistics["mean"])
                ]
            if normalize:
                feature_list = [
                    f / std for f, std in zip(feature_list, statistics["std"])
                ]

            for i in range(len(statistics["mean"])):
                print("tf - mean", statistics["mean"][i])
                print("tf - var", statistics["var"][i])

            return feature_list

        features1, features2 = normalize_features(
            [features1, features2], True, True, True, True
        )

        tf_features1 = self.tf_to_torch(features1)
        tf_features2 = self.tf_to_torch(features2)

        diff_features1 = torch_features1 - tf_features1
        diff_features2 = torch_features2 - tf_features2

        print(torch.max(torch.abs(diff_features1)))
        print(torch.max(torch.abs(diff_features2)))
        pass

    def test_correlation_volume(self):

        features1 = torch.randn((10, 3, 480, 640)) * 10
        features2 = torch.randn((10, 3, 480, 640)) * 10
        max_displacement = 4

        torch_features1 = features1.clone()
        torch_features2 = features2.clone()

        torch_features1, torch_features2 = helpers.normalize_features(
            torch_features1, torch_features2
        )

        torch_cost_volume = helpers.compute_cost_volume(
            x1=torch_features1, x2=torch_features2, max_displacement=max_displacement
        )

        features1 = self.torch_to_tf(features1)
        features2 = self.torch_to_tf(features2)

        def normalize_features(
            feature_list,
            normalize,
            center,
            moments_across_channels,
            moments_across_images,
        ):
            """Normalizes feature tensors (e.g., before computing the cost volume).

            Args:
              feature_list: list of tf.tensors, each with dimensions [b, h, w, c]
              normalize: bool flag, divide features by their standard deviation
              center: bool flag, subtract feature mean
              moments_across_channels: bool flag, compute mean and std across channels
              moments_across_images: bool flag, compute mean and std across images

            Returns:
              list, normalized feature_list
            """

            # Compute feature statistics.
            import collections

            statistics = collections.defaultdict(list)
            axes = [-3, -2, -1] if moments_across_channels else [-3, -2]
            for feature_image in feature_list:
                mean, variance = tf.nn.moments(
                    x=feature_image, axes=axes, keepdims=True
                )
                statistics["mean"].append(mean)
                statistics["var"].append(variance)

            if moments_across_images:
                statistics["mean"] = [
                    tf.reduce_mean(input_tensor=statistics["mean"])
                ] * len(feature_list)
                statistics["var"] = [
                    tf.reduce_mean(input_tensor=statistics["var"])
                ] * len(feature_list)

            statistics["std"] = [tf.sqrt(v + 1e-16) for v in statistics["var"]]

            # Center and normalize features.

            if center:
                feature_list = [
                    f - mean for f, mean in zip(feature_list, statistics["mean"])
                ]
            if normalize:
                feature_list = [
                    f / std for f, std in zip(feature_list, statistics["std"])
                ]

            for i in range(len(statistics["mean"])):
                print("tf - mean", statistics["mean"][i])
                print("tf - var", statistics["var"][i])

            return feature_list

        features1, features2 = normalize_features(
            [features1, features2], True, True, True, True
        )

        """Compute the cost volume between features1 and features2.

        Displace features2 up to max_displacement in any direction and compute the
        per pixel cost of features1 and the displaced features2.

        Args:
          features1: tf.tensor of shape [b, h, w, c]
          features2: tf.tensor of shape [b, h, w, c]
          max_displacement: int, maximum displacement for cost volume computation.

        Returns:
          tf.tensor of shape [b, h, w, (2 * max_displacement + 1) ** 2] of costs for
          all displacements.
        """

        # Set maximum displacement and compute the number of image shifts.
        _, height, width, _ = features1.shape.as_list()
        if max_displacement <= 0 or max_displacement >= height:
            raise ValueError(f"Max displacement of {max_displacement} is too large.")

        max_disp = max_displacement
        num_shifts = 2 * max_disp + 1

        # Pad features2 and shift it while keeping features1 fixed to compute the
        # cost volume through correlation.

        # Pad features2 such that shifts do not go out of bounds.
        features2_padded = tf.pad(
            tensor=features2,
            paddings=[[0, 0], [max_disp, max_disp], [max_disp, max_disp], [0, 0]],
            mode="CONSTANT",
        )
        cost_list = []
        for i in range(num_shifts):
            for j in range(num_shifts):
                corr = tf.reduce_mean(
                    input_tensor=features1
                    * features2_padded[:, i : (height + i), j : (width + j), :],
                    axis=-1,
                    keepdims=True,
                )
                cost_list.append(corr)
        tf_cost_volume = tf.concat(cost_list, axis=-1)
        tf_cost_volume = self.tf_to_torch(tf_cost_volume)

        diff_cost_volumes = tf_cost_volume - torch_cost_volume

        print(torch.max(diff_cost_volumes))
        pass

    def test_census_transform(self):
        x1 = torch.rand((10, 3, 48, 64)) * 10.0 + 10.0

        torch_x1 = x1.clone()

        torch_x1_census = helpers.census_transform(torch_x1, patch_size=7)

        tf_x1 = self.torch_to_tf(x1)
        tf_x1_census = uflow_helpers.census_transform(tf_x1, patch_size=7)
        tf_x1_census = self.tf_to_torch(tf_x1_census)

        diff_x1_census = torch_x1_census - tf_x1_census

        # print('torch', torch_x1_census.cpu().numpy())
        # print('tf', tf_x1_census.cpu().numpy())
        print("max difference: torch-tf", torch.max(torch.abs(diff_x1_census)))
        pass

    def test_census_loss(self):
        x1 = torch.randn((10, 3, 480, 640)) * 10
        x2 = torch.randn((10, 3, 480, 640)) * 10

        masks = torch.rand((10, 1, 480, 640))

        num_pairs = 1

        torch_census_loss = helpers.calc_census_loss(
            x1, x2, patch_size=7, masks_flow_valid=masks
        )

        tf_x1 = self.torch_to_tf(x1)
        tf_x2 = self.torch_to_tf(x2)
        tf_masks = self.torch_to_tf(masks)

        tf_x1 = tf_x1.gpu()
        tf_x2 = tf_x2.gpu()
        tf_masks = tf_masks.gpu()

        weight_census = 1.0

        def abs_robust_loss(diff, eps=0.01, q=0.4):
            """The so-called robust loss used by DDFlow."""
            return tf.pow((tf.abs(diff) + eps), q)

        tf_census_loss = (
            weight_census
            * uflow_helpers.census_loss(
                tf_x1, tf_x2, tf_masks, distance_metric_fn=abs_robust_loss
            )
            / num_pairs
        )

        torch_census_loss = torch_census_loss.numpy()
        print("torch_census_loss", torch_census_loss)
        tf_census_loss = tf_census_loss.numpy()
        print("tf_census_loss", tf_census_loss)

        diff_census_losses = torch_census_loss - tf_census_loss
        print("diff_census_losses", diff_census_losses)

    def test_calc_transform_between_pointclouds(self):
        from scipy.spatial.transform import Rotation
        from tensor_operations.pointcloud_transforms import calc_transform_between_pointclouds

        dtype = torch.float32
        device = torch.device("cpu:0")
        transf = helpers.get_stereo_points_transf(dtype, device)
        rot = Rotation.from_euler("z", 90, degrees=True)
        transf[:3, :3] = torch.tensor(rot.as_matrix(), dtype=dtype, device=device)

        print("transf", transf)

        pts1 = torch.randn(3, 100, 30) * 3.0 + 1.0
        pts2 = helpers.transformation(pts1, transf)

        pts1 = pts1.reshape(3, -1)
        pts2 = pts2.reshape(3, -1)

        estimated_transf = calc_transform_between_pointclouds(pts1, pts2)

        diff_transf = transf - estimated_transf

        print("diff transformations: \n", diff_transf)

    def test_calc_fundamentalmatrix_from_opticalflow_v2(self):
        from scipy.spatial.transform import Rotation

        # from pointcloud_transforms import calc_transform_between_pointclouds
        from tensor_operations.pointcloud_transforms import calc_fundamentalmatrix_from_opticalflow

        dtype = torch.float32
        device = torch.device("cpu:0")
        transf = helpers.get_stereo_points_transf(dtype, device)
        rot = Rotation.from_euler("z", 45, degrees=True)
        transf[:3, :3] = torch.tensor(rot.as_matrix(), dtype=dtype, device=device)
        transf[0, 3] = transf[0, 3] * 100.0
        print("transf \n", transf)

        H = 100
        W = 30

        grid_y, grid_x = torch.meshgrid(
            [
                torch.arange(0.0, H + 0.0, dtype=dtype, device=device),
                torch.arange(0.0, W + 0.0, dtype=dtype, device=device),
            ]
        )
        grid_x = (grid_x / (W - 1.0) * 2.0 - 1.0) * W / 2.0
        grid_y = (grid_y / (H - 1.0) * 2.0 - 1.0) * H / 2.0
        grid_1 = torch.ones((H, W), dtype=dtype, device=device)
        pts1 = torch.stack((grid_x, grid_y, grid_1), dim=0)
        pts1[2] = torch.rand(H, W) * 1000.0 + 1000.0
        pts1[:2] = pts1[:2] * pts1[2]

        pts2 = helpers.transformation(pts1, transf)

        # helpers.visualize_point_cloud(torch.cat((pts1, pts2), dim=1))
        scene_flow = pts2 - pts1

        coords1 = pts1[:2] / pts1[2]
        coords2 = pts2[:2] / pts2[2]
        optical_flow = coords2 - coords1

        mask = ((pts1[2] > 0.0) * (pts2[2] > 0.0)).unsqueeze(0)
        projection_matrix = torch.eye(3)[:2, :]

        transf_pred = calc_fundamentalmatrix_from_opticalflow(
            optical_flow, mask, projection_matrix
        )

        print("transf_pred \n", transf_pred)

    def test_calc_fundamentalmatrix_from_opticalflow(self):
        from tensor_operations.pointcloud_transforms import calc_fundamentalmatrix_from_opticalflow

        kitti_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            calibs_dir=os.path.join(
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            preload=False,
            dev="cpu",
            max_num_imgs=args.val_dataset_max_num_imgs,
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=False
        )

        for i, (
            imgpairs_forward,
            gt_flow_noc_uv_forward,
            gt_flow_noc_valid_forward,
            gt_flow_occ_uv_forward,
            gt_flow_occ_valid_forward,
            projection_matrix,
            reprojection_matrix,
        ) in enumerate(kitti_dataloader):

            if i == 30:
                flow = gt_flow_occ_uv_forward[0]
                mask = gt_flow_occ_valid_forward[0]
                # helpers.visualize_flow(flow)
                transf_pred = calc_fundamentalmatrix_from_opticalflow(
                    flow, mask.unsqueeze(0), projection_matrix[0]
                )
                print("transf_pred \n", transf_pred)

            else:
                if i < 30:
                    continue
                else:
                    break

    def test_smoothness_loss(self):

        B = 3
        H = 1000
        W = 6
        flow = torch.randn(B, 2, H, W) * 1000
        img1 = torch.rand(B, 3, H, W)

        torch_loss = helpers.calc_smoothness_loss(flow, img1, edge_weight=150, order=2)

        img1 = self.torch_to_tf(img1)
        flow = self.torch_to_tf(flow)

        tf_loss = uflow_helpers.calc_smoothness_loss_order2(img1=img1, flow=flow)

        print("torch", torch_loss)
        print("tf", tf_loss)

        pass

    def test_downsampling(self):
        from torchvision.transforms import transforms
        import PIL

        kitti_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            calibs_dir=os.path.join(
                "..",
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            preload=False,
            dev="cpu",
            max_num_imgs=args.val_dataset_max_num_imgs,
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=False
        )

        for i, (
            imgpairs_forward,
            gt_flow_noc_uv_forward,
            gt_flow_noc_valid_forward,
            gt_flow_occ_uv_forward,
            gt_flow_occ_valid_forward,
            projection_matrix,
            reprojection_matrix,
        ) in enumerate(kitti_dataloader):

            B = 1
            H = 640
            W = 640
            img1s = torch.rand(B, 3, H, W)
            img1s = imgpairs_forward[:, :3]

            torch_img1s_down = torch.nn.functional.interpolate(
                img1s, scale_factor=1.0 / 2.0, mode="bilinear", align_corners=False
            )

            # torch_img1s_down = torch.nn.functional.interpolate(torch_img1s_down, scale_factor=1.0 / 2.0,
            #                                                   mode='bilinear',
            #                                                   align_corners=False)
            torch_img1s_down = torch.nn.functional.interpolate(
                img1s,
                size=(int(H) // 2, int(W) // 2),
                mode="bilinear",
                align_corners=False,
            )

            resize_transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(
                        (int(H) // 2, int(W) // 2), interpolation=PIL.Image.BILINEAR
                    ),
                    transforms.ToTensor(),
                ]
            )

            # for i in range(B):
            #    torch_img1s_down[i] = resize_transforms(img1s[i])

            # helpers.visualize_img(torch_img1s_down[0])

            img1s = self.torch_to_tf(img1s)

            tf_img1s_down = uflow_helpers.resize(
                img1s, int(H) // 2, int(W) // 2, is_flow=False
            )
            # tf_img1s_down = uflow_helpers.resize(tf_img1s_down, int(H) // 4, int(W) // 4, is_flow=False)

            tf_img1s_down = self.tf_to_torch(tf_img1s_down)

            # helpers.visualize_imgpair(torch.cat((torch_img1s_down, tf_img1s_down), dim=1)[0])

            helpers.visualize_img(torch.abs(torch_img1s_down[0] - tf_img1s_down[0]))

            print(torch.max(torch.abs(torch_img1s_down - tf_img1s_down)))

    def test_upsampling(self):

        kitti_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                "..", args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            calibs_dir=os.path.join(
                "..",
                args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            preload=False,
            dev="cpu",
            max_num_imgs=args.val_dataset_max_num_imgs,
        )

        kitti_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=False
        )

        for i, (
            imgpairs_forward,
            gt_flow_noc_uv_forward,
            gt_flow_noc_valid_forward,
            gt_flow_occ_uv_forward,
            gt_flow_occ_valid_forward,
            projection_matrix,
            reprojection_matrix,
        ) in enumerate(kitti_dataloader):

            B = 3
            H = 640
            W = 640
            img1s = torch.rand(B, 3, H, W)
            img1s = imgpairs_forward[:, :3]

            torch_img1s_up = torch.nn.functional.interpolate(
                img1s, scale_factor=4.0, mode="bilinear", align_corners=False
            )

            # torch_img1s_up = torch.nn.functional.interpolate(torch_img1s_up, scale_factor=4.0, mode='bilinear',
            #                                                   align_corners=False)

            # helpers.visualize_img(torch_img1s_down[0])

            img1s = self.torch_to_tf(img1s)

            tf_img1s_up = uflow_helpers.upsample(img1s, is_flow=False)
            tf_img1s_up = uflow_helpers.upsample(tf_img1s_up, is_flow=False)

            tf_img1s_up = self.tf_to_torch(tf_img1s_up)

            # helpers.visualize_imgpair(torch.cat((torch_img1s_down, tf_img1s_down), dim=1)[0])

            # helpers.visualize_img(torch.abs(torch_img1s_up[0] - tf_img1s_up[0]))

            print(torch.max(torch.abs(torch_img1s_up - tf_img1s_up)))

    def test_read_rgb(self):
        import PIL
        from torchvision import transforms

        height = 640
        width = 640
        image_path = "C:/Users/Leonh/PycharmProjects/optical-flow/datasets/KITTI_flow/training/image_2/000000_10.png"
        image_data = tf.compat.v1.gfile.FastGFile(image_path, "rb").read()

        image_uint = tf.image.decode_png(image_data, channels=3)
        image_float = tf.image.convert_image_dtype(image_uint, tf.float32)
        image_resized = tf.image.resize(
            image_float[None], [height, width], method=tf.image.ResizeMethod.BILINEAR
        )[0]

        tf_img = image_resized
        tf_img = tf.expand_dims(tf_img, 0)
        tf_img = self.tf_to_torch(tf_img)[0]
        img = PIL.Image.open(image_path)

        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )[0]

        torch_img = img

        print(torch.max(torch.abs(tf_img - torch_img)))


if __name__ == "__main__":
    unittest.main()
