import torch
import unittest
from util import helpers

import os
from datasets.datasets import KittiDataset


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

    def test_shape2pxlcoords(self):

        B, H, W = (2, 640, 640)
        flow = torch.zeros(size=(B, 2, H, W))
        dtype = flow.dtype
        device = flow.device

        pxlcoords1 = helpers.shape2pxlcoords(B, H, W, dtype=dtype, device=device)

        pxlcoords2 = helpers.oflow2pxlcoords(flow)

        pxlcoords_diff = pxlcoords1 - pxlcoords2

        print("pxlcoords_diff - l1", torch.norm(pxlcoords_diff, p=1))
        print("pxlcoords_diff - sum-abs", torch.sum(torch.abs(pxlcoords_diff)))

    def test_shape2uv1(self):
        B, H, W = (2, 640, 640)

        dtype = torch.float32
        device = "cpu"

        grid_v, grid_u = torch.meshgrid(
            [
                torch.arange(0.0, H, dtype=dtype, device=device),
                torch.arange(0.0, W, dtype=dtype, device=device),
            ]
        )

        grid_1 = torch.ones_like(grid_u)

        grid_uv1_v1 = torch.stack((grid_u, grid_v, grid_1))

        grid_uv1_v2 = helpers.shape2uv1(B=0, H=H, W=W, dtype=dtype, device=device)

        grid_uv1_diff = grid_uv1_v1 - grid_uv1_v2
        print("grid_uv1_diff - l1", torch.norm(grid_uv1_diff, p=1))

    def test_depth_oflow2xyz(self):

        B, H, W = (2, 640, 640)
        oflow = torch.zeros(size=(B, 2, H, W))
        dtype = oflow.dtype
        device = oflow.device

        depth = torch.randn(size=(B, 1, H, W))
        reproj_mats = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)

        xyz1 = helpers.depth2xyz(depth, reproj_mats)
        xyz2 = helpers.depth_oflow2xyz(depth, oflow, reproj_mats)

        xyz_diff = xyz1 - xyz2

        print("xyz_diff - l1", torch.norm(xyz_diff, p=1))

    def test_rescale_intrinsics(self):

        B = 2
        scale_factor = 32.0

        reproj_mats_left = torch.randn(size=(3, 3)).unsqueeze(0).repeat(B, 1, 1)
        proj_mats_left = torch.randn(size=(3, 3)).unsqueeze(0).repeat(B, 1, 1)
        proj_mats_left = proj_mats_left[:, :2, :]
        print(proj_mats_left)

        (
            proj_mats_left_scaled_v2,
            reproj_mats_left_scaled_v2,
        ) = helpers.rescale_intrinsics(
            proj_mats_left.clone(),
            reproj_mats_left.clone(),
            sx=1.0 / scale_factor,
            sy=1.0 / scale_factor,
        )
        proj_mats_left_scaled_v1 = proj_mats_left.clone() / scale_factor
        # without clone this is like an assignment
        reproj_mats_left_scaled_v1 = reproj_mats_left.clone()
        reproj_mats_left_scaled_v1[:, :, :2] = (
            reproj_mats_left_scaled_v1[:, :, :2] * scale_factor
        )

        print(
            "proj_mats - l1",
            torch.norm(proj_mats_left_scaled_v1 - proj_mats_left_scaled_v2, p=1),
        )
        print(
            "reproj_mats - l1",
            torch.norm(reproj_mats_left_scaled_v1 - reproj_mats_left_scaled_v2, p=1),
        )

    def test_mask2rgb(self):
        import cv2

        torch_mask = torch.zeros(size=(5, 640, 640))
        torch_mask[:1, :400, :400] = 1.0
        torch_mask[1:2, 300:, 300:] = 1.0
        torch_mask[2:3, :400, 300:] = 1.0
        torch_mask = (torch_mask > 0.5).type(torch.int64)
        cv_img = helpers.mask2rgb(torch_mask)

        cv2.imshow("mask", cv_img)
        cv2.waitKey(0)
