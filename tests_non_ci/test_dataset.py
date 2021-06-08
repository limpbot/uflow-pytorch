import unittest

from util import helpers
import torch
from tensor_operations import transforms3d

# import tensorflow as tf

from usflow import options
import os
from datasets.datasets import KittiDataset


parser = options.setup_comon_options()

preliminary_args = [
    "-s",
    "config/config_setup_0.yaml",
    "-c",
    "config/config_coach_def_usceneflow.yaml",
]
args = parser.parse_args(preliminary_args)


class Tests(unittest.TestCase):
    def torch_to_tf(self, x, device="cpu"):

        x = tf.convert_to_tensor(x.permute(0, 2, 3, 1).cpu().detach().numpy())
        return x

    def tf_to_torch(self, x, dev=None):
        x = torch.from_numpy(x.numpy()).permute(0, 3, 1, 2)
        if dev is not None:
            x = x.to(device=dev)

        return x

    def test_raw_dataset(self):
        from datasets.datasets import KittiDataset
        import os

        kitti_dataset = KittiDataset(
            raw_dataset=True,
            fp_imgs_filenames=os.path.join(
                "..",
                "kitti_raw_meta",
                "lists_imgpair_filenames",
                "raw_train_monosf_kittisplit.txt",
            ),
            raw_dir=os.path.join(args.datasets_dir, "KITTI_complete"),
            return_left_and_right=True,
            return_projection_matrices=True,
            calibs_dir=os.path.join("..", "kitti_raw_meta", "cam_intrinsics"),
        )

        """
        kitti_dataset = KittiDataset(imgs_left_dir=os.path.join('..', args.datasets_dir, 'KITTI_flow_multiview/training/image_2'),
                                     imgs_right_dir=os.path.join('..', args.datasets_dir, 'KITTI_flow_multiview/training/image_3'),
                                     return_left_and_right=True)
        """

        train_dataloader = torch.utils.data.DataLoader(
            kitti_dataset, batch_size=1, shuffle=False
        )

        for i, data in enumerate(train_dataloader):
            if i % 100 == 0:
                print(i)
            if i % 1000 == 0:
                imgpairs_left, imgpairs_right, proj_mats, reproj_mats = data

                print(proj_mats[0])
                imgpairs = torch.cat((imgpairs_left, imgpairs_right), dim=2)

                imgpairs = torch.nn.functional.interpolate(imgpairs, size=(720, 640))
                helpers.visualize_imgpair(imgpairs[0])
            pass

    def test_read_obj_map(self):
        import cv2
        import numpy as np

        obj_map_fn = "/media/driveD/datasets/KITTI_flow/training/obj_map/000007_10.png"
        # TODO: check if cv2.imread(disp_fn, cv2.IMREAD_UNCHANGED) reads as np.uint16
        # obj_map = cv2.imread(obj_map_fn)#, cv2.IMREAD_ANYDEPTH)
        obj_map = cv2.imread(obj_map_fn, cv2.IMREAD_UNCHANGED).astype(np.uint8)

        # shape: 375x1242x3 range: [0, 1]

        # H x W : dtype=np.uint16 note: maximum > 256
        obj_map = torch.from_numpy(obj_map)
        obj_map = obj_map.unsqueeze(0)
        # helpers.visualize_img((obj_map == 0)* 1.0)

        for i in range(obj_map.max() + 1):
            print(i)
            helpers.visualize_img((obj_map == i) * 1.0)

    def test_get_transf(self):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataloader_device = self.device
        self.dataloader_pin_memory = False

        if self.args.dataloader_num_workers > 0:
            self.dataloader_device = "cpu"
            self.dataloader_pin_memory = True

        self.val_dataset = KittiDataset(
            imgs_left_dir=os.path.join(
                self.args.datasets_dir, "KITTI_flow/training/image_2"
            ),
            flows_noc_dir=os.path.join(
                self.args.datasets_dir, "KITTI_flow/training/flow_noc"
            ),
            flows_occ_dir=os.path.join(
                self.args.datasets_dir, "KITTI_flow/training/flow_occ"
            ),
            return_flow=True,
            calibs_dir=os.path.join(
                self.args.datasets_dir,
                "KITTI_flow/data_scene_flow_calib/training/calib_cam_to_cam",
            ),
            return_projection_matrices=True,
            disps0_dir=os.path.join(
                self.args.datasets_dir, "KITTI_flow/training/disp_noc_0"
            ),
            disps1_dir=os.path.join(
                self.args.datasets_dir, "KITTI_flow/training/disp_noc_1"
            ),
            return_disp=True,
            masks_objects_dir=os.path.join(
                self.args.datasets_dir, "KITTI_flow/training/obj_map"
            ),
            return_mask_objects=True,
            preload=False,
            max_num_imgs=self.args.val_dataset_max_num_imgs,
            dev=self.dataloader_device,
            width=self.args.arch_res_width,
            height=self.args.arch_res_height,
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=self.dataloader_pin_memory,
        )

        val_epoch_length = len(self.val_dataloader)

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
            gt_masks_objects_left_fwd,
            proj_mats_left_fwd,
            reproj_mats_left_fwd,
        ) in enumerate(self.val_dataloader):
            imgpairs_left_fwd = imgpairs_left_fwd.to(self.device)
            gt_flow_noc_uv_fwd = gt_flow_noc_uv_fwd.to(self.device)
            gt_flow_noc_valid_fwd = gt_flow_noc_valid_fwd.to(self.device)
            gt_flow_occ_uv_fwd = gt_flow_occ_uv_fwd.to(self.device)
            gt_flow_occ_valid_fwd = gt_flow_occ_valid_fwd.to(self.device)
            gt_disps_left_fwd = gt_disps_left_fwd.to(self.device)
            gt_disps_masks_left_fwd = gt_disps_masks_left_fwd.to(self.device)
            gt_disps_left_bwd = gt_disps_left_bwd.to(self.device)
            gt_disps_masks_left_bwd = gt_disps_masks_left_bwd.to(self.device)
            gt_masks_objects_left_fwd = gt_masks_objects_left_fwd.to(self.device)
            proj_mats_left_fwd = proj_mats_left_fwd.to(self.device)
            reproj_mats_left_fwd = reproj_mats_left_fwd.to(self.device)

            _, _, gt_H, gt_W = gt_flow_occ_uv_fwd.shape
            _, _, H, W = imgpairs_left_fwd.shape
            sx = gt_W / W
            sy = gt_H / H
            print("sx, sy", sx, sy)
            proj_mats_left_fwd, reproj_mats_left_fwd = helpers.rescale_intrinsics(
                proj_mats_left_fwd, reproj_mats_left_fwd, sx, sy
            )

            gt_pts3d_left1 = helpers.disp2xyz(
                gt_disps_left_fwd, proj_mats_left_fwd, reproj_mats_left_fwd
            )
            gt_pts3d_left2 = helpers.disp2xyz(
                gt_disps_left_bwd,
                proj_mats_left_fwd,
                reproj_mats_left_fwd,
                gt_flow_occ_uv_fwd,
            )

            own_mask = torch.zeros_like(gt_disps_masks_left_fwd)
            _, _, H, W = own_mask.shape
            # own_mask[:, :, :3*int(W/4), int(W/3):int(2*W/3)] = 1
            gt_mask_pts3d_valid = (
                gt_disps_masks_left_fwd
                * gt_disps_masks_left_bwd
                * (gt_masks_objects_left_fwd == 0)
                * gt_flow_noc_valid_fwd
            )  # * own_mask

            # from transforms3d import visualize_pcds
            # visualize_pcds(gt_pts3d_left1[0].flatten(1), gt_pts3d_left2[0].flatten(1), gt_mask_pts3d_valid[0].flatten(1))

            # gt_mask_pts3d_valid = gt_mask_pts3d_valid.repeat(1, 3, 1, 1)

            # transf = transforms3d.calc_transform_between_pointclouds(
            #    gt_pts3d_left1[0, :, gt_mask_pts3d_valid[0, 0]],
            #    gt_pts3d_left2[0, :, gt_mask_pts3d_valid[0, 0]],
            # )

            # transf = transforms3d.calc_transform_between_pointclouds_v2(gt_pts3d_left1[0, :, gt_mask_pts3d_valid[0, 0]],
            #                                                            gt_pts3d_left2[0, :, gt_mask_pts3d_valid[0, 0]],
            #                                                            so3_type='exponential')

            # oflow = ogeo.pt3d_2_oflow(gt_pts3d_left1, proj_mats_left_fwd)
            # helpers.visualize_flow(oflow[0])
            transf = transforms3d.calc_transformation_via_optical_flow(
                gt_pts3d_left1[0],
                gt_pts3d_left2[0],
                gt_oflow=gt_flow_occ_uv_fwd[0],
                mask_valid=gt_mask_pts3d_valid[0, 0],
                proj_mat=proj_mats_left_fwd[0],
                reproj_mat=reproj_mats_left_fwd[0],
            )

            nH = 400
            nW = 200
            imgs_left_fwd_n = torch.nn.functional.interpolate(
                imgpairs_left_fwd[:, :3], size=(nH, nW), mode="nearest"
            )

            gt_pts3d_left1_n = torch.nn.functional.interpolate(
                gt_pts3d_left1, size=(nH, nW), mode="nearest"
            )

            gt_pts3d_left2_n = torch.nn.functional.interpolate(
                gt_pts3d_left2, size=(nH, nW), mode="nearest"
            )

            gt_mask_pts3d_valid_n = (
                torch.nn.functional.interpolate(
                    gt_mask_pts3d_valid * 1.0, size=(nH, nW), mode="nearest"
                )
                == 1.0
            )

            # transf = transforms3d.calc_transform_between_pointclouds_v2(gt_pts3d_left1_n[0].flatten(1),
            ##                                                            gt_pts3d_left2_n[0].flatten(1),
            #                                                            so3_type='exponential',
            #                                                            img=imgs_left_fwd_n[0],
            #                                                            mask_valid=gt_mask_pts3d_valid_n[0])

            print(batch_id)
            gt_pts3d_left1_ftf = transforms3d.pts3d_transform(
                gt_pts3d_left1, transf.unsqueeze(0).to(self.device)
            )
            gt_sflow = gt_pts3d_left1_ftf - gt_pts3d_left1
            gt_oflow = helpers.sflow2oflow(
                gt_sflow, gt_disps_left_fwd, proj_mats_left_fwd, reproj_mats_left_fwd
            )

            gt_oflow[0, :, ~gt_mask_pts3d_valid[0, 0]] = 0.0
            helpers.visualize_flow(
                torch.cat((gt_flow_occ_uv_fwd[0], gt_oflow[0]), dim=1)
            )
            # helpers.visualize_img(gt_mask_pts3d_valid[0])
            # helpers.visualize_img((gt_masks_objects_left_fwd[0] == 0))
            print(transf)


if __name__ == "__main__":
    unittest.main()
