import torch
import tensor_operations.geometric as ops_geo
import tensor_operations.rearrange as ops_rearr
import tensor_operations.warp as ops_warp
import tensor_operations.mask as ops_mask
import tensor_operations.vision as ops_vis
import tensor_operations.loss as ops_loss


class USFlowLossVal:
    def __init__(self, model, args):
        self.model = model
        self.args = args

        # self.logger = logger

    def calc_losses(self, batch, batch_idx):
        (
            imgpairs_left_fwd,
            gt_flow_noc_uv_fwd,
            gt_flow_noc_valid_fwd,
            gt_flow_l1_occ_uv,
            gt_flow_occ_valid_fwd,
            gt_disps_left_fwd,
            gt_disps_masks_left_fwd,
            gt_disps_left_bwd,
            gt_disps_masks_left_bwd,
            gt_se3s_l1,
            projs_mat_l1,
            reprojs_mat_l1,
        ) = batch

        metrics = {}
        images = {}
        images_ids = [0, 10, 20, 22]

        num_imgpairs_l1 = imgpairs_left_fwd.size(0)

        _, _, gt_H, gt_W = gt_flow_l1_occ_uv.shape

        imgpairs_left_bwd = torch.cat(
            (imgpairs_left_fwd[:, 3:, :, :], imgpairs_left_fwd[:, :3, :, :]), dim=1
        )
        imgpairs_l12 = torch.cat((imgpairs_left_fwd, imgpairs_left_bwd), dim=0)
        projs_mat_l12 = projs_mat_l1.repeat(2, 1, 1)
        reprojs_mat_l12 = reprojs_mat_l1.repeat(2, 1, 1)

        flows_l12, disps_l12, se3s_6d_l12, masks_l12 = self.model.forward(
            imgpairs_l12, projs_mat_l12, reprojs_mat_l12
        )

        flows_l12 = flows_l12[0]
        disps_l12 = disps_l12[0]
        masks_l12 = masks_l12[0]

        if se3s_6d_l12 is not None:
            se3s_l12 = ops_geo.se3_6d_to_se3mat(se3s_6d_l12)
            if self.args.test_sflow_via_disp_se3:
                if batch_idx in images_ids:
                    _, _, H, W = flows_l12.shape
                    sx = gt_W / W
                    sy = gt_H / H

                    oflows_l1 = torch.nn.functional.interpolate(
                        flows_l12,
                        size=(gt_H, gt_W),
                        mode="bilinear",
                        align_corners=True,
                    )
                    oflows_l1[:, 0] = oflows_l1[:, 0] * sx
                    oflows_l1[:, 1] = oflows_l1[:, 1] * sy
                    images["uflow"] = ops_vis.flow2rgb(flows_l12[0], draw_arrows=True)

                pts3d_l12 = ops_geo.disp_2_pt3d(
                    disps_l12, projs_mat_l12, reprojs_mat_l12
                )
                flows_l12, _ = ops_geo.sflow_via_transform(
                    pts3d_l12,
                    se3s_l12,
                    masks_l12,
                    egomotion_addition=self.args.arch_se3_egomotion_addition,
                )

            if torch.sum(torch.isnan(flows_l12)) or torch.sum(torch.isnan(disps_l12)):
                a = 0

            se3s_l1 = se3s_l12[:num_imgpairs_l1]
            # disps_left_img2 = disps_left[num_imgpairs_left_fwd:]

            # choose mask 0
            se3s_error_dist, se3s_error_angle = ops_loss.dist_angle_transfs(
                se3s_l1[:, 0], gt_se3s_l1
            )
            metrics["se3s-err-angle"] = se3s_error_angle.sum()
            metrics["se3s-err-dist"] = se3s_error_dist.sum()

        flows_l1 = flows_l12[:num_imgpairs_l1]
        disps_l1 = disps_l12[:num_imgpairs_l1]
        depths_l1 = ops_geo.disp_2_depth(disps_l1, fx=projs_mat_l1[:, 0, 0])

        if flows_l1.size(1) == 2:
            disps_l2 = disps_l12[num_imgpairs_l1:]
            oflows_l1 = flows_l1
        elif flows_l1.size(1) == 3:
            pts3d_l1 = ops_geo.depth_2_pt3d(depths_l1, reprojs_mat_l1)
            pts3d_l1_flow_ftf = pts3d_l1 + flows_l1
            disps_l2 = ops_geo.depth_2_disp(
                pts3d_l1_flow_ftf[:, 2:3], fx=projs_mat_l1[:, 0, 0]
            )

            # pts3d_l1_flow_ftf = ops_geo.pts3d_transform_obj_ego(
            #    pts3d_l1,
            #    se3s_l1,
            #    masks_l1,
            #    egomotion_addition=self.args.arch_se3_egomotion_addition,
            # )

            pxls2d_l1_wrpd = ops_geo.pt3d_2_pxl2d(pts3d_l1_flow_ftf, projs_mat_l1)
            oflows_l1 = ops_geo.pxl2d_2_oflow(pxls2d_l1_wrpd)
        else:
            print(
                "error: bad arch_flow_out_channels:",
                self.args.arch_flow_out_channels,
            )

        _, _, H, W = oflows_l1.shape
        sx = gt_W / W
        sy = gt_H / H

        oflows_l1 = torch.nn.functional.interpolate(
            oflows_l1, size=(gt_H, gt_W), mode="bilinear", align_corners=True
        )
        oflows_l1[:, 0] = oflows_l1[:, 0] * sx
        oflows_l1[:, 1] = oflows_l1[:, 1] * sy

        # disps_left_img1 : 1x1x640x640
        disps_l1 = torch.nn.functional.interpolate(
            disps_l1, size=(gt_H, gt_W), mode="bilinear", align_corners=True
        )
        disps_l1[:, 0] = disps_l1[:, 0] * sx

        disps_l2 = torch.nn.functional.interpolate(
            disps_l2, size=(gt_H, gt_W), mode="bilinear", align_corners=True
        )
        disps_l2[:, 0] = disps_l2[:, 0] * sx

        disps_left_img2_wrpd = ops_warp.warp(disps_l2, gt_flow_l1_occ_uv)

        metrics["outl-D1"] = ops_loss.calc_disp_outlier_percentage(
            disps_l1, gt_disps_left_fwd, gt_disps_masks_left_fwd
        )

        metrics["outl-D2"] = ops_loss.calc_disp_outlier_percentage(
            disps_left_img2_wrpd, gt_disps_left_bwd, gt_disps_masks_left_bwd
        )

        metrics["outl-F"] = ops_loss.calc_flow_outlier_percentage(
            oflows_l1, gt_flow_l1_occ_uv, gt_flow_occ_valid_fwd
        )

        metrics["outl-SF"] = ops_loss.calc_sflow_outlier_percentage(
            disps_l1,
            gt_disps_left_fwd,
            gt_disps_masks_left_fwd,
            disps_left_img2_wrpd,
            gt_disps_left_bwd,
            gt_disps_masks_left_bwd,
            oflows_l1,
            gt_flow_l1_occ_uv,
            gt_flow_occ_valid_fwd,
        )

        flow_uv_orig_pred_vs_gt = torch.cat(
            (oflows_l1[0], gt_flow_noc_uv_fwd[0]), dim=1
        )

        if self.args.eval_save_visualizations:
            flow_uv_orig_pred_vs_gt_rgb = ops_vis.flow2rgb(
                flow_uv_orig_pred_vs_gt, draw_arrows=True
            )

            # self.logger.log_img(dir=self.logger.eval_flows_dir, name="flow_" + str(batch_idx) + ".png",
            #                    img=flow_uv_orig_pred_vs_gt_rgb)

            if batch_idx in images_ids:
                images["flow"] = flow_uv_orig_pred_vs_gt_rgb

            if masks_l12 is not None:
                masks_l12_rgb = ops_vis.mask2rgb(masks_l12[0])
                # self.logger.log_img(dir=self.logger.eval_masks_dir, name="mask_" + str(batch_idx) + ".png",
                #                   img=masks_l12_rgb)

                if batch_idx in images_ids:
                    images["mask"] = masks_l12_rgb

        metrics["oflow-epe-noc"] = torch.sum(
            gt_flow_noc_valid_fwd
            * torch.norm(
                gt_flow_noc_uv_fwd - oflows_l1[:num_imgpairs_l1],
                dim=1,
                p=None,
            ),
            dim=(0, 1, 2),
        )

        metrics["oflow-epe-noc"] = metrics["oflow-epe-noc"] / torch.sum(
            gt_flow_noc_valid_fwd
        )

        metrics["oflow-epe-occ"] = torch.sum(
            gt_flow_occ_valid_fwd
            * torch.norm(
                gt_flow_l1_occ_uv - oflows_l1[:num_imgpairs_l1],
                dim=1,
                p=None,
            ),
            dim=(0, 1, 2),
        )
        metrics["oflow-epe-occ"] = metrics["oflow-epe-occ"] / torch.sum(
            gt_flow_occ_valid_fwd
        )

        depths_l1 = torch.nn.functional.interpolate(
            depths_l1,
            size=(gt_H, gt_W),
            mode="bilinear",
            align_corners=True,
        )
        gt_depths = ops_geo.disp_2_depth(
            gt_disps_left_fwd, fx=projs_mat_l1[:, 0, 0] * sx
        )

        depths_pred_vs_gt = torch.cat((depths_l1[0], gt_depths[0]), dim=1)
        if self.args.eval_save_visualizations:
            depths_pred_vs_gt_rgb = ops_vis.depth2rgb(depths_pred_vs_gt)
            # self.logger.log_img(dir=self.logger.eval_depths_dir, name="depth_" + str(batch_idx) + ".png",
            #                    img=depths_pred_vs_gt_rgb)

            if batch_idx in images_ids:
                images["depth"] = depths_pred_vs_gt_rgb

        metrics["depth-sq-rel"] = torch.mean(
            (depths_l1[gt_disps_masks_left_fwd] - gt_depths[gt_disps_masks_left_fwd])
            ** 2
            / gt_depths[gt_disps_masks_left_fwd]
        )

        metrics["depth-abs-rel"] = torch.mean(
            torch.abs(
                depths_l1[gt_disps_masks_left_fwd] - gt_depths[gt_disps_masks_left_fwd]
            )
            / gt_depths[gt_disps_masks_left_fwd]
        )

        return metrics, images
