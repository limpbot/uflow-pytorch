import torch
import tensor_operations.geometric as ops_geo
import tensor_operations.rearrange as ops_rearr
import tensor_operations.warp as ops_warp
import tensor_operations.mask as ops_mask
import tensor_operations.vision as ops_vis
import tensor_operations.loss as ops_loss


class USFlowLossTrain:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def calc_losses(self, batch, batch_idx, vwriter=None):
        """
        (
            _imgpairs_left_fwd,
            _imgpairs_right_fwd,
            _proj_mats_left_fwd,
            _reproj_mats_left_fwd,
        ) = batch

        imgpairs_left_fwd = _imgpairs_left_fwd.to(self.device)
        imgpairs_right_fwd = _imgpairs_right_fwd.to(self.device)
        proj_mats_left_fwd = _proj_mats_left_fwd.to(self.device)
        reproj_mats_left_fwd = _reproj_mats_left_fwd.to(self.device)
        """

        (
            imgpairs_left_fwd,
            imgpairs_right_fwd,
            proj_mats_left_fwd,
            reproj_mats_left_fwd,
        ) = batch

        states = {}
        losses = {}

        num_imgpairs_left_fwd = imgpairs_left_fwd.size(0)
        H_in = imgpairs_left_fwd.size(2)
        W_in = imgpairs_left_fwd.size(3)

        imgpairs_left_bwd = ops_rearr.imgpairs_swap_order(imgpairs_left_fwd)
        imgpairs_right_bwd = ops_rearr.imgpairs_swap_order(imgpairs_right_fwd)

        imgpairs_left = torch.cat((imgpairs_left_fwd, imgpairs_left_bwd), dim=0)
        imgpairs_right = torch.cat((imgpairs_right_fwd, imgpairs_right_bwd), dim=0)

        # batch: left_fwd, left_bwd, right_fwd, right_bwd
        projs_mat_l1 = proj_mats_left_fwd.repeat(2, 1, 1)
        reprojs_mat_l1 = reproj_mats_left_fwd.repeat(2, 1, 1)

        proj_mats = proj_mats_left_fwd.repeat(4, 1, 1)
        reproj_mats = reproj_mats_left_fwd.repeat(4, 1, 1)

        num_imgpairs_left = imgpairs_left.size(0)

        imgpairs = torch.cat((imgpairs_left, imgpairs_right), dim=0)

        num_imgpairs = imgpairs.size(0)

        if self.args.loss_flow_teacher_crop_lambda > 0.0:

            imgpairs_left_cropped = imgpairs_left[
                :,
                :,
                self.args.loss_flow_teacher_crop_reduction_size : -self.args.loss_flow_teacher_crop_reduction_size,
                self.args.loss_flow_teacher_crop_reduction_size : -self.args.loss_flow_teacher_crop_reduction_size,
            ]

            imgpairs_left_cropped = torch.nn.functional.interpolate(
                imgpairs_left_cropped,
                size=(H_in, W_in),
                mode="bilinear",
                align_corners=True,
            )

            num_cropped_imgpairs = imgpairs_left_cropped.size(0)

            list_flows, list_disps, se3s_6d, list_masks = self.model(
                torch.cat((imgpairs, imgpairs_left_cropped), dim=0),
                torch.cat((proj_mats, projs_mat_l1), dim=0),
                torch.cat((reproj_mats, reprojs_mat_l1), dim=0),
            )

            list_student_flow_forward = []
            list_student_flow_backward = []
            list_teacher_flow_forward = []
            list_teacher_flow_backward = []
            for lvl_id in range(len(list_flows)):
                flows = list_flows[lvl_id]
                list_student_flow_forward.append(
                    flows[-num_cropped_imgpairs : -num_cropped_imgpairs // 2]
                )
                list_student_flow_backward.append(flows[-num_cropped_imgpairs // 2 :])

                list_teacher_flow_forward.append(flows[: num_cropped_imgpairs // 2])
                list_teacher_flow_backward.append(
                    flows[num_cropped_imgpairs // 2 : num_cropped_imgpairs]
                )

                list_flows[lvl_id] = flows[:num_imgpairs]
                list_disps[lvl_id] = list_disps[lvl_id][:num_imgpairs]
                list_masks[lvl_id] = list_masks[lvl_id][:num_imgpairs]
            se3s_6d = se3s_6d[:num_imgpairs]

        else:
            list_flows, list_disps, se3s_6d, list_masks = self.model.forward(
                imgpairs, proj_mats, reproj_mats
            )

        imgs_l1 = imgpairs[:num_imgpairs_left, :3]
        imgs_r1 = imgpairs[num_imgpairs_left:, :3]
        imgs_l2 = imgpairs[:num_imgpairs_left, 3:]

        lvl_weights = self.args.loss_lvl_weights

        for lvl_id in range(len(list_flows)):

            if lvl_weights[lvl_id] == 0.0:
                continue

            flows = list_flows[lvl_id]
            disps = list_disps[lvl_id]
            masks = list_masks[lvl_id]

            _, _, H_out, W_out = disps.shape

            scale_inout = H_in / H_out

            disps_l1 = disps[:num_imgpairs_left]
            disps_r1 = disps[num_imgpairs_left:]

            flows_l1 = flows[:num_imgpairs_left]
            if se3s_6d is not None:
                se3s_6d_l1 = se3s_6d[:num_imgpairs_left]
            else:
                se3s_6d_l1 = None

            if masks is not None:
                masks_l1 = masks[:num_imgpairs_left]
            else:
                masks_l1 = None

            imgs_l1_rs = ops_vis.resize(imgs_l1, H_out=H_out, W_out=W_out)
            imgs_r1_rs = ops_vis.resize(imgs_r1, H_out=H_out, W_out=W_out)
            imgs_l2_rs = ops_vis.resize(imgs_l2, H_out=H_out, W_out=W_out)

            # disparity loss:
            #   photometric loss based on warp of right image using disparity
            #   smoothness loss

            pxls2d_l1_disp_ftf = ops_geo.disp_2_pxl2d(-disps_l1)
            pxls2d_r1_disp_ftf = ops_geo.disp_2_pxl2d(disps_r1)

            imgs_r1_rs_bwrpd, masks_disp_inside_l1 = ops_warp.interpolate2d(
                imgs_r1_rs,
                pxls2d_l1_disp_ftf,
                return_masks_flow_inside=True,
            )

            masks_disp_non_occl_l1 = ops_mask.pxl2d_2_mask_non_occl(
                pxls2d_r1_disp_ftf, binary=self.args.loss_masks_non_occlusion_binary
            )

            masks_disp_valid_l1 = masks_disp_non_occl_l1 * masks_disp_inside_l1

            masks_disp_valid_l1 = ops_mask.mask_ensure_non_zero(
                masks_disp_valid_l1, thresh_perc=self.args.loss_nonzeromask_thresh_perc
            )

            if self.args.loss_disp_photo_lambda > 0.0:
                loss_disp_photo = (
                    self.args.loss_disp_photo_lambda
                    * ops_loss.calc_photo_loss(
                        imgs_l1_rs,
                        imgs_r1_rs_bwrpd,
                        masks_flow_valid=masks_disp_valid_l1,
                        type=self.args.loss_photo_type,
                    )
                )

                if "disp_photo" not in losses:
                    losses["disp_photo"] = loss_disp_photo * lvl_weights[lvl_id]
                else:
                    losses["disp_photo"] += loss_disp_photo * lvl_weights[lvl_id]

            if self.args.loss_disp_smooth_lambda > 0.0:
                loss_disp_smooth = (
                    self.args.loss_disp_smooth_lambda
                    * ops_loss.calc_smoothness_loss(
                        disps_l1,
                        imgs_l1_rs,
                        edge_weight=self.args.loss_disp_smooth_edgeweight,
                        order=self.args.loss_disp_smooth_order,
                        smooth_type=self.args.loss_smooth_type,
                    )
                )

                if "disp_smooth" not in losses:
                    losses["disp_smooth"] = loss_disp_smooth * lvl_weights[lvl_id]
                else:
                    losses["disp_smooth"] += loss_disp_smooth * lvl_weights[lvl_id]

            # scene flow loss:
            #   photometric loss based on warp of second image to first image (forward and backward)
            #   smoothness loss
            #   consistency3d loss

            (projs_mat_l1_rs, reprojs_mat_l1_rs,) = ops_vis.rescale_intrinsics(
                projs_mat_l1.clone(),
                reprojs_mat_l1.clone(),
                sx=1.0 / scale_inout,
                sy=1.0 / scale_inout,
            )

            pts3d_l1 = ops_geo.disp_2_pt3d(
                disps_l1, proj_mats=projs_mat_l1_rs, reproj_mats=reprojs_mat_l1_rs
            )
            pts3d_l1_norm = torch.norm(pts3d_l1, p=2, dim=1, keepdim=True)
            if self.args.loss_pts3d_norm_detach:
                pts3d_l1_norm = pts3d_l1_norm.detach()

            if self.args.arch_flow_out_channels == 2:
                pxls2d_l1_flow_ftf = ops_geo.oflow_2_pxl2d(flows_l1)

            elif self.args.arch_flow_out_channels == 3:
                pts3d_l1_flow_ftf = pts3d_l1 + flows_l1
                pxls2d_l1_flow_ftf = ops_geo.pt3d_2_pxl2d(
                    pts3d_l1_flow_ftf, proj_mats=projs_mat_l1_rs
                )
            else:
                print(
                    "error: invalid dimensions of flow:",
                    self.args.arch_flow_out_channels,
                )
            # fixed num_imgpairs_left -> num_imgpairs_left_fwd
            pxls2d_l2_flow_ftf = ops_rearr.batches_swap_order(pxls2d_l1_flow_ftf)

            # change warping: 1st flow2coords / sceneflow2coords / 2nd: warp(img, coords)
            imgs_l2_rs_flow_bwrpd, masks_flow_inside_l1 = ops_warp.interpolate2d(
                imgs_l2_rs,
                pxls2d_l1_flow_ftf,
                return_masks_flow_inside=True,
            )

            masks_flow_non_occl_l1 = ops_mask.pxl2d_2_mask_non_occl(pxls2d_l2_flow_ftf)

            masks_flow_valid_l1 = masks_flow_non_occl_l1 * masks_flow_inside_l1
            masks_flow_valid_l1 = ops_mask.mask_ensure_non_zero(
                masks_flow_valid_l1, thresh_perc=self.args.loss_nonzeromask_thresh_perc
            )

            # masks_flow_disp_valid_l1 = masks_flow_valid_l1 * masks_disp_valid_l1

            if self.args.loss_flow_photo_lambda > 0.0:
                loss_flow_photo = (
                    self.args.loss_flow_photo_lambda
                    * ops_loss.calc_photo_loss(
                        imgs_l1_rs,
                        imgs_l2_rs_flow_bwrpd,
                        masks_flow_valid=masks_flow_valid_l1,
                        type=self.args.loss_photo_type,
                    )
                )
                if "flow_photo" not in losses:
                    losses["flow_photo"] = loss_flow_photo * lvl_weights[lvl_id]
                else:
                    losses["flow_photo"] += loss_flow_photo * lvl_weights[lvl_id]

            if self.args.loss_flow_teacher_crop_lambda > 0.0:

                student_flow_forward = list_student_flow_forward[lvl_id]
                student_flow_backward = list_student_flow_backward[lvl_id]
                teacher_flow_forward = list_teacher_flow_forward[lvl_id]
                teacher_flow_backward = list_teacher_flow_backward[lvl_id]
                crop_reduction_size = int(
                    self.args.loss_flow_teacher_crop_reduction_size / scale_inout
                )

                losses[
                    "teacher_student_crops"
                ] = self.args.loss_flow_teacher_crop_lambda * ops_loss.calc_selfsup_loss(
                    student_flow_forward,
                    student_flow_backward,
                    teacher_flow_forward,
                    teacher_flow_backward,
                    crop_reduction_size=crop_reduction_size,
                )

                # epoch_with_selfsup = max(self.epoch-self.args.loss_flow_teacher_crop_begin_epoch, 0)
                # losses["teacher_student_crops"] = losses["teacher_student_crops"] * min(
                #        self.epoch_with_selfsup / self.args.loss_flow_teacher_crop_rampup_epochs, 1.0
                #    )

            if self.args.loss_disp_flow_cons3d_lambda > 0.0:
                if self.args.arch_flow_out_channels == 2:
                    disps_l2 = ops_rearr.batches_swap_order(disps_l1)
                    pts3d_l2 = ops_geo.disp_2_pt3d(
                        disps_l2,
                        proj_mats=projs_mat_l1_rs,
                        reproj_mats=reprojs_mat_l1_rs,
                    )
                    pts3d_l1_flow_ftf = ops_warp.interpolate2d(
                        pts3d_l2, pxls2d_l1_flow_ftf
                    )

                loss_disp_flow_cons3d = (
                    self.args.loss_disp_flow_cons3d_lambda
                    * ops_loss.calc_consistency3d_loss(
                        pts3d_l1_norm,
                        pts3d_l1_flow_ftf,
                        pts3d_l1,
                        pxls2d_l1_flow_ftf,
                        masks_flow_valid_l1,
                        type=self.args.loss_disp_flow_cons3d_type,
                    )
                )

                if "disp_flow_cons3d" not in losses:
                    losses["disp_flow_cons3d"] = (
                        loss_disp_flow_cons3d * lvl_weights[lvl_id]
                    )
                else:
                    losses["disp_flow_cons3d"] += (
                        loss_disp_flow_cons3d * lvl_weights[lvl_id]
                    )

                # if torch.sum(torch.isnan(loss_disp_flow_cons3d)) or torch.sum(
                #    torch.isnan(loss_flow_photo)
                # ):
                #    a = 0

            if self.args.loss_flow_smooth_lambda > 0.0:
                if self.args.arch_flow_out_channels == 2:
                    weights_inv = None
                else:
                    weights_inv = pts3d_l1_norm + 1e-8
                loss_sf_smooth = (
                    self.args.loss_flow_smooth_lambda
                    * (1.0 / scale_inout)
                    * ops_loss.calc_smoothness_loss(
                        flows_l1,
                        imgs_l1_rs,
                        edge_weight=self.args.loss_flow_smooth_edgeweight,
                        order=self.args.loss_flow_smooth_order,
                        weights_inv=weights_inv,
                        smooth_type=self.args.loss_smooth_type,
                    )
                )

                if "flow_smooth" not in losses:
                    losses["flow_smooth"] = loss_sf_smooth * lvl_weights[lvl_id]
                else:
                    losses["flow_smooth"] += loss_sf_smooth * lvl_weights[lvl_id]

            if se3s_6d is not None:
                se3s_mat_l1 = ops_geo.se3_6d_to_se3mat(se3s_6d_l1)
                pts3d_l1_se3_ftf = ops_geo.pts3d_transform_obj_ego(
                    pts3d_l1,
                    se3mat=se3s_mat_l1,
                    mask=masks_l1,
                    egomotion_addition=self.args.arch_se3_egomotion_addition,
                )
                pxl2d_l1_se3_ftf = ops_geo.pt3d_2_pxl2d(
                    pts3d_l1_se3_ftf, proj_mats=projs_mat_l1_rs
                )

                (
                    imgs_l2_rs_disp_se3_bwrpd,
                    masks_l1_se3_ftf_inside,
                ) = ops_warp.interpolate2d(
                    imgs_l2_rs,
                    pxl2d_l1_se3_ftf,
                    return_masks_flow_inside=True,
                )
                masks_l1_se3_ftf_valid = masks_l1_se3_ftf_inside * masks_flow_valid_l1

                if self.args.loss_disp_se3_photo_lambda > 0.0:
                    loss_disp_se3_photo = (
                        self.args.loss_disp_se3_photo_lambda
                        * ops_loss.calc_photo_loss(
                            imgs_l1_rs,
                            imgs_l2_rs_disp_se3_bwrpd,
                            masks_flow_valid=masks_l1_se3_ftf_valid,
                            type=self.args.loss_photo_type,
                            fwdbwd=self.args.loss_disp_se3_photo_fwdbwd,
                        )
                    )
                    if "disp_se3_photo" not in losses:
                        losses["disp_se3_photo"] = (
                            loss_disp_se3_photo * lvl_weights[lvl_id]
                        )
                    else:
                        losses["disp_se3_photo"] += (
                            loss_disp_se3_photo * lvl_weights[lvl_id]
                        )

                if self.args.loss_se3_diversity_lambda > 0.0:
                    if self.args.arch_se3_egomotion_addition:
                        se3s_6d_l1_sep = se3s_6d_l1[:, 1:]
                    else:
                        se3s_6d_l1_sep = se3s_6d_l1
                    loss_se3_reg = None
                    _, K, _ = se3s_6d_l1_sep.shape
                    for k in range(K):
                        se3_6d_l1_diff = se3s_6d_l1_sep[:, k : k + 1] - se3s_6d_l1_sep
                        se3_6d_l1_diff[:, k : k + 1] = 100.0
                        se3_6d_l1_diff[
                            :, :, :3
                        ] /= self.args.arch_se3_rotation_sensitivity
                        se3_6d_l1_diff[
                            :, :, 3:
                        ] /= self.args.arch_se3_translation_sensitivity
                        loss_se3_reg_single = (
                            torch.exp(
                                -torch.norm(
                                    se3_6d_l1_diff,
                                    dim=2,
                                    p=1,
                                    keepdim=True,
                                )
                            ).mean()
                            * self.args.loss_se3_diversity_lambda
                        )
                        if k == 0:
                            loss_se3_reg = loss_se3_reg_single
                        else:
                            loss_se3_reg += loss_se3_reg_single

                    if "se3_diversity" not in losses:
                        losses["se3_diversity"] = loss_se3_reg * lvl_weights[lvl_id]
                    else:
                        losses["se3_diversity"] += loss_se3_reg * lvl_weights[lvl_id]

                if self.args.loss_disp_se3_cons3d_lambda > 0.0:
                    # points3d_left_img2_bwrpd = torch.zeros_like(points3d_left_img2_bwrpd)
                    loss_disp_se3_cons3d = (
                        self.args.loss_disp_se3_cons3d_lambda
                        * ops_loss.calc_consistency3d_loss(
                            pts3d_l1_norm,
                            pts3d_l1_se3_ftf,
                            pts3d_l1,
                            pxl2d_l1_se3_ftf,
                            masks_l1_se3_ftf_valid,
                            type=self.args.loss_disp_se3_cons3d_type,
                            fwdbwd=self.args.loss_disp_se3_cons3d_fwdbwd,
                        )
                    )
                    if "disp_se3_cons3d" not in losses:
                        losses["disp_se3_cons3d"] = (
                            loss_disp_se3_cons3d * lvl_weights[lvl_id]
                        )
                    else:
                        losses["disp_se3_cons3d"] += (
                            loss_disp_se3_cons3d * lvl_weights[lvl_id]
                        )

                if self.args.loss_disp_se3_cons_oflow_lambda > 0.0:

                    pts3d_l1_se3_ftf_pts3d_l1_detach = ops_geo.pts3d_transform_obj_ego(
                        pts3d_l1.detach(),
                        se3mat=se3s_mat_l1,
                        mask=masks_l1,
                        egomotion_addition=self.args.arch_se3_egomotion_addition,
                    )

                    # points3d_left_img2_bwrpd = torch.zeros_like(points3d_left_img2_bwrpd)
                    disp_se3_cons_oflow = (
                        self.args.loss_disp_se3_cons_oflow_lambda
                        * ops_loss.calc_consistency3d_loss(
                            pts3d_l1_norm,
                            pts3d_l1_se3_ftf_pts3d_l1_detach,
                            pts3d_l1.detach(),
                            pxls2d_l1_flow_ftf,
                            masks_l1_se3_ftf_valid,
                            type=self.args.loss_disp_se3_cons_oflow_type,
                            fwdbwd=self.args.loss_disp_se3_cons_oflow_fwdbwd,
                        )
                    )
                    if "disp_se3_cons_oflow" not in losses:
                        losses["disp_se3_cons_oflow"] = (
                            disp_se3_cons_oflow * lvl_weights[lvl_id]
                        )
                    else:
                        losses["disp_se3_cons_oflow"] += (
                            disp_se3_cons_oflow * lvl_weights[lvl_id]
                        )

                if masks_l1 is not None and self.args.loss_mask_cons_oflow_lambda > 0.0:
                    # points3d_left_img2_bwrpd = torch.zeros_like(points3d_left_img2_bwrpd)
                    loss_mask_cons_oflow = (
                        self.args.loss_mask_cons_oflow_lambda
                        * ops_loss.calc_consistency_mask_loss(
                            masks_l1,
                            pxls2d_l1_flow_ftf,
                            masks_l1_se3_ftf_valid,
                            fwdbwd=self.args.loss_mask_cons_oflow_fwdbwd,
                        )
                    )
                    if "loss_mask_cons_oflow" not in losses:
                        losses["loss_mask_cons_oflow"] = (
                            loss_mask_cons_oflow * lvl_weights[lvl_id]
                        )
                    else:
                        losses["loss_mask_cons_oflow"] += (
                            loss_mask_cons_oflow * lvl_weights[lvl_id]
                        )

                if (
                    masks_l1 is not None
                    and self.args.loss_mask_reg_nonzero_lambda > 0.0
                ):
                    loss_mask_reg = (
                        ops_loss.calc_mask_reg_loss(masks_l1)
                        * self.args.loss_mask_reg_nonzero_lambda
                    )
                    if "mask_reg" not in losses:
                        losses["mask_reg"] = loss_mask_reg * lvl_weights[lvl_id]
                    else:
                        losses["mask_reg"] += loss_mask_reg * lvl_weights[lvl_id]

                if (
                    self.args.loss_disp_se3_proj2oflow_corr3d_separate_lambda > 0.0
                    and (
                        not self.args.loss_disp_se3_proj2oflow_level0_only
                        or (lvl_id == 0)
                    )
                ):
                    if self.args.arch_flow_out_channels != 2:
                        print(
                            "error: cannot calculate disp_se3_proj2oflow loss for flow channels != 2"
                        )
                    else:
                        if self.args.loss_disp_se3_proj2oflow_fwdbwd:
                            B = pts3d_l1.size(0)
                        else:
                            B = pts3d_l1.size(0) // 2

                        if (
                            masks_l1 is not None
                            and self.args.loss_disp_se3_proj2oflow_corr3d_separate_mask_lambda
                            > 0.0
                        ):
                            (
                                loss_disp_se3_proj2oflow_corr3d_separate_mask,
                                masks_l1_from_loss,
                            ) = ops_loss.calc_mask_consistency_oflow_loss(
                                pts3d_l1[:B],
                                masks_l1[:B],
                                se3s_mat_l1[:B],
                                flows_l1[:B],
                                projs_mat_l1_rs[:B],
                                reprojs_mat_l1_rs[:B],
                                masks_flow_valid_l1[
                                    :B
                                ],  # masks_flow_valid_l1[:B],  # masks_l1_se3_ftf_valid[:B],
                                self.args.loss_disp_se3_proj2oflow_cross3d_score_const_weight,
                                self.args.loss_disp_se3_proj2oflow_cross3d_score_linear_weight,
                                self.args.loss_disp_se3_proj2oflow_cross3d_score_exp_weight,
                                self.args.loss_disp_se3_proj2oflow_cross3d_score_exp_slope,
                                self.args.loss_disp_se3_proj2oflow_cross3d_outlier_slope,
                                self.args.loss_disp_se3_proj2oflow_cross3d_outlier_min,
                                self.args.loss_disp_se3_proj2oflow_cross3d_outlier_max,
                                self.args.loss_disp_se3_proj2oflow_cross3d_max,
                                self.args.loss_disp_se3_proj2oflow_cross3d_min,
                                egomotion_addition=self.args.arch_se3_egomotion_addition,
                                visualize=self.args.train_visualize,
                                vwriter=vwriter,
                            )

                            loss_disp_se3_proj2oflow_corr3d_separate_mask *= (
                                self.args.loss_disp_se3_proj2oflow_corr3d_separate_mask_lambda
                            )

                            if "disp_se3_proj2oflow_corr3d_separate_mask" not in losses:
                                losses["disp_se3_proj2oflow_corr3d_separate_mask"] = (
                                    loss_disp_se3_proj2oflow_corr3d_separate_mask
                                    * lvl_weights[lvl_id]
                                )
                            else:
                                losses["disp_se3_proj2oflow_corr3d_separate_mask"] += (
                                    loss_disp_se3_proj2oflow_corr3d_separate_mask
                                    * lvl_weights[lvl_id]
                                )

                        if (
                            self.args.loss_disp_se3_proj2oflow_corr3d_separate_use_mask_from_loss
                        ):
                            masks_se3 = masks_l1_from_loss
                        else:
                            masks_se3 = masks_l1

                        if masks_se3 is not None:
                            masks_se3 = masks_se3[:B]
                        loss_disp_se3_proj2oflow_corr3d_separate = ops_loss.calc_consistency_oflow_corr3d_single_loss(
                            pts3d_l1[:B],
                            masks_se3,
                            se3s_mat_l1[:B],
                            flows_l1[:B],
                            projs_mat_l1_rs[:B],
                            reprojs_mat_l1_rs[:B],
                            masks_flow_valid_l1[
                                :B
                            ],  # masks_flow_valid_l1[:B],  # masks_l1_se3_ftf_valid[:B],
                            egomotion_addition=self.args.arch_se3_egomotion_addition,
                        )

                        loss_disp_se3_proj2oflow_corr3d_separate *= (
                            self.args.loss_disp_se3_proj2oflow_corr3d_separate_lambda
                        )

                        if "disp_se3_proj2oflow_corr3d_separate" not in losses:
                            losses["disp_se3_proj2oflow_corr3d_separate"] = (
                                loss_disp_se3_proj2oflow_corr3d_separate
                                * lvl_weights[lvl_id]
                            )
                        else:
                            losses["disp_se3_proj2oflow_corr3d_separate"] += (
                                loss_disp_se3_proj2oflow_corr3d_separate
                                * lvl_weights[lvl_id]
                            )

                if masks_l1 is not None and self.args.loss_mask_smooth_lambda > 0.0:

                    loss_mask_smooth = (
                        self.args.loss_mask_smooth_lambda
                        * (1.0 / scale_inout)
                        * ops_loss.calc_smoothness_loss(
                            masks_l1,
                            imgs_l1_rs,
                            edge_weight=self.args.loss_mask_smooth_edgeweight,
                            order=self.args.loss_mask_smooth_order,
                            smooth_type=self.args.loss_smooth_type,
                        )
                    )

                    if "mask_smooth" not in losses:
                        losses["mask_smooth"] = loss_mask_smooth * lvl_weights[lvl_id]
                    else:
                        losses["mask_smooth"] += loss_mask_smooth * lvl_weights[lvl_id]

                if (
                    masks_l1 is not None
                    and self.args.loss_disp_se3_proj2oflow_corr3d_joint_lambda > 0.0
                    and (
                        not self.args.loss_disp_se3_proj2oflow_level0_only
                        or (lvl_id == 0)
                    )
                ):

                    pts3d_l1_se3_ftf_pts3d_l1_detach = ops_geo.pts3d_transform_obj_ego(
                        pts3d_l1.detach(),
                        se3mat=se3s_mat_l1,
                        mask=masks_l1,
                        egomotion_addition=self.args.arch_se3_egomotion_addition,
                    )
                    pxl2d_l1_se3_ftf_pts3d_l1_detach = ops_geo.pt3d_2_pxl2d(
                        pts3d_l1_se3_ftf_pts3d_l1_detach, proj_mats=projs_mat_l1_rs
                    )

                    if self.args.arch_flow_out_channels != 2:
                        print(
                            "error: cannot calculate disp_se3_proj2oflow loss for flow channels != 2"
                        )
                    else:
                        if self.args.loss_disp_se3_proj2oflow_fwdbwd:
                            B = pxl2d_l1_se3_ftf_pts3d_l1_detach.size(0)
                        else:
                            B = pxl2d_l1_se3_ftf_pts3d_l1_detach.size(0) // 2

                        # oflow_l1_se3_ftf = ops_geo.pxl2d_2_oflow(pxl2d_l1_se3_ftf_pts3d_l1_detach)

                        (
                            loss_disp_se3_proj2oflow_corr3d_joint,
                            _,
                        ) = ops_loss.calc_consistency_oflow_corr3d_loss(
                            pts3d_l1_se3_ftf_pts3d_l1_detach[:B],
                            flows_l1[:B],
                            masks_l1_se3_ftf_valid[:B],
                            projs_mat_l1_rs[:B],
                            reprojs_mat_l1_rs[:B],
                        )
                        loss_disp_se3_proj2oflow_corr3d_joint = (
                            loss_disp_se3_proj2oflow_corr3d_joint.mean()
                        )
                        loss_disp_se3_proj2oflow_corr3d_joint *= (
                            self.args.loss_disp_se3_proj2oflow_corr3d_joint_lambda
                        )

                        if "disp_se3_proj2oflow_corr3d_joint" not in losses:
                            losses["disp_se3_proj2oflow_corr3d_joint"] = (
                                loss_disp_se3_proj2oflow_corr3d_joint
                                * lvl_weights[lvl_id]
                            )
                        else:
                            losses["disp_se3_proj2oflow_corr3d_joint"] += (
                                loss_disp_se3_proj2oflow_corr3d_joint
                                * lvl_weights[lvl_id]
                            )

            if lvl_id == 0:
                states["pts3d-lvl0-norm"] = torch.mean(
                    torch.norm(pts3d_l1, p=2, dim=1)
                ).item()
            if lvl_id == 5:
                states["pts3d-lvl4-norm"] = torch.mean(
                    torch.norm(pts3d_l1, p=2, dim=1)
                ).item()

        """
        loss_disp = 0.
        loss_flow = 0.
        for key, val in losses.items():
            if key.startswith('flow'):
                loss_flow += val
            if key.startswith('disp'):
                loss_disp += val
        losses["flow"] = loss_flow
        losses["disp"] = loss_disp

        if self.args.loss_balance_sf_disp:
            loss_sflow_lambda = losses["disp"].detach() / losses["sf"].detach()

            if loss_sflow_lambda > 1.0:
                losses["sf"] = loss_sflow_lambda * losses["sf"]
            else:
                losses["disp"] = (1. / loss_sflow_lambda) * losses["disp"]

        losses["total"] = losses["sf"] + losses["disp"]
        """

        loss_total = 0.0
        for key, val in losses.items():
            loss_total += val

        if "disp_se3_photo" in losses.keys():
            weight_batch = losses["disp_se3_photo"].item()
        else:
            weight_batch = 1.0

        loss_total *= weight_batch

        losses["total"] = loss_total
        states["disp-lvl0-avg"] = torch.mean(list_disps[0]).item() / 832.0
        states["disp-lvl4-avg"] = torch.mean(list_disps[-1]).item() / 52.0

        states["flow-lvl0-avg"] = torch.mean(torch.norm(list_flows[0], dim=1)).item()
        states["flow-lvl4-avg"] = torch.mean(torch.norm(list_flows[-1], dim=1)).item()

        if se3s_6d is not None:
            states["transl-norm"] = torch.norm(se3s_6d[:, :, 3:], dim=2).mean().item()
            states["rot-norm"] = torch.norm(se3s_6d[:, :, :3], dim=2).mean().item()

        # print(transf.get_matrix().shape)
        """
        states["disp-lvl01-avg"] = torch.mean(list_disps[1]).item() / 832.
        states["sflow-lvl0-abs-avg"] = torch.mean(torch.abs(list_sflows[0])).item()
        states["sflow-lvl01-abs-avg"] = torch.mean(torch.abs(list_sflows[1])).item()
        states["sflow-lvl5-abs-avg"] = torch.mean(torch.abs(list_sflows[5])).item()
        """

        return losses, states
