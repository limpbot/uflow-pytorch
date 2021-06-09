import os
import torch
import cv2
from usflow.data import create_present_dataloader

from clerk import Clerk
from usflow.plusflow import PLUSFlow
from run_manager import RunManager

import tensor_operations.vision as ops_vis
import tensor_operations.geometric as ops_geo
import tensor_operations.mask as ops_mask
import tensor_operations.warp as ops_warp
import tensor_operations.rearrange as ops_rearr
import tensor_operations.registration as ops_reg

from usflow import options
import sys


class Presenter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        repro_dir = None
        # repro_dir = os.path.dirname(os.path.realpath(__file__))
        if repro_dir is None:
            repro_dir = "."
        # os.chdir(repro_dir)

        parser = options.setup_comon_options()
        args = parser.parse_args()

        args.repro_dir = repro_dir

        # args.model_load_modules
        # args = args
        args.debug_mode = "_pydev_bundle.pydev_log" in sys.modules.keys()
        if args.debug_mode:
            print("DEBUG MODE")
            args.dataloader_num_workers = 0

        run_manager = RunManager(args)

        self.args = run_manager.args

        self.clerk = Clerk(self.args)

        self.model = PLUSFlow(args=self.args)

        tag = "latest"
        self.clerk.load_model_state(
            self.model.architecture, tag, modules=self.args.model_load_modules
        )

        self.args.dataloader_device = None  # self.device
        self.args.dataloader_pin_memory = False

        if self.args.dataloader_num_workers > 0:
            self.args.dataloader_device = "cpu"
            self.args.dataloader_pin_memory = True

        self.args.present_fname_imgs_filenames = "present_short_files.txt"
        self.args.present_fname_imgs_filenames = "present_files.txt"
        # self.args.present_fname_imgs_filenames = "train_files.txt"
        # self.args.present_fname_imgs_filenames = "filtered_files.txt"
        self.args.present_dataset_name = "smsf_eigen_zhou"
        self.args.present_dataset_val = True
        # self.args.present_dataset_name = "smsf"
        self.present_dataloader = create_present_dataloader(self.args)

    def present(self):

        valid_indices = torch.zeros(size=(len(self.present_dataloader),))

        save_video = True
        save_imgs = True
        fps = 1 #10
        save_filtered = False
        show_duration = 0  # None, 1
        id_start = 2 # 0 video: 100 img: 394
        id_end = 20
        name = "flow_comp"
        # name = "se3_mask"
        # name = "disp"
        # name = "se3_oflow"
        # name = "net_oflow_disp"
        name = "net_oflow"
        # name = "se3-sflow"
        # name = "se3_oflow_disp"
        # name = "se3_mask"
        if name == "flow_comp":
            outs = ["net_oflow", "se3_oflow"]
        elif name == "se3_mask_disp":
            outs = ["disp", "mask", "se3_oflow"]
        elif name == "se3_mask":
            outs = ["mask", "se3_oflow"]
        elif name == "net_oflow_disp":
            outs = ["net_oflow", "disp"]
        elif name == "se3_oflow_disp":
            outs = ["disp", "se3_oflow"]
        else:
            outs = [name]
        outs = ["labels"]
        # "mask", "net_oflow", "se3_oflow", "disp"

        if save_video:
            vwriter = ops_vis.create_vwriter(
                name, self.args.arch_res_width, self.args.arch_res_height * len(outs),
            fps)
        else:
            vwriter = None

        self.model.to(self.device)
        self.model.eval()
        for (i, batch) in enumerate(self.present_dataloader):
            if (i + 1) % 100 == 0:
                print(i + 1)
            print('id', i)
            if i < id_start or i > id_end:
                continue
            visuals = {}
            # if i > 1000:
            #    break

            for el_id in range(len(batch)):
                batch[el_id] = batch[el_id].to(self.device)

            if not self.args.present_dataset_val:
                (
                    imgpairs_left_fwd,
                    imgpairs_right_fwd,
                    proj_mats_left_fwd,
                    proj_mats_left_fwd,
                ) = batch
            else:
                (
                    imgpairs_left_fwd,
                    gt_flow_noc_uv_fwd,
                    gt_flow_noc_valid_fwd,
                    gt_flow_occ_uv_fwd,
                    gt_flow_occ_valid_fwd,
                    gt_disps_left_fwd,
                    gt_disps_masks_left_fwd,
                    gt_disps_left_bwd,
                    gt_disps_masks_left_bwd,
                    gt_se3s_l1,
                    proj_mats_left_fwd,
                    reproj_mats_left_fwd,
                ) = batch

            imgs = imgpairs_left_fwd[:, :3]

            imgpairs_left_bwd = ops_rearr.imgpairs_swap_order(imgpairs_left_fwd)

            imgpairs_left = torch.cat((imgpairs_left_fwd, imgpairs_left_bwd), dim=0)
            proj_mats_left = proj_mats_left_fwd.repeat(2, 1, 1)
            reproj_mats_left = reproj_mats_left_fwd.repeat(2, 1, 1)

            list_flows, list_disps, se3s_6d, list_masks = self.model.forward(
                imgpairs_left, proj_mats_left, reproj_mats_left
            )

            # list_flows, list_disps, se3s_6d, list_masks = self.model.forward(
            #    imgpairs_left_fwd, proj_mats_left_fwd, reproj_mats_left_fwd
            # )

            se3s_6d = se3s_6d[:1]
            flows = list_flows[0][:1]
            disps_l1 = list_disps[0][:1]
            disps_l2 = list_disps[0][1:]
            masks = list_masks[0][:1]

            gt_pts3d_l1 = ops_geo.disp_2_pt3d(
                gt_disps_left_fwd, proj_mats_left_fwd, reproj_mats_left_fwd
            )
            gt_pts3d_l2 = ops_geo.disp_2_pt3d(
                gt_disps_left_bwd,
                proj_mats_left_fwd,
                reproj_mats_left_fwd,
                gt_flow_occ_uv_fwd,
            )
            gt_mask_pts3d_valid = (
                gt_disps_masks_left_fwd
                * gt_disps_masks_left_bwd
                * gt_flow_noc_valid_fwd
            )

            pts3d_l1 = ops_geo.disp_2_pt3d(
                disps_l1, proj_mats_left_fwd, reproj_mats_left_fwd
            )

            pts3d_l2 = ops_geo.disp_2_pt3d(
                disps_l2, proj_mats_left_fwd, reproj_mats_left_fwd
            )

            if se3s_6d is not None:
                se3s_mat = ops_geo.se3_6d_to_se3mat(se3s_6d)
                pts3d_l1_se3_ftf = ops_geo.pts3d_transform_obj_ego(
                    pts3d_l1,
                    se3s_mat,
                    masks,
                    egomotion_addition=self.args.arch_se3_egomotion_addition,
                )
            else:
                pts3d_l1_se3_ftf = pts3d_l1 + flows

            pts3d_l2_bwrpd = ops_warp.warp(pts3d_l2, flows)
            #flows = ops_geo.pt3d_2_oflow(pts3d_l2_bwrpd, proj_mats=proj_mats_left_fwd)

            down_scale = 0.1

            pts3d_l1_down = ops_vis.resize(
                pts3d_l1, scale_factor=down_scale, mode="nearest"
            )

            pts3d_l2_bwrpd_down = ops_vis.resize(
                pts3d_l2_bwrpd, scale_factor=down_scale, mode="nearest"
            )

            flows_down = ops_vis.resize(
                flows, scale_factor=down_scale, mode="nearest"
            )

            H_down = pts3d_l1_down.size(2)
            W_down = pts3d_l1_down.size(3)

            #x1 = pts3d_l1_down.flatten(2)
            #x2 = pts3d_l2_bwrpd_down.flatten(2)

            #eps = 0.05 # 0.01
            min_samples = 5
            thresh_overlap = 0.8
            num_rounds = 1

            #eps = 0.05 for gt
            rigid_dists_max_div = 0.05 # 0.05
            rigid_dist_max = 1.0
            grid_sample_rate = 0.05 # 0.05

            thresh_transf_dist = 0.1 # 0.2
            se3_sim_dist_thresh = 0.1 # 0.1 # 0.1 / 5
            se3_sim_angle_thresh = 5 #5 / 360 * 2 * np.pi
            se3_reg_thresh = 0.1

            erode_patchsize = 3
            erode_threshold = 0.2

            visualize_single_masks = True
            no_gt = False

            if no_gt:
                gt_pts3d_l1 = pts3d_l1
                gt_pts3d_l2 = pts3d_l2_bwrpd
                gt_mask_pts3d_valid = gt_mask_pts3d_valid
                gt_mask_pts3d_valid = ops_vis.resize(
                    gt_mask_pts3d_valid * 1.0, H_out=gt_pts3d_l1.size(2), W_out=gt_pts3d_l1.size(3), mode="nearest"
                ) == 1.0
                gt_mask_pts3d_valid[:] = True

            gt_pts3d_l1_down = ops_vis.resize(
                gt_pts3d_l1, scale_factor=down_scale, mode="nearest"
            )
            gt_pts3d_l2_down = ops_vis.resize(
                gt_pts3d_l2, scale_factor=down_scale, mode="nearest"
            )

            gt_mask_pts3d_valid_down = ops_vis.resize(
                gt_mask_pts3d_valid * 1.0, scale_factor=down_scale, mode="nearest"
            ) == 1.0

            H_down = gt_pts3d_l1_down.size(2)
            W_down = gt_pts3d_l1_down.size(3)
            N_down = H_down * W_down

            x1 = gt_pts3d_l1_down.flatten(2)
            x2 = gt_pts3d_l2_down.flatten(2)

            v1 = torch.norm((x1[:, :, :, None] - x1[:, :, None, :]), dim=1)
            v2 = torch.norm((x2[:, :, :, None] - x2[:, :, None, :]), dim=1)

            flows_down = flows_down.flatten(2)
            flows_down_dists = (flows_down[:, :, :, None] - flows_down[:, :, None, :])
            dists_div = torch.abs(v2 - v1) # / (v1 + v2)
            dists_div = dists_div[0]

            N = dists_div.size(0)
            connected = (dists_div < rigid_dists_max_div) * gt_mask_pts3d_valid_down[0].flatten(1)
            connected = connected * (v1[0] < rigid_dist_max) * (v2[0] < rigid_dist_max)

            core_ids_pot = torch.arange(0, N, int(1/grid_sample_rate))
            objects_masks = connected[core_ids_pot]

            K = len(objects_masks)

            objects_masks = objects_masks.reshape(K, H_down, W_down)

            for r in range(num_rounds):
                K = len(objects_masks)
                print("init masks", K)

                objects_masks = ops_mask.filter_size(objects_masks, min_samples=min_samples)
                K = len(objects_masks)
                print("masks after filter size", K)
                if visualize_single_masks:
                    ops_vis.visualize_img(objects_masks.reshape(1, K * H_down, W_down))

                objects_masks = ops_mask.filter_interconnected(objects_masks, dists_div, rigid_dists_max_div)
                print("masks after interconnected check", K)
                if visualize_single_masks:
                    ops_vis.visualize_img(objects_masks.reshape(1, K * H_down, W_down))

                objects_masks = ops_mask.filter_size(objects_masks, min_samples=min_samples)
                K = len(objects_masks)
                print("masks after filter size", K)
                if visualize_single_masks:
                    ops_vis.visualize_img(objects_masks.reshape(1, K * H_down, W_down))

                objects_masks = ops_mask.filter_overlap(objects_masks, max_overlap=thresh_overlap)
                K = len(objects_masks)
                print("masks after filter overlap", K)
                if visualize_single_masks:
                    ops_vis.visualize_img(objects_masks.reshape(1, K * H_down, W_down))


                objects_se3s, objects_se3s_devs = ops_reg.register_objects(objects_masks,
                                                                           gt_pts3d_l1_down[0],
                                                                           gt_pts3d_l2_down[0],
                                                                           thresh=se3_reg_thresh)

                K = len(objects_se3s)
                print("masks after se3 registration", K)

                objects_se3s = ops_reg.filter_sim_se3(objects_se3s, objects_se3s_devs,
                                                      dist_thresh=se3_sim_dist_thresh,
                                                      angle_thresh=se3_sim_angle_thresh)

                K = len(objects_se3s)
                print("masks after se3 filter similarity", K)

                # assign labels
                gt_pts3d_l1_down_ftf = ops_geo.pts3d_transform(gt_pts3d_l1_down.repeat(K, 1, 1, 1), objects_se3s)
                dev = torch.norm(gt_pts3d_l1_down_ftf - gt_pts3d_l2_down, dim=1)
                objects_masks = (dev < thresh_transf_dist) * gt_mask_pts3d_valid_down[0]

                print("masks after threshold testing")
                #objects_masks = ops_vis.choose_from_neighborhood(1.0 * objects_masks[None, ], patch_size=3)[0]

                if visualize_single_masks:
                    ops_vis.visualize_img(objects_masks.reshape(1, K * H_down, W_down))


            K = len(objects_se3s)

            gt_pts3d_l1_ftf = ops_geo.pts3d_transform(gt_pts3d_l1.repeat(K, 1, 1, 1), objects_se3s)
            dev = torch.norm(gt_pts3d_l1_ftf - gt_pts3d_l2, dim=1, keepdim=True)
            objects_masks = torch.argmin(dev, 0, keepdim=True)
            objects_masks[:, :, ~gt_mask_pts3d_valid[0, 0]] = K
            objects_masks = ops_vis.label2onehot(objects_masks)

            #objects_masks = 1.0 * ops_vis.choose_from_neighborhood(objects_masks, patch_size=3)

            #dists = torch.exp(-dists**2 / 0.5)
            # ops_vis.visualize_img(dists)
            #dists = torch.norm(flows_down_dists[:, :2], dim=1)
            #dists = dists[0].detach().cpu().numpy()

            #dists = (dists > 0.05) * 10.0
            #optical flow 0.2
            # clustering = DBSCAN(eps=0.0005, min_samples=3, metric="precomputed").fit(dists)
            # discretize / kmeans
            # clustering = SpectralClustering(n_clusters=10, assign_labels='discretize', random_state=0, affinity='precomputed').fit(dists)
            # labels = torch.from_numpy(clustering.labels_).to(device)
            # dists_masks = torch.arange(labels.min(), labels.max()+1).to(device)[:, None] == labels[None, :]
            # K = labels.max()+1 - labels.min()
            # dists_masks = dists_masks.reshape(K, H_down, W_down)
            #ops_vis.visualize_img(ops_vis.mask2rgb(dists_masks))
            #for i in range(H_down * W_down):
            #    dists_single = dists[:, i]
            #    ops_vis.visualize_img(dists_single.reshape(1, H_down, W_down))


            pxls2d_wrpd = ops_geo.pt3d_2_pxl2d(pts3d_l1_se3_ftf, proj_mats_left_fwd)
            oflows = ops_geo.pxl2d_2_oflow(pxls2d_wrpd)

            mask_inside = ops_mask.oflow_2_mask_inside(oflows)

            """
            pts1 = pts3d[0]
            pts1 = (
                pts1[(mask_inside[0] > 0.0).repeat(3, 1, 1)]
                .reshape(3, -1)
                .permute(1, 0)
            )
            pts2 = pts3d_l1_se3_ftf[0]
            pts2 = (
                pts2[(mask_inside[0] > 0.0).repeat(3, 1, 1)]
                .reshape(3, -1)
                .permute(1, 0)
            )

            img = imgs[0]
            img = (
                img[(mask_inside[0] > 0.0).repeat(3, 1, 1)].reshape(3, -1).permute(1, 0)
            )
            img_sflow = ext_vis.visualize_sflow(pts1, pts2, img)
            img_sflow = torch.from_numpy(np.array(img_sflow)).permute(2, 0, 1)
            
            visuals["se3-sflow"] = img_sflow
            """

            # ext_vis.visualize_img(img_sflow, duration=1)
            # oflows_diff = (torch.norm(oflows - flows, dim=1, p=2) * mask_inside).mean()
            # print("oflows_diff", oflows_diff)
            # if oflows_diff >= 5:
            #    valid_indices[i] = 1

            alpha_img_mask = 0.2
            alpha_img_oflow = 0.05

            #visuals["labels"] = ops_vis.resize(ops_vis.mask2rgb(dists_masks)[None, ], H_out=disps_l1.size(2),W_out=disps_l1.size(3), mode="nearest")[0]
            visuals["labels"] = ops_vis.resize(ops_vis.mask2rgb(objects_masks[0])[None, ], H_out=disps_l1.size(2),W_out=disps_l1.size(3), mode="nearest")[0]

            visuals["disp"] = ops_vis.disp2rgb(disps_l1[0])

            if masks is not None:
                visuals["mask"] = alpha_img_mask * imgs[0] + (
                    1 - alpha_img_mask
                ) * ops_vis.mask2rgb(masks[0], True)

            if flows.size(1) == 2:
                visuals["net_oflow"] = alpha_img_oflow * imgs[0] + (
                    1 - alpha_img_oflow
                ) * ops_vis.flow2rgb(flows[0], draw_arrows=False)

            visuals["se3_oflow"] = alpha_img_oflow * imgs[0] + (
                1 - alpha_img_oflow
            ) * ops_vis.flow2rgb(oflows[0], draw_arrows=False)

            visual = None
            for key in outs:
                if visual == None:
                    visual = visuals[key]
                else:
                    visual = torch.cat((visual, visuals[key]), dim=1)

            if visual is not None:
                if save_imgs:
                    fpath = 'images/img' + str(i) + '.png'
                ops_vis.visualize_img(
                    visual,
                    duration=show_duration,
                    vwriter=vwriter,
                    fpath=fpath
                )

        if save_filtered:

            fp_imgs_filenames = os.path.join(
                self.args.repro_dir,
                "datasets/kitti_raw_meta",
                "lists_imgpair_filenames",
                self.args.present_dataset_name,
                self.args.present_fname_imgs_filenames,
            )

            fp_imgs_filenames_filtered = os.path.join(
                self.args.repro_dir,
                "datasets/kitti_raw_meta",
                "lists_imgpair_filenames",
                self.args.present_dataset_name,
                "train_filtered_files.txt",
            )
            with open(fp_imgs_filenames_filtered, "w") as fout:
                with open(fp_imgs_filenames, "r") as fin:
                    for (i, lin) in enumerate(fin.readlines()):
                        if valid_indices[i] != 0:
                            fout.write(lin)

        if vwriter is not None:
            vwriter.release()
        cv2.destroyAllWindows()


def main():
    presenter = Presenter()
    presenter.present()


if __name__ == "__main__":
    main()
