
import torch
import torch.nn as nn
from architectures.se3 import SE3Net
from architectures.disp import DispNet
from architectures.mask import MaskNet
from architectures.uflow import UFlow

class USFlow(nn.Module):
    def __init__(self, args):
        super(USFlow, self).__init__()

        self.args = args

        if self.args.arch_modules_masks_num_outs > 0:
            self.module_mask = MaskNet(args)
        else:
            self.module_mask = None

        if self.args.arch_modules_masks_num_outs > 0 or (
                self.args.arch_modules_masks_num_outs == 0
                and self.args.arch_se3_egomotion_addition
        ):
            self.module_se3 = SE3Net(args)
        else:
            self.module_se3 = None

        self.module_flow = UFlow(args)
        self.module_disp = DispNet(args)

        self.module_upsample_x2 = torch.nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=True
        )

        self.weights_means = None
        self.weights_vars = None
        self.bias_means = None
        self.bias_vars = None
        self.check_params()
        self.freeze_params()

    def forward(self, imgpair, proj_mats, reproj_mats):
        B, _, H, W = imgpair.shape

        if self.args.arch_flow_out_channels == 2:
            flows = self.module_flow(imgpair, proj_mats, reproj_mats)
        else:
            flows, disps = self.module_flow(imgpair, proj_mats, reproj_mats)

        if self.args.arch_disp_separate:
            disp_in = imgpair[:, :3, :, :]
            disps = self.module_disp(disp_in)
            disps.insert(0, disps[0])
            for lvl_id in range(len(disps)):
                disps[lvl_id] = 2.0 * self.module_upsample_x2(disps[lvl_id])

        if self.module_se3 is not None:
            se3_feats = self.module_se3.encode(imgpair)

            if self.args.arch_se3_encoded_cat_oflow:
                for lvl_id in range(len(se3_feats)):
                    _, _, H_lvl, W_lvl = se3_feats[lvl_id].shape
                    se3_feats[lvl_id] = torch.cat(
                        (
                            se3_feats[lvl_id],
                            torch.nn.functional.interpolate(
                                flows[2],
                                size=(H_lvl, W_lvl),
                                mode="bilinear",
                                align_corners=True,
                            )
                            / (W_lvl * 2 ** lvl_id),
                        ),
                        dim=1,
                    )
            transf_6d = self.module_se3.decode(se3_feats[-1])
        else:
            transf_6d = None

        if self.module_mask is not None:
            masks = self.module_mask(se3_feats)
            masks.insert(0, masks[0])

            for lvl_id in range(len(masks)):
                masks[lvl_id] = self.module_upsample_x2(masks[lvl_id])
        else:
            masks = [None] * len(disps)

        return flows, disps, transf_6d, masks

    def freeze_params(self):
        if self.args.model_freeze_modules is not None:
            for name, param in self.named_parameters():
                for module_freeze in self.args.model_freeze_modules:
                    if name.startswith(module_freeze):
                        param.requires_grad = False

    def check_params(self):
        mean_thresh = 1e-06
        var_thresh = 1e-06

        weights_means = {}
        weights_vars = {}
        bias_means = {}
        bias_vars = {}

        # self.modules_features
        # self.modules_context
        # self.modules_flow
        # self.modules_context_upsampling
        # self.module_refinement

        for name, param in self.named_parameters():
            if len(param.size()) == 1:
                var, mean = torch.var_mean(param)
                bias_means[name] = mean
                bias_vars[name] = var
                if self.bias_means is not None:
                    if torch.max(torch.abs(mean - self.bias_means[name])) < mean_thresh:
                        print("warning: ", name, "consistent bias")

            else:
                var, mean = torch.var_mean(param)
                weights_means[name] = mean
                weights_vars[name] = var

                if self.weights_means is not None:
                    if (
                        torch.max(torch.abs(mean - self.weights_means[name]))
                        < mean_thresh
                    ):
                        print("warning: ", name, "consistent weight")

        self.weights_means = weights_means
        self.weights_vars = weights_vars
        self.bias_means = bias_means
        self.bias_vars = bias_vars