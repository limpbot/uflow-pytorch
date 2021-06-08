import unittest

from util import helpers
import torch

from usflow import options

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

    def test_charbonnier_loss(self):
        B = 3
        H = 160
        W = 160
        fb_sigma = 0.3
        teacher_flow = torch.randn(size=(B, 2, H, W)) * 1 + 30
        student_flow = torch.randn(size=(B, 2, H, W)) * 1 + 30

        teacher_mask = torch.rand(size=(B, 1, H, W))
        student_mask = torch.rand(size=(B, 1, H, W))

        num_pairs = B

        mask = student_mask * teacher_mask  # * valid_warp_mask (forward)
        mask = mask.detach()

        teacher_flow = teacher_flow.detach()
        torch_loss = helpers.calc_charbonnier_loss(student_flow, teacher_flow, mask)

        teacher_flow = torch.flip(teacher_flow, [1])
        student_flow = torch.flip(student_flow, [1])
        teacher_flow = self.torch_to_tf(teacher_flow)
        student_flow = self.torch_to_tf(student_flow)
        student_mask = self.torch_to_tf(student_mask)
        teacher_mask = self.torch_to_tf(teacher_mask)

        error = uflow_helpers.robust_l1(tf.stop_gradient(teacher_flow) - student_flow)
        mask = tf.stop_gradient(teacher_mask * student_mask)
        tf_loss = tf.reduce_sum(input_tensor=mask * error) / (
            tf.reduce_sum(input_tensor=tf.ones_like(mask)) + 1e-16
        )

        print(tf_loss.numpy(), torch_loss.item())
        print(tf_loss.numpy() - torch_loss.item())

    def test_fb_consistency(self):
        B = 3
        H = 160
        W = 160
        fb_sigma = 0.3
        flow_forward = torch.randn(size=(B, 2, H, W)) * 1 + 30
        flow_backward = torch.randn(size=(B, 2, H, W)) * 1 + 30

        # torch_coords_flow_forward = helpers.flow2coords(flow_forward)
        torch_mask_flow_inside = helpers.calc_masks_flow_inside(flow_forward)
        # torch_mask_flow_inside = torch.ones(size=(B, 1, H, W))
        torch_fb_consistency = helpers.calc_fb_consistency(
            flow_forward, flow_backward, fb_sigma
        )

        flow_forward = torch.flip(flow_forward, [1])
        flow_backward = torch.flip(flow_backward, [1])
        flow_forward = self.torch_to_tf(flow_forward)
        flow_backward = self.torch_to_tf(flow_backward)

        tf_coords_flow_forward = uflow_helpers.flow_to_warp(flow_forward)
        tf_mask_flow_inside = uflow_helpers.mask_invalid(
            uflow_helpers.flow_to_warp(flow_forward)
        )
        # tf_mask_flow_inside = self.torch_to_tf(torch_mask_flow_inside)

        flow_backward_warped = uflow_helpers.resample(
            flow_backward, uflow_helpers.flow_to_warp(flow_forward)
        )
        fb_sq_diff = tf.reduce_sum(
            input_tensor=(flow_forward + flow_backward_warped) ** 2,
            axis=-1,
            keepdims=True,
        )
        tf_fb_consistency = (
            tf.exp(-fb_sq_diff / (fb_sigma ** 2 * (H ** 2 + W ** 2)))
            * tf_mask_flow_inside
        )

        tf_fb_consistency = self.tf_to_torch(tf_fb_consistency)
        tf_mask_flow_inside = self.tf_to_torch(tf_mask_flow_inside)
        tf_coords_flow_forward = self.tf_to_torch(tf_coords_flow_forward)
        # tf_coords_flow_forward = torch.flip(tf_coords_flow_forward, [1])

        """
        for i in range(B):
            helpers.visualize_flow(torch.cat((torch_coords_flow_forward[i], tf_coords_flow_forward[i]), dim=2))
            helpers.visualize_flow(torch.abs((torch_coords_flow_forward[i] - tf_coords_flow_forward[i])))
            print('coords diff:', torch.sum(torch.abs(torch_coords_flow_forward - tf_coords_flow_forward)))
        
        for i in range(B):
            helpers.visualize_img(torch.cat((torch_mask_flow_inside[i], tf_mask_flow_inside[i]), dim=2))
            helpers.visualize_img(torch.abs((torch_mask_flow_inside[i] - tf_mask_flow_inside[i])))
            print('mask diff:', torch.sum(torch.abs(torch_mask_flow_inside - tf_mask_flow_inside)))
        for i in range(B):
            helpers.visualize_img(torch.cat((torch_fb_consistency[i], tf_fb_consistency[i]), dim=2))
            helpers.visualize_img(torch.abs((torch_fb_consistency[i] - tf_fb_consistency[i])))
            print('fb consistency diff:', torch.sum(torch.abs(torch_fb_consistency - tf_fb_consistency)))
        """

        print(torch.max(torch.abs(torch_mask_flow_inside - tf_mask_flow_inside)))
        print(torch.max(torch.abs(torch_fb_consistency - tf_fb_consistency)))

    def test_ssim(self):
        def _SSIM(x, y):
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            # (C3 = C2 / 2)
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
            SSIM = SSIM_n / SSIM_d  # range [-1, 1]
            DSSIM = (1 - SSIM) / 2  # range [ 0, 1]

            return DSSIM

        B = 2
        C = 3
        H = 280
        W = 640
        x1 = torch.rand(size=(B, C, H, W))
        x2 = torch.rand(size=(B, C, H, W))

        ssim = _SSIM(x1, x2)

        pass

    def test_chamfer(self):
        B = 2
        C = 3
        H = 160
        W = 160
        # 2x3x160x160 -> 1.3 GB allocation
        pts1 = torch.randn(size=(B, C, H, W)).cuda()
        pts2 = pts1
        helpers.calc_chamfer_dist(pts1, pts2)

    def test_rec_loss(self):

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
        pts_diff1[~occ_map_f].detach_()
        pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

    def test_sf_smoothness_loss(self):

        ## 3D motion smoothness loss
        loss_3d_s = (
            (
                _smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)
            ).mean()
            + (
                _smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)
            ).mean()
        ) / (2 ** ii)


if __name__ == "__main__":
    unittest.main()
