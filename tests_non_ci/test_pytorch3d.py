import unittest

import torch
import pytorch3d.transforms as transf3d

from tensor_operations import transforms3d

from usflow import options

parser = options.setup_comon_options()

# preliminary_args = ['-s', '../config/config_setup_0.yaml', '-c', '../config/config_coach_uflow_0.yaml']
# args = parser.parse_args(preliminary_args)


class Tests(unittest.TestCase):
    def test_clone(self):
        B = 1
        N = 1
        H = 10
        W = 10
        transf_6d = torch.randn(N, 6).cuda()
        transf_6d[:, :3] = transf_6d[:, :3] * 0.0
        transf_6d[:, 3:] = transf_6d[:, 3:] * 0.0 + 1.0

        t1 = transf3d.Rotate(
            transf3d.axis_angle_to_matrix(transf_6d[:, :3]), device="cuda"
        ).translate(transf_6d[:, 3:])

        pts1 = torch.randn(B, 3, H, W).cuda()

        pts2 = (
            t1.transform_points(pts1.reshape(B, 3, H * W).permute(0, 2, 1))
            .permute(0, 2, 1)
            .reshape(B, 3, H, W)
        )

        pass

    def test_create_transf3d(self):
        def transformation_from_parameters(axisangle, translation, invert=False):
            """Convert the network's (axisangle, translation) output into a 4x4 matrix"""
            R = rot_from_axisangle(axisangle)
            t = translation.clone()

            if invert:
                R = R.transpose(1, 2)
                t *= -1

            T = get_translation_matrix(t)

            if invert:
                M = torch.matmul(R, T)
            else:
                M = torch.matmul(T, R)

            return M

        def get_translation_matrix(translation_vector):
            """Convert a translation vector into a 4x4 transformation matrix"""
            T = torch.zeros(translation_vector.shape[0], 4, 4).to(
                device=translation_vector.device
            )

            t = translation_vector.contiguous().view(-1, 3, 1)

            T[:, 0, 0] = 1
            T[:, 1, 1] = 1
            T[:, 2, 2] = 1
            T[:, 3, 3] = 1
            T[:, :3, 3, None] = t

            return T

        def rot_from_axisangle(vec):
            """Convert an axisangle rotation into a 4x4 transformation matrix
            (adapted from https://github.com/Wallacoloo/printipi)
            Input 'vec' has to be Bx1x3
            """
            angle = torch.norm(vec, 2, 2, True)
            axis = vec / (angle + 1e-7)

            ca = torch.cos(angle)
            sa = torch.sin(angle)
            C = 1 - ca

            x = axis[..., 0].unsqueeze(1)
            y = axis[..., 1].unsqueeze(1)
            z = axis[..., 2].unsqueeze(1)

            xs = x * sa
            ys = y * sa
            zs = z * sa
            xC = x * C
            yC = y * C
            zC = z * C
            xyC = x * yC
            yzC = y * zC
            zxC = z * xC

            rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

            rot[:, 0, 0] = torch.squeeze(x * xC + ca)
            rot[:, 0, 1] = torch.squeeze(xyC - zs)
            rot[:, 0, 2] = torch.squeeze(zxC + ys)
            rot[:, 1, 0] = torch.squeeze(xyC + zs)
            rot[:, 1, 1] = torch.squeeze(y * yC + ca)
            rot[:, 1, 2] = torch.squeeze(yzC - xs)
            rot[:, 2, 0] = torch.squeeze(zxC - ys)
            rot[:, 2, 1] = torch.squeeze(yzC + xs)
            rot[:, 2, 2] = torch.squeeze(z * zC + ca)
            rot[:, 3, 3] = 1

            return rot

        B = 6
        H = 10
        W = 10
        transf_6d = torch.randn(B, 6).cuda() * 10.0 + 3.0
        transf_6d.requires_grad = True
        transf1_se3mat = transformation_from_parameters(
            transf_6d[:, :3].unsqueeze(1), transf_6d[:, 3:].unsqueeze(1)
        )

        transf2_se3mat = transforms3d.se3_6d_to_se3mat(transf_6d, type="axisangle")

        transf3_se3mat = transforms3d.se3_6d_to_se3mat(transf_6d, type="exponential")

        transf4_se3mat = transforms3d.se3_6d_to_se3mat(transf_6d, type="rodrigues")

        print(transf1_se3mat - transf2_se3mat)

        pts3d = torch.randn(B, 3, H * W).cuda()
        pts4d = torch.cat(
            (
                pts3d,
                torch.ones(size=(B, 1, H * W), dtype=pts3d.dtype, device=pts3d.device),
            ),
            dim=1,
        )
        pts3d_ftf1 = torch.matmul(transf1_se3mat, pts4d)[:, :3]

        pts3d_ftf2 = transforms3d.pts3d_transform(
            pts3d.permute(0, 2, 1), transf1_se3mat
        ).permute(0, 2, 1)

        print(torch.norm(pts3d_ftf1 - pts3d_ftf2, dim=1).mean())
        pass

    def test_transf3d(self):
        B = 2
        N = 2
        H = 10
        W = 10
        transf_6d = torch.zeros(N, 6).cuda() + 3.0
        # transf_6d[:, :3] = transf_6d[:, :3] * 0.
        transf_6d[:, 4] = 0.1
        # transf_6d[:, 3:] = transf_6d[:, 3:] * 0.

        # so3_type = 'axisangle'
        so3_type = "exponential"
        # so3_type = 'rodrigues'
        se3mat = transforms3d.se3_6d_to_se3mat(transf_6d, type=so3_type)
        print(transf_6d, se3mat)
        torch.manual_seed(26)
        pts1 = torch.randn(B, 3, H, W).cuda()
        # pts1[:, 0] = 1
        # pts1[:, 1] = 0
        # pts1[:, 2] = 0

        pts2 = transforms3d.pts3d_transform(pts1, se3mat)

        transf_est = transforms3d.calc_transform_between_pointclouds_v2(
            pts1[0, :].flatten(1), pts2[0, :].flatten(1), so3_type=so3_type
        )
        print(transf_est)
        print(transf_est - se3mat[0])

        # print(pts1[0, :, 0, 0])
        # print(pts2[0, :, 0, 0])

        # pts_diff = pts2 - pts1
        pass
