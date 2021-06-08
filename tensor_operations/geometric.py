import torch
import pytorch3d.transforms as t3d


def disp_2_depth(disp, fx):
    # disp: Bx1xHxW
    # note for kitti-dataset: baseline=0.54 -> 54 cm
    # depth = focal-length * baseline / disparity
    disp = torch.clamp(disp, 0)
    fx = fx[..., None, None, None]

    depth = fx * 0.54 / (disp + 1e-8)
    depth = torch.clamp(depth, 1e-3, 80)

    # https: // github.com / visinf / self - mono - sf / tree / master / models

    return depth


def depth_2_disp(depth, fx):
    fx = fx[..., None, None, None]

    disp = fx * 0.54 / (depth + 1e-8)

    return disp


def disp_2_pt3d(disp, proj_mats, reproj_mats, oflow=None):
    depth = disp_2_depth(disp, fx=proj_mats[:, 0, 0])
    xyz = depth_2_pt3d(depth, reproj_mats, oflow)
    return xyz


def pt3d_2_pxl2d(pt3d, proj_mats):
    # 3D-2D Projection:
    # u = (fx*x + cx * z) / z
    # v = (fy*y + cy * z) / z
    # shift on plane: delta_x = (fx * bx) / z
    #                 delta_y = (fy * by) / z
    # uv = (P * xyz) / z
    # P = [ fx   0  cx]
    #     [ 0   fy  cy]

    B, _, H, W = pt3d.shape

    pt3d = pt3d.reshape(B, 3, -1)
    # 3 x N
    # z = torch.abs(xyz[:, 2].clone()) + 1e-8
    # uv = torch.matmul(proj_mats, xyz[:, :2]) / z.unsqueeze(1)

    pt3d[:, 2] = torch.abs(pt3d.clone()[:, 2]) + 1e-8
    # z = torch.abs(pt3d[:, 2]) + 1e-8
    uv = torch.matmul(proj_mats, pt3d / (pt3d[:, 2] + 1e-8).unsqueeze(1))
    # uv = uv.type_as(pt3d)
    # uv = torch.div(torch.matmul(proj_mats, xyz), (xyz[:, 2] + 1e-8).unsqueeze(1))

    # 2xN
    uv = uv.reshape(B, 2, H, W)

    return uv


def pt3d_2_oflow(pt3d, proj_mats):
    pxl2d = pt3d_2_pxl2d(pt3d, proj_mats)
    oflow = pxl2d_2_oflow(pxl2d)
    return oflow


def oflow_2_vec3d(oflow, reproj_mats):
    B, _, H, W = oflow.shape
    depth = torch.ones(size=(B, 1, H, W), dtype=oflow.dtype, device=oflow.device)

    vec3d = depth_2_pt3d(depth, reproj_mats, oflow)

    vec3d = vec3d / torch.norm(vec3d, p=2, dim=1, keepdim=True)

    return vec3d


def oflow_2_vec2d(oflow, reproj_mats):
    B, _, H, W = oflow.shape
    depth = torch.ones(size=(B, 1, H, W), dtype=oflow.dtype, device=oflow.device)

    vec2d = depth_2_pt3d(depth, reproj_mats, oflow)[:, :2]
    # vec2d = vec2d / torch.norm(vec2d, p=2, dim=1, keepdim=True)

    return vec2d


def depth_2_pt3d(depth, reproj_mats, oflow=None):
    B, _, H, W = depth.shape

    dtype = depth.dtype
    device = depth.device

    if oflow == None:
        grid_uv1 = shape_2_pxl3d(B=0, H=H, W=W, dtype=dtype, device=device)
        uv1 = grid_uv1.reshape(3, -1)
        # 3 x N
    else:
        grid_uv1 = oflow_2_pxl3d(oflow)
        uv1 = grid_uv1.reshape(B, 3, -1)

    xyz = torch.matmul(reproj_mats, uv1) * depth.flatten(2)
    # B x 3 x 3 * 3 x N = (B x 3 x N)

    xyz = xyz.reshape(B, 3, H, W)

    # 2D-3D Re-Projection:
    # x = (u/fx - cx/fx) * z
    # y = (v/fy - cy/fy) * z
    # z = z
    # xyz = (RP * uv1) * z
    # RP = [ 1/fx     0  -cx/fx ]
    #      [    0  1/fy  -cy/fy ]
    #      [    0      0      1 ]

    return xyz


def shape_2_pxl3d(B, H, W, dtype, device):

    grid_uv = shape_2_pxl2d(B, H, W, dtype, device)

    if B != 0:
        grid_1 = torch.ones(size=(B, 1, H, W), dtype=dtype, device=device)
        grid_uv1 = torch.cat((grid_uv, grid_1), dim=1)

    else:
        grid_1 = torch.ones(size=(1, H, W), dtype=dtype, device=device)
        grid_uv1 = torch.cat((grid_uv, grid_1), dim=0)

    return grid_uv1


def shape_2_pxl2d(B, H, W, dtype, device):
    grid_v, grid_u = torch.meshgrid(
        [
            torch.arange(0.0, H, dtype=dtype, device=device),
            torch.arange(0.0, W, dtype=dtype, device=device),
        ]
    )

    grid_uv = torch.stack((grid_u, grid_v), dim=0)
    # 2xHxW

    if B != 0:
        grid_uv = grid_uv.unsqueeze(0).repeat(repeats=(B, 1, 1, 1))
        # Bx2xHxW

    return grid_uv


def disp_2_pxl2d(disp):
    # in: Bx1xHxW
    B, _, H, W = disp.shape
    dtype = disp.dtype
    device = disp.device

    grid_uv = shape_2_pxl2d(B=B, H=H, W=W, dtype=dtype, device=device)

    grid_1 = torch.ones(size=(B, 1, H, W), dtype=dtype, device=device)
    grid_uv = grid_uv + torch.cat((disp, grid_1), dim=1)

    return grid_uv


def pxl2d_2_oflow(pxlcoords):
    B, _, H, W = pxlcoords.shape
    dtype = pxlcoords.dtype
    device = pxlcoords.device

    grid_y, grid_x = torch.meshgrid(
        [
            torch.arange(0.0, H, dtype=dtype, device=device),
            torch.arange(0.0, W, dtype=dtype, device=device),
        ]
    )

    grid_xy = torch.stack((grid_x, grid_y), dim=0)

    flow = pxlcoords - grid_xy

    return flow


def oflow_2_pxl2d(flow):
    B, _, H, W = flow.shape
    dtype = flow.dtype
    device = flow.device

    grid_uv = shape_2_pxl2d(B=B, H=H, W=W, dtype=dtype, device=device)

    grid_uv = grid_uv + flow

    return grid_uv


def oflow_2_pxl3d(flow):
    B, _, H, W = flow.shape
    dtype = flow.dtype
    device = flow.device

    grid_uv = oflow_2_pxl2d(flow)

    grid_1 = torch.ones(size=(B, 1, H, W), dtype=dtype, device=device)
    grid_uv1 = torch.cat((grid_uv, grid_1), dim=1)

    return grid_uv1


def oflow_2_pxl2d_normalized(flow):
    grid_xy = oflow_2_pxl2d(flow)

    grid_xy = pxl2d_2_pxl2d_normalized(grid_xy)

    return grid_xy


def pxl2d_2_pxl2d_normalized(grid_xy):
    # ensure normalize pxlcoords is no inplace
    grid_xy = grid_xy.clone()
    B, C, H, W = grid_xy.shape

    grid_xy[:, 0] = grid_xy[:, 0] / (W - 1.0) * 2.0 - 1.0
    grid_xy[:, 1] = grid_xy[:, 1] / (H - 1.0) * 2.0 - 1.0

    return grid_xy


### Apply weight-sharpening to the masks across the channels of the input
### output = Normalize( (sigmoid(input) + noise)^p + eps )
### where the noise is sampled from a 0-mean, sig-std dev distribution (sig is increased over time),
### the power "p" is also increased over time and the "Normalize" operation does a 1-norm normalization
def sharpen_masks(input, add_noise=True, noise_std=0, pow=1):
    input = torch.sigmoid(input)
    if add_noise and noise_std > 0:
        noise = input.new_zeros(*input.size()).normal_(
            mean=0.0, std=noise_std
        )  # Sample gaussian noise
        input = input + noise
    input = (
        torch.clamp(input, min=0, max=100000) ** pow
    ) + 1e-12  # Clamp to non-negative values, raise to a power and add a constant
    return torch.nn.Functional.normalize(
        input, p=1, dim=1, eps=1e-12
    )  # Normalize across channels to sum to 1
    # return input


#############################
### NEW SE3 LAYER (se3toSE3 with some additional transformation for the translation vector)

# Create a skew-symmetric matrix "S" of size [(Bk) x 3 x 3] (passed in) given a [(Bk) x 3] vector
def create_skew_symmetric_matrix(vector):
    # Create the skew symmetric matrix:
    # [0 -z y; z 0 -x; -y x 0]
    N = vector.size(0)
    output = vector.view(N, 3, 1).expand(N, 3, 3).clone()
    output[:, 0, 0] = 0
    output[:, 1, 1] = 0
    output[:, 2, 2] = 0
    output[:, 0, 1] = -vector[:, 2]
    output[:, 1, 0] = vector[:, 2]
    output[:, 0, 2] = vector[:, 1]
    output[:, 2, 0] = -vector[:, 1]
    output[:, 1, 2] = -vector[:, 0]
    output[:, 2, 1] = vector[:, 0]
    return output


# Compute the rotation matrix R & translation vector from the axis-angle parameters using Rodriguez's formula:
# (R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2)
# (t = I + (1-cos(theta))/theta^2 * K + ((theta-sin(theta)/theta^3) * K^2
# where K is the skew symmetric matrix based on the un-normalized axis & theta is the norm of the input parameters
# From Wikipedia: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# Eqns: 77-84 (PAGE: 10) of http://ethaneade.com/lie.pdf
def se3ToRt(input):
    eps = 1e-12
    # Get the un-normalized axis and angle
    N = input.size(0)
    axis = input.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    angle = torch.sqrt(angle2)  # Angle

    # Compute skew-symmetric matrix "K" from the axis of rotation
    K = create_skew_symmetric_matrix(axis)
    K2 = torch.bmm(K, K)  # K * K

    # Compute sines
    S = torch.sin(angle) / angle
    S.masked_fill_(angle2.lt(eps), 1)  # sin(0)/0 ~= 1

    # Compute cosines
    C = (1 - torch.cos(angle)) / angle2
    C.masked_fill_(angle2.lt(eps), 0)  # (1 - cos(0))/0^2 ~= 0

    # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
    rot = (
        torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(input)
    )  # R = I (avoid use expand as it does not allocate new memory)
    rot += K * S.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K
    rot += K2 * C.expand(N, 3, 3)
    return rot

    """
    # Check dimensions
    bsz, nse3, ndim = input.size()
    N = bsz * nse3
    eps = 1e-12
    assert ndim == 6

    # Trans | Rot params
    input_v = input.view(N, ndim, 1)
    rotparam = input_v.narrow(1, 3, 3)  # so(3)
    transparam = input_v.narrow(1, 0, 3)  # R^3

    # Get the un-normalized axis and angle
    axis = rotparam.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    angle = torch.sqrt(angle2)  # Angle
    small = (
        angle2 < eps
    ).detach()  # Don't need gradient w.r.t this operation (also because of pytorch error: https://discuss.pytorch.org/t/get-error-message-maskedfill-cant-differentiate-the-mask/9129/4)

    # Create Identity matrix
    I = angle2.expand(N, 3, 3).clone()
    I[:] = 0  # Fill will all zeros
    I[:, 0, 0], I[:, 1, 1], I[:, 2, 2] = 1.0, 1.0, 1.0  # Diagonal elements = 1

    # Compute skew-symmetric matrix "K" from the axis of rotation
    K = create_skew_symmetric_matrix(axis)
    K2 = torch.bmm(K, K)  # K * K

    # Compute A = (sin(theta)/theta)
    A = torch.sin(angle) / angle
    A[small] = 1.0  # sin(0)/0 ~= 1

    # Compute B = (1 - cos(theta)/theta^2)
    B = (1 - torch.cos(angle)) / angle2
    B[small] = 1 / 2  # lim 0-> 0 (1 - cos(0))/0^2 = 1/2

    # Compute C = (theta - sin(theta))/theta^3
    C = (1 - A) / angle2
    C[small] = 1 / 6  # lim 0-> 0 (0 - sin(0))/0^3 = 1/6

    # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
    R = I + K * A.expand(N, 3, 3) + K2 * B.expand(N, 3, 3)

    # Compute the translation tfm matrix: V = I + ((1-cos(theta))/theta^2) * K + ((theta-sin(theta))/theta^3)*K^2
    V = I + K * B.expand(N, 3, 3) + K2 * C.expand(N, 3, 3)

    # Compute translation vector
    t = torch.bmm(V, transparam)  # V*u

    # Final tfm
    return torch.cat([R, t], 2).view(bsz, nse3, 3, 4).clone()  # B x K x 3 x 4
    """


def so3d_to_so3_mat(so3d):
    # so3_3d Bx3
    B, _ = so3d.shape

    angle1d = torch.norm(so3d, dim=1, p=2)
    axis3d = so3d / (angle1d.unsqueeze(-1) + 1e-7)
    cos_angle1d = torch.cos(angle1d)
    sin_angle1d = torch.sin(angle1d)

    so3_mat = (
        torch.stack(
            [
                cos_angle1d + axis3d[:, 0] ** 2 * (1 - cos_angle1d),
                axis3d[:, 0] * axis3d[:, 1] * (1 - cos_angle1d)
                - axis3d[:, 2] * sin_angle1d,
                axis3d[:, 0] * axis3d[:, 2] * (1 - cos_angle1d)
                + axis3d[:, 1] * sin_angle1d,
                axis3d[:, 1] * axis3d[:, 0] * (1 - cos_angle1d)
                + axis3d[:, 2] * sin_angle1d,
                cos_angle1d + axis3d[:, 1] ** 2 * (1 - cos_angle1d),
                axis3d[:, 1] * axis3d[:, 2] * (1 - cos_angle1d)
                - axis3d[:, 0] * sin_angle1d,
                axis3d[:, 2] * axis3d[:, 0] * (1 - cos_angle1d)
                - axis3d[:, 1] * sin_angle1d,
                axis3d[:, 2] * axis3d[:, 1] * (1 - cos_angle1d)
                + axis3d[:, 0] * sin_angle1d,
                cos_angle1d + axis3d[:, 2] ** 2 * (1 - cos_angle1d),
            ]
        )
        .reshape(3, 3, -1)
        .permute(2, 0, 1)
    )

    return so3_mat


def se3_6d_to_se3mat(se3_6d, type="exponential"):
    # input shape Bx6 or BxKx6
    K = None
    if len(se3_6d.shape) == 3:
        K = se3_6d.shape[1]
        se3_6d = se3_6d.view(-1, 6)
    B, _ = se3_6d.shape
    dtype = se3_6d.dtype
    device = se3_6d.device

    se3mat = torch.zeros(size=(B, 4, 4), dtype=dtype, device=device)

    if type == "axisangle":
        se3mat[:, :3, :3] = so3d_to_so3_mat(se3_6d[:, :3])
        se3mat[:, :3, 3] = se3_6d[:, 3:]
    elif type == "exponential":
        se3mat[:, :3, :3] = t3d.so3_exponential_map(se3_6d[:, :3])
        se3mat[:, :3, 3] = se3_6d[:, 3:]
    elif type == "hind4sight":
        # se3mat[:, :3, :4] = se3ToRt(se3_6d[:, :].unsqueeze(1))
        se3mat[:, :3, :3] = se3ToRt(se3_6d[:, :3])
        se3mat[:, :3, 3] = se3_6d[:, 3:]

    # transl_3d = torch.matmul(se3mat[:, :3, :3], se3_6d[:, 3:].unsqueeze(-1)).squeeze(-1)
    # se3mat[:, :3, 3] = transl_3d

    # se3mat[:, :3, :3] = torch.eye(3)

    se3mat[:, 3, 3] = 1.0

    if K is not None:
        se3mat = se3mat.view(-1, K, 4, 4)

    return se3mat


def pts3d_transform(pts3d, se3mat, mask=None):
    # pts3d: BxNx3 or Bx3xHxW
    # se3mat: Bx4x4 or BxKx4x4
    # masks: BxKxHxW
    pts3d_shape = pts3d.shape
    dtype = pts3d.dtype
    device = pts3d.device

    if len(pts3d_shape) == 4:
        B, _, H, W = pts3d_shape
        pts3d = pts3d.reshape(B, 3, H * W).permute(0, 2, 1)

    B, N, _ = pts3d.shape
    # se3mat Bx4x4
    # pts3d BxNx3
    pts4d = torch.cat(
        (pts3d, torch.ones(size=(B, N, 1), dtype=dtype, device=device)), dim=2
    )
    if mask is None:
        # se3mat: Bx1x4x4, pts3d: BxNx4x1
        pts4d_ftf = torch.matmul(se3mat.unsqueeze(1), pts4d.unsqueeze(-1)).squeeze(-1)
        # pts4d: BxNx4
        pts3d_ftf = pts4d_ftf[:, :, :3]
    else:
        # se3mat: BxKx1x4x4, pts3d: Bx1xNx4x1
        # se3mat[:, 0] = torch.eye(4, dtype=se3mat.dtype, device=se3mat.device)
        # se3mat = se3mat * 0.
        pts4d_ftf = torch.matmul(
            se3mat.unsqueeze(2), pts4d.unsqueeze(1).unsqueeze(-1)
        ).squeeze(-1)
        # pts4d: BxKxNx4 -> BxKxNx3
        pts3d_ftf = pts4d_ftf[:, :, :, :3]
        B, K, H, W = mask.shape
        # B x K x H x W
        mask = mask.reshape(B, K, H * W).unsqueeze(-1)
        # BxKxNx1 * BxKxNx3
        pts3d_ftf = pts3d + torch.sum(mask * (pts3d_ftf - pts3d.unsqueeze(1)), dim=1)

    if len(pts3d_shape) == 4:
        pts3d_ftf = pts3d_ftf.permute(0, 2, 1).reshape(B, 3, H, W)

    return pts3d_ftf


def pts3d_transform_obj_ego(pts3d, se3mat, mask, egomotion_addition=True):

    # transform with egomotion

    if egomotion_addition:
        pts3d = pts3d_transform(pts3d, se3mat[:, 0])

    if mask is not None:
        # transform with masks + obj_motions
        if egomotion_addition:
            se3mat = se3mat.clone()
            se3mat[:, 0] = torch.eye(4, dtype=se3mat.dtype, device=se3mat.device)
        pts3d = pts3d_transform(pts3d, se3mat[:, 0:], mask[:, 0:])

    return pts3d


def sflow_via_transform(pts3d, se3mat, mask, egomotion_addition):
    pts3d_ftf = pts3d_transform_obj_ego(
        pts3d, se3mat, mask, egomotion_addition=egomotion_addition
    )
    flow = pts3d_ftf - pts3d.detach()
    return flow, mask
