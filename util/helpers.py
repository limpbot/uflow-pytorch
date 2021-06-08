import torch
import numpy as np
import cv2
import matplotlib

# matplotlib.use('Agg')  # None-interactive plots do not need tk
from matplotlib import pyplot as plt

# from util.chamfer_distance import ChamferDistance

# Load optimizer


def load_optimizer(optim_type, parameters, lr=1e-4, momentum=0.9, weight_decay=1e-4):
    if optim_type == "sgd":
        optimizer = torch.optim.SGD(
            params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(
            params=parameters, lr=lr, weight_decay=weight_decay
        )
    elif optim_type == "asgd":
        optimizer = torch.optim.ASGD(
            params=parameters, lr=lr, weight_decay=weight_decay
        )
    elif optim_type == "rmsprop":
        optimizer = torch.optim.RMSprop(
            params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        assert False, "Unknown optimizer type: " + optim_type
    return optimizer


def rgb_to_grayscale(x):
    # x: Bx3xHxW

    # rgb_weights: 3x1x1
    # https://en.wikipedia.org/wiki/Luma_%28video%29

    dtype = x.dtype
    device = x.device

    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=dtype, device=device)

    rgb_weights = rgb_weights.view(-1, 1, 1)

    x = x * rgb_weights

    x = torch.sum(x, dim=1, keepdim=True)

    return x


# input: flow: torch.tensor 2xHxW
# output: flow_rgb: numpy.ndarray 3xHxW
def flow2rgb_old(flow, max_value=100):
    flow_map_np = flow.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float("nan")
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    rgb_map = rgb_map.clip(0, 1)
    rgb_map = torch.from_numpy(rgb_map)
    rgb_map = rgb_map.to(flow.device)
    return rgb_map


def flow2startendpoints(flow):
    _, H, W = flow.shape

    endpoints = oflow2pxlcoords(flow.unsqueeze(0))[0]

    startpoints = oflow2pxlcoords(0.0 * flow.unsqueeze(0))[0]

    return startpoints, endpoints


def depth2rgb(depth):
    if len(depth.shape) == 2:
        H, W = depth.shape
    elif len(depth.shape) == 3:
        _, H, W = depth.shape
        depth = depth[0]

    device = depth.device
    dtype = depth.dtype

    np_depth = depth.detach().cpu().numpy()
    # cv2.COLORMAP_PLASMA, cv2.COLORMAP_MAGMA, cv2.COLORMAP_INFERNO
    np_depth_rgb = (
        cv2.applyColorMap((np_depth * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)
        / 255.0
    )

    depth_rgb = torch.from_numpy(np_depth_rgb).permute(2, 0, 1)
    depth_rgb = torch.flip(depth_rgb, dims=(0,))
    depth_rgb = depth_rgb.to(device)

    return depth_rgb


def get_colors(K, device=None):
    torch_colors = (
        torch.from_numpy(
            cv2.applyColorMap(
                tensor_to_cv_img(
                    (torch.arange(K).repeat(1, 1, 1).type(torch.float32) + 1.0)
                    / (K + 1)
                ),
                cv2.COLORMAP_JET,
            )
        ).squeeze()
        / 255.0
    )

    if device is not None:
        torch_colors = torch_colors.to(device)
    # K x 3
    return torch_colors


def mask2rgb(torch_mask):
    K, H, W = torch_mask.shape
    device = torch_mask.device
    torch_mask = torch_mask.type(torch.float32)
    torch_mask_normalized = torch_mask / (
        torch.sum(torch_mask, dim=0, keepdim=True) + 1e-7
    )

    torch_colors = get_colors(K, device=device)

    # Kx3x1x1 * Kx1xHxW
    torch_mask_rgb = torch.sum(
        torch_colors.unsqueeze(-1).unsqueeze(-1) * torch_mask_normalized.unsqueeze(1),
        dim=0,
    )

    torch_mask_rgb = torch_mask_rgb.to(device)

    return torch_mask_rgb


def flow2rgb(flow_torch, draw_arrows=False, srcs_flow=None):
    _, H, W = flow_torch.shape
    flow_torch_1 = flow_torch.clone()

    flow = flow_torch_1.detach().cpu().numpy()
    # 2 x H x W
    flow[0] = -flow[0]
    flow[1] = -flow[1]

    scaling = 50.0 / (H ** 2 + W ** 2) ** 0.5
    motion_angle = np.arctan2(flow[0], flow[1])
    motion_magnitude = (flow[0] ** 2 + flow[0] ** 2) ** 0.5
    flow_hsv = np.stack(
        [
            ((motion_angle / np.math.pi) + 1.0) / 2.0,
            np.clip(motion_magnitude * scaling, 0.0, 1.0),
            np.ones_like(motion_magnitude),
        ],
        axis=-1,
    )

    flow_rgb = matplotlib.colors.hsv_to_rgb(flow_hsv)

    """
    srcs_flow = srcs_flow.flatten(1).permute(1, 0)
    num_srcs = srcs_flow.shape[0]
    srcs_flow = srcs_flow.detach().cpu().numpy()
    
    for i in range(num_srcs):
        flow_rgb = cv2.circle(flow_rgb, (srcs_flow[i, 0], srcs_flow[i, 1]), radius=0, color=(0, 0, 255), thickness=-1)
    """

    flow_rgb = torch.from_numpy(flow_rgb).permute(2, 0, 1)

    flow_rgb = flow_rgb.to(flow_torch.device)

    if srcs_flow is not None:
        start, end = flow2startendpoints(srcs_flow.clone())
        flow_rgb = draw_arrows_in_rgb(flow_rgb, start, end)

    if draw_arrows:
        start, end = flow2startendpoints(flow_torch.clone())
        flow_rgb = draw_arrows_in_rgb(flow_rgb, start, end)

    return flow_rgb


def delta_uv_kernels(patch_size, dtype, device):
    delta_uv = shape2pxlcoords(
        B=1, H=patch_size, W=patch_size, dtype=dtype, device=device
    ) - (patch_size // 2)

    return delta_uv


def neighbors_to_channels(x, patch_size):
    # input: BxCxHxW
    # output: BxP*P*CxHxW
    # first channel upper-left
    # for top-to-bottom
    #   for left-to-right
    # note: second channel = first row, second element
    # last channel lower-right

    dtype = x.dtype
    device = x.device

    kernels = torch.eye(patch_size ** 2, dtype=dtype, device=device).view(
        patch_size ** 2, 1, patch_size, patch_size
    )

    # kernels: P*Px1xPxP : out_channels x in_channels x H x W
    kernels = kernels.repeat(x.size(1), 1, 1, 1)
    # kernels: P*Px1xPxP : out_channels x in_channels x H x W

    x = torch.nn.functional.conv2d(
        input=x, weight=kernels, padding=int((patch_size - 1) / 2), groups=x.size(1)
    )

    return x


def neighbors_to_channels_v2(x, patch_size):
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    num_channels_out = patch_size ** 2 * C

    max_displacement = int((patch_size - 1) / 2)

    x_volume = torch.zeros(size=(B, num_channels_out, H, W), device=device, dtype=dtype)

    padding_module = torch.nn.ConstantPad2d(max_displacement, 0.0)
    x_pad = padding_module(x)

    for i in range(patch_size):
        for j in range(patch_size):
            channel_index = (i * patch_size + j) * C
            x_volume[:, channel_index : channel_index + C, :, :] = x_pad[
                :, :, i : i + H, j : j + W
            ]

    return x_volume


def neighbors_to_channels_v3(x, patch_size):
    # input: BxCxHxW
    # output: BxP*P*CxHxW

    B, C, H, W = x.shape
    dtype = x.dtype
    device = x.device

    grid_y, grid_x = torch.meshgrid(
        [
            torch.arange(0.0, H, dtype=torch.int32, device=device),
            torch.arange(0.0, W, dtype=torch.int32, device=device),
        ]
    )
    grid_yx = torch.stack((grid_y, grid_x), dim=0)
    grid_yx = grid_yx.unsqueeze(1)

    patch_grid_y, patch_grid_x = torch.meshgrid(
        [
            torch.arange(0.0, patch_size, dtype=torch.long, device=device),
            torch.arange(0.0, patch_size, dtype=torch.long, device=device),
        ]
    )
    patch_grid_yx = torch.stack((patch_grid_y, patch_grid_x), dim=0) - (
        (patch_size - 1) // 2
    )
    patch_grid_yx = patch_grid_yx.flatten(1).unsqueeze(2).unsqueeze(3)

    grid_yx = grid_yx + patch_grid_yx
    indices = (grid_yx[0] * W + grid_yx[1]).flatten(1).unsqueeze(0)
    x = x.flatten(2)

    y = torch.zeros(size=(B, patch_size ** 2, H * W), device=device, dtype=dtype)

    # mask = (indices >= 0) * (indices < H*W)

    y.scatter_(dim=2, src=x, index=indices[:, :1])

    return x


def census_transform(x, patch_size):
    """
    census transform:
    input: rgb image
    output: difference for each pixel to its neighbors 7x7
    1. rgb to gray: bxhxwxc -> bxhxwx1
    2. neighbor intensities as channels: bxhxwx1 -> bxhxwx7*7 (padding with zeros)
    3. difference calculation: L1 / sqrt(0.81 + L1^2): bxhxwx7*7 (coefficient from DDFlow)
    """

    # x: Bx3xHxW
    x = rgb_to_grayscale(x) * 255.0

    # x: Bx1xHxW
    x = neighbors_to_channels(x, patch_size=patch_size)

    # x: BxP^2xHxW - Bx1xHxW
    dist_per_pixel_per_neighbor = x - x[:, 24].unsqueeze(1)

    # L1: BxP^2xHxW
    dist_per_pixel_per_neighbor = dist_per_pixel_per_neighbor / torch.sqrt(
        0.81 + dist_per_pixel_per_neighbor ** 2
    )
    # neighbor_dist: BxP^2xHxW
    # neighbor_dist in [0, 1]

    return dist_per_pixel_per_neighbor


def soft_hamming_distance(x1, x2):
    """
    soft hamming distance:
    input: census transformed images bxhxwxk
    output: difference between census transforms per pixel
    1. difference calculation per pixel, per features: L2 / (0.1 + L2)
    2. summation over features: bxhxwxk -> bxhxwx1
    """

    # x1, x2: BxCxHxW

    squared_dist_per_pixel_per_feature = (x1 - x2) ** 2
    # squared_dist_per_pixel_per_feature: BxCxHxW

    dist_per_pixel_per_feature = squared_dist_per_pixel_per_feature / (
        0.1 + squared_dist_per_pixel_per_feature
    )
    # dist_per_pixel_per_feature: BxCxHxW

    dist_per_pixel = torch.sum(dist_per_pixel_per_feature, dim=1)
    # dist_per_pixel_per: BxHxW

    return dist_per_pixel


def calc_photo_loss(x1, x2, type="census", masks_flow_valid=None):

    if type == "census":
        return calc_census_loss(x1, x2, patch_size=7, masks_flow_valid=masks_flow_valid)

    elif type == "ssim":

        l1_per_pixel = calc_l1_per_pixel(x1, x2)
        dssim_per_pixel = calc_ddsim_per_pixel(x1, x2)
        dssim_per_pixel = torch.nn.functional.pad(
            dssim_per_pixel, (1, 1, 1, 1), mode="constant", value=0
        )
        loss_per_pixel = (0.85 * dssim_per_pixel + 0.15 * l1_per_pixel).mean(
            dim=1, keepdim=True
        )

        loss_per_pixel = loss_per_pixel * masks_flow_valid

        return torch.sum(loss_per_pixel) / (torch.sum(masks_flow_valid) + 1e-8)


def calc_census_loss(x1, x2, patch_size, masks_flow_valid=None):
    """
    census loss:
    1. hamming distance from census transformed rgb images
    2. robust loss for per pixel hamming distance: (|diff|+0.01)^0.4   (as in DDFlow)
    3. per pixel multiplication with zero mask at border s.t. every loss value close to border = 0
    4. sum over all pixel and divide by number of pixel which were not zeroed out: sum(per_pixel_loss)/ (num_pixels + 1e-6)
    """

    dtype = x1.dtype
    device = x1.device

    # x1, x2: Bx3xHxW
    x1_census = census_transform(x1, patch_size)
    x2_census = census_transform(x2, patch_size)
    # x1_census, x2_census: Bxpatch_size^2xHxW

    soft_hamming_dist_per_pixel = soft_hamming_distance(x1_census, x2_census)
    # soft_hamming_dist: BxHxW

    robust_soft_hamming_dist_per_pixel = (soft_hamming_dist_per_pixel + 0.01) ** (0.4)
    # robust_soft_hamming_dist_per_pixel: BxHxW

    masks_valid_pixel = torch.zeros(
        (robust_soft_hamming_dist_per_pixel.size()), dtype=dtype, device=device
    )

    pad = int((patch_size - 1) / 2)

    if masks_flow_valid is not None:
        masks_valid_pixel[:, pad:-pad, pad:-pad] = masks_flow_valid[
            :, 0, pad:-pad, pad:-pad
        ]
    else:
        masks_valid_pixel[:, pad:-pad, pad:-pad] = 1.0

    # mask = mask.repeat(robust_soft_hamming_dist_per_pixel.size(0), 1, 1)
    # mask: BxHxW
    #  # * valid_warp_mask: adds if warping is outside of frame

    valid_pixel_mask_total_weight = torch.sum(masks_valid_pixel, dim=(0, 1, 2))

    # q: why does uflow stop gradient computation for mask in mask_total_weight, but not for mask in general?

    return torch.sum(
        robust_soft_hamming_dist_per_pixel * masks_valid_pixel, dim=(0, 1, 2)
    ) / (valid_pixel_mask_total_weight + 1e-6)


def calc_l1_loss(x1, x2):
    return torch.sum(torch.abs(x1 - x2)) / x1.size(0)


def calc_l1_per_pixel(x1, x2):
    return torch.norm(x1 - x2, p=1, dim=1, keepdim=True)


def calc_charbonnier_loss(x1, x2, weights):
    # x1, x2: BxCxHxW
    # weights: Bx1xHxW
    # return torch.sum((((x1 - x2) ** 2 + 0.001 ** 2) ** 0.5) * weights) / (torch.sum(weights) + 1e-16)

    return torch.sum((((x1 - x2) ** 2 + 0.001 ** 2) ** 0.5) * weights) / (
        weights.size(0) * weights.size(2) * weights.size(3) + 1e-16
    )


def calc_reconstruction_loss(
    pts1_norm, pts1_ftf, pts1, pxlcoords1_ftf, mask_valid, type="smsf"
):
    # forward_transformed
    # backward_warped
    # x1, x2: B x 3 x H x W

    num_imgpairs_left_fwd, _, _, _ = pts1.shape

    # reorder
    pts2 = torch.cat(
        (
            pts1[num_imgpairs_left_fwd:],
            pts1[:num_imgpairs_left_fwd],
        ),
        dim=0,
    )

    if type == "smsf":
        pts2_bwrpd = interpolate2d(
            pts2,
            pxlcoords1_ftf,
            return_masks_flow_inside=False,
        )

        res = pts1_ftf - pts2_bwrpd
        epe = torch.norm(res, p=2, dim=1, keepdim=True)

        # * 2.0 because forward and backward reconstruction are summed in self-mono-sf
        rec_loss = torch.sum((mask_valid * epe) / (pts1_norm + 1e-8)) / (
            torch.sum(mask_valid) + 1e-8
        )

    elif type == "chamfer":
        return None
        # rec_loss = calc_chamfer_loss(pts1_ftf, pts2, mask_valid)

    return rec_loss


def warp(x, flow, return_masks_flow_inside=False):
    B, C, H, W = x.size()
    dtype = x.dtype
    device = x.device

    pxlcoords = oflow2pxlcoords(flow)
    # input: (B, C, Hin​, Win​) and grid: (B, 2, Hout​, Wout​)
    # output: (B, C, Hin, Win

    return interpolate2d(
        x, pxlcoords, return_masks_flow_inside=return_masks_flow_inside
    )


def sflow2oflow(sflow, disp, proj_mats, reproj_mats):
    depth = disp2depth(disp, proj_mats[:, 0, 0])
    points3d = depth2xyz(depth, reproj_mats)
    points3d_fwdwrpd = points3d + sflow
    pxlcoords_fwdwrpd = xyz2uv(points3d_fwdwrpd, proj_mats)

    oflow = pxlcoords2flow(pxlcoords_fwdwrpd)

    return oflow


def warp3d(x, flow, disp, proj_mats, reproj_mats):
    depth = disp2depth(disp, proj_mats[:, 0, 0])
    points3d = depth2xyz(depth, reproj_mats)
    points3d_fwdwrpd = points3d + flow
    pxlcoords_fwdwrpd = xyz2uv(points3d_fwdwrpd, proj_mats)

    return interpolate2d(x, pxlcoords_fwdwrpd, return_masks_flow_inside=False)


def interpolate2d(x, pxlcoords, return_masks_flow_inside=False):
    pxlcoords_normalized = normalize_pxlcoords(pxlcoords)
    pxlcoords_normalized_perm = pxlcoords_normalized.transpose(1, 2).transpose(
        2, 3
    )  # .permute(0, 2, 3, 1)

    B, C, H, W = x.size()
    dtype = x.dtype
    device = x.device

    # input: (B, C, Hin​, Win​) and grid: (1, Hout​, Wout​, 2)
    # output: (B, C, Hin, Win
    x_warped = torch.nn.functional.grid_sample(
        input=x, grid=pxlcoords_normalized_perm, mode="bilinear", align_corners=True
    )
    # BxCxHxW

    mask_valid_flow = torch.ones(
        size=(B, H, W), dtype=dtype, device=device, requires_grad=False
    )
    mask_valid_flow[pxlcoords_normalized_perm[:, :, :, 0] > 1.0] = 0.0
    mask_valid_flow[pxlcoords_normalized_perm[:, :, :, 0] < -1.0] = 0.0
    mask_valid_flow[pxlcoords_normalized_perm[:, :, :, 1] > 1.0] = 0.0
    mask_valid_flow[pxlcoords_normalized_perm[:, :, :, 1] < -1.0] = 0.0
    mask_valid_flow = mask_valid_flow.unsqueeze(1)
    # Bx1xHxW

    x_warped = x_warped * mask_valid_flow

    if return_masks_flow_inside:
        return x_warped, mask_valid_flow
    else:
        return x_warped


def calc_masks_flow_inside(flow):
    B, C, H, W = flow.size()
    dtype = flow.dtype
    device = flow.device

    pxlcoords_normalized = flow2pxlcoords_normalized(flow)
    pxlcoords_normalized = pxlcoords_normalized.permute(0, 2, 3, 1)
    # input: (B, C, Hin​, Win​) and grid: (1, 2, Hout​, Wout​)
    # output: (B, C, Hin, Win
    # BxCxHxW

    mask_valid_flow = torch.ones(size=(B, H, W), dtype=dtype, device=device)
    mask_valid_flow[pxlcoords_normalized[:, :, :, 0] > 1.0] = 0.0
    mask_valid_flow[pxlcoords_normalized[:, :, :, 0] < -1.0] = 0.0
    mask_valid_flow[pxlcoords_normalized[:, :, :, 1] > 1.0] = 0.0
    mask_valid_flow[pxlcoords_normalized[:, :, :, 1] < -1.0] = 0.0
    mask_valid_flow = mask_valid_flow.unsqueeze(1)
    # Bx1xHxW

    return mask_valid_flow


def normalize_features(x1, x2):
    # over channel and spatial dimensions
    # and moments average over batch size and features
    # x1 : B x C x H x W
    """
    x1 = torch.rand((12, 3, 9, 16))

    import tensorflow as tf
    x1 = tf.convert_to_tensor(x1.detach().cpu().numpy())
    #x2 = tf.convert_to_tensor(x2.detach().cpu().numpy())
    mean1, var1 = tf.nn.moments(x1, axes=[-1, -2, -3])
    mean1 = torch.from_numpy(mean1.numpy()).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    var1 = torch.from_numpy(var1.numpy()).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    """

    var1, mean1 = torch.var_mean(x1, dim=(1, 2, 3), keepdim=True, unbiased=False)
    var2, mean2 = torch.var_mean(x2, dim=(1, 2, 3), keepdim=True, unbiased=False)

    mean = torch.mean(torch.cat((mean1, mean2)))
    var = torch.mean(torch.cat((var1, var2)))

    # 1e-16 for robustness of division by std, this can be found in uflow implementation
    std = torch.sqrt(var + 1e-16)
    x1_normalized = (x1 - mean) / std
    x2_normalized = (x2 - mean) / std

    return x1_normalized, x2_normalized


def compute_cost_volume(x1, x2, max_displacement):
    # x1 : B x C x H x W
    # x2 : B x C x H x W

    dtype = x1.dtype
    device = x1.device

    B, C, H, W = x1.size()

    padding_module = torch.nn.ConstantPad2d(max_displacement, 0.0)
    x2_pad = padding_module(x2)

    num_shifts = 2 * max_displacement + 1

    cost_volume_channel_dim = num_shifts ** 2
    cost_volume = torch.zeros(
        (B, cost_volume_channel_dim, H, W), dtype=dtype, device=device
    )

    for i in range(num_shifts):
        for j in range(num_shifts):
            cost_volume_single_layer = torch.mean(
                x1 * x2_pad[:, :, i : i + H, j : j + W], dim=1
            )
            cost_volume[:, i * num_shifts + j, :, :] = cost_volume_single_layer

    return cost_volume


def compute_cost_volume_v2(x1, x2, max_displacement):

    dtype = x1.dtype
    device = x1.device

    B, C, H, W = x1.size()

    num_shifts = 2 * max_displacement + 1

    cost_volume_channel_dim = num_shifts ** 2

    x2_channels = neighbors_to_channels(x2, patch_size=9).reshape(
        B, cost_volume_channel_dim, C, H, W
    )
    x1_channels = x1.unsqueeze(1)
    # x1_channels = x1.repeat_interleave(cost_volume_channel_dim, dim=1)
    cost_volume = torch.mean((x1_channels * x2_channels), dim=2)

    return cost_volume


def calc_batch_gradients(batch, margin=0):
    # batch: torch.tensor: BxCxHxW
    shift = 1 + margin
    batch_grad_x = batch[:, :, :, shift:] - batch[:, :, :, :-shift]
    # B x 2 x H x W-shift

    batch_grad_y = batch[:, :, shift:, :] - batch[:, :, :-shift, :]
    # B x 2 x H-shift x W

    return batch_grad_x, batch_grad_y


def calc_batch_k_gradients(batch, order):

    print(batch.size())
    batch_k_grad_x, batch_k_grad_y = calc_batch_gradients(batch)
    print(batch_k_grad_x.size())
    for k in range(order - 1):
        batch_k_grad_x, _ = calc_batch_gradients(batch_k_grad_x)
        print(batch_k_grad_x.size())
        _, batch_k_grad_y = calc_batch_gradients(batch_k_grad_y)

    return batch_k_grad_x, batch_k_grad_y


def robust_l1(x):
    """Robust L1 metric."""
    return (x ** 2 + 0.001 ** 2) ** 0.5


def calc_disp_outlier_percentage(disps_pred, disps_gt, masks_disps_gt):
    diff_disps = torch.abs(disps_pred[masks_disps_gt] - disps_gt[masks_disps_gt])
    disp_outlier_percentage = (
        100
        * torch.sum(
            (diff_disps >= 3.0) * ((diff_disps / disps_gt[masks_disps_gt]) >= 0.05)
        )
        / torch.sum(masks_disps_gt)
    )

    return disp_outlier_percentage


def calc_flow_outlier_percentage(flows_pred, flows_gt, masks_flows_gt):
    masks_flows_gt = masks_flows_gt.unsqueeze(0)
    diff_flows = torch.norm(flows_pred - flows_gt, dim=1, keepdim=True)[masks_flows_gt]
    norm_flows_gt = torch.norm(flows_gt, dim=1, keepdim=True)[masks_flows_gt]
    flow_outlier_percentage = (
        100
        * torch.sum((diff_flows >= 3.0) * ((diff_flows / norm_flows_gt) >= 0.05))
        / torch.sum(masks_flows_gt)
    )

    return flow_outlier_percentage


def calc_sflow_outlier_percentage(
    disps0_pred,
    disps0_gt,
    masks_disps0_gt,
    disps1_pred,
    disps1_gt,
    masks_disps1_gt,
    flows_pred,
    flows_gt,
    masks_flows_gt,
):
    masks_flows_gt = masks_flows_gt.unsqueeze(0)
    diff_flows = torch.norm(flows_pred - flows_gt, dim=1, keepdim=True)[masks_flows_gt]
    norm_flows_gt = torch.norm(flows_gt, dim=1, keepdim=True)[masks_flows_gt]

    diff_disps0 = torch.abs(disps0_pred[masks_disps0_gt] - disps0_gt[masks_disps0_gt])
    diff_disps1 = torch.abs(disps1_pred[masks_disps1_gt] - disps1_gt[masks_disps1_gt])

    flows_outls = (diff_flows >= 3.0) * ((diff_flows / norm_flows_gt) >= 0.05)
    disps0_outls = (diff_disps0 >= 3.0) * (
        (diff_disps0 / disps0_gt[masks_disps0_gt]) >= 0.05
    )
    disps1_outls = (diff_disps0 >= 3.0) * (
        (diff_disps0 / disps0_gt[masks_disps0_gt]) >= 0.05
    )

    sflow_outlier_percentage = (
        100
        * torch.sum(flows_outls | disps0_outls | disps1_outls)
        / torch.sum(masks_flows_gt | masks_disps0_gt | masks_disps1_gt)
    )

    return sflow_outlier_percentage


def calc_smoothness_loss(
    flow, img1, edge_weight=150, order=1, weights_inv=None, smooth_type="uflow"
):
    # flow: torch.tensor: Bx2xHxW
    # img1: torch.tensor: Bx3xHxW

    margin = 0
    flow_k_grad_x, flow_k_grad_y = calc_batch_gradients(flow)
    for k in range(order - 1):
        flow_k_grad_x, _ = calc_batch_gradients(flow_k_grad_x)
        _, flow_k_grad_y = calc_batch_gradients(flow_k_grad_y)

    if smooth_type == "uflow":
        # Bx2xHxW-(order * 2)
        flow_k_grad_x = robust_l1(flow_k_grad_x)
        flow_k_grad_y = robust_l1(flow_k_grad_y)

        img1_grad_x, img1_grad_y = calc_batch_gradients(img1, margin=order - 1)
    elif smooth_type == "smsf":
        flow_k_grad_x = torch.abs(flow_k_grad_x)
        flow_k_grad_y = torch.abs(flow_k_grad_y)

        img1_grad_x, img1_grad_y = calc_batch_gradients(img1, margin=0)
        img1_grad_x = img1_grad_x[:, :, :, order - 1 :]
        img1_grad_y = img1_grad_y[:, :, order - 1 :, :]
    else:
        print("error: unknown smooth-type: ", smooth_type)

    img1_grad_x = torch.abs(img1_grad_x)
    # Bx3xHxW-1
    img1_grad_y = torch.abs(img1_grad_y)
    # Bx3xH-1xW

    if weights_inv is not None:
        weights_inv_x = weights_inv[:, :, :, 1:-1]
        weights_inv_y = weights_inv[:, :, 1:-1, :]

        loss = (
            torch.mean(
                torch.exp(-edge_weight * torch.mean(img1_grad_x, dim=1, keepdim=True))
                * flow_k_grad_x
                / weights_inv_x
            )
            + torch.mean(
                torch.exp(-edge_weight * torch.mean(img1_grad_y, dim=1, keepdim=True))
                * flow_k_grad_y
                / weights_inv_y
            )
        ) / 2.0

    else:
        loss = (
            torch.mean(
                torch.exp(-edge_weight * torch.mean(img1_grad_x, dim=1, keepdim=True))
                * flow_k_grad_x
            )
            + torch.mean(
                torch.exp(-edge_weight * torch.mean(img1_grad_y, dim=1, keepdim=True))
                * flow_k_grad_y
            )
        ) / 2.0

    return loss


def calc_ddsim_per_pixel(x, y):
    # patch size = 3x3
    # x = BxCxHxW
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    # (c3 = c2 / 2)
    mu_x = torch.nn.AvgPool2d(3, 1)(x)
    mu_y = torch.nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = torch.nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = torch.nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    ssim_n = (2 * mu_x_mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d  # range [-1, 1]
    dssim = (1 - ssim) / 2  # range [ 0, 1]

    return dssim


def calc_chamfer_dist(pts1, pts2):
    # pts1, pts2: Bx3xHxW
    # patch_size = kernel_size
    B, C, H, W = pts1.shape
    kernel_size = int(W * 0.3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    print(kernel_size)
    pts1_neighbors = neighbors_to_channels(pts1, patch_size=kernel_size).reshape(
        B, C, -1, H, W
    )
    dist, _ = torch.min(
        torch.norm(pts1_neighbors - pts2.unsqueeze(2), dim=1, keepdim=True), dim=2
    )
    # BxK*CxHxW
    return dist


"""
def calc_chamfer_loss(
    pts1,
    pts2,
    masks_valid=None,
    img=None,
    edge_weight=150,
    order=2,
    lambda_smoothness=0.1,
):
    # B x 3 x H x W
    B, C, H, W = pts1.shape
    chamfer_dist = ChamferDistance()
    # ...
    # points and points_reconstructed are n_points x 3 matrices
    pts1 = pts1.flatten(2).permute(0, 2, 1)
    pts2 = pts2.flatten(2).permute(0, 2, 1)
    # Bx N x 3
    dist1, dist2 = chamfer_dist(pts1, pts2)
    dist1 = dist1.reshape(B, 1, H, W)
    dist2 = dist2.reshape(B, 1, H, W)

    if masks_valid is not None:
        dist1 = dist1 * masks_valid
        dist2 = dist2 * masks_valid
    chamfer_loss = (torch.mean(dist1)) + (torch.mean(dist2))

    if img is not None:
        smoothness_loss = calc_smoothness_loss(
            dist1, img, edge_weight=edge_weight, order=order, smooth_type="smsf"
        ) + calc_smoothness_loss(
            dist2, img, edge_weight=edge_weight, order=order, smooth_type="smsf"
        )

        # print('chamfer-dist', chamfer_loss)
        # print('chamfer-smooth', lambda_smoothness * smoothness_loss)
        chamfer_loss += lambda_smoothness * smoothness_loss

    return chamfer_loss
"""


def calc_masks_non_occlusion(
    flow_forward_orig,
    flow_backward_orig,
    mask_type="wang",
    return_forward_and_backward=True,
    binary=False,
):
    # flow: torch.tensor : Bx2xHxW
    B, C, H, W = flow_forward_orig.size()
    dtype = flow_forward_orig.dtype
    device = flow_forward_orig.device

    if return_forward_and_backward:
        flow_forward = torch.ones(size=(B * 2, C, H, W), dtype=dtype, device=device)
        flow_forward[:B] = flow_forward_orig
        flow_forward[B:] = flow_backward_orig
        flow_backward = torch.ones(size=(B * 2, C, H, W), dtype=dtype, device=device)
        flow_backward[:B] = flow_backward_orig
        flow_backward[B:] = flow_forward_orig
    else:
        flow_backward = flow_backward_orig
        flow_forward = flow_forward_orig

    B, _, _, _ = flow_forward.size()

    masks_non_occlusion = torch.ones(size=(B, 1, H, W), dtype=dtype, device=device)
    if mask_type == "brox":

        flow_backward_warped = warp(
            flow_backward, flow_forward, return_masks_flow_inside=False
        )

        flows_forward_diff_squared = torch.sum(
            (flow_forward + flow_backward_warped) ** 2, dim=1, keepdim=True
        )
        # visualize_flow(torch.cat((flow_forward[0], flow_backward[0], flow_backward_warped[0]), dim=2))
        flows_forward_squared_sum = torch.sum(
            (flow_forward ** 2 + flow_backward_warped ** 2), dim=1, keepdim=True
        )

        masks_non_occlusion[
            flows_forward_diff_squared > (0.01 * flows_forward_squared_sum + 0.5)
        ] = 0.0

    elif mask_type == "wang":

        grid_xy = oflow2pxlcoords(flow_backward)
        # Bx2xHxW
        masks_non_occlusion = pxlcoords2mask(grid_xy)
        masks_non_occlusion = masks_non_occlusion.detach()

    # options self-mono-sf
    # opt1: use bool output:
    if binary:
        masks_non_occlusion = masks_non_occlusion > 0.5

    # opt2: set no occlusion if mask too small:
    # if torch.sum(masks_non_occlusion) < (B * H * W / 2):
    #    masks_non_occlusion[:] = 1.

    return masks_non_occlusion


def masks_ensure_non_zero(masks, thresh_perc=0.5):

    B, _, H, W = masks.shape

    if torch.sum(masks) < thresh_perc * (B * H * W):
        masks[:] = 1.0

    return masks


def pxlcoords2flow(pxlcoords):
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


def oflow2uv1(flow):
    B, _, H, W = flow.shape
    dtype = flow.dtype
    device = flow.device

    grid_uv = shape2pxlcoords(B=B, H=H, W=W, dtype=dtype, device=device)

    grid_uv = grid_uv + flow

    grid_1 = torch.ones(size=(B, 1, H, W), dtype=dtype, device=device)
    grid_uv1 = torch.cat((grid_uv, grid_1), dim=1)

    return grid_uv1


def oflow2pxlcoords(flow):
    B, _, H, W = flow.shape
    dtype = flow.dtype
    device = flow.device

    grid_uv = shape2pxlcoords(B=B, H=H, W=W, dtype=dtype, device=device)

    grid_uv = grid_uv + flow

    return grid_uv


def flow2pxlcoords_normalized(flow):
    B, C, H, W = flow.shape

    grid_xy = oflow2pxlcoords(flow)

    grid_xy = normalize_pxlcoords(grid_xy)

    return grid_xy


def shape2uv1(B, H, W, dtype, device):

    grid_uv = shape2pxlcoords(B, H, W, dtype, device)

    if B != 0:
        grid_1 = torch.ones(size=(B, 1, H, W), dtype=dtype, device=device)
        grid_uv1 = torch.cat((grid_uv, grid_1), dim=1)

    else:
        grid_1 = torch.ones(size=(1, H, W), dtype=dtype, device=device)
        grid_uv1 = torch.cat((grid_uv, grid_1), dim=0)

    return grid_uv1


def shape2pxlcoords(B, H, W, dtype, device):
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


def normalize_pxlcoords(grid_xy):
    # ensure normalize pxlcoords is no inplace
    grid_xy = grid_xy.clone()
    B, C, H, W = grid_xy.shape

    grid_xy[:, 0] = grid_xy[:, 0] / (W - 1.0) * 2.0 - 1.0
    grid_xy[:, 1] = grid_xy[:, 1] / (H - 1.0) * 2.0 - 1.0

    return grid_xy


def pxlcoords2mask(grid_xy, binary=False):
    # input: B x 2 x H x W
    B, _, H, W = grid_xy.shape
    dtype = grid_xy.dtype
    device = grid_xy.device

    grid_xy_floor = torch.floor(grid_xy).long()
    # B x 2 x H x W

    grid_xy_offset = grid_xy - grid_xy_floor
    # B x 2 x H x W

    kernel_xy = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int64, device=device
    )
    kernel_xy = kernel_xy.unsqueeze(0).unsqueeze(3).unsqueeze(4)
    K = 4
    # 1 x 4 x 2 x 1 x 1
    grid_weights_offset_x = 1.0 - torch.abs(
        kernel_xy[:, :, 0] - grid_xy_offset[:, 0].unsqueeze(1)
    )
    grid_weights_offset_y = 1.0 - torch.abs(
        kernel_xy[:, :, 1] - grid_xy_offset[:, 1].unsqueeze(1)
    )
    #                                      1 x 4 x 1 x 1      - B x 1 x H x W

    grid_weights = grid_weights_offset_x * grid_weights_offset_y
    # B x 4 x H x W

    grid_weights = grid_weights.reshape(B, K, -1)
    # grid_weights = torch.ones_like(grid_weights)

    grid_indices_x = grid_xy_floor[:, 0].unsqueeze(1) + kernel_xy[:, :, 0]
    grid_indices_y = grid_xy_floor[:, 1].unsqueeze(1) + kernel_xy[:, :, 1]
    #                 B x 1 x H x W                   + 1 x 4 x 1 x 1
    # -> B x 4 x H x W

    grid_indices_x = grid_indices_x.reshape(B, K, -1)
    grid_indices_y = grid_indices_y.reshape(B, K, -1)
    # B x 4 x H*W

    grid_xy_indices = grid_indices_y * W + grid_indices_x
    # B x 4 x H*W

    grid_counts = torch.zeros(
        (B, K, H * W), requires_grad=False, dtype=dtype, device=device
    )

    # problem not in parallel: the number of valid indices are different for each batch and kernel_offset
    for b in range(B):
        for i in range(K):
            valid_indices_mask = (
                (grid_indices_x[b, i] >= 0)
                & (grid_indices_x[b, i] < W)
                & (grid_indices_y[b, i] >= 0)
                & (grid_indices_y[b, i] < H)
            )
            # B x H*W

            grid_xy_indices_i = grid_xy_indices[b, i]
            # B x H*W
            grid_xy_indices_i = grid_xy_indices_i[valid_indices_mask].reshape(-1)
            # B x valid(H*W)

            grid_weights_i = grid_weights[b, i]
            # B x H*W
            grid_weights_i = grid_weights_i[valid_indices_mask].reshape(-1)
            # B x valid(H*W)

            grid_counts[b, i] = grid_counts[b, i].scatter_add(
                dim=0, index=grid_xy_indices_i, src=grid_weights_i
            )
            # B x 4 x H*W

    grid_counts = grid_counts.reshape((B, K, H, W))
    grid_counts = torch.sum(grid_counts, dim=1, keepdim=True)

    grid_counts = torch.clamp(grid_counts, 0.0, 1.0)
    masks_non_occlusion = grid_counts

    masks_non_occlusion = masks_non_occlusion.detach()

    if binary:
        masks_non_occlusion = masks_non_occlusion > 0.5

    return masks_non_occlusion


def backproject_pixels(intrinsics, disp):
    # disp: B x 1 x H x W

    B, C, H, W = disp.shape
    dtype = disp.dtype
    device = disp.device

    grid_y, grid_x = torch.meshgrid(
        [
            torch.arange(0.0, H, dtype=dtype, device=device),
            torch.arange(0.0, W, dtype=dtype, device=device),
        ]
    )

    grid_z = torch.ones(size=(H, W), dtype=dtype, device=device)
    grid_xyz = torch.stack((grid_x, grid_y, grid_z), dim=1)
    # B x 3 x H x W

    # Nx3 * 3x3 * Nx1 = Nx3


#    pts_mat = torch.matmul(torch.inverse(intrinsics.cpu()).cuda(), pixel_mat) * depth_mat
#    pts = pts_mat.view(b, -1, h, w)
#
#    return points_3d

# def sceneflow2

# monodepth 2 implementation photometric loss


def compute_reprojection_loss(self, pred, target):
    """Computes reprojection loss between a batch of predicted and target images"""
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if self.opt.no_ssim:
        reprojection_loss = l1_loss
    else:
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


##


def calc_fb_consistency(
    flow_forward, flow_backward, fb_sigma, masks_flow_inside_forward=None
):
    # in : Bx2xHxW
    # out: Bx1xHxW
    H = flow_forward.size(2)
    W = flow_forward.size(3)

    if masks_flow_inside_forward == None:
        flow_backward_warped, masks_flow_inside_forward = warp(
            flow_backward, flow_forward, return_masks_flow_inside=True
        )
    else:
        flow_backward_warped = warp(
            flow_backward, flow_forward, return_masks_flow_inside=False
        )

    flows_diff_squared = torch.sum(
        (flow_forward + flow_backward_warped) ** 2, dim=1, keepdim=True
    )

    fb_consistency = torch.exp(
        -(flows_diff_squared / (fb_sigma ** 2 * (H ** 2 + W ** 2)))
    )

    fb_consistency = fb_consistency * masks_flow_inside_forward
    # flows_squared_sum = torch.sum((flow_forward ** 2 + flow_backward ** 2), dim=1, keepdim=True)

    return fb_consistency


def calc_selfsup_loss(
    student_flow_forward,
    student_flow_backward,
    teacher_flow_forward,
    teacher_flow_backward,
    crop_reduction_size=16,
):

    B, _, H, W = student_flow_forward.shape

    teacher_fb_sigma = 0.003
    student_fb_sigma = 0.03

    student_fb_consistency_forward = calc_fb_consistency(
        student_flow_forward, student_flow_backward, student_fb_sigma
    )
    teacher_fb_consistency_forward = calc_fb_consistency(
        teacher_flow_forward, teacher_flow_backward, teacher_fb_sigma
    )
    teacher_fb_consistency_forward = teacher_fb_consistency_forward[
        :,
        :,
        crop_reduction_size:-crop_reduction_size,
        crop_reduction_size:-crop_reduction_size,
    ]
    teacher_fb_consistency_forward = torch.nn.functional.interpolate(
        teacher_fb_consistency_forward,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )

    student_fb_consistency_backward = calc_fb_consistency(
        student_flow_backward, student_flow_forward, student_fb_sigma
    )
    teacher_fb_consistency_backward = calc_fb_consistency(
        teacher_flow_backward, teacher_flow_forward, teacher_fb_sigma
    )

    teacher_fb_consistency_backward = teacher_fb_consistency_backward[
        :,
        :,
        crop_reduction_size:-crop_reduction_size,
        crop_reduction_size:-crop_reduction_size,
    ]
    teacher_fb_consistency_backward = torch.nn.functional.interpolate(
        teacher_fb_consistency_backward,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    # Bx1xHxW

    teacher_flow_forward = teacher_flow_forward[
        :,
        :,
        crop_reduction_size:-crop_reduction_size,
        crop_reduction_size:-crop_reduction_size,
    ]

    teacher_flow_forward = torch.nn.functional.interpolate(
        teacher_flow_forward, size=(H, W), mode="bilinear", align_corners=False
    )

    teacher_flow_forward[:, 0] = (
        teacher_flow_forward[:, 0] * W / (W - 2 * crop_reduction_size)
    )
    teacher_flow_forward[:, 1] = (
        teacher_flow_forward[:, 1] * H / (H - 2 * crop_reduction_size)
    )

    teacher_flow_backward = teacher_flow_backward[
        :,
        :,
        crop_reduction_size:-crop_reduction_size,
        crop_reduction_size:-crop_reduction_size,
    ]

    teacher_flow_backward[:, 0] = (
        teacher_flow_backward[:, 0] * W / (W - 2 * crop_reduction_size)
    )
    teacher_flow_backward[:, 1] = (
        teacher_flow_backward[:, 1] * H / (H - 2 * crop_reduction_size)
    )

    teacher_flow_backward = torch.nn.functional.interpolate(
        teacher_flow_backward, size=(H, W), mode="bilinear", align_corners=False
    )

    # valid_warp_mask adds mask for if warping is outisde of frame
    weights_forward = (1.0 - student_fb_consistency_forward) * (
        teacher_fb_consistency_forward
    )  # * valid_warp_mask (forward)
    weights_forward = weights_forward.detach()

    weights_backward = (1.0 - student_fb_consistency_backward) * (
        teacher_fb_consistency_backward
    )  # * valid_warp_mask (forward)
    weights_backward = weights_backward.detach()
    # weights.requires_grad = False
    # Bx1xHxW

    teacher_flow_forward = teacher_flow_forward.detach()
    teacher_flow_backward = teacher_flow_backward.detach()
    selfsup_loss_forward = calc_charbonnier_loss(
        student_flow_forward, teacher_flow_forward, weights_forward
    )
    selfsup_loss_backward = calc_charbonnier_loss(
        student_flow_backward, teacher_flow_backward, weights_backward
    )

    return (selfsup_loss_forward + selfsup_loss_backward) / 2.0


def draw_arrows_in_rgb(img, start, end):
    _, H, W = img.shape
    device = img.device
    dtype = img.dtype

    img = tensor_to_cv_img(img.clone())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    threshold = 2.0

    start = start.clone().permute(1, 2, 0)
    end = end.clone().permute(1, 2, 0)

    start = start.detach().cpu().numpy()
    end = end.detach().cpu().numpy()

    norm = np.linalg.norm(end - start, axis=2)
    norm[norm < threshold] = 0

    # Draw all the nonzero values
    nz = np.nonzero(norm)

    skip_amount = (len(nz[0]) // 100) + 1

    for i in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(
            img,
            pt1=tuple(start[y, x]),
            pt2=tuple(end[y, x]),
            color=(0, 200, 0),
            thickness=2,
            tipLength=0.2,
        )

    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.to(device)

    return img


def visualize_flow(flow, draw_arrows=False, resize=True):
    # x_in: 2xHxW

    if resize:
        _, H, W = flow.shape
        max_H = 720
        max_W = 1280
        scale_factor = min(max_H / H, max_W / W)
        flow = torch.nn.functional.interpolate(
            flow.unsqueeze(0), scale_factor=(scale_factor, scale_factor)
        )[0]

    rgb = flow2rgb(flow, draw_arrows=draw_arrows)

    img = tensor_to_cv_img(rgb)

    cv2.imshow("flow", img)
    cv2.waitKey(0)


def visualize_hist(x):
    x = x.flatten().cpu().detach().numpy()
    plt.hist(x)
    plt.show()


def visualize_img(rgb):
    # img: 3xHxW
    rgb = rgb.clone()
    img = tensor_to_cv_img(rgb)

    cv2.imshow("img", img)
    cv2.waitKey(0)


def visualize_imgpair(imgpair):
    # imgpair: 6xHxW
    img1 = imgpair[:3]
    img2 = imgpair[3:]
    img = torch.cat((img1, img2), dim=2)

    visualize_img(img)


def visualize_point_cloud(xyz, rgb=None, mask=None):
    import pptk

    # input: 3xHxW
    if mask == None:
        mask = (xyz[2, :, :] > 0.0) * (xyz[2, :, :] < 80.0)

    mask = mask.reshape(-1)
    xyz = xyz.reshape(3, -1)
    xyz = xyz[:, mask]
    xyz = xyz.permute(1, 0)
    xyz_np = xyz.cpu().detach().numpy()
    # input pptk.viewer: numpy.ndarray, xyz: Nx3, (np.float64 works for sure)

    v = pptk.viewer(xyz_np)

    if rgb is not None:
        rgb = rgb.reshape(3, -1)
        rgb = rgb[:, mask]
        rgb = rgb.permute(1, 0)
        rgb_np = rgb.cpu().detach().numpy()
        v.attributes(rgb_np)

    v.set(point_size=0.01)
    v.set(phi=1.5 * np.pi)
    v.set(theta=-0.5 * np.pi)

    print("press enter to continue.")
    v.wait()
    v.close()

    """
    poses = []
    #(x, y, z, phi, theta, r)
    poses.append([0, 0, 0, 0 * np.pi/2, np.pi/4, 5])

    v.play(poses, 2 * np.arange(5), repeat=True, interp='linear')
    """

    # while True:
    # print('lookat:', v.get('lookat'))
    # print('view:', v.get('view'))
    # print('eye:', v.get('eye'), 'r:', v.get('r'))
    # print('phi:', v.get('phi'), 'theta:', v.get('theta'))


def rescale_intrinsics(proj_mats, reproj_mats, sx, sy):
    # sx = target_W / W
    # sy = target_H / H

    proj_mats[:, 0, :] = proj_mats[:, 0, :] * sx
    proj_mats[:, 1, :] = proj_mats[:, 1, :] * sy
    reproj_mats[:, :, 0] = reproj_mats[:, :, 0] / sx
    reproj_mats[:, :, 1] = reproj_mats[:, :, 1] / sy

    return proj_mats, reproj_mats


def depth2xyz(depth, reproj_mats, oflow=None):
    B, _, H, W = depth.shape

    dtype = depth.dtype
    device = depth.device

    if oflow == None:
        grid_uv1 = shape2uv1(B=0, H=H, W=W, dtype=dtype, device=device)
        uv1 = grid_uv1.reshape(3, -1)
        # 3 x N
    else:
        # grid_uv1 = shape2uv1(B=B, H=H, W=W, dtype=dtype, device=device)
        # grid_uv1 += flow
        grid_uv1 = oflow2uv1(oflow)
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


def depth_oflow2xyz(depth, oflow, reproj_mats):
    B, _, H, W = depth.shape

    dtype = depth.dtype
    device = depth.device

    grid_uv1 = oflow2uv1(oflow)
    # B x 3 x H x W
    # grid_uv1 = shape2uv1(B=0, H=H, W=W, dtype=dtype, device=device)

    uv1 = grid_uv1.reshape(B, 3, -1)
    # 3 x N

    xyz = torch.matmul(reproj_mats, uv1) * depth.flatten(2)
    # (B x 3 x N) = (B x 3 x 3 * B x 3 x N)  .* B x 3 x N

    xyz = xyz.reshape(B, 3, H, W)

    return xyz


def xyz2uv(xyz, proj_mats):
    # 3D-2D Projection:
    # u = (fx*x + cx * z) / z
    # v = (fy*y + cy * z) / z
    # shift on plane: delta_x = (fx * bx) / z
    #                 delta_y = (fy * by) / z
    # uv = (P * xyz) / z
    # P = [ fx   0  cx]
    #     [ 0   fy  cy]

    B, _, H, W = xyz.shape

    xyz = xyz.reshape(B, 3, -1)
    # 3 x N
    # z = torch.abs(xyz[:, 2].clone()) + 1e-8
    # uv = torch.matmul(proj_mats, xyz[:, :2]) / z.unsqueeze(1)

    xyz[:, 2] = torch.abs(xyz.clone()[:, 2]) + 1e-8
    uv = torch.matmul(proj_mats, xyz) / (xyz[:, 2] + 1e-8).unsqueeze(1)

    # uv = torch.div(torch.matmul(proj_mats, xyz), (xyz[:, 2] + 1e-8).unsqueeze(1))

    # 2xN
    uv = uv.reshape(B, 2, H, W)

    return uv


def disp2depth(disp, fx):
    # disp: Bx1xHxW
    # note for kitti-dataset: baseline=0.54 -> 54 cm
    # depth = focal-length * baseline / disparity
    disp = torch.clamp(disp, 0)
    fx = fx[..., None, None, None]

    depth = fx * 0.54 / (disp + 1e-8)
    depth = torch.clamp(depth, 1e-3, 80)

    # https: // github.com / visinf / self - mono - sf / tree / master / models

    return depth


def depth2disp(depth, fx):
    fx = fx[..., None, None, None]

    disp = fx * 0.54 / (depth + 1e-8)

    return disp


def disp2xyz(disp, proj_mats, reproj_mats, oflow=None):
    depth = disp2depth(disp, fx=proj_mats[:, 0, 0])
    xyz = depth2xyz(depth, reproj_mats, oflow)
    return xyz


def disp_oflow2xyz(disp, oflow, proj_mats, reproj_mats):
    depth = disp2depth(disp, fx=proj_mats[:, 0, 0])
    xyz = depth_oflow2xyz(depth, oflow, reproj_mats)
    return xyz


def transformation(xyz, transformation_matrix):
    # xyz: 3xHxW, transformation_matrix=4x4
    _, H, W = xyz.shape
    dtype = xyz.dtype
    device = xyz.device

    grid_1 = torch.ones(size=(1, H, W), dtype=dtype, device=device)

    xyz1 = torch.cat((xyz, grid_1), dim=0)

    xyz1 = xyz1.reshape(4, -1)
    xyz1 = torch.mm(transformation_matrix, xyz1)

    xyz = xyz1.reshape(4, H, W)[:3]

    return xyz


def transformation2flow(xyz, transformation_matrix, proj_mat):

    _, H, W = xyz.shape
    dtype = xyz.dtype
    device = xyz.device

    xyz = transformation(xyz, transformation_matrix)

    uv_transf = xyz2uv(xyz, proj_mat)

    v, u = torch.meshgrid(
        [
            torch.arange(0.0, H, dtype=dtype, device=device),
            torch.arange(0.0, W, dtype=dtype, device=device),
        ]
    )

    uv = torch.stack((u, v))

    flow = uv_transf - uv

    return flow


def get_stereo_points_transf(dtype, device):

    transf = torch.tensor(
        [
            [1.0, 0.0, 0.0, -0.54],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )

    return transf


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def tensor_to_cv_img(x_in):
    # x_in : CxHxW float32
    # x_out : HxWxC uint8
    x_in = x_in * 1.0
    x_in = torch.clamp(x_in, min=0.0, max=1.0)
    x_out = (x_in.permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
    x_out = x_out[:, :, ::-1]
    return x_out
