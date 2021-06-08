import torch
import tensor_operations.geometric as og
import tensor_operations.vision as ops_vis

def pxl2d_2_mask_non_occl(grid_xy, binary=False):
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


def oflow_2_mask_inside(flow):

    pxl2d_normalized = og.oflow_2_pxl2d_normalized(flow)
    mask_inside = pxl2d_normalized_2_mask_inside(pxl2d_normalized)

    return mask_inside


def pxl2d_2_mask_inside(pxl2d):
    pxl2d_normalized = og.pxl2d_2_pxl2d_normalized(pxl2d)
    mask_inside = pxl2d_normalized_2_mask_inside(pxl2d_normalized)
    return mask_inside


def pxl2d_normalized_2_mask_inside(pxl2d_normalized):
    B, C, H, W = pxl2d_normalized.size()
    dtype = pxl2d_normalized.dtype
    device = pxl2d_normalized.device

    pxl2d_normalized = pxl2d_normalized.permute(0, 2, 3, 1)
    # B x H x W x C
    mask_valid_flow = torch.ones(size=(B, H, W), dtype=dtype, device=device)
    mask_valid_flow[pxl2d_normalized[:, :, :, 0] > 1.0] = 0.0
    mask_valid_flow[pxl2d_normalized[:, :, :, 0] < -1.0] = 0.0
    mask_valid_flow[pxl2d_normalized[:, :, :, 1] > 1.0] = 0.0
    mask_valid_flow[pxl2d_normalized[:, :, :, 1] < -1.0] = 0.0
    mask_valid_flow = mask_valid_flow.unsqueeze(1)
    # Bx1xHxW

    return mask_valid_flow


def pxl2d_2_mask_valid(pxl2d_tf_fwd, pxl2d_tf_bwd):
    # in: Bx2xHxW
    # out: Bx1xHxW

    mask_inside = pxl2d_2_mask_inside(pxl2d_tf_fwd)
    mask_non_occl = pxl2d_2_mask_non_occl(pxl2d_tf_bwd)

    return mask_inside * mask_non_occl


def mask_ensure_non_zero(masks, thresh_perc=0.5):

    B, _, H, W = masks.shape

    if torch.sum(masks) < thresh_perc * (B * H * W):
        masks[:] = 1.0

    return masks


def filter_size(masks, min_samples):
    K = len(masks)

    counts = torch.sum(masks.flatten(1), dim=1)

    masks = masks[counts > min_samples, :, :]

    return masks


def filter_multiple_assignment(masks, min_samples, max_assignments=1):
    # in: K x H x W
    # out: K x H x W
    # filter multiple assignments

    mask_multiple_assignments = (torch.sum(masks, dim=0) > max_assignments)
    masks[:, mask_multiple_assignments] = False

    # masks = torch.cat((masks, mask_multiple_assignments[None, ]), dim=0)

    masks = filter_size(masks, min_samples)

    return masks

def filter_overlap(masks, max_overlap=0.5):
    # in: K x N or K x H x W

    if len(masks.shape) == 2:
        sums = torch.sum(masks[:, None] * masks[None,], dim=2)
        sums_diag = torch.diag(sums)[:, None] + torch.diag(sums)[None, :]
    elif len(masks.shape) == 3:
        sums = torch.sum(masks.flatten(1)[:, None] * masks.flatten(1)[None,], dim=2)
        sums_diag = torch.diag(sums)[:, None] + torch.diag(sums)[None, :]
    else:
        print("error: filter_overlap requires input shape to have length 2 or 3")
        return masks
    overlaps = 2 * sums / sums_diag
    min_ids = torch.diag(-sums).argsort()

    ids_filtered = []
    unselected = torch.zeros_like(min_ids) == 0
    for id in min_ids:
        if unselected[id]:
            ids_filtered.append(id)
        unselected[overlaps[id] > max_overlap] = False

    masks_filtered = []
    for id in ids_filtered:
        masks_filtered.append(masks[id])

    masks = masks_filtered
    masks = torch.stack(masks)

    return masks


def filter_erode(objects_masks, erode_patchsize, erode_threshold, min_samples, valid_pts):
    objects_masks = ops_vis.erode(objects_masks[:, None, ], patch_size=erode_patchsize, thresh=erode_threshold)[:, 0]
    # objects_masks = ops_vis.dilate(objects_masks[:, None,], patch_size=3)[:, 0]
    objects_masks = objects_masks * valid_pts
    objects_masks = filter_size(objects_masks, min_samples)

    return objects_masks

def filter_interconnected(objects_masks, dists_div, rigid_dists_max_div):
    K, H_down, W_down = objects_masks.shape
    objects_masks = objects_masks.reshape(K, H_down * W_down)
    for k in range(K):
        objects_masks[k] = objects_masks[k]
        dists_div_masked = dists_div[objects_masks[k, None, :] * objects_masks[k, :, None]]
        num_pixel = int(dists_div_masked.shape[0] ** 0.5)
        dists_div_masked = dists_div_masked.reshape(num_pixel, num_pixel)
        interconnected_masked = torch.mean(1.0 * (dists_div_masked < rigid_dists_max_div), dim=1) > 0.5
        interconnected = torch.ones_like(objects_masks[0], dtype=torch.bool)
        interconnected[objects_masks[k]] = interconnected_masked
        objects_masks[k] *= interconnected
    objects_masks = objects_masks.reshape(K, H_down, W_down)

    return objects_masks