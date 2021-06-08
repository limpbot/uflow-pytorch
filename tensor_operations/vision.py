import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
import os
import tensor_operations.geometric as ops_geo
import tensor_operations.rearrange as ops_rearr
import open3d as o3d


def create_vwriter(name, width, height, fps=10):
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    vwriter = cv2.VideoWriter(
        os.path.join("videos", name + ".mp4"),
        fourcc,
        fps,
        (width, height),
    )
    return vwriter


def resize(x, H_out=None, W_out=None, scale_factor=None, mode="bilinear"):
    # mode 'nearest' or 'bilinear'
    # in: BxCxHxW
    # out: BxCxHxW

    if H_out != None and W_out != None:

        if mode != "nearest":
            x_out = torch.nn.functional.interpolate(
                x,
                size=(H_out, W_out),
                mode=mode,
                align_corners=True,
            )
        else:
            x_out = torch.nn.functional.interpolate(x, size=(H_out, W_out), mode=mode)
    elif scale_factor != None:
        if mode != "nearest":
            x_out = torch.nn.functional.interpolate(
                x,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=True,
            )
        else:
            x_out = torch.nn.functional.interpolate(
                x,
                scale_factor=scale_factor,
                mode=mode,
            )
    else:
        print("warning: could not resize. pls specify H_out/W_out or scale")
        x_out = x
    return x_out


def rescale_intrinsics(proj_mats, reproj_mats, sx, sy):
    # sx = target_W / W
    # sy = target_H / H

    proj_mats[:, 0, :] = proj_mats[:, 0, :] * sx
    proj_mats[:, 1, :] = proj_mats[:, 1, :] * sy
    reproj_mats[:, :, 0] = reproj_mats[:, :, 0] / sx
    reproj_mats[:, :, 1] = reproj_mats[:, :, 1] / sy

    return proj_mats, reproj_mats


def rgb_2_grayscale(x):
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

    endpoints = ops_geo.oflow_2_pxl2d(flow.unsqueeze(0))[0]

    startpoints = ops_geo.oflow_2_pxl2d(0.0 * flow.unsqueeze(0))[0]

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
    # depth range: 1/80 - 1/ 1e-3
    np_depth = 1.0 / np_depth * 7
    np_depth = np.clip(np_depth, 0, 1)

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


def mask2rgb(torch_mask, binary_mask=False):
    K, H, W = torch_mask.shape
    device = torch_mask.device
    torch_mask = torch_mask.type(torch.float32)

    if binary_mask:
        min_prob = 0.5  # 1.0 / K
        torch_mask = (torch_mask > min_prob) * 1.0

    torch_colors = get_colors(K, device=device)

    # Kx3x1x1 * Kx1xHxW
    torch_mask_rgb = torch.sum(
        torch_colors.unsqueeze(-1).unsqueeze(-1) * torch_mask.unsqueeze(1),
        dim=0,
    )
    # 3xHxW
    torch_mask_rgb = torch_mask_rgb / (
        torch.clamp(torch.max(torch_mask_rgb, dim=0, keepdim=True)[0], 1.0, np.inf)
        + 1e-7
    )

    # 3 x H x W
    for i in range(K):
        width = int(W / 20)
        height = int(H / K)
        torch_mask_rgb[:, i * height : (i + 1) * height - int(height / 10), :width] = (
            torch_colors[i].unsqueeze(-1).unsqueeze(-1)
        )

    torch_mask_rgb = torch_mask_rgb.to(device)

    return torch_mask_rgb

def label2onehot(labels):
    device = labels.device
    onehot = torch.arange(labels.min(), labels.max() + 1).to(device)[None, :, None, None] == labels
    onehot = onehot * 1.0
    return onehot

def disp2rgb(disp):
    disp_np = (disp).detach().cpu().numpy()

    vmax = 80  # np.percentile(disp_np, 95)
    disp_np = disp_np / vmax

    disp_np = torch.from_numpy(disp_np).permute(1, 2, 0)

    disp_rgb = matplotlib.cm.get_cmap("magma")(disp_np)[:, :, 0, :3]

    disp_rgb = torch.from_numpy(disp_rgb).permute(2, 0, 1)

    disp_rgb = disp_rgb.to(disp.device)

    return disp_rgb


def flow2rgb(flow_torch, draw_arrows=False, srcs_flow=None):
    _, H, W = flow_torch.shape
    flow_torch_1 = flow_torch.clone()

    flow = flow_torch_1.detach().cpu().numpy()

    size_wheel = 70
    offset_x = W - 3 * int(size_wheel / 2)
    offset_y = H - 3 * int(size_wheel / 2)
    for y in range(size_wheel):
        for x in range(size_wheel):
            radius = ((y - size_wheel / 2.0) ** 2 + (x - size_wheel / 2.0) ** 2) ** 0.5
            if radius <= size_wheel / 2.0:
                flow[0, offset_y + y, offset_x + x] = x - size_wheel / 2.0
                flow[1, offset_y + y, offset_x + x] = y - size_wheel / 2.0

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


def draw_arrows_in_rgb(img, start, end):
    _, H, W = img.shape
    device = img.device
    dtype = img.dtype

    img = tensor_to_cv_img(img.clone())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


import matplotlib.pyplot as plt


def get_image_from_plot(fig, ax):
    ax.axis("off")
    fig.tight_layout(pad=0)

    # To remove the huge white borders
    ax.margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )

    return image_from_plot


def visualize_flow(flow, draw_arrows=False, resize=True, duration=0):
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
    cv2.waitKey(duration)


def visualize_hist(x, split_dim=1, max=0.01):
    if split_dim == None:
        x = x.flatten().cpu().detach().numpy()
        x = x[x <= max]
    else:
        x = list(torch.split(x, 1, dim=split_dim))
        for i in range(len(x)):
            x[i] = x[i].flatten().cpu().detach().numpy()
            x[i] = x[i][x[i] <= max]

    plt.clf()
    plt.hist(x)
    plt.show(block=False)
    plt.pause(0.001)


def visualize_img(rgb, duration=0, vwriter=None, fpath=None):
    # img: 3xHxW
    rgb = rgb.clone()
    img = tensor_to_cv_img(rgb)

    if vwriter is not None:
        vwriter.write(img)

    if fpath is not None:
        cv2.imwrite(fpath, img)

    cv2.imshow("img", img)
    cv2.waitKey(duration)


def visualize_imgpair(imgpair):
    # imgpair: 6xHxW
    img1 = imgpair[:3]
    img2 = imgpair[3:]
    img = torch.cat((img1, img2), dim=2)

    visualize_img(img)


def tensor_to_cv_img(x_in):
    # x_in : CxHxW float32
    # x_out : HxWxC uint8
    x_in = x_in * 1.0
    x_in = torch.clamp(x_in, min=0.0, max=1.0)
    x_out = (x_in.permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
    x_out = x_out[:, :, ::-1]
    return x_out


def render_pts3d(pts):
    pass
    # pcd = o3d.io.read_point_cloud("pcds/entire_hall_3d_cropped_binary.pcd")
    # downpcd.paint_uniform_color(np.zeros([3, 1]))
    # o3d.visualization.draw_geometries([downpcd])


def visualize_sflow(pts1, pts2, img=None):
    dims = pts1.dim()

    if dims == 4:
        _, _, H, W = pts1.shape
        pts = torch.cat(
            (pts1.permute(1, 0, 2, 3).flatten(1), pts2.permute(1, 0, 2, 3).flatten(1)),
            dim=1,
        )
        pts = pts.permute(1, 0)
        if img is not None:
            img = img.permute(1, 0, 2, 3).flatten(1).permute(1, 0)

    elif dims == 3:
        _, H, W = pts1.shape
        # 3 x H x W
        pts = torch.cat(
            (pts1.flatten(1), pts2.flatten(1)),
            dim=1,
        )
        pts = pts.permute(1, 0)
        if img is not None:
            img = img.flatten(1).permute(1, 0)
    elif dims == 2:
        H = 256
        W = 832
        pts = torch.cat(
            (pts1, pts2),
            dim=0,
        )
        # expects to be Nx3 then
        pass
    else:
        print("error: input for scene flow visualization must be 2D,3D or 4D.")
        return 0
    # pts: N x 3
    N, _ = pts.shape
    N = int(N / 2)
    pts = pts.detach().cpu().numpy()
    pairs = torch.arange(N).repeat(2, 1)
    pairs[1:2, :] += N
    pairs = pairs.permute(1, 0)
    pairs = pairs.detach().cpu().numpy()
    # N x 2
    pts = np.expand_dims(pts, axis=2)
    pairs = np.expand_dims(pairs, axis=2)
    if img == None:
        colors = [[0, 0.5, 0] for i in range(len(pairs))]
    else:
        # Nx3
        img = img.detach().cpu().numpy() * 0.5
        img = np.expand_dims(img, axis=2)
        colors = img

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(pairs),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    up = np.expand_dims(np.array([0.0, -1.0, 0.0], dtype=np.float), axis=1)
    # looking direction
    lookat = np.expand_dims(np.array([0.0, 0.0, 10.0], dtype=np.float), axis=1)
    # pos
    front = np.expand_dims(np.array([0.0, -0.2, -1.1], dtype=np.float), axis=1)
    zoom = 0.0000000001
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=W, height=H)
    vis.add_geometry(line_set)
    # o3d.visualization.draw_geometries(
    #    [line_set], lookat=lookat, up=up, front=front, zoom=0.01
    # )
    vis.poll_events()
    vis.update_renderer()

    view_control = vis.get_view_control()
    view_control.set_lookat(lookat)
    view_control.set_up(up)
    view_control.set_front(front)
    view_control.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()

    # vis.run()
    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    return img

def erode(x, patch_size, thresh=1.0):
    x = ops_rearr.neighbors_to_channels(x * 1.0, patch_size)
    avg = torch.mean(x, dim=1, keepdim=True)
    return (avg >= thresh)

def dilate(x, patch_size):
    x = ops_rearr.neighbors_to_channels(x * 1.0, patch_size)
    avg = torch.mean(x, dim=1, keepdim=True)
    return (avg > 0.)

def choose_from_neighborhood(x, patch_size):
    x = x.permute(1, 0, 2, 3)
    x = ops_rearr.neighbors_to_channels(x * 1.0, patch_size)
    avg = torch.mean(x, dim=1, keepdim=True)
    id = torch.argmax(avg, dim=0, keepdim=True)
    return label2onehot(id)


class PhotoTransform:
    def __init__(self, device=None):
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def __call__(self, imgpair):
        imgpair = self.random_swap_channels(imgpair)
        imgpair = self.random_adjust_hue(imgpair)

        return imgpair

    def random_swap_channels(self, imgpair):
        # imgpair: 6 x H x W
        _, H, W = imgpair.shape

        channel_indices = torch.arange(3)
        # 3
        offset = torch.randint(low=0, high=3, size=(1,))
        reverse = torch.randint(low=0, high=2, size=(1,))
        # 1
        channel_indices = ((3 ** reverse) - 1) + ((channel_indices + offset) % 3) * (
            (-1) ** reverse
        )
        # 3
        channel_indices = torch.cat((channel_indices, channel_indices + 3))
        imgpair = imgpair[channel_indices]
        # 6 x H x W

        return imgpair

    def random_adjust_hue(self, imgpair):

        device = imgpair.device

        imgpair[:3] = kornia.color.rgb_to_hsv(imgpair[:3])
        imgpair[3:] = kornia.color.rgb_to_hsv(imgpair[3:])

        hue = torch.rand(size=(1,), device=device) - 0.5

        imgpair[2] = imgpair[2] + hue
        imgpair[5] = imgpair[5] + hue

        imgpair[:3] = kornia.color.hsv_to_rgb(imgpair[:3])
        imgpair[3:] = kornia.color.hsv_to_rgb(imgpair[3:])

        imgpair = torch.clamp(input=imgpair, min=0.0, max=1.0)

        return imgpair
