import torch


def calc_batch_gradients(batch, shift=1):
    # batch: torch.tensor: BxCxHxW
    # shift = 1 + margin
    batch_grad_x = batch[:, :, :, shift:] - batch[:, :, :, :-shift]
    # B x 2 x H x W-shift

    batch_grad_y = batch[:, :, shift:, :] - batch[:, :, :-shift, :]
    # B x 2 x H-shift x W

    return batch_grad_x, batch_grad_y


def calc_img_gradients(img, margin=1):
    margin = 1
    _, H, W = img.shape
    batch = torch.nn.functional.pad(
        img.unsqueeze(0), pad=[margin, margin, margin, margin], mode="replicate"
    )
    batch_grad_x, batch_grad_y = calc_batch_gradients(batch=batch, shift=margin + 1)
    batch_grad_x = batch_grad_x[:, :, margin : H + margin, :]
    batch_grad_y = batch_grad_y[:, :, :, margin : W + margin]

    img_grad_x = batch_grad_x[0]
    img_grad_y = batch_grad_y[0]

    return img_grad_x, img_grad_y


def calc_div2d(img2d):
    _, H, W = img2d.shape

    img2d_grad_x, img2d_grad_y = calc_img_gradients(img2d, margin=1)

    # img2d_div = torch.stack((img2d_grad_x[0], img2d_grad_y[1]), dim=0)
    img2d_div = img2d_grad_x[0] + img2d_grad_y[1]
    return img2d_div.unsqueeze(0)


def minimize_tv(img, step_size=10):
    alpha = 0.8
    sigma = 0.25
    tau = 1.0

    _, H, W = img.shape
    u = torch.rand(size=(1, H, W))
    p = 1.0 * torch.ones(size=(2, H, W))

    for i in range(step_size):

        u_grad_x, u_grad_y = calc_img_gradients(u, margin=1)
        u_grad = torch.cat((u_grad_x, u_grad_y), dim=0)

        p = (p + sigma * alpha * u_grad) / torch.max(
            torch.ones(size=(2, H, W)) / alpha, (torch.abs(p + sigma * alpha * u_grad))
        )

        u = (
            u + tau * (2 * torch.mean(img, dim=0, keepdim=True) + alpha * calc_div2d(p))
        ) / (1.0 + 2.0 * tau)

        # u = u_new + theta * (u_new - u)
        u = torch.clamp(u, 0.0, 1.0)
    return u
