import math
from torch import nn
from torch.autograd import Function
import torch
import NTfm3D_cuda

torch.manual_seed(42)


class NTfm3DFunction(Function):
    @staticmethod
    def forward(ctx, points, masks, transforms):
        batch_size, num_channels, data_height, data_width = points.size()
        num_se3 = masks.size()[1]
        assert num_channels == 3
        assert masks.size() == torch.Size(
            [batch_size, num_se3, data_height, data_width]
        )
        assert transforms.size() == torch.Size(
            [batch_size, num_se3, 3, 4]
        )  # Transforms [R|t]

        # Create output (or reshape)
        output = points.new_zeros(*points.size())

        # se3layers.NTfm3D_forward_cuda(points, masks, transforms, output)

        NTfm3D_cuda.forward(
            points.contiguous(), masks.contiguous(), transforms.contiguous(), output
        )

        # print("output", output)

        # variables = output
        # ctx.save_for_backward(*variables) # Save for BWD pass
        ctx.save_for_backward(points, masks, transforms, output)  # Save for BWD pass

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Get saved tensors
        points, masks, transforms, output = ctx.saved_tensors
        # print("grad_output",grad_output)
        # print("masks",masks)
        # print("transforms",transforms)
        # print("output",output)
        assert grad_output.is_same_size(output)

        # Initialize grad input
        grad_points = points.new_zeros(*points.size())
        grad_masks = masks.new_zeros(*masks.size())
        grad_transforms = transforms.new_zeros(*transforms.size())
        # print("grad_masks",grad_masks)

        NTfm3D_cuda.backward(
            points.contiguous(),
            masks.contiguous(),
            transforms.contiguous(),
            output.contiguous(),
            grad_points,
            grad_masks,
            grad_transforms,
            grad_output.contiguous(),
            1,
        )
        # print("grad_points",grad_points)
        # print("grad_masks",grad_masks)
        # print("grad_transforms",grad_transforms)
        # self.use_mask_gradmag
        # outputs = NTfm3D_cuda.backward(grad_output.contiguous() *ctx.saved_variables)
        # d_old_h = outputs[0]
        # return d_old_h
        return grad_points, grad_masks, grad_transforms


class NTfm3D(nn.Module):
    def __init__(self):
        super(NTfm3D, self).__init__()

    def forward(self, points, masks, transforms):
        return NTfm3DFunction.apply(points, masks, transforms)
