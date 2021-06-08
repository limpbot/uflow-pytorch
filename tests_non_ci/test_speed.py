import unittest

from util import helpers
import torch
import tensorflow as tf

from usflow import options

parser = options.setup_comon_options()

preliminary_args = [
    "-s",
    "../config/config_setup_0.yaml",
    "-c",
    "../config/config_coach_uflow_0.yaml",
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

    def test_neighbors_to_channels(self):
        import time

        B = 10
        C = 3
        H = 340
        W = 340
        # torch.set_flush_denormal(True)
        x = torch.rand(size=(B, C, H, W), device="cuda:0")
        dtype = x.dtype
        device = x.device
        patch_size = 9

        time_start = time.time()
        cv3 = helpers.neighbors_to_channels_v3(x=x, patch_size=9)
        cv1 = helpers.neighbors_to_channels(x=x, patch_size=9)

        print("max diff cv3 cv1", torch.max(torch.abs(cv3 - cv1)))
        time_end = time.time()
        print("end", time_end - time_start)

    def test_compute_cost_volume(self):
        import time

        B = 10
        C = 3
        H = 340
        W = 340

        x1 = torch.rand(size=(B, C, H, W), device="cuda:0")
        x2 = torch.rand(size=(B, C, H, W), device="cuda:0")
        time_start = time.time()
        cv1 = helpers.compute_cost_volume(x1=x1, x2=x2, max_displacement=4)
        time_end = time.time()
        print("v1", time_end - time_start)

        time_start = time.time()
        cv2 = helpers.compute_cost_volume_v2(x1=x1, x2=x2, max_displacement=4)
        time_end = time.time()

        print("v2", time_end - time_start)

        print("diff", torch.max(torch.abs(cv1 - cv2)))
        pass


if __name__ == "__main__":
    unittest.main()
