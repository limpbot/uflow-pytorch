import unittest

import torch

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
    def test_clone(self):

        x = torch.randn(1, requires_grad=True)
        print(x)
        # y = x.clone()

        # y.retain_grad()

        z1 = x.clone() ** 2
        z1.mean().backward()

        z2 = x ** 2
        z2.mean().backward()

        # print(y.grad)
        print(x.grad)
