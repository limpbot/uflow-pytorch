import torch
import numpy as np


def save_torch_as_nptxt(torch_out, fpath):
    np_out = torch_out.numpy()
    np.savetxt(fpath, np_out)


def read_nptxt_as_torch(fpath):
    np_in = np.loadtxt(fpath)
    torch_in = torch.from_numpy(np_in)
    return torch_in
