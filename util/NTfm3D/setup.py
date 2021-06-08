from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="NTfm3D_cuda",
    ext_modules=[
        CUDAExtension(
            "NTfm3D_cuda",
            [
                "NTfm3D_cuda.cpp",
                "NTfm3D_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
