from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
            "ssim.cu",
            "ssim_bfloat.cu",
            "ssim_half.cu",
            "ext.cpp"])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
