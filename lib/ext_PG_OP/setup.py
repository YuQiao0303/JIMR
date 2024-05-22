from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ext_PG_OP",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "ext_PG_OP",
            ["pytorch/ext_ops.cpp", "kernel/ext_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)