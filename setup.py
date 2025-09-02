from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_envshade",
    ext_modules=[
        CUDAExtension(
            name="_diff_envshade",
            sources=[
                "etc.cu",
                "bruteforce_shader.cu",
                "bindings.cpp"
            ]
         )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
