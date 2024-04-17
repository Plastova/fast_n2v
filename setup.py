import pybind11
from setuptools import setup, Extension

sources = [
    "src/n2v/node2vec_pybind.cpp",
    "src/n2v/n2v.cpp",
    "src/n2v/biasedrandomwalk.cpp",
    "src/n2v/word2vec.cpp",
]
include_dirs = [pybind11.get_include(), "src/snap"]
extra_objects = ["src/snap/Snap.o"]
ela = ["-lgomp"]
eca = ["-std=c++17", "-O3", "-DNDEBUG", "-fopenmp"]

ext_modules = [
    Extension(
        "n2vcpp_opt",
        sources=[*sources, "src/n2v/batch_rnd.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_objects=extra_objects,
        extra_link_args=ela,
        extra_compile_args=[*eca, "-DOPTIMIZED"],
    ),
    Extension(
        "n2vcpp_ref",
        sources=sources,
        include_dirs=include_dirs,
        language="c++",
        extra_objects=extra_objects,
        extra_link_args=ela,
        extra_compile_args=eca,
    ),
]

setup(
    name="n2v_ext",
    packages=["n2v_ext"],
    version="0.0.1",
    ext_modules=ext_modules,
    requires=["pybind11"],
)
