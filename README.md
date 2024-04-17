Code for the paper "Fast Implementation of the Node2Vec".

This repository provides  reference (from [snap-stanford/snap](https://github.com/snap-stanford/snap?ysclid=lu1c1ki2dc379755454)) and our optimized node2vec implementations. There are also scripts for creating Python package and providing Python API for both implementations.

## Prerequisites
* GCC compiler
* pybind11 - it's only needed if you want to create the Python package with node2vec.
Please install pybind11 by `pip install pybind11`.
## References
1. [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653).   Aditya Grover, Jure Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.

2. Stanford Network Analysis Platform (SNAP) [on GitHub](https://github.com/snap-stanford/snap), [node2vec project](https://snap.stanford.edu/node2vec/).
    
    `node2vec/src/snap/` contains node2vec dependencies from SNAP. The source code is published unchanged, except for the following files:
   * base.cpp
   * base.h
   * Snap.cpp
   * Snap.h

    The changes in these files are that we have removed those `#include`s in them that are not necessary for node2vec or added new ones. Similarly in the following files from `node2vec/src/n2v/`:
    * biasedrandomwalk.cpp
    * biasedrandomwalk.h
    * n2v.cpp
    * n2v.h
    
    In addition to the optimized version, `word2vec.h` and `word2vec.cpp` also contains a reference implementation from SNAP. Files `node2vec.cpp` and `node2vec.h` also contain code from `snap/examples/node2vec/node2vec.cpp`.
    

    The `node2vec/src/snap/Makefile` has been adapted for the purposes of our repository.



## Build instructions

### C++
* Run `make node2vec_optimized` to build optimized C++ node2vec.

* Run `make node2vec_reference` to build reference C++ node2vec.

Run examples for optimized and reference implementations by

`./node2vec_opt -i:example/graph/graph.edgelist -o:example/emb/result_opt.emb -l:5 -d:32 -p:4 -q:1 -v`

`./node2vec_ref -i:example/graph/graph.edgelist -o:example/emb/result_ref.emb -l:5 -d:32 -p:4 -q:1 -v`

The same node2vec parameters are supported as in [SNAP](https://github.com/snap-stanford/snap/tree/master/examples/node2vec).
### Python API
1. Use `make wheel` to create Python package for node2vec.
2. Install package by `pip install dist/n2v_ext-{platform}.whl`
3. Run example `python example/example.py`
