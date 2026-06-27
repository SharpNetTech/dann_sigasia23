<div align="center">
<h1>SharpNet: Enhancing MLPs to Represent Functions with Controlled Non&#8288;-&#8288;differentiability</h1>

[ACM DL](https://doi.org/10.1145/3811330) | [Homepage](https://sharpnettech.github.io) | [SharpNet2D](https://github.com/SharpNetTech/SharpNet2D) | [SharpNet3D](https://github.com/SharpNetTech/SharpNet3D)

ACM Transactions on Graphics (SIGGRAPH 2026)

<p><span><b>Hanting Niu</b><sup>1,2,*</sup></span> · <span><b>Junkai Deng</b><sup>3,*</sup></span> · <span><b>Fei Hou</b><sup>1,2</sup></span> · <span><b>Wencheng Wang</b><sup>1,2</sup></span> · <span><b>Ying He</b><sup>3</sup></span></p>
<p><sup>1</sup> Institute of Software, Chinese Academy of Sciences<br>
<sup>2</sup> University of Chinese Academy of Sciences<br>
<sup>3</sup> Nanyang Technological University</p>
<p><sup>*</sup> Equal contributions</p>
</div>

## Discontinuity-Aware Neural Network ##

This is a supplemental code release that modifies the official code release of paper "Discontinuity-Aware 2D Neural Fields" by Belhe et al. (SIGGRAPH Asia 2023). The modified code is designed to run an experiment introduced in Section 4.3 of our paper.

## What does this repo do? ##
This repo should be able to reproduce the following experiments:
| | Geodesic<br>(Section 4.1) | Medial axis<br>(Section 4.2) | Belhe<br>(Section 4.3) |
|--|:--:|:----:|:----:|
| Raw MLP | ✗ | - | ✗ |
| InstantNGP | ✗ | - | ✗ |
| SharpNet w/ ReLU | ✗ | - | ✗ |
| SharpNet w/ Softplus (Ours) | ✗ | ✗ | ✗ |
| Belhe et al | - | - | ✓ |
| Liu et al | - | - | ✗ |

Note: The experiment in Section 4.3 is conveniently named "Belhe" because the feature edges are taken directly from Belhe et al. It should not be confused with the actual method.

## The original readme is retained and is also a must-read ##
You can find the original README file at [README_orig.md](README_orig.md). The original readme file contains information on how to setup the environment, project, and how to run the code.

> [!IMPORTANT]
> The original readme contains obsolete information.
>
> One of the dependencies in the original readme, `slangpy`, has since been renamed to `slangtorch`.
> 
> Please install `slangtorch`.

> [!IMPORTANT]
> `diffvg` will not build with Python >= 3.11

### If you have difficulties with setup ###
We provide a Dockerfile that can build a working environment. The Dockerfile is not optimized for image space but at least it works.

```bash
docker build -t dann:latest .
docker run --rm -it --gpus=all --ipc=host -v /path/to/code:/workspace dann:latest bash
```

## Our modifications to the original code ##
We did the following modifications:
* Renamed the original readme and written this readme;
* Written a `Dockerfile`, a `requirements.txt` and a `constraints.txt`;
* Import `slangtorch` as `slangpy`;
* Modified the definition of MLP from ReLU network to the Softplus network used in our experiments;
* Added Belhe experiment training code.

## After you set up the project ##
The "Belhe experiment" gets its name from this repository. We reuse their preprocessed dataset of `wos/overview`.

Out of the ~4GB downloaded dataset (~8GB after expansion), you will only need:
* File `wos/overview/img.msh_curved.npz` (12KB).

We assume the said file is located at `./data/wos/overview/img.msh_curved.npz`; this location is hard-coded in our modified code.

## Running the code ##
Our training code for the Belhe experiment can be run directly. That is,
```bash
python train_belhe.py
```

Results will be stored to `./results/belhe`.

## A deeper dive into DANN dataset format ##
We tried our best to reverse engineer the dataset format, especially the aforementioned `.npz` file.

The `Sampler` class in DANN is functionally equivalent to a `Dataset` class. The `msh_curved.npz` file defines the mesh segmentation of the space. The `.npz` file contains the following fields:

* `vertices` is a `[V, 2]` array of type `float32`. This is the locations of each vertex. It is worth noting that
  * The coordinates are normalized to range [0.0, 1.0]
  * The coordinates are given in [Y, X] order
  * The origin is located at the upper left corner, X axis goes right and Y axis goes down
* `linear_triangles` is an `[L, 3]` array of type `int32`. The integers reference the index of the vertices array. The edge connecting the first point and the second point in this list seems to be the discontinuous edge.
* `curved_triangles` is not quite understood. It is possible to indicate curved edges.
* `continuous_triangles` is a `[C, 3]` array of type `int32`. The integers reference the index of the vertices array. These triangles are continuous, they don't contain any edge that are part of the discontinuous feature.

In our Belhe experiment, we load the preprocessed `.npz` file for its mesh structure, then immediately merge the linear triangles into continuous triangles because our field to learn is continuous everywhere.

## Citation ##
If you find our work useful, please cite SharpNet.
```bibtex
@article{niu2026sharpnet,
    author = {Niu, Hanting and Deng, Junkai and Hou, Fei and Wang, Wencheng and He, Ying},
    title = {{SharpNet}: Enhancing {MLP}s to Represent Functions with Controlled Non-differentiability},
    year = {2026},
    issue_date = {July 2026},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {45},
    number = {4},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3811330},
    doi = {10.1145/3811330},
    journal = {ACM Transactions on Graphics},
    month = jul,
    articleno = {113},
    numpages = {19},
    keywords = {MLP, Sharp features, Poisson's equation, Jump Neumann boundary condition, Green's function, CAD},
}
```

## Acknowledgments ##
We thank the authors of DANN for their code. The original DANN codebase can be found at [yashbelhe/dann_sigasia23](https://github.com/yashbelhe/dann_sigasia23).

Furthermore, we thank the authors for their swift email response to our questions about the environment setup.