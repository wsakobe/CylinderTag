# CylinderTag: An Accurate and Flexible Marker for Cylinder-Shape Objects Pose Estimation Based on Projective Invariants

<div align="center"> <img src="pose.bmp" width = 480 height = 300 /> </div>

## Table of Contents

- [Background](#background)
- [Requirements](#Requirements)
- [Usage](#usage)
	- [Generator](#generator)
  - [Recognizer](#Recognizer)
- [License](#License)

## Background

We propose a novel visual marker called CylinderTag, which is specifically designed for curved surfaces. CylinderTag is a cyclic marker crafted to be firmly attached to objects with cylinder-like shapes. It is encoded by leveraging the manifold assumption for surfaces with cylindrical properties and through the utilization of cross-ratio in projection invariant.

## Requirements

1. OpenCV 4 (test on 4.5.3)
2. Ceres 2 (test on 2.0)

## Usage

We provide a basic CylinderTag (12c2f) for use, if you need other kinds of markers (such as 15c3f, 18c4f, etc.) need to be generated automatically by the user using the generator. For the recognizer, a marker encoding file (.marker) and a 3D model file (.model) are required for proper use, and a camera parameter file (cameraParams.yaml) is required for positioning.

### Generator

We provide a simple MATLAB-based marker generator where the user can modify the type of marker, the number of markers generated, and draw the markers from within the generator.

### Recognizer

We have defined a CylinderTag class in the recognizer, which contains the functions detect, estimatePose, drawAxis, etc. You can find its usage and definition in the source file.

## License

[MIT](LICENSE)

## Citation
If you utilize this code for any academic purposes, kindly reference the following work in your citations:

```bibtex
@article{wang2024cylindertag,
  title={CylinderTag: An Accurate and Flexible Marker for Cylinder-Shape Objects Pose Estimation Based on Projective Invariants},
  author={Wang, Shaoan and Zhu, Mingzhu and Hu, Yaoqing and Li, Dongyue and Yuan, Fusong and Yu, Junzhi},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2024},
  publisher={IEEE}
}
```
