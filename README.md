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

### Regression tests

```bash
cmake -S . -B build/tests -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCYLINDERTAG_BUILD_TESTS=ON
cmake --build build/tests -j
ctest --test-dir build/tests --output-on-failure
```

These tests cover image-border thresholds, right-side projective decoding,
unmatched column codes, column-width averaging, and stale results after a failed detection.

### Experimental robust detector (opt-in)

This Draft implementation builds on the separate bug fixes in [PR #12](https://github.com/ShaoanWang/CylinderTag/pull/12).
The existing `detect()` implementation and its defaults are preserved. Select the
new path explicitly for experiments with the supplied **12c2f dictionary**:

```cpp
CylinderTag tag("CTag_2f12c.marker");
std::vector<MarkerInfo> markers;
tag.detectRobust(gray8, markers, 5, true, 5);
```

The path combines several adaptive threshold windows, contour quadrilateral
proposals, gradient-based edge line fitting, and continuous projective symbol
scores. Decoding compares every dictionary ID, cyclic phase, reading direction,
and vertical reflection. It requires at least four supported columns and rejects
ambiguous best/runner-up scores. Overlapping hypotheses are deduplicated.
`MarkerInfo` adds `mirrored`, `decodeError`, `decodeMargin`, and `supportColumns`.
Mirrored feature coordinates have **not** been validated for pose estimation.
Other dictionary configurations are not validated by this experimental decoder.

The proposal and refinement stages take inspiration from the
[OpenCV ArUco detection pipeline](https://docs.opencv.org/4.5.4/d5/dae/tutorial_aruco_detection.html)
and [AprilTag 2](https://april.eecs.umich.edu/papers/details.php?name=wang2016iros).
The CylinderTag cross-ratio geometry and dictionary are retained.

```bash
cmake -S . -B build/robust -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCYLINDERTAG_BUILD_TESTS=ON -DCYLINDERTAG_BUILD_ROBUST_TOOLS=ON
cmake --build build/robust -j
ctest --test-dir build/robust --output-on-failure
./build/robust/cylindertag_video input.mov CTag_2f12c.marker /tmp/detection 1 5 1 87
```

The optional video tool writes an overlay AVI, frame CSV, detection JSONL,
stage JSONL, and stage PNGs for the comma-separated snapshot frame indices
(zero-based; `87` above). Output parent directories must already exist. The CLI
reports ID 0 counts for the development fixture; the detector searches all IDs.
`captureDebugImages(true)`, `debugImages()`, `detectionStats()` and
`detectionCandidates()` expose intermediate results. `useMultiThreshold(false)`
selects only the requested threshold window. Six CTest entries cover the five
legacy regressions plus 984 exact decoder cases across IDs, phases, reading
directions, reflections and missing columns, with short/blank/tiny rejection checks.

#### Development evidence and limits

A single development clip contains 266 frames of a vertically mirrored printed
ID 0 marker. The mirror came from our STL export, so **all three runs use the
same complete 41-ID mirrored dictionary** (`c -> 63-c`), the unchanged video,
full resolution, threshold window 5, and refinement radius 5. No ROI, tracking,
or expected-ID restriction is used.

| Implementation | ID 0 detected frames | Frames containing another ID |
| --- | ---: | ---: |
| Upstream `8fb42c9` | 158/266 (59.40%) | 21 |
| Bug fixes in #12 | 159/266 (59.77%) | 19 |
| Opt-in robust path | 170/266 (63.91%) | 0 |

The gain against upstream is **4.51 percentage points**. An earlier 0.38%
baseline used the wrong dictionary orientation and is withdrawn. These are
frame-level observations from one development clip, not held-out accuracy or a
general false-positive-rate estimate. The private input video is not included.

A reproducible synthetic check renders all 41 IDs in canonical orientation,
four views each, plus 36 marker-free controls. Upstream detects the correct ID
in 6/164 positive frames, the robust path in 55/164; frames with a wrong ID fall
from 12 to 1. Both detect nothing in the 36 controls. The robust path detects
54/82 near-view positives but only **1/82 far-view positives**. Small markers,
real-world negative scenes, calibrated pose accuracy, runtime, and independent
camera/lighting conditions still need evaluation before changing the default.

```bash
python3 tests/generate_validation_video.py /tmp/cylindertag-validation --canonical-only
./build/robust/cylindertag_video /tmp/cylindertag-validation/validation.avi \
  CTag_2f12c.marker /tmp/cylindertag-validation/robust 1 5 1
python3 tests/evaluate_validation.py /tmp/cylindertag-validation/labels.json \
  /tmp/cylindertag-validation/robust.jsonl
```

Python tooling requires NumPy and OpenCV. The default renderer also includes
mirrored scenes; use `--canonical-only` for comparisons with a detector that
does not support reflections. The four-column minimum can reject views accepted
by the original detector, and the extra reflection hypotheses increase the
search space. Score thresholds are experimental, not calibrated probabilities.

#### Additional real-video check: regression on the upstream sample

The unchanged upstream `test.avi` (1920×1200, 259 frames) was evaluated after
submitting this Draft, without further tuning. All methods use the same full
canonical 41-ID dictionary, scale 1, threshold window 5, refinement radius 5,
and no ROI or tracking; the robust path keeps its default multiple windows.

| Implementation | Frames with detections | Output IDs |
| --- | ---: | --- |
| Upstream `8fb42c9` | 259/259 (100.00%) | 5 only |
| Bug fixes #12 | 259/259 (100.00%) | 5 only |
| Opt-in robust path | 245/259 (94.59%) | 5 only |

**The robust path regresses by 14 frames (5.41 percentage points).** No ID/corner
ground-truth annotations are provided with this sample; these are detection
availability counts, not independently verified recognition accuracy. Agreement
on ID 5 is not an independent ground-truth label. In five missed frames the
selected threshold's largest group contains three columns, below the decoder's
four-column minimum; other rejection causes have not yet been isolated. This
is further evidence for keeping the path opt-in and this PR a Draft.

`tests/sample_video_results.json` records hashes, implementation commits, counts
and missing-frame indices. The sample's SHA-256 matches upstream `8fb42c9`.
To rerun with headless executables built against the three revisions:

```bash
python3 tests/compare_sample_videos.py test.avi \
  --original /path/to/original/fair_video \
  --bugfix /path/to/bugfix/fair_video \
  --improved build/robust/cylindertag_video --output /tmp/sample-comparison
```

The standard-library-only comparison script supports multiple canonical input
videos, saves exact commands and SHA-256 hashes, and checks sequential frame
indices and equal frame counts. `--summarize-existing` reads prior outputs.
The legacy headless driver uses the same scale/window/refinement settings;
the GUI demo must not be substituted with different defaults. No runtime
comparison is claimed by these runs.

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
