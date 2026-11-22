# Machine Vision Using Particle Analysis for Robotic End-Effector Tracking — MSc Dissertation

Public engineering repository accompanying the MSc dissertation
(University of Sussex, module 876H1Z).

**What this is.** Engineering repository for an MSc project (University of
Sussex, module 876H1Z): tracking of surgical end-effectors on real
operating-room stereo recordings whose camera is re-mounted between
sessions. A particle filter carries the hand--eye calibration in its
state and re-estimates it online from stereo video, built on exact
forward kinematics. On the SurgPose corpus (34 recordings, 68
instrument trajectories) the full system reaches a 5.00 mm median 3D
error with every run below the 10 mm criterion, at 127 frames per
second; the task-level analysis segments 332 episodes into four phases.
The dissertation text itself is not part of this repository.

## Repository structure

```
_gen_figs2.py   Generates the analysis figures from the archived JSONs below (outputs are local-only)
results/        Archived result records — the source of every reported number
                (results_all34.json, phase_metrics.json, demo mp4s, run logs)
_mvp/           Earlier completed stage of the project (Apollo hand tracking);
               data roots via APOLLO_WORK / APOLLO_DATASET env vars
_tools/         Small utility scripts
```

## Regenerate the analysis figures

From the repository root:

```
python _gen_figs2.py
```

Reads `results/results_all34.json`, `results/phase_metrics.json` and
the two single-panel demo videos, and writes figures 5.8–5.12 into
`paper/figs/` and `paper_cn/figs/` (created if absent; not tracked).
Requires matplotlib, Pillow and ffmpeg. The demo videos are not tracked
(size); regenerate them with `pipeline/render_demo.py` (`DET=1 SPLIT=0`)
as described below.

## Datasets: where to download and how to use

| Dataset | Download | Used for |
|---|---|---|
| **SurgPose** — 34 da Vinci stereo recordings with joint states and ground-truth 3D keypoints | Zenodo: https://doi.org/10.5281/zenodo.15278516 (34 archives, 21.5 GB, CC BY 4.0) | all tracking experiments and the phase analysis |
| per-frame gripper angles | https://github.com/zijianwu1231/SurgPose (2025-10 supplement, `gripper_angle.yaml`) | grasp state of the phase rules |
| **EndoVis 2017** robotic instrument segmentation sub-challenge | official registration at https://endovis.grand-challenge.org/ (challenge report: arXiv:1902.06426); official 1800/1200-frame split | U-Net segmentation training and evaluation |

**Unpack layout.** Each SurgPose archive unpacks to
`<SURG_WORK>/surgpose_unpack/<six-digit-id>/` containing
`regular/left_video.mp4`, `regular/right_video.mp4`,
`api_jp_data.yaml`, `api_cp_data.yaml`, camera calibration and keypoint
annotations; the gripper yaml goes to
`<SURG_WORK>/_meta/gripper_angle.yaml`.

## Reproduce the experiments

The experiment scripts live in a separate working tree
(`pipeline/`: `render_demo.py`, `phase_metrics.py`, `track_filter.py`,
`track_online.py`, `train_detector.py`, …) with the mapping from each
script to its archived output documented in Appendix C of the paper.
Set two environment variables before running:

```
SURG_WORK=/path/to/surg_work          # unpacked data + pipeline scripts
DISSERTATION_ROOT=/path/to/this/repo  # where results/ is read and written
```

Geometry/rendering runs under Python 3.13; network training and
inference run under Python 3.11 with PyTorch (nothing is installed or
upgraded by the scripts). Example — regenerate the demo videos used by
figure 5.12:

```
cd $SURG_WORK/pipeline
DET=1 SPLIT=0 SEGS=000021:PSM1,000027:PSM1 \
CKPT=$DISSERTATION_ROOT/runtime/models/detector_14kp_20261110.pt \
OUT_OV=$DISSERTATION_ROOT/results/demo_single python render_demo.py
```

## Path policy

All scripts inside this repository locate files relative to their own
location — no absolute paths. The `pipeline/` scripts resolve their two
roots through `pipeline/_paths.py` (env-overridable, defaults resolve
to the historical layout); the archived `_mvp/` stage uses
`APOLLO_WORK` / `APOLLO_DATASET` the same way.

## Citation

If you use the datasets, cite SurgPose (Wu, Z., 2025, Zenodo,
DOI 10.5281/zenodo.15278516) and the EndoVis 2017 challenge report
(Allan et al., arXiv:1902.06426).
