# BHUNTER

**BHUNTER: Generating Blind-Spot Driving Scenarios for Robustness Testing of Autonomous Driving Perception**

ACM Transactions on Software Engineering and Methodology (TOSEM), 2026.

**Project page:** https://alexps9.github.io/BHUNTER/

BHUNTER searches for driving scenarios in which surrounding traffic creates a blind spot for the ego vehicle. It runs in the CARLA simulator, mutates scenario seeds toward a target occlusion level, and exports camera and LiDAR data in a KITTI-style layout for detection and tracking. A separate appearance-transfer model maps simulated camera frames closer to real imagery. This repository contains the fuzzer, the appearance-transfer code and checkpoint, and the occlusion benchmark released with the paper.

## Repository layout

```
BHUNTER/
├── Blindhunter/          # occlusion-guided scenario fuzzer
│   ├── exp_fuzzer.py     # search loop (guided, random, no scheduling)
│   ├── mutator/          # CARLA scenario mutation and KITTI export
│   ├── data/corpus/seeds # initial scenarios
│   └── run_fuzzer.sh
├── style_transfer/       # sim-to-real appearance transfer and checkpoint
├── evaluation/           # released benchmark and qualitative results
│   ├── benchmark/
│   └── evaluation_visualization/
├── requirements.txt
├── LICENSE               # MIT, for the BHUNTER code
└── NOTICE                # third-party components
```

## Environment

- Linux host that can run CARLA 0.9.15 (the fuzzer starts `CarlaUE4.sh`)
- Python 3.8
- A GPU is optional for the fuzzer and required for practical appearance-transfer training

CARLA itself is not included. Install the simulator, then install the matching Python API together with the other dependencies:

```bash
git clone --recurse-submodules https://github.com/alexps9/BHUNTER.git
cd BHUNTER
pip install -r requirements.txt
```

`Blindhunter/scenario_runner` is a submodule used for actor bookkeeping. If the clone did not populate it:

```bash
git submodule update --init Blindhunter/scenario_runner
```

Point-cloud downsampling in `Blindhunter/utils/ground_truth` links a small PCL library. Build it only if you call `downsample`:

```bash
cd Blindhunter/utils/ground_truth
mkdir -p build && cd build
cmake .. && cmake --build .
```

## Generate scenarios

Seeds live in `Blindhunter/data/corpus/seeds`. Each seed is a JSON scenario: map, weather, ego route, candidate actor, and the KITTI sensor rig (`datasets/KITTI/config/kitti_sensor.yaml`).

From `Blindhunter/`:

```bash
bash run_fuzzer.sh \
  --corpus data/corpus/seeds \
  --workdir output \
  --map Town02 \
  --strategy guided \
  --desired_occlusion 0.8 \
  --carla_path /path/to/CARLA_0.9.15/CarlaUE4.sh
```

`--strategy` is one of:

| Strategy | Behavior |
| --- | --- |
| `guided` | Keep mutating seeds whose occlusion moves toward the target |
| `random` | Sample seeds without the guided queue |
| `no_scheduling` | Replay queued scenarios with a weaker schedule |

`--desired_occlusion` is a float in `[0, 1]`. The search writes:

```
output/<strategy>/<desired_occlusion>/
├── queue/      # scenarios kept for further mutation
├── dataset/    # exported sensor frames
├── trace/      # scenarios that reached the target occlusion
├── info        # score log
└── snapshot    # resumable search state
```

The same entry point can be launched from the repository root:

```bash
python -m Blindhunter.exp_fuzzer --help
```

## Appearance transfer

Released weights are epoch 50 of the unpaired translation model, under `style_transfer/checkpoints/bhunter_sim2real/`. Setup, data layout, and the test command are in [`style_transfer/README.md`](style_transfer/README.md).

## Benchmark

`evaluation/benchmark/` is the KITTI-style dataset used in the paper. Both `detection/` and `tracking/` contain five occlusion levels:

- `No_Occlusion`
- `Low_Occlusion`
- `Moderate_Occlusion`
- `Severe_Occlusion`
- `Extreme_Occlusion`

Each level contains:

- `calib/`: calibration
- `image_2/`: RGB images
- `label_2/`: ground-truth annotations
- `planes/`: ground-plane estimates
- `velodyne/`: LiDAR point clouds
- `velodyne_depth/`: virtual points projected from depth

Tracking sequences also include `pose/` and `occlusion_win.txt`.

Qualitative perception outputs, before and after retraining, are in `evaluation/evaluation_visualization/`:

- Detection: `detection/<Occlusion_Level>/{before_retrain,after_retrain}/*_camera.png` and `*_lidar.jpg`
- Tracking: `tracking/<Occlusion_Level>/{before_retrain,after_retrain}/pred_camera_bbox.mp4` and `pred_lidar_bbox.mp4`

These files are for inspection. They do not retrain a detector or tracker.

## Citation

If you use BHUNTER, please cite the TOSEM 2026 paper. The v1.0.0 software archive is on Zenodo: [10.5281/zenodo.23072481](https://doi.org/10.5281/zenodo.23072481). The concept DOI [10.5281/zenodo.23072480](https://doi.org/10.5281/zenodo.23072480) always points at the latest version.

```bibtex
@article{bhunter2026tosem,
  author  = {Peng, Songyang and Dai, Jiarun and Lv, Yanghao and Luo, Jiaqi and Huang, Zongan and Zhang, Yuan and Yang, Min},
  title   = {BHUNTER: Generating Blind-Spot Driving Scenarios for Robustness Testing of Autonomous Driving Perception},
  journal = {ACM Transactions on Software Engineering and Methodology},
  year    = {2026}
}
```

```bibtex
@software{bhunter2026zenodo,
  author    = {Peng, Songyang and Dai, Jiarun and Lv, Yanghao and Luo, Jiaqi and Huang, Zongan and Zhang, Yuan and Yang, Min},
  title     = {BHUNTER: Generating Blind-Spot Driving Scenarios for Robustness Testing of Autonomous Driving Perception},
  month     = oct,
  year      = 2026,
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.23072481},
  url       = {https://doi.org/10.5281/zenodo.23072481}
}
```

## License

BHUNTER code is released under the [MIT License](LICENSE). Third-party code and the CARLA simulator keep their own terms; see [NOTICE](NOTICE).
