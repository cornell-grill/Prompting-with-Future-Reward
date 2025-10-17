# Prompting with the Future: Open-World Model Predictive Control with Interactive Digital Twins
[Project Page](https://prompting-with-the-future.github.io/) 

<img  src="intro.gif" width="800">

Official implementation of Prompting with the Future (RSS 2025). We provide a demo on a scanned environment and also provide a pipeline for scanning your own environment for open-world manipulation.

## Installation

### Requirements for running the demo

```
git clone https://github.com/cornell-grill/Prompting-with-Future-Reward.git
cd prompting-with-the-future
conda env create --file environment.yaml
conda activate pwfr
pip install --upgrade mani_skill
conda install pytorch3d -c pytorch3d
```

Please download the pre-optimized gaussian splatting checkpoint and data from [here](https://drive.google.com/drive/folders/1HviE902s5g9fdUJHpfJ2HwhT4gy6kxi9?usp=sharing) and put them in the `gaussians/` folder.

Please add your ChatGPT API key in the `utils/prompt_gpt.py` script.

### Requirements for scanning your own environment

Please follow the instructions in [COLMAP](https://colmap.github.io/) to install the dependencies for reconstruction.

Since the environment for [SAM2](https://github.com/facebookresearch/sam2) is not compatible with the main environment and it is only used for reconstruction, we provide a separate environment for SAM2.

```
conda create -n sam2 python=3.10.0
conda activate sam2
cd sam2
pip install -e .
```

## Run our demo
We prepared a scanned environment for testing the Prompting with the Future.

``` 
python main.py --scene_name basket_world --instruction "put the green cucumber into the basket"
```

The resulting trajectory and joint actions will be saved in the `results` folder.

## Scan your own environment
We provide two ways to scan your environment for open-world manipulation.

### 1a. Scan with a phone (recommended)
Firstly, print a checkerboard (utils/reconstruct/checker_board.svg) and put it in the workspace.

Then, use a phone camera to flexibly scan your environment. (60fps is recommended)

Name your scene as `{SCENE_NAME}` and put the video in the `gaussians/colmap/{SCENE_NAME}` folder.

Run the following script to reconstruct the interactive digital twin with movable meshes and gaussians.

```
sh build.sh {SCENE_NAME} {INSTRUCTION}
```

### 1b. Scan with a robot
We also provide a script to use the robot to scan your environment.

Due to different robot platforms, we provide an example script on [Droid](https://droid-dataset.github.io/) setup for scanning with a robot.

You can adapt the `utils/reconstruct/robot_scan.py` script for your own robot.

### 2. Post-processing (optional)
We found that VLMs are quite robust to the artifacts in the reconstructed scene.
However, we still provide a post-processing step to improve the visual quality.

Change the `box` parameters in the `gaussians/gaussian_world.py` script to the bounding box of your workspace. This will remove the gaussians outside the bounding box.

The post-processing flag will also fill the holes under the objects.

### 3. Planning
Start the planning on your scanned environment by running the following command.

```
python main.py --scene_name {SCENE_NAME} --instruction {INSTRUCTION} --name {EXP_NAME}
```

## Citation
If you find our code or paper is useful, please consider citing:
```bibtex
@inproceedings{ning2025prompting,
  title={Prompting with the Future: Open-World Model Predictive Control with Interactive Digital Twins},
  author={Ning, Chuanruo and Fang, Kuan and Ma, Wei-Chiu},
  booktitle={RSS},
  year={2025}
}
```
