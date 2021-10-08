# Trajectory

## Contents

* [Download the Project](#downloading-the-project)
* [Setting Up the project](#setting-up-the-project)
	* [Manual Setup](#manual-setup)
* [Running the pipeline](#running-the-pipeline)

## Downloading the project
---

Clone this repo using the following command 

```shell
git clone https://github.com/sleepingsaint/yolo_trajectory_prediction.git
```

## Setting up the project
---

* Run the download.py script and everything required will be downloaded and placed in corresponding directories

```shell
	python download.py
```

## Manual Setup 
---
* download objection detection weights from the following link and save it in detector/YOLOV3/weight/obj_detection.weights

	* https://drive.google.com/file/d/1C3Kqqu9gDXNNXr5WDpmhGTr-UaQ2ckqQ/view?usp=sharing 

* download weights for trajectory prediction of arm from the following link and save it in Trajectory/models/Individual/traj_arm.pth
	* https://drive.google.com/file/d/1VfRVvc-7EowI540S0_6FxJOMVg9XyXef/view?usp=sharing 

* download weights for trajectory prediction of end effector from the following link and save it in Trajectory/models/Individual/traj_endeffector.pth 
	* https://drive.google.com/file/d/1t_qok3BNNHN6EK_Uw3WiXrfGtLfQJpCs/view?usp=sharing 

* download weights for trajectory prediction of probe from the following link and save it in Trajectory/models/Individual/traj_probe.pth 
	* https://drive.google.com/file/d/1SEZGUvLB2gfVwA-yGTBlF5k3YDiOzYWj/view?usp=sharing 

* download weights for trajectory prediction of person from the following link and save it in Trajectory/models/Individual/00013.pth 
	* https://drive.google.com/file/d/1p8vo9rig9Q0i0WLVudQBgzkdjtJXcDiS/view?usp=sharing 

* Download the pretrained checkpoint file for deepsort and save it in deep_sort/deep/checkpoint/
	* https://drive.google.com/file/d/1MlXnCSjD5yOfxnnJMkruE0rCgLCMZJlB/view?usp=sharing

* download the testing video for running inference and save it in data\test_video.mp4
	* https://drive.google.com/file/d/13mVaJTsJ7rN-Bz5KtsS20dX1TZFAHLkU/view?usp=sharing


## Running the pipeline
---
Run the below command to run inference on a video file

```shell
python yolov3_deepsort.py -v <path to video file> -f <number of frames to run inference on>
```

example:

* Running for the first 100 frames

```python
python yolov3_deepsort.py -v data/test_video.mp4 -f 100
```

* Running for the entire video
```shell
python yolov3_deepsort.py -v data\test_video.mp4
```