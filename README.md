# Aeye
CCTV 인물 자동 탐지 어플리케이션





## Getting started

#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate tracker-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate tracker-gpu
```


##Getting Started

1. yolo_on_video.py
 - 동영사 내 등장인물에 대하여 yolo르 이용해서 detection을 진행 
 - detection 결과들을 crop해서 persons 디렉토리에 데이터 저장
 - you need yolov3.cfg , yolov3.weights , coco.names in your directory
 
How to execute yolo_on_video.py 

'''bash
python yolo_on_video.py --video=1554.mp4 --frame_rate=30
'''
- --video="your target video name"
- --frame_rate=number of frame users want to cut 

