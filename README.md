# Aeye
CCTV 인물 자동 탐지 어플리케이션


## Getting started

1. yolo_on_video.py
 - 동영사 내 등장인물에 대하여 yolo르 이용해서 detection을 진행 
 - detection 결과들을 crop해서 persons 디렉토리에 데이터 저장
 - you need yolov3.cfg , yolov3.weights , coco.names in your directory


```bash
# Tensorflow CPU
python yolo_on_video.py --video=1554.mp4 --frame_rate=30
```
- --video="your target video name"
- --frame_rate=number of frame users want to cut 


#### yolo_on_video Result
<img src='yolo_on_video_example.jpg' />

2. image_clustering.py
 - 동영사 내 등장인물에 대하여 Clustering으 진행
 - Clustering 결과르 json 파일로 저장

```bash
python image_clustering.py
```



3. Target_matching.py
 - 입력 타겟에 가장 유사한 클러스터 인물 비교
 - Pops up and returns the closest persons by comparing cosine similarity
 - required parameter: target_image: person you want to find in this video

```bash
python3 Target_matching.py --target_image=current_path/target_image.jpg
```


