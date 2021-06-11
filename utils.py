from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import os

def make_group(filename, kmeans):
    groups = {}
    for file, cluster in zip(filename ,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    
    return groups

def feature_vectors(img_dict, model):
    f_vect = {}
    print('Feature extracting...')
    for fn, img in img_dict.items():
        if img.shape[2] == 1:
            img = img.repeat(3, axis=2)
        arr4d = np.expand_dims(img, axis=0) # (1, 224, 224, 3)
        arr4d_pp = preprocess_input(arr4d)
        f_vect[fn] = model.predict(arr4d_pp)[0, :]
    return f_vect

def get_images(path, filenames, target_size=(299, 299)):
    images = []
    for filename in filenames:
        img = Image.open(os.path.join(path, 'persons', filename))
        resized_img = img.resize(target_size)
        images.append([filename, np.array(resized_img)])

    return dict(images)

def get_person_coord(path):
    f = open(os.path.join(path, 'darknet/result.txt'), "r")
    lines = f.readlines()
    frame_idx = 0
    boxes = []
    all_boxes = {}

    for line in lines:
        if "Enter Image Path" in line:
            if len(boxes) != 0:
                all_boxes[img_name] = boxes
            boxes = []
            img_name = line.split(":")[1].split('/')[-1]
            frame_idx += 1
        elif "person" in line:
            coordinates = line.split(":")[2].split(" ")
            if not coordinates[2].isdigit():
                continue
            x_min = int(coordinates[2])
            y_min = int(coordinates[4])
            x_max = int(coordinates[6])
            y_max = int(coordinates[8])

            boxes.append((x_min, y_min, x_max, y_max))
            bbox = (x_min,y_min,x_max,y_max)
    return all_boxes

def crop_person_image(data_dict, path):
    all_frame = list(data_dict.keys())
    j=1
    for frame_idx, file_name in enumerate(all_frame):
        im = Image.open(os.path.join(path, 'frame', file_name))
        for idx in range(len(data_dict[file_name])):
            bbox = data_dict[file_name][idx]
            crop = im.crop(bbox)
            if not os.path.isdir(os.path.join(path, 'persons')):
                os.makedirs(os.path.join(path, 'persons'))
            crop.save(os.path.join(path, 'persons', f'person_{frame_idx}_{j}.jpg'))
            j += 1
    print('# Successful Save')

def cut_frames(videos_path):
    path_name = videos_path
    vidcap = cv2.VideoCapture(path_name)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("video/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def get_frame_from_video(path, video, frame_rate=0.5):
    if not os.path.isdir(os.path.join(path, 'frame')):
        os.makedirs(os.path.join(path, 'frame'))
    vidcap = cv2.VideoCapture(os.path.join(path, video))
    sec, count = 0, 1
    frameRate = frame_rate # capture image in each 0.5 second
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(os.path.join(path, 'frame', f'image{count}.jpg'), image)
        return hasFrames
    success = getFrame(sec)
    print("framming.....")
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
    print("# Successful framming")


def write_train_list(path):
    frame_path = os.path.join(path, 'frame')
    frame_list = os.listdir(frame_path)
    f = open(os.path.join(path,'darknet/data/train.txt'), 'w')
    for frame_name in frame_list:
        f.write(frame_path + '/' + frame_name + '\n')
    f.close()
    print("# Successful Writing")

def excute_YOLO():
    os.system('./darknet/darknet detector test \
               ./darknet/cfg/coco.data \
               ./darknet/cfg/yolov3.cfg \
               ./darknet/yolov3.weights \
               < ./darknet/data/train.txt > ./darknet/result.txt')