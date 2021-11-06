import cv2
import os
import numpy as np
from keras.models import load_model, Model
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.image as mpimg
import pickle
import json
from absl import app
from absl import flags
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from scipy import spatial

classNames= []
classFile = 'coco.names'
configPath = '/Users/suminbae/PycharmProjects/tf_cv2/aeye/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/Users/suminbae/PycharmProjects/tf_cv2/aeye/frozen_inference_graph.pb'


with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


path = "/Users/suminbae/PycharmProjects/tf_cv2/aeye"
debug_path = "/Users/suminbae/PycharmProjects/tf_cv2/aeye/debug_frame/"
save_path="/Users/suminbae/PycharmProjects/tf_cv2/aeye/test_data/target_appear_frame"


vidcap = cv2.VideoCapture("people-detection.mp4")
vidcap.set(3,1280)
vidcap.set(4,720)
vidcap.set(10,70)

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(debug_path+"frame_"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

# sec = 0
# frameRate = 3 #//it will capture image in each 0.5 second
# count=1
# success = getFrame(sec)
#
# frame_data=[]
# appear_frame_list =[]
#
# while success:
#     count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = getFrame(sec)





def save_file_name(path):
    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
        # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpg'):
                # adds only the image files to the flowers list
                frame_data.append(file.name)


def get_model():
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def extract_features(img, model):
    # load the image as a 224x224 array
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img
    #reshaped_img = img.reshape(224,224,3)

    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    imgx = np.expand_dims(imgx, axis=0)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features



def detect_ssd(img,weightsPath,configPath):
    thres = 0.5
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    return classIds,confs,bbox


def target_matching(target_img):
    taget_feature = extract_features(target_img)



def cos_sim_sci(A, B):
    return 1 - spatial.distance.cosine(A, B)


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


if __name__=="__main__":

    sec = 0

    #input
    frameRate = 1  # //it will capture image in each 0.5 second
    count = 1
    success = getFrame(sec)

    frame_data = []
    appear_frame_list = []

    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

    save_file_name(debug_path)



    os.chdir(debug_path)

    model = get_model()
    target_img = cv2.imread(path+'/test7.jpg')
    #target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    cv2.imshow("detected frame", target_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    target_img = cv2.resize(target_img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    target_feature = extract_features(target_img, model)
    print(target_feature.shape)

    #sim 0.7이상의 매칭결과가 있는 프레임들 결과 담아주기
    #ui에 이 프레임들을 띄워주는 방식으로 진행하자
    #appear_frame_list =[]

    # 11/4까지 완ㄹㅅ 20:56
    for frame in frame_data:
        print(frame)
        # # if frame=="target.jpg":
        # #     continue
        # if frame!="image5.jpg":
        #     continue
        img = cv2.imread(frame)
        # os.chdir(path)

        classIds, confs, bbox = detect_ssd(img, weightsPath, configPath)
        try:

            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if classId == 1:
                    print(box)
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    roi = img[y:y + h, x:x + w]
                    print(roi.shape)
                    # cv2.imshow("roi",roi)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    roi = cv2.resize(roi, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                    p_feature = extract_features(roi, model)
                    print(p_feature.shape)

                    print(cos_sim_sci(p_feature, target_feature))

                    #threshold는 0.6이 최상인것같이 보인다
                    if cos_sim_sci(p_feature, target_feature) > 0.6:
                        appear_frame_list.append(frame)
                        break
        except:
            print("error")
            pass
    print(appear_frame_list)

    for f in appear_frame_list:
        frame_name = f
        f = cv2.imread(f,1)
        cv2.imshow("apper",f)
        cv2.imwrite(save_path+"/"+frame_name,f)
        cv2.waitKey(0)
        cv2.destroyAllWindows()







