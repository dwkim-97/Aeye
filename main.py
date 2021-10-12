import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pprint import pprint

classFile='coco.names'
classnames=[]

whT=320
confidenceThreshold =0.5
nmsThreshold = 0.3
#lower it is acc



modelconfig = 'yolov3.cfg'
modelweights = 'yolov3.weights'

#yolo model
net = cv2.dnn.readNetFromDarknet(modelconfig,modelweights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


with open(classFile,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
print(classnames)
print(len(classnames))



def findobject(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds= []
    confs=[]

    for output in outputs:
        for det in output:
            scores =det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence>confidenceThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y =  int((det[0]*wT) - w/2) , int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    print(len(bbox))
    indicies = cv2.dnn.NMSBoxes(bbox,confs,confidenceThreshold,nmsThreshold)

    for i in indicies:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)



        #cv2.putText(img,f'{classnames[classIds[i]].upper()} {int(confs[i]*100)}%',
        #(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),1)

    print(x,y,x+w,y+h)


def yolo_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        success, img = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 200)

        print(success)
        print(img.shape)

        # input to networ
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        # extract only output layer

        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # print(net.getUnconnectedOutLayers())
        # print(outputNames)

        outputs = net.forward(outputNames)
        # print(outputs[0].shape)
        # print(outputs[1].shape)
        # print(outputs[2].shape)
        # print(outputs[0][0])


        findobject(outputs, img)

        #확인차 보여줄 때는 imshwo하셈
        #cv2.imshow("image", img)
        cv2.imwrite("video/" + f"test{count}.jpg", img)
        #cv2.waitKey(1)

        # 임시방편용
        if count == 30:
            break

        count += 1


if __name__=="__main__":
    cap = cv2.VideoCapture('1554.mp4')
    count = 0

    while True:
        success, img = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 200)

        print(success)
        print(img.shape)

        # input to networ
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        # extract only output layer

        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # print(net.getUnconnectedOutLayers())
        # print(outputNames)

        outputs = net.forward(outputNames)
        # print(outputs[0].shape)
        # print(outputs[1].shape)
        # print(outputs[2].shape)
        # print(outputs[0][0])

        findobject(outputs, img)

        cv2.imshow("image", img)
        cv2.imwrite("video/" + f"test{count}.jpg", img)
        cv2.waitKey(1)

        # 임시방편용
        if count == 30:
            break

        count += 1






