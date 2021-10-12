import os
import cv2
import numpy as np

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

    #outputs는 3개 (300,85) , (1200,85), (4---,85)
    for output in outputs:
        for det in output:
            # 5부터는 score
            scores =det[5:]
            classId = np.argmax(scores)

            ##############################################################
            # 사람만 detect 하고 싶으면 아래 classId 0 아닐때 아예
            # 루프 안돌게 하면 된다
            #                                                            #
            #                                                            #
            #                                                            #
            #                                                            #
            #                                                            #
            ##############################################################

            if classId!=0:
                continue

            confidence = scores[classId]

            if confidence>confidenceThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y =  int((det[0]*wT) - w/2) , int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indicies = cv2.dnn.NMSBoxes(bbox,confs,confidenceThreshold,nmsThreshold)

    for i in indicies:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        roi = img[y:y + h, x:x + w]
        #cv2.imwrite(f"{c}"+"roi.jpg", roi)

        #cv2.putText(img,f'{classnames[classIds[i]].upper()} {int(confs[i]*100)}%',
        #(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),1)

    return bbox , classIds


def yolo_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while True:
        success, img = cap.read()
        print("frame" + f"{frame_number}" + " handling")

        # 조절할려면 여기
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 200)

        # input to network
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

        # findobject(outputs, img)
        box_coord, classIds = findobject(outputs, img)

        # frame saving
        print("detecting persons")
        # cv2.imshow("image", img)
        cv2.imwrite("frame_data/" + f"test{frame_number}.jpg", img)
        # cv2.waitKey(1)
        print("detecting is over")




        print("croppin images ")
        crop_person(img,box_coord,frame_number)
        print("cropping over")

        # 임시방편용
        if frame_number == 5:
            break

        frame_number += 1

        ##############################################################
        # 여기는 프레임률 조절 아직 방법 못찾아서 count돌다가 30개 이상이면 멈추게
        #
        #                                                            #
        #                                                            #
        #                                                            #
        #                                                            #
        #                                                            #
        ##############################################################



def crop_person(img,box_coord,frame_number):
    person_number=1

    try:
        for bbox in box_coord:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            roi = img[y:y + h, x:x + w]
            cv2.imwrite("persons/" + "frame_" + f"{frame_number}" + "_" + "person_" + f"{person_number}" + ".jpg", roi)
            # cv2.imwrite("persons/" +"frame_"+f"{person_number}" + "_" + f"{frame_number}.jpg", roi)
            person_number += 1

    except Exception as e:
        print(e)
        print("in the frame"+f"{frame_number}"+"this person has error"+f"{person_number}")
        pass





if __name__=="__main__":

    #10/12 19:11 작동완료 확인
    #10/12 crop function 까지 만들어서 함수화
    #10/12 19:30 완성


    yolo_on_video("1554.mp4")















