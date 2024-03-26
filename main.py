import cv2
import numpy as np

video = cv2.VideoCapture(0)
classes = []
with open('coco.names','rt') as arquivo:
    classes = arquivo.read().rstrip('\n').split('\n')

modeloConf = 'yolov3-tiny.cfg'
modeloWeights = 'yolov3-tiny.weights'
confThresh = 0.2

net = cv2.dnn.readNetFromDarknet(modeloConf,modeloWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    check,img = video.read()
    img = cv2.resize(img,(1090,720))
    blob = cv2.dnn.blobFromImage(img,1/255,(320,320),[0,0,0],1,crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    imH, imW, imC = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThresh:
                w,h = int(det[2]*imW), int(det[3]*imH)
                x,y = int((det[0]*imW)-w/2), int((det[1]*imH)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThresh,0.3)

    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(37, 217, 100),2)
        cv2.putText(img,f'{classes[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(191, 133, 6),2)

    cv2.imshow('Video',img)
    cv2.waitKey(1)