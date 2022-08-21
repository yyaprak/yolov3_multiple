import cv2
import numpy as np


import glob
import random
net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

classes = ["cat","dog","lion"]
'''
with open("/home/iyilmaz/Desktop/nesne_takibi/downloaded_images/animals_images/classes.txt", "r") as f:
    classes = f.read().splitlines()
'''
'''
cap = cv2.VideoCapture('/mydrive/yolo/yolov3/mygeneratedvideo.avi')
'''
font = cv2.FONT_HERSHEY_PLAIN

colors = np.random.uniform(0, 255, size=(len(classes), 3))
images_path = glob.glob(r"/home/iyilmaz/Desktop/nesne_takibi/downloaded_images/animals_images/*.jpeg")

# Insert here the path of your images
random.shuffle(images_path)

count=0
#print(images_path)
for img_path in images_path:
    #print("i :",count)
    #count=count+1
#while True:
#    _, img = cap.read()
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #print("confidence ",confidence)
            if confidence > 0.3:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+0), font, 2, (255,255,255), 2)
            
            #resize image
            width = 640
            height = 480
            dim = (width, height)
                
            # resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #from google.colab.patches import cv2_imshow
        cv2.imshow("resim",resized)
        key = cv2.waitKey(1)
        if cv2.waitKey(0) == ord('q'):
            break


cv2.destroyAllWindows()
