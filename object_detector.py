import numpy as np
import argparse
import cv2

#argument parsing for command line
arg = argparse.ArgumentParser()
arg.add_argument("-i","--image", required=True) #input image
arg.add_argument("-p","--prototxt", required= True) #prototxt model file
arg.add_argument("-m","--model",required=True) #caffe model file
arg.add_argument("-r","--probability", type=float , default=0.2) #minimum probability of the detection
args = vars(arg.parse_args())


CLASSES =["Background", "Airplane", "Bicycle", "Bird", "Boat",
	"Bottle", "Bus", "Car", "Cat", "Chair", "Cow", "Dining table",
	"Dog", "Horse", "Motorbike", "Person", "Potted Plant", "Sheep",
	"Sofa", "Train", "TV"] #names of objects that can be detected- using a PRE-trained model
COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3)) #generates the color of the text/boxes displayed on image

print("***Loading model...")
mode = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])#loading the model from disk using OpenCV dnn

image = cv2.imread(args["image"])
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007843, (300,300), 127.5)
#loads image and sets up input for the image- resizing and normalizing


print("*** Detecting objects...")
mode.setInput(blob)
detections =  mode.forward()
#sending the blob through the network

for i in np.arange(0, detections.shape[2]):

        probability= detections[0,0,i,2]


        if probability > args["probability"]:

            nametag = int(detections[0,0,i,1])
            box = detections[0,0,i, 3:7] * np.array([w,h,w,h])
            (startX,startY, endX, endY) = box.astype("int")
             
            label = "{}:{:.2f}%".format(CLASSES[nametag], probability*100)
            print("--> {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX ,endY), COLORS[nametag], 2)
            y= startY-15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[nametag], 2)

cv2.imshow("Output", image)
cv2.waitKey(0)
