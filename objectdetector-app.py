import streamlit as st
import numpy as np
import argparse
import cv2
from PIL import Image

st.title('Object Detection Application')

CLASSES =["Background", "Airplane", "Bicycle", "Bird", "Boat",
	"Bottle", "Bus", "Car", "Cat", "Chair", "Cow", "Dining table",
	"Dog", "Horse", "Motorbike", "Person", "Potted Plant", "Sheep",
	"Sofa", "Train", "TV"] #names of objects that can be detected- using a PRE-trained model
COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3)) #generates the color of the text/boxes displayed on image


model = cv2.dnn.readNetFromCaffe('models\MobileNetSSD_deploy.prototxt.txt', 'models\MobileNetSSD_deploy.caffemodel') #loading the model from disk using OpenCV dnn
model_load_state = st.text('Loading models...')
model_load_state.text('Model Loaded!')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Using PIL
    disp_image = Image.open(uploaded_file)
    st.image(disp_image, caption='Uploaded Image.')
    image = np.array(disp_image)
    image = image[:, :, ::-1].copy() 
    (h,w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007843, (300,300), 127.5)
    model_load_state.text('Image Loaded!')
    model_state = st.text("*** Detecting objects...")
    model.setInput(blob)
    detections =  model.forward()
    #loads image and sets up input for the image- resizing and normalizing

    #sending the blob through the network
    objects = 0
    for i in np.arange(0, detections.shape[2]):

            probability= detections[0,0,i,2]


            if probability > 0.7:
                
                objects+=1

                nametag = int(detections[0,0,i,1])
                box = detections[0,0,i, 3:7] * np.array([w,h,w,h])
                (startX,startY, endX, endY) = box.astype("int")
                
                label = "{}:{:.2f}%".format(CLASSES[nametag], probability*100)
                print("--> {}".format(label))
                cv2.rectangle(image, (startX, startY), (endX ,endY), COLORS[nametag], 2)
                y= startY-15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[nametag], 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(image)
    img_caption = "Objects Detected:"+str(objects)
    st.image(output_image)
    model_state.text(img_caption)





