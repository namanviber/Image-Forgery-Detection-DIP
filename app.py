import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from PIL import Image,ImageChops,ImageEnhance
import cv2
import keras.backend as K

#Function for ela 
def convert_to_ela_image(image, quality):
    temp_filename = 'temp_file.jpg'
    ela_filename = 'temp_ela_file.png'
    
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

#Function for filters 
q = [4.0, 12.0, 2.0]
filter1 = [[0, 0, 0, 0, 0],
           [0, -1, 2, -1, 0],
           [0, 2, -4, 2, 0],
           [0, -1, 2, -1, 0],
           [0, 0, 0, 0, 0]]
filter2 = [[-1, 2, -2, 2, -1],
           [2, -6, 8, -6, 2],
           [-2, 8, -12, 8, -2],
           [2, -6, 8, -6, 2],
           [-1, 2, -2, 2, -1]]
filter3 = [[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 1, -2, 1, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]]


filter1 = np.asarray(filter1, dtype=float) / q[0]
filter2 = np.asarray(filter2, dtype=float) / q[1]
filter3 = np.asarray(filter3, dtype=float) / q[2]
    
filters = filter1+filter2+filter3



image_size = (128, 128)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 85).resize(image_size)).flatten() / 255.0

#IoU Score
def iou_score(y_true, y_pred, smooth=1e-5):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
    
#Load model 
phase1 = load_model("phase1.h5")
phase2 = load_model('phase2.h5', custom_objects={'iou_score': iou_score})

def predict(img_path,model) :
    pi=prepare_image(img_path)
    pi=pi.reshape(1,128,128,3)
    predict=phase1.predict(pi)

    return predict[0]

def predictRegion(img):
    img = cv2.resize(img, (512, 512))
    srm_img = cv2.filter2D(img, -1, filters)
    
    img = np.expand_dims(img, axis=0)
    srm_img = np.expand_dims(srm_img, axis=0)

    prediction = phase2.predict([img, srm_img])

    prediction = prediction.squeeze()

    for i in range(prediction.shape[0]) :
        for j in range(prediction.shape[1]) :
            if prediction[i][j]>0.75 :
                prediction[i][j]=1.0
            else :
                prediction[i][j]=0.0

    return prediction

 
st.title("Image Forgery Detection")
st.header("Upload a image ")

# To View Uploaded Image
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

# You don't have handy image 
st.image(image_file)
ela_img = convert_to_ela_image(image_file)
pred=predict(image_file)
st.text("ELA image for this image")
st.image(ela_img)
pred=pred[0]
st.markdown("Probability of input image to be real is " + str(pred[0]))
st.markdown("Probability of input image to be fake is " + str(1-pred[0]))

if pred >= 0.5 :
    st.title("This is a pristine image")
else :
    st.title("This is a potentially forged image")
    predi=predictRegion(image_file)
    st.image(predi)
    st.write("##### NOTE : Black region is part of image where original image may be tempered Please have a close look on these regions . Thankyou ❤️") 
        