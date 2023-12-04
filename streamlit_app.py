import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance

model1 = load_model("phase1.h5")
model2 = load_model("phase2.h5")

st.title('Image Forgery Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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

import keras.backend as K

def iou_score(y_true, y_pred, smooth=1e-5):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


# Preprocess Image
def convert_to_ela_image(image, quality):
    temp_filename = 'temp_file.jpg'
    
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

image_size = (128, 128)

def prepare_image(image):
    img = image.convert('RGB')
    return np.array(convert_to_ela_image(image, 85).resize(image_size)).flatten() / 255.0

# Predict
def predict(img):
    img1 = prepare_image(img)
    img1=img1.reshape(1,128,128,3)
    prediction = model1.predict(img1)
    if prediction[0]>0.5 :
        pred = 0
    else:
        pred = 1
        img=cv2.resize(img,(512,512))
        preprocess_img=cv2.filter2D(img,-1,filters)
        pred=model2.predict([img,preprocess_img])[0]
        pred=pred.reshape(512,512)
    if pred == 0:
        for i in range(pred.shape[0]) :
            for j in range(pred.shape[1]) :
                if pred[i][j]>0.75 :
                    pred[i][j]=1.0
                else :
                    pred[i][j]=0.0
        st.image(pred, caption="potentially Forged Image", use_column_width=True)
        st.error("This Image is potentially forged")
        st.error(f'There is {prediction[0][0]*100:.2f} % chance of image being manupulated')
    else:
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.success("This Image is Authentic")
        st.success(f'There is {prediction[0][1]*100:.2f} % chance that the image is authentic one')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    trigger = st.button('Predict', on_click=lambda: predict(image))