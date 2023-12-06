import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
import keras.backend as K
import io

# Function for ela
def convert_to_ela_image(image, quality):
    temp_filename = 'temp_file.jpg'
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

# Function for filters
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

filters = filter1 + filter2 + filter3

image_size = (128, 128)

def prepare_image(image_path):
    image = Image.open(io.BytesIO(image_path))
    return np.array(convert_to_ela_image(image, 85).resize(image_size)).flatten() / 255.0

# IoU Score
def iou_score(y_true, y_pred, smooth=1e-5):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

# Load model
phase1 = load_model("phase1.h5")
phase2 = load_model('phase2.h5', custom_objects={'iou_score': iou_score})

def predict(content):
    pi = prepare_image(content)
    pi = pi.reshape(1, 128, 128, 3)
    prediction = phase1.predict(pi)

    return prediction[0]

def predictRegion(content):
    image = Image.open(io.BytesIO(content))
    image = np.array(image)

    img = cv2.resize(image, (512, 512))
    srm_img = cv2.filter2D(img, -1, filters)

    img = np.expand_dims(img, axis=0)
    srm_img = np.expand_dims(srm_img, axis=0)

    prediction = phase2.predict([img, srm_img])

    prediction = prediction.squeeze()

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i][j] > 0.75:
                prediction[i][j] = 1.0
            else:
                prediction[i][j] = 0.0

    return prediction

def highlight(img, mask):

    # Resize the original image using OpenCV
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Create an RGBA version of the original image
    original_array_rgba = np.concatenate([img, np.full(img.shape[:-1] + (1,), 255, dtype=np.uint8)], axis=-1)

    # Prepare a red semi-transparent overlay
    overlay = np.zeros(original_array_rgba.shape, dtype=np.uint8)
    red_color = [255, 0, 0, 128]  # Semi-transparent red

    # Apply the red overlay where the mask is black
    overlay[mask == 1] = red_color

    # Convert the numpy arrays back to PIL image for alpha compositing
    original_img_rgba = Image.fromarray(original_array_rgba)
    overlay_img = Image.fromarray(overlay)

    # Composite the images together
    highlighted_img = Image.alpha_composite(original_img_rgba, overlay_img)

    # Convert back to RGB to save in JPG format
    highlighted_img_rgb = highlighted_img.convert("RGB")

    return highlighted_img_rgb

st.title("Image Forgery Detection")
st.header("Upload an image ")

# To View Uploaded Image
image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

if image_file is not None:
    st.image(image_file)
    content = image_file.read()
    pred = predict(content)
    pred = pred[0]

    if pred >= 0.6:
        st.success(f"This is a real image\n\nProbability of input image to be real is {pred * 100:.2f} %")
    else:
        st.error(f"This is a potentially forged image\n\nProbability of input image to be fake is {(1 - pred) * 100:.2f} %")
        predi = predictRegion(content)
        img = cv2.imdecode(np.frombuffer(content, np.uint8), 1)
        highlight_img = highlight(img, predi)
        st.image(highlight_img)
