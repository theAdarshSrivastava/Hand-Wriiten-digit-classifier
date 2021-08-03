import time
import urllib
import numpy as np
import cv2 as cv
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img


html_temp = """
    <body>
    <div style ="padding-bottom: 20px; 
                 padding-top: 20px; 
                 padding-left: 5px; 
                 padding-right: 5px">
    <center><h1>Handwritten Digit Classifier</h1></center>
    </div>
    <style>
    body{
    background-image:url("https://cdn.pixabay.com/photo/2016/05/05/02/37/sunset-1373171__340.jpg");
    background-repeat: no-repeat;
  background-attachment: fixed;
  background-size: cover;
    }
    </style>
    </body>
    """

st.markdown(html_temp, unsafe_allow_html=True)

opt = st.selectbox("How do you want to upload the image for recognition?\n",
                  ( "Please Choose", "Upload image from device","Draw the Digit!"),)


if opt == "Upload image from device":
    file = st.file_uploader("Select", type=["jpg", "png", "jpeg"])
    st.set_option("deprecation.showfileUploaderEncoding", False)
    if file is not None:
        image = Image.open(file)

            
elif opt == 'Draw the Digit!':
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color ="#000"
    bg_color ="#eee"
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw",)
               )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#000",
    background_color="#eee",
    update_streamlit=realtime_update,
    height=200,
    width=300,
    drawing_mode=drawing_mode,
    key="canvas",
    )
    try:
        cv.imwrite("test.jpg",canvas_result.image_data)
        image=Image.open("test.jpg")
    except:
        pass
try:
    if image is not None:
        st.image(image, width = 300, caption = 'Uploaded Image')
        if st.button('Predict'):
            model = keras.models.load_model('/home/adarshsrivastava/Github/Hand-Wriiten-digit-classifier/Model/Model.h5')
            ""
            
            ""
            prediction = model.predict(image)
            st.success('Hey! The uploaded digit has been predicted as {}'.format(np.argmax(prediction)))

except:
    pass
