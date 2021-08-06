import numpy as np
import cv2 as cv
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://cdn.statically.io/img/i.pinimg.com/originals/2d/8e/c0/2d8ec08d2a34ac48fcad7824bf00d7d1.jpg");
        background-repeat: no-repeat;
        background-attachment: fixed;
  background-size: 100% 100%;
    }
""",
    unsafe_allow_html=True
)

html_temp = """
    <body>
    <div style ="padding-bottom: 20px; 
                 padding-top: 20px; 
                 padding-left: 5px; 
                 padding-right: 5px;
                 ">
    <center><h1 style="color:#D4C20B">Handwritten Digit Classifier</h1></center>
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
        <p style="color:white"><br><center>The handwritten digit recognition is the ability of computers to recognize human handwritten digits. It is a hard task for the machine because handwritten digits are not perfect and can be made with many different flavors. The handwritten digit recognition is the solution to this problem which uses the image of a digit and recognizes the digit present in the image.</center></p> 
    """

st.markdown(html_temp, unsafe_allow_html=True)

opt = st.selectbox("How do you want to upload the image for recognition?\n",
                  ( "Please Choose","Draw the Digit!"),)


            
if opt == 'Draw the Digit!':
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 8)
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
            model = keras.models.load_model('Model/model.h5')
            im=cv.imread("test.jpg")
            im=cv.cvtColor(im,cv.COLOR_BGR2GRAY)
            im=cv.resize(im,(28,28))
            im=np.array(im.reshape(28,28,1))
            im = np.invert(np.array([im]))
            im=im/255.
            print(im.shape)
            print(im)
            prediction = model.predict(im)
            st.success('Hey! The uploaded digit has been predicted as {}'.format(np.argmax(prediction)))
            st.balloons()

except:
    pass
