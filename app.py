import time
import urllib
import numpy as np
import cv2 as cv
from PIL import Image
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img


# @st.cache(allow_output_mutation=True, suppress_st_warning=True)

html_temp = """
    <div style =  padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Handwritten Digit Recognizer</h1></center>
    </div>
    """

st.markdown(html_temp, unsafe_allow_html=True)

st.set_option("deprecation.showfileUploaderEncoding", False)
# st.markdown(html_temp, unsafe_allow_html=True)

opt = st.selectbox(
    "How do you want to upload the image for classification?\n",
    ("Please Select", "Upload image via link", "Upload image from device"),
)
if opt == "Upload image from device":
    file = st.file_uploader("Select", type=["jpg", "png", "jpeg"])
    st.set_option("deprecation.showfileUploaderEncoding", False)
    if file is not None:
        image = Image.open(file)

elif opt == "Upload image via link":

    try:
        img = st.text_input("Enter the Image Address")
        image1 = Image.open(urllib.request.urlopen(img))

    except:
        if st.button("Submit"):
            show = st.error("Please Enter a valid Image Address!")
            time.sleep(4)
            show.empty()

if image is not None:

    try:
        st.image(image, width=100, caption="Uploaded Image")

        if st.button("Classify"):
            # img_array = np.array(image.resize((28, 28), Image.ANTIALIAS))
            # img_array = np.array(img_array, dtype="float32")
            # img_array = np.array(img_array) / 255.0

            model_dir = "Model/model.h5"
            model = keras.models.load_model(model_dir)

            # prediction = model.predict(img_array)
            # result = np.argmax(prediction)
            # -------------------------------------------

            image1 = cv.imread("6.png")
            # shape if (352, 324, 3) for screen snap, this could be different based on read image.

            image1 = cv.resize(image1, (28, 28, 1), interpolation=cv.INTER_AREA)
            # now its in shape (28, 28, 3) which is~ 2352(28x28x3)

            image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)  # now gray image
            image1 = image.img_to_array(image1)  # shape (28, 28, 1) i.e channel 1
            image1 = image1.flatten()
            # flatten it as model is expecting (None,784) , this will be (784,) i.e 28x28x1 =

            image1 = np.expand_dims(image1, axis=0)  # will move it to (1,784)
            prediction = model.predict(image1)  # predict
            pred = np.argmax(prediction)

            # Displaying output
            st.info(f"The uploaded image has been classified as {pred}")
    except:
        st.error("Please try again")
