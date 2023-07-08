import numpy as np
import streamlit as st
from PIL import Image
from camera_input_live import camera_input_live
from streamlit_cropper import st_cropper
from streamlit_sortables import sort_items
from streamlit_toggle import st_toggle_switch

from cv_functions import *


def main_loop():

    st.title("OpenCV Pipeline Generator")
    st.subheader("This will allow you to generate OpenCV code for your FTC robot!")
    st.text("Created with Streamlit and OpenCV")
    st.sidebar.header("Stage Options")

    with st.sidebar:
        selected_stages = st.sidebar.multiselect(
            "Pipeline stages",
            ["Crop", "Blur", "Erode", "Dilate", "Threshold"],
            ["Crop", "Blur", "Erode", "Dilate", "Threshold"]
        )
        pipeline = sort_items([x for x in selected_stages if not x == "Crop"], header="Pipeline order")


    c1, c2 = st.columns([1, 7], gap="small")
    with c1:
        st_toggle_switch(
            label="Webcam input?",
            key="enable_webcam",
            default_value=False
        )
    image_file = None
    if st.session_state["enable_webcam"]:
        image_file = camera_input_live()
    else:
        with c2:
            image_file = st.file_uploader(label="Upload a test image", type=["jpg", "png", "jpeg"])
            if not image_file:
                return None

    input_image = Image.open(image_file)
    if "Crop" in selected_stages:
        st.subheader("Crop:")
        st.text("Crop result shown as \"Input\" below")
        input_image = st_cropper(input_image, realtime_update=True, box_color="#FF0000", aspect_ratio=None)

    input_image = np.array(input_image)
    display_images = [np.copy(input_image)]
    captions = ["Input"]

    for stage in pipeline:
        match stage:
            case "Blur":
                st.sidebar.slider("Blur", min_value=0.0, max_value=20.0, key="blur_amt")
                input_image = gaussian_blur(input_image, st.session_state["blur_amt"])
                display_images.append(np.copy(input_image))
                captions.append("Blur")

            case "Erode":
                st.sidebar.subheader("Erode")
                st.sidebar.slider("Iterations", min_value=1, max_value=50, key="erode_amt")
                input_image = erode(input_image, st.session_state["erode_amt"])
                display_images.append(np.copy(input_image))
                captions.append("Erode")

            case "Dilate":
                st.sidebar.subheader("Dilate")
                st.sidebar.slider("Iterations", min_value=1, max_value=50, key="dilate_amt")
                input_image = erode(input_image, st.session_state["dilate_amt"])
                display_images.append(np.copy(input_image))
                captions.append("Dilate")

            case "Threshold":
                st.sidebar.subheader("Threshold")
                st.sidebar.slider("Hue", 0, 180, (0, 180), key="hue")
                st.sidebar.slider("Saturation", 0, 255, (0, 255), key="sat")
                st.sidebar.slider("Value", 0, 255, (0, 255), key="val")

                hue = st.session_state["hue"]
                sat = st.session_state["sat"]
                val = st.session_state["val"]

                lower_bound = (hue[0], sat[0], val[0])
                upper_bound = (hue[1], sat[1], val[1])
                input_image = hsv_threshold(input_image, lower_bound, upper_bound)
                display_images.append(np.copy(input_image))
                captions.append("HSV Threshold")

    st.write("---")
    st.header("Pipeline")
    i = 0
    for col in st.columns(len(display_images), gap="small"):
        col.subheader(captions[i])
        col.image(display_images[i])
        i += 1

if __name__ == '__main__':
    main_loop()