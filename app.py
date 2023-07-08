import numpy as np
import cv2 as cv
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
from streamlit_sortables import sort_items
from streamlit_toggle import st_toggle_switch

from cv_functions import *


def main_loop():

    st.set_page_config(layout="wide")

    st.title("OpenCV Pipeline Generator")
    st.subheader("This will allow you to generate OpenCV code for your FTC robot!")
    st.text("Created with Streamlit and OpenCV")

    with st.sidebar:
        st.header("Pipeline Options")
        selected_stages = st.sidebar.multiselect(
            "Pipeline stages",
            ["Crop", "Blur", "Erode", "Dilate", "Threshold", "Contours"],
            ["Threshold", "Erode", "Dilate", "Contours"]
        )
        pipeline = sort_items([x for x in selected_stages if not x == "Crop"], header="Pipeline order")
        if "Contours" in pipeline and ("Threshold" not in pipeline or pipeline.index("Contours") < pipeline.index("Threshold")):
            st.error("Contours stage must come AFTER Threshold stage")
            return None

    c1, c2 = st.columns([1, 7], gap="small")
    with c1:
        enable_webcam = st_toggle_switch(
            label="Webcam input?",
            key="enable_webcam",
            default_value=False
        )
    image_file = None
    with c2:
        if enable_webcam:
            image_file = st.camera_input("Capture from your webcam")
        else:
            image_file = st.file_uploader(label="Upload a test image", type=["jpg", "png", "jpeg"])

    if not image_file:
        image_file = "example_pic.jpg"
    source_image = Image.open(image_file)

    if "Crop" in selected_stages:
        st.sidebar.subheader("Crop")
        st.sidebar.color_picker("Crop box color", value="#FF0000", key="crop_color")
        st.subheader("Crop")
        st.text("Crop result shown as \"Input\" below")
        source_image = st_cropper(source_image, realtime_update=True, box_color=st.session_state["crop_color"], aspect_ratio=None)

    source_image = np.array(source_image)
    input_image = np.copy(source_image)
    display_images = [input_image]
    captions = ["Input"]

    for stage in pipeline:
        match stage:
            case "Blur":
                st.sidebar.subheader("Blur")
                st.sidebar.slider("Blur", min_value=0.0, max_value=20.0, key="blur_amt")
                input_image = gaussian_blur(input_image, st.session_state["blur_amt"])
                display_images.append(np.copy(input_image))
                captions.append("Blur")

            case "Erode":
                st.sidebar.subheader("Erode")
                st.sidebar.selectbox("Shape", ["RECT", "ELLIPSE", "CROSS"], key="erode_shape")
                st.sidebar.slider("Size", min_value=0, max_value=12, key="erode_size", value=1)
                st.sidebar.slider("Iterations", min_value=1, max_value=25, key="erode_iter", value=5)

                shape = match_shape(st.session_state["erode_shape"])
                if shape is None:
                    st.error("Invalid selection. Someone messed up this list")
                    return None

                input_image = erode(input_image, st.session_state["erode_size"], st.session_state["erode_iter"], shape)
                display_images.append(np.copy(input_image))
                captions.append("Erode")

            case "Dilate":
                st.sidebar.subheader("Dilate")
                st.sidebar.selectbox("Shape", ["RECT", "ELLIPSE", "CROSS"], key="dilate_shape")
                st.sidebar.slider("Size", min_value=0, max_value=12, key="dilate_size", value=1)
                st.sidebar.slider("Iterations", min_value=1, max_value=25, key="dilate_iter", value=6)

                shape = match_shape(st.session_state["dilate_shape"])
                if shape is None:
                    st.error("Invalid selection. Someone messed up this list")
                    return None

                input_image = dilate(input_image, st.session_state["dilate_size"], st.session_state["dilate_iter"], shape)
                display_images.append(np.copy(input_image))
                captions.append("Dilate")

            case "Threshold":
                st.sidebar.subheader("Threshold")
                st.sidebar.slider("Hue", 0, 180, (61, 126), key="hue")
                st.sidebar.slider("Saturation", 0, 255, (110, 255), key="sat")
                st.sidebar.slider("Value", 0, 255, (0, 255), key="val")

                hue = st.session_state["hue"]
                sat = st.session_state["sat"]
                val = st.session_state["val"]

                lower_bound = (hue[0], sat[0], val[0])
                upper_bound = (hue[1], sat[1], val[1])
                input_image = hsv_threshold(input_image, lower_bound, upper_bound)
                display_images.append(np.copy(input_image))
                captions.append("HSV Threshold")

            case "Contours":
                st.sidebar.subheader("Contours")
                st.sidebar.selectbox("Approximation Method", ("CHAIN NONE", "CHAIN SIMPLE", "CHAIN TC89 L1", "CHAIN TC89 KCOS"), key="contour_approx_method")

                approx_method = match_approx_method(st.session_state["contour_approx_method"])
                if approx_method is None:
                    st.error("Invalid selection. Someone messed up this list")
                    return None

                contours, hierarchy = contour(input_image, approx_method)
                display_images.append(cv.drawContours(source_image, contours, -1, (0,255,0), 3))
                captions.append("Contours")

    st.write("---")
    st.header("Pipeline")
    i = 0
    for col in st.columns(len(display_images), gap="small"):
        col.subheader(captions[i])
        col.image(display_images[i])
        i += 1

if __name__ == '__main__':
    main_loop()