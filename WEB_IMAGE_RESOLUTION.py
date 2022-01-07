from PIL import Image
import streamlit as st
import os.path as osp
import cv2
import torch
import RRDBNet_arch as arch
from PIL import Image
import webbrowser
from keras_preprocessing.image import img_to_array
import os
import PIL
import numpy as np
from tensorflow.python.keras.models import load_model

# for loading image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Title
title=st.title("CI7520: Machine Learning and Artificial Intelligence")

# sidebars
st.sidebar.header(("CI7520: Machine Learning and Artificial Intelligence"))
select_box = st.sidebar.selectbox("Select Model for Image Resolution...",("Home","ESRGAN", "GSRM-OWN"))

if(select_box=="Home"):
    st.title("Course : Data Science")
    st.header("Kingston University, London")
    st.header("Topic: Image Super Resolution")
    st.header("CourseWork-2, Group-2")
    st.subheader("Snehal Desai (K2039389)")
    st.subheader("Madhavi Prajapati (K2044348)")
    st.subheader("Geetanjali Sawant (K1144343)")
    st.subheader("Revathipriya Selvamani (K2007882)")

if (select_box=="ESRGAN"):
    title.header("Selected model is: ESRGAN")
    data = st.file_uploader("Choose Low Resolution Image you want to convert into High Resolution Image...",
                            type=["png","jpg","jpeg","gif"])

    # To display code for user
    if st.button("View Code of ESRGAN"):
        url = 'https://drive.google.com/drive/folders/1WS8wsZFyQpsPqV-D3HvfYPDuf3-X_icP?usp=sharing'
        webbrowser.open_new_tab(url)
        # st.code(get_file_content_as_string(url))

    if data is not None:
        # For loading and saving image in file
        file_details = {"FileName": data.name, "FileType": data.type}
        st.write("Low Resolution Image is...")
        img = load_image(data)
        st.image(img, caption='Uploaded Low Resolution Image.', use_column_width=True)
        with open(os.path.join("LR",data.name), "wb") as f:
            f.write(data.getbuffer())

        # Button to deploy ESRGAN model
        button_ESRGAN = st.button("Deploy Result using ESRGAN Model")
        if (button_ESRGAN):
            st.write("Your model was deployed and visualize the results...")

            # Running ESRGAN Model
            model_path = 'models/RRDB_ESRGAN_x4.pth'
            device = torch.device('cpu')
            test_img_folder = 'LR/' + data.name

            model = arch.RRDBNet(3, 3, 64, 23, gc=32)
            model.load_state_dict(torch.load(model_path), strict=True)
            model.eval()
            model = model.to(device)

            print('Model path {:s}. \nTesting...'.format(model_path))
            base = osp.splitext(osp.basename(test_img_folder))
            print(base)

            # Read images
            img = cv2.imread(test_img_folder, cv2.IMREAD_COLOR)
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            st.write("Super Resolution Image is...")

            # Displaying High Resolution Image
            st.image(output, caption='Uploaded High Resolution Image.', use_column_width=True, channels='BGR')

if (select_box=="GSRM-OWN"):
    title.header("Selected model is: GSRM-OWN")
    data = st.file_uploader("Choose Low Resolution Image you want to convert into High Resolution Image...",
                            type=["png", "jpg", "jpeg", "gif"])

    # To display code for user
    if st.button("View Code of GSRM-OWN"):
        url = 'https://drive.google.com/drive/folders/1iMSaZR28xMUHBdFp4eUR1twWp6oczkeT?usp=sharing'
        webbrowser.open_new_tab(url)
        # st.code(get_file_content_as_string(url))

    # Image we took from user is not none, then display in High Resolution Image
    if data is not None:
        # For loading and saving image in file
        file_details = {"FileName": data.name, "FileType": data.type}
        st.write("Low Resolution Image is...")
        img = load_image(data)
        st.image(img, caption='Uploaded Low Resolution Image.', use_column_width=True)
        with open(os.path.join("GSRM_LR_IMAGES", data.name), "wb") as f:
            f.write(data.getbuffer())

        # Button to deploy GSRM-OWN model
        button_GSRM_OWN = st.button("Deploy Result using our GSRM-OWN Model")
        if (button_GSRM_OWN):
            st.write("Your model was deployed and visualize the results...")

            # Loading GSRM-own model
            model = load_model('OWN_MODEL.h5')
            upscale_factor = 3

            def predicted_output_image(model, input_image):
                # Convert into ycbcr
                ycbcr = input_image.convert("YCbCr")
                y, cb, cr = ycbcr.split()
                y = img_to_array(y)
                y = y.astype("float32") / 255.0
                input = np.expand_dims(y, axis=0)

                # Predict the iamge based on input image
                out = model.predict(input)
                out_img_y = out[0]
                out_img_y *= 255.0

                # Restore the image in RGB color space.
                out_img_y = out_img_y.clip(0, 255)
                out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
                out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
                out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
                out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
                out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
                    "RGB"
                )
                return out_img

            higher_img = predicted_output_image(model, img)  # Predicted output Image

            st.write("Super Resolution Image is...")

            # Displaying High Resolution Image
            st.image(higher_img, caption='Uploaded High Resolution Image.', use_column_width=True, channels='RGB')







