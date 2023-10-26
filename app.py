import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os
from dataset import SegDataTest
from torch.utils.data import DataLoader
import dill

if os.path.exists("./data/images_for_TEST/test_image.jpg"):
    os.remove("./data/images_for_TEST/test_image.jpg")

st.markdown("""## Semantic segmentation of blood cells in medical images.""")
st.image('image.jpg')
st.markdown('---')
st.header("Try segmenting your image or an image from a local database")

uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, channels="BGR")

    def save_uploaded_file(uploadedfile):
        with open(os.path.join("./data/images_for_TEST",uploadedfile.name),"wb") as f:
            f.write(uploadedfile.getbuffer())
        return uploadedfile.name
        # Apply Function here
    file_name = save_uploaded_file(uploaded_file)


    src = "./data/images_for_TEST/" + file_name
    dst = "./data/images_for_TEST/test_image.jpg"
    if os.path.isfile(src):
        os.rename(src, dst)

    st.markdown('---')

    st.subheader("Segmenated image")

    @st.cache_resource
    def load_model():
        with open('./model/segmentation_model1.pth', 'rb') as f:
            model = dill.load(f)
            st.success("Model is Loaded")
            return model

    st.markdown('---')

    TEST_ds = SegDataTest('TEST')
    TEST_dl = DataLoader(TEST_ds, batch_size=1,collate_fn=TEST_ds.collate_fn)
    im = next(iter(TEST_dl))

    model = load_model()
    _mask = model(im)
    # Заберем канал который имеет наивысшую вероятность
    _, _mask = torch.max(_mask, dim=1)

    def display_image_and_mask(original_image, predicted_mask):
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        ax[0].imshow(original_image)
        ax[0].set_title('Original image')
        ax[0].axis('off')
        ax[1].imshow(predicted_mask)
        ax[1].set_title('Predicted mask')
        ax[1].axis('off')
        plt.tight_layout()
        return fig
    # Assuming you have the original image as 'original_image' and predicted mask as 'predicted_mask'
    fig = display_image_and_mask(im[0].permute(1,2,0).detach().cpu()[:,:,0],
                        _mask.permute(1,2,0).detach().cpu()[:,:,0])

    # Render the figure using st.pyplot
    st.pyplot(fig)
