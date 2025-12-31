import streamlit as st
import pandas as pd
import os
import re
from AHBA_label_transfer import generate_image_label
from AHBA_label_transfer import colorize_labels
import numpy as np
from PIL import Image

def get_all_file_numbers(folder_path):
    get_train_files = os.listdir(folder_path)
    all_numbers = []
    for file in get_train_files:
        numbers_in_file = re.findall(r'\d+', file)
        all_numbers.extend(numbers_in_file)
    all_numbers = sorted([int(num) for num in all_numbers])
    return all_numbers


st.header("""Brain Image Segmentation & Labeling""")

train_image_numbers = get_all_file_numbers("./training_images/images/")
train_image_num = st.selectbox("**Pre-labeled brain slabs (training set, ordered):**", train_image_numbers, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose an option", disabled=False, label_visibility="visible")

train_col1, train_col2 = st.columns(2)
with st.container():
    with train_col1:
        st.markdown("**Original image (training)**") 
        st.image(f"./training_images/images/Brodmann_Nissl_{train_image_num}.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
    with train_col2:
        st.markdown("**Segmented mask (training)**") 
        mask_img = np.array(Image.open(f"./training_images/masks/Brodmann_Nissl_{train_image_num}.png"))
        st.image(colorize_labels(mask_img, bg_label=-1), caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)


st.markdown("---")


test_image_numbers = get_all_file_numbers("./testing_images/")
test_image_anno = []
test_col1, test_col2 = st.columns(2)
with st.container():
    with test_col1:
        test_image_num = st.selectbox("**Unlabeled brain slabs (testing set, same order):**", test_image_numbers, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose an option", disabled=False, label_visibility="visible")
    with test_col2:
        st.write("")
        st.write("")
        if st.button('Generate label:'):
            test_image = Image.open(f"./testing_images/Brodmann_Nissl_{test_image_num}.png")
            test_image_array = np.array(test_image)

            closest_train_image_to_test = min(train_image_numbers, key=lambda x: abs(x - test_image_num))
            train_image = Image.open(f"./training_images/images/Brodmann_Nissl_{closest_train_image_to_test}.png")
            train_image_array = np.array(train_image)
            
            anno_image = Image.open(f"./training_images/masks/Brodmann_Nissl_{closest_train_image_to_test}.png")
            anno_image_array = np.array(anno_image)

            test_image_anno = generate_image_label(test_image_array, train_image_array, anno_image_array)
            
test_anno_col1, test_anno_col2 = st.columns(2)
with st.container():
    with test_anno_col1:  
        st.markdown("**Original image (testing)**") 
        st.image(f"./testing_images/Brodmann_Nissl_{test_image_num}.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
    with test_anno_col2:
        if len(test_image_anno) > 1:
            st.markdown("**Segmented mask (testing)**") 
            st.image(colorize_labels(test_image_anno, bg_label=-1), caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
