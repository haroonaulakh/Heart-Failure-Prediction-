import streamlit as st
import pandas as pd
import base64




# title 
st.title("Heart Failure Prediction")

uploaded_file = st.file_uploader("https://www.magicpattern.design/tools/starry-sky-generator", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img_path = uploaded_file.name
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    base64_img = get_base64_of_bin_file(img_path)

    # Custom CSS for background image
    page_bg_img = f"""
    <style>
    body {{
        background-image: url("data:image/jpeg;base64,{base64_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Apply the CSS
st.markdown(page_bg_img, unsafe_allow_html=True)