import streamlit as st
import base64
import os

# Title
st.title("Heart Failure Prediction")

# Debug: Check file path
img_path = r"steptodown.com601822.jpg"
if not os.path.exists(img_path):
    st.write("Image file not found! Check the file path.")
else:
    # Base64 encoding
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    base64_img = get_base64_of_bin_file(img_path)

    # Apply CSS for background image
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
    st.write("Background image applied successfully!")
