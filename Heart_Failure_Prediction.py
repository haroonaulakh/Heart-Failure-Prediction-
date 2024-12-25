import streamlit as st
import pandas as pd
import base64




# title 
st.title("Heart Failure Prediction")

# background
# Function to encode a local image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your local image
img_path = "C:\\Users\\DELL\\Desktop\\IDS-SEM-PROJ\\steptodown.com601822.jpg"
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

# Apply the CSS
st.markdown(page_bg_img, unsafe_allow_html=True)