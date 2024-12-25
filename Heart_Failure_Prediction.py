import streamlit as st
import base64

page_bg_img = """
<style>
body {
    background-image: url("https://www.w3schools.com/w3images/mountains.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.write("This verifies if an online image works.")