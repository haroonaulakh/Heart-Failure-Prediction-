import streamlit as st
import base64

st.markdown(
    """
    <style>
        .main { background-color: #080606; padding: 20px; font-family: 'Arial', sans-serif; }
        .sidebar .sidebar-content { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
        h1 { color: #2c3e50; font-size: 28px; font-weight: bold; margin-bottom: 10px; }
        h2, h3 { color: #34495e; font-weight: bold; }
        p { font-size: 16px; color: #555555; }
        .footer { margin-top: 20px; text-align: center; font-size: 14px; color: #777777; }
        .card { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
        .metric { display: flex; justify-content: space-around; margin-bottom: 20px; }
        .metric > div { text-align: center; padding: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚗 Traffic Accident Analysis")
st.write("Explore insights and statistics on road accidents across the U.S.")