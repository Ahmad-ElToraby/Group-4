import streamlit as st
import requests
from PIL import Image

st.title("Emotion Detection App ")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    # Show image
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=250)

    if st.button("Predict Emotion"):
        files = {"file": uploaded.getvalue()}

        # Call FastAPI endpoint
        res = requests.post("http://127.0.0.1:8000/predict", files=files)
        # json file for emotion

        emotion = res.json()["emotion"]
        st.success(f"Detected Emotion: {emotion}")
        #
