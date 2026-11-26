
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Rare Genetic Disease Screening",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AI-Assisted Deep Phenotyping")
st.subheader("Rare Genetic Disease Screening in India")

st.write(
    "An AI-assisted research prototype using "
    "Deep Learning and Computer Vision."
)

st.info("Upload a facial image to begin.")

uploaded_file = st.file_uploader(
    "Upload Patient Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    st.warning(
        "Model integration is pending. "
        "This research prototype does not provide "
        "a medical diagnosis."
    )

st.divider()
st.caption("Research Project | BIT Mesra")

