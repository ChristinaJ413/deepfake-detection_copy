import streamlit as st
from inference import predict_deepfake

st.title("Deepfake Face Detector")

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded:
    image_bytes = uploaded.getvalue()
    st.image(image_bytes, caption="Uploaded image", use_column_width=True)
    with st.spinner("Analyzing..."):
        result = predict_deepfake(image_bytes)

    if result["label"] == "Fake":
        st.error(f"**Prediction: Fake** ({result['confidence']:.1%} confidence)")
    else:
        st.success(f"**Prediction: Real** ({result['confidence']:.1%} confidence)")

