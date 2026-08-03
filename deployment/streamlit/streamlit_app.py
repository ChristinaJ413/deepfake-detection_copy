from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import DeepfakeDetector


APP_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = (
    APP_DIRECTORY
    / "models"
    / "resnet50_model.onnx"
)

# This must match the threshold selected using validation data.
DECISION_THRESHOLD = 0.45

# Keep this True when the ONNX model returns a raw binary logit.
OUTPUT_IS_LOGIT = True


st.set_page_config(
    page_title="Synthetic Sight",
    page_icon="👁️",
    layout="centered",
)


@st.cache_resource
def load_detector() -> DeepfakeDetector:
    """
    Load the model once and reuse it across predictions.

    Streamlit reruns the script after user interactions, so caching the
    detector prevents the model from loading repeatedly.
    """
    return DeepfakeDetector(
        model_path=MODEL_PATH,
        decision_threshold=DECISION_THRESHOLD,
        output_is_logit=OUTPUT_IS_LOGIT,
    )


st.title("Synthetic Sight")

st.subheader("AI-Generated Face Image Detection")

st.write(
    "Upload a face image to receive an estimate of whether the image "
    "is real or AI-generated. The prediction is produced by our "
    "fine-tuned ResNet-50 model."
)

st.warning(
    "This application is an educational research prototype. Its "
    "prediction is not definitive proof that an image is authentic "
    "or AI-generated."
)

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Select a JPG, JPEG, PNG, or WebP image containing a face.",
)

if uploaded_file is None:
    st.info("Upload an image to begin.")

else:
    try:
        image = Image.open(uploaded_file)
        image.load()
        image = image.convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        st.error(
            "The uploaded file could not be opened as an image. "
            "Please try a different JPG, PNG, or WebP file."
        )
        st.stop()

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True,
    )

    analyze_button = st.button(
        "Analyze image",
        type="primary",
        use_container_width=True,
    )

    if analyze_button:
        try:
            with st.spinner("Analyzing the image..."):
                detector = load_detector()
                result = detector.predict(image)

            prediction = str(result["prediction"])
            fake_probability = float(
                result["fake_probability"]
            )
            real_probability = float(
                result["real_probability"]
            )
            threshold = float(
                result["decision_threshold"]
            )

            if prediction == "Fake":
                st.error(f"Prediction: {prediction}")
            else:
                st.success(f"Prediction: {prediction}")

            probability_columns = st.columns(2)

            with probability_columns[0]:
                st.metric(
                    "Fake probability",
                    f"{fake_probability:.1%}",
                )

            with probability_columns[1]:
                st.metric(
                    "Real probability",
                    f"{real_probability:.1%}",
                )

            st.write("Fake probability")

            st.progress(
                fake_probability,
                text=f"{fake_probability:.1%}",
            )

            st.caption(
                f"Decision threshold: {threshold:.2f}. "
                "The model labels an image Fake when its estimated "
                "fake probability is equal to or greater than this "
                "threshold."
            )

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "The model could not complete the prediction. "
                f"Technical details: {error}"
            )


st.divider()

st.caption(
    "Synthetic Sight · AI4ALL Ignite · Educational prototype"
)
