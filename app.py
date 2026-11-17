import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# 1. Page Configuration
st.set_page_config(
    page_title="Potato Leaf Disease Classifier",
    page_icon="🥔",
    layout="centered"
)

# 2. Constants (Matching your training notebook)
MODEL_PATH = Path(__file__).parent / "models" / "potato_model.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# 3. Load Model Once & Cache It
@st.cache_resource
def load_cnn_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: `{MODEL_PATH}`\n\nPlease run `model.save('models/potato_model.keras')` in your notebook first.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_cnn_model()

# 4. Image Preprocessing Function
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(image)
    # Uncomment the line below ONLY if your model does NOT have a Rescaling(1./255) layer inside it:
    # img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

# 5. User Interface
st.title("🥔 Potato Leaf Disease Classification")
st.write("Upload a potato leaf image to predict whether it has **Early Blight**, **Late Blight**, or is **Healthy**.")

uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    if st.button("Predict Disease", type="primary"):
        with st.spinner("Analyzing leaf..."):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)

            # Apply softmax if your model's final Dense layer outputs raw logits
            if np.max(predictions[0]) > 1.0 or np.min(predictions[0]) < 0.0:
                scores = tf.nn.softmax(predictions[0]).numpy()
            else:
                scores = predictions[0]

            predicted_idx = int(np.argmax(scores))
            raw_class = CLASS_NAMES[predicted_idx]
            clean_label = raw_class.replace("Potato___", "").replace("_", " ")
            confidence = round(float(scores[predicted_idx]) * 100, 2)

            # Display Main Result
            st.subheader("Prediction Result")
            if "healthy" in raw_class.lower():
                st.success(f"**Diagnosis:** {clean_label} ({confidence}% confidence)")
            else:
                st.warning(f"**Diagnosis:** {clean_label} ({confidence}% confidence)")

            # Display All Class Scores
            st.write("### Model Score Breakdown")
            for i, cls_name in enumerate(CLASS_NAMES):
                label = cls_name.replace("Potato___", "").replace("_", " ")
                score_pct = round(float(scores[i]) * 100, 2)
                st.write(f"**{label}:** {score_pct}%")
                st.progress(min(int(score_pct), 100))