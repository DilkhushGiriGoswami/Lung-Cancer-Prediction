import streamlit as st
st.set_page_config(page_title=" Lung Cancer Detector", layout="centered")
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = 256
MODEL_PATH = 'LungcancerPrediction.h5'
CLASS_NAMES = ['No Cancer', 'Chance of Cancer', 'Lung Cancer']

# Load the model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Page settings
st.markdown("## 🩺 Lung Cancer Detection from Chest X-ray")
st.markdown("Upload a chest X-ray image and click **Predict** to get a diagnosis based on deep learning.")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def plot_probabilities(probs, class_names):
    fig, ax = plt.subplots()
    ax.barh(class_names, probs, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

# UI when image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Chest X-ray", use_column_width=True)
    st.markdown("### Click below to start analysis.")

    if st.button("🔎 Predict"):
        with st.spinner("Analyzing X-ray..."):
            try:
                input_data = preprocess_image(image)
                prediction = model.predict(input_data)[0]  # Shape: (3,)
                predicted_class = int(np.argmax(prediction))
                confidence = float(prediction[predicted_class])

                st.markdown("---")
                st.subheader(" Model Prediction")
                st.success(f" **Result:** {CLASS_NAMES[predicted_class]}")
                st.info(f"**Confidence:** {confidence:.2%}")

                # Confidence-based message
                if predicted_class == 2 and confidence > 0.7:
                    st.error("High likelihood of Lung Cancer. Please consult a specialist.")
                elif predicted_class == 1:
                    st.warning("There is a chance of Lung Cancer. Further screening is recommended.")
                else:
                    st.success("No signs of Lung Cancer detected with high confidence.")

                # Visualize probabilities
                st.markdown("### 📈 Prediction Confidence")
                plot_probabilities(prediction, CLASS_NAMES)

            except Exception as e:
                st.error(f"Prediction failed due to: {e}")

else:
    st.info("Please upload a chest X-ray image (JPG, JPEG, PNG) to begin.")
