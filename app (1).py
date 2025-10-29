import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load your trained model
model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")

st.title("🧠 Document Layout Analysis (PubLayNet)")
st.write("Upload any document image to visualize detected layout elements.")

uploaded_file = st.file_uploader("📄 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(image, caption='Uploaded Document', use_column_width=True)

    # Run model prediction
    results = model.predict(source=img_array, conf=0.5)
    annotated = results[0].plot()  # Draw bounding boxes

    st.image(annotated, caption="🟩 Text | 🟦 Table | 🟥 Figure | 🟨 Title | 🟧 List", use_column_width=True)
    st.success("✅ Layout detected successfully!")
