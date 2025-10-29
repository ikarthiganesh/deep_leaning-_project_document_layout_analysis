import streamlit as st
from ultralytics import YOLO
import os
import requests
import numpy as np
import cv2
from PIL import Image

# ✅ Download model from Google Drive if not found locally
url = "https://drive.google.com/uc?id=1JYGpsZm7AnNTewf0-phE0Hrc_aUswXmR"
model_path = "best.pt"

if not os.path.exists(model_path):
    st.info("📥 Downloading YOLO model... Please wait...")
    r = requests.get(url, allow_redirects=True)
    open(model_path, 'wb').write(r.content)
    st.success("✅ Model downloaded successfully!")

# ✅ Load model
model = YOLO(model_path)

# ✅ Streamlit UI
st.title("📰 Document Layout Analysis using YOLOv8")
st.write("Upload any document image (magazine, paper, or report) to visualize its layout structure.")

uploaded_file = st.file_uploader("📂 Upload a document image", type=["jpg", "jpeg", "png"])

colors = {
    "text": (0, 255, 0),
    "title": (255, 215, 0),
    "list": (255, 140, 0),
    "table": (255, 69, 0),
    "figure": (30, 144, 255)
}

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # ✅ Run prediction
    results = model.predict(source=img_array, conf=0.5, verbose=False)

    # ✅ Draw bounding boxes
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        color = colors.get(label, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        text = f"{label.upper()} ({conf:.2f})"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_array, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        cv2.putText(img_array, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    st.image(img_array, caption="🧾 Detected Layout", use_column_width=True)
    st.success("✅ Layout Analysis Completed Successfully!")
