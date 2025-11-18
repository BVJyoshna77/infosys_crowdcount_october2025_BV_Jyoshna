import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import time

# ----------------------------
# 🎨 Page Setup
# ----------------------------
st.set_page_config(page_title="Apple Detection & Counting", page_icon="🍎", layout="wide")

st.title("🍎 Real-Time Apple Detection and Counting System")
st.write("This app detects and counts apples in real-time using a YOLOv8 model and your webcam/video/image.")

# ----------------------------
# 🧠 Load YOLO Model
# ----------------------------
model_path = "yolov8n.pt"  # replace with 'best.pt' if you have a trained apple model
model = YOLO(model_path)

# ----------------------------
# 📷 Mode Selection
# ----------------------------
mode = st.radio("Select Mode", ["Upload Image", "Upload Video", "Live Webcam"])

# ----------------------------
# 🧩 Detection Function
# ----------------------------
def detect_apples(image):
    results = model(image)
    detected_img = results[0].plot()
    apple_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if "apple" in label.lower():
            apple_count += 1

    return detected_img, apple_count

# ----------------------------
# 🖼 Image Upload Mode
# ----------------------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Detecting apples..."):
            img_array = np.array(image)
            output_img, count = detect_apples(img_array)
            st.image(output_img, caption=f"Detected Apples: {count}", use_column_width=True)
            st.success(f"🍏 Total Apples Detected: {count}")

# ----------------------------
# 📹 Video Upload Mode
# ----------------------------
elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)  # show original video
        stframe = st.empty()
        apple_counter_placeholder = st.empty()
        cap = cv2.VideoCapture(video_path)

        st.success("Processing video... 🍎")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            output_frame = results[0].plot()
            apple_count = 0
            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if "apple" in label.lower():
                    apple_count += 1

            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            stframe.image(output_frame, channels="RGB", use_column_width=True)
            apple_counter_placeholder.success(f"🍎 Apples Detected: {apple_count}")

        cap.release()
        st.info("Video processing completed ✅")

# ----------------------------
# 🎥 Live Webcam Mode
# ----------------------------
elif mode == "Live Webcam":
    stframe = st.empty()
    apple_counter_placeholder = st.empty()

    run = st.checkbox("Start Webcam")
    stop = st.button("Stop")

    camera = cv2.VideoCapture(0)

    while run and not stop:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to access webcam.")
            break

        # Perform detection
        results = model(frame)
        output_frame = results[0].plot()
        apple_count = 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if "apple" in label.lower():
                apple_count += 1

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        stframe.image(output_frame, channels="RGB", use_column_width=True)
        apple_counter_placeholder.success(f"🍎 Apples Detected: {apple_count}")

        time.sleep(0.1)  # controls refresh rate

    camera.release()
    st.info("Webcam stopped. ✅")

st.markdown("---")
#st.caption("Developed by Divyasree • Powered by YOLOv8 + Streamlit • © 2025")
