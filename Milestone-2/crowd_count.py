
import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import time
import hashlib

# ----------------------------
# 🎨 Page Setup
# ----------------------------
st.set_page_config(page_title="Person Detection & Counting", page_icon="🧍", layout="wide")

# ----------------------------
# 🧠 Load YOLO Model
# ----------------------------
model_path = "yolov8n.pt"  # replace with 'best.pt' if you have a trained custom person model
model = YOLO(model_path)

# ----------------------------
# 🛠 User Authentication
# ----------------------------
if "users" not in st.session_state:
    st.session_state["users"] = {}

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "username" not in st.session_state:
    st.session_state["username"] = ""


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def login_user(username, password):
    if username in st.session_state["users"]:
        return st.session_state["users"][username] == hash_password(password)
    return False


def register_user(username, password):
    if username in st.session_state["users"]:
        return False
    st.session_state["users"][username] = hash_password(password)
    return True


# ----------------------------
# 🧩 Detection Function
# ----------------------------
def detect_persons(image):
    results = model(image)
    detected_img = results[0].plot()
    person_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if "person" in label.lower():
            person_count += 1

    return detected_img, person_count


# ----------------------------
# 🔑 Unified Auth Page
# ----------------------------
if not st.session_state["logged_in"]:
    st.title("🧍 Welcome to Person Detection & Counting")
    st.write("Login or Register below to access the Dashboard.")

    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome {username}! Redirecting to Dashboard...")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_username = st.text_input("Choose a Username", key="reg_user")
        new_password = st.text_input("Choose a Password", type="password", key="reg_pass")
        if st.button("Register"):
            if register_user(new_username, new_password):
                st.success("Registration successful! Please log in.")
            else:
                st.error("Username already exists!")

# ----------------------------
# 📊 Dashboard
# ----------------------------
else:
    st.sidebar.title(f"👋 Hello, {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.rerun()

    st.title("🧍 Person Detection Dashboard")
    st.write("Detect and count people using images, videos, or your webcam.")

    mode = st.radio("Select Mode", ["Upload Image", "Upload Video", "Live Webcam"])

    # Image Mode
    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            with st.spinner("Detecting persons..."):
                img_array = np.array(image)
                output_img, count = detect_persons(img_array)
                st.image(output_img, caption=f"Detected Persons: {count}", use_column_width=True)
                st.success(f"🧍 Total Persons Detected: {count}")

    # Video Mode
    elif mode == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            st.video(video_path)
            stframe = st.empty()
            person_counter_placeholder = st.empty()
            cap = cv2.VideoCapture(video_path)

            st.success("Processing video... 🧍")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                output_frame = results[0].plot()
                person_count = 0
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if "person" in label.lower():
                        person_count += 1

                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                stframe.image(output_frame, channels="RGB", use_column_width=True)
                person_counter_placeholder.success(f"🧍 Persons Detected: {person_count}")

            cap.release()
            st.info("Video processing completed ✅")

    # Webcam Mode
    elif mode == "Live Webcam":
        stframe = st.empty()
        person_counter_placeholder = st.empty()
        run = st.checkbox("Start Webcam")
        stop = st.button("Stop")

        camera = cv2.VideoCapture(0)
        while run and not stop:
            ret, frame = camera.read()
            if not ret:
                st.warning("Failed to access webcam.")
                break

            results = model(frame)
            output_frame = results[0].plot()
            person_count = 0
            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if "person" in label.lower():
                    person_count += 1

            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            stframe.image(output_frame, channels="RGB", use_column_width=True)
            person_counter_placeholder.success(f"🧍 Persons Detected: {person_count}")

            time.sleep(0.1)

        camera.release()
        st.info("Webcam stopped. ✅")

st.markdown("---")
#st.caption("Developed by • Powered by YOLOv8 + Streamlit • © 2025")
