import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import bcrypt
import time
import pandas as pd
import altair as alt
import certifi

# ---------------------------- LOAD ENV ----------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# ---------------------------- DATABASE ----------------------------
try:
    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    db = client["weapon_detection"]
    users_col = db["users"]
    history_col = db["history"]
    db_status = True
except Exception as e:
    db_status = False
    st.error(f" Database connection failed: {e}")

# ---------------------------- SESSION STATE ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None

# ---------------------------- LOAD YOLO ----------------------------
model_path = "yolov8n.pt"
model = YOLO(model_path)

# ---------------------------- SETTINGS ----------------------------
RED_ZONE_THRESHOLD = 5  # Persons count to trigger Red Zone

# ---------------------------- CSS ----------------------------
st.markdown("""
    <style>
    .main {background-color: #F9F5EC;}
    section[data-testid="stSidebar"] {background-color: #B76E79 !important; color: white;}
    .stButton>button {background-color: #B76E79; color: white; border-radius: 8px;}
    .stButton>button:hover {background-color: #9d5c66;}
    h1, h2, h3, h4, h5, h6 {color: #B76E79;}
    </style>
""", unsafe_allow_html=True)

# Hide sidebar when not logged in
if not st.session_state.logged_in:
    hide_sidebar = """
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
    """
    st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------------------- FUNCTIONS ----------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def login_user(username, password):
    if not db_status:
        st.error("⚠ Cannot login. No database connection.")
        return False
    user = users_col.find_one({"username": username})
    if user and check_password(password, user["password"]):
        st.session_state.logged_in = True
        st.session_state.user = user
        st.rerun()
        return True
    return False

def register_user(username, password):
    if not db_status:
        st.error("⚠ Cannot register. No database connection.")
        return False
    if users_col.find_one({"username": username}):
        return False
    users_col.insert_one({
        "username": username,
        "password": hash_password(password)
    })
    return True

def detect_persons(image):
    results = model(image)
    output_img = results[0].plot()
    count = sum(1 for box in results[0].boxes if "person" in model.names[int(box.cls[0])].lower())
    if count >= RED_ZONE_THRESHOLD:
        cv2.putText(output_img, f"⚠ RED ZONE: {count} persons", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    return output_img, count

# ---------------------------- AUTHENTICATION ----------------------------
if not st.session_state.logged_in:
    st.title("Welcome to Crowd Count Detection")
    #st.write("Login or Register below to access the Dashboard.")

    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login_user(username, password):
                st.success(f"Welcome {username}! Redirecting to Dashboard...")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_username = st.text_input("Choose Username", key="reg_user")
        new_password = st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("Register"):
            if register_user(new_username, new_password):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username already exists!")

# ---------------------------- MAIN DASHBOARD ----------------------------
else:
    st.sidebar.title(f" Hello, {st.session_state.user['username']}")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Upload & Detect", "Live Webcam"])
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()

    # ---------------- DASHBOARD (with History integrated) ----------------
    if page == "Dashboard":
        st.title(" Dashboard & History Overview")

        if db_status:
            total_uploads = history_col.count_documents({"user": st.session_state.user["username"]})
            st.metric("Total Uploads Analyzed", total_uploads)

            st.subheader(" Upload History & Analysis")
            user_history = list(history_col.find({"user": st.session_state.user["username"]}))
            if not user_history:
                st.info("No history found yet. Upload images/videos to generate history!")
            else:
                df = pd.DataFrame(user_history)
                for col in ["filename", "result", "timestamp"]:
                    if col not in df.columns:
                        df[col] = None

                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df["result"] = pd.to_numeric(df["result"], errors="coerce").fillna(0)
                st.dataframe(df[["filename", "result", "timestamp"]].fillna("N/A"))

                csv = df[["filename", "result", "timestamp"]].fillna("N/A").to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download CSV", csv, "history.csv", "text/csv")

                if len(df) > 0:
                    st.subheader(" Person Count per File")
                    chart_df = df[["filename", "result", "timestamp"]].fillna(0)
                    chart = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X('filename', sort=None),
                        y='result',
                        tooltip=['filename','result','timestamp']
                    ).properties(width=700, height=400)
                    st.altair_chart(chart, use_container_width=True)

                    st.subheader(" Person Detection Trend Over Time")
                    df_valid = df.dropna(subset=['timestamp'])
                    if not df_valid.empty:
                        df_valid = df_valid.sort_values("timestamp")
                        trend_chart = alt.Chart(df_valid).mark_line(point=True).encode(
                            x=alt.X('timestamp:T', title='Time'),
                            y=alt.Y('result:Q', title='Persons Detected'),
                            tooltip=['filename', 'result', 'timestamp']
                        ).properties(width=700, height=400)
                        st.altair_chart(trend_chart, use_container_width=True)
                    else:
                        st.info("No valid data available for visualization.")
        else:
            st.warning("Database connection unavailable.")

    # ---------------- UPLOAD & DETECT ----------------
    elif page == "Upload & Detect":
        st.title(" Upload Image/Video for Detection")
        mode = st.radio("Select Mode", ["Image", "Video"])

        if mode == "Image":
            uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    img_array = np.array(image)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    output_img, count = detect_persons(img_array)
                    st.image(output_img, caption=f"Detected Persons: {count}", use_column_width=True)
                    st.success(f" Persons Detected: {count}")

                    if db_status:
                        history_col.insert_one({
                            "user": st.session_state.user["username"],
                            "filename": uploaded_file.name,
                            "result": count,
                            "timestamp": datetime.utcnow()
                        })
                        st.success("✅ Detection result saved to MongoDB!")
                except Exception as e:
                    st.error(f"⚠ Failed to read image: {e}")

        elif mode == "Video":
            uploaded_video = st.file_uploader("Upload Video", type=["mp4","mov","avi","mkv"])
            if uploaded_video:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                video_path = tfile.name

                st.video(video_path)
                stframe = st.empty()
                person_counter = st.empty()
                cap = cv2.VideoCapture(video_path)
                stop_button = st.button("⏹ Stop Processing")

                count = 0
                st.success("Processing video...")

                while cap.isOpened():
                    if stop_button:
                        st.warning("⏹ Video processing stopped by user.")
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    output_frame, count = detect_persons(frame)
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(output_frame, channels="RGB", use_container_width=True)
                    person_counter.info(f" Persons in Current Frame: {count}")
                cap.release()
                st.info("✅ Video processing completed!")

                if db_status:
                    history_col.insert_one({
                        "user": st.session_state.user["username"],
                        "filename": uploaded_video.name,
                        "result": count,
                        "timestamp": datetime.utcnow()
                    })
                    st.success("✅ Detection result saved to MongoDB!")

    # ---------------- LIVE WEBCAM ----------------
    elif page == "Live Webcam":
        st.title(" Live Webcam Detection")
        stframe = st.empty()
        person_counter = st.empty()
        run = st.checkbox("Start Webcam")
        stop = st.button("Stop")
        camera = cv2.VideoCapture(0)

        while run and not stop:
            ret, frame = camera.read()
            if not ret:
                st.warning("⚠ Failed to access webcam frame.")
                break
            output_frame, count = detect_persons(frame)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            stframe.image(output_frame, channels="RGB", use_container_width=True)
            person_counter.info(f"Persons in Current Frame: {count}")
            time.sleep(0.1)
        camera.release()
        st.info("Webcam stopped. ✅")
