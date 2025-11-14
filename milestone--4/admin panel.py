# Analytics — Crowd Count (YOLOv8s, CPU-friendly) + Red Zone Support
# -------------------------------------------------------------------
# pip install ultralytics streamlit plotly opencv-python bcrypt pyjwt pandas numpy streamlit-drawable-canvas pillow
# Optional (newer Streamlit): pip install streamlit-drawable-canvas-fix
# streamlit run (appname).py
# -------------------------------------------------------------------

import os
import io
import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import cv2
from PIL import Image
import bcrypt
import jwt

# --- Streamlit <-> drawable-canvas compatibility shim for image_to_url path ---
# Some versions of streamlit-drawable-canvas import image_to_url from
# "streamlit.elements.lib.image_utils" while newer Streamlit exposes it at
# "streamlit.elements.image_utils". We alias whichever exists to both paths
# so imports inside the canvas package succeed without errors.
import sys
try:
    from streamlit.elements import image_utils as _iu  # Newer layout
except Exception:
    try:
        from streamlit.elements.lib import image_utils as _iu  # Older layout
    except Exception:
        _iu = None
if _iu is not None:
    sys.modules.setdefault("streamlit.elements.image_utils", _iu)
    sys.modules.setdefault("streamlit.elements.lib.image_utils", _iu)
# -----------------------------------------------------------------------------

# Canvas import with typo guard
try:
    from streamlit_drawable_camus import st_canvas  # will fail if typo
    DRAWING_AVAILABLE = True
except Exception:
    try:
        from streamlit_drawable_canvas import st_canvas
        DRAWING_AVAILABLE = True
    except Exception:
        st_canvas = None  # type: ignore
        DRAWING_AVAILABLE = False

# ----------------------- App Config ------------------------

APP_TITLE = "CrowdCount"
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "dev-super-secret-jwt-key-change-me")
JWT_ALGO = "HS256"
TOKEN_TTL_HOURS = 8

DB_PATH = os.getenv("CROWD_DB_PATH", "crowd_count.db")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# YOLO model + performance knobs
YOLO_MODEL_NAME = "yolov8s.pt"
DETECT_EVERY_N_FRAMES = 5  # detect every 5th frame for CPU

# ----------------------- DB Helpers ------------------------

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash BLOB NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source_url TEXT NOT NULL,
    created_by INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    points_json TEXT NOT NULL,
    is_red INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(camera_id) REFERENCES cameras(id)
);

CREATE TABLE IF NOT EXISTS counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    camera_id INTEGER,
    zone_id INTEGER,
    total_count INTEGER,
    img_density REAL,
    source_kind TEXT, -- image | video | live
    note TEXT,
    FOREIGN KEY(camera_id) REFERENCES cameras(id),
    FOREIGN KEY(zone_id) REFERENCES zones(id)
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    user_email TEXT
);
"""

with get_conn() as conn:
    exec_result = conn.executescript(SCHEMA_SQL)
    cols = [c[1] for c in conn.execute("PRAGMA table_info(zones)").fetchall()]
    if "is_red" not in cols:
        conn.execute("ALTER TABLE zones ADD COLUMN is_red INTEGER NOT NULL DEFAULT 0")
        conn.commit()

# ----------------------- Utilities ------------------------

def log(level: str, message: str, user_email: Optional[str] = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO logs(ts, level, message, user_email) VALUES (?,?,?,?)",
            (datetime.utcnow().isoformat(), level.upper(), message, user_email),
        )
        conn.commit()

def hash_password(pw: str) -> bytes:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt())

def check_password(pw: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), hashed)
    except Exception:
        return False

def create_token(payload: dict) -> str:
    exp = datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS)
    to_encode = {**payload, "exp": exp}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGO)

def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except Exception:
        return None

def get_current_user():
    token = st.session_state.get("token")
    if not token:
        return None
    return decode_token(token)

def is_admin(user: Optional[dict]) -> bool:
    return bool(user and user.get("role") == "admin")

def point_in_polygon(point, polygon):
    # Ray casting
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# ------------------ YOLOv8 Person Detector ----------------

from ultralytics import YOLO

@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLO(YOLO_MODEL_NAME)

def yolo_detect_persons(frame: np.ndarray):
    """Return list of (x,y,w,h) and an annotated frame."""
    if frame is None:
        return [], np.zeros((1,1,3), dtype=np.uint8)
    model = load_yolo()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, verbose=False)[0]
    persons = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            persons.append((x1, y1, x2 - x1, y2 - y1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    return persons, frame

# -------- Frame helpers for zone editing from video/live --------

def grab_frame_from_video(video_bytes_or_path, at_seconds: float = 1.0) -> Optional[np.ndarray]:
    """
    Accepts either a file-like bytes object (uploaded video) or a file path.
    Seeks to at_seconds and returns a single BGR frame (numpy array) or None.
    """
    tmp_path = None
    try:
        if isinstance(video_bytes_or_path, (bytes, bytearray)):
            tmp_path = f"./vid_tmp{int(time.time()*1000)}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(video_bytes_or_path)
            src = tmp_path
        else:
            src = str(video_bytes_or_path)

        cap = cv2.VideoCapture(src)
        if at_seconds > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return frame
        return None
    finally:
        if tmp_path:
            try: os.remove(tmp_path)
            except: pass


def grab_frame_from_source(src: str = "0") -> Optional[np.ndarray]:
    """
    Grabs one frame from a webcam (0) or RTSP/HTTP stream URL and returns BGR frame.
    """
    source = 0 if src.strip() == "0" else src.strip()
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    return frame if ok and frame is not None else None

# ----------------------- THEME (Light Purple & White) -----------------------


def apply_theme():
    # Subtle light-purple -> white background, purple accents, white cards/inputs
    st.markdown("""
        <style>
        :root {
            --brand-purple: #7C3AED;   /* purple-600 */
            --brand-purple-500: #8B5CF6;/* purple-500 */
            --light-purple: #F3E8FF;   /* purple-100 */
            --bg-white: #FFFFFF;
            --text-dark: #1F2937;
            --muted: #6B7280;
            --radius: 14px;
            --padding-top:20px;
        }
        /* page background */
        .stApp {
            background: linear-gradient(120deg, var(--light-purple), var(--bg-white));
        }
        /* center content a bit narrower for a clean card look */
        .block-container { max-width: 1100px; }

        /* headers */
        h1, h2, h3, h4 { color: var(--text-dark) !important; }
        h1 { font-weight: 800; }

        /* generic "card" look for containers that Streamlit renders (forms, tables) */
        .stMarkdown, .stDataFrame, .stPlotlyChart, .stTextInput, .stNumberInput, .stSelectbox, .stFileUploader, .stCheckbox, .stRadio {
            border-radius: var(--radius) !important;
        }
        /* inputs background */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stDateInput>div>div>input {
            background: var(--bg-white) !important;
            border: 1px solid #E5E7EB !important;
            border-radius: var(--radius) !important;
        }
        .stSelectbox > div > div { background: var(--bg-white) !important; border-radius: var(--radius) !important; }

        /* buttons */
        .stButton>button {
            background: var(--brand-purple) !important;
            color: white !important;
            border: 0 !important;
            border-radius: 12px !important;
            padding: 0.5rem 1rem !important;
        }
        .stButton>button:hover { background: var(--brand-purple-500) !important; }

        /* sidebar keep white card look to match theme */
        section[data-testid="stSidebar"] {
            background: var(--bg-white) !important;
            border-right: 1px solid #E5E7EB !important;
        }
        /* info/success boxes softened */
        .stAlert { border-radius: var(--radius) !important; }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Create Account", layout="wide")
    apply_theme()  # <-- Call this FIRST!
    # ... REST OF YOUR FUNCTIONALITY (login/register routing, etc.) ...


# ----------------------- Auth UI ---------------------------

def ui_register():
    st.header("Create Account")
    with st.form("register_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["user", "admin"], help="Admins see the Admin Panel")
        submitted = st.form_submit_button("Register")
    if submitted:
        if not (name and email and pw):
            st.error("Please fill all fields")
            return
        with get_conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                    (name, email, hash_password(pw), role, datetime.utcnow().isoformat()),
                )
                conn.commit()
                log("INFO", f"New user registered: {email}")
                st.success("Account created. You can now log in.")
            except sqlite3.IntegrityError:
                st.error("Email already exists.")

def ui_login():
    st.header("Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        selected_role = st.selectbox("Role", ["user", "admin"])  # Role selector
        submitted = st.form_submit_button("Sign In")
    if submitted:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT id, name, email, password_hash, role FROM users WHERE email=?",
                (email,)
            ).fetchone()
        if row:
            uid, name, email_db, pw_hash, role = row
            if check_password(pw, pw_hash):
                if selected_role != role:
                    st.error(f"Role mismatch: your account role is '{role}'. Please select '{role}'.")
                    return
                token = create_token({"user_id": uid, "email": email_db, "name": name, "role": role})
                st.session_state["token"] = token
                st.success("Logged in!")
                log("INFO", "User logged in", email)
                st.rerun()
            else:
                st.error("Invalid credentials.")
        else:
            st.error("Invalid credentials.")

def ui_logout_button():
    if st.button("Logout"):
        st.session_state.pop("token", None)
        st.info("Logged out")
        st.rerun()

# ----------------------- Camera & Zones --------------------

def list_cameras() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT id,name,source_url,created_at FROM cameras ORDER BY id DESC", conn)
    return df

def create_camera(name: str, source_url: str, user_email: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO cameras(name,source_url,created_by,created_at) VALUES(?,?,?,?)",
            (name, source_url, None, datetime.utcnow().isoformat()),
        )
        conn.commit()
    log("INFO", f"Camera added: {name}", user_email)

def delete_camera(cam_id: int, user_email: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
        conn.commit()
    log("WARN", f"Camera deleted id={cam_id}", user_email)

def list_zones(camera_id: Optional[int]) -> pd.DataFrame:
    if not camera_id:
        return pd.DataFrame(columns=["id","name","points_json","is_red","created_at","camera_id"])
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT id,camera_id,name,points_json,is_red,created_at FROM zones WHERE camera_id=? ORDER BY id DESC",
            conn,
            params=(camera_id,),
        )
    return df

def save_zone(camera_id: int, name: str, points: List[Tuple[int,int]], is_red: bool = False):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO zones(camera_id,name,points_json,is_red,created_at) VALUES(?,?,?,?,?)",
            (camera_id, name, json.dumps(points), int(is_red), datetime.utcnow().isoformat()),
        )
        conn.commit()

def update_zone_label(zone_id: int, new_label: str):
    with get_conn() as conn:
        conn.execute("UPDATE zones SET name=? WHERE id=?", (new_label, zone_id))
        conn.commit()

def update_zone_is_red(zone_id: int, is_red: bool):
    with get_conn() as conn:
        conn.execute("UPDATE zones SET is_red=? WHERE id=?", (int(is_red), zone_id))
        conn.commit()

def delete_zone(zone_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM zones WHERE id=?", (zone_id,))
        conn.commit()

# ----------------------- Counting + Storage ---------------

def save_count(ts: datetime, camera_id: Optional[int], zone_id: Optional[int], total_count: int, density: float, source_kind: str, note: str=""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO counts(ts,camera_id,zone_id,total_count,img_density,source_kind,note) VALUES(?,?,?,?,?,?,?)",
            (ts.isoformat(), camera_id, zone_id, total_count, float(density), source_kind, note),
        )
        conn.commit()

def get_counts_df(days: int = 7) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM counts WHERE ts >= ? ORDER BY ts ASC",
            conn,
            params=((datetime.utcnow() - timedelta(days=days)).isoformat(),),
        )
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def get_scalar(sql: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(sql)
        return int(cur.fetchone()[0])

# ----------------------- Analytics State -----------------------

ALERT_THRESHOLD = 7
analytics_state = {
    "zone_occupancy": {},       # {zone_name: count}
    "events": deque(maxlen=10),
    "alerts": [],
    "chart_data": {}
}

# --- Streaming / processing (updates analytics_state["zone_occupancy"]) ---

def process_video_stream_streamlit(video_path: str, camera_id: Optional[int] = None, mode: str = "analytics"):
    if not video_path or not os.path.exists(video_path):
        return
    cap = cv2.VideoCapture(video_path)

    # Load zones
    zones = []
    red_flags = {}  # {zone_name: is_red}
    if camera_id:
        zdf = list_zones(camera_id)
        for _, r in zdf.iterrows():
            pts = json.loads(r["points_json"])
            zones.append((r["name"], pts))
            red_flags[r["name"]] = bool(r["is_red"])

    # Reset state
    if mode == "analytics":
        analytics_state["zone_occupancy"] = {name: 0 for name, _ in zones}
        analytics_state["chart_data"] = {name: deque(maxlen=30) for name, _ in zones}
        analytics_state["events"].clear()
        analytics_state["alerts"].clear()

    heatmap = None
    frame_idx = 0
    last_boxes = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if heatmap is None:
            h, w = frame.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)

        if frame_idx % DETECT_EVERY_N_FRAMES == 0 or frame_idx == 1:
            boxes, annotated = yolo_detect_persons(frame.copy())
            last_boxes = boxes
        else:
            annotated = frame.copy()
            for (x, y, w, h) in last_boxes:
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,0), 2)

        # Analytics by polygon zones (red/normal)
        if mode == "analytics" and zones:
            current = {name: 0 for name, _ in zones}
            for (x, y, w, h) in last_boxes:
                cx, cy = x + w//2, y + h//2
                cv2.circle(heatmap, (cx, cy), 20, 1, -1)
                for (zname, pts) in zones:
                    if point_in_polygon((cx, cy), pts):
                        current[zname] += 1
                        break

            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            cv2.addWeighted(heatmap_colored, 0.5, annotated, 0.5, 0, annotated)
            heatmap *= 0.95

            # draw polygons: red ones in red, normal in blue
            for (zname, pts) in zones:
                color = (0,0,255) if red_flags.get(zname, False) else (255,0,0)
                if len(pts) >= 2:
                    cv2.polylines(annotated, [np.array(pts, dtype=np.int32)], True, color, 3)
                    cv2.putText(annotated, zname, (pts[0][0], max(15, pts[0][1]-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            analytics_state["zone_occupancy"] = current
            for name, cnt in current.items():
                analytics_state["chart_data"][name].append(int(cnt))
            analytics_state["alerts"] = [
                f"ALERT: '{name}' exceeded threshold ({cnt}/{ALERT_THRESHOLD})!"
                for name, cnt in current.items() if cnt > ALERT_THRESHOLD
            ]

        ok2, buff = cv2.imencode(".jpg", annotated)
        if not ok2:
            continue
        yield buff.tobytes()

    cap.release()

def analytics_data_api_streamlit():
    return {
        "zone_occupancy": analytics_state.get("zone_occupancy", {}),
        "events": list(analytics_state.get("events", [])),
        "alerts": analytics_state.get("alerts", []),
        "chart_data": {k: list(v) for k, v in analytics_state.get("chart_data", {}).items()}
    }

# ----------------------- UI Pages --------------------------

def page_dashboard(user):
    st.title(APP_TITLE)
    #st.caption("Secure login • Red/Normal zones • Live analytics and alerts")

    # Logout option on dashboard page
    ui_logout_button()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        users_count = get_scalar("SELECT COUNT(*) FROM users")
        st.metric("Users", users_count)
    with col2:
        cams_count = get_scalar("SELECT COUNT(*) FROM cameras")
        st.metric("Cameras", cams_count)
    with col3:
        zones_count = get_scalar("SELECT COUNT(*) FROM zones")
        st.metric("Zones", zones_count)
    with col4:
        rows_count = get_scalar("SELECT COUNT(*) FROM counts")
        st.metric("Count Records", rows_count)

    df = get_counts_df(days=7)
    if df.empty:
        st.info("No counts yet. Use the People Count tab to run an analysis.")
        return

    st.subheader("People Count ")
    df_line = df.groupby(pd.Grouper(key="ts", freq="1min"))[["total_count"]].sum().reset_index()
    fig1 = px.line(df_line, x="ts", y="total_count", title="Total People Count (last 7 days)")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Density View ")
    fig2 = px.scatter(df, x="ts", y="img_density", color="source_kind", title="Image Density over time")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Zone Occupancy ")
    df_bar = df.copy()
    df_bar["zone_label"] = df_bar["zone_id"].fillna(-1).astype(int).astype(str)
    fig3 = px.bar(df_bar, x="zone_label", y="total_count", color="source_kind", title="Counts by Zone (saved records)")
    st.plotly_chart(fig3, use_container_width=True)

    st.download_button(
        "Download Counts CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"counts_{int(time.time())}.csv",
        mime="text/csv",
    )

def page_people_count(user):
    st.header("People Count & Zone Management")

    if not DRAWING_AVAILABLE:
        st.warning(
            "Drawable canvas is currently unavailable. If you're on a new Streamlit version, try 'pip install streamlit-drawable-canvas-fix' or keep the shim above and restart."
        )

    cams_df = list_cameras()
    cam_label_to_id = {f"#{row.id} {row.name}": int(row.id) for _, row in cams_df.iterrows()} if not cams_df.empty else {}
    cam_choice = st.selectbox("Select Camera", options=["(None)"] + list(cam_label_to_id.keys()))
    cam_id = cam_label_to_id.get(cam_choice)

    # Show existing zones table and basic edit controls
    zdf = list_zones(cam_id)
    if not zdf.empty:
        zdf_show = zdf[["id","name","is_red","created_at"]].copy()
        zdf_show["type"] = np.where(zdf_show["is_red"].astype(bool), "Red", "Normal")
        st.dataframe(zdf_show.drop(columns=["is_red"]))

        c1, c2, c3 = st.columns(3)
        with c1:
            zid = st.selectbox("Zone ID", [int(x) for x in zdf.id.tolist()])
        with c2:
            new_label = st.text_input("New Label")
            if st.button("Rename"):
                update_zone_label(int(zid), new_label or "Renamed Zone")
                st.success("Zone renamed.")
                st.rerun()
        with c3:
            flip_red = st.checkbox("Mark as Red Zone", value=bool(zdf.loc[zdf.id==zid,"is_red"].values[0]))
            if st.button("Save Red/Normal"):
                update_zone_is_red(int(zid), bool(flip_red))
                st.success("Zone type updated.")
                st.rerun()

        if st.button("Delete Selected Zone"):
            delete_zone(int(zid))
            st.success("Zone deleted.")
            st.rerun()

    st.subheader("Count People")
    source_kind = st.tabs(["Image (draw zones)", "Video File", "Live (Webcam/IP)"])

    # ---------- Image Tab ----------
    with source_kind[0]:
        up = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
        if up is not None:
            file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            boxes, annotated = yolo_detect_persons(frame.copy())
            count = len(boxes)
            density = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) / 255.0
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detections: {count}")

            if DRAWING_AVAILABLE and cam_id:
                bg_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Polygon tool selector (Draw vs Drag/Resize)
                mode = st.radio(
                    "Select Zone Tool",
                    ["Draw Polygon", "Drag/Resize Polygon"],
                    horizontal=True
                )
                drawing_mode = "polygon" if mode == "Draw Polygon" else "transform"

                canvas_res = st_canvas(
                    fill_color="rgba(255,0,0,0.3)",
                    stroke_width=3,
                    stroke_color="#FF0000",
                    background_image=bg_img,
                    update_streamlit=True,
                    height=min(720, bg_img.height),
                    width=min(1080, bg_img.width),
                    drawing_mode=drawing_mode,
                    key="canvas_img",
                )
                zone_name = st.text_input("Zone name", value="Zone from image", key="zone_name_img")
                mark_red = st.checkbox("Mark as Red Zone (danger area)", value=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Save polygon"):
                        try:
                            objs = canvas_res.json_data.get("objects", []) if canvas_res and canvas_res.json_data else []
                            poly = None
                            for o in reversed(objs):
                                if o.get("type") == "path" and "path" in o:
                                    poly = [(int(p[1]), int(p[2])) for p in o["path"] if isinstance(p, list) and len(p) >= 3]
                                    if poly:
                                        break
                            if poly and zone_name and cam_id:
                                save_zone(cam_id, zone_name, poly, is_red=bool(mark_red))
                                st.success(f"Zone saved ({'Red' if mark_red else 'Normal'}).")
                                st.rerun()
                            else:
                                st.error("No polygon found. Draw or select a polygon, then save.")
                        except Exception as e:
                            st.error(f"Polygon parse error: {e}")
                with c2:
                    if st.button("Save polygon (API-compat)"):
                        try:
                            if 'poly' in locals() and poly:
                                save_zone(cam_id, zone_name or "Zone", poly, is_red=bool(mark_red))
                                st.success("Zone saved via compat.")
                                st.rerun()
                            else:
                                st.error("No polygon available to save.")
                        except Exception as e:
                            st.error(f"Compat save error: {e}")

            if st.button("Save Count Record (Image)"):
                save_count(datetime.utcnow(), cam_id, None, count, density, "image", note=up.name)
                st.success("Saved count record.")

    # ---------- Video File Tab ----------
    with source_kind[1]:
        upv = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"], key="video_up")
        threshold = st.number_input("Alert threshold", min_value=1, max_value=1000, value=10)

        # --- Zone Editor from a captured frame of the uploaded video ---
        if upv is not None:
            #st.markdown("*Zone Editor (Video):* capture a frame → draw/drag polygon → save zone.")
            cve1, cve2 = st.columns(2)
            with cve1:
                snap_sec = st.number_input("Capture at second", min_value=0.0, max_value=6000.0, value=1.0, step=0.5)
            with cve2:
                edit_now = st.button("Open Zone Editor (from Video)")

            if edit_now:
                vb = upv.getvalue()
                frame_edit = grab_frame_from_video(vb, at_seconds=float(snap_sec))
                if frame_edit is None:
                    st.error("Could not capture a frame from the video.")
                else:
                    st.image(cv2.cvtColor(frame_edit, cv2.COLOR_BGR2RGB), caption="Captured frame for zone editing")
                    if DRAWING_AVAILABLE and cam_id:
                        bg_img = Image.fromarray(cv2.cvtColor(frame_edit, cv2.COLOR_BGR2RGB))
                        mode = st.radio(
                            "Zone Tool (Video)",
                            ["Draw Polygon", "Drag/Resize Polygon"],
                            horizontal=True, key="vid_zone_mode"
                        )
                        drawing_mode = "polygon" if mode == "Draw Polygon" else "transform"

                        canvas_res_v = st_canvas(
                            fill_color="rgba(255,0,0,0.3)",
                            stroke_width=3,
                            stroke_color="#FF0000",
                            background_image=bg_img,
                            update_streamlit=True,
                            height=min(720, bg_img.height),
                            width=min(1080, bg_img.width),
                            drawing_mode=drawing_mode,
                            key="canvas_video_img",
                        )
                        zone_name_v = st.text_input("Zone name", value="Zone from video", key="zone_name_video")
                        mark_red_v = st.checkbox("Mark as Red Zone (danger area)", value=True, key="mark_red_video")

                        if st.button("Save polygon (Video)"):
                            try:
                                objs = canvas_res_v.json_data.get("objects", []) if canvas_res_v and canvas_res_v.json_data else []
                                poly = None
                                for o in reversed(objs):
                                    if o.get("type") == "path" and "path" in o:
                                        poly = [(int(p[1]), int(p[2])) for p in o["path"] if isinstance(p, list) and len(p) >= 3]
                                        if poly:
                                            break
                                if poly and zone_name_v and cam_id:
                                    save_zone(cam_id, zone_name_v, poly, is_red=bool(mark_red_v))
                                    st.success(f"Zone saved from video ({'Red' if mark_red_v else 'Normal'}).")
                                    st.rerun()
                                else:
                                    st.error("No polygon found. Draw or select a polygon, then save.")
                            except Exception as e:
                                st.error(f"Polygon parse error (video): {e}")

        # --- Analytics processing for the uploaded video ---
        if upv is not None and st.button("Process with Analytics (zones/heatmap)"):
            tmp_path2 = f"./tmp_{int(time.time())}_{upv.name}"
            with open(tmp_path2, "wb") as f:
                f.write(upv.getvalue())
            frame_box2 = st.empty()
            st.info("Running analytics (Red/Normal zones)…")
            for i, jpg in enumerate(process_video_stream_streamlit(tmp_path2, camera_id=cam_id, mode="analytics")):
                nparr = np.frombuffer(jpg, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    frame_box2.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Frame {i+1}")
                time.sleep(0.01)
            try:
                os.remove(tmp_path2)
            except:
                pass
            st.success("Analytics complete.")

    # ---------- Live (Webcam/IP) Tab ----------
    with source_kind[2]:
        st.write("Connect to webcam or Paste URL")
        url = st.text_input("Source ", value="0")
        run = st.checkbox("Start stream")
        threshold = st.number_input("Live alert threshold", min_value=1, max_value=1000, value=15, key="th_live")

        # Zone Editor from a LIVE snapshot
       # st.markdown("*Zone Editor (Live):* take a snapshot → draw/drag polygon → save zone.")
        if st.button("Snapshot & Open Zone Editor"):
            frame_live = grab_frame_from_source(url)
            if frame_live is None:
                st.error("Could not capture a frame from the live source.")
            else:
                st.image(cv2.cvtColor(frame_live, cv2.COLOR_BGR2RGB), caption="Live snapshot for zone editing")
                if DRAWING_AVAILABLE and cam_id:
                    bg_img = Image.fromarray(cv2.cvtColor(frame_live, cv2.COLOR_BGR2RGB))
                    mode = st.radio(
                        "Zone Tool (Live)",
                        ["Draw Polygon", "Drag/Resize Polygon"],
                        horizontal=True, key="live_zone_mode"
                    )
                    drawing_mode = "polygon" if mode == "Draw Polygon" else "transform"

                    canvas_res_l = st_canvas(
                        fill_color="rgba(255,0,0,0.3)",
                        stroke_width=3,
                        stroke_color="#FF0000",
                        background_image=bg_img,
                        update_streamlit=True,
                        height=min(720, bg_img.height),
                        width=min(1080, bg_img.width),
                        drawing_mode=drawing_mode,
                        key="canvas_live_img",
                    )
                    zone_name_l = st.text_input("Zone name", value="Zone from live", key="zone_name_live")
                    mark_red_l = st.checkbox("Mark as Red Zone (danger area)", value=True, key="mark_red_live")

                    if st.button("Save polygon (Live)"):
                        try:
                            objs = canvas_res_l.json_data.get("objects", []) if canvas_res_l and canvas_res_l.json_data else []
                            poly = None
                            for o in reversed(objs):
                                if o.get("type") == "path" and "path" in o:
                                    poly = [(int(p[1]), int(p[2])) for p in o["path"] if isinstance(p, list) and len(p) >= 3]
                                    if poly:
                                        break
                            if poly and zone_name_l and cam_id:
                                save_zone(cam_id, zone_name_l, poly, is_red=bool(mark_red_l))
                                st.success(f"Zone saved from live snapshot ({'Red' if mark_red_l else 'Normal'}).")
                                st.rerun()
                            else:
                                st.error("No polygon found. Draw or select a polygon, then save.")
                        except Exception as e:
                            st.error(f"Polygon parse error (live): {e}")

        # Existing live streaming & analytics
        frame_box = st.empty()
        count_chart = st.empty()
        counts_series = []
        if run:
            try:
                src = 0 if url.strip() == "0" else url.strip()
                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    st.error("Could not open source.")
                else:
                    last_plot = time.time()
                    frame_idx = 0
                    last_boxes = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_idx += 1
                        if frame_idx % DETECT_EVERY_N_FRAMES == 0 or frame_idx == 1:
                            boxes, annotated = yolo_detect_persons(frame.copy())
                            last_boxes = boxes
                        else:
                            annotated = frame.copy()
                            for (x, y, w, h) in last_boxes:
                                cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
                        c = len(last_boxes)
                        density = float(np.mean(frame)/255.0)
                        frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Live count={c}")
                        save_count(datetime.utcnow(), cam_id, None, c, density, "live", note=str(url))
                        counts_series.append({"ts": datetime.utcnow(), "count": c})
                        if c >= threshold:
                            st.warning(f"ALERT: Live count {c} ≥ {threshold}")
                        if time.time() - last_plot > 1.0 and len(counts_series) > 1:
                            dfc = pd.DataFrame(counts_series)
                            fig = px.line(dfc, x="ts", y="count", title="Live Count")
                            count_chart.plotly_chart(fig, use_container_width=True)
                            last_plot = time.time()
                        time.sleep(0.02)
                cap.release()
            except Exception as e:
                st.error(f"Live error: {e}")

# ----------------------- Admin Page -----------------------

def page_admin(user):
    if not is_admin(user):
        st.error("Admin only.")
        return

    st.header("Admin Panel & Analytics")
    tabs = st.tabs(["Users", "Cameras", "Logs", "Reports"])

    with tabs[0]:
        with get_conn() as conn:
            udf = pd.read_sql_query("SELECT id,name,email,role,created_at FROM users ORDER BY id DESC", conn)
        st.dataframe(udf)
        st.subheader("Create/Update User")
        with st.form("admin_user"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            pw = st.text_input("Password ", type="password")
            role = st.selectbox("Role", ["user","admin"])
            submit = st.form_submit_button("Save User")
        if submit:
            with get_conn() as conn:
                row = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
                if row:
                    if pw:
                        conn.execute("UPDATE users SET name=?, role=?, password_hash=? WHERE email=?", (name, role, hash_password(pw), email))
                    else:
                        conn.execute("UPDATE users SET name=?, role=? WHERE email=?", (name, role, email))
                else:
                    conn.execute("INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)", (name, email, hash_password(pw or "changeme123"), role, datetime.utcnow().isoformat()))
                conn.commit()
            st.success("User saved.")

    with tabs[1]:
        st.subheader("Manage Cameras")
        cdf = list_cameras()
        st.dataframe(cdf)
        with st.form("add_cam"):
            cname = st.text_input("Camera Name")
            curl = st.text_input("Source URL (0 for default webcam or RTSP/HTTP)")
            sub = st.form_submit_button("Add Camera")
        if sub and cname and curl:
            create_camera(cname, curl, user.get("email"))
            st.success("Camera added.")
            st.rerun()
        if not cdf.empty:
            del_id = st.selectbox("Delete Camera", options=["-"] + [int(i) for i in cdf.id.tolist()])
            if del_id != "-" and st.button("Delete Selected Camera"):
                delete_camera(int(del_id), user.get("email"))
                st.success("Camera deleted.")
                st.rerun()

    with tabs[2]:
        with get_conn() as conn:
            ldf = pd.read_sql_query("SELECT * FROM logs ORDER BY ts DESC LIMIT 1000", conn)
        st.dataframe(ldf)
        st.download_button("Download Logs CSV", ldf.to_csv(index=False).encode("utf-8"), file_name="logs.csv")

    with tabs[3]:
        st.subheader("Generate / Download Reports")
        days = st.slider("Include last N days", 1, 90, 7)
        df = get_counts_df(days)
        if df.empty:
            st.info("No counts yet.")
        else:
            by_cam = df.groupby("camera_id")["total_count"].sum().reset_index().rename(columns={"total_count":"sum_count"})
            fig = px.bar(by_cam, x="camera_id", y="sum_count", title="Counts by Camera (Report)")
            st.plotly_chart(fig, use_container_width=True)
            ts_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(REPORTS_DIR, f"report_{ts_name}.csv")
            df.to_csv(path, index=False)
            st.success(f"Report saved: {path}")
            st.download_button("Download This Report", df.to_csv(index=False).encode("utf-8"), file_name=f"report_{ts_name}.csv")

# ----------------------- App Router -----------------------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="👥", layout="wide")
    apply_theme()

    user = get_current_user()
    if not user:
        # Removed Auth radio from sidebar — show tabs in MAIN content instead
        tabs = st.tabs(["Login", "Register"])
        with tabs[0]:
            ui_login()
        with tabs[1]:
            ui_register()
        return

    # Sidebar remains for logged-in navigation
    st.sidebar.success(f"Hello, {user.get('name')} ({user.get('role')})")
    choice = st.sidebar.radio("Go to", ["Analytics", "People Count", "Admin" if is_admin(user) else "(Admin only)"])
    st.sidebar.divider()

    if choice == "Analytics":
        page_dashboard(user)   # logout button is on this page
    elif choice == "People Count":
        page_people_count(user)
    elif choice == "Admin" and is_admin(user):
        page_admin(user)
    else:
        st.info("You do not have access to this page.")

if __name__ == "__main__":
    main()
