import streamlit as st
import cv2
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Vision Lab",
    page_icon="🧠",
    layout="wide"
)

# ---------------- FUTURISTIC UI ----------------
st.markdown("""
<style>

/* ----- Global App ----- */
.stApp {
    background: radial-gradient(circle at top, #0f2027, #0a0f14 70%);
    color: #e0e0e0;
}

/* Remove default white blocks */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* ----- Headings ----- */
h1, h2, h3 {
    color: #00e5ff;
    font-weight: 700;
}

/* ----- Glass Cards ----- */
.glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 0 25px rgba(0,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}

/* ----- Upload box ----- */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, #0a1f2d, #06131d);
    border-radius: 15px;
    border: 2px dashed #00e5ff;
    padding: 20px;
}

/* ----- Buttons ----- */
.stButton > button {
    background: linear-gradient(135deg, #00e5ff, #00b0ff);
    color: black;
    border-radius: 30px;
    padding: 10px 24px;
    font-weight: 700;
    border: none;
    box-shadow: 0 0 15px rgba(0,229,255,0.6);
}

/* ----- Metrics ----- */
[data-testid="stMetric"] {
    background: rgba(0,229,255,0.08);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(0,229,255,0.2);
}

/* ----- Dataframe ----- */
.stDataFrame {
    background: rgba(0,0,0,0.4);
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="glass">
    <h1 style="font-size:42px; text-align:center;">🧠 AI Vision Lab</h1>
    <h3 style="text-align:center; color:#90caf9;">Contour-Based Object Intelligence System</h3>
    <p style="text-align:center; color:#aaa; font-size:14px;">
        Developed by <b>Salai Nimalan (23MIA1064)</b> | Computer Vision • Geometry • AI
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ----------------
st.markdown("""
<div class="glass">
    <h2>📤 Upload Visual Data</h2>
    <p style="color:#aaa; font-size:14px;">Supported formats: JPG, PNG. Upload an image containing geometric shapes.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop image here",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

# ---------------- UTILS ----------------
def resize_for_display(img):
    h, w = img.shape[:2]
    MAX_W = 450
    MAX_H = 350
    scale = min(MAX_W / w, MAX_H / h)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# ---------------- SHAPE DETECTION ----------------
def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        vertices = len(approx)

        shape = "Unknown"

        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            shape = "Square" if aspect < 1.15 else "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        elif vertices == 6:
            shape = "Hexagon"
        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            shape = "Circle" if circularity > 0.8 else "Irregular"

        cv2.drawContours(image, [approx], -1, (0, 255, 255), 3)
        cv2.putText(
            image,
            shape,
            (approx[0][0][0], approx[0][0][1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            2
        )

        results.append([shape, round(area, 2), round(perimeter, 2)])

    return image, results

# ---------------- MAIN ----------------
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    processed_img, data = detect_shapes(image.copy())

    image_disp = resize_for_display(image)
    processed_disp = resize_for_display(processed_img)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='glass'><h3>📷 Raw Input</h3>", unsafe_allow_html=True)
        st.image(image_disp, channels="BGR", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass'><h3>🧩 AI Detection Output</h3>", unsafe_allow_html=True)
        st.image(processed_disp, channels="BGR", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if data:
        df = pd.DataFrame(data, columns=["Shape", "Area (px²)", "Perimeter (px)"])

        st.markdown("""
        <div class="glass">
            <h2>📊 Object Intelligence Report</h2>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Objects", len(df))
        with col2:
            st.metric("Unique Shapes", df["Shape"].nunique())
        with col3:
            st.metric("Largest Area", f"{df['Area (px²)'].max():.1f}")

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("📥 Upload an image to begin AI-based contour analysis.")
