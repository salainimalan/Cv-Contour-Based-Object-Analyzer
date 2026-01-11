import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Contour Object Analyzer",
    page_icon="Analysis",
    layout="wide"
)

st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(135deg, #0b1220, #05080f);
    color: #eaeaf0;
}

/* Remove Streamlit clutter */
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 1.5rem; }

/* Typography */
h1 { color: #4fc3f7; font-size: 40px; }
h2 { color: #90caf9; }
h3 { color: #bbdefb; }

/* Glass panels */
.panel {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(16px);
    border-radius: 18px;
    padding: 24px;
    border: 1px solid rgba(79,195,247,0.25);
    box-shadow: 0 0 30px rgba(79,195,247,0.15);
    margin-bottom: 22px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: rgba(10,25,45,0.7);
    border: 2px dashed #4fc3f7;
    border-radius: 14px;
    padding: 25px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4fc3f7, #1976d2);
    color: #05121c;
    font-weight: 700;
    border-radius: 25px;
    padding: 10px 28px;
    border: none;
    box-shadow: 0 0 12px rgba(79,195,247,0.6);
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(79,195,247,0.1);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(79,195,247,0.3);
}

/* Data table */
.stDataFrame {
    background: rgba(0,0,0,0.4);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.12);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="panel">
    <h1 style="text-align:center;">Contour Object Analyzer</h1>
    <h3 style="text-align:center;">Geometric Shape Detection and Measurement</h3>
    <p style="text-align:center; color:#9aa7b2; font-size:14px;">
        Developed by <b>Salai Nimalan (23MIA1064)</b>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="panel">
    <h2>Upload Image</h2>
    <p style="color:#9aa7b2;">Upload an image containing geometric objects to analyze their contours.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

def resize_for_display(img):
    h, w = img.shape[:2]
    scale = min(500 / w, 380 / h)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        v = len(approx)

        if v == 3:
            shape = "Triangle"
        elif v == 4:
            w, h = cv2.minAreaRect(cnt)[1]
            if min(w,h) == 0:
                continue
            shape = "Square" if max(w,h)/min(w,h) < 1.15 else "Rectangle"
        elif v == 5:
            shape = "Pentagon"
        elif v == 6:
            shape = "Hexagon"
        else:
            shape = "Circle" if (4*np.pi*area)/(perimeter**2) > 0.8 else "Irregular"

        cv2.drawContours(image, [approx], -1, (0, 220, 120), 3)
        cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 2)

        results.append([shape, round(area,2), round(perimeter,2)])

    return image, results

if uploaded_file:
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(img_bytes, 1)

    processed, data = detect_shapes(image.copy())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='panel'><h3>Original Image</h3>", unsafe_allow_html=True)
        st.image(resize_for_display(image), channels="BGR", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='panel'><h3>Detected Contours</h3>", unsafe_allow_html=True)
        st.image(resize_for_display(processed), channels="BGR", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if data:
        df = pd.DataFrame(data, columns=["Shape", "Area", "Perimeter"])

        st.markdown("<div class='panel'><h2>Measurements</h2></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Objects", len(df))
        c2.metric("Shape Types", df["Shape"].nunique())
        c3.metric("Max Area", f"{df['Area'].max():.1f}")

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload an image to start contour analysis.")

