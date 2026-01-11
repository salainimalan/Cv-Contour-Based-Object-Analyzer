import streamlit as st
import cv2
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Contour Object Analyzer",
    page_icon="📐",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #e3f2fd;
}
h1, h2, h3 {
    color: #1976d2;
}
.small {
    font-size: 13px;
    color: #424242;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.stButton>button {
    background-color: #1976d2;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title(" DA1-Contour-Based Object Analyzer")
st.markdown("**By: 23MIA1064 - SALAI NIMALAN**")
st.divider()

# ---------------- UPLOAD SECTION ----------------
st.header("Upload Your Image")
uploaded_file = st.file_uploader(
    "Choose an image file (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    help="Upload an image to analyze shapes and contours."
)

# ---------------- UTILS ----------------
def resize_for_display(img):
    h, w = img.shape[:2]

    MAX_W = 400
    MAX_H = 300

    scale = min(MAX_W / w, MAX_H / h)

    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    return img

# ---------------- SHAPE DETECTION ----------------
def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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

        cv2.drawContours(image, [approx], -1, (80, 200, 120), 3)
        cv2.putText(
            image,
            shape,
            (approx[0][0][0], approx[0][0][1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,   # 👈 reduced font size
            (60, 120, 220),
            2
        )

        results.append([shape, round(area, 2), round(perimeter, 2)])

    return image, results

# ---------------- MAIN ----------------
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    processed_img, data = detect_shapes(image.copy())

    # Resize images for display
    image_disp = resize_for_display(image)
    processed_disp = resize_for_display(processed_img)

    # Display images side by side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(" Original Image")
        st.image(image_disp, channels="BGR", use_container_width=True)

    with col2:
        st.subheader(" Analyzed Image")
        st.image(processed_disp, channels="BGR", use_container_width=True)

    if data:
        df = pd.DataFrame(data, columns=["Shape", "Area", "Perimeter"])

        st.divider()
        st.header(" Analysis Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Objects Detected", len(df))
        with col2:
            st.metric("Unique Shape Types", df["Shape"].nunique())
        with col3:
            st.metric("Largest Area", f"{df['Area'].max():.1f} px²")

        st.subheader("Detailed Shape Measurements")
        st.dataframe(df, use_container_width=True)

else:
    st.info(" Please upload an image above to start the analysis.")