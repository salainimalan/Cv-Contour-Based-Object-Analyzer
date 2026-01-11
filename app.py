import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("🔷 Shape & Contour Analyzer")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def show_image(image, caption):
    if len(image.shape) == 2:
        st.image(image, caption=caption)
    else:
        st.image(image, caption=caption, channels="BGR")

if uploaded:
    img = Image.open(uploaded)
    img = np.array(img)

    # Convert to grayscale safely
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # ---- Noise reduction ----
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # ---- Edge detection ----
    edges = cv2.Canny(blur, 40, 120)

    # ---- Strengthen edges ----
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ---- Find contours ----
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    results = []
    count = 0

    h, w = gray.shape
    min_area = 0.001 * h * w   # remove tiny noise

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)

        # ---- Circularity ----
        circularity = 4 * np.pi * area / (peri * peri + 1e-6)

        # ---- Shape classification ----
        if circularity > 0.82:
            shape = "Circle"
        else:
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                x,y,w1,h1 = cv2.boundingRect(approx)
                ar = w1 / float(h1)
                shape = "Square" if 0.90 <= ar <= 1.10 else "Rectangle"
            elif len(approx) == 5:
                shape = "Pentagon"
            else:
                shape = "Polygon"

        count += 1

        # ---- Draw results ----
        cv2.drawContours(output, [cnt], -1, (0,255,0), 2)
        x,y,w1,h1 = cv2.boundingRect(cnt)
        cv2.putText(output, shape, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        results.append((shape, int(area), int(peri)))

    col1, col2 = st.columns(2)
    with col1:
        show_image(img, "Original Image")
    with col2:
        show_image(output, "Detected Objects")

    st.subheader(f"Detected Objects: {count}")
    st.subheader("📊 Feature Table")

    for i, r in enumerate(results):
        st.write(f"Object {i+1} → Shape: {r[0]} | Area: {r[1]} | Perimeter: {r[2]}")
