import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🔷 Shape & Contour Analyzer")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    count = 0

    data = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:   # remove small noise
            count += 1

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                shape = "Rectangle"
            elif len(approx) == 5:
                shape = "Pentagon"
            else:
                shape = "Circle"

            cv2.drawContours(output, [cnt], -1, (0,255,0), 2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.putText(output, shape, (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            data.append((shape, int(area), int(peri)))

    st.subheader("Detected Objects: " + str(count))
    st.image(output, channels="BGR")

    st.subheader("Feature Table")
    for i, d in enumerate(data):
        st.write(f"Object {i+1}: Shape = {d[0]}, Area = {d[1]}, Perimeter = {d[2]}")

