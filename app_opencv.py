# app_opencv.py
import streamlit as st
import cv2
import numpy as np
import ezdxf
import tempfile

def image_to_contours(img, n_colors=6, min_area=50, simplify=2.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    Z = lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(Z, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten().reshape(img.shape[:2])

    contours_by_color = {}
    for i in range(len(centers)):
        mask = np.uint8(labels == i) * 255
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        good = []
        for c in cnts:
            if cv2.contourArea(c) > min_area:
                eps = simplify * cv2.arcLength(c, True) / 1000.0
                approx = cv2.approxPolyDP(c, eps, True)
                good.append(approx)
        if good:
            contours_by_color[f"color_{i}"] = good
    return contours_by_color

def save_dxf(contours_by_color, path, scale=1.0):
    doc = ezdxf.new()
    msp = doc.modelspace()
    for layer, cnts in contours_by_color.items():
        if layer not in doc.layers:
            doc.layers.new(name=layer)
        for cnt in cnts:
            pts = [(p[0][0]*scale, -p[0][1]*scale) for p in cnt]
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})
    doc.saveas(path)

st.title("App 1 – OpenCV contornos")
f = st.file_uploader("Subí una imagen", ["png","jpg","jpeg"])
n_colors = st.slider("N colores", 2, 12, 6)
simplify = st.slider("Simplificación", 0.5, 5.0, 2.0)
min_area = st.slider("Área mínima", 10, 500, 80)
ancho_mm = st.number_input("Ancho en mm", 10.0, 1000.0, 100.0)

if f:
    data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, 1)
    h, w = img.shape[:2]
    scale = ancho_mm / w
    st.write(f"Tamaño final: {ancho_mm:.1f}mm x {h*scale:.1f}mm")
    cnts = image_to_contours(img, n_colors, min_area, simplify)
    preview = img.copy()
    for c in cnts.values():
        cv2.drawContours(preview, c, -1, (0,255,0), 1)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
    if st.button("Exportar DXF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            save_dxf(cnts, tmp.name, scale)
            st.download_button("Descargar DXF", open(tmp.name,"rb"), file_name="opencv_out.dxf")
