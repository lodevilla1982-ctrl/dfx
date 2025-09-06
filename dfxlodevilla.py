import streamlit as st
import cv2
import numpy as np
import ezdxf
from sklearn.cluster import KMeans
import tempfile
import matplotlib.pyplot as plt

def image_to_contours(img, mode="edges", n_colors=6):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mode == "edges":
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif mode == "colors":
        Z = img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=n_colors, n_init="auto").fit(Z)
        labels = kmeans.labels_.reshape(img.shape[:2])
        contours = []
        for i in range(n_colors):
            mask = np.uint8(labels == i) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(cnts)
    else:
        contours = []
    return contours

def save_dxf(contours, output_path):
    doc = ezdxf.new()
    msp = doc.modelspace()
    for cnt in contours:
        if len(cnt) > 2:
            points = [(p[0][0], p[0][1]) for p in cnt]
            msp.add_lwpolyline(points, close=True)
    doc.saveas(output_path)

# -------------------
# STREAMLIT APP
# -------------------
st.title("Imagen a DXF")

uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
mode = st.radio("Modo de vectorizado:", ["edges", "colors"])
n_colors = st.slider("Número de colores (solo para 'colors')", 2, 12, 5)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    contours = image_to_contours(img, mode=mode, n_colors=n_colors)

    # Preview
    preview = img.copy()
    cv2.drawContours(preview, contours, -1, (0, 255, 0), 1)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Preview del vector")

    if st.button("Generar DXF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            save_dxf(contours, tmp.name)
            st.success("DXF generado con éxito")
            st.download_button("Descargar DXF", open(tmp.name, "rb"), file_name="salida.dxf")
