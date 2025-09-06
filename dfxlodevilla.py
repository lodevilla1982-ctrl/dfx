import streamlit as st
import cv2
import numpy as np
import ezdxf
import tempfile

# -------------------
# FUNCIONES
# -------------------

def image_to_contours(img, mode="edges", n_colors=6):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mode == "edges":
        # Threshold adaptativo (mejor para logos y letras que Canny)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

    elif mode == "colors":
        # Reducción de colores usando cv2.kmeans
        Z = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            Z, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        labels = labels.flatten().reshape(img.shape[:2])

        contours = []
        for i in range(n_colors):
            mask = np.uint8(labels == i) * 255
            # Morfología → cerrar huecos
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours.extend(cnts)
    else:
        contours = []

    # Filtrar ruido (descartar áreas muy pequeñas)
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    # Simplificar contornos
    simplified = [cv2.approxPolyDP(cnt, 1.5, True) for cnt in filtered]

    return simplified


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
st.title("🖼️ Imagen a DXF")

uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
mode = st.radio("Modo de vectorizado:", ["edges", "colors"])
n_colors = st.slider("Número de colores (solo en modo 'colors')", 2, 12, 6)

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
            st.success("✅ DXF generado con éxito")
            st.download_button("⬇️ Descargar DXF", open(tmp.name, "rb"), file_name="salida.dxf")
