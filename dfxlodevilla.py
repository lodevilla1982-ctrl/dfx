import streamlit as st
import cv2
import numpy as np
import ezdxf
import tempfile
from skimage import color

# -------------------
# FUNCIONES
# -------------------

def merge_similar_colors(centers, delta_e=4.0):
    """Fusiona colores cercanos en LAB por ΔE."""
    merged = []
    used = set()
    for i, c in enumerate(centers):
        if i in used:
            continue
        group = [c]
        for j, d in enumerate(centers):
            if j != i and j not in used:
                dist = np.linalg.norm(c - d)
                if dist < delta_e:
                    group.append(d)
                    used.add(j)
        merged.append(np.mean(group, axis=0))
    return np.array(merged)


def image_to_contours(img, n_colors=6, delta_e=0, simplify_eps=1.5, min_area=50):
    """Segmenta por colores y devuelve contornos jerárquicos."""
    # Convertir a LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    Z = lab.reshape((-1, 3)).astype(np.float32)

    # K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(Z, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Fusionar colores similares (opcional)
    if delta_e > 0:
        centers = merge_similar_colors(centers, delta_e=delta_e)

    labels = labels.flatten().reshape(img.shape[:2])
    contours_by_color = {}

    for i in range(len(centers)):
        mask = np.uint8(labels == i) * 255
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        final_cnts = []
        for c in cnts:
            if cv2.contourArea(c) > min_area:
                approx = cv2.approxPolyDP(c, simplify_eps, True)
                final_cnts.append(approx)
        if final_cnts:
            contours_by_color[f"color_{i}"] = final_cnts

    return contours_by_color


def save_dxf(contours_by_color, output_path):
    """Exporta contornos a DXF con capas por color."""
    doc = ezdxf.new()
    msp = doc.modelspace()

    for layer, cnts in contours_by_color.items():
        if layer not in doc.layers:
            doc.layers.new(name=layer)
        for cnt in cnts:
            if len(cnt) > 2:
                points = [(int(p[0][0]), -int(p[0][1])) for p in cnt]  # Y invertido
                msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})

    doc.saveas(output_path)


# -------------------
# STREAMLIT APP
# -------------------
st.title("🖼️ Imagen → DXF (v2 con capas)")

uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
n_colors = st.slider("Número de colores", 2, 12, 6)
delta_e = st.slider("Fusionar colores similares (ΔE LAB)", 0, 20, 0)
simplify_eps = st.slider("Simplificar contornos (px)", 0.5, 5.0, 1.5)
min_area = st.slider("Filtrar áreas pequeñas (px²)", 10, 500, 80)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    contours_by_color = image_to_contours(img, n_colors=n_colors,
                                          delta_e=delta_e,
                                          simplify_eps=simplify_eps,
                                          min_area=min_area)

    # Preview
    preview = img.copy()
    for cnts in contours_by_color.values():
        cv2.drawContours(preview, cnts, -1, (0, 255, 0), 1)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
             caption="Preview del vector", use_container_width=True)

    if st.button("Generar DXF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            save_dxf(contours_by_color, tmp.name)
            st.success("✅ DXF generado con capas por color")
            st.download_button("⬇️ Descargar DXF", open(tmp.name, "rb"), file_name="salida_v2.dxf")
