import streamlit as st
import cv2
import numpy as np
import ezdxf
import tempfile

# -------------------
# FUNCIONES
# -------------------

def image_to_contours(img, n_colors=6, simplify_eps=1.5, min_area=50):
    """Convierte la imagen a contornos segmentando por colores."""
    # Convertir a LAB para mejor segmentación perceptual
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    Z = lab.reshape((-1, 3)).astype(np.float32)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(Z, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    labels = labels.flatten().reshape(img.shape[:2])
    contours_by_color = {}

    for i in range(len(centers)):
        mask = np.uint8(labels == i) * 255
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        final_cnts = []
        for c in cnts:
            if cv2.contourArea(c) > min_area:
                # Epsilon bajo para mantener curvas
                eps = simplify_eps * cv2.arcLength(c, True) / 1000.0
                approx = cv2.approxPolyDP(c, eps, True)
                final_cnts.append(approx)
        if final_cnts:
            contours_by_color[f"color_{i}"] = final_cnts

    return contours_by_color


def save_dxf(contours_by_color, output_path, scale=1.0):
    """Exporta contornos como SPLINE a DXF con capas por color."""
    doc = ezdxf.new()
    msp = doc.modelspace()

    for layer, cnts in contours_by_color.items():
        if layer not in doc.layers:
            doc.layers.new(name=layer)
        for cnt in cnts:
            if len(cnt) > 2:
                points = [(float(p[0][0]) * scale, -float(p[0][1]) * scale) for p in cnt]
                # Usar SPLINE para curvas suaves
                try:
                    msp.add_spline(points, dxfattribs={"layer": layer})
                except Exception:
                    # Fallback a polilínea cerrada si falla spline
                    msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})

    doc.saveas(output_path)

# -------------------
# STREAMLIT APP
# -------------------
st.title("🖼️ Imagen → DXF (v3 final, escalado en mm y splines)")

uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
n_colors = st.slider("Número de colores", 2, 12, 6)
simplify_eps = st.slider("Simplificar contornos (menor = más detalle)", 0.5, 5.0, 1.5)
min_area = st.slider("Filtrar áreas pequeñas (px²)", 10, 500, 80)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Configuración de escala en mm
    st.markdown(f"**Tamaño original:** {w}px × {h}px")
    ancho_mm = st.number_input("Ancho final en mm", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
    escala = ancho_mm / w
    alto_mm = h * escala
    st.markdown(f"📏 Resultado: {ancho_mm:.1f}mm × {alto_mm:.1f}mm  (escala {escala:.4f} mm/px)")

    contours_by_color = image_to_contours(img, n_colors=n_colors,
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
            save_dxf(contours_by_color, tmp.name, scale=escala)
            st.success("✅ DXF generado con splines y escalado real en mm")
            st.download_button("⬇️ Descargar DXF", open(tmp.name, "rb"), file_name="salida_v3.dxf")
