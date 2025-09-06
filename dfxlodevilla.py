
import streamlit as st
import cv2
import numpy as np
import ezdxf
import tempfile

# ============== utilidades de color ==================
def to_lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

def bilateral_if(img, enabled=True):
    if not enabled:
        return img
    # Suaviza preservando bordes
    return cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=5)

def kmeans_lab(img_bgr, k, seed=0):
    lab = to_lab(img_bgr)
    Z = lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # reproducible
    np.random.seed(seed)
    compactness, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(lab.shape[:2])
    return labels, centers

def deltaE76(c1, c2):
    # Euclidean distance in LAB
    return float(np.linalg.norm(c1 - c2))

def merge_clusters_by_deltaE(labels, centers, threshold=5.0):
    # agrupa centros de color cercanos usando enlace simple (orden-dependiente pero efectivo)
    n = len(centers)
    groups = []
    assigned = np.full(n, -1, dtype=int)
    gidx = 0
    for i in range(n):
        if assigned[i] != -1:
            continue
        assigned[i] = gidx
        base = centers[i]
        for j in range(i+1, n):
            if assigned[j] != -1:
                continue
            if deltaE76(base, centers[j]) < threshold:
                assigned[j] = gidx
        gidx += 1
    # remap labels
    map_old_new = {i: assigned[i] for i in range(n)}
    new_labels = np.vectorize(map_old_new.get)(labels)
    # calcular nuevos centros promedio
    new_n = gidx
    new_centers = np.zeros((new_n, 3), dtype=np.float32)
    counts = np.zeros(new_n, dtype=np.int64)
    for old_idx, new_idx in map_old_new.items():
        new_centers[new_idx] += centers[old_idx]
        counts[new_idx] += 1
    for i in range(new_n):
        if counts[i] > 0:
            new_centers[i] /= counts[i]
    return new_labels, new_centers

# ============== contornos ==================
def find_contours_with_holes(mask):
    # CCOMP devuelve jerarquía de 2 niveles: contornos externos y sus agujeros
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    result = []
    for i, h in enumerate(hierarchy):
        parent = h[3]
        # incluimos TODOS los contornos (externos y agujeros). Para DXF los exportamos como loops separados.
        result.append(cnts[i])
    return result, hierarchy, cnts

def simplify_contour(cnt, simplify_pct=1.5):
    # epsilon basado en porcentaje del perímetro
    epsilon = (simplify_pct / 100.0) * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return approx

def image_to_contours_colors(img_bgr, n_colors=6, seed=0, merge_deltaE=5.0, min_area=50, simplify_pct=1.5, bilateral=True, close_kernel=3):
    img = img_bgr.copy()
    if bilateral:
        img = bilateral_if(img, True)

    labels, centers_lab = kmeans_lab(img, n_colors, seed=seed)
    if merge_deltaE > 0:
        labels, centers_lab = merge_clusters_by_deltaE(labels, centers_lab, threshold=merge_deltaE)

    contours_all = []
    h, w = labels.shape
    kernel = np.ones((close_kernel, close_kernel), np.uint8) if close_kernel > 1 else None

    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = np.uint8(labels == lbl) * 255
        if kernel is not None:
            # cerrar huecos pequeños y luego abrir para limpiar ruido
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cnts, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for cnt in cnts:
            if cv2.contourArea(cnt) >= float(min_area):
                contours_all.append(simplify_contour(cnt, simplify_pct))
    return contours_all

def image_to_contours_edges(img_bgr, min_area=50, simplify_pct=1.5, block_size=11, C=2, close_kernel=3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # umbral adaptativo en canal L mejora textos y logos
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size | 1, C)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_all = []
    for cnt in cnts:
        if cv2.contourArea(cnt) >= float(min_area):
            contours_all.append(simplify_contour(cnt, simplify_pct))
    return contours_all

# ============== DXF ==================
def contours_to_dxf(contours, dxf_path, img_shape):
    h, w = img_shape[:2]
    doc = ezdxf.new("R2010")
    # 4 => millimeters (solo como referencia al importar en CAD)
    try:
        doc.header["$INSUNITS"] = 4
    except Exception:
        pass
    msp = doc.modelspace()

    for cnt in contours:
        if len(cnt) < 3:
            continue
        # OpenCV tiene (0,0) arriba-izquierda. DXF usa (0,0) abajo-izquierda
        pts = [(float(p[0][0]), float(h - p[0][1])) for p in cnt]
        msp.add_lwpolyline(pts, close=True)

    doc.saveas(dxf_path)

# ============== UI ==================
st.set_page_config(page_title="Imagen a DXF mejorado", layout="centered")
st.title("🖼️ Imagen → DXF (mejorado)")
st.caption("Segmentación por colores en LAB con fusión ΔE, contornos con agujeros y DXF listo para Fusion 360.")

uploaded_file = st.file_uploader("Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"])

mode = st.radio("Modo de vectorizado:", ["colors (recomendado)", "edges"], index=0)
col1, col2 = st.columns(2)

with col1:
    min_area = st.slider("Filtrar áreas pequeñas (px²)", 10, 2000, 80, step=10)
    simplify_pct = st.slider("Simplificar contornos (%)", 0.5, 5.0, 1.5, step=0.1)
    close_kernel = st.slider("Cierre/limpieza morfológica (kernel)", 1, 9, 3, step=2)

with col2:
    seed = st.number_input("Semilla (reproducible)", value=0, step=1)
    if mode.startswith("colors"):
        n_colors = st.slider("Número de colores", 2, 12, 7)
        merge_deltaE = st.slider("Fusionar colores similares (ΔE LAB)", 0, 30, 6)
        bilateral = st.checkbox("Suavizado bilateral (anti-ruido)", value=True)
    else:
        n_colors = None
        merge_deltaE = None
        bilateral = False

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if mode.startswith("colors"):
        contours = image_to_contours_colors(
            img, n_colors=n_colors, seed=seed, merge_deltaE=merge_deltaE,
            min_area=min_area, simplify_pct=simplify_pct, bilateral=bilateral,
            close_kernel=close_kernel
        )
    else:
        contours = image_to_contours_edges(
            img, min_area=min_area, simplify_pct=simplify_pct, close_kernel=close_kernel
        )

    # Preview
    preview = img.copy()
    if len(contours) > 0:
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 1)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Preview del vector", use_column_width=True)

    if st.button("Generar DXF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            contours_to_dxf(contours, tmp.name, img.shape)
            st.success("✅ DXF generado con éxito")
            st.download_button("⬇️ Descargar DXF", open(tmp.name, "rb"), file_name="salida.dxf")
