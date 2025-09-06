# evdualmode.py
import streamlit as st
import cv2
import numpy as np
import ezdxf
import tempfile
from math import ceil

# try import skimage; if missing, we'll fallback to OpenCV-only mode
try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

st.set_page_config(page_title="Imagen → DXF (v-final)", layout="centered")

# ---------------- utility: smoothing and decimation ----------------
def smooth_contour(pts, window=7):
    """Moving-average smoothing on closed contour pts (Nx2)."""
    N = len(pts)
    if N < 3 or window <= 1:
        return pts
    w = int(window)
    # wrap pad
    pad = w//2
    xp = np.concatenate([pts[-pad:], pts, pts[:pad]])
    # moving average
    cumsum = np.cumsum(xp, axis=0)
    sm = (cumsum[w:] - cumsum[:-w]) / w
    # ensure same length
    return sm

def decimate_contour(pts, max_points=800):
    N = len(pts)
    if N <= max_points:
        return pts
    step = int(np.ceil(N / max_points))
    return pts[::step]

# ---------------- segmentation & contour extraction ----------------
def segment_kmeans_lab(img_bgr, n_colors=6, attempts=10):
    """Return labels (H x W) from kmeans in LAB color space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    Z = lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _ret, labels, centers = cv2.kmeans(Z, n_colors, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(img_bgr.shape[:2]), centers

def masks_to_contours_skimage(mask):
    """mask: 2D uint8 (0/255). Returns list of contours (each Nx2 float) in x,y coords."""
    # skimage.measure.find_contours returns coords as (row, col)
    raw = measure.find_contours(mask.astype(np.uint8), level=127.5)
    contours = []
    for r in raw:
        # r[:,0] -> row (y), r[:,1] -> col (x)
        pts = np.stack([r[:,1], r[:,0]], axis=1)  # x,y float
        contours.append(pts)
    return contours

def masks_to_contours_opencv(mask):
    """mask: uint8 0/255. Returns list of contours as Nx2 int (x,y)."""
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = []
    for c in cnts:
        if c.shape[0] >= 3:
            pts = c[:,0,:].astype(float)  # x,y
            contours.append(pts)
    return contours

# ---------------- convert and export to DXF ----------------
def contours_to_dxf(contours_by_layer, path_out, scale_mm_per_px=1.0, use_splines=True):
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    # set units header as millimeter if possible (optional)
    try:
        doc.header['$INSUNITS'] = 4
    except Exception:
        pass

    for layer, contours in contours_by_layer.items():
        if layer not in doc.layers:
            doc.layers.new(layer)
        for pts in contours:
            if len(pts) < 3: 
                continue
            # convert pixel coords -> mm and invert Y for CAD
            h = max(int(np.max(pts[:,1]) + 1), 1)
            pts_mm = [(float(x)*scale_mm_per_px, -float(y)*scale_mm_per_px) for x,y in pts]
            try:
                if use_splines:
                    # ezdxf accepts list of control points (x, y). We use it as fit points.
                    msp.add_spline(pts_mm, dxfattribs={"layer": layer})
                else:
                    msp.add_lwpolyline(pts_mm, close=True, dxfattribs={"layer": layer})
            except Exception:
                # fallback to polyline (robust)
                msp.add_lwpolyline(pts_mm, close=True, dxfattribs={"layer": layer})
    doc.saveas(path_out)

# ---------------- UI ----------------
st.title("Imagen → DXF (v-final, suavizado & mm reales)")
st.markdown("Modo recomendado: `skimage` (curvas suaves). Si `skimage` no está disponible se usa fallback OpenCV.")

uploaded = st.file_uploader("Sube una imagen (PNG/JPG)", type=["png","jpg","jpeg"])
mode = st.radio("Modo vectorizado:", ("skimage (recomendado)", "opencv (fallback, más áspero)"))
n_colors = st.slider("Número de colores (segmentación)", 2, 12, 6)
morph_kernel = st.slider("Kernel morfológico (cierre) - usa para unir pequeñas grietas", 1, 11, 3, step=2)
min_area = st.slider("Filtrar áreas pequeñas (px²)", 10, 2000, 80)
smooth_win = st.slider("Suavizado (window en px)", 1, 21, 7, step=2)
max_points = st.slider("Máx. puntos por contorno (mayores = más detalle)", 200, 2000, 800, step=100)
use_splines = st.checkbox("Exportar como SPLINE en DXF (curvas suaves)", value=True)
show_layers = st.checkbox("Separar por capas (color_n)", value=True)

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("No se pudo leer la imagen.")
        st.stop()
    h, w = img.shape[:2]
    st.markdown(f"**Tamaño original:** {w}px × {h}px")

    ancho_mm = st.number_input("Ancho final en mm (anchura X)", min_value=1.0, max_value=5000.0, value=100.0, step=1.0)
    scale_mm_per_px = float(ancho_mm) / float(w)
    alto_mm = h * scale_mm_per_px
    st.markdown(f"📏 Resultado: **{ancho_mm:.1f} mm × {alto_mm:.1f} mm**  (escala = {scale_mm_per_px:.6f} mm/px)")

    # Segment and extract contours
    labels, centers = segment_kmeans_lab(img, n_colors=n_colors)
    contours_by_layer = {}

    kernel = None
    if morph_kernel > 1:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)

    for i in range(n_colors):
        mask = np.uint8(labels == i) * 255
        if kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # filter small islands
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        # choose method
        if mode.startswith("skimage") and SKIMAGE_AVAILABLE:
            raw_contours = masks_to_contours_skimage(mask)
        else:
            raw_contours = masks_to_contours_opencv(mask)
        useful = []
        for c in raw_contours:
            # area filter: approximate polygon area (use Green formula)
            if len(c) < 3: 
                continue
            area = abs(np.dot(c[:,0], np.roll(c[:,1],1)) - np.dot(c[:,1], np.roll(c[:,0],1))) / 2.0
            if area < min_area:
                continue
            # smoothing and decimation
            sm = smooth_contour(c, window=smooth_win)
            sm = decimate_contour(sm, max_points)
            useful.append(sm)
        if useful:
            layer_name = f"color_{i}" if show_layers else "combined"
            contours_by_layer.setdefault(layer_name, []).extend(useful)

    # Preview: draw contours over thumbnail for quick check
    preview = img.copy()
    for layer, cls in contours_by_layer.items():
        for c in cls:
            pts_i = np.round(c).astype(np.int32)
            if pts_i.shape[0] >= 2:
                cv2.polylines(preview, [pts_i.reshape(-1,1,2)], isClosed=True, color=(0,255,0), thickness=1)

    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Preview del vector (verde)", use_container_width=True)

    if st.button("Generar y descargar DXF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            try:
                contours_to_dxf(contours_by_layer, tmp.name, scale_mm_per_px, use_splines=use_splines)
                st.success("DXF generado con éxito")
                st.download_button("⬇️ Descargar DXF", open(tmp.name,"rb"), file_name="salida_vfinal.dxf")
            except Exception as e:
                st.error(f"Error generando DXF: {e}")
                raise
