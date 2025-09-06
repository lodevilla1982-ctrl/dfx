# app_potrace.py
import streamlit as st
import cv2, numpy as np, ezdxf, tempfile, potrace

def bitmap_to_paths(bw):
    bmp = potrace.Bitmap(bw)
    return bmp.trace()

def save_dxf(paths, path, scale=1.0):
    doc = ezdxf.new()
    msp = doc.modelspace()
    for curve in paths:
        points = []
        for segment in curve.segments:
            if segment.is_corner:
                x, y = segment.c
                points.append((x*scale, -y*scale))
            else:
                # Curva Bezier -> aproximar con polilínea
                for p in [segment.c1, segment.c2, segment.end]:
                    points.append((p[0]*scale, -p[1]*scale))
        if len(points) > 2:
            msp.add_spline(points)
    doc.saveas(path)

st.title("App 2 – Potrace (curvas suaves)")
f = st.file_uploader("Subí una imagen", ["png","jpg","jpeg"])
ancho_mm = st.number_input("Ancho en mm", 10.0, 1000.0, 100.0)

if f:
    data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    _, bw = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
    h, w = bw.shape[:2]
    scale = ancho_mm / w
    st.write(f"Tamaño final: {ancho_mm:.1f}mm x {h*scale:.1f}mm")
    paths = bitmap_to_paths(bw)
    preview = cv2.cvtColor(bw*255, cv2.COLOR_GRAY2BGR)
    st.image(preview, caption="Preview B/N", use_container_width=True)
    if st.button("Exportar DXF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            save_dxf(paths, tmp.name, scale)
            st.download_button("Descargar DXF", open(tmp.name,"rb"), file_name="potrace_out.dxf")
