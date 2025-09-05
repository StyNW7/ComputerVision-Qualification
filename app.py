# NW25-1, Stanley Nathanael Wijaya

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import os
import time

# Try to import streamlit-webrtc for live auto-capture (But I think there is still some error, however there is no need based on the qualification case)

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
    WEBSUPPORTED = True
except Exception:
    WEBSUPPORTED = False


st.set_page_config(page_title="Computer Vision Mini App", layout="wide")


# -------------------------
# Utils / Helper functions
# -------------------------


def to_bytes(img: np.ndarray):
    """Convert OpenCV image (BGR) to bytes for download."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return buf.getvalue()

def pil_to_cv2(pil_img: Image.Image):
    """PIL -> OpenCV BGR"""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def apply_brightness_contrast(img, brightness=0, contrast=0):
    """brightness [-100..100], contrast [-100..100]"""
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
    return img

def cartoonize(img_bgr):
    """Simple cartoon effect."""
    img_color = img_bgr.copy()
    for _ in range(2):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=75, sigmaSpace=75)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    edges = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img_color, edges)

def detect_shapes_from_edges(edges, orig_img):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_out = orig_img.copy()
    shapes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # ignore small noise
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        if len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            # check aspect ratio to distinguish square/rectangle
            ar = w / float(h)
            shape_name = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
        elif len(approx) > 4:
            shape_name = "Circle/Polygon"
        else:
            shape_name = "Unknown"
        shapes.append((shape_name, (x, y, w, h)))
        cv2.drawContours(img_out, [approx], -1, (0,255,0), 2)
        cv2.putText(img_out, shape_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img_out, shapes


# Haarcascade path from cv2 installation
CASCADE_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_FACE):
    st.error("Haarcascade face file not found in cv2.data.haarcascades. Pastikan OpenCV terinstal dengan benar.")
face_cascade = cv2.CascadeClassifier(CASCADE_FACE)


# -------------------------
# Streamlit UI (Bonus: for application)
# -------------------------


st.title("Computer Vision Qualification — Mini App — Image / Edge / Shape / Haarcascade + Auto-capture")
st.markdown("""
Aplikasi demo menggunakan **Python + OpenCV + Streamlit**.  
Fitur: Image processing, Edge detection (Canny/Sobel), Shape detection (contours), Face detection (Haarcascade), Live auto-capture (jika streamlit-webrtc terpasang), dan filter untuk hasil capture.
""")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Input Mode")
    mode = st.selectbox("Pilih mode", ["Image Upload / Snapshot", "Live Auto-Capture (realtime)"])
    if mode == "Live Auto-Capture (realtime)" and not WEBSUPPORTED:
        st.warning("streamlit-webrtc tidak terpasang — Live mode tidak tersedia. Gunakan Image Upload / Snapshot atau install streamlit-webrtc.")
        st.info("Install via: pip install streamlit-webrtc")
    st.write("---")
    st.subheader("Image Processing Options")
    do_resize = st.checkbox("Resize", value=False)
    resize_w = st.number_input("Width (px)", value=640, step=1)
    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast", -100, 100, 0)
    blur_kernel = st.slider("Global Blur kernel (0 = none)", 0, 51, 0, step=2)
    st.write("---")
    st.subheader("Edge detection")
    edge_algo = st.selectbox("Edge algorithm", ["None", "Canny", "Sobel"])
    if edge_algo == "Canny":
        canny1 = st.slider("Canny threshold1", 0, 500, 100)
        canny2 = st.slider("Canny threshold2", 0, 500, 200)
    elif edge_algo == "Sobel":
        sobel_ksize = st.selectbox("Sobel kernel size", [1,3,5,7])
    st.write("---")
    st.subheader("Shape Detection")
    do_shape = st.checkbox("Enable Shape Detection (from edges)", value=True)
    st.write("---")
    st.subheader("Pattern Recognition (Haarcascade)")
    detect_face = st.checkbox("Enable face detection (Haarcascade)", value=True)
    min_face_size = st.number_input("Min face size (px)", value=40, step=1)
    st.write("---")
    st.subheader("Filters for captured face")
    filter_choice = st.selectbox("Default filter for post-capture preview", ["None", "Blur face", "Grayscale", "Edge effect", "Cartoon"])
    auto_save_dir = st.text_input("Auto-save directory for captures", value="captures")
    if st.button("Make sure save dir exists"):
        os.makedirs(auto_save_dir, exist_ok=True)
        st.success(f"Directory ensured: {auto_save_dir}")

with col2:
    st.header("Preview & Actions")

    # Input either upload or camera_input
    frame = None
    if mode == "Image Upload / Snapshot" or not WEBSUPPORTED:
        upload_tab, cam_tab = st.tabs(["Upload Image", "Camera Snapshot"])
        with upload_tab:
            uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
            if uploaded is not None:
                image = Image.open(uploaded)
                frame = pil_to_cv2(image)
        with cam_tab:
            cam = st.camera_input("Take a camera snapshot (this requires permission)")
            if cam is not None:
                image = Image.open(cam)
                frame = pil_to_cv2(image)
    else:

        # Live mode using webrtc: realtime detection + auto-capture
        st.info("Live mode: face will be auto-captured when detected. Tekan 'Start' untuk memulai. (Butuh streamlit-webrtc terpasang)")
        start_live = st.button("Start Live Auto-Capture")
        stop_live = st.button("Stop Live")
        captured_images = st.empty()
        
        if WEBSUPPORTED:
            class LiveFaceCapture(VideoTransformerBase):
                def __init__(self):
                    self.captured = False
                    self.counter = 0
                def transform(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(min_face_size, min_face_size))
                    for (x,y,w,h) in faces:
                        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                    # Auto-capture: when face present and not yet captured
                    if len(faces) > 0 and (not self.captured):
                        # crop first face
                        (x,y,w,h) = faces[0]
                        face_img = img[y:y+h, x:x+w]
                        # save full frame as well
                        os.makedirs(auto_save_dir, exist_ok=True)
                        fname = os.path.join(auto_save_dir, f"capture_live_{int(time.time())}.jpg")
                        cv2.imwrite(fname, img)
                        # also save face region
                        fname_face = os.path.join(auto_save_dir, f"face_live_{int(time.time())}.jpg")
                        cv2.imwrite(fname_face, face_img)
                        self.captured = True
                    # Reset captured flag after some frames so it can capture again later
                    self.counter += 1
                    if self.counter > 120:  # about ~4 seconds depending on fps
                        self.captured = False
                        self.counter = 0
                    return img

            if start_live:
                webrtc_ctx = webrtc_streamer(
                    key="live-face",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=LiveFaceCapture,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

    # If we have a static frame (from upload or camera)
    if frame is not None:
        display_col1, display_col2 = st.columns(2)
        with display_col1:
            st.subheader("Original / Input")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        # Process according to options
        proc = frame.copy()
        if do_resize:
            h, w = proc.shape[:2]
            new_w = int(resize_w)
            new_h = int(h * (new_w / w))
            proc = cv2.resize(proc, (new_w, new_h))
        if brightness != 0 or contrast != 0:
            proc = apply_brightness_contrast(proc, brightness=brightness, contrast=contrast)
        if blur_kernel and blur_kernel > 0:
            k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            proc = cv2.GaussianBlur(proc, (k,k), 0)

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        faces = []
        if detect_face:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(min_face_size,min_face_size))
            # Draw face boxes
            for (x,y,w,h) in faces:
                cv2.rectangle(proc, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.putText(proc, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # Edge detection
        edge_visual = None
        if edge_algo == "Canny":
            edges = cv2.Canny(gray, canny1, canny2)
            edge_visual = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif edge_algo == "Sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = np.uint8(np.clip(sobel, 0, 255))
            edge_visual = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

        # Shape detection (from edges)
        shape_img = None
        shapes_list = []
        if do_shape and edge_visual is not None:
            edges_gray = cv2.cvtColor(edge_visual, cv2.COLOR_BGR2GRAY)
            shape_img, shapes_list = detect_shapes_from_edges(edges_gray, proc)

        # Show processed results
        with display_col2:
            st.subheader("Processed")
            st.image(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.write("---")
        st.subheader("Edge / Shape / Face details (results)")

        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Edges**")
            if edge_visual is not None:
                st.image(cv2.cvtColor(edge_visual, cv2.COLOR_BGR2RGB))
            else:
                st.info("Edge detection not applied")

        with cols[1]:
            st.markdown("**Shapes**")
            if shape_img is not None:
                st.image(cv2.cvtColor(shape_img, cv2.COLOR_BGR2RGB))
                st.write("Detected shapes:")
                for sname, bbox in shapes_list:
                    st.write(f"- {sname} at {bbox}")
            else:
                st.info("Shape detection not run or not enough edges")

        with cols[2]:
            st.markdown("**Faces (Haarcascade)**")
            if len(faces) > 0:
                st.write(f"Detected {len(faces)} face(s)")
                for idx, (x, y, w, h) in enumerate(faces):
                    crop = proc[y:y+h, x:x+w]
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"Face {idx+1}")
                    if st.button(f"Save face {idx+1}", key=f"save_{idx}"):
                        os.makedirs(auto_save_dir, exist_ok=True)
                        name = os.path.join(auto_save_dir, f"face_{int(time.time())}_{idx}.jpg")
                        cv2.imwrite(name, crop)
                        st.success(f"Saved {name}")
            else:
                st.info("No face detected")


        st.write("---")
        st.subheader("Capture & Apply Filter")

        # If faces present, auto-capture the first face (simulate auto-capture for snapshot mode)
        captured = None
        if detect_face and len(faces) > 0:
            (x,y,w,h) = faces[0]
            captured = proc.copy()
            # Auto-save full frame
            os.makedirs(auto_save_dir, exist_ok=True)
            fname = os.path.join(auto_save_dir, f"capture_{int(time.time())}.jpg")
            cv2.imwrite(fname, captured)
            st.success(f"Auto-captured (saved) to {fname}")

        # Allow manual capture if no face or just to re-capture
        if st.button("Manual Capture (save current frame)"):
            os.makedirs(auto_save_dir, exist_ok=True)
            name = os.path.join(auto_save_dir, f"manual_capture_{int(time.time())}.jpg")
            cv2.imwrite(name, proc)
            st.success(f"Saved {name}")
            captured = proc.copy()

        # If we have capture, show filter options and apply selected filter
        if captured is not None:
            st.write("Captured image (first face auto-captured or manual). Pilih filter untuk preview/save.")
            cap_cols = st.columns([2,1])
            with cap_cols[0]:
                st.image(cv2.cvtColor(captured, cv2.COLOR_BGR2RGB), caption="Captured")
            with cap_cols[1]:
                chosen = st.selectbox("Filter for captured image", ["None", "Blur face region", "Grayscale", "Edge overlay", "Cartoonize"])
                apply_btn = st.button("Apply & Save Filtered Image")
                preview = captured.copy()
                if chosen == "Blur face region":
                    # blur face region(s)
                    if len(faces) > 0:
                        (x,y,w,h) = faces[0]
                        face_region = preview[y:y+h, x:x+w]
                        k = 35
                        k = k if k %2 ==1 else k+1
                        face_region_blur = cv2.GaussianBlur(face_region, (k,k), 0)
                        preview[y:y+h, x:x+w] = face_region_blur
                    else:
                        preview = cv2.GaussianBlur(preview, (25,25), 0)
                elif chosen == "Grayscale":
                    g = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                    preview = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                elif chosen == "Edge overlay":
                    g = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                    e = cv2.Canny(g, 100,200)
                    e3 = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
                    preview = cv2.addWeighted(preview, 0.7, e3, 0.3, 0)
                elif chosen == "Cartoonize":
                    preview = cartoonize(preview)
                # show preview
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Filtered Preview")
                if apply_btn:
                    os.makedirs(auto_save_dir, exist_ok=True)
                    namef = os.path.join(auto_save_dir, f"captured_filtered_{int(time.time())}.jpg")
                    cv2.imwrite(namef, preview)
                    st.success(f"Filtered image saved to {namef}")
                    # downloadable
                    st.download_button("Download filtered image", to_bytes(preview), file_name=os.path.basename(namef), mime="image/jpeg")
        else:
            st.info("Belum ada capture — aktifkan camera atau upload gambar lalu deteksi wajah untuk auto-capture.")