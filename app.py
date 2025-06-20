import streamlit as st
import cv2
import os
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Pengaturan Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Multi-Model",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("😷 Deteksi Masker Real-time (Multi-Model)")
st.write("Aplikasi ini memungkinkan Anda memilih model YOLO yang berbeda untuk melakukan deteksi secara langsung.")

# --- Inisialisasi Session State ---
if "model" not in st.session_state:
    st.session_state.model = None

if "model_name" not in st.session_state:
    st.session_state.model_name = None

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.5

# --- Fungsi untuk memuat model (dengan cache) ---
@st.cache_resource
def load_yolo_model(model_path):
    """Memuat model YOLO dari path yang diberikan."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- Sidebar untuk pemilihan model dan pengaturan ---
with st.sidebar:
    st.header("Pengaturan")

    weights_folder = 'weights'
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)

    model_files = [f for f in os.listdir(weights_folder) if f.endswith('.pt')]

    if not model_files:
        st.warning("Tidak ada file model (.pt) yang ditemukan di folder 'weights'.")
        selected_model_file = None
    else:
        selected_model_file = st.selectbox("Pilih Model:", model_files)

    confidence_threshold = st.slider(
        "Tingkat Keyakinan (Confidence)", 0.0, 1.0, st.session_state.confidence_threshold, 0.05
    )
    st.session_state.confidence_threshold = confidence_threshold

# --- Logika Pemuatan Model menggunakan Session State ---
if selected_model_file:
    model_path = os.path.join(weights_folder, selected_model_file)

    if (
        st.session_state.model is None
        or st.session_state.model_name != selected_model_file
    ):
        with st.spinner(f"Memuat model '{selected_model_file}'..."):
            model = load_yolo_model(model_path)
            if model:
                st.session_state.model = model
                st.session_state.model_name = selected_model_file
                st.success(f"Model '{selected_model_file}' berhasil dimuat!")
            else:
                st.error("Gagal memuat model. Periksa file dan log.")

# --- Kelas untuk memproses frame video ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, confidence_threshold):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=self.confidence_threshold, verbose=False)
        result = results[0]

        # Inisialisasi counter per frame
        counter = {
            'with_mask': 0,
            'without_mask': 0,
            'mask_weared_incorrect': 0
        }

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                label = f"{class_name} {conf:.2f}"

                if class_name == 'with_mask':
                    color = (0, 255, 0)
                    counter['with_mask'] += 1
                elif class_name == 'without_mask':
                    color = (0, 0, 255)
                    counter['without_mask'] += 1
                elif class_name == 'mask_weared_incorrect':
                    color = (0, 255, 255)
                    counter['mask_weared_incorrect'] += 1
                else:
                    color = (255, 255, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Tampilkan counter realtime di frame
        y_offset = 30
        for key, value in counter.items():
            text = f"{key}: {value}"
            cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 30

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Stream Kamera Real-time ---
if st.session_state.model is not None:
    st.divider()
    st.header(f"Deteksi Real-time Menggunakan: `{st.session_state.model_name}`")

    # Salin ke variabel lokal agar tidak error saat digunakan di thread lain
    current_model = st.session_state.model
    current_confidence = st.session_state.confidence_threshold

    webrtc_streamer(
        key="detection-stream",
        video_processor_factory=lambda: VideoTransformer(
            current_model,
            current_confidence
        ),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    st.info("Letakkan file model .pt Anda di dalam folder 'weights' lalu pilih model dari sidebar untuk memulai.")
