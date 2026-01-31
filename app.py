import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import os
import pandas as pd

# =====================================================
# PAGE CONFIG (لازم أول حاجة)
# =====================================================
st.set_page_config(
    page_title="Fire & Smoke AI Guard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PATHS SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def asset_path(filename):
    return os.path.join(BASE_DIR, "assets", filename)

def notebook_path(filename):
    return os.path.join(BASE_DIR, "notebooks", filename)

# =====================================================
# SAFE IMAGE DISPLAY
# =====================================================
def show_image(filename, caption=None):
    path = asset_path(filename)
    if os.path.exists(path):
        try:
            img = Image.open(path)
            st.image(img, caption=caption, use_container_width=True)
        except:
            st.error(f"Cannot open image: {filename}")
    else:
        st.warning(f"Missing file: {filename}")

# =====================================================
# CSS (MODERN UI)
# =====================================================
st.markdown("""
<style>
.main {background-color: #f6f7fb;}
.stTabs [data-baseweb="tab"] {
    background-color: #ffffff;
    border-radius: 12px 12px 0 0;
    padding: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #ffecec;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO HEADER
# =====================================================
st.markdown("""
<div style="
    background: linear-gradient(90deg, #c62828, #ff7043);
    padding: 30px;
    border-radius: 18px;
    color: white;
">
    <h1 style="margin-bottom:5px;">🔥 Fire & Smoke AI Guard</h1>
    <h4 style="font-weight:400;">
        Real-time Fire & Smoke Detection using YOLOv26n
    </h4>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "best.pt")
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error("❌ Model file best.pt not found")
        return None

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")

    conf_threshold = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.25, 0.01
    )

    iou_threshold = st.slider(
        "IoU Threshold",
        0.0, 1.0, 0.45, 0.01
    )

    st.markdown("---")
    st.markdown("""👨‍💻 **Developed by**        
                **Mohamed Mostafa Hassan Afify**""")
    st.markdown("🔗 [GitHub Repository](https://github.com/mohamedmhafify)")
    st.markdown("🔗 [Linked In](https://www.linkedin.com/in/mohamedmhafify)")

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs([
    "🚀 Detection Demo",
    "📊 Performance Dashboard",
    "🧠 Training Story"
])

# =====================================================
# TAB 1 — DETECTION
# =====================================================
with tab1:
    st.subheader("🚀 Run Detection")

    source_type = st.radio(
        "Choose Input Type:",
        ("🖼 Image", "🎥 Video"),
        horizontal=True
    )

    st.markdown("---")

    if source_type == "🖼 Image":
        uploaded_file = st.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file and model:
            col1, col2 = st.columns(2)

            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)

            with col2:
                if st.button("🔥 Start Detection", type="primary", use_container_width=True):
                    with st.spinner("Running YOLOv26n inference..."):
                        res = model.predict(
                            image,
                            conf=conf_threshold,
                            iou=iou_threshold
                        )
                    st.success("✅ Detection completed successfully!")
                    st.image(
                        res[0].plot(),
                        caption="Detection Result",
                        use_container_width=True
                    )

    else:
        uploaded_video = st.file_uploader(
            "Upload Video", type=["mp4", "avi", "mov"]
        )

        if uploaded_video and model:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            if st.button("🎬 Start Video Analysis", type="primary"):
                st.info("⏱ Processing video frames in real-time")
                cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(
                        frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )

                    frame_rgb = cv2.cvtColor(
                        results[0].plot(),
                        cv2.COLOR_BGR2RGB
                    )

                    st_frame.image(frame_rgb, use_container_width=True)

                cap.release()
                st.success("🎉 Video analysis finished")

# =====================================================
# TAB 2 — PERFORMANCE DASHBOARD
# =====================================================
with tab2:
    st.markdown("## 📊 Model Performance Overview")

    # الأرقام الكلية الحقيقية (Overall)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("mAP@0.5 (All)", "0.739")
    m2.metric("Precision", "0.716")
    m3.metric("Recall", "0.671")
    m4.metric("Peak F1-Score", "0.79")

    st.markdown("---")

# إضافة تحليل النار والدخان (Per-Class Analysis)
    st.markdown("### 🔍 Detailed Class Analysis")
    col_fire, col_smoke = st.columns(2)

    with col_fire:
        st.markdown("""
        <div style="background-color: #fff5f5; padding: 15px; border-radius: 12px; border-left: 6px solid #d32f2f; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
            <h4 style="color: #000000; margin: 0;">🔥 Fire Detection</h4>
            <p style="color: #000000; margin: 5px 0;"><b>mAP@0.5:</b> 0.865</p>
            <p style="color: #000000; margin: 5px 0;"><b>Precision:</b> 0.819</p>
            <p style="color: #000000; margin: 5px 0;"><b>Recall:</b> 0.787</p>
        </div>
        """, unsafe_allow_html=True)

    with col_smoke:
        st.markdown("""
        <div style="background-color: #f7f7f7; padding: 15px; border-radius: 12px; border-left: 6px solid #757575; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
            <h4 style="color: #000000; margin: 0;">💨 Smoke Detection</h4>
            <p style="color: #000000; margin: 5px 0;"><b>mAP@0.5:</b> 0.613</p>
            <p style="color: #000000; margin: 5px 0;"><b>Precision:</b> 0.614</p>
            <p style="color: #000000; margin: 5px 0;"><b>Recall:</b> 0.556</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # الرسوم البيانية داخل Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Confusion Matrix")
        show_image("confusion_matrix_normalized.png")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Training Curves")
        show_image("results.png")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📈 Precision-Recall & F1 Analysis")
    col3, col4 = st.columns(2)
    with col3:
        show_image("BoxPR_curve.png", "Precision-Recall Curve")
    with col4:
        show_image("BoxF1_curve.png", "F1 Score vs Confidence")

    with st.expander("🖼 View Dataset Distribution (Labels)"):
        show_image("labels.jpg")

    # عرض البيانات الخام
    csv_path = asset_path("results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.markdown("### 📄 Raw Training Metrics (Last 5 Epochs)")
        st.dataframe(df.tail(), use_container_width=True)

# =====================================================
# TAB 3 — TRAINING STORY
# =====================================================
with tab3:
    st.markdown("## 🧠 Training Story")

    st.markdown("""
    This project focuses on **early fire & smoke detection** in complex surveillance environments using the cutting-edge **YOLO26 Nano** architecture. The goal was to build a system that is not only accurate but also fast enough for real-time edge deployment.
    
    **Key Challenges & Solutions:**
    * **The Smoke Dilemma:** Detecting **small, low-contrast smoke** is difficult because it lacks a fixed shape. I optimized the model over **100 epochs** to capture these "non-rigid" features effectively.
    * **Precision Tuning:** Based on the **F1-Confidence Curve**, I identified **0.23** as the optimal threshold to balance detection sensitivity and minimize false alarms.
    * **Performance:** Achieved a blistering **5ms inference speed**, making it ideal for integration with high-speed CCTV systems.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Epochs", "100")
    col2.metric("Batch Size", "16")
    col3.metric("Image Size", "640×640")

    st.markdown("### ⚙️ Training Code")
    st.code("""
!pip install ultralytics

from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="Fire-Smoke-1/data.yaml", epochs=100, imgsz=640)
""", language="python")

    nb_path = notebook_path("training_code.ipynb")
    if os.path.exists(nb_path):
        with open(nb_path, "rb") as f:
            st.download_button(
                "📥 Download Full Training Notebook",
                f,
                file_name="Fire_Detection_Training.ipynb",
                use_container_width=True
            )

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr>
<div style="text-align:center; color:gray;">
🚀 Built with YOLOv26n & Streamlit <br>
© 2026 Mohamed Mostafa Hassan Afify — Computer Vision Engineer
</div>
""", unsafe_allow_html=True)
