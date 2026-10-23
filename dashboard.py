import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import datetime
import os

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Dashboard Klasifikasi dan Deteksi Buah",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CSS THEME (UNICORN: PINK - UNGU - PUTIH)
# ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #ffd1ec 0%, #e3d0ff 50%, #ffffff 100%);
    color: #4a2a6b;
    font-family: 'Arial', sans-serif;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #f9d6ee, #e6dcff); }

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 1100px !important;
}

[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #4a2a6b !important; }

/* TOMBOL */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #ff9bd9, #b388ff);
    border: none; border-radius: 999px; padding: 0.6rem 1.4rem;
    font-weight: 700; box-shadow: 0 6px 16px rgba(179,136,255,0.35);
    transition: all 0.25s ease;
}
.stButton > button *, .stFormSubmitButton > button * { color: #ffffff !important; font-weight: 700; }
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-3px) scale(1.03);
    background: linear-gradient(135deg, #ff7fcf, #9c6bff);
    box-shadow: 0 10px 22px rgba(179,136,255,0.5);
}

@keyframes fadeSlideIn { 0% {opacity:0; transform:translateY(20px);} 100% {opacity:1; transform:translateY(0);} }
@keyframes floaty { 0%,100% {transform:translateY(0);} 50% {transform:translateY(-8px);} }

.hero {
    background: linear-gradient(135deg, #ff9bd9, #b388ff);
    border: 3px solid rgba(255,255,255,0.7);
    border-radius: 28px; padding: 40px 30px; text-align: center; color: #fff;
    box-shadow: 0 12px 30px rgba(179,136,255,0.35); animation: fadeSlideIn 0.8s ease forwards;
}
.hero h1 { color:#fff !important; font-size: 34px; margin: 6px 0; }
.hero p { color:#fff !important; font-size: 16px; margin: 0; }
.hero .fruits { font-size: 42px; letter-spacing: 8px; animation: floaty 3s ease-in-out infinite; }

.main-title {
    background: linear-gradient(145deg, #ff9bd9, #b388ff);
    border: 2px solid rgba(255,255,255,0.7); border-radius: 22px; color: #fff;
    text-align: center; padding: 18px; font-size: 26px; font-weight: bold;
    margin-bottom: 18px; box-shadow: 0 8px 20px rgba(179,136,255,0.3);
    animation: fadeSlideIn 0.8s ease forwards;
}
.section-title {
    font-size: 18px; font-weight: 700; background: rgba(255,255,255,0.7);
    padding: 8px 15px; border-radius: 14px; color: #7b3fa8; margin: 10px 0 14px 0;
    text-align: center; border: 1px solid #f3b8e3; animation: fadeSlideIn 0.8s ease forwards;
}
.explain-box, .detect-result {
    background: rgba(255,255,255,0.78); border: 1px solid #f3b8e3; border-radius: 14px;
    padding: 14px 18px; color: #4a2a6b; margin-top: 10px; font-weight: 500;
    animation: fadeSlideIn 0.8s ease forwards;
}

/* KARTU FITUR & STATISTIK */
.feature-card {
    background: rgba(255,255,255,0.8); border: 1px solid #f3b8e3; border-radius: 20px;
    padding: 22px 16px; text-align: center; height: 100%;
    box-shadow: 0 6px 16px rgba(179,136,255,0.18); transition: all 0.3s ease;
}
.feature-card:hover { transform: translateY(-6px); box-shadow: 0 14px 26px rgba(179,136,255,0.35); }
.feature-card .ico { font-size: 38px; }
.feature-card h4 { color:#7b3fa8 !important; margin: 8px 0 4px 0; }
.feature-card p { font-size: 14px; margin: 0; }

.stat-card {
    background: rgba(255,255,255,0.85); border: 1px solid #f3b8e3; border-radius: 18px;
    padding: 14px; text-align: center; box-shadow: 0 6px 14px rgba(179,136,255,0.15);
}
.stat-card .lbl { font-size: 12px; color:#9a6bbf; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.stat-card .val { font-size: 24px; font-weight: 800; color:#7b3fa8; }

/* STEPPER */
.stepper { display:flex; justify-content:center; gap:10px; margin: 0 0 20px 0; flex-wrap: wrap; }
.step { padding: 6px 16px; border-radius: 999px; font-size: 13px; font-weight: 700;
        background: rgba(255,255,255,0.6); color:#9a6bbf; border:1px solid #f3b8e3; }
.step.active { background: linear-gradient(135deg,#ff9bd9,#b388ff); color:#fff; border-color: transparent; }

.footer-center { text-align: center; color: #7b3fa8; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ==========================
# GLOBALS & SESSION INIT
# ==========================
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ''

if 'user_campus' not in st.session_state:
    st.session_state['user_campus'] = ''

if 'used_feature' not in st.session_state:
    st.session_state['used_feature'] = False

if 'last_mode' not in st.session_state:
    st.session_state['last_mode'] = None

if 'last_pred' not in st.session_state:
    st.session_state['last_pred'] = None

if 'last_conf' not in st.session_state:
    st.session_state['last_conf'] = None

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/najwa_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/best_model.h5")
    return yolo_model, classifier

# attempt load, but catch errors to keep app responsive
try:
    yolo_model, classifier = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    yolo_model, classifier = None, None

# ==========================
# CLASS NAMES
# ==========================
# urutan alfabet, sama dengan urutan kelas pada model YOLO
class_names = ["Apel", "Cherry", "Grapes", "Jeruk", "Kiwi", "Mangga", "Pisang", "Sawo", "Strawberry"]

# ==========================
# PAGES
# ==========================
from plotly import express as px


def stepper(active):
    labels = ["1 · Beranda", "2 · Registrasi", "3 · Dashboard", "4 · Feedback"]
    html = "".join(
        f'<div class="step {"active" if i == active else ""}">{t}</div>' for i, t in enumerate(labels)
    )
    st.markdown(f'<div class="stepper">{html}</div>', unsafe_allow_html=True)


def halaman_awal():
    stepper(0)
    st.markdown(
        '<div class="hero"><div class="fruits">🍎🍓🍇🍊🥝🥭🍌</div>'
        '<h1>Selamat Datang di Dashboard Najwa</h1>'
        '<p>Aplikasi Deteksi &amp; Klasifikasi Buah (YOLO + CNN)</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="feature-card"><div class="ico">📦</div><h4>Deteksi Objek</h4>'
                '<p>YOLO menemukan dan menandai posisi buah pada gambar.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="feature-card"><div class="ico">🧠</div><h4>Klasifikasi Gambar</h4>'
                '<p>CNN menebak jenis buah beserta tingkat keyakinannya.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="feature-card"><div class="ico">💬</div><h4>Feedback</h4>'
                '<p>Berikan rating dan saran setelah mencoba aplikasi.</p></div>', unsafe_allow_html=True)

    st.write("")
    _, mid, _ = st.columns([2, 2, 2])
    with mid:
        if st.button("✨ Mulai Sekarang", use_container_width=True):
            st.session_state['page'] = 'register'
            st.rerun()


def halaman_registrasi():
    stepper(1)
    st.markdown('<div class="main-title">📝 Registrasi Singkat</div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="section-title">Isi data berikut sebelum masuk ke dashboard</div>', unsafe_allow_html=True)
        with st.form("form_reg"):
            name = st.text_input("👤 Nama Lengkap", value=st.session_state.get('user_name', ''))
            campus = st.text_input("🏫 Nama Kampus", value=st.session_state.get('user_campus', ''))
            submitted = st.form_submit_button("Masuk ke Dashboard 🚀", use_container_width=True)
            if submitted:
                if name.strip() == '' or campus.strip() == '':
                    st.warning("Mohon isi Nama dan Nama Kampus sebelum melanjutkan.")
                else:
                    st.session_state['user_name'] = name.strip()
                    st.session_state['user_campus'] = campus.strip()
                    st.session_state['page'] = 'main'
                    st.rerun()
        if st.button("⬅️ Kembali", use_container_width=True):
            st.session_state['page'] = 'home'
            st.rerun()


def stat_card(label, value):
    return f'<div class="stat-card"><div class="lbl">{label}</div><div class="val">{value}</div></div>'


def halaman_main():
    stepper(2)
    if os.path.exists("model/LOGO UUSK.jpg"):
        st.sidebar.image("model/LOGO UUSK.jpg", use_container_width=True)
    st.sidebar.markdown("### 🦄 Menu Analisis")
    menu = st.sidebar.radio("Pilih Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.sidebar.info("📤 Unggah gambar sesuai mode yang dipilih untuk mulai analisis AI.")
    st.sidebar.markdown("---")
    if st.sidebar.button("⬅️ Kembali ke Registrasi", use_container_width=True):
        st.session_state['page'] = 'register'
        st.rerun()

    st.markdown('<div class="main-title">🔎 Dashboard Utama — Deteksi & Klasifikasi Buah</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class=\"explain-box\">Halo <b>{st.session_state.get('user_name','')}</b> dari "
        f"<b>{st.session_state.get('user_campus','')}</b> 👋 — mode aktif: <b>{menu}</b>. "
        f"Unggah gambar untuk memulai.</div>", unsafe_allow_html=True)
    st.write("")

    uploaded_file = st.file_uploader("📂 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("👆 Silakan unggah gambar terlebih dahulu untuk mulai deteksi atau klasifikasi.")
        return

    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="section-title">🖼️ Gambar Diupload</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)

    stats = []

    # DETECTION
    if menu == "Deteksi Objek (YOLO)":
        if yolo_model is None:
            st.error("Model YOLO belum tersedia.")
        else:
            with st.spinner("🚀 AI sedang mendeteksi objek... harap tunggu sebentar!"):
                results = yolo_model(np.array(img))
                result_img = results[0].plot()
                boxes = results[0].boxes

            with col2:
                st.markdown('<div class="section-title">📦 Hasil Deteksi Objek</div>', unsafe_allow_html=True)
                if boxes is not None and len(boxes) > 0:
                    st.image(result_img, use_container_width=True)

                    data, confidences = [], []
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        data.append({
                            "Label": results[0].names[cls],
                            "Confidence": f"{conf*100:.2f}%",
                            "x1": int(xyxy[0]), "y1": int(xyxy[1]),
                            "x2": int(xyxy[2]), "y2": int(xyxy[3])
                        })
                        confidences.append(conf * 100)

                    avg_conf = float(np.mean(confidences))
                    st.session_state['last_mode'] = 'detection'
                    st.session_state['last_pred'] = f"{len(boxes)} objek terdeteksi"
                    st.session_state['last_conf'] = avg_conf
                    st.session_state['used_feature'] = True
                    stats = [("Mode", "Deteksi"), ("Objek Terdeteksi", len(boxes)), ("Rata-rata Confidence", f"{avg_conf:.1f}%")]

                    st.markdown("---")
                    st.subheader("📋 Rincian Hasil Deteksi")
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                    st.caption("💡 Semakin tinggi nilai confidence, semakin yakin model terhadap deteksi tersebut.")
                else:
                    st.warning("⚠️ Tidak ada objek yang terdeteksi dalam gambar ini.")

    # CLASSIFICATION
    else:
        if classifier is None:
            st.error("Model klasifikasi belum tersedia.")
        else:
            with st.spinner("🧠 Sedang melakukan klasifikasi gambar..."):
                target_size = classifier.input_shape[1:3]
                img_resized = img.resize(target_size)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(img_array)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction)) * 100.0

            with col2:
                st.markdown('<div class="section-title">📊 Hasil Klasifikasi</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="hero" style="padding:22px;"><p>Kategori Prediksi</p>'
                    f'<h1>{class_names[class_index]}</h1></div>', unsafe_allow_html=True)
                st.progress(min(max(confidence / 100.0, 0.0), 1.0))
                st.write(f"**Tingkat Keyakinan Model:** {confidence:.2f}%")

                prob_df = pd.DataFrame({
                    "Kelas": class_names,
                    "Probabilitas (%)": [float(prediction[0][i]) * 100 for i in range(len(class_names))]
                }).sort_values("Probabilitas (%)", ascending=True)
                fig_bar = px.bar(prob_df, x="Probabilitas (%)", y="Kelas", orientation="h",
                                 color_discrete_sequence=["#b388ff"])
                fig_bar.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

                st.session_state['last_mode'] = 'classification'
                st.session_state['last_pred'] = class_names[class_index]
                st.session_state['last_conf'] = confidence
                st.session_state['used_feature'] = True
                stats = [("Mode", "Klasifikasi"), ("Prediksi", class_names[class_index]), ("Keyakinan", f"{confidence:.1f}%")]

    # RINGKASAN + DONUT
    if stats and st.session_state.get('used_feature', False):
        st.markdown('<div class="section-title">✨ Ringkasan Hasil</div>', unsafe_allow_html=True)
        sc = st.columns(len(stats))
        for c, (lbl, val) in zip(sc, stats):
            c.markdown(stat_card(lbl, val), unsafe_allow_html=True)

        val = st.session_state.get('last_conf', None)
        if val is not None:
            benar = float(val)
            salah = max(0.0, 100.0 - benar)
            pie_data = pd.DataFrame({"Hasil": ["Benar", "Salah"], "Persentase": [benar, salah]})
            fig = px.pie(pie_data, values='Persentase', names='Hasil',
                         color='Hasil', color_discrete_map={'Benar': '#b388ff', 'Salah': '#ffc2e6'},
                         hole=0.45)
            fig.update_traces(textinfo='label+percent', marker=dict(line=dict(color='#ffffff', width=2)))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=0, b=0),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.markdown('<div class="section-title">🍩 Donut Chart Persen Akurasi</div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns([1, 2, 1])
            with d2:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        _, b, _ = st.columns([2, 2, 2])
        with b:
            if st.button("💬 Berikan Feedback", use_container_width=True):
                st.session_state['page'] = 'feedback'
                st.rerun()


def halaman_feedback():
    stepper(3)
    st.markdown('<div class="main-title">💬 Feedback Pengguna</div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="section-title">Terima kasih telah menggunakan aplikasi — beri masukanmu 💖</div>', unsafe_allow_html=True)
        with st.form("form_feedback"):
            rating = st.slider("⭐ Rating (1 - 5)", min_value=1, max_value=5, value=5)
            comment = st.text_area("Komentar / Saran", height=120)
            submitted = st.form_submit_button("Kirim Feedback 💌", use_container_width=True)

            if submitted:
                now = datetime.datetime.now().isoformat()
                row = {
                    'timestamp': now,
                    'user_name': st.session_state.get('user_name', ''),
                    'user_campus': st.session_state.get('user_campus', ''),
                    'mode': st.session_state.get('last_mode', ''),
                    'prediction': st.session_state.get('last_pred', ''),
                    'confidence': st.session_state.get('last_conf', ''),
                    'rating': rating,
                    'comment': comment
                }
                df_row = pd.DataFrame([row])
                csv_path = 'feedbacks.csv'
                try:
                    if not os.path.exists(csv_path):
                        df_row.to_csv(csv_path, index=False, mode='w', encoding='utf-8')
                    else:
                        df_row.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8')
                    st.success("✅ Feedback berhasil dikirim — terima kasih!")
                    st.markdown(f"<div class='footer-center'>Feedback tersimpan ke <b>{csv_path}</b></div>", unsafe_allow_html=True)
                    st.session_state['used_feature'] = False
                except Exception as e:
                    st.error(f"Gagal menyimpan feedback: {e}")

        if st.button("⬅️ Kembali ke Dashboard", use_container_width=True):
            st.session_state['page'] = 'main'
            st.rerun()


# ==========================
# ROUTING
# ==========================
if st.session_state['page'] == 'home':
    halaman_awal()
elif st.session_state['page'] == 'register':
    halaman_registrasi()
elif st.session_state['page'] == 'main':
    halaman_main()
elif st.session_state['page'] == 'feedback':
    halaman_feedback()
else:
    halaman_awal()

# FOOTER
st.markdown("---")
st.markdown("<center style='color:#7b3fa8;'>created by <b>Najwa | NIM_KAMU</b><br>Proyek UAS Praktikum Pemrograman Big Data | Universitas Syiah Kuala</center>", unsafe_allow_html=True)
