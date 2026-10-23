import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import datetime

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Dashboard Klasifikasi dan Deteksi Objek",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CSS THEME (BIRU LEMBUT)
# ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #b3c7ff, #8e9fff);
    color: #f2f4ff;
    font-family: 'Arial', sans-serif;
    padding: 2rem 3rem;
    margin: 0;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }

.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

@keyframes fadeSlideIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

.main-title, .section-title, .detect-result, .explain-box {
    transition: all 0.3s ease-in-out;
    transform: scale(1);
    animation: fadeSlideIn 0.8s ease forwards;
}

.main-title:hover, .section-title:hover, .detect-result:hover, .explain-box:hover {
    transform: scale(1.03);
    box-shadow: 6px 6px 15px rgba(0,0,0,0.12);
}

.main-title:active, .section-title:active, .detect-result:active, .explain-box:active {
    transform: scale(0.97);
    box-shadow: 2px 2px 6px rgba(0,0,0,0.12);
}

.main-title {
    background: linear-gradient(145deg, #6b7bd6, #4f63c8);
    border: 2px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    color: #f8fbff;
    text-align: center;
    padding: 20px;
    font-size: 28px;
    font-weight: bold;
    margin: 20px auto 25px auto;
    box-shadow: 4px 6px 12px rgba(0,0,0,0.08);
    width: 100%;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    background-color: rgba(255,255,255,0.06);
    padding: 8px 15px;
    border-radius: 12px;
    color: #f2f6ff;
    margin-bottom: 15px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.04);
}

.detect-result {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px;
    color: #f2f6ff;
    font-weight: 600;
    text-align: center;
}

.explain-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 15px;
    color: #f2f6ff;
    margin-top: 10px;
    font-weight: 500;
    text-align: justify;
}

.footer-center { text-align: center; color: #eef2ff; font-weight: 500; }
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
    yolo_model = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
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
class_names = ["Ballet Flat", "Boat", "Brogue", "Clog", "Sneaker"]

# ==========================
# PAGES
# ==========================

def halaman_awal():
    st.markdown("<div class=\"main-title\">👟 <b>SELAMAT DATANG DI DASHBOARD SEFTYA PRATISTA</b> 👟</div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Aplikasi Deteksi & Klasifikasi Sepatu (YOLO + CNN)</div>', unsafe_allow_html=True)
    st.markdown('<div class="explain-box">\
        Halo! Ini adalah aplikasi demo yang memperlihatkan bagaimana model deteksi (YOLO) dan klasifikasi (CNN) bekerja pada citra sepatu.\
        Klik tombol di bawah untuk melanjutkan ke proses registrasi singkat sebelum menggunakan dashboard.</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("Lanjut ke Registrasi →"):
        st.session_state['page'] = 'register'


def halaman_registrasi():
    st.markdown('<div class="main-title">📝 Registrasi Singkat</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Isi data berikut sebelum masuk ke dashboard</div>', unsafe_allow_html=True)

    with st.form("form_reg"):
        name = st.text_input("Nama Lengkap", value=st.session_state.get('user_name', ''))
        campus = st.text_input("Nama Kampus", value=st.session_state.get('user_campus', ''))
        submitted = st.form_submit_button("Masuk ke Dashboard Utama")
        if submitted:
            if name.strip() == '' or campus.strip() == '':
                st.warning("Mohon isi Nama dan Nama Kampus sebelum melanjutkan.")
            else:
                st.session_state['user_name'] = name.strip()
                st.session_state['user_campus'] = campus.strip()
                st.session_state['page'] = 'main'

    if st.button("⬅️ Kembali ke Halaman Awal"):
        st.session_state['page'] = 'home'


from plotly import express as px

def halaman_main():
    st.sidebar.image("model/LOGO UUSK.jpg", use_container_width=True)
    menu = st.sidebar.radio("🧭 Pilih Mode Analisis:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]) 
    st.sidebar.info("📤 Unggah gambar sesuai mode yang dipilih untuk mulai analisis AI.")

    st.markdown('<div class="main-title">🔎 Dashboard Utama — Deteksi & Klasifikasi</div>', unsafe_allow_html=True)
    st.markdown(f"<div class=\"explain-box\">Halo <b>{st.session_state.get('user_name','')}</b> dari <b>{st.session_state.get('user_campus','')}</b> — pilih mode dan unggah gambar untuk memulai.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Unggah Gambar", type=["jpg", "jpeg", "png"])    

    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(img, caption="🖼️ Gambar Diupload", use_container_width=True)

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
                    if boxes is not None and len(boxes) > 0:
                        st.image(result_img, caption="📦 Hasil Deteksi Objek", use_container_width=True)
                        st.success(f"✅ {len(boxes)} objek berhasil terdeteksi!")

                        # Buat tabel hasil deteksi
                        data = []
                        confidences = []
                        for box in boxes:
                            xyxy = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            data.append({
                                "Label": results[0].names[cls],
                                "Confidence": f"{conf*100:.2f}%",
                                "x1": int(xyxy[0]),
                                "y1": int(xyxy[1]),
                                "x2": int(xyxy[2]),
                                "y2": int(xyxy[3])
                            })
                            confidences.append(conf*100)

                        df = pd.DataFrame(data)
                        st.subheader("📋 Rincian Hasil Deteksi")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.markdown("> 💡 Semakin tinggi nilai *confidence*, semakin yakin model terhadap deteksi tersebut.")

                        # Simpan ringkasan sebagai last prediction
                        avg_conf = float(np.mean(confidences)) if len(confidences)>0 else 0.0
                        st.session_state['last_mode'] = 'detection'
                        st.session_state['last_pred'] = f"{len(boxes)} objek terdeteksi"
                        st.session_state['last_conf'] = avg_conf
                        st.session_state['used_feature'] = True

                    else:
                        st.warning("⚠️ Tidak ada objek yang terdeteksi dalam gambar ini.")

        # CLASSIFICATION
        elif menu == "Klasifikasi Gambar":
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
                    st.subheader("📊 Hasil Klasifikasi")
                    st.metric(label="Kategori Prediksi", value=class_names[class_index])
                    st.progress(confidence/100.0)
                    st.write(f"**Tingkat Keyakinan Model:** {confidence:.2f}%")

                    st.write("🔢 Probabilitas per kelas:")
                    prob_dict = {class_names[i]: f"{prediction[0][i]*100:.2f}%" for i in range(len(class_names))}
                    st.json(prob_dict)

                    # Simpan ringkasan sebagai last prediction
                    st.session_state['last_mode'] = 'classification'
                    st.session_state['last_pred'] = class_names[class_index]
                    st.session_state['last_conf'] = confidence
                    st.session_state['used_feature'] = True

        # Donut Chart (jika sudah ada prediksi)
        if st.session_state.get('used_feature', False):
            val = st.session_state.get('last_conf', None)
            if val is not None:
                benar = float(val)
                salah = max(0.0, 100.0 - benar)
                pie_data = pd.DataFrame({"Hasil":["Benar","Salah"], "Persentase":[benar,salah]})
                fig = px.pie(pie_data, values='Persentase', names='Hasil',
                             color='Hasil', color_discrete_map={'Benar':'#4f63c8','Salah':'#dce4ff'},
                             hole=0.35)
                fig.update_traces(textinfo='label+percent+value', marker=dict(line=dict(color='rgba(0,0,0,0)')))
                fig.update_layout(width=300, height=300, margin=dict(l=0,r=0,t=0,b=0),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

                st.markdown('<div class="section-title">Donut Chart Persen Akurasi</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False})

                if st.button("Berikan Feedback"):
                    st.session_state['page'] = 'feedback'
    else:
        st.info("👆 Silakan unggah gambar terlebih dahulu untuk mulai deteksi atau klasifikasi.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Kembali ke Registrasi"):
        st.session_state['page'] = 'register'


def halaman_feedback():
    st.markdown('<div class="main-title">💬 Feedback Pengguna</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Terima kasih telah menggunakan aplikasi — beri masukanmu di bawah ini</div>', unsafe_allow_html=True)

    with st.form("form_feedback"):
        rating = st.slider("Rating (1 - 5)", min_value=1, max_value=5, value=5)
        comment = st.text_area("Komentar / Saran", height=120)
        submitted = st.form_submit_button("Kirim Feedback")

        if submitted:
            # simpan ke CSV
            now = datetime.datetime.now().isoformat()
            row = {
                'timestamp': now,
                'user_name': st.session_state.get('user_name',''),
                'user_campus': st.session_state.get('user_campus',''),
                'mode': st.session_state.get('last_mode',''),
                'prediction': st.session_state.get('last_pred',''),
                'confidence': st.session_state.get('last_conf',''),
                'rating': rating,
                'comment': comment
            }
            df_row = pd.DataFrame([row])
            csv_path = 'feedbacks.csv'
            try:
                # append to csv (create if not exists)
                if not pd.io.common.file_exists(csv_path):
                    df_row.to_csv(csv_path, index=False, mode='w', encoding='utf-8')
                else:
                    df_row.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8')
                st.success("✅ Feedback berhasil dikirim — terima kasih!")
                st.markdown(f"<div class='footer-center'>Feedback tersimpan ke <b>{csv_path}</b></div>", unsafe_allow_html=True)

                # reset used_feature supaya tidak mengirim ganda
                st.session_state['used_feature'] = False

            except Exception as e:
                st.error(f"Gagal menyimpan feedback: {e}")

    if st.button("⬅️ Kembali ke Dashboard"):
        st.session_state['page'] = 'main'


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
st.markdown("<center>created by <b>Seftya Pratista | 2208108010054</b><br>Proyek UAS Praktikum Pemrograman Big Data | Universitas Syiah Kuala</center>", unsafe_allow_html=True)
