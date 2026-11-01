import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import os
from collections import Counter
from plotly import express as px

# ==========================
# CONFIG
# ==========================
st.set_page_config(
    page_title="Dashboard Klasifikasi dan Deteksi Buah",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================
# CSS THEME (UNICORN GLASS: PINK - UNGU - PUTIH)
# ==========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&family=Fredoka:wght@500;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: 'Poppins', sans-serif; }

/* ---------- LATAR BERGERAK ---------- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #ffd6f1, #e6d4ff, #d6e6ff, #ffe3f4);
    background-size: 300% 300%;
    animation: bgmove 14s ease infinite;
    color: #4a2a6b;
}
@keyframes bgmove { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

[data-testid="stAppViewContainer"]::before, [data-testid="stAppViewContainer"]::after {
    content: ""; position: fixed; border-radius: 50%; filter: blur(70px); opacity: .55;
    pointer-events: none; z-index: 0;
}
[data-testid="stAppViewContainer"]::before { width: 420px; height: 420px; background: #ff9bd9; top: -120px; left: -100px; animation: blob 9s ease-in-out infinite; }
[data-testid="stAppViewContainer"]::after  { width: 460px; height: 460px; background: #b388ff; bottom: -160px; right: -120px; animation: blob 11s ease-in-out infinite reverse; }
@keyframes blob { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(50px,40px) scale(1.15)} }

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }
.block-container { position: relative; z-index: 1; padding-top: 1.5rem !important; max-width: 1150px !important; }

[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li { color: #4a2a6b !important; }

/* ---------- ANIMASI ---------- */
@keyframes rise { 0%{opacity:0; transform:translateY(24px)} 100%{opacity:1; transform:translateY(0)} }
@keyframes floaty { 0%,100%{transform:translateY(0) rotate(-2deg)} 50%{transform:translateY(-12px) rotate(2deg)} }
@keyframes shine { 0%{background-position:0% 50%} 100%{background-position:200% 50%} }

/* ---------- KARTU KACA ---------- */
.glass {
    background: rgba(255,255,255,0.55); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1.5px solid rgba(255,255,255,0.85); border-radius: 26px; padding: 26px;
    box-shadow: 0 12px 34px rgba(150,100,220,0.18); animation: rise .7s ease both;
}
.glass.tight { padding: 16px 20px; border-radius: 20px; }

/* ---------- TEKS ---------- */
.badge { display:inline-block; padding:6px 16px; border-radius:999px; font-size:13px; font-weight:600;
         background: rgba(255,255,255,.75); color:#8b4fc0; border:1px solid #f1b6e2; margin-bottom: 14px; }
.big-title { font-family:'Fredoka',sans-serif; font-weight:700; font-size: 56px; line-height: 1.05; margin: 0 0 12px 0; color:#4a2a6b; }
.grad-text {
    background: linear-gradient(90deg, #ff6fc8, #b388ff, #6fa8ff, #ff6fc8); background-size: 200% auto;
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    animation: shine 5s linear infinite;
}
.sub { font-size: 17px; color:#6b4a8a; margin-bottom: 18px; }
.h2 { font-family:'Fredoka',sans-serif; font-size: 26px; font-weight: 700; color:#4a2a6b; margin: 6px 0 4px 0; }
.muted { color:#8a6aa8; font-size: 14px; }

/* ---------- BUAH MELAYANG ---------- */
.fruit-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.fruit-tile {
    background: rgba(255,255,255,.75); border-radius: 22px; padding: 16px 6px; text-align:center;
    border: 1px solid #f6c6ea; box-shadow: 0 8px 18px rgba(179,136,255,.2);
    animation: floaty 4s ease-in-out infinite;
}
.fruit-tile .e { font-size: 44px; }
.fruit-tile .n { font-size: 12px; font-weight: 600; color:#7b3fa8; }

/* ---------- LANGKAH ---------- */
.step-num { width: 46px; height: 46px; border-radius: 50%; display:flex; align-items:center; justify-content:center;
    font-weight: 800; font-size: 20px; color:#fff; background: linear-gradient(135deg,#ff6fc8,#a97bff);
    box-shadow: 0 6px 14px rgba(169,123,255,.45); margin-bottom: 10px; }
.step-card h4 { margin: 0 0 4px 0; color:#7b3fa8; font-family:'Fredoka',sans-serif; font-size: 20px; }
.step-card p { margin:0; font-size: 14px; }

/* ---------- PROGRESS TITIK ---------- */
.dots { display:flex; gap:22px; align-items:center; justify-content:center; margin-bottom: 18px; }
.dot { font-size: 24px; opacity: .35; filter: grayscale(.7); transition: all .3s ease; }
.dot.on { opacity: 1; filter: none; transform: scale(1.5); animation: floaty 3s ease-in-out infinite; }

/* ---------- BUAH LATAR BELAKANG ---------- */
.fruit-bg { position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }
.fruit-bg span { position: absolute; opacity: .22; animation: floaty 6s ease-in-out infinite; }

/* ---------- TOMBOL ---------- */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #ff6fc8, #a97bff); border: none; border-radius: 16px;
    padding: .7rem 1.4rem; box-shadow: 0 8px 20px rgba(169,123,255,.4); transition: all .25s ease;
}
.stButton > button *, .stFormSubmitButton > button * { color:#fff !important; font-weight: 700; font-size: 16px; }
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-3px) scale(1.03); box-shadow: 0 14px 28px rgba(169,123,255,.55);
    background: linear-gradient(135deg, #ff4fbb, #8f5cff);
}

/* ---------- INPUT ---------- */
[data-baseweb="input"], [data-baseweb="textarea"] { border-radius: 14px !important; background: rgba(255,255,255,.85) !important; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea { color:#4a2a6b !important; }

/* ---------- RADIO JADI TOMBOL PIL ---------- */
div[role="radiogroup"] { gap: 10px; flex-wrap: wrap; }
div[role="radiogroup"] label {
    background: rgba(255,255,255,.7); border: 1.5px solid #f1b6e2; border-radius: 999px;
    padding: 8px 20px !important; cursor: pointer; transition: all .2s ease;
}
div[role="radiogroup"] label > div:first-child { display: none; }
div[role="radiogroup"] label:hover { transform: translateY(-2px); background: #fff; }
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg,#ff6fc8,#a97bff); border-color: transparent; box-shadow: 0 8px 18px rgba(169,123,255,.45);
}
div[role="radiogroup"] label:has(input:checked) * { color:#fff !important; font-weight: 700; }

/* ---------- UPLOADER ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.65); border: 2.5px dashed #c58bf2; border-radius: 24px; padding: 28px;
}
[data-testid="stFileUploaderDropzone"] * { color:#7b3fa8 !important; }
[data-testid="stFileUploaderDropzone"] button { background: linear-gradient(135deg,#ff6fc8,#a97bff) !important; border:none !important; border-radius: 12px !important; }
[data-testid="stFileUploaderDropzone"] button * { color:#fff !important; }

/* ---------- KARTU STATISTIK ---------- */
.stat { background: linear-gradient(135deg,#ff8fd6,#a97bff); border-radius: 22px; padding: 18px; text-align:center;
        box-shadow: 0 10px 22px rgba(169,123,255,.35); animation: rise .7s ease both; }
.stat .l { font-size: 12px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: rgba(255,255,255,.9); }
.stat .v { font-family:'Fredoka',sans-serif; font-size: 30px; font-weight: 700; color:#fff; }

.result-big { text-align:center; }
.result-big .emo { font-size: 74px; animation: floaty 3.5s ease-in-out infinite; }
.result-big .name { font-family:'Fredoka',sans-serif; font-size: 42px; font-weight: 700; }

.chip { display:inline-block; margin: 4px 6px 4px 0; padding: 6px 14px; border-radius: 999px; font-weight: 600; font-size: 14px;
        background: #fff; color:#7b3fa8; border: 1.5px solid #f1b6e2; }

/* ---------- KOTAK PENJELASAN HASIL ---------- */
.interp { background: rgba(255,255,255,.78); border: 1.5px solid #f1b6e2; border-left: 7px solid #ff6fc8;
          border-radius: 20px; padding: 16px 22px; margin: 14px 0; color:#4a2a6b; animation: rise .7s ease both; }
.interp .ititle { font-family:'Fredoka',sans-serif; font-size: 20px; font-weight: 700; color:#7b3fa8; margin-bottom: 6px; }
.interp ul { margin: 0; padding-left: 20px; }
.interp li { margin: 6px 0; font-size: 15px; line-height: 1.5; }
.interp b { color:#7b3fa8; }
.interp.active { border-left-color:#a97bff; box-shadow: 0 8px 22px rgba(169,123,255,.3); }
.lvl { display:inline-block; padding: 2px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; margin-left: 6px; }
.lvl-1 { background:#d6f5e3; color:#17613f; }
.lvl-2 { background:#e3dcff; color:#4b2d9a; }
.lvl-3 { background:#fff0c7; color:#7a5200; }
.lvl-4 { background:#ffdbe3; color:#9c1d3d; }

/* ---------- TEKS BAWAAN STREAMLIT DI LATAR TERANG -> UNGU TUA ---------- */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
[data-testid="stSpinner"], [data-testid="stSpinner"] *,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span,
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"],
[data-testid="stAlert"] *, [data-testid="stNotification"] *,
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * { color: #4a2a6b !important; }
::placeholder { color: #a58bc4 !important; opacity: 1; }

/* teks putih DI DALAM bentuk ungu/pink tetap putih */
.stButton > button *, .stFormSubmitButton > button *,
div[role="radiogroup"] label:has(input:checked) *,
[data-testid="stFileUploaderDropzone"] button *, .step-num { color: #ffffff !important; }

.footer-center { text-align:center; color:#7b3fa8; font-size: 13px; margin-top: 30px; }
hr { border-color: rgba(160,110,210,.25) !important; }
</style>
""", unsafe_allow_html=True)


# ---------- BUAH MELAYANG DI LATAR ----------
_bg = [("🍎", 4, 8, 54, 0), ("🍒", 14, 82, 40, 1.2), ("🍇", 30, 3, 48, 2.1), ("🍊", 44, 90, 58, .6),
       ("🥝", 58, 6, 44, 1.8), ("🥭", 70, 86, 56, 2.6), ("🍌", 84, 12, 52, .9), ("🍓", 88, 78, 46, 1.5),
       ("🌰", 22, 94, 36, 3.0), ("🍎", 50, 48, 30, 2.4), ("🍇", 94, 45, 38, .3)]
st.markdown(
    '<div class="fruit-bg">' + "".join(
        f'<span style="top:{t}%; left:{l}%; font-size:{fs}px; animation-delay:{d}s;">{e}</span>'
        for e, t, l, fs, d in _bg) + '</div>',
    unsafe_allow_html=True)

# ==========================
# GLOBALS & SESSION INIT
# ==========================
for key, default in {
    'page': 'home', 'user_name': '', 'user_campus': '', 'photo_source': 'upload',
    'used_feature': False, 'last_mode': None, 'last_pred': None, 'last_conf': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/najwa_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/best_model.h5")
    return yolo_model, classifier

try:
    yolo_model, classifier = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    yolo_model, classifier = None, None

# ==========================
# CLASS NAMES
# ==========================
class_names = ["Apel", "Cherry", "Grapes", "Jeruk", "Kiwi", "Mangga", "Pisang", "Sawo", "Strawberry"]
fruit_emoji = {"Apel": "🍎", "Cherry": "🍒", "Grapes": "🍇", "Jeruk": "🍊", "Kiwi": "🥝",
               "Mangga": "🥭", "Pisang": "🍌", "Sawo": "🌰", "Strawberry": "🍓"}

# ==========================
# HELPERS
# ==========================
def dots(active):
    icons = ["🍎", "🍌", "🥝", "🍇", "🍓"]
    html = "".join(f'<div class="dot {"on" if i == active else ""}">{e}</div>' for i, e in enumerate(icons))
    st.markdown(f'<div class="dots">{html}</div>', unsafe_allow_html=True)


def go(page):
    st.session_state['page'] = page
    st.rerun()


def stat(label, value):
    return f'<div class="stat"><div class="l">{label}</div><div class="v">{value}</div></div>'


def level_info(pct):
    if pct >= 90:
        return "Sangat yakin", "lvl-1"
    if pct >= 70:
        return "Cukup yakin", "lvl-2"
    if pct >= 50:
        return "Kurang yakin", "lvl-3"
    return "Ragu-ragu", "lvl-4"


def badge(pct):
    label, cls = level_info(pct)
    return f'<span class="lvl {cls}">{label}</span>'


def interpret_classification(pred, conf, probs):
    """Penjelasan hasil klasifikasi. probs = [(nama, persen), ...] urut terbesar."""
    emo = fruit_emoji.get(pred, "🍎")
    return ('<div class="interp"><div class="ititle">📖 Penjelasan Klasifikasi</div><ul>'
            f'<li>Berdasarkan hasil klasifikasi, AI menebak buah di foto itu adalah <b>{emo} {pred}</b> '
            f'dengan tingkat keyakinan <b>{conf:.2f}%</b>{badge(conf)}</li></ul></div>')


def interpret_detection(labels, confs):
    """Penjelasan hasil deteksi YOLO. confs dalam persen."""
    grouped = {}
    for l, c in zip(labels, confs):
        grouped.setdefault(l, []).append(c)
    parts = [f'<b>{len(v)} {fruit_emoji.get(l, "🍎")} {l}</b> dengan tingkat keyakinan <b>{np.mean(v):.1f}%</b>'
             for l, v in grouped.items()]
    kalimat = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " dan " + parts[-1]
    low = sum(1 for c in confs if c < 50)
    low_note = (f'<li>⚠️ Ada <b>{low} objek</b> dengan keyakinan di bawah 50%. Bisa jadi itu salah deteksi, '
                'cek kotaknya pada gambar.</li>') if low else ""
    return ('<div class="interp"><div class="ititle">📖 Hasil Deteksi</div><ul>'
            f'<li>Berdasarkan hasil deteksi ditemukan {kalimat}.</li>'
            f'{low_note}'
            '<li>🟪 <b>Kotak berwarna</b> pada gambar menandai posisi tiap buah. Tulisan di atas kotak adalah jenis buah '
            'dan angka keyakinannya (misal <b>0.87</b> berarti 87%).</li>'
            '</ul></div>')


def no_detection_note():
    return ('<div class="interp"><div class="ititle">📖 Kenapa tidak ada yang terdeteksi?</div><ul>'
            '<li>Buah di foto mungkin bukan salah satu dari 9 buah yang dikenal AI.</li>'
            '<li>Foto terlalu gelap, buram, atau buahnya terlalu kecil.</li>'
            '<li>💡 Coba foto dengan cahaya terang dan buah terlihat jelas.</li></ul></div>')


def donut_note(benar, salah):
    simpul = ""
    if benar >= 90:
        simpul = "AI sangat yakin dengan hasil ini dan hasilnya dapat dipercaya."
    elif benar >= 70:
        simpul = "keyakinan jauh lebih besar daripada keraguan dan hasil ini cukup dapat dipercaya."
    elif benar >= 50:
        simpul = ("keyakinan sedikit lebih besar daripada keraguan, namun hasil ini masih bisa keliru "
                  "dan sebaiknya dicek kembali dengan foto lain.")
    kesimpulan = f'<li>Sehingga dapat disimpulkan bahwa {simpul}</li>' if simpul else ""
    return ('<div class="interp"><div class="ititle">📖 Penjelasan Donut Chart</div><ul>'
            f'<li>Berdasarkan visualisasi di atas dapat diketahui bahwa keyakinan pada hasil sebesar <b>{benar:.1f}%</b> '
            f'dan keraguannya sebesar <b>{salah:.1f}%</b>.</li>'
            f'{kesimpulan}</ul></div>')


def mode_info(active):
    yolo = ('<div class="interp {a}"><div class="ititle">🍊 Apa itu Deteksi Objek (YOLO)?</div><ul>'
            '<li>AI mencari dan menandai <b>di mana</b> buah berada pada gambar menggunakan kotak, sekaligus menyebutkan jenisnya.</li>'
            '<li>Cocok untuk foto yang berisi <b>lebih dari satu buah</b>.</li></ul></div>')
    cnn = ('<div class="interp {a}"><div class="ititle">🍇 Apa itu Klasifikasi Gambar (CNN)?</div><ul>'
           '<li>AI menebak <b>jenis</b> buah dari satu foto, lengkap dengan persentase kemungkinan tiap jenis buah.</li>'
           '<li>Cocok untuk foto <b>satu buah</b> yang terlihat jelas.</li></ul></div>')
    return (yolo.format(a="active" if active == "Deteksi Objek (YOLO)" else ""),
            cnn.format(a="active" if active == "Klasifikasi Gambar" else ""))


# ==========================
# PAGES
# ==========================
def halaman_awal():
    dots(0)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(
            '<div style="padding-top:30px; animation: rise .7s ease both;">'
            '<span class="badge">🍇 Proyek Deteksi &amp; Klasifikasi Buah</span>'
            '<div class="big-title">Kenali 9 Jenis Buah<br>dengan <span class="grad-text">Kekuatan AI</span></div>'
            '<div class="sub">Unggah foto buah, biarkan YOLO menemukan posisinya dan CNN menebak jenisnya. '
            'Cepat, lucu, dan gampang dipakai ✨</div></div>', unsafe_allow_html=True)
        if st.button("🍎 Mulai Sekarang", key="start"):
            go('register')
    with right:
        tiles = "".join(
            f'<div class="fruit-tile" style="animation-delay:{i*0.3}s"><div class="e">{fruit_emoji[n]}</div><div class="n">{n}</div></div>'
            for i, n in enumerate(["Apel", "Cherry", "Grapes", "Jeruk", "Kiwi", "Mangga", "Pisang", "Sawo", "Strawberry"])
        )
        st.markdown(f'<div class="glass"><div class="fruit-grid">{tiles}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="h2" style="text-align:center;">Cara Kerjanya</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3, gap="medium")
    steps = [("1", "Isi Registrasi", "Masukkan nama dan kampusmu terlebih dahulu."),
             ("2", "Unggah Gambar", "Pilih mode deteksi atau klasifikasi, lalu upload foto buah."),
             ("3", "Lihat Hasil", "Dapatkan hasil analisis AI dan beri feedback.")]
    for col, (n, t, d) in zip((c1, c2, c3), steps):
        col.markdown(f'<div class="glass step-card"><div class="step-num">{n}</div><h4>{t}</h4><p>{d}</p></div>',
                     unsafe_allow_html=True)


def halaman_registrasi():
    dots(1)
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(
            '<div class="glass" style="text-align:center; padding:50px 24px;">'
            '<div style="font-size:80px; animation: floaty 4s ease-in-out infinite;">🍎🍌🍇</div>'
            '<div class="h2">Halo, Teman Baru!</div>'
            '<div class="muted">Kenalan dulu yuk sebelum masuk ke dashboard.</div></div>',
            unsafe_allow_html=True)
    with right:
        st.markdown('<div class="h2">🍌 Registrasi Singkat</div><div class="muted">Isi data berikut untuk melanjutkan</div>',
                    unsafe_allow_html=True)
        st.write("")
        with st.form("form_reg", border=False):
            name = st.text_input("👤 Nama Lengkap", value=st.session_state.get('user_name', ''))
            campus = st.text_input("🏫 Nama Kampus", value=st.session_state.get('user_campus', ''))
            submitted = st.form_submit_button("Masuk ke Dashboard 🍊", use_container_width=True)
            if submitted:
                if name.strip() == '' or campus.strip() == '':
                    st.warning("Mohon isi Nama dan Nama Kampus sebelum melanjutkan.")
                else:
                    st.session_state['user_name'] = name.strip()
                    st.session_state['user_campus'] = campus.strip()
                    go('input')
        if st.button("⬅️ Kembali ke Beranda", use_container_width=True):
            go('home')


def halaman_input():
    dots(2)
    st.markdown(
        '<div style="text-align:center; animation: rise .7s ease both;">'
        '<div class="big-title" style="font-size:42px;">Mau pakai <span class="grad-text">foto dari mana</span>?</div>'
        '<div class="sub">Pilih cara memasukkan foto buah, lalu lanjut ke dashboard untuk dianalisis AI.</div></div>',
        unsafe_allow_html=True)
    opsi = [
        ("upload", "📂", "Unggah Foto", "Pilih foto buah dari galeri atau penyimpanan perangkatmu."),
        ("camera", "📸", "Ambil Foto", "Foto buah langsung memakai kamera HP atau laptop."),
    ]
    cols = st.columns(2, gap="large")
    for col, (kode, emo, judul, desc) in zip(cols, opsi):
        with col:
            st.markdown(
                f'<div class="glass step-card" style="text-align:center;"><div style="font-size:56px;">{emo}</div>'
                f'<h4>{judul}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
            if st.button(f"{emo} {judul}", key=f"src_{kode}", use_container_width=True):
                st.session_state['photo_source'] = kode
                go('main')
    st.write("")
    _, mid, _ = st.columns([2, 2, 2])
    with mid:
        if st.button("⬅️ Kembali ke Registrasi", use_container_width=True):
            go('register')


def halaman_main():
    dots(3)
    top1, top2 = st.columns([4, 1])
    with top1:
        st.markdown(
            f'<div class="glass tight"><span class="h2">🍓 Dashboard Buah</span> &nbsp; '
            f'<span class="muted">Halo <b>{st.session_state.get("user_name","")}</b> dari '
            f'<b>{st.session_state.get("user_campus","")}</b> 👋</span></div>', unsafe_allow_html=True)
    with top2:
        if st.button("⬅️ Sumber Foto", use_container_width=True):
            go('input')

    st.write("")
    st.markdown('<div class="muted" style="margin-bottom:6px;">Pilih mode analisis</div>', unsafe_allow_html=True)
    menu_pick = st.radio("Mode", ["🍊 Deteksi Objek (YOLO)", "🍇 Klasifikasi Gambar"],
                         horizontal=True, label_visibility="collapsed")
    menu = "Deteksi Objek (YOLO)" if "Deteksi" in menu_pick else "Klasifikasi Gambar"
    info_yolo, info_cnn = mode_info(menu)
    ci1, ci2 = st.columns(2, gap="medium")
    ci1.markdown(info_yolo, unsafe_allow_html=True)
    ci2.markdown(info_cnn, unsafe_allow_html=True)

    st.write("")
    sumber = st.session_state.get('photo_source', 'upload')
    sumber_nama = "📂 Unggah Foto" if sumber == "upload" else "📸 Ambil Foto"
    st.markdown(f'<div class="muted">Sumber foto: <b>{sumber_nama}</b> (ganti lewat tombol "Sumber Foto" di kanan atas)</div>',
                unsafe_allow_html=True)
    st.write("")
    uploaded_file = None
    if sumber == "upload":
        uploaded_file = st.file_uploader("📂 Unggah gambar buah (JPG / PNG)", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("📸 Arahkan kamera ke buah, lalu ambil foto")

    if uploaded_file is None:
        st.markdown('<div class="glass tight" style="text-align:center;">👆 Masukkan foto terlebih dahulu '
                    'untuk mulai deteksi atau klasifikasi.</div>', unsafe_allow_html=True)
        return

    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="h2">🍎 Gambar Kamu</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)

    stats = []

    # ---------- DETEKSI ----------
    if menu == "Deteksi Objek (YOLO)":
        if yolo_model is None:
            st.error("Model YOLO belum tersedia.")
        else:
            with st.spinner("🍊 AI sedang mendeteksi objek... harap tunggu sebentar!"):
                results = yolo_model(np.array(img))
                result_img = results[0].plot()[:, :, ::-1]  # BGR -> RGB
                boxes = results[0].boxes

            with col2:
                st.markdown('<div class="h2">🍊 Hasil Deteksi</div>', unsafe_allow_html=True)
                if boxes is not None and len(boxes) > 0:
                    st.image(result_img, use_container_width=True)

                    data, confidences, labels = [], [], []
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = results[0].names[cls]
                        labels.append(label)
                        data.append({
                            "Jenis Buah": label, "Tingkat Keyakinan": f"{conf*100:.2f}%",
                            "Kiri (x1)": int(xyxy[0]), "Atas (y1)": int(xyxy[1]),
                            "Kanan (x2)": int(xyxy[2]), "Bawah (y2)": int(xyxy[3])
                        })
                        confidences.append(conf * 100)

                    avg_conf = float(np.mean(confidences))
                    st.session_state.update(last_mode='detection', last_pred=f"{len(boxes)} objek terdeteksi",
                                            last_conf=avg_conf, used_feature=True)
                    stats = [("Mode", "Deteksi"), ("Objek", len(boxes)), ("Rata-rata Confidence", f"{avg_conf:.1f}%")]

                    chips = "".join(f'<span class="chip">{fruit_emoji.get(l, "🍎")} {l} × {n}</span>'
                                    for l, n in Counter(labels).items())
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Tidak ada objek yang terdeteksi dalam gambar ini.")
                    st.markdown(no_detection_note(), unsafe_allow_html=True)
                    data = []

            if data:
                st.markdown('<div class="h2">🍌 Rincian Hasil Deteksi</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                st.markdown(interpret_detection(labels, confidences), unsafe_allow_html=True)

    # ---------- KLASIFIKASI ----------
    else:
        if classifier is None:
            st.error("Model klasifikasi belum tersedia.")
        else:
            with st.spinner("🍇 Sedang melakukan klasifikasi gambar..."):
                target_size = classifier.input_shape[1:3]
                img_resized = img.resize(target_size)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(img_array)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction)) * 100.0
                pred_name = class_names[class_index]

            with col2:
                st.markdown('<div class="h2">🍇 Hasil Klasifikasi</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="glass result-big"><div class="emo">{fruit_emoji.get(pred_name, "🍎")}</div>'
                    f'<div class="name grad-text">{pred_name}</div>'
                    f'<div class="muted">Keyakinan model: <b>{confidence:.2f}%</b></div></div>',
                    unsafe_allow_html=True)
                st.progress(min(max(confidence / 100.0, 0.0), 1.0))
                probs_sorted = sorted(((class_names[i], float(prediction[0][i]) * 100) for i in range(len(class_names))),
                                      key=lambda x: -x[1])
                st.markdown(interpret_classification(pred_name, confidence, probs_sorted), unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Kelas": class_names,
                "Probabilitas (%)": [float(prediction[0][i]) * 100 for i in range(len(class_names))]
            }).sort_values("Probabilitas (%)", ascending=True)
            colors = ['#ff6fc8' if k == pred_name else '#cdb2ff' for k in prob_df["Kelas"]]
            fig_bar = px.bar(prob_df, x="Probabilitas (%)", y="Kelas", orientation="h")
            fig_bar.update_traces(marker_color=colors, text=[f"{v:.1f}%" for v in prob_df["Probabilitas (%)"]],
                                  textposition="outside", cliponaxis=False,
                                  textfont=dict(color='#4a2a6b', size=13))
            fig_bar.update_layout(height=380, margin=dict(l=10, r=60, t=10, b=10),
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='#4a2a6b', size=14), bargap=0.25)
            fig_bar.update_xaxes(range=[0, 115], tickfont=dict(color='#4a2a6b'), title_font=dict(color='#4a2a6b'),
                                 gridcolor='rgba(123,63,168,0.15)', automargin=True)
            fig_bar.update_yaxes(title_text="", showticklabels=True, ticklabelposition="outside left",
                                 tickfont=dict(color='#4a2a6b', size=15), automargin=True)
            st.markdown('<div class="h2">🥝 Probabilitas per Kelas</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False}, theme=None)
            st.markdown('<div class="interp"><div class="ititle">📖 Penjelasan Grafik</div><ul>'
                        f'<li>Berdasarkan grafik di atas, dapat diketahui batang berwarna <b>pink</b> merupakan tebakan utama AI, '
                        f'yaitu <b>{fruit_emoji.get(pred_name, "🍎")} {pred_name}</b> dengan nilai probabilitas sebesar <b>{confidence:.1f}%</b>, '
                        'sedangkan batang yang berwarna <b>ungu muda</b> adalah tebakan kemungkinan buah lain.</li></ul></div>',
                        unsafe_allow_html=True)

            st.session_state.update(last_mode='classification', last_pred=pred_name,
                                    last_conf=confidence, used_feature=True)
            stats = [("Mode", "Klasifikasi"), ("Prediksi", pred_name), ("Keyakinan", f"{confidence:.1f}%")]

    # ---------- RINGKASAN + DONUT ----------
    if stats and st.session_state.get('used_feature', False):
        st.write("")
        st.markdown('<div class="h2">🥭 Ringkasan</div>', unsafe_allow_html=True)
        sc = st.columns(len(stats), gap="medium")
        for c, (lbl, val) in zip(sc, stats):
            c.markdown(stat(lbl, val), unsafe_allow_html=True)

        val = st.session_state.get('last_conf', None)
        if val is not None:
            benar = float(val)
            salah = max(0.0, 100.0 - benar)
            pie_data = pd.DataFrame({"Hasil": ["Benar", "Salah"], "Persentase": [benar, salah]})
            fig = px.pie(pie_data, values='Persentase', names='Hasil', color='Hasil',
                         color_discrete_map={'Benar': '#a97bff', 'Salah': '#ffc2e6'}, hole=0.55)
            fig.update_traces(textinfo='label+percent', textfont=dict(color='#4a2a6b', size=15),
                              marker=dict(line=dict(color='#ffffff', width=3)))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=0, b=0),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#4a2a6b', size=14),
                                  legend=dict(font=dict(color='#4a2a6b', size=15)))
            st.write("")
            st.markdown('<div class="h2" style="text-align:center;">🍒 Persen Akurasi</div>', unsafe_allow_html=True)
            _, d2, _ = st.columns([1, 2, 1])
            with d2:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, theme=None)
            st.markdown(donut_note(benar, salah), unsafe_allow_html=True)

        _, b, _ = st.columns([2, 2, 2])
        with b:
            if st.button("🍓 Berikan Feedback", use_container_width=True):
                go('feedback')


def halaman_feedback():
    dots(4)
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(
            '<div class="glass" style="text-align:center; padding:50px 24px;">'
            '<div style="font-size:90px; animation: floaty 4s ease-in-out infinite;">🍓</div>'
            '<div class="h2">Terima Kasih!</div>'
            '<div class="muted">Masukanmu bikin aplikasi ini makin keren.</div></div>',
            unsafe_allow_html=True)
    with right:
        st.markdown('<div class="h2">🍓 Feedback Pengguna</div><div class="muted">Beri rating dan saranmu</div>',
                    unsafe_allow_html=True)
        st.write("")
        with st.form("form_feedback", border=False):
            pilihan = st.radio("Rating", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                               index=4, horizontal=True)
            rating = len(pilihan)
            comment = st.text_area("Komentar / Saran", height=120)
            submitted = st.form_submit_button("Kirim Feedback 🍓", use_container_width=True)

            if submitted:
                row = {
                    'timestamp': datetime.datetime.now().isoformat(),
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
                    st.markdown(f"<div class='footer-center'>Feedback tersimpan ke <b>{csv_path}</b></div>",
                                unsafe_allow_html=True)
                    st.session_state['used_feature'] = False
                except Exception as e:
                    st.error(f"Gagal menyimpan feedback: {e}")

        if st.button("⬅️ Kembali ke Dashboard", use_container_width=True):
            go('main')


# ==========================
# ROUTING
# ==========================
if st.session_state['page'] == 'home':
    halaman_awal()
elif st.session_state['page'] == 'register':
    halaman_registrasi()
elif st.session_state['page'] == 'input':
    halaman_input()
elif st.session_state['page'] == 'main':
    halaman_main()
elif st.session_state['page'] == 'feedback':
    halaman_feedback()
else:
    halaman_awal()

# FOOTER
st.markdown(
    "<div class='footer-center'>created by <b>Najwa Putri Aqilah | 2408108010053</b><br>"
    "Proyek UAS Praktikum Pemrograman Big Data | Universitas Syiah Kuala</div>",
    unsafe_allow_html=True)
