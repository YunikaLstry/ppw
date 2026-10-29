
# ============================================================
# APLIKASI KLASIFIKASI BERITA DETIK.COM
# Word2Vec Skip-gram + Gaussian Naive Bayes
# ============================================================

import re
import html
import ipaddress
import socket
from urllib.parse import urlparse
from pathlib import Path

import joblib
import numpy as np
import requests
import streamlit as st

from bs4 import BeautifulSoup
from gensim.models import Word2Vec

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Berita Detik.com",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(135deg, #102a43, #176b87);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 24px;
}

.hero h1 {
    color: white;
    font-size: 30px;
    margin-bottom: 10px;
}

.hero p {
    color: #e6f4f7;
    font-size: 15px;
    margin-bottom: 0;
}

.step-card {
    background: white;
    border: 1px solid #e5eaf0;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}

.result-finance {
    background: #e8f7ed;
    border: 1px solid #b7e4c7;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
}

.result-sport {
    background: #eaf3ff;
    border: 1px solid #b6d4fe;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
}

.small-note {
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. LOKASI DAN PEMUATAN MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "hasil_skipgram"

W2V_PATH = MODEL_DIR / "w2v_model.model"
GNB_PATH = MODEL_DIR / "gnb_model.pkl"

LABEL_MAP = {
    0: "Finance",
    1: "Sport"
}


@st.cache_resource
def load_models():
    """Memuat model Word2Vec dan Gaussian Naive Bayes."""

    if not W2V_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {W2V_PATH}"
        )

    if not GNB_PATH.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {GNB_PATH}"
        )

    word2vec = Word2Vec.load(str(W2V_PATH))
    gaussian_nb = joblib.load(GNB_PATH)

    if (
        hasattr(gaussian_nb, "n_features_in_")
        and gaussian_nb.n_features_in_ != word2vec.vector_size
    ):
        raise ValueError(
            "Jumlah fitur model tidak sesuai. "
            "Pastikan GaussianNB dilatih menggunakan "
            "vektor rata-rata Word2Vec tanpa PCA."
        )

    return word2vec, gaussian_nb


@st.cache_resource
def load_preprocessing_tools():
    """Menyiapkan stopword remover dan stemmer."""

    stopword_remover = (
        StopWordRemoverFactory()
        .create_stop_word_remover()
    )

    stemmer = StemmerFactory().create_stemmer()

    # Samakan kamus ini dengan preprocessing saat training.
    normalisasi = {
        "yg": "yang",
        "dgn": "dengan",
        "utk": "untuk",
        "dr": "dari",
        "tdk": "tidak",
        "gak": "tidak",
        "ga": "tidak",
        "nggak": "tidak",
        "krn": "karena",
        "sdh": "sudah",
        "blm": "belum"
    }

    return stopword_remover, stemmer, normalisasi


try:
    w2v_model, gnb_model = load_models()
    stopword_remover, stemmer, normalisasi = (
        load_preprocessing_tools()
    )
    MODEL_READY = True
    MODEL_ERROR = None

except Exception as error:
    MODEL_READY = False
    MODEL_ERROR = str(error)


# ============================================================
# 3. VALIDASI URL DETIK.COM
# ============================================================

def validate_url(url):
    """Memastikan URL berasal dari domain Detik.com."""

    try:
        parsed = urlparse(url.strip())

        if parsed.scheme not in ("http", "https"):
            return False, "URL harus diawali http:// atau https://."

        hostname = (parsed.hostname or "").lower().rstrip(".")

        if not (
            hostname == "detik.com"
            or hostname.endswith(".detik.com")
        ):
            return False, "Gunakan URL artikel dari Detik.com."

        # Menolak alamat IP lokal dan jaringan privat.
        try:
            ip = ipaddress.ip_address(hostname)
            if not ip.is_global:
                return False, "Alamat IP nonpublik tidak diizinkan."
        except ValueError:
            pass

        try:
            addresses = socket.getaddrinfo(
                hostname, None, type=socket.SOCK_STREAM
            )

            for address in addresses:
                ip_text = address[4][0].split("%")[0]
                ip = ipaddress.ip_address(ip_text)

                if not ip.is_global:
                    return False, "Domain mengarah ke alamat nonpublik."

        except socket.gaierror:
            pass

        return True, ""

    except Exception:
        return False, "Format URL tidak valid."


# ============================================================
# 4. SCRAPING ARTIKEL
# ============================================================

def get_article(url):
    """Mengambil judul dan isi artikel dari halaman Detik.com."""

    valid, message = validate_url(url)

    if not valid:
        raise ValueError(message)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
        allow_redirects=True
    )

    response.raise_for_status()

    # Pastikan URL pengalihan masih menuju Detik.com.
    valid, message = validate_url(response.url)

    if not valid:
        raise ValueError(
            "Halaman dialihkan ke URL yang tidak diizinkan. "
            + message
        )

    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = (
        soup.select_one("h1.detail__title")
        or soup.select_one("h1")
        or soup.select_one('meta[property="og:title"]')
    )

    if title_tag:
        if title_tag.name == "meta":
            title = title_tag.get("content", "").strip()
        else:
            title = title_tag.get_text(" ", strip=True)
    else:
        title = "Judul tidak ditemukan"

    body = (
        soup.select_one("div.detail__body-text")
        or soup.select_one("div.itp_bodycontent")
        or soup.select_one("article")
    )

    if body:
        for tag in body.select(
            "script, style, aside, table, figure, iframe, "
            "button, form, nav"
        ):
            tag.decompose()

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in body.find_all("p")
            if p.get_text(" ", strip=True)
        ]
    else:
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(" ", strip=True)) > 35
        ]

    text = " ".join(paragraphs)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 100:
        raise ValueError(
            "Isi artikel tidak berhasil diambil atau terlalu pendek. "
            "Coba URL artikel Detik.com lainnya."
        )

    return title, text


# ============================================================
# 5. PREPROCESSING TEKS
# ============================================================

def preprocess_text(text):
    """Membersihkan teks dan mengubahnya menjadi token."""

    # A. Case folding
    text = html.unescape(str(text)).lower()

    # B. Cleaning
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b\S+@\S+\.\S+\b", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # C. Tokenisasi
    tokens = text.split()

    # D. Normalisasi kata
    tokens = [
        normalisasi.get(token, token)
        for token in tokens
    ]

    # E. Stopword removal
    text_without_stopwords = stopword_remover.remove(
        " ".join(tokens)
    )

    # F. Stemming
    stemmed_text = stemmer.stem(text_without_stopwords)

    return {
        "case_folding": str(text).lower(),
        "cleaning": text,
        "tokens": tokens,
        "stopword_removed": text_without_stopwords,
        "stemming": stemmed_text,
        "final_tokens": stemmed_text.split()
    }


# ============================================================
# 6. EKSTRAKSI FITUR WORD2VEC
# ============================================================

def document_vector(tokens):
    """
    Menghitung rata-rata vektor kata dari model Word2Vec.
    Hasilnya menjadi fitur untuk Gaussian Naive Bayes.
    """

    known_words = [
        token for token in tokens
        if token in w2v_model.wv
    ]

    if not known_words:
        raise ValueError(
            "Tidak ada kata yang cocok dengan kosakata Word2Vec. "
            "Periksa preprocessing atau coba artikel lain."
        )

    vectors = np.array(
        [w2v_model.wv[token] for token in known_words]
    )

    return np.mean(vectors, axis=0), known_words


# ============================================================
# 7. PREDIKSI KATEGORI
# ============================================================

def classify_article(text):
    """Menjalankan preprocessing, Word2Vec, dan klasifikasi."""

    steps = preprocess_text(text)

    vector, known_words = document_vector(
        steps["final_tokens"]
    )

    features = vector.reshape(1, -1)

    prediction = gnb_model.predict(features)[0]
    probabilities = gnb_model.predict_proba(features)[0]

    probability_map = {
        int(label): float(probability)
        for label, probability in zip(
            gnb_model.classes_, probabilities
        )
    }

    prediction = int(prediction)

    if prediction not in LABEL_MAP:
        raise ValueError(
            f"Label {prediction} tidak sesuai dengan LABEL_MAP."
        )

    return {
        "category": LABEL_MAP[prediction],
        "finance": probability_map.get(0, 0.0),
        "sport": probability_map.get(1, 0.0),
        "steps": steps,
        "known_words": known_words
    }


# ============================================================
# 8. TAMPILAN UTAMA
# ============================================================

st.markdown("""
<div class="hero">
    <h1>📰 Klasifikasi Berita Detik.com</h1>
    <p>
        Prediksi kategori berita secara otomatis menggunakan
        Word2Vec Skip-gram dan Gaussian Naive Bayes.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🔗 Masukkan URL Artikel")

st.write(
    "Tempelkan tautan artikel Detik.com. "
    "Aplikasi akan mengambil isi berita dan memprediksi kategorinya."
)

if not MODEL_READY:
    st.error("Model gagal dimuat.")
    st.code(MODEL_ERROR or "Kesalahan tidak diketahui")
    st.stop()

with st.form("article_form"):
    url = st.text_input(
        "URL artikel",
        placeholder="https://finance.detik.com/...",
    )

    submit = st.form_submit_button(
        "🔍 Analisis Berita",
        use_container_width=True
    )


# ============================================================
# 9. MENAMPILKAN HASIL
# ============================================================

if submit:
    if not url.strip():
        st.warning("Masukkan URL artikel terlebih dahulu.")

    else:
        try:
            with st.spinner("Langkah 1/3 — Mengambil artikel..."):
                title, article_text = get_article(url.strip())

            st.success("Artikel berhasil diambil.")

            st.markdown("### 📄 Artikel yang Dianalisis")
            st.write(f"**Judul:** {title}")

            with st.expander("Lihat isi artikel"):
                st.write(article_text)

            with st.spinner("Langkah 2/3 — Memproses teks..."):
                steps = preprocess_text(article_text)

            with st.spinner("Langkah 3/3 — Melakukan klasifikasi..."):
                result = classify_article(article_text)

            st.divider()
            st.markdown("## 🎯 Hasil Klasifikasi")

            if result["category"] == "Finance":
                st.markdown(
                    """
                    <div class="result-finance">
                        <div style="font-size:42px">💰</div>
                        <h2 style="color:#166534;margin:5px 0">
                            FINANCE
                        </h2>
                        <p style="color:#166534;margin:0">
                            Model memprediksi artikel termasuk kategori
                            Finance.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="result-sport">
                        <div style="font-size:42px">⚽</div>
                        <h2 style="color:#1d4ed8;margin:5px 0">
                            SPORT
                        </h2>
                        <p style="color:#1d4ed8;margin:0">
                            Model memprediksi artikel termasuk kategori
                            Sport.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("### 📊 Probabilitas Prediksi")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Finance",
                    f"{result['finance'] * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Sport",
                    f"{result['sport'] * 100:.2f}%"
                )

            chart_data = {
                "Kategori": ["Finance", "Sport"],
                "Probabilitas (%)": [
                    result["finance"] * 100,
                    result["sport"] * 100
                ]
            }

            st.bar_chart(
                chart_data,
                x="Kategori",
                y="Probabilitas (%)"
            )

            # Penjelasan tahapan analisis
            st.divider()
            st.markdown("## 🔬 Tahapan Analisis")

            with st.expander(
                "1. Scraping — Pengambilan artikel",
                expanded=False
            ):
                st.write(
                    "Aplikasi mengambil judul dan paragraf artikel "
                    "dari URL Detik.com."
                )
                st.write(f"Panjang teks: {len(article_text):,} karakter")

            with st.expander("2. Preprocessing — Pembersihan teks"):
                st.write(
                    "Teks dibersihkan, diubah menjadi huruf kecil, "
                    "ditokenisasi, dinormalisasi, dihapus stopword-nya, "
                    "dan dilakukan stemming."
                )

                st.markdown("**Hasil cleaning**")
                st.code(steps["cleaning"][:2000])

                st.markdown("**Hasil tokenisasi**")
                st.write(steps["tokens"][:100])

                st.markdown("**Hasil stopword removal**")
                st.code(steps["stopword_removed"][:2000])

                st.markdown("**Hasil stemming**")
                st.code(steps["stemming"][:2000])

            with st.expander("3. Word2Vec Skip-gram — Ekstraksi fitur"):
                st.write(
                    "Word2Vec mengubah kata menjadi vektor numerik. "
                    "Vektor dokumen dibuat dengan menghitung rata-rata "
                    "vektor kata yang ditemukan dalam kosakata model."
                )
                st.write(
                    f"Kata yang dikenali model: "
                    f"{len(result['known_words'])}"
                )
                st.write(
                    f"Dimensi vektor dokumen: "
                    f"{w2v_model.vector_size}"
                )
                st.write("Contoh kata yang dikenali:")
                st.write(result["known_words"][:50])

            with st.expander(
                "4. Gaussian Naive Bayes — Klasifikasi"
            ):
                st.write(
                    "Vektor dokumen diberikan kepada model Gaussian "
                    "Naive Bayes untuk memprediksi Finance atau Sport."
                )
                st.write(
                    f"Kategori hasil prediksi: **{result['category']}**"
                )
                st.write(
                    "Nilai probabilitas merupakan keluaran model dan "
                    "tidak menjamin prediksi selalu benar."
                )

        except requests.exceptions.HTTPError as error:
            st.error(f"Gagal mengakses halaman artikel: {error}")

        except requests.exceptions.RequestException as error:
            st.error(f"Masalah koneksi saat mengambil artikel: {error}")

        except Exception as error:
            st.error(f"Analisis gagal: {error}")


# ============================================================
# 10. INFORMASI METODE
# ============================================================

st.divider()

st.markdown("### 🧠 Metode yang Digunakan")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="step-card">
        <h4>🔤 Word2Vec Skip-gram</h4>
        <p>
        Mengubah kata menjadi representasi vektor numerik
        berdasarkan hubungan antarkata.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card">
        <h4>📈 Gaussian Naive Bayes</h4>
        <p>
        Menggunakan fitur numerik dari vektor dokumen untuk
        memprediksi kategori Finance atau Sport.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<p class="small-note">Hasil prediksi bergantung pada kualitas '
    'data pelatihan dan kesesuaian preprocessing dengan pipeline '
    'training model.</p>',
    unsafe_allow_html=True
)
