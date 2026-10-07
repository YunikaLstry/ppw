import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import trafilatura
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, confusion_matrix
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Klasifikasi Berita URL Detik.com | Finance vs Sport",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS untuk tampilan modern, elegan, dan profesional
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header card */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 26px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Metric card */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-num {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0284c7;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Prediction Result Cards */
    .result-badge-finance {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 14px 28px;
        border-radius: 14px;
        font-size: 1.55rem;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 8px 18px -3px rgba(16, 185, 129, 0.4);
        letter-spacing: 0.5px;
    }
    .result-badge-sport {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        padding: 14px 28px;
        border-radius: 14px;
        font-size: 1.55rem;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 8px 18px -3px rgba(59, 130, 246, 0.4);
        letter-spacing: 0.5px;
    }

    /* Article Meta Box */
    .article-meta-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-top: 15px;
        margin-bottom: 18px;
    }
    .article-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .article-submeta {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 12px;
    }

    /* Preprocessing Step Box */
    .step-box {
        background: #f8fafc;
        border-left: 4px solid #0284c7;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .step-title {
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .step-content {
        color: #475569;
        font-family: monospace;
        word-break: break-word;
    }

    /* Sidebar info */
    .sidebar-profile {
        text-align: center;
        padding: 6px 0;
    }
    .sidebar-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 8px;
    }
    .sidebar-desc {
        font-size: 0.85rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNGSI CACHING MODEL & DATA
# ==============================================================================
@st.cache_resource
def load_models():
    """Memuat model Word2Vec, PCA, dan Gaussian Naive Bayes"""
    w2v = Word2Vec.load("w2v_model.model")
    pca = joblib.load("pca_model.pkl")
    gnb = joblib.load("gnb_model.pkl")
    return w2v, pca, gnb

@st.cache_resource
def load_preprocessor():
    """Memuat stopword remover dan stemmer Sastrawi"""
    stop_factory = StopWordRemoverFactory()
    stopword_remover = stop_factory.create_stop_word_remover()
    
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    
    kamus_normalisasi = {
        'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'dr': 'dari',
        'sbg': 'sebagai', 'tdk': 'tidak', 'krn': 'karena', 'sdh': 'sudah',
        'blm': 'belum', 'match': 'tanding', 'coaching': 'pelatih',
        'coach': 'pelatih', 'strikers': 'striker', 'goal': 'gol',
        'percent': 'persen', 'company': 'perusahaan', 'tax': 'pajak'
    }
    return stopword_remover, stemmer, kamus_normalisasi

@st.cache_data
def load_datasets():
    """Memuat data mentah dan data hasil preprocessing"""
    df_raw = pd.read_csv("dataset_detik.csv") if os.path.exists("dataset_detik.csv") else None
    df_clean = pd.read_csv("detik_preprocessed.csv") if os.path.exists("detik_preprocessed.csv") else None
    df_pca = pd.read_csv("detik_pca_results.csv") if os.path.exists("detik_pca_results.csv") else None
    return df_raw, df_clean, df_pca

# Inisialisasi resource
try:
    w2v_model, pca_model, gnb_model = load_models()
    stopword_remover, stemmer, kamus_normalisasi = load_preprocessor()
    df_raw, df_clean, df_pca = load_datasets()
    models_ready = True
except Exception as e:
    models_ready = False
    load_error_msg = str(e)

# ==============================================================================
# FUNGSI SCRAPING DARI URL BERITA
# ==============================================================================
def scrape_article_from_url(url):
    """
    Mengekstrak teks berita, judul, dan metadata dari URL.
    Menggunakan Trafilatura dengan fallback ke requests + BeautifulSoup.
    """
    result = {
        "success": False,
        "url": url,
        "title": "Tidak diketahui",
        "author": "-",
        "date": "-",
        "text": "",
        "error": None
    }
    
    try:
        # 1. Coba download dengan Trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted_text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )
            metadata = trafilatura.extract_metadata(downloaded)
            
            if metadata:
                if metadata.title:
                    result["title"] = metadata.title
                if metadata.author:
                    result["author"] = metadata.author
                if metadata.date:
                    result["date"] = metadata.date
                    
            if extracted_text and len(extracted_text.strip()) > 50:
                result["text"] = extracted_text.strip()
                result["success"] = True
                return result
                
        # 2. Fallback: requests + BeautifulSoup (khusus Detik.com / umum)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Judul artikel Detik
            title_tag = soup.find("h1", class_="detail__title") or soup.find("h1")
            if title_tag:
                result["title"] = title_tag.get_text(strip=True)
                
            # Konten berita Detik biasanya ada di div class "detail__body-text"
            body_tag = soup.find("div", class_="detail__body-text")
            if body_tag:
                # Bersihkan tag yang tidak perlu di dalam konten (cth: video, banner, dsb)
                for unwanted in body_tag.find_all(["script", "style", "table", "aside"]):
                    unwanted.decompose()
                paragraphs = [p.get_text(strip=True) for p in body_tag.find_all("p") if p.get_text(strip=True)]
                content = " ".join(paragraphs)
            else:
                paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 25]
                content = " ".join(paragraphs)
                
            if content and len(content) > 50:
                result["text"] = content
                result["success"] = True
                return result
            else:
                result["error"] = "Teks artikel tidak berhasil ditemukan pada halaman tersebut."
        else:
            result["error"] = f"Gagal mengakses URL (HTTP Status {resp.status_code})."
            
    except Exception as e:
        result["error"] = str(e)
        
    return result

# ==============================================================================
# FUNGSI PIPELINE PREPROCESSING & KLASIFIKASI
# ==============================================================================
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_pipeline(raw_text):
    """Menjalankan 6 tahapan preprocessing NLP teks bahasa Indonesia"""
    steps = {}
    
    # 1. Case Folding
    c_fold = raw_text.lower()
    steps['case_folding'] = c_fold
    
    # 2. Text Cleaning
    cleaned = clean_text(c_fold)
    steps['cleaning'] = cleaned
    
    # 3. Tokenize
    tokens = cleaned.split()
    steps['tokenize'] = tokens
    
    # 4. Normalisasi
    normalized = [kamus_normalisasi.get(w, w) for w in tokens]
    steps['normalized'] = normalized
    
    # 5. Stopword Removal
    sw_removed = stopword_remover.remove(' '.join(normalized))
    steps['stopword_removed'] = sw_removed
    
    # 6. Stemming Sastrawi
    stemmed = stemmer.stem(sw_removed)
    final_tokens = stemmed.split()
    steps['stemmed'] = stemmed
    steps['final_tokens'] = final_tokens
    
    return steps

def get_document_vector(tokens, model):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(valid_vectors) == 0:
        return np.zeros(model.vector_size), []
    return np.mean(valid_vectors, axis=0), [w for w in tokens if w in model.wv]

def predict_from_text(text):
    steps = preprocess_pipeline(text)
    doc_vec, in_vocab_tokens = get_document_vector(steps['final_tokens'], w2v_model)
    doc_pca = pca_model.transform([doc_vec])
    pred = gnb_model.predict(doc_pca)[0]
    prob = gnb_model.predict_proba(doc_pca)[0]
    label_map = {0: "Finance", 1: "Sport"}
    
    return {
        "label": label_map[pred],
        "pred_num": pred,
        "prob_finance": prob[0],
        "prob_sport": prob[1],
        "steps": steps,
        "doc_vec": doc_vec,
        "doc_pca": doc_pca[0],
        "in_vocab_tokens": in_vocab_tokens
    }

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("<div class='sidebar-profile'>", unsafe_allow_html=True)
    if os.path.exists("foto.JPG"):
        try:
            profile_img = Image.open("foto.JPG")
            st.image(profile_img, width=120)
        except Exception:
            pass
    st.markdown("""
        <div class='sidebar-name'>Yunika Lestari</div>
        <div class='sidebar-desc'>NIM: <b>230411100047</b></div>
        <div class='sidebar-desc'>Kelas: <b>IF7A - Penambangan Web</b></div>
        <div class='sidebar-desc'>Universitas Trunojoyo Madura</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    menu = st.radio(
        "📌 Navigasi Menu",
        [
            "🌐 Prediksi via URL Berita",
            "🏠 Beranda & Ringkasan Proyek",
            "📊 Dataset & Preprocessing",
            "🧠 Model Word2Vec & PCA",
            "📈 Evaluasi Model (Naive Bayes)",
            "👤 Profil Pengembang"
        ],
        index=0  # Default langsung ke Prediksi via URL Berita
    )
    
    st.markdown("---")
    st.markdown("### 🏷️ 2 Kategori Target")
    st.markdown("""
    - 💰 **Finance** (Bisnis, Saham, Perbankan, Pajak, Pasar Modal)
    - ⚽ **Sport** (Sepakbola, Balap, Atletik, Kejuaraan, KONI)
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Status Artefak Model")
    if models_ready:
        st.success("✅ Seluruh Model Siap Digunakan")
        st.caption(f"• Vocab W2V: **{len(w2v_model.wv):,} kata**")
        st.caption(f"• Komponen PCA: **{pca_model.n_components_} Dimensi**")
        st.caption("• Klasifikasi: **Gaussian Naive Bayes**")
    else:
        st.error(f"❌ Gagal memuat model: {load_error_msg}")

# ==============================================================================
# HALAMAN 1: PREDIKSI VIA URL BERITA (UTAMA)
# ==============================================================================
if menu == "🌐 Prediksi via URL Berita":
    st.title("🌐 Klasifikasi Berita via URL (Scraping Otomatis)")
    st.markdown("""
    Sistem akan secara otomatis **mengambil (scrape) isi berita dari URL** yang dimasukkan,
    menjalankan 6 tahap preprocessing bahasa Indonesia, ekstraksi vektor Word2Vec Skip-gram,
    reduksi dimensi PCA, dan mengklasifikasikannya ke salah satu dari **2 Kategori**: **Finance** atau **Sport**.
    """)
    
    # Preset Contoh URL dari Detik.com
    st.markdown("##### 📌 Coba Contoh URL Berita Detik.com (Klik tombol untuk memilih):")
    sample_urls = {
        "Finance 1 (LRT & BPI Danantara)": "https://finance.detik.com/infrastruktur/d-8665852/pramono-blak-blakan-ke-prabowo-minta-bantuan-rute-lrt-diperpanjang",
        "Finance 2 (MRT Jakarta & Integrasi Kota)": "https://finance.detik.com/infrastruktur/d-8666137/blok-m-kota-tua-dan-misi-mrt-hidupkan-sudut-sudut-jakarta",
        "Sport 1 (Program Sport Diplomacy KONI)": "https://sport.detik.com/g-sport/d-8666151/ketum-koni-tutup-program-sport-diplomacy-indonesia-timor-leste",
        "Sport 2 (Atlet Tolak Peluru Papua)": "https://sport.detik.com/sport-lain/d-8666076/lina-hisage-atlet-muda-potensial-cabor-tolak-peluru-dari-papua"
    }
    
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    if col_u1.button("💰 Finance: Rute LRT"):
        st.session_state['selected_url'] = sample_urls["Finance 1 (LRT & BPI Danantara)"]
    if col_u2.button("💰 Finance: Proyek MRT"):
        st.session_state['selected_url'] = sample_urls["Finance 2 (MRT Jakarta & Integrasi Kota)"]
    if col_u3.button("⚽ Sport: Diplomasi KONI"):
        st.session_state['selected_url'] = sample_urls["Sport 1 (Program Sport Diplomacy KONI)"]
    if col_u4.button("⚽ Sport: Atlet Papua"):
        st.session_state['selected_url'] = sample_urls["Sport 2 (Atlet Tolak Peluru Papua)"]
        
    default_url = st.session_state.get(
        'selected_url',
        "https://finance.detik.com/infrastruktur/d-8665852/pramono-blak-blakan-ke-prabowo-minta-bantuan-rute-lrt-diperpanjang"
    )
    
    input_url = st.text_input(
        "🔗 Masukkan URL Artikel Berita (Detik.com atau portal berita lainnya):",
        value=default_url,
        placeholder="https://finance.detik.com/... atau https://sport.detik.com/..."
    )
    
    col_btn1, col_btn2 = st.columns([1.5, 3])
    with col_btn1:
        start_scrape_btn = st.button("🚀 Scrape & Klasifikasikan Berita", type="primary", use_container_width=True)
        
    if start_scrape_btn:
        if not input_url.strip():
            st.warning("Silakan masukkan URL berita terlebih dahulu.")
        else:
            with st.spinner("Mengunduh artikel dan mengekstrak teks berita dari URL..."):
                scrape_res = scrape_article_from_url(input_url.strip())
                
            if not scrape_res["success"]:
                st.error(f"❌ Gagal mengekstrak berita dari URL: {scrape_res.get('error', 'Teks tidak ditemukan')}")
            else:
                raw_article_text = scrape_res["text"]
                
                # Tampilkan info artikel yang berhasil di-scrape
                st.markdown(f"""
                <div class='article-meta-box'>
                    <div class='article-title'>📰 {scrape_res['title']}</div>
                    <div class='article-submeta'>
                        ✍️ Penulis: <b>{scrape_res['author']}</b> &nbsp;|&nbsp; 
                        📅 Tanggal: <b>{scrape_res['date']}</b> &nbsp;|&nbsp; 
                        🔗 <a href='{input_url}' target='_blank'>Buka Sumber Berita Asli</a>
                    </div>
                    <div style='color:#334155; font-size:0.92rem; line-height:1.6;'>
                        <b>Cuplikan Konten yang Berhasil Diambil:</b><br>
                        {raw_article_text[:350]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Jalankan klasifikasi
                with st.spinner("Menjalankan Preprocessing (Stemming, Stopword Removal), Word2Vec, PCA, dan Naive Bayes..."):
                    pred_res = predict_from_text(raw_article_text)
                    
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader("🎯 Hasil Klasifikasi (2 Kategori: Finance vs Sport):")
                
                c_pred1, c_pred2 = st.columns([1, 1.3])
                
                with c_pred1:
                    if pred_res['label'] == "Finance":
                        st.markdown(f"""
                        <div style='text-align:center; padding: 22px; background:#f0fdf4; border-radius:14px; border:1px solid #bbf7d0;'>
                            <div style='font-size:0.9rem; color:#166534; font-weight:700; text-transform:uppercase;'>HASIL KLASIFIKASI KATEGORI</div>
                            <div class='result-badge-finance' style='margin-top:10px;'>💰 FINANCE</div>
                            <div style='margin-top:16px; font-size:1.15rem; color:#15803d; font-weight:700;'>
                                Probabilitas: <b>{pred_res['prob_finance'] * 100:.2f}%</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='text-align:center; padding: 22px; background:#eff6ff; border-radius:14px; border:1px solid #bfdbfe;'>
                            <div style='font-size:0.9rem; color:#1e40af; font-weight:700; text-transform:uppercase;'>HASIL KLASIFIKASI KATEGORI</div>
                            <div class='result-badge-sport' style='margin-top:10px;'>⚽ SPORT</div>
                            <div style='margin-top:16px; font-size:1.15rem; color:#1d4ed8; font-weight:700;'>
                                Probabilitas: <b>{pred_res['prob_sport'] * 100:.2f}%</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric("Kata Aktif Cocok di Kosakata W2V", f"{len(pred_res['in_vocab_tokens'])} kata")
                    
                with c_pred2:
                    st.markdown("##### 📊 Distribusi Probabilitas Model (Gaussian Naive Bayes)")
                    prob_df = pd.DataFrame({
                        "Kategori": ["Finance (0)", "Sport (1)"],
                        "Probabilitas (%)": [pred_res['prob_finance'] * 100, pred_res['prob_sport'] * 100]
                    })
                    
                    fig, ax = plt.subplots(figsize=(5, 2.7))
                    colors = ["#10b981", "#3b82f6"]
                    bars = ax.barh(prob_df["Kategori"], prob_df["Probabilitas (%)"], color=colors, height=0.45)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Tingkat Keyakinan (%)")
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w + 2, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%",
                                va='center', ha='left', fontweight='bold', color='#1e293b')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
                    
                    # Nilai PCA
                    st.markdown("##### 📍 Koordinat Proyeksi Dokumen (PCA 3 Komponen):")
                    st.code(f"PC1: {pred_res['doc_pca'][0]:.4f} | PC2: {pred_res['doc_pca'][1]:.4f} | PC3: {pred_res['doc_pca'][2]:.4f}")
                    
                # Expander Detail Preprocessing
                with st.expander("🔍 Lihat Rincian 6 Tahapan Preprocessing Teks dari Artikel", expanded=False):
                    st.markdown(f"**1. Case Folding:**\n`{pred_res['steps']['case_folding'][:300]}...`")
                    st.markdown(f"**2. Cleaning Regex:**\n`{pred_res['steps']['cleaning'][:300]}...`")
                    st.markdown(f"**3. Tokenize:**\n`{pred_res['steps']['tokenize'][:20]}` (Total {len(pred_res['steps']['tokenize'])} token)")
                    st.markdown(f"**4. Normalisasi Slang & Istilah:**\n`{pred_res['steps']['normalized'][:20]}`")
                    st.markdown(f"**5. Stopword Removal Sastrawi:**\n`{pred_res['steps']['stopword_removed'][:300]}...`")
                    st.markdown(f"**6. Stemming Sastrawi:**\n`{pred_res['steps']['stemmed'][:300]}...`")
                    st.markdown(f"**Kata yang Masuk Vektor Word2Vec:**\n`{pred_res['in_vocab_tokens'][:30]}`")
                    
                with st.expander("📄 Tampilkan Teks Lengkap Artikel yang Berhasil Diambil", expanded=False):
                    st.text_area("Isi Lengkap Berita:", value=raw_article_text, height=200, disabled=True)

# ==============================================================================
# HALAMAN 2: BERANDA & RINGKASAN PROYEK
# ==============================================================================
elif menu == "🏠 Beranda & Ringkasan Proyek":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>📰 Sistem Klasifikasi Berita Detik.com via URL</div>
        <div class='hero-subtitle'>
            Implementasi Natural Language Processing (NLP) untuk mengklasifikasikan artikel berita web
            secara otomatis ke dalam 2 kategori: <b>Finance</b> atau <b>Sport</b> menggunakan
            Word2Vec Skip-gram, PCA, dan Gaussian Naive Bayes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-num'>2 Kategori</div>
            <div class='metric-label'>Target Kelas Klasifikasi</div>
            <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>Finance (0) & Sport (1)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-num'>{len(w2v_model.wv) if models_ready else 5328:,}</div>
            <div class='metric-label'>Kosakata Word2Vec</div>
            <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>Skip-gram, 100 Dimensi</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-num'>3 PC</div>
            <div class='metric-label'>Komponen Utama PCA</div>
            <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>Varians Terjaga: 91.48%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-num' style='color:#059669;'>87.50%</div>
            <div class='metric-label'>Akurasi Model Uji</div>
            <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>Gaussian Naive Bayes</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔄 Alur Pipeline Sistem dari URL hingga Hasil")
    st.markdown("""
    1. **Input URL Berita**: Pengguna memasukkan tautan (URL) artikel dari portal berita (misal: detik.com).
    2. **Web Scraping Otomatis**: Konten teks dan metadata artikel diekstraksi secara otomatis menggunakan Trafilatura dan BeautifulSoup.
    3. **6 Tahap Preprocessing Teks**:
       - *Case Folding* (Lowercase)
       - *Text Cleaning* (Regex pembersihan URL, angka, simbol)
       - *Tokenization* (Pemotongan kata)
       - *Normalisasi* (Kamus singkatan & slang)
       - *Stopword Removal* (Sastrawi)
       - *Stemming* (Sastrawi Nazief-Adriani)
    4. **Ekstraksi Fitur Word2Vec Skip-gram**: Kata-kata ditransformasikan menjadi representasi vektor numerik 100 dimensi dan dihitung rata-rata vektor dokumen (*Average Word Vector*).
    5. **Reduksi Dimensi PCA**: Vektor dokumen 100 dimensi diproyeksikan ke 3 Komponen Utama (PC) dengan menjaga 91.48% varians informasi.
    6. **Klasifikasi Gaussian Naive Bayes**: Menghitung probabilitas apakah berita tersebut bertema **Finance** atau **Sport**.
    """)

# ==============================================================================
# HALAMAN 3: DATASET & PREPROCESSING
# ==============================================================================
elif menu == "📊 Dataset & Preprocessing":
    st.title("📊 Eksplorasi Dataset & Tahap Preprocessing")
    st.markdown("Memahami struktur dataset berita Detik.com (200 data artikel: 100 Finance & 100 Sport).")
    
    tab1, tab2 = st.tabs(["📁 Tinjauan Dataset", "⚙️ 6 Tahap Preprocessing"])
    
    with tab1:
        if df_clean is not None:
            col_filter1, col_filter2 = st.columns([1, 2])
            with col_filter1:
                selected_label = st.selectbox("Filter Kategori:", ["Semua Kategori (2)", "finance", "sport"])
            with col_filter2:
                search_query = st.text_input("🔍 Cari kata dalam isi berita:", "")
                
            filtered_df = df_clean.copy()
            if selected_label != "Semua Kategori (2)":
                filtered_df = filtered_df[filtered_df['label'] == selected_label]
            if search_query:
                filtered_df = filtered_df[filtered_df['isi_berita_clean'].str.contains(search_query.lower(), na=False)]
                
            st.write(f"Menampilkan **{len(filtered_df)}** dari total **{len(df_clean)}** artikel:")
            st.dataframe(filtered_df, use_container_width=True, height=350)
            
            # Distribusi Kelas Bar Chart
            st.subheader("Distribusi 2 Kelas Dataset")
            dist = df_clean['label'].value_counts()
            col_d1, col_d2 = st.columns([1, 2])
            with col_d1:
                st.metric("Total Finance", dist.get('finance', 0))
                st.metric("Total Sport", dist.get('sport', 0))
            with col_d2:
                fig, ax = plt.subplots(figsize=(6, 2.5))
                sns.barplot(x=dist.index, y=dist.values, hue=dist.index, palette=["#3b82f6", "#10b981"], legend=False, ax=ax)
                ax.set_ylabel("Jumlah Berita")
                ax.set_xlabel("Kategori")
                for p in ax.patches:
                    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
                st.pyplot(fig)
        else:
            st.warning("File dataset tidak ditemukan.")
            
    with tab2:
        st.markdown("### 📝 Rincian 6 Tahapan Pembersihan Teks (Sesuai `preprocessing.ipynb`)")
        st.markdown("""
        <div class='step-box'>
            <div class='step-title'>1. Case Folding</div>
            <div class='step-content'>Mengubah semua karakter huruf menjadi huruf kecil (lowercase). Contoh: "IHSG Menguat" ➔ "ihsg menguat"</div>
        </div>
        <div class='step-box'>
            <div class='step-title'>2. Cleaning (Regex)</div>
            <div class='step-content'>Menghapus URL/link (http/https), angka, tanda baca, simbol non-alfanumerik, dan spasi berlebih.</div>
        </div>
        <div class='step-box'>
            <div class='step-title'>3. Tokenization</div>
            <div class='step-content'>Memecah kalimat menjadi daftar token kata tunggal menggunakan spasi sebagai pemisah (.split()).</div>
        </div>
        <div class='step-box'>
            <div class='step-title'>4. Normalisasi Slang & Istilah Serapan</div>
            <div class='step-content'>Mengganti singkatan/slang (contoh: 'yg'➔'yang', 'dgn'➔'dengan', 'match'➔'tanding', 'company'➔'perusahaan').</div>
        </div>
        <div class='step-box'>
            <div class='step-title'>5. Stopword Removal</div>
            <div class='step-content'>Menghilangkan kata umum/penghubung yang tidak memiliki makna diskriminatif menggunakan Sastrawi StopWordRemover.</div>
        </div>
        <div class='step-box'>
            <div class='step-title'>6. Stemming (Sastrawi)</div>
            <div class='step-content'>Mengembalikan setiap kata berimbuhan ke bentuk kata dasarnya (lemma) menggunakan algoritma Nazief & Adriani (Sastrawi Stemmer).</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# HALAMAN 4: MODEL WORD2VEC & PCA
# ==============================================================================
elif menu == "🧠 Model Word2Vec & PCA":
    st.title("🧠 Eksplorasi Model Word2Vec & Reduksi Dimensi PCA")
    st.markdown("Fitur representasi kata Word2Vec Skip-gram dan visualisasi ruang vektor dokumen setelah reduksi PCA.")
    
    col_w1, col_w2 = st.columns([1, 1])
    
    with col_w1:
        st.subheader("🔍 Uji Kata Serupa (Semantic Similarity)")
        st.caption("Mencari kata yang memiliki konteks paling dekat berdasarkan bobot embedding Word2Vec Skip-gram.")
        
        sample_words = ["saham", "sepak", "pelatih", "pasar", "emas", "liga", "modal", "tanding", "juara", "bank"]
        input_word = st.selectbox("Pilih contoh kata:", sample_words)
        custom_word = st.text_input("Atau ketik kata sendiri (dalam kata dasar):", "").strip().lower()
        
        target_word = custom_word if custom_word else input_word
        
        if st.button("🔎 Temukan Kata Terdekat"):
            if target_word in w2v_model.wv:
                similar_words = w2v_model.wv.most_similar(target_word, topn=8)
                sim_df = pd.DataFrame(similar_words, columns=["Kata Serupa", "Cosine Similarity"])
                sim_df["Similarity (%)"] = (sim_df["Cosine Similarity"] * 100).round(2)
                st.dataframe(sim_df[["Kata Serupa", "Similarity (%)"]], use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.barplot(x="Cosine Similarity", y="Kata Serupa", data=sim_df, palette="Blues_r", ax=ax)
                ax.set_title(f"Kata Terdekat untuk: '{target_word}'", fontweight="bold")
                st.pyplot(fig)
            else:
                st.error(f"Kata '{target_word}' tidak ditemukan dalam kosakata model Word2Vec (total 5,328 kata).")
                
    with col_w2:
        st.subheader("📊 Visualisasi Sebaran Dokumen (PCA)")
        st.caption("Visualisasi 200 dokumen berita yang telah direduksi dari 100 dimensi menjadi Komponen Utama (PC1 & PC2).")
        
        if df_pca is not None and 'PC1' in df_pca.columns and 'PC2' in df_pca.columns:
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            sns.scatterplot(
                x='PC1', y='PC2', hue='label',
                data=df_pca,
                palette={'finance': '#10b981', 'sport': '#3b82f6'},
                alpha=0.85, s=70, edgecolor='black', linewidth=0.5,
                ax=ax
            )
            ax.set_title("Distribusi Dokumen: PC1 vs PC2", fontweight="bold")
            ax.set_xlabel("Principal Component 1 (PC1)")
            ax.set_ylabel("Principal Component 2 (PC2)")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
            st.success("Tampak pemisahan kluster yang jelas antara berita kategori **Finance** (hijau) dan **Sport** (biru).")
        else:
            st.info("Data hasil PCA tidak tersedia.")

# ==============================================================================
# HALAMAN 5: EVALUASI MODEL (NAIVE BAYES)
# ==============================================================================
elif menu == "📈 Evaluasi Model (Naive Bayes)":
    st.title("📈 Evaluasi & Metrik Kinerja Model")
    st.markdown("Hasil pengujian model **Gaussian Naive Bayes** pada 2 kategori berita Detik (Finance vs Sport).")
    
    col_ev1, col_ev2 = st.columns([1, 1.2])
    
    with col_ev1:
        st.subheader("Confusion Matrix")
        cm_matrix = np.array([[17, 3], [2, 18]]) # Akurasi 87.5%
        
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Finance (0)", "Sport (1)"],
            yticklabels=["Finance (0)", "Sport (1)"],
            annot_kws={"size": 13, "weight": "bold"},
            cbar=False, ax=ax
        )
        ax.set_title("Confusion Matrix - Data Uji (40 Berita)", fontweight="bold")
        ax.set_xlabel("Prediksi Model")
        ax.set_ylabel("Label Sebenarnya")
        st.pyplot(fig)
        
    with col_ev2:
        st.subheader("Classification Report (2 Kelas)")
        report_data = {
            "Kategori": ["Finance (0)", "Sport (1)", "Accuracy", "Macro Avg", "Weighted Avg"],
            "Precision": [0.89, 0.86, "-", 0.88, 0.88],
            "Recall": [0.85, 0.90, "-", 0.88, 0.88],
            "F1-Score": [0.87, 0.88, 0.875, 0.87, 0.87],
            "Support": [20, 20, 40, 40, 40]
        }
        df_rep = pd.DataFrame(report_data)
        st.dataframe(df_rep, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div style='background:#f8fafc; border-radius:10px; padding:15px; border:1px solid #e2e8f0; margin-top:10px;'>
            <h5 style='margin-bottom:8px; color:#0f172a;'>Kesimpulan Metrik Evaluasi:</h5>
            <ul style='color:#475569; font-size:0.92rem; margin-bottom:0;'>
                <li><b>Total Kelas:</b> Tepat 2 kategori (Finance dan Sport).</li>
                <li><b>Akurasi Keseluruhan:</b> 87.50% (35 dari 40 berita uji terprediksi tepat).</li>
                <li><b>Presisi Finance:</b> 89% (minim kesalahan salah deteksi sport sebagai finance).</li>
                <li><b>Recall Sport:</b> 90% (kemampuan mendeteksi 90% seluruh berita olahraga).</li>
                <li><b>F1-Score:</b> Keseimbangan sangat stabil pada rentang 87% - 88%.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# HALAMAN 6: PROFIL PENGEMBANG
# ==============================================================================
elif menu == "👤 Profil Pengembang":
    st.title("👤 Profil Pengembang & Informasi Proyek")
    
    col_prof1, col_prof2 = st.columns([1, 2])
    
    with col_prof1:
        if os.path.exists("foto.JPG"):
            try:
                prof_img = Image.open("foto.JPG")
                st.image(prof_img, caption="Yunika Lestari", use_container_width=True)
            except Exception:
                pass
                
    with col_prof2:
        st.markdown("""
        ### Yunika Lestari
        - **NIM**: 230411100047
        - **Kelas**: IF7A (Pencarian dan Penambangan Web)
        - **Program Studi**: Teknik Informatika
        - **Fakultas**: Fakultas Teknik
        - **Universitas**: Universitas Trunojoyo Madura (UTM)
        
        ---
        #### 📚 Tentang Mata Kuliah Penambangan Web (PPW)
        Penambangan Web (PPW) merupakan mata kuliah di program studi Teknik Informatika Universitas Trunojoyo Madura (UTM) yang berfokus pada teknik ekstraksi, pemrosesan, dan analisis data berskala besar dari internet.
        
        Mahasiswa mempelajari cara mengumpulkan data secara otomatis menggunakan metode crawling dan scraping dari situs web berita, meletakkan dasar text processing (NLP), ekstraksi fitur embedding (Word2Vec), reduksi dimensi (PCA), dan pemodelan klasifikasi berita untuk menghasilkan wawasan serta sistem cerdas.
        """)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "Aplikasi Web Mining Klasifikasi Berita Detik.com via URL &bull; 2 Kategori: Finance & Sport &bull; "
    "Yunika Lestari (230411100047) &bull; Teknik Informatika UTM"
    "</div>",
    unsafe_allow_html=True
)
