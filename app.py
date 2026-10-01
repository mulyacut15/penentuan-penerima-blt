import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import io

# -----------------------------------------------
# 1. Konfigurasi Halaman & Tema Custom (CSS)
# -----------------------------------------------
st.set_page_config(
    page_title="Sistem Klasifikasi BLT C4.5",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeksi CSS Modern
st.markdown("""
    <style>
    /* Styling Kartu Utama & Metrik */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    
    /* Custom Styling Tombol Submit */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    
    /* Kotak Pesan Hasil Klasifikasi */
    .result-box-layak {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .result-box-tidak {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# 2. Fungsi Pemrosesan Data & Pelabelan
# -----------------------------------------------
def proses_data(df):
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()
    
    # Cleaning
    df_clean['PENGHASILAN'] = df_clean['PENGHASILAN'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df_clean['PENGHASILAN'] = pd.to_numeric(df_clean['PENGHASILAN'], errors='coerce').fillna(0)
    df_clean['USIA'] = pd.to_numeric(df_clean['USIA'], errors='coerce').fillna(0)
    
    # Aturan dasar jika belum ada target
    def cek_kelayakan(row):
        skor = 0
        if str(row.get('WARGA NEGARA', '')).upper() == 'WNI':
            skor += 1
        if row.get('USIA', 0) >= 17:
            skor += 1
        if str(row.get('JENIS PEKERJAAN', '')).upper() not in ['ASN', 'TNI', 'POLRI', 'PEGAWAI NEGERI SIPIL']:
            skor += 1
        if str(row.get('KONDISI RUMAH LAYAK/TIDAK', '')).upper() in ['TIDAK', 'TIDAK LAYAK']:
            skor += 1
        if str(row.get('RENTAN/DISABILITAS', '')).upper() == 'YA':
            skor += 1
        if row.get('PENGHASILAN', 0) < 1500000:
            skor += 1
        return 'Layak' if skor >= 4 else 'Tidak Layak'
    
    if 'Kelayakan BLT' not in df_clean.columns:
        df_clean['Kelayakan BLT'] = df_clean.apply(cek_kelayakan, axis=1)
        
    return df_clean

@st.cache_data
def muat_data_awal():
    try:
        df = pd.read_excel('FINAL DATA.xlsx')
        return proses_data(df)
    except Exception:
        return None

data_utama = muat_data_awal()

# Persiapan Model
fitur = ['WARGA NEGARA', 'USIA', 'JENIS PEKERJAAN', 'KONDISI RUMAH LAYAK/TIDAK', 'RENTAN/DISABILITAS', 'PENGHASILAN']

def latih_model(df):
    X = pd.get_dummies(df[fitur])
    y = df['Kelayakan BLT']
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=1)
    model.fit(X, y)
    return model, X.columns

model, fitur_kolom = None, None
if data_utama is not None:
    model, fitur_kolom = latih_model(data_utama)

# -----------------------------------------------
# 3. Sidebar Info
# -----------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/tree-structure.png", width=70)
    st.title("Sistem Kelayakan BLT")
    st.caption("Algoritma Decision Tree C4.5")
    st.divider()
    
    if data_utama is not None:
        st.success("🟢 **Data Latih:** Aktif (`FINAL DATA.xlsx`)")
        st.write(f"**Total Sampel:** {len(data_utama)} Data")
    else:
        st.error("🔴 **Data Latih:** Belum Dimuat")
        st.warning("Unggah file di tab Upload terlebih dahulu.")
    
    st.divider()
    st.info("💡 **Tips:** Kamu bisa langsung mengecek kelayakan individu pada tab Form Input tanpa perlu upload ulang.")

# -----------------------------------------------
# 4. Header Utama
# -----------------------------------------------
st.title("🤝 Klasifikasi Kelayakan Penerima BLT")
st.write("Sistem pendukung keputusan penerima Bantuan Langsung Tunai berbasis algoritma **C4.5 (Entropy & Gain Ratio)**.")
st.divider()

# -----------------------------------------------
# 5. Navigasi Tab
# -----------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📝 **Input Form Warga**", 
    "📁 **Upload File & Batch Process**", 
    "🌳 **Pohon Keputusan & Evaluasi**"
])

# TAB 1: FORM INPUT INDIVIDU
with tab1:
    st.subheader("Cek Kelayakan Individu")
    st.write("Isi formulir di bawah ini untuk menganalisis kelayakan secara *real-time*.")
    
    with st.form("form_warga", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Informasi Identitas & Demografi**")
            nama = st.text_input("Nama Lengkap Warga", value="Ahmad")
            nik = st.text_input("NIK", value="1116032501820000")
            usia = st.number_input("Usia (Tahun)", min_value=0, max_value=120, value=35)
            warga_negara = st.selectbox("Kewarganegaraan", ["WNI", "WNA"])
            pekerjaan = st.selectbox("Jenis Pekerjaan", [
                "BURUH", "PETANI", "KARYAWAN SWASTA", "PEDAGANG", 
                "BELUM/TIDAK BEKERJA", "KARYAWAN HONORER", "ASN", "TNI", "POLRI"
            ])

        with col2:
            st.markdown("**Kondisi Ekonomi & Fasilitas**")
            penghasilan = st.number_input("Penghasilan Bulanan (Rp)", min_value=0, value=1200000, step=100000)
            kondisi_rumah = st.selectbox("Kondisi Rumah", ["TIDAK LAYAK", "LAYAK"])
            rentan = st.selectbox("Kategori Rentan / Disabilitas", ["Tidak", "Ya"])
        
        st.divider()
        tombol_submit = st.form_submit_button("🔍 Cek Kelayakan Sekarang", type="primary")

    if tombol_submit:
        if model is not None:
            input_dict = {
                'WARGA NEGARA': warga_negara,
                'USIA': usia,
                'JENIS PEKERJAAN': pekerjaan,
                'KONDISI RUMAH LAYAK/TIDAK': kondisi_rumah,
                'RENTAN/DISABILITAS': rentan,
                'PENGHASILAN': penghasilan
            }
            df_input = pd.DataFrame([input_dict])
            df_input_encoded = pd.get_dummies(df_input).reindex(columns=fitur_kolom, fill_value=0)
            
            hasil_prediksi = model.predict(df_input_encoded)[0]
            
            st.markdown("### 📌 Hasil Analisis Sistem")
            
            # Kartu Rincian Input
            res_c1, res_c2, res_c3 = st.columns([1, 1, 2])
            res_c1.metric("Nama Warga", nama)
            res_c2.metric("Penghasilan", f"Rp {penghasilan:,.0f}")
            
            with res_c3:
                if hasil_prediksi == 'Layak':
                    st.markdown('<div class="result-box-layak">✅ MEREKOMENDASIKAN: LAYAK MENERIMA BLT</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-box-tidak">❌ MEREKOMENDASIKAN: TIDAK LAYAK MENERIMA BLT</div>', unsafe_allow_html=True)
        else:
            st.error("Model belum siap. Harap pastikan file 'FINAL DATA.xlsx' berada di folder yang sama atau unggah file di tab Upload.")

# TAB 2: UPLOAD FILE & BATCH PROCESS
with tab2:
    st.subheader("Pemrosesan Data Massal (Batch Processing)")
    st.write("Unggah dokumen spreadsheet jika ingin mengklasifikasikan banyak data sekaligus.")
    
    uploaded_file = st.file_uploader("Unggah berkas Excel (.xlsx)", type=["xlsx"])
    data_diproses = uploaded_file if uploaded_file is not None else (data_utama if data_utama is not None else None)

    if uploaded_file:
        try:
            df_upload = pd.read_excel(uploaded_file)
            data_diproses = proses_data(df_upload)
            model, fitur_kolom = latih_model(data_diproses)
            st.success("File berhasil diunggah dan model C4.5 telah diperbarui!")
        except Exception as e:
            st.error(f"Gagal memproses file upload: {e}")

    if data_diproses is not None:
        st.divider()
        st.markdown("### 📊 Ringkasan Statistik Data")
        
        m1, m2, m3 = st.columns(3)
        total_warga = len(data_diproses)
        total_layak = (data_diproses['Kelayakan BLT'] == 'Layak').sum()
        total_tidak = (data_diproses['Kelayakan BLT'] == 'Tidak Layak').sum()
        
        m1.metric("Total Data Warga", f"{total_warga} Orang")
        m2.metric("Jumlah Layak BLT", f"{total_layak} Orang", delta=f"{(total_layak/total_warga)*100:.1f}%")
        m3.metric("Jumlah Tidak Layak", f"{total_tidak} Orang", delta=f"-{(total_tidak/total_warga)*100:.1f}%", delta_color="inverse")

        st.markdown("### 📋 Tabel Data Warga")
        st.dataframe(data_diproses, use_container_width=True, height=350)

        # Download Hasil
        output = io.BytesIO()
        data_diproses.to_excel(output, index=False)
        st.download_button(
            label="📥 Unduh Hasil Klasifikasi Lengkap (Excel)",
            data=output.getvalue(),
            file_name="Hasil_Klasifikasi_BLT.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

# TAB 3: VISUALISASI POHON KEPUTUSAN & EVALUASI
with tab3:
    if model is not None and data_utama is not None:
        st.subheader("🌳 Pohon Keputusan C4.5")
        st.write("Struktur logika keputusans yang dihasilkan oleh algoritma berdasarkan *Entropy*:")
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model, 
            feature_names=fitur_kolom, 
            class_names=model.classes_, 
            filled=True, 
            rounded=True,
            fontsize=8
        )
        st.pyplot(fig)

        st.divider()
        st.subheader("📈 Evaluasi Performa Model")
        
        X = pd.get_dummies(data_utama[fitur])
        y = data_utama['Kelayakan BLT']
        prediksi_total = model.predict(X)
        akurasi = accuracy_score(y, prediksi_total)
        
        st.metric("Tingkat Akurasi Model", f"{akurasi:.2%}")
        
        with st.expander("🔍 Laporan Detail (Classification Report)"):
            st.code(classification_report(y, prediksi_total))
    else:
        st.info("Unggah data terlebih dahulu untuk menampilkan struktur pohon keputusan.")