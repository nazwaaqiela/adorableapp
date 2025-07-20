import streamlit as st
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import ngrams
import pickle
import io
import ast
import gensim
from gensim import corpora
from gensim.models import Phrases, CoherenceModel
from gensim.models.phrases import Phraser
import matplotlib.pyplot as plt
import numpy as np

USERNAME = "admincs"
PASSWORD = "adorable123"

def login():
    st.title("Halaman Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Berhasil masuk!")
            st.rerun()
        else:
            st.error("Username atau Password salah!")

def home():
    st.header("Halaman Utama")
    st.write("Selamat datang di aplikasi analisis data ulasan!")
    st.write("""
    Aplikasi ini dapat melakukan analisis data ulasan secara interaktif dan visual. Berikut ini adalah langkah-langkah untuk analisis data yang harus dilakukan secara **berurutan**:

    1. **Unggah dataset** dengan memilih opsi 'Unggah Data' di menu.
    2. Pilih '**Analisis Data Eksploratori**' untuk melihat gambaran umum dari data melalui grafik dan statistik deskriptif.
    3. Pilih '**Analisis Sentimen**' untuk menganalisis teks dalam dataset dan melihat sentimen yang terkandung di dalamnya.
    4. Pilih '**Filter Ulasan**' untuk melakukan filter pada ulasan setiap sentimen berdasarkan nama produk dan kategori produk.
    5. Pilih '**Analisis Topik**' untuk mengidentifikasi topik-topik utama yang terkandung dalam dataset.
    
    Setelah analisis selesai, hasil analisis dalam bentuk file excel dapat diunduh.

    """)


def upload_data():
    st.header("Unggah Data")

    uploaded_file = st.file_uploader("Unggah file Excel", type="xlsx")

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            st.session_state.df = df

            st.subheader("Informasi Dataset")
            st.write(f"Jumlah baris: {df.shape[0]}")
            st.write(f"Jumlah kolom: {df.shape[1]}")

            missing = df.isnull().sum()
            missing_cols = missing[missing > 0]
            if not missing_cols.empty:
                st.write("Missing values per kolom:")
                st.dataframe(missing_cols)

                if st.button("Hapus Missing Values"):
                    df = df.dropna()
                    st.success("Missing values berhasil dihapus!")
            else:
                st.write("Tidak ada missing values.")
            
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                st.write(f"Jumlah baris duplikat: {dup_count}")
                if st.checkbox("Tampilkan baris duplikat"):
                    st.dataframe(df[df.duplicated()])
                
                if st.button("Hapus Duplikat"):
                    df = df.drop_duplicates()
                    st.success("Duplikat berhasil dihapus!")
            else:
                st.write("Tidak ada baris duplikat.")
            
            st.write("Tipe Data per Kolom:")
            st.dataframe(df.dtypes.astype(str))
            
            st.write("10 Data Teratas")
            st.dataframe(df.head(10))

        except Exception as e:
            st.error(f"Gagal membaca file Excel: {e}")

    else:
        st.write("Silakan unggah file terlebih dahulu.")

    kamus_slang_path = "kamus_slang.xlsx"
    stopwords_path = "stopwords.xlsx"

    try:
        kamus_slang_df = pd.read_excel(kamus_slang_path)
        stopwords_df = pd.read_excel(stopwords_path)
    
        kamus_slang = dict(zip(kamus_slang_df["slang"], kamus_slang_df["formal"]))
        list_stopwords = set(stopwords_df["stopword"])
    except Exception as e:
        st.error(f"Error loading files: {e}")
    
    kata_hapus = {'nya', 'ya', 'sih', 'banget', 'gitu', 'deh', 'huhu', 'sayang', 'kali', 'wkwk', 'eh', 'ku', 'kak', 'adorable', 'sepatu', 'pakai', 'sih', 'dah', 'moga', 'semoga', 'x', 'projects', 'beli', 'pokok'}

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Step 1 - cleaning karakter, huruf berulang, spasi
    def clean_text(text):
        text = re.sub(r'[^a-z\s]', '', str(text), flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        return text.lower()

    # Step 2 - ganti slang
    def replace_slang(text):
        words = text.split()
        return ' '.join([kamus_slang.get(w, w) for w in words])

    # Step 3 - hapus stopwords
    def remove_stopwords(text):
        words = text.split()
        return ' '.join([w for w in words if w not in list_stopwords])

    # Step 4 - stemming
    def apply_stemming(text):
        return stemmer.stem(text)

    # Step 5 - hapus noise
    def remove_noise(text):
        words = text.split()
        return ' '.join([w for w in words if w not in kata_hapus])

    # Step 6 - tokenisasi
    def tokenize(text):
        return text.split()

    if "Ulasan" not in df.columns:
        st.error("Kolom 'Ulasan' tidak ditemukan dalam data.")
        return

    status_placeholder = st.empty()
    status_placeholder.write("Memproses pembersihan teks...")

    df["Ulasan_Cleaned"] = df["Ulasan"].apply(clean_text)
    df["Ulasan_Normalized"] = df["Ulasan_Cleaned"].apply(replace_slang)
    df["Ulasan_Removed"] = df["Ulasan_Normalized"].apply(remove_stopwords)
    df["Ulasan_Stemmed"] = df["Ulasan_Removed"].apply(apply_stemming)
    df["Ulasan_Stemmed2"] = df["Ulasan_Stemmed"].apply(remove_noise)
    df["Ulasan_Tokenized"] = df["Ulasan_Stemmed2"].apply(tokenize)

    status_placeholder.empty()
    st.success("Teks berhasil dibersihkan!")

    # Tampilkan hasil
    st.subheader("Cuplikan Hasil Pembersihan Data")
    st.dataframe(df["Ulasan_Tokenized"].head())


def exploratory_data_analysis():
    st.header("Analisis Data Eksploratori")

    if "df" not in st.session_state or "Ulasan_Tokenized" not in st.session_state.df.columns:
        st.warning("⚠ Silakan upload dan bersihkan data terlebih dahulu.")
        return

    df = st.session_state.df.copy()

    # Jumlah ulasan per produk
    st.subheader("Jumlah Ulasan per Produk")
    product_counts = df["Produk"].value_counts()  # Menghitung jumlah ulasan per produk
    fig, ax = plt.subplots(figsize=(15, 6))  # Menyesuaikan ukuran agar sama
    product_counts.head(15).plot(kind="bar", color="lightgreen", ax=ax)
    ax.set_title("Top 15 Produk dengan Ulasan Terbanyak")
    ax.set_xlabel("Produk")
    ax.set_ylabel("Jumlah Ulasan")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    df["Ulasan_String"] = df["Ulasan_Tokenized"].apply(lambda tokens: ' '.join(tokens))

    # Explode kata
    words_exploded = df["Ulasan_String"].str.split().explode()
    word_counts = words_exploded.value_counts()

    # Kata paling banyak muncul
    st.subheader("Kata yang paling banyak muncul")
    fig, ax = plt.subplots(figsize=(10, 6))
    word_counts.head(20).plot(kind="bar", color="skyblue", ax=ax)
    ax.set_title("Top 15 Kata yang Paling Banyak Muncul")
    ax.set_xlabel("Kata")
    ax.set_ylabel("Frekuensi")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # semua token jadi satu string
    all_tokens = sum(df["Ulasan_Tokenized"], [])
    text = ' '.join(all_tokens)

    # WordCloud
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    st.subheader("WordCloud")
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("WordCloud Ulasan", fontsize=20)
    st.pyplot(fig)

def analisis_sentimen():
    st.header("Analisis Sentimen Ulasan")

    if "df" not in st.session_state or "Ulasan_Tokenized" not in st.session_state.df.columns:
        st.warning("⚠ Silakan upload dan bersihkan data terlebih dahulu.")
        return

    df = st.session_state.df.copy()

    # Gabungkan token jadi string ulasan
    df["Ulasan_String"] = df["Ulasan_Tokenized"].apply(lambda tokens: ' '.join(tokens))

    # Load model pipeline
    try:
        with open("model_sentimen.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return

    # Prediksi sentimen
    try:
        df["Prediksi_Sentimen"] = model.predict(df["Ulasan_String"])
        st.success("Sentimen berhasil diprediksi!")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
        return

    label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
    export_df = df.copy()
    export_df["Sentimen"] = export_df["Prediksi_Sentimen"].map(label_map)

    st.subheader("Distribusi Sentimen")

    sentimen_counts = export_df["Sentimen"].value_counts().reindex(["Negatif", "Netral", "Positif"], fill_value=0)
    colors = {"Negatif": "red", "Netral": "gold", "Positif": "green"}

    fig, ax = plt.subplots(figsize=(5, 3.5))  
    bars = ax.bar(sentimen_counts.index, sentimen_counts.values,
                  color=[colors[s] for s in sentimen_counts.index])

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)  

    ax.set_title("Jumlah Ulasan per Sentimen", fontsize=8)
    ax.set_xlabel("Sentimen", fontsize=7)
    ax.set_ylabel("Jumlah", fontsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    st.pyplot(fig)
    
    st.subheader("Ulasan Berdasarkan Sentimen")

    tab_neg, tab_net, tab_pos = st.tabs(["**Negatif**", "**Netral**", "**Positif**"])

    with tab_neg:
        st.write("Ulasan dengan sentimen **Negatif**")
        neg_df = df[df["Prediksi_Sentimen"] == 0]

        if not neg_df.empty:
             # WordCloud 
            all_tokens = sum(neg_df["Ulasan_Tokenized"], [])
            text = ' '.join(all_tokens)
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("WordCloud Sentimen Negatif", fontsize=18)
            st.pyplot(fig)
    
            # Visualisasi produk dengan ulasan terbanyak 
            neg_product_counts = neg_df["Produk"].value_counts().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            neg_product_counts.plot(kind="barh", color="lightcoral", ax=ax)
            ax.invert_yaxis() 
            ax.set_title("Top 15 Produk dengan Ulasan Negatif", fontsize=14, pad=20)
            ax.set_xlabel("Jumlah Ulasan Negatif", fontsize=12) 
            ax.set_ylabel("Nama Produk", fontsize=12) 
            plt.tight_layout()
            st.pyplot(fig)

        st.dataframe(neg_df[["Produk", "Ulasan"]])
                        
    with tab_net:
        st.write("Ulasan dengan sentimen **Netral**")
        net_df = df[df["Prediksi_Sentimen"] == 1]

        if not net_df.empty:
             # WordCloud 
            all_tokens = sum(net_df["Ulasan_Tokenized"], [])
            text = ' '.join(all_tokens)
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("WordCloud Sentimen Netral", fontsize=18)
            st.pyplot(fig)
    
            # Visualisasi produk dengan ulasan terbanyak 
            net_product_counts = net_df["Produk"].value_counts().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            net_product_counts.plot(kind="barh", color="lightcoral", ax=ax)
            ax.invert_yaxis() 
            ax.set_title("Top 15 Produk dengan Ulasan Netral", fontsize=14, pad=20)
            ax.set_xlabel("Jumlah Ulasan Netral", fontsize=12) 
            ax.set_ylabel("Nama Produk", fontsize=12) 
            plt.tight_layout()
            st.pyplot(fig)
            
        st.dataframe(net_df[["Produk", "Ulasan"]])
        
    with tab_pos:
        st.write("Ulasan dengan sentimen **Positif**")
        pos_df = df[df["Prediksi_Sentimen"] == 2]

        if not pos_df.empty:
            # WordCloud 
            all_tokens = sum(pos_df["Ulasan_Tokenized"], [])
            text = ' '.join(all_tokens)
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("WordCloud Sentimen Positif", fontsize=18)
            st.pyplot(fig)
    
            # Visualisasi produk dengan ulasan terbanyak 
            pos_product_counts = pos_df["Produk"].value_counts().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            pos_product_counts.plot(kind="barh", color="lightcoral", ax=ax)
            ax.invert_yaxis() 
            ax.set_title("Top 15 Produk dengan Ulasan Positif", fontsize=14, pad=20)
            ax.set_xlabel("Jumlah Ulasan Positif", fontsize=12) 
            ax.set_ylabel("Nama Produk", fontsize=12) 
            plt.tight_layout()
            st.pyplot(fig)

        st.dataframe(pos_df[["Produk", "Ulasan"]])

    kolom_terpilih = ["No", "Tanggal", "Produk", "Ulasan", "Ulasan_Tokenized", "Sentimen"]
    export_df = export_df[kolom_terpilih]
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name="Hasil Sentimen")

    xlsx_data = output.getvalue()  

    st.download_button(
        label="📥 Unduh Hasil Sentimen",
        data=xlsx_data,
        file_name="hasil_sentimen.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Simpan kembali ke session_state
    st.session_state.df = df

def filter_ulasan():
    st.header("Filter Ulasan")
    
    if "df" not in st.session_state or "Prediksi_Sentimen" not in st.session_state.df.columns:
        st.warning("⚠ Silakan lakukan analisis sentimen terlebih dahulu")
        return

    df = st.session_state.df.copy()
    
    categories = {
        "Heels": r"(?i)\bheels?\b",        
        "Sneakers": r"(?i)\bsneakers?\b",   
        "Boots": r"(?i)\bboots?\b",         
        "Platform": r"(?i)\bplatform\b",    
        "Sandals": r"(?i)\bsandals?\b",     
        "Mules": r"(?i)\bmules?\b",         
        "Oxford": r"(?i)\boxford\b",        
        "Wedges": r"(?i)\bwedges?\b",      
        "Loafer": r"(?i)\bloafer\b",        
        "Flat Shoes": r"(?i)\bflat shoes?\b" 
    }
    df["Kategori"]= "Lainnya"
    for cat, pattern in categories.items():
        df.loc[df["Produk"].str.contains(pattern, regex=True), "Kategori"] = cat

    # Tab sentimen
    tab_neg, tab_net, tab_pos = st.tabs(["**Negatif**", "**Netral**", "**Positif**"])
    
    with tab_neg:
        build_sentiment_tab(df[df["Prediksi_Sentimen"] == 0], "Negatif")
    with tab_net:
        build_sentiment_tab(df[df["Prediksi_Sentimen"] == 1], "Netral")
    with tab_pos:
        build_sentiment_tab(df[df["Prediksi_Sentimen"] == 2], "Positif")

def build_sentiment_tab(df, label):
    st.subheader(f"Ulasan {label}")
    
    if df.empty:
        st.warning(f"Tidak ada ulasan {label.lower()}")
        return

    tab_produk, tab_kategori = st.tabs(["Filter Berdasarkan Produk", "Filter Berdasarkan Kategori"])
    
    with tab_produk:
        selected_products = st.multiselect(
            "Pilih Produk:",
            options=df["Produk"].unique(),
            default=df["Produk"].unique()[0:3],
            key=f"prod_{label}"
        )
        produk_df = df[df["Produk"].isin(selected_products)]
        
        st.dataframe(
            produk_df[["Produk", "Ulasan"]],
            height=300,
            hide_index=True
        )
        st.markdown(f"**Total Ulasan:** {len(produk_df)}")

    with tab_kategori:
        selected_cats = st.multiselect(
            "Pilih Kategori:",
            options=df["Kategori"].unique(),
            default=df["Kategori"].unique()[0:2],
            key=f"cat_{label}"
        )
        kategori_df = df[df["Kategori"].isin(selected_cats)]
        
        st.dataframe(
            kategori_df[["Kategori", "Produk", "Ulasan"]],
            height=300,
            hide_index=True
        )
        st.markdown(f"**Total Ulasan:** {len(kategori_df)}")

def analisis_topik():
    st.header("Analisis Topik Ulasan")
    
    if "df" not in st.session_state or "Ulasan_Tokenized" not in st.session_state.df.columns:
        st.warning("⚠ Silakan upload dan bersihkan data terlebih dahulu.")
        return
    st.subheader("Topik Berdasarkan Sentimen")

    tab_neg, tab_net, tab_pos = st.tabs(["**Negatif**", "**Netral**", "**Positif**"])
    
    with tab_neg:
        datanegatif = [
            ['Topik #1', 'kelupas', 'jahit', 'kulit_ngelupas', 'pas', 'robek', 'jebol', 'tumit_sakit', 'kena_air', 'rapi', 'kualitas'],
            ['Topik #2', 'lem_rapi', 'kecil', 'kualitas', 'lem', 'ukur', 'beda', 'kaki', 'sakit_dipakenya', 'pinggir', 'sesuai'],
            ['Topik #3', 'berat', 'nyaman', 'kaki', 'lecet', 'jalan', 'lepas', 'kaos_kaki', 'nyaman_jalan', 'pas', 'besar'],
            ['Topik #4', 'kecil_ukur', 'ukur_kecil', 'sesuai', 'kaki_lecet', 'berat_keras', 'jahit_lepas', 'ukur', 'rusak', 'pilih', 'ribet'],
            ['Topik #5', 'sakit', 'kulit', 'lecet', 'sol_berat', 'jalan', 'upper', 'pas', 'kaki', 'bagus', 'coba'],
            ['Topik #6', 'lem_buka', 'kecewa', 'sol', 'lem', 'ukur', 'rapi', 'sedih', 'lubang', 'produk', 'ngelopek'],
            ['Topik #7', 'nyaman', 'sol_buka', 'kaki_lecet', 'aksesoris', 'jalan', 'kaki_sakit', 'kotor', 'kulit_kelupas', 'paku', 'ok'],
            ['Topik #8', 'nyaman', 'keras', 'rusak', 'lecet', 'kaki', 'kecewa', 'kardus', 'pas', 'produk', 'empuk'],
            ['Topik #9', 'nyaman', 'sakit', 'tali', 'pas', 'belah', 'lepas', 'kecil', 'keras', 'sol', 'ukur_ngepas'],
            ['Topik #10', 'ukur', 'sempit', 'kaki', 'lebar', 'cukup', 'tali', 'pas', 'lepas', 'warna', 'kurang'],
            ['Topik #11', 'sakit_kaki', 'pas', 'bagus', 'tali_putus', 'minggu', 'produk', 'jalan', 'upper_kelupas', 'kualitas', 'lem'],
            ['Topik #12', 'lecet', 'pas', 'tali', 'kaki', 'sol_lepas', 'jalan_sakit', 'warna', 'jalan', 'tumit', 'kulit'],
            ['Topik #13', 'sandal', 'nyaman', 'buka', 'kaku', 'langsung', 'berat', 'jalan', 'pas', 'ukur', 'belah']
        ]
    
        df_negatif = pd.DataFrame(datanegatif, columns=["Topik", "Kata 1", "Kata 2", "Kata 3", "Kata 4", "Kata 5", "Kata 6", "Kata 7", "Kata 8", "Kata 9", "Kata 10"])

        st.dataframe(df_negatif)
    
    with tab_net:
        datanetral = [
            ['Topik #1', 'bagus', 'nyaman', 'empuk', 'lecet', 'sandal', 'kaos_kaki', 'pas_kaki', 'penyok', 'tipis', 'juga'],
            ['Topik #2', 'sempit', 'pas', 'bagus', 'suka', 'naik_ukur', 'nyaman', 'ukur_kecil', 'empuk', 'besar', 'dus_penyok'],
            ['Topik #3', 'kecil', 'bagus', 'sendal', 'tali', 'saran', 'packingnya', 'tingkat', 'sandal', 'putus', 'sandal_bagus'],
            ['Topik #4', 'bagus', 'nyaman', 'berat', 'gede', 'boots', 'kaki', 'cocok_kaki', 'ukur', 'sempit', 'lucu'],
            ['Topik #5', 'bagus', 'tali', 'kecil', 'ukur', 'pas', 'ukur_pas', 'ok', 'berat', 'empuk', 'awet'],
            ['Topik #6', 'ukur', 'kaki', 'nyaman', 'besar', 'produk', 'kaki_lebar', 'bagus', 'pas', 'ngepas', 'ukur_pas'],
            ['Topik #7', 'lebar', 'bagus', 'pesan', 'longgar', 'model', 'ukur', 'besar', 'kaki', 'nyaman', 'sendal'],
            ['Topik #8', 'besar', 'bagus', 'ukur', 'pas', 'jalan', 'nyaman', 'berat', 'kecil', 'longgar', 'ukur_pas'],
            ['Topik #9', 'bagus', 'sakit', 'kecil', 'nyaman', 'pas', 'lumayan', 'langganan', 'licin', 'tali', 'besar']
        ]
        
        df_netral = pd.DataFrame(datanetral, columns=["Topik", "Kata 1", "Kata 2", "Kata 3", "Kata 4", "Kata 5", "Kata 6", "Kata 7", "Kata 8", "Kata 9", "Kata 10"])

        st.dataframe(df_netral)
    
    with tab_pos:
        datapositif = [
            ['Topik #1', 'bagus', 'enak', 'ukur', 'suka', 'nyaman', 'empuk', 'pas', 'lucu', 'gemas', 'kualitas_bagis'],
            ['Topik #2', 'nyaman', 'bagus', 'empuk', 'kaki', 'cantik', 'lucu', 'ukur_pas', 'empuk_nyaman', 'pas', 'cakep'],
            ['Topik #3', 'suka', 'ukur_pas', 'bahan', 'sol_empuk', 'empuk', 'bagus', 'nyaman_jalan', 'model', 'makasih', 'suka_model']
        ]
        
        df_positif = pd.DataFrame(datapositif, columns=["Topik", "Kata 1", "Kata 2", "Kata 3", "Kata 4", "Kata 5", "Kata 6", "Kata 7", "Kata 8", "Kata 9", "Kata 10"])
        
        st.dataframe(df_positif)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_negatif.to_excel(writer, index=False, sheet_name="Topik Negatif")
        df_netral.to_excel(writer, index=False, sheet_name="Topik Netral")
        df_positif.to_excel(writer, index=False, sheet_name="Topik Positif")
    
    output.seek(0) 
    
    st.download_button(
        label="📥 Unduh Hasil Analisis Topik",
        data=output,
        file_name="hasil_topik.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
    else:
        st.sidebar.title("Navigasi")
        page = st.sidebar.radio("Pilih Halaman", [
            "Halaman Utama", "Unggah Data", "Analisis Data Eksploratori",
            "Analisis Sentimen", "Filter Ulasan", "Analisis Topik", "Keluar"
        ])

        if page == "Halaman Utama":
            home()
        elif page == "Unggah Data":
            upload_data()
        elif page == "Analisis Data Eksploratori":
            exploratory_data_analysis()
        elif page == "Analisis Sentimen":
            analisis_sentimen()
        elif page == "Filter Ulasan":
            filter_ulasan()
        elif page == "Analisis Topik":
            analisis_topik()
        elif page == "Keluar":
            st.session_state.logged_in = False
            st.success("Logout berhasil.")
            st.rerun()

if __name__ == "__main__":
    main()
