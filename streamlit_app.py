import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pickle
from wordcloud import WordCloud


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# GLOBAL VARIABLE
st.set_page_config(page_title="Dashboard - Sentiment Analysis IKD")
factory = StemmerFactory()
stemmer = factory.create_stemmer()
all_stopwords = stopwords.words('indonesian')
kata_negasi = ['ga','gak','tidak', 'bukan', 'tak', 'tiada', 'belum', 'jangan', 'enggak', 'takkan', 'tidaklah', 'takluk']
stopwords = [word for word in all_stopwords if word not in kata_negasi]
df = pd.read_csv('pasd_ikd_dataset_final.csv')
df2 = pd.read_csv('psd_ikd_dataset_clean.csv')

# Meload model dari file
with open('model_catboost.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load tfidf_vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

def preprocess_text(text):
    # Konversi teks ke huruf kecil
    text = text.lower()
    # Menghapus teks yang memiliki awalan @ dan URL/Path menggunakan regular expression
    text = re.sub(r'@[A-Za-z0-9_]+|https?://\S+|www\.\S+|/[A-Za-z0-9]+', '', text)
    # Membuat kamus untuk translasi
    translation_table = str.maketrans('', '', '=[],.:()?"+&-!')
    # Menggunakan translate() untuk mengganti simbol dengan string kosong
    text = text.translate(translation_table)
    # Menghapus karakter non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Menghapus string yang diawali dengan [ dan diakhiri dengan ]
    text = re.sub(r'\[.*?\]', '', text)
    # Menghapus teks yang diawali dengan "+"
    text = re.sub(r'\+[A-Za-z0-9_]+', '', text)
    # Menghapus teks yang diawali dan diakhiri dengan tanda garis bawah
    text = re.sub(r'^_|_$', '', text)
    text = re.sub(r'(?<=\S)_\w+', '', text)
    text = re.sub(r'_\w+_', '', text)
    text = re.sub(r'\s*_[xX][0-9a-fA-F]{2}\s*', '', text)
    # Menghapus whitespace ekstra
    text = ' '.join(text.split())
    # Mengganti karakter yang berulang dengan satu kemunculan saja
    text = re.sub(r'(.)\1+', r'\1', text)
   # Menghapus stopwords yang sudah ditentukan
    words = [word for word in text.split() if word not in stopwords]
    # Gabungkan kata-kata kembali menjadi satu string
    processed_text = ' '.join(words)

    return text

dataset_referensi = {
    "ap": "aplikasi",
    "app": "aplikasi",
    "apk": "aplikasi",
    "ak": "aku",
    "km": "kamu",
    "sy": "saya",
    "tdk": "tidak",
    "bka": "buka",
    "jg": "juga",
    "jk": "jika",
    "msh": "masih",
    "yg": "yang",
    "klo": "kalau",
    "gblk": "goblok",
    "la": "lah",
    "gk": "gak",
    "blm": "belum",
    "belom": "belum",
    "buk": "buka",
    "tlg": "tolong",
    "maf": "maaf"
}

def perpanjang_kata_singkatan_teks(teks):
    teks_diperpanjang = teks
    for kata in teks.split():
        if kata in dataset_referensi:
            teks_diperpanjang = teks_diperpanjang.replace(kata, dataset_referensi[kata])
    return teks_diperpanjang

def predict_sentiment(word):
    word_vector = tfidf_vectorizer.transform([word])
    prediction = loaded_model.predict(word_vector)
    print(prediction)

    if prediction[0] == 1:
        return "Positif"
    else:
        return "Negatif"

def main():
    st.title("Sentiment Analysis Aplikasi Identitas Kependudukan Digital ✨")

    # Sidebar
    st.sidebar.title("IKD")
    st.sidebar.subheader("Menu")
    option = st.sidebar.selectbox(
        'Pilih opsi:',
        ('Beranda', 'EDA', 'Predict')
    )

    # Konten Utama
    if option == 'Beranda':
        st.header("Sambutan 🎉")
        st.write("Selamat datang ! Situs ini akan menjelaskan tentang pengalaman pengguna dalam menggunakan \
                 aplikasi Identitas Kependudukan Digital. Kami menyajikan analisis sentimen dari berbagai tanggapan \
                 pengguna untuk memberikan gambaran tentang penerimaan dan pengalaman pengguna secara keseluruhan. \
                 Mari kita jelajahi bersama bagaimana aplikasi ini menghadirkan proses digitalisasi KTP yang baru, \
                 serta melihat bagaimana respons pengguna terhadap aplikasi ini.")
        
        st.header("Tentang Aplikasi Identitas Kependudukan Digital 😎")
        st.write("Aplikasi Identitas Kependudukan Digital adalah sebuah solusi inovatif yang memungkinkan pengguna \
                  untuk dengan mudah mengkonversi informasi dari KTP fisik mereka ke dalam bentuk digital. Dengan \
                 menyediakan platform yang aman dan efisien, aplikasi ini memfasilitasi proses digitalisasi identitas, \
                 memberikan kemudahan akses dan manajemen identitas bagi pengguna, serta menyokong upaya pemerintah dalam  \
                 mewujudkan transformasi digital di sektor kependudukan.")
                 
    elif option == 'EDA':
        st.write("Selamat datang ! ini adalah halaman EDA yang berfungsi untuk menampilkan visualisasi hasil eksplorasi data")

        st.subheader('Nilai Sampel dari Dataset')
        st.write(df2.head())

        # Mengambil nilai perhitungan distribusi kelas
        class_counts = df['score'].value_counts()
        
        st.subheader("Distribusi Skor Aplikasi Identitas Kependudukan Digital")
        # Menampilkan diagram batang
        st.bar_chart(class_counts)
        st.divider()
        st.write("**Insight**: aplikasi Identitas Kependudukan Digital memiliki dua kelompok pengguna yang dominan: satu kelompok yang sangat tidak puas dan satu kelompok yang sangat puas. Hal ini mengindikasikan bahwa ada aspek-aspek tertentu dari aplikasi yang sangat berhasil bagi sebagian pengguna, sementara aspek lain mungkin menyebabkan frustrasi atau kekecewaan bagi yang lain.")
        st.divider()

        # Mengambil nilai perhitungan distribusi kelas
        class_counts = df['tone'].value_counts()

        st.subheader("Distribusi Tone Aplikasi Identitas Kependudukan Digital")
        # Menampilkan diagram batang
        st.bar_chart(class_counts)
        st.divider()
        st.write("**Insight**: banyak pengguna memiliki sentimen negatif terhadap aplikasi ini dibandingkan dengan yang memiliki sentimen positif. Meski demikian, perbedaan antara tone negatif dan positif tidak terlalu besar, menunjukkan adanya kelompok signifikan yang tetap puas dengan aplikasi ini meskipun ada kritik yang dominan.")
        st.divider()

        # Fungsi untuk menentukan dominasi tone
        def determine_dominant_tone(group):
            positive_count = (group['tone'] == 1).sum()
            negative_count = (group['tone'] == -1).sum()
            if positive_count > negative_count:
                return 1
            elif negative_count > positive_count:
                return -1
            else:
                return 0

        # Contoh DataFrame
        # df2 = pd.read_csv('path_to_your_data.csv')

        # Menghitung dominasi tone per bulan
        dominant_tone_per_month = df2.groupby(['year', 'month']).apply(determine_dominant_tone).reset_index(name='dominant_tone')

        # Menentukan rentang tahun dan bulan
        start_year = df2['year'].min()
        end_year = df2['year'].max()
        start_month = df2['month'].min()
        end_month = 5

        # Membuat daftar semua bulan dalam rentang tahun yang ada
        all_months = pd.date_range(start=f'{start_year}-{start_month}', end=f'{end_year}-{end_month}', freq='MS').strftime("%Y-%m").tolist()

        # Menggabungkan hasil dominasi tone dengan semua bulan
        dominant_tone_per_month['year_month'] = dominant_tone_per_month['year'].astype(str) + '-' + dominant_tone_per_month['month'].astype(str).str.zfill(2)
        dominant_tone_per_month = dominant_tone_per_month.set_index('year_month').reindex(all_months).reset_index()
        dominant_tone_per_month = dominant_tone_per_month.rename(columns={'index': 'year_month'}).fillna(0)

        # Visualisasi dengan Streamlit
        st.title('Dominasi Tone per Bulan')

        # Plot dominasi tone per bulan menggunakan Matplotlib
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(dominant_tone_per_month['year_month'], dominant_tone_per_month['dominant_tone'], marker='o')

        # Menambahkan detail visualisasi
        ax.set_title('Dominasi Tone per Bulan')
        ax.set_xlabel('Tahun-Bulan')
        ax.set_ylabel('Dominasi Tone')
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Negatif', 'Netral', 'Positif'])
        ax.set_xticks(dominant_tone_per_month['year_month'][::3])  # Show every 3rd month for better readability
        ax.set_xticklabels(dominant_tone_per_month['year_month'][::3], rotation=90)
        ax.grid(True)
        plt.tight_layout()

        # Menampilkan plot di Streamlit
        st.pyplot(fig)
        st.divider()
        st.write("**Insight**: Akhir-akhir ini aplikasi sedang mendapatkan sentimen yang buruk mengingat sekitar 6 bulan terakhir tone negatif lebih dominan daripada positif ataupun netral.")
        st.divider()

        st.subheader('WordCloud untuk Tone Positif')

        # Menggabungkan teks dari kolom "content" menjadi satu string
        text = " ".join(df[df['tone'] == 1]['content'])

        # Membuat WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

        # Menampilkan WordCloud menggunakan st.image
        st.image(wordcloud.to_array(), use_column_width=True)
        st.divider()
        st.write("**Insight**: pengguna sering menggunakan kata-kata seperti 'sangat membantu', 'terima kasih', 'lebih baik', dan 'bermanfaat'. Kata-kata ini mengindikasikan bahwa pengguna merasa aplikasi ini memberikan manfaat yang signifikan dan memudahkan proses administrasi kependudukan. Ucapan terima kasih yang sering muncul menunjukkan apresiasi tinggi terhadap kemudahan dan efisiensi yang disediakan oleh aplikasi ini. Selain itu, ada juga harapan untuk peningkatan lebih lanjut, seperti yang tercermin dalam kata-kata 'lebih baik' dan 'semoga.'")
        st.divider()

        st.subheader('WordCloud untuk Tone Negatif')

        # Menggabungkan teks dari kolom "content" menjadi satu string
        text = " ".join(df[df['tone'] == -1]['content'])

        # Membuat WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

        # Menampilkan WordCloud menggu
        # nakan st.image
        st.image(wordcloud.to_array(), use_column_width=True)
        st.divider()
        st.write("**Insight**: pengguna sering menggunakan kata-kata seperti 'tidak bisa', 'scan barcode', 'ke dukcapil', dan 'ke kantor'. Kata-kata ini mengindikasikan bahwa pengguna merasa aplikasi ini terkadang memberikan masalah yang signifikan dan menyulitkan proses administrasi kependudukan.")
        st.divider()

        st.title('Top 15 Bigrams untuk Tone Positif')

        df_positive_tone = df[df['tone'] == 1]

        # Gabungkan semua teks dalam satu string
        all_text = ' '.join(df_positive_tone['content'])

        # Inisialisasi CountVectorizer dengan ngram_range=(2, 2) untuk bigram
        vectorizer = CountVectorizer(ngram_range=(2, 2))

        # Transformasi teks menjadi vektor fitur bigram
        X = vectorizer.fit_transform([all_text])

        # Ambil fitur (bigram) dan frekuensi masing-masing
        features = vectorizer.get_feature_names_out()
        frequencies = X.toarray()[0]

        # Buat dictionary dengan pasangan bigram dan frekuensinya
        bigram_freq_dict = dict(zip(features, frequencies))

        # Ambil 15 bigram teratas
        top_15_bigrams = dict(sorted(bigram_freq_dict.items(), key=lambda item: item[1], reverse=True)[:15])

        # Buat visualisasi diagram batang menggunakan st.bar_chart
        st.bar_chart(pd.Series(top_15_bigrams))
        st.divider()
        st.write("**Insight**: Kita bisa melihat kata-kata yang sering muncul dengan mudah menggunakan visualisasi ini. Terlihat pengguna sering menggunakan kata-kata seperti 'sangat membantu', 'sangat bermanfaat', 'terima kasih', dan sebagainya yang menunjukan bahwa aplikasi ini membantu pengguna tersebut dalam hal administratif.")
        st.divider()


        st.subheader('Top 15 Bigrams untuk Tone Negatif')
        

        df_negative_tone = df[df['tone'] == -1]

        # Gabungkan semua teks dalam satu string
        all_text = ' '.join(df_negative_tone['content'])

        # Inisialisasi CountVectorizer dengan ngram_range=(2, 2) untuk bigram
        vectorizer = CountVectorizer(ngram_range=(2, 2))

        # Transformasi teks menjadi vektor fitur bigram
        X = vectorizer.fit_transform([all_text])

        # Ambil fitur (bigram) dan frekuensi masing-masing
        features = vectorizer.get_feature_names_out()
        frequencies = X.toarray()[0]

        # Buat dictionary dengan pasangan bigram dan frekuensinya
        bigram_freq_dict = dict(zip(features, frequencies))

        # Ambil 15 bigram teratas
        top_15_bigrams = dict(sorted(bigram_freq_dict.items(), key=lambda item: item[1], reverse=True)[:15])

        # Buat visualisasi diagram batang menggunakan st.bar_chart
        st.bar_chart(pd.Series(top_15_bigrams))
        st.divider()
        st.write("**Insight**: pengguna sering menggunakan kata-kata seperti 'tidak bisa', 'scan barcode', 'ke dukcapil', dan 'ke kantor'. Kata-kata ini mengindikasikan bahwa terdapat malfungsi pada aplikasi seperti permasalahan pada scan barcode, lalu terdapat masalah lain yang menyebabkan pengguna harus datang ke kantor dukcapil.")
        st.divider()
        
    elif option == 'Predict':
        st.write("Selamat datang ! Ini adalah halaman prediksi yang berfungsi untuk memprediksi sentiment berdasarkan \
                 tanggapan yang dimasukan. Tanggapan yang dimasukan berupa teks secara langsung")
        
        st.header("Prediksi Kalimat Secara Langsung")
        input_text = st.text_area("Masukkan teks:")
        text = preprocess_text(input_text)
        text = perpanjang_kata_singkatan_teks(text)

        if st.button("Predict"):
            if(len(text) != 0):
                sentimen = predict_sentiment(text)
                if(sentimen) == "Positif":
                    st.balloons()
                    st.success(f"Text yang Anda masukan memiliki sentimen: {sentimen} 🥳")
                else:
                    st.error(f"Text yang Anda masukan memiliki sentimen: {sentimen} 😢")
            else:
                st.error(f"Harap masukan teks !")
            

if __name__ == "__main__":
    main()