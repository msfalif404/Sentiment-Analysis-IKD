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
factory = StemmerFactory()
stemmer = factory.create_stemmer()
all_stopwords = stopwords.words('indonesian')
kata_negasi = ['tidak', 'bukan', 'tak', 'tiada', 'belum', 'jangan', 'enggak', 'takkan', 'tidaklah', 'takluk']
stopwords = [word for word in all_stopwords if word not in kata_negasi]
df = pd.read_csv('pasd_ikd_dataset_final.csv')

# Meload model dari file
with open('model_catboost.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load tfidf_vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

def preprocess_text(text):
    # Hapus karakter non-alphanumeric dan ubah ke huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    # Lematisasi dan hapus stop words
    processed_words = [stemmer.stem(word) for word in text.split() if word not in all_stopwords]
    # Gabungkan kata-kata kembali menjadi satu string
    processed_text = ' '.join(processed_words)

    return processed_text

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
    st.sidebar.title("Menu")
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

        # Mengambil nilai perhitungan distribusi kelas
        class_counts = df['score'].value_counts()
        
        # Menampilkan diagram batang
        st.bar_chart(class_counts)

        # Mengambil nilai perhitungan distribusi kelas
        class_counts = df['tone'].value_counts()

        # Menampilkan diagram batang
        st.bar_chart(class_counts)

        st.title('WordCloud')

        # Menggabungkan teks dari kolom "content" menjadi satu string
        text = " ".join(df[df['tone'] == 1]['content'])

        # Membuat WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

        # Menampilkan WordCloud menggunakan st.image
        st.image(wordcloud.to_array(), use_column_width=True)

        st.title('WordCloud')

        # Menggabungkan teks dari kolom "content" menjadi satu string
        text = " ".join(df[df['tone'] == -1]['content'])

        # Membuat WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

        # Menampilkan WordCloud menggunakan st.image
        st.image(wordcloud.to_array(), use_column_width=True)

        st.title('Top 15 Bigrams for Positive Reviews')

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

        st.title('Top 15 Bigrams for Negative Reviews')

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
        
    elif option == 'Predict':
        st.write("Selamat datang ! Ini adalah halaman prediksi yang berfungsi untuk memprediksi sentiment berdasarkan \
                 tanggapan yang dimasukan. Tanggapan yang dimasukan bisa berupa teks secara langsung ataupun file excel.")
        
        st.header("Prediksi Kalimat Secara Langsung")
        input_text = st.text_area("Masukkan teks:")
        text = preprocess_text(input_text)

        if st.button("Predict"):
            sentimen = predict_sentiment(text)
            st.write(sentimen)
            

if __name__ == "__main__":
    main()