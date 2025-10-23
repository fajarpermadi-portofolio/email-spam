import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan TF-IDF vocabulary
model_fraud = pickle.load(open('model_fraud.sav', 'rb'))
vocab = pickle.load(open('new_selected_feature_tf-idf.sav', 'rb'))

# Inisialisasi TF-IDF Vectorizer dengan vocabulary hasil training
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=vocab)

# Judul aplikasi
st.title('💌 Prediksi Email Spam')

# Input teks
clean_teks = st.text_area('Masukkan isi email di bawah ini:')

if st.button('🔍 Deteksi Email'):
    if clean_teks.strip() == "":
        st.warning("Harap masukkan teks email terlebih dahulu!")
    else:
        # Transformasi teks input ke bentuk TF-IDF
        teks_tfidf = loaded_vec.transform([clean_teks])

        # Prediksi
        predict_fraud = model_fraud.predict(teks_tfidf)[0]

        # Tampilkan hasil prediksi
        if predict_fraud == 'ham':
            st.success("✅ Email Normal (Bukan Spam)")
        else:
            st.error("🚨 Email Spam Terdeteksi!")

