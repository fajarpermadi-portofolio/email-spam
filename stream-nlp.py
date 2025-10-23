import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan vocab (pastikan vocab adalah dict)
model_fraud = pickle.load(open('model_fraud.sav', 'rb'))
vocab = pickle.load(open('new_selected_feature_tf-idf.sav', 'rb'))

# Inisialisasi TF-IDF Vectorizer
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=vocab)

# Fit vectorizer pakai dummy data agar idf_ terisi
loaded_vec.fit(["dummy text just to initialize vectorizer"])

# Streamlit UI
st.title("💌 Prediksi Email Spam")

clean_teks = st.text_area("Masukkan isi email di bawah ini:")

if st.button("🔍 Deteksi Email"):
    if clean_teks.strip() == "":
        st.warning("Harap masukkan teks email terlebih dahulu!")
    else:
        teks_tfidf = loaded_vec.transform([clean_teks])
        predict_fraud = model_fraud.predict(teks_tfidf)[0]

        if predict_fraud == 'ham':
            st.success("✅ Email Normal (Bukan Spam)")
        else:
            st.error("🚨 Email Spam Terdeteksi!")
