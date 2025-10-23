# ======================================================
# app_email_bert.py — Streamlit Email Spam Detector
# ======================================================
import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================
# 1️⃣ LOAD MODEL DAN TOKENIZER
# ======================================================
@st.cache_resource
def load_model():
    model_dir = "./bert_email_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ======================================================
# 2️⃣ PREDIKSI TEKS EMAIL
# ======================================================
def predict_text(text: str):
    # Tokenisasi input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Prediksi tanpa gradient (hemat GPU)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label]) * 100
    return pred_label, confidence, probs

# ======================================================
# 3️⃣ KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Email Spam Detector (IndoBERT)",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Deteksi Email Spam dengan IndoBERT")
st.write("Gunakan model IndoBERT hasil fine-tuning untuk mendeteksi apakah email termasuk **Spam** atau **Bukan Spam**.")

# ======================================================
# 4️⃣ INPUT TEKS PENGGUNA
# ======================================================
email_text = st.text_area(
    "Masukkan isi email di bawah ini:",
    height=200,
    placeholder="Contoh: 'Selamat! Anda mendapatkan hadiah undian Rp10 juta. Klik tautan berikut...'"
)

# ======================================================
# 5️⃣ TOMBOL DETEKSI
# ======================================================
if st.button("🔍 Deteksi Sekarang"):
    if not email_text.strip():
        st.warning("Silakan masukkan teks email terlebih dahulu.")
    else:
        with st.spinner("Menganalisis isi email..."):
            label, confidence, probs = predict_text(email_text)

            # Mapping label numerik -> teks
            label_map = {
                0: "Bukan Spam (Normal)",
                1: "Spam"
            }
            pred_name = label_map.get(label, f"Label {label}")

            # Tampilkan hasil
            st.success(f"**Hasil:** {pred_name} ({confidence:.2f}% yakin)")
            st.progress(int(confidence))

            # Detail probabilitas
            st.subheader("📊 Probabilitas Tiap Kelas:")
            for i, (idx, name) in enumerate(label_map.items()):
                st.write(f"- **{name}**: {probs[i]*100:.2f}%")