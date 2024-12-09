# app.py
import streamlit as st
import joblib
import numpy as np

# Load model dan label encoder
try:
    model = joblib.load('model/stress_model_tuned.pkl')
    encoder = joblib.load('model/label_encoder.pkl')
except FileNotFoundError:
    st.error("Model atau label encoder tidak ditemukan. Pastikan file berada di folder 'model'.")

# Judul aplikasi
st.title("Prediksi Tingkat Stres Mahasiswa")

# Penjelasan aplikasi
st.markdown("""
    Aplikasi ini memprediksi tingkat stres mahasiswa berdasarkan beberapa faktor,
    seperti jam belajar, jam tidur, kegiatan ekstrakurikuler, dan aktivitas fisik.
    Silakan masukkan data pribadi Anda di bawah ini untuk melihat prediksi tingkat stres Anda.
""")

# Input data dari pengguna
study_hours = st.slider("Jam Belajar per Hari", min_value=0, max_value=24, step=1)
extracurricular_hours = st.slider("Jam Kegiatan Ekstrakurikuler per Hari", min_value=0, max_value=24, step=1)
sleep_hours = st.slider("Jam Tidur per Hari", min_value=0, max_value=24, step=1)
physical_activity_hours = st.slider("Jam Aktivitas Fisik per Hari", min_value=0, max_value=24, step=1)

# Preprocessing input data
input_data = np.array([[study_hours, extracurricular_hours, sleep_hours, physical_activity_hours]])

# Normalisasi input data
scaler = joblib.load('model/scaler.pkl')  # Pastikan Anda menyimpan scaler saat pelatihan model
input_data_scaled = scaler.transform(input_data)

# Prediksi ketika tombol ditekan
if st.button("Prediksi"):
    if model and encoder:
        # Prediksi tingkat stres
        prediction = model.predict(input_data_scaled)
        stress_level = encoder.inverse_transform(prediction)[0]

        # Menampilkan hasil prediksi
        st.success(f"Tingkat Stres Anda: {stress_level}")
        
        # Menyediakan tips untuk mengurangi stres
        st.markdown("""
        *Tips Mengurangi Stres:*
        - *Tidur yang cukup*: Minimal 7-8 jam per malam.
        - *Atur waktu belajar*: Jangan belajar terlalu lama tanpa istirahat.
        - *Kegiatan sosial*: Berinteraksi dengan teman dapat membantu mengurangi stres.
        - *Aktivitas fisik*: Berolahraga minimal 30 menit sehari.
        """)

    else:
        st.error("Model atau label encoder belum dimuat. Pastikan file yang diperlukan tersedia.")
