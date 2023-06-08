import pickle
import numpy as np
import streamlit as st
from PIL import Image

#load save model
DTR=pickle.load(open('DTR_modelkematian.pkl','rb'))
sc=pickle.load(open('Scaler_model.pkl','rb'))

#judul web
st.title("Prediksi Kematian Pasien Covid-19 dengan Decision Tree")
image = Image.open('mask.jpg')

st.image(image, caption = 'Covid-19 Gunakan Masker')

NAME = st.text_input("NAMA LENGKAP")

# Input data
SEX = st.radio("Gender", [ 'Perempuan','Laki-Laki'])
if SEX == 'Perempuan':
    SEX = 1.0
else:
    SEX = 2.0

AGE = st.text_input("Umur")
if AGE != '':
    AGE = float(AGE)  # Convert to float

USMER = st.radio("Tingkat Unit Medis", ['Pertama', 'Kedua', 'Ketiga'])
if USMER == 'Pertama':
    USMER = 1
elif USMER == 'Kedua':
    USMER = 2
elif USMER == 'Ketiga':
    USMER = 3

MEDICAL_UNIT = MEDICAL_UNIT = st.slider("Medical Unit", 1, 13, step=1)

PATIENT_TYPE = st.radio("Tipe Pasien", ['Pulang ke rumah', 'Rawat Inap'])
if PATIENT_TYPE == 'Pulang ke rumah':
     PATIENT_TYPE = 1.0
else:
    PATIENT_TYPE = 2.0

PNEUMONIA = st.radio("PNEUMONIA", ['Ya', 'Tidak'])
if PNEUMONIA == 'Ya':
    PNEUMONIA = 1.0
else:
    PNEUMONIA = 2.0

DIABETES = st.radio("DIABETES", ['Ya', 'Tidak'])
if DIABETES == 'Ya':
    DIABETES = 1.0
else:
    DIABETES = 2.0

COPD = st.radio("COPD", ['Ya', 'Tidak'])
if COPD == 'Ya':
    COPD = 1.0
else:
    COPD = 2.0

ASTHMA = st.radio("ASTHMA", ['Ya', 'Tidak'])
if ASTHMA == 'Ya':
    ASTHMA = 1.0
else:
    ASTHMA = 2.0

INMSUPR = st.radio("IMUNOSUPRESI", ['Ya', 'Tidak'])
if INMSUPR == 'Ya':
    INMSUPR = 1.0
else:
    INMSUPR = 2.0

HIPERTENSION = st.radio("HIPERTENSION", ['Ya', 'Tidak'])
if HIPERTENSION == 'Ya':
    HIPERTENSION = 1.0
else:
    HIPERTENSION = 2.0

OTHER_DISEASE = st.radio("Penyakit Lain", ['Ya', 'Tidak'])
if OTHER_DISEASE == 'Ya':
    OTHER_DISEASE = 1.0
else:
    OTHER_DISEASE = 2.0

CARDIOVASCULAR = st.radio("CARDIOVASCULAR", ['Ya', 'Tidak'])
if CARDIOVASCULAR == 'Ya':
    CARDIOVASCULAR = 1.0
else:
    CARDIOVASCULAR = 2.0

OBESITY = st.radio("OBESITAS", ['Ya', 'Tidak'])
if OBESITY == 'Ya':
    OBESITY = 1.0
else:
    OBESITY = 2.0

RENAL_CHRONIC = st.radio("Penyakit GINJAL", ['Ya', 'Tidak'])
if RENAL_CHRONIC == 'Ya':
    RENAL_CHRONIC = 1.0
else:
    RENAL_CHRONIC = 2.0

TOBACCO = st.radio("Pengguna Tembakau", ['Ya', 'Tidak'])
if TOBACCO == 'Ya':
    TOBACCO = 1.0
else:
    TOBACCO = 2.0

COVID_CARRIER=st.radio("COVID CARRIER",['Yes','No'])
if COVID_CARRIER == 'Yes':
    COVID_CARRIER = 1.0
else:
    COVID_CARRIER = 2.0

#kode untuk prediksi
Prediksi_Kematian =''
if st.button("Prediksi SEKARANG"):
    # Mengubah argumen menjadi array numpy dua dimensi
    data = [[USMER, MEDICAL_UNIT, SEX, PATIENT_TYPE, PNEUMONIA, AGE, DIABETES, COPD, ASTHMA, INMSUPR,HIPERTENSION, OTHER_DISEASE, CARDIOVASCULAR, OBESITY, RENAL_CHRONIC, TOBACCO, COVID_CARRIER]]
    # Melakukan scaling pada data input
    scaled_data = sc.transform(data)
    # Melakukan prediksi dengan KNN
    Prediksi = DTR.predict(scaled_data)

    if Prediksi[0]==0:
        Prediksi_Kematian ="DIED"
    elif Prediksi[0] == 1:
        Prediksi_Kematian = "LIFE"
    else:
        Prediksi_Kematian = "UNKNOWN"

st.success(Prediksi_Kematian)
st.write("Prediksi Pasien :",Prediksi_Kematian)

st.write("Hasil Pembelajaran Mata Kuliah Statistika dan Sains Data Kelompok 3")
st.write("Harum Aprelina R (2017031092)")
st.write("Gustina Saputri (2057031015)")
st.write("Afra Nabila Zury (2057031002)")

