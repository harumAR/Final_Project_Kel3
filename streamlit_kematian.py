import pickle
import numpy as np
import streamlit as st
from PIL import Image

#load save model
DTR=pickle.load(open('DTR_modelkematian.pkl','rb'))
sc=pickle.load(open('Scaler_model.pkl','rb'))

#judul web
st.title("Prediksi Kematian Pasien covid dengan Decision Tree")
image = Image.open('mask.jpg')

st.image(image, caption = 'selalu berjaga jaga gunakan masker')

NAME = st.text_input("NAME")

# Input data
SEX = st.radio("SEX", [ 'Female','Male'])
if SEX == 'Female':
    SEX = 1.0
else:
    SEX = 2.0

AGE = st.text_input("AGE")
if AGE != '':
    AGE = float(AGE)  # Convert to float

USMER = st.radio("USMER", ['First', 'Second', 'Third'])
if USMER == 'First':
    USMER = 1
elif USMER == 'Second':
    USMER = 2
elif USMER == 'Third':
    USMER = 3

MEDICAL_UNIT = MEDICAL_UNIT = st.slider("MEDICAL_UNIT", 0, 13, step=1)

PATIENT_TYPE = st.radio("PATIENT_TYPE", ['Return Home', 'Hospitalization'])
if PATIENT_TYPE == 'Return Home':
     PATIENT_TYPE = 1.0
else:
    PATIENT_TYPE = 2.0

PNEUMONIA = st.radio("PNEUMONIA", ['Yes', 'No'])
if PNEUMONIA == 'Yes':
    PNEUMONIA = 1.0
else:
    PNEUMONIA = 2.0

DIABETES = st.radio("DIABETES", ['Yes', 'No'])
if DIABETES == 'Yes':
    DIABETES = 1.0
else:
    DIABETES = 2.0

COPD = st.radio("COPD", ['Yes', 'No'])
if COPD == 'Yes':
    COPD = 1.0
else:
    COPD = 2.0

ASTHMA = st.radio("ASTHMA", ['Yes', 'No'])
if ASTHMA == 'Yes':
    ASTHMA = 1.0
else:
    ASTHMA = 2.0

INMSUPR = st.radio("INMSUPR", ['Yes', 'No'])
if INMSUPR == 'Yes':
    INMSUPR = 1.0
else:
    INMSUPR = 2.0

HIPERTENSION = st.radio("HIPERTENSION", ['Yes', 'No'])
if HIPERTENSION == 'Yes':
    HIPERTENSION = 1.0
else:
    HIPERTENSION = 2.0

OTHER_DISEASE = st.radio("OTHER_DISEASE", ['Yes', 'No'])
if OTHER_DISEASE == 'Yes':
    OTHER_DISEASE = 1.0
else:
    OTHER_DISEASE = 2.0

CARDIOVASCULAR = st.radio("CARDIOVASCULAR", ['Yes', 'No'])
if CARDIOVASCULAR == 'Yes':
    CARDIOVASCULAR = 1.0
else:
    CARDIOVASCULAR = 2.0

OBESITY = st.radio("OBESITY", ['Yes', 'No'])
if OBESITY == 'Yes':
    OBESITY = 1.0
else:
    OBESITY = 2.0

RENAL_CHRONIC = st.radio("RENAL_CHRONIC", ['Yes', 'No'])
if RENAL_CHRONIC == 'Yes':
    RENAL_CHRONIC = 1.0
else:
    RENAL_CHRONIC = 2.0

TOBACCO = st.radio("TOBACCO", ['Yes', 'No'])
if TOBACCO == 'Yes':
    TOBACCO = 1.0
else:
    TOBACCO = 2.0

COVID_CARRIER=st.radio("COVID_CARRIER",['Yes','No'])
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
st.write("Prediksi Pasien bernama",NAME, "diprediksi menggunakan Decision Tree akan" Prediksi_Kematian)

st.write("Hasil Pembelajaran Mata Kuliah Statistika dan Sains Data Kelompok 3")
st.write("Harum Aprelina R (2017031092)")
st.write("Gustina Saputri (2057031015)")
st.write("Afra Nabila Zury (2057031002)")

