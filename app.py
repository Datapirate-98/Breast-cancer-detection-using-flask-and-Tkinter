import numpy as np
import pandas as pd
import streamlit as st
import pickle

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

def wellcome():
    return "WELCOME DOCS"

def predict_result(Thickness, Size, Shape, Adhesion, Epithelial_Size, Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses):
    prediction = classifier.predict([[Thickness, Size, Shape, Adhesion, Epithelial_Size, Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses]])
    print(prediction)
    return prediction
def main():
    st.title("Breast Cancer Detection")
    html_temp = """
<div style = "background-color:red;padding:10px">
<h2 style="color:white;text-align:center;">BREAST CANCER DETECTION APP</h2>
</div>
"""
    st.markdown(html_temp, unsafe_allow_html = True)
    Thickness = st.text_input("Thickness", "Type Here")
    Size = st.text_input("Size", "Type Here")
    Shape = st.text_input("Shape", "Type Here")
    Adhesion = st.text_input("Adhesion", "Type Here")
    Epithelial_Size = st.text_input("Epithelial Size", "Type Here")
    Bare_Nuclei = st.text_input("Bare Nuclei", "Type Here")
    Bland_Chromatin = st.text_input("Bland Chromatin", "Type Here")
    Normal_Nucleoli = st.text_input("Normal Nucleoli", "Type Here")
    Mitoses = st.text_input("Mitoses", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_result(Thickness, Size, Shape, Adhesion, Epithelial_Size, Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses)
    st.success("The output is {}".format(result))
    if st.button("About"):
        st.text("BUILT BY SATYAM")
        st.text("Using Streamlit")
if __name__=="__main__":
    main()
    
