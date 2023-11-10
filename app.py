import streamlit as st
import classes
import pandas as pd
import requests
estrato = tuple([f'Estrato {x}' for x in range(1,7)])
internet = tuple(['Si', 'No'])
situa= tuple(["Igual", "Mejor", "Peor"])

estrato_elegido=st.selectbox("Estrato", options=estrato, index=0)
internet_elegido = st.checkbox("¿Tiene internet?")
situacion_elegida = st.selectbox("Situación Económica", options=situa, index=2)

predecir = st.button("PREDECIR")

@st.cache_data
def hacer_prediccion(estrato_elegido, internet_elegido, situacion_elegida):
    request_data =       str({"fami_estratovivienda": estrato_elegido,
        "tiene_internet": int(internet_elegido),
    "fami_situacioneconomica": situacion_elegida}).replace("'",'"')

    url_api="http://localhost:8000/predecir"
    return requests.post(url=url_api, data=request_data).text

if predecir:

    val = hacer_prediccion(estrato_elegido, internet_elegido, situacion_elegida)
    #modelo = classes.ModeloMLParaAPI(estrato_elegido, internet_elegido, situacion_elegida)




    st.metric(value=val, label='Prob. alumno tenga puntaje alto')
    #st.dataframe(modelo._preprocesar_datos())

