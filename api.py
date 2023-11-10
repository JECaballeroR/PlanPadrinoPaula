from fastapi import FastAPI
from classes import ModeloMLParaAPI, EntradaModelo


app = FastAPI(title="Ejemplo Paula",
              version="0.42.69"
              )
"""Esto me dejaría documentar la API :)"""


@app.post("/predecir", tags=["ModeloML"])
async def predecir_probabilidad(entrada: EntradaModelo):
    """Endpoint para predecir con el modelo de clasificación desarrollado"""
    modelo = ModeloMLParaAPI(entrada.fami_estratovivienda, entrada.tiene_internet, entrada.fami_situacioneconomica)
    prediccion = modelo.predecir()[0][0]
    return prediccion