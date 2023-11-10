from pydantic import BaseModel as BM
from typing import Literal
import pandas as pd
import joblib

class EntradaModelo(BM):
    fami_estratovivienda: Literal['Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 'Estrato 5', 'Estrato 6']
    tiene_internet: bool
    fami_situacioneconomica: Literal['Igual', 'Mejor', 'Peor']
    class Config:
        schema_extra={
            "example":{
                "fami_estratovivienda": "Estrato 3",
                "tiene_internet": 1,
                "fami_situacioneconomica": "Igual",
            }
        }

class ModeloMLParaAPI:
    def __init__(self,
                 fami_estratovivienda,
                 tiene_internet,
                 fami_situacioneconomica
                 ):
        self.fami_estratovivienda = fami_estratovivienda
        self.tiene_internet = tiene_internet
        self.fami_situacioneconomica = fami_situacioneconomica
    def _cargar_modelo(self):
        self.modelo = joblib.load("modelo_clasificacion.pkl")
    def _preprocesar_datos(self):
        estrato_elegido, internet_elegido, situacion_elegida =      self.fami_estratovivienda, self.tiene_internet, self.fami_situacioneconomica
        estratos_bin = [int(f'FAMI_ESTRATOVIVIENDA_{estrato_elegido}' == x) for x in ['FAMI_ESTRATOVIVIENDA_Estrato 1',
                                                                                      'FAMI_ESTRATOVIVIENDA_Estrato 2',
                                                                                      'FAMI_ESTRATOVIVIENDA_Estrato 3',
                                                                                      'FAMI_ESTRATOVIVIENDA_Estrato 4',
                                                                                      'FAMI_ESTRATOVIVIENDA_Estrato 5',
                                                                                      'FAMI_ESTRATOVIVIENDA_Estrato 6']]
        internet_bin = [0, 1] if internet_elegido else [1, 0]
        situa_bin = [int(f'FAMI_SITUACIONECONOMICA_{situacion_elegida}' == x) for x in
                     ['FAMI_SITUACIONECONOMICA_Igual', 'FAMI_SITUACIONECONOMICA_Mejor',
                      'FAMI_SITUACIONECONOMICA_Peor']]
        data_cols = ['FAMI_ESTRATOVIVIENDA_Estrato 1', 'FAMI_ESTRATOVIVIENDA_Estrato 2',
                     'FAMI_ESTRATOVIVIENDA_Estrato 3', 'FAMI_ESTRATOVIVIENDA_Estrato 4',
                     'FAMI_ESTRATOVIVIENDA_Estrato 5', 'FAMI_ESTRATOVIVIENDA_Estrato 6',
                     'FAMI_TIENEINTERNET_No', 'FAMI_TIENEINTERNET_Si',
                     'FAMI_SITUACIONECONOMICA_Igual', 'FAMI_SITUACIONECONOMICA_Mejor',
                     'FAMI_SITUACIONECONOMICA_Peor']
        data_pred = pd.DataFrame(columns=data_cols, data=[[*estratos_bin, *internet_bin, *situa_bin]])
        return data_pred
    def predecir(self):
        self._cargar_modelo()
        x = self._preprocesar_datos()
        y_pred = pd.DataFrame(self.modelo.predict_proba(x)[:,1]).rename(
            columns={'0':'Prob_Puntaje_Alto'}

        )
        return y_pred.to_dict(orient='records')