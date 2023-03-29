from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel

class Input_model(BaseModel):
    sk_id : int

app = FastAPI()
with open('./model.pkl', 'rb') as md:
    model = pickle.load(md)

X_test = pd.read_csv("./X_test_sample.csv",index_col='SK_ID_CURR')
id_list = X_test.index.tolist()

shap_values = np.load("./model_explanations_shap/shap_values.npy")
shap_values_dict = dict(zip(id_list,shap_values))

@app.get("/")
def presentation():
    return{'message':"Implémentez un modèle de scoring \nby Emanuele Partenza"}

@app.post('/predict')
def predict_(input_sk_id :Input_model):
    sk_id = dict(input_sk_id)['sk_id']
    if sk_id not in id_list:
        return {'ERROR' : "Client ID not found"}
    y_pred = model.predict(X_test.loc[X_test.index == sk_id])
    y_pred_proba = model.predict_proba(X_test.loc[X_test.index == sk_id])
    return {'costumer_id':sk_id,
            'classe_solvab': int(y_pred[0]),
            'proba_classe_0':round(float(y_pred_proba[0][0])*100,2),
            'proba_classe_1':round(float(y_pred_proba[0][1])*100,2)
            }

@app.post('/features_importance')
def feature_importance(input_sk_id :Input_model):
    sk_id = dict(input_sk_id)['sk_id']
    if sk_id not in id_list:
        return f"ERROR {sk_id} not in the list of clients"
    return dict(zip(X_test.columns,shap_values_dict[sk_id]))