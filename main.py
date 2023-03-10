from fastapi import FastAPI
import pandas as pd
import numpy as np
# import pickle
from pydantic import BaseModel
# import os
# import joblib
import mlflow


class Input_model(BaseModel):
    sk_id : int

app = FastAPI()

# model = pickle.load(open('./model.pkl','rb'))
model = mlflow.sklearn.load_model("models:/LGBM_Hyperopt_specificity/1")

# model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
# model = joblib.load(model_path)


X_test = pd.read_csv("./X_test_sample.csv",index_col='SK_ID_CURR')
id_list = X_test.index.tolist()

# prediction = pd.read_csv("./prediction_X_test_sample_df.csv",index_col='id')
# prediction = prediction.to_dict(orient='index')
# base_values = np.load("./model_explanations_shap/base_values.npy")

shap_values = np.load("./model_explanations_shap/shap_values.npy")
shap_values_dict = dict(zip(id_list,shap_values))

@app.get("/")
def presentation():
    return{'message':"Implémentez un modèle de scoring \nby Emanuele Partenza"}

@app.post('/predict')
def predict_(input_sk_id :Input_model):
    sk_id = dict(input_sk_id)['sk_id']
    # if sk_id not in id_list:
    #     return {'ERROR' : "Client ID not found in the client's list"}
    y_pred = model.predict(X_test.loc[X_test.index == sk_id])
    y_pred_proba = model.predict_proba(X_test.loc[X_test.index == sk_id])
    return {'costumer_id':sk_id,
            'y_pred': int(y_pred[0]),
            'y_pred_proba':round(float(y_pred_proba[0].max()),2)} #prediction[sk_id]

@app.post('/features_importance')
def feature_importance(input_sk_id :Input_model):
    sk_id = dict(input_sk_id)['sk_id']
    if sk_id not in id_list:
        return f"ERROR {sk_id} not in the list of clients"
    return dict(zip(X_test.columns,shap_values_dict[sk_id]))
    
    