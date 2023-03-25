# import pandas as pd
# import random
from fastapi.testclient import TestClient

# import costumers id list and random_choice a client
# X_test = pd.read_csv("./X_test_sample.csv",index_col='SK_ID_CURR')
# 
# id_list = X_test.index.tolist()
# del X_test
# rand_id = random.choice(id_list)


from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code==200

def test_predict():
    response = client.post("/predict",json={'sk_id':141817})
    assert response.status_code==200
    # assert response.json() == {}

def test_prediction_client_id():
    response = client.post('/predict',json={'sk_id':000})
    assert response.status_code==200
    assert response.json() == {'ERROR' : "Client ID not found"}


