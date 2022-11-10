from asyncio import runners
import bentoml 
from bentoml.io import JSON
import numpy as np

price_predictor = bentoml.sklearn.get("used_price_prediction:latest")

dv = price_predictor.custom_objects['dictVectorizer']

price_predictor_runner = price_predictor.to_runner()

svc = bentoml.Service("used_price_pridictor", runners=[price_predictor_runner])

@svc.api(input=JSON(), output=JSON())
def price_predictor(input_data):
    X_test = dv.transform(input_data)
    result = price_predictor_runner.predict.run(X_test)
    suggestion = np.expm1(result)
    return suggestion