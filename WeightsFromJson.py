import json
from keras.models import model_from_json

with open('./model.structure') as fp:
    model = model_from_json(fp.read())
    print(model)
    model.load_weights('./model.weights.json')
    print(model.summary())
    print(model.weights[0])
#    print(model.structure())
