#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import random
import os
from keras.preprocessing import image
from keras.models import model_from_json
from keras import regularizers, optimizers

from load import *

#server
app = Flask(__name__)

import tensorflow as tf
json_file = open('model_diplom_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model_diplom_h5.h5')
print('loaded')

loaded_model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])

json_file = open('model_diplom_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


global graph
graph = tf.compat.v1.get_default_graph()
# to session
session = tf.compat.v1.Session(graph=graph)

classes = ['тиран', 'духовный','Ответственный', 'узкомыслящий',           'хитрый', 'семейный','самоуверенный','мыслящий широко',           'чувствительный', 'материальный', 'добродушный','артистичный','мягкий']

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/predict/',methods=['POST', 'GET'])
def predict():
    
    # pcture
    n_size = 100
    #print('Enter picture path')
    #pic_path = str(input())
    # open and preprocess
    
    x = np.expand_dims(image.img_to_array(image.load_img('61547474_2431238130443189_6036326193769218048_n.jpg', target_size=(n_size, n_size))), axis=0).reshape(n_size,n_size,3)
    # predict
    x = x.reshape([-1,100,100,3])
    print(x.shape)
    
    tf.compat.v1.global_variables_initializer()
    
    with session.as_default():
        with graph.as_default():
            
            pred_d = loaded_model.predict(x)
            print(pred_d[0])
            pred_text =[]
            for i in range(len(classes)):
                if pred_d[0][i]>0.9:
                    pred_text.append(classes[i])
            print(pred_text)
            return str(pred_text)
        
        tf.reset_default_graph()
        
if __name__ == "__main__":
    #port = int(os.environ.get('PORT', 5008))
    #app.run(host = '0.0.0.0', port=port)
    
    app.run(debug=True)


# In[ ]:





# In[ ]:




