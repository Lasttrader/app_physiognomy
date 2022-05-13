#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import flask
from flask import Flask, request, redirect, jsonify, render_template, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

#import for model and preprocessing
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers, optimizers
import random
import os
import numpy as np
from load import *

import tensorflow as tf
global graph

#####################

#server
app = Flask(__name__)
dropzone = Dropzone(app)
app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

#Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

#Uploads settings and path
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads' # path
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app) # max file size, default - 16Mb


graph = tf.compat.v1.get_default_graph()
# to session
session_tf = tf.compat.v1.Session(graph=graph)

#define classes
classes = ['тиран', 'духовный','Ответственный', 'узкомыслящий',           'хитрый', 'семейный','самоуверенный','мыслящий широко',           'чувствительный', 'материальный', 'добродушный','артистичный','мягкий']

######################

# work with app
@app.route('/', methods=['POST', 'GET'])
def index():
    # clear upload folder
    upload_dir = [f for f in os.listdir(app.config['UPLOADED_PHOTOS_DEST']+'/')]
    for f in upload_dir:
        os.remove(os.path.join(app.config['UPLOADED_PHOTOS_DEST']+'/',f))
    
    #set session fo images result
    if 'file_urls' not in session:
        session['file_urls'] = []
    
    #List to hold images urls
    file_urls = session['file_urls']
    
    #Dropzone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            print(file.filename)
            
            #save file in folder
            filename = photos.save(file, name=file.filename)
            print(filename)
            
            #append image urls
            file_urls.append(photos.url(filename))
        
        #session
        session['file_urls'] = file_urls 
        session['filename'] = filename
        #print(file_urls)
        #return "uploading..."

    return render_template('index.html')


@app.route('/results', methods=['POST','GET'])
def results():
    
    # redirect if no images
    if 'file_urls' not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
    # set the files
    file_urls = session['file_urls']
    session.pop('file_urls', None)
    
    #set file
    filename = session['filename']
    session.pop('filename', None)
    print(filename)

    #return render_template('results.html', file_urls=file_urls)


#@app.route('/predict',methods=['POST', 'GET'])
#def predict():
    
     
    # pcture
    n_size = 100
    #print('Enter picture path')
    #pic_path = str(input())
    # open and preprocess
    
    x = np.expand_dims(image.img_to_array(image.load_img(app.config['UPLOADED_PHOTOS_DEST']+'/'+filename, target_size=(n_size, n_size))), axis=0).reshape(n_size,n_size,3)
    # predict
    x = x.reshape([-1,100,100,3])
    print(x.shape)
    
    tf.compat.v1.global_variables_initializer()
    
    with session_tf.as_default():
        with graph.as_default():
            #work with model

            json_file = open('model_json.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights('model_h5.h5')
            print('model is loaded')
            loaded_model.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

            # model json
            json_file = open('model_json.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)

            pred_d = loaded_model.predict(x)
            print(pred_d[0])
            pred_text =[]
            for i in range(len(classes)):
                if pred_d[0][i]>0.99:
                    pred_text.append(classes[i])
            print(pred_text)
            #return str(pred_text)
            return render_template('results.html', file_urls=file_urls, prediction_text = str(pred_text))
        
        tf.reset_default_graph()


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5008))
    app.run(host = '0.0.0.0', port=port)
    
    #app.run(debug=True)


# In[ ]:





# In[ ]:




