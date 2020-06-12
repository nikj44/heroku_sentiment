from flask import Flask,render_template,url_for,request
import numpy as np
import nltk
import pandas as pd
import string
import re
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import Sequential
import nltk
import re

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
        from tensorflow.keras.models import model_from_json
        json_file = open('model.json', 'r')
        model_json = json_file.read()
        model2 = model_from_json(model_json)
        model2.load_weights("model.h5")
        from nltk.corpus import stopwords
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import one_hot
        voc_size=10000


        if request.method == 'POST':
                message = request.form['message']
        df1 = pd.DataFrame(columns=['messages'])
        df1 = df1.append({'messages': message}, ignore_index=True)
        from nltk.stem.porter import PorterStemmer

        ps = PorterStemmer()
        corpus = []
        for i in range(0, len(df1)):
            mess = re.sub('[^a-zA-Z]', ' ', df1['messages'].iloc[i])
            mess = mess.lower()
            mess = mess.split()
            mess = [ps.stem(word) for word in mess if not word in stopwords.words('english')]
            mess = ' '.join(mess)
            corpus.append(mess)
        onehot_repr=[one_hot(words,voc_size)for words in corpus]
        sent_length=40
        embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

        #predictions
        X_special=np.array(embedded_docs)
        y_hats = model2.predict_classes(X_special)
        y_hats = y_hats[0].astype(int)
        return render_template('result.html',prediction = y_hats)




if __name__ == '__main__':
	app.run(debug=True)