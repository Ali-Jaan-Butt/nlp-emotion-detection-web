from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import urllib.request
import zipfile
import os
import time
from keras.models import load_model
from transformers import AutoTokenizer, AutoModel
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from django.contrib import messages

def myapp(request):
    if os.path.exists('model/nlp_model.h5')==False:
        num_classes = 5
        embed_num_dims = 300
        max_seq_len = 500
        class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
        data_train = pd.read_csv('data/data_train.csv', encoding='utf-8')
        data_test = pd.read_csv('data/data_test.csv', encoding='utf-8')
        X_train = data_train.Text
        X_test = data_test.Text
        y_train = data_train.Emotion
        y_test = data_test.Emotion
        data = pd.concat([data_train, data_test], axis=0, ignore_index=True)
        def clean_text(data):
            data = re.sub(r"(#[\d\w\.]+)", '', data)
            data = re.sub(r"(@[\d\w\.]+)", '', data)
            data = word_tokenize(data)
            return data
        texts = [' '.join(clean_text(text)) for text in data.Text]
        texts_train = [' '.join(clean_text(text)) for text in X_train]
        texts_test = [' '.join(clean_text(text)) for text in X_test]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        sequence_train = tokenizer.texts_to_sequences(texts_train)
        sequence_test = tokenizer.texts_to_sequences(texts_test)
        index_of_words = tokenizer.word_index
        vocab_size = len(index_of_words) + 1
        print('Number of unique words: {}'.format(len(index_of_words)))
        X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len )
        X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len )
        X_train_pad
        encoding = {
            'joy': 0,
            'fear': 1,
            'anger': 2,
            'sadness': 3,
            'neutral': 4
        }
        y_train = [encoding[x] for x in data_train.Emotion]
        y_test = [encoding[x] for x in data_test.Emotion]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_train
        def create_embedding_matrix(filepath, word_index, embedding_dim):
            vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
            embedding_matrix = np.zeros((vocab_size, embedding_dim))
            with open(filepath) as f:
                for line in f:
                    word, *vector = line.split()
                    if word in word_index:
                        idx = word_index[word] 
                        embedding_matrix[idx] = np.array(
                            vector, dtype=np.float32)[:embedding_dim]
            return embedding_matrix
        fname = 'embeddings/wiki-news-300d-1M.vec'
        if not os.path.isfile(fname):
            print('Downloading word vectors...')
            urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                                      'wiki-news-300d-1M.vec.zip')
            print('Unzipping...')
            with zipfile.ZipFile('wiki-news-300d-1M.vec.zip', 'r') as zip_ref:
                zip_ref.extractall('embeddings')
            print('done.')
            os.remove('wiki-news-300d-1M.vec.zip')
        embedd_matrix = create_embedding_matrix(fname, index_of_words, embed_num_dims)
        embedd_matrix.shape
        new_words = 0
        for word in index_of_words:
            entry = embedd_matrix[index_of_words[word]]
            if all(v == 0 for v in entry):
                new_words = new_words + 1
        embedd_layer = Embedding(input_dim=vocab_size,
                         output_dim=embed_num_dims,
                         embeddings_initializer=tf.keras.initializers.Constant(embedd_matrix),
                         trainable=False)
        kernel_size = 3
        filters = 256
        model = Sequential()
        model.add(embedd_layer)
        model.add(Conv1D(filters, kernel_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        model.summary()
        batch_size = 256
        epochs = 6
        hist = model.fit(X_train_pad, y_train, 
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(X_test_pad,y_test))
        predictions = model.predict(X_test_pad)
        predictions = np.argmax(predictions, axis=1)
        predictions = [class_names[pred] for pred in predictions]
        accuracy = accuracy_score(data_test.Emotion, predictions) * 100
        f1_score_test = f1_score(data_test.Emotion, predictions, average='micro') * 100
        result_data = {'Accuracy': accuracy, 'F1 Score': f1_score_test}
        with open('model/data.json', 'w') as json_file:
            json.dump(result_data, json_file)
        print('Message: {}\nPredicted: {}'.format(X_test[4], predictions[4]))
        model.save('model/nlp_model.h5')
        tokenizer_json = tokenizer.to_json()
        with open('model/tokenizer.json', 'w') as json_file:
            json.dump(tokenizer_json, json_file)
    else:
        pass
    prediction(request)
    return render(request, 'comp/home.html')

def prediction(request):
    if request.method == 'POST':
        max_seq_len = 500
        prediction_res = 'None'
        accuracy = 0
        f1_result = 0
        data_test = pd.read_csv('data/data_test.csv', encoding='utf-8')
        class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
        if os.path.exists('model/nlp_model.h5')==True:
            with open('model/tokenizer.json', 'r') as json_file:
                tokenizer_json = json.load(json_file)
                tokenizer = tokenizer_from_json(tokenizer_json)
            model = tf.keras.models.load_model('model/nlp_model.h5')
            txt = request.POST.getlist('textinput')
            seq = tokenizer.texts_to_sequences(txt)
            padded = pad_sequences(seq, maxlen=max_seq_len)
            start_time = time.time()
            pred = model.predict(padded)
            prediction_res = class_names[np.argmax(pred)]
            with open('model/data.json', 'r') as json_file:
                loaded_dict = json.load(json_file)
            accuracy = loaded_dict['Accuracy']
            f1_result = loaded_dict['F1 Score']
            if prediction_res!='None' and accuracy!=0 and f1_result!=0:
                messages.info(request, 'Prediction is '+prediction_res)
                messages.info(request, 'Accuracy is '+str(accuracy))
                messages.info(request, 'F1 Score is '+str(f1_result))
            else:
                pass
        else:
            pass
    return render(request, 'comp/home.html')