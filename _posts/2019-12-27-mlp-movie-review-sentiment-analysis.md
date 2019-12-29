---
layout: post
title: Multilayer Perceptron for Movie Sentiment Analysis
categories: [Processed Data]
tags: [NLP, MLP, Keras, Tensorflow, NLTK, Bag of Words]
---

Use TF Keras to build and test various MLPs on Movie Sentiment Analysis. Use NLTK to clean data.

Part 1: clean text data, generate vocabulary, transform data  
Part 2: build various MLP models (1 hidden layer, 2 hidden layers)  
Part 3: build testing harness  
Part 4: test various MLP models and encoding schemes  
Part 5: test on two real reviews  

Win condition: >87% accuracy on test split (87% is the upper bound for SVM and other traditional ML techniques on this data, see: http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf

Attributions:
machinelearningmastery.com DL for NLP book

polarity dataset v2.0 ( 3.0Mb) (includes README v2.0): 1000 positive and 1000 negative processed reviews. Introduced in Pang/Lee ACL 2004. Released June 2004.

## Import Libraries


```python
import nltk
from nltk.corpus import stopwords
from collections import Counter
from os import listdir
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import tensorflow.keras as tk
%load_ext tensorboard
```

    The tensorboard extension is already loaded. To reload it, use:
      %reload_ext tensorboard


## Data Engineering


```python
root_dir = 'review_polarity/txt_sentoken/'
neg_train_dir = root_dir + 'neg_train'
neg_test_dir = root_dir + 'neg_test'
pos_train_dir = root_dir + 'pos_train'
pos_test_dir = root_dir + 'pos_test'
```

Data cleaning function


```python
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(text):
    words = nltk.word_tokenize(text)
    alpha_words = [w for w in words if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    relevant_words = [w for w in alpha_words if w not in stop_words]
    filtered_words = [w for w in relevant_words if len(w)>1]
    return filtered_words
```

Build a vocabulary with the training data


```python
def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

def process_docs_to_vocab(directory, vocab):
    i=0
    for filename in listdir(directory):
        if filename.startswith('cv'):
            path = directory + '/' + filename
            add_doc_to_vocab(path, vocab)
            i+=1
    print(f'Processed {i} docs.')
    return vocab
```


```python
vocab = Counter()
process_docs_to_vocab(pos_train_dir, vocab)
process_docs_to_vocab(neg_train_dir, vocab)
print(len(vocab))
```

    Processed 900 docs.
    Processed 900 docs.
    36388



```python
print(vocab.most_common(25))
```

    [('film', 8513), ('movie', 5032), ('one', 5002), ('like', 3196), ('even', 2262), ('good', 2076), ('time', 2041), ('would', 2037), ('story', 1932), ('much', 1825), ('character', 1783), ('also', 1757), ('get', 1728), ('characters', 1655), ('two', 1645), ('first', 1588), ('see', 1558), ('way', 1516), ('well', 1479), ('could', 1444), ('make', 1420), ('really', 1400), ('little', 1350), ('films', 1345), ('life', 1343)]



```python
def filter_vocab(vocab, min_occurrences=5):
    tokens = [k for k, c in vocab.items() if c >= min_occurrences]
    print(len(tokens))
    return tokens
```


```python
filtered_vocab = filter_vocab(vocab, 2)
```

    23548



```python
def save_list(tokens, filename):
    if type(tokens[0]) != str:
        tokens = str(tokens)
    data = '\n'.join(tokens)
    file = open(filename, 'w')
    file.write(data)
    file.close()
```


```python
save_list(filtered_vocab, 'vocab.txt')
```

Now user our vocabulary to process our data


```python
vocab_set = set(load_doc('vocab.txt').split())
print(len(vocab_set))
```

    23548



```python
def doc_to_line(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab_tokens = [w for w in tokens if w in vocab_set]
    return ' '.join(vocab_tokens)

def process_docs_to_lines(directory, vocab):
    lines = list()
    for filename in listdir(directory):
        if filename.startswith('cv'):
            path = directory + '/' + filename
            line = doc_to_line(path, vocab)
            lines.append(line)
    return lines
```


```python
neg_train = process_docs_to_lines(neg_train_dir, vocab_set)
pos_train = process_docs_to_lines(pos_train_dir, vocab_set)
neg_test = process_docs_to_lines(neg_test_dir, vocab_set)
pos_test = process_docs_to_lines(pos_test_dir, vocab_set)
```


```python
trainX, trainY = neg_train+pos_train, [0]*len(neg_train)+[1]*len(pos_train)
testX, testY = neg_test+pos_test, [0]*len(neg_test)+[1]*len(pos_test)

print(len(trainX), len(trainY))
print(len(testX), len(testY))
```

    1800 1800
    200 200



```python
save_list(trainX, 'trainX.txt')
save_list(trainY, 'trainY.txt')
save_list(testX, 'testX.txt')
save_list(testY, 'testY.txt')
```

Transform data to prepare for modelling, using BOW representation


```python
processed_data = {}
processed_data['trainX'] = trainX
processed_data['trainY'] = trainY
processed_data['testX'] = testX
processed_data['testY'] = testY
```


```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainX)

def gen_encodings(X, tokenizer):
    output={}
    modes = ['binary', 'count', 'tfidf', 'freq']
    for mode in modes:
        output[mode] = tokenizer.texts_to_matrix(X, mode=mode)
    return output

trainX_dict, testX_dict = gen_encodings(processed_data['trainX'], tokenizer), gen_encodings(processed_data['testX'], tokenizer)

print(trainX_dict['binary'][:10])
```

    [[0. 1. 1. ... 0. 0. 0.]
     [0. 1. 1. ... 0. 0. 0.]
     [0. 1. 1. ... 0. 0. 0.]
     ...
     [0. 1. 1. ... 0. 0. 0.]
     [0. 1. 0. ... 0. 0. 0.]
     [0. 1. 1. ... 0. 0. 0.]]


## Build Models


```python
input_vec_len = trainX_dict['binary'].shape[1]
```


```python
mlp1 = Sequential(name='mlp1')

mlp1.add(Dense(50, input_shape=(input_vec_len, ), activation='relu'))
mlp1.add(Dense(1, activation='sigmoid'))

mlp1.summary()
plot_model(mlp1)
```

    Model: "mlp1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_11 (Dense)             (None, 50)                1177500   
    _________________________________________________________________
    dense_12 (Dense)             (None, 1)                 51        
    =================================================================
    Total params: 1,177,551
    Trainable params: 1,177,551
    Non-trainable params: 0
    _________________________________________________________________





![png](Multilayer%20Perceptron%20Movie%20Review%20Sentiment%20Analysis_files/Multilayer%20Perceptron%20Movie%20Review%20Sentiment%20Analysis_26_1.png)




```python
mlp2 = Sequential(name='mlp2')

mlp2.add(Dense(25, input_shape=(input_vec_len,), activation='relu'))
mlp2.add(Dense(25, activation='relu'))
mlp2.add(Dense(1, activation='sigmoid'))

mlp2.summary()
plot_model(mlp2)
```

    Model: "mlp2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_13 (Dense)             (None, 25)                588750    
    _________________________________________________________________
    dense_14 (Dense)             (None, 25)                650       
    _________________________________________________________________
    dense_15 (Dense)             (None, 1)                 26        
    =================================================================
    Total params: 589,426
    Trainable params: 589,426
    Non-trainable params: 0
    _________________________________________________________________





![png](Multilayer%20Perceptron%20Movie%20Review%20Sentiment%20Analysis_files/Multilayer%20Perceptron%20Movie%20Review%20Sentiment%20Analysis_27_1.png)



## Build a Testing Harness


```python
def gen_model(name, input_vec_len):
    if name == 'mlp1':
        model = Sequential(name='mlp1')
        model.add(Dense(50, input_shape=(input_vec_len, ), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    if name =='mlp2':
        model = Sequential(name='mlp2')
        model.add(Dense(25, input_shape=(input_vec_len,), activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model        
```


```python
def evaluate_model(model_name, data, n_repeats=5):
    trainX, trainY, testX, testY = data
    scores = []

    # create tensorboard callback
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    for i in range(n_repeats):
        model = gen_model(model_name, trainX.shape[1])
        H = model.fit(trainX, trainY,
                      validation_data=(testX, testY),
                      epochs=10,
                      callbacks=[tb_callback],
                      verbose=2)
        scores.append(H.history['val_accuracy'])

    return scores
```


```python
models = ['mlp1', 'mlp2']
```


```python
trainY = np.array(trainY)
testY = np.array(testY)
```


```python
results = pd.DataFrame()
for model_name in models:
    for mode in trainX_dict.keys():
        data = trainX_dict[mode], trainY, testX_dict[mode], testY
        results[model_name,'and',mode] = evaluate_model(model_name, data)
```

    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 5s - loss: 0.4730 - accuracy: 0.7856 - val_loss: 0.2800 - val_accuracy: 0.9150
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0608 - accuracy: 0.9944 - val_loss: 0.2422 - val_accuracy: 0.9200
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0175 - accuracy: 1.0000 - val_loss: 0.2269 - val_accuracy: 0.9100
    Epoch 4/10
    1800/1800 - 3s - loss: 0.0077 - accuracy: 1.0000 - val_loss: 0.2221 - val_accuracy: 0.9200
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0042 - accuracy: 1.0000 - val_loss: 0.2201 - val_accuracy: 0.9200
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0026 - accuracy: 1.0000 - val_loss: 0.2208 - val_accuracy: 0.9200
    Epoch 7/10
    1800/1800 - 2s - loss: 0.0017 - accuracy: 1.0000 - val_loss: 0.2201 - val_accuracy: 0.9200
    Epoch 8/10
    1800/1800 - 2s - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.2210 - val_accuracy: 0.9200
    Epoch 9/10
    1800/1800 - 2s - loss: 9.2178e-04 - accuracy: 1.0000 - val_loss: 0.2225 - val_accuracy: 0.9200
    Epoch 10/10
    1800/1800 - 2s - loss: 7.2100e-04 - accuracy: 1.0000 - val_loss: 0.2233 - val_accuracy: 0.9200
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 3s - loss: 0.4734 - accuracy: 0.7761 - val_loss: 0.2825 - val_accuracy: 0.9000
    Epoch 2/10
    1800/1800 - 3s - loss: 0.0632 - accuracy: 0.9928 - val_loss: 0.2215 - val_accuracy: 0.9200
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0179 - accuracy: 1.0000 - val_loss: 0.2136 - val_accuracy: 0.9350
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0085 - accuracy: 1.0000 - val_loss: 0.2117 - val_accuracy: 0.9300
    Epoch 5/10
    1800/1800 - 3s - loss: 0.0047 - accuracy: 1.0000 - val_loss: 0.2114 - val_accuracy: 0.9200
    Epoch 6/10
    1800/1800 - 3s - loss: 0.0027 - accuracy: 1.0000 - val_loss: 0.2113 - val_accuracy: 0.9200
    Epoch 7/10
    1800/1800 - 2s - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.2129 - val_accuracy: 0.9250
    Epoch 8/10
    1800/1800 - 2s - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.2139 - val_accuracy: 0.9200
    Epoch 9/10
    1800/1800 - 2s - loss: 9.7307e-04 - accuracy: 1.0000 - val_loss: 0.2161 - val_accuracy: 0.9250
    Epoch 10/10
    1800/1800 - 2s - loss: 7.5617e-04 - accuracy: 1.0000 - val_loss: 0.2172 - val_accuracy: 0.9250
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.4931 - accuracy: 0.7678 - val_loss: 0.2886 - val_accuracy: 0.9000
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0865 - accuracy: 0.9861 - val_loss: 0.2353 - val_accuracy: 0.9300
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0214 - accuracy: 1.0000 - val_loss: 0.2124 - val_accuracy: 0.9200
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0087 - accuracy: 1.0000 - val_loss: 0.2074 - val_accuracy: 0.9250
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0043 - accuracy: 1.0000 - val_loss: 0.2063 - val_accuracy: 0.9250
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0023 - accuracy: 1.0000 - val_loss: 0.2053 - val_accuracy: 0.9300
    Epoch 7/10
    1800/1800 - 2s - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.2059 - val_accuracy: 0.9300
    Epoch 8/10
    1800/1800 - 2s - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.2061 - val_accuracy: 0.9250
    Epoch 9/10
    1800/1800 - 2s - loss: 7.5248e-04 - accuracy: 1.0000 - val_loss: 0.2098 - val_accuracy: 0.9200
    Epoch 10/10
    1800/1800 - 3s - loss: 5.7766e-04 - accuracy: 1.0000 - val_loss: 0.2111 - val_accuracy: 0.9200
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.4953 - accuracy: 0.7633 - val_loss: 0.2893 - val_accuracy: 0.9050
    Epoch 2/10
    1800/1800 - 3s - loss: 0.0819 - accuracy: 0.9922 - val_loss: 0.2240 - val_accuracy: 0.9250
    Epoch 3/10
    1800/1800 - 3s - loss: 0.0202 - accuracy: 1.0000 - val_loss: 0.2119 - val_accuracy: 0.9400
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0076 - accuracy: 1.0000 - val_loss: 0.2058 - val_accuracy: 0.9350
    Epoch 5/10
    1800/1800 - 3s - loss: 0.0035 - accuracy: 1.0000 - val_loss: 0.2025 - val_accuracy: 0.9300
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.2054 - val_accuracy: 0.9300
    Epoch 7/10
    1800/1800 - 2s - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.2075 - val_accuracy: 0.9300
    Epoch 8/10
    1800/1800 - 2s - loss: 8.4367e-04 - accuracy: 1.0000 - val_loss: 0.2090 - val_accuracy: 0.9300
    Epoch 9/10
    1800/1800 - 2s - loss: 6.2169e-04 - accuracy: 1.0000 - val_loss: 0.2114 - val_accuracy: 0.9250
    Epoch 10/10
    1800/1800 - 3s - loss: 4.7267e-04 - accuracy: 1.0000 - val_loss: 0.2123 - val_accuracy: 0.9250
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.4785 - accuracy: 0.7683 - val_loss: 0.2871 - val_accuracy: 0.9150
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0659 - accuracy: 0.9928 - val_loss: 0.2442 - val_accuracy: 0.9100
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0205 - accuracy: 0.9994 - val_loss: 0.2298 - val_accuracy: 0.9300
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0095 - accuracy: 1.0000 - val_loss: 0.2253 - val_accuracy: 0.9300
    Epoch 5/10
    1800/1800 - 3s - loss: 0.0047 - accuracy: 1.0000 - val_loss: 0.2215 - val_accuracy: 0.9300
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0025 - accuracy: 1.0000 - val_loss: 0.2232 - val_accuracy: 0.9200
    Epoch 7/10
    1800/1800 - 2s - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.2244 - val_accuracy: 0.9200
    Epoch 8/10
    1800/1800 - 2s - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.2264 - val_accuracy: 0.9250
    Epoch 9/10
    1800/1800 - 2s - loss: 7.1370e-04 - accuracy: 1.0000 - val_loss: 0.2274 - val_accuracy: 0.9250
    Epoch 10/10
    1800/1800 - 2s - loss: 5.2696e-04 - accuracy: 1.0000 - val_loss: 0.2305 - val_accuracy: 0.9150
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 5s - loss: 0.4456 - accuracy: 0.7950 - val_loss: 0.2984 - val_accuracy: 0.9000
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0493 - accuracy: 0.9939 - val_loss: 0.2747 - val_accuracy: 0.8900
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0142 - accuracy: 1.0000 - val_loss: 0.2811 - val_accuracy: 0.8850
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0061 - accuracy: 1.0000 - val_loss: 0.2878 - val_accuracy: 0.8900
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0034 - accuracy: 1.0000 - val_loss: 0.2974 - val_accuracy: 0.8850
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0022 - accuracy: 1.0000 - val_loss: 0.3024 - val_accuracy: 0.8900
    Epoch 7/10
    1800/1800 - 2s - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.3101 - val_accuracy: 0.8900
    Epoch 8/10
    1800/1800 - 2s - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.3171 - val_accuracy: 0.8900
    Epoch 9/10
    1800/1800 - 2s - loss: 8.8085e-04 - accuracy: 1.0000 - val_loss: 0.3226 - val_accuracy: 0.8950
    Epoch 10/10
    1800/1800 - 2s - loss: 6.9886e-04 - accuracy: 1.0000 - val_loss: 0.3281 - val_accuracy: 0.8950
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.4814 - accuracy: 0.7644 - val_loss: 0.3684 - val_accuracy: 0.8550
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0658 - accuracy: 0.9894 - val_loss: 0.3094 - val_accuracy: 0.8900
    Epoch 3/10
    1800/1800 - 3s - loss: 0.0161 - accuracy: 1.0000 - val_loss: 0.3134 - val_accuracy: 0.9000
    Epoch 4/10
    1800/1800 - 3s - loss: 0.0077 - accuracy: 1.0000 - val_loss: 0.3204 - val_accuracy: 0.9000
    Epoch 5/10
    1800/1800 - 3s - loss: 0.0047 - accuracy: 1.0000 - val_loss: 0.3259 - val_accuracy: 0.9050
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0032 - accuracy: 1.0000 - val_loss: 0.3326 - val_accuracy: 0.9050
    Epoch 7/10
    1800/1800 - 2s - loss: 0.0023 - accuracy: 1.0000 - val_loss: 0.3388 - val_accuracy: 0.9000
    Epoch 8/10
    1800/1800 - 2s - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.3435 - val_accuracy: 0.9000
    Epoch 9/10
    1800/1800 - 2s - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.3505 - val_accuracy: 0.9000
    Epoch 10/10
    1800/1800 - 2s - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.3547 - val_accuracy: 0.9000
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 5s - loss: 0.4860 - accuracy: 0.7639 - val_loss: 0.3213 - val_accuracy: 0.8850
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0761 - accuracy: 0.9900 - val_loss: 0.2792 - val_accuracy: 0.9100
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0217 - accuracy: 1.0000 - val_loss: 0.2805 - val_accuracy: 0.9000
    Epoch 4/10
    1800/1800 - 4s - loss: 0.0092 - accuracy: 1.0000 - val_loss: 0.2888 - val_accuracy: 0.9050
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0047 - accuracy: 1.0000 - val_loss: 0.3042 - val_accuracy: 0.8950
    Epoch 6/10
    1800/1800 - 4s - loss: 0.0026 - accuracy: 1.0000 - val_loss: 0.3159 - val_accuracy: 0.9000
    Epoch 7/10
    1800/1800 - 6s - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.3250 - val_accuracy: 0.9000
    Epoch 8/10
    1800/1800 - 5s - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.3321 - val_accuracy: 0.9050
    Epoch 9/10
    1800/1800 - 4s - loss: 7.6161e-04 - accuracy: 1.0000 - val_loss: 0.3406 - val_accuracy: 0.9050
    Epoch 10/10
    1800/1800 - 8s - loss: 5.8231e-04 - accuracy: 1.0000 - val_loss: 0.3458 - val_accuracy: 0.9000
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 12s - loss: 0.4570 - accuracy: 0.7800 - val_loss: 0.3003 - val_accuracy: 0.8900
    Epoch 2/10
    1800/1800 - 5s - loss: 0.0491 - accuracy: 0.9956 - val_loss: 0.3032 - val_accuracy: 0.8950
    Epoch 3/10
    1800/1800 - 4s - loss: 0.0145 - accuracy: 1.0000 - val_loss: 0.3008 - val_accuracy: 0.9000
    Epoch 4/10
    1800/1800 - 4s - loss: 0.0063 - accuracy: 1.0000 - val_loss: 0.3107 - val_accuracy: 0.9000
    Epoch 5/10
    1800/1800 - 3s - loss: 0.0037 - accuracy: 1.0000 - val_loss: 0.3183 - val_accuracy: 0.9050
    Epoch 6/10
    1800/1800 - 3s - loss: 0.0025 - accuracy: 1.0000 - val_loss: 0.3227 - val_accuracy: 0.9100
    Epoch 7/10
    1800/1800 - 3s - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.3286 - val_accuracy: 0.9100
    Epoch 8/10
    1800/1800 - 4s - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.3339 - val_accuracy: 0.9100
    Epoch 9/10
    1800/1800 - 4s - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.3371 - val_accuracy: 0.9050
    Epoch 10/10
    1800/1800 - 3s - loss: 8.6128e-04 - accuracy: 1.0000 - val_loss: 0.3419 - val_accuracy: 0.9050
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 11s - loss: 0.4637 - accuracy: 0.7761 - val_loss: 0.3359 - val_accuracy: 0.8650
    Epoch 2/10
    1800/1800 - 10s - loss: 0.0678 - accuracy: 0.9811 - val_loss: 0.3050 - val_accuracy: 0.9000
    Epoch 3/10
    1800/1800 - 7s - loss: 0.0153 - accuracy: 1.0000 - val_loss: 0.3023 - val_accuracy: 0.9150
    Epoch 4/10
    1800/1800 - 4s - loss: 0.0065 - accuracy: 1.0000 - val_loss: 0.3092 - val_accuracy: 0.9050
    Epoch 5/10
    1800/1800 - 3s - loss: 0.0036 - accuracy: 1.0000 - val_loss: 0.3173 - val_accuracy: 0.8950
    Epoch 6/10
    1800/1800 - 5s - loss: 0.0023 - accuracy: 1.0000 - val_loss: 0.3244 - val_accuracy: 0.8950
    Epoch 7/10
    1800/1800 - 8s - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.3304 - val_accuracy: 0.9000
    Epoch 8/10
    1800/1800 - 6s - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.3374 - val_accuracy: 0.9000
    Epoch 9/10
    1800/1800 - 5s - loss: 8.3588e-04 - accuracy: 1.0000 - val_loss: 0.3444 - val_accuracy: 0.9000
    Epoch 10/10
    1800/1800 - 6s - loss: 6.3720e-04 - accuracy: 1.0000 - val_loss: 0.3509 - val_accuracy: 0.8950
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 22s - loss: 0.4787 - accuracy: 0.7700 - val_loss: 0.2471 - val_accuracy: 0.9000
    Epoch 2/10
    1800/1800 - 3s - loss: 0.0171 - accuracy: 0.9989 - val_loss: 0.2410 - val_accuracy: 0.9000
    Epoch 3/10
    1800/1800 - 3s - loss: 0.0037 - accuracy: 1.0000 - val_loss: 0.2419 - val_accuracy: 0.9000
    Epoch 4/10
    1800/1800 - 4s - loss: 0.0020 - accuracy: 1.0000 - val_loss: 0.2488 - val_accuracy: 0.9000
    Epoch 5/10
    1800/1800 - 4s - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.2571 - val_accuracy: 0.8950
    Epoch 6/10
    1800/1800 - 5s - loss: 8.1434e-04 - accuracy: 1.0000 - val_loss: 0.2640 - val_accuracy: 0.8950
    Epoch 7/10
    1800/1800 - 5s - loss: 5.7475e-04 - accuracy: 1.0000 - val_loss: 0.2728 - val_accuracy: 0.8900
    Epoch 8/10
    1800/1800 - 5s - loss: 4.2358e-04 - accuracy: 1.0000 - val_loss: 0.2817 - val_accuracy: 0.8900
    Epoch 9/10
    1800/1800 - 3s - loss: 3.2424e-04 - accuracy: 1.0000 - val_loss: 0.2890 - val_accuracy: 0.8800
    Epoch 10/10
    1800/1800 - 4s - loss: 2.5493e-04 - accuracy: 1.0000 - val_loss: 0.2955 - val_accuracy: 0.8850
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 6s - loss: 0.4819 - accuracy: 0.7689 - val_loss: 0.2794 - val_accuracy: 0.8900
    Epoch 2/10
    1800/1800 - 4s - loss: 0.0222 - accuracy: 0.9978 - val_loss: 0.2757 - val_accuracy: 0.9050
    Epoch 3/10
    1800/1800 - 3s - loss: 0.0051 - accuracy: 0.9994 - val_loss: 0.2756 - val_accuracy: 0.9050
    Epoch 4/10
    1800/1800 - 7s - loss: 0.0023 - accuracy: 1.0000 - val_loss: 0.2855 - val_accuracy: 0.9000
    Epoch 5/10
    1800/1800 - 9s - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.2921 - val_accuracy: 0.8950
    Epoch 6/10
    1800/1800 - 8s - loss: 9.3610e-04 - accuracy: 1.0000 - val_loss: 0.2995 - val_accuracy: 0.9000
    Epoch 7/10
    1800/1800 - 10s - loss: 6.6391e-04 - accuracy: 1.0000 - val_loss: 0.3052 - val_accuracy: 0.8950
    Epoch 8/10
    1800/1800 - 6s - loss: 4.9056e-04 - accuracy: 1.0000 - val_loss: 0.3124 - val_accuracy: 0.8950
    Epoch 9/10
    1800/1800 - 5s - loss: 3.7110e-04 - accuracy: 1.0000 - val_loss: 0.3193 - val_accuracy: 0.8950
    Epoch 10/10
    1800/1800 - 4s - loss: 2.8737e-04 - accuracy: 1.0000 - val_loss: 0.3270 - val_accuracy: 0.8950
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 10s - loss: 0.4798 - accuracy: 0.7761 - val_loss: 0.2964 - val_accuracy: 0.8600
    Epoch 2/10
    1800/1800 - 4s - loss: 0.0174 - accuracy: 0.9983 - val_loss: 0.2792 - val_accuracy: 0.8800
    Epoch 3/10
    1800/1800 - 3s - loss: 0.0041 - accuracy: 1.0000 - val_loss: 0.2817 - val_accuracy: 0.8750
    Epoch 4/10
    1800/1800 - 3s - loss: 0.0022 - accuracy: 1.0000 - val_loss: 0.2854 - val_accuracy: 0.8800
    Epoch 5/10
    1800/1800 - 4s - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.2919 - val_accuracy: 0.8800
    Epoch 6/10
    1800/1800 - 3s - loss: 9.7101e-04 - accuracy: 1.0000 - val_loss: 0.2989 - val_accuracy: 0.8700
    Epoch 7/10
    1800/1800 - 5s - loss: 7.0166e-04 - accuracy: 1.0000 - val_loss: 0.3048 - val_accuracy: 0.8700
    Epoch 8/10
    1800/1800 - 3s - loss: 5.2488e-04 - accuracy: 1.0000 - val_loss: 0.3098 - val_accuracy: 0.8700
    Epoch 9/10
    1800/1800 - 4s - loss: 4.0910e-04 - accuracy: 1.0000 - val_loss: 0.3150 - val_accuracy: 0.8750
    Epoch 10/10
    1800/1800 - 6s - loss: 3.2565e-04 - accuracy: 1.0000 - val_loss: 0.3202 - val_accuracy: 0.8750
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 14s - loss: 0.4564 - accuracy: 0.7806 - val_loss: 0.3002 - val_accuracy: 0.8950
    Epoch 2/10
    1800/1800 - 9s - loss: 0.0125 - accuracy: 0.9989 - val_loss: 0.2863 - val_accuracy: 0.8950
    Epoch 3/10
    1800/1800 - 6s - loss: 0.0037 - accuracy: 1.0000 - val_loss: 0.2946 - val_accuracy: 0.9000
    Epoch 4/10
    1800/1800 - 4s - loss: 0.0021 - accuracy: 1.0000 - val_loss: 0.3022 - val_accuracy: 0.9000
    Epoch 5/10
    1800/1800 - 3s - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.3096 - val_accuracy: 0.8950
    Epoch 6/10
    1800/1800 - 4s - loss: 9.0126e-04 - accuracy: 1.0000 - val_loss: 0.3178 - val_accuracy: 0.8950
    Epoch 7/10
    1800/1800 - 3s - loss: 6.4940e-04 - accuracy: 1.0000 - val_loss: 0.3243 - val_accuracy: 0.8950
    Epoch 8/10
    1800/1800 - 3s - loss: 4.9022e-04 - accuracy: 1.0000 - val_loss: 0.3297 - val_accuracy: 0.8950
    Epoch 9/10
    1800/1800 - 3s - loss: 3.8148e-04 - accuracy: 1.0000 - val_loss: 0.3355 - val_accuracy: 0.8900
    Epoch 10/10
    1800/1800 - 3s - loss: 3.0373e-04 - accuracy: 1.0000 - val_loss: 0.3399 - val_accuracy: 0.8800
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 6s - loss: 0.4792 - accuracy: 0.7667 - val_loss: 0.2925 - val_accuracy: 0.8850
    Epoch 2/10
    1800/1800 - 3s - loss: 0.0167 - accuracy: 0.9983 - val_loss: 0.2943 - val_accuracy: 0.8850
    Epoch 3/10
    1800/1800 - 3s - loss: 0.0041 - accuracy: 1.0000 - val_loss: 0.3024 - val_accuracy: 0.8850
    Epoch 4/10
    1800/1800 - 4s - loss: 0.0021 - accuracy: 1.0000 - val_loss: 0.3065 - val_accuracy: 0.8900
    Epoch 5/10
    1800/1800 - 5s - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.3098 - val_accuracy: 0.8900
    Epoch 6/10
    1800/1800 - 6s - loss: 8.8100e-04 - accuracy: 1.0000 - val_loss: 0.3159 - val_accuracy: 0.8850
    Epoch 7/10
    1800/1800 - 2s - loss: 6.3283e-04 - accuracy: 1.0000 - val_loss: 0.3214 - val_accuracy: 0.8900
    Epoch 8/10
    1800/1800 - 3s - loss: 4.7008e-04 - accuracy: 1.0000 - val_loss: 0.3260 - val_accuracy: 0.8900
    Epoch 9/10
    1800/1800 - 6s - loss: 3.6328e-04 - accuracy: 1.0000 - val_loss: 0.3293 - val_accuracy: 0.8900
    Epoch 10/10
    1800/1800 - 14s - loss: 2.8839e-04 - accuracy: 1.0000 - val_loss: 0.3335 - val_accuracy: 0.8900
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 9s - loss: 0.6917 - accuracy: 0.5711 - val_loss: 0.6892 - val_accuracy: 0.7150
    Epoch 2/10
    1800/1800 - 4s - loss: 0.6827 - accuracy: 0.6300 - val_loss: 0.6804 - val_accuracy: 0.8250
    Epoch 3/10
    1800/1800 - 3s - loss: 0.6648 - accuracy: 0.9011 - val_loss: 0.6648 - val_accuracy: 0.8500
    Epoch 4/10
    1800/1800 - 3s - loss: 0.6349 - accuracy: 0.9094 - val_loss: 0.6422 - val_accuracy: 0.8600
    Epoch 5/10
    1800/1800 - 3s - loss: 0.5949 - accuracy: 0.9389 - val_loss: 0.6147 - val_accuracy: 0.8700
    Epoch 6/10
    1800/1800 - 3s - loss: 0.5494 - accuracy: 0.9356 - val_loss: 0.5874 - val_accuracy: 0.8500
    Epoch 7/10
    1800/1800 - 2s - loss: 0.5018 - accuracy: 0.9472 - val_loss: 0.5568 - val_accuracy: 0.8700
    Epoch 8/10
    1800/1800 - 3s - loss: 0.4538 - accuracy: 0.9561 - val_loss: 0.5285 - val_accuracy: 0.8700
    Epoch 9/10
    1800/1800 - 2s - loss: 0.4091 - accuracy: 0.9606 - val_loss: 0.5016 - val_accuracy: 0.8800
    Epoch 10/10
    1800/1800 - 2s - loss: 0.3667 - accuracy: 0.9661 - val_loss: 0.4765 - val_accuracy: 0.8800
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 5s - loss: 0.6919 - accuracy: 0.5117 - val_loss: 0.6891 - val_accuracy: 0.5500
    Epoch 2/10
    1800/1800 - 2s - loss: 0.6831 - accuracy: 0.7250 - val_loss: 0.6809 - val_accuracy: 0.8500
    Epoch 3/10
    1800/1800 - 2s - loss: 0.6656 - accuracy: 0.8867 - val_loss: 0.6654 - val_accuracy: 0.8250
    Epoch 4/10
    1800/1800 - 2s - loss: 0.6363 - accuracy: 0.9128 - val_loss: 0.6434 - val_accuracy: 0.8700
    Epoch 5/10
    1800/1800 - 2s - loss: 0.5980 - accuracy: 0.9317 - val_loss: 0.6184 - val_accuracy: 0.8200
    Epoch 6/10
    1800/1800 - 2s - loss: 0.5548 - accuracy: 0.9367 - val_loss: 0.5889 - val_accuracy: 0.8650
    Epoch 7/10
    1800/1800 - 2s - loss: 0.5079 - accuracy: 0.9533 - val_loss: 0.5621 - val_accuracy: 0.8450
    Epoch 8/10
    1800/1800 - 2s - loss: 0.4623 - accuracy: 0.9506 - val_loss: 0.5320 - val_accuracy: 0.8800
    Epoch 9/10
    1800/1800 - 3s - loss: 0.4167 - accuracy: 0.9644 - val_loss: 0.5050 - val_accuracy: 0.8800
    Epoch 10/10
    1800/1800 - 3s - loss: 0.3750 - accuracy: 0.9683 - val_loss: 0.4806 - val_accuracy: 0.8750
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 5s - loss: 0.6917 - accuracy: 0.5406 - val_loss: 0.6889 - val_accuracy: 0.7550
    Epoch 2/10
    1800/1800 - 2s - loss: 0.6813 - accuracy: 0.8183 - val_loss: 0.6796 - val_accuracy: 0.5750
    Epoch 3/10
    1800/1800 - 2s - loss: 0.6625 - accuracy: 0.7983 - val_loss: 0.6639 - val_accuracy: 0.6900
    Epoch 4/10
    1800/1800 - 2s - loss: 0.6336 - accuracy: 0.8044 - val_loss: 0.6427 - val_accuracy: 0.8350
    Epoch 5/10
    1800/1800 - 2s - loss: 0.5969 - accuracy: 0.9111 - val_loss: 0.6197 - val_accuracy: 0.7950
    Epoch 6/10
    1800/1800 - 2s - loss: 0.5557 - accuracy: 0.9406 - val_loss: 0.5910 - val_accuracy: 0.8500
    Epoch 7/10
    1800/1800 - 2s - loss: 0.5088 - accuracy: 0.9550 - val_loss: 0.5642 - val_accuracy: 0.8850
    Epoch 8/10
    1800/1800 - 2s - loss: 0.4641 - accuracy: 0.9561 - val_loss: 0.5365 - val_accuracy: 0.8450
    Epoch 9/10
    1800/1800 - 2s - loss: 0.4196 - accuracy: 0.9622 - val_loss: 0.5108 - val_accuracy: 0.8750
    Epoch 10/10
    1800/1800 - 2s - loss: 0.3791 - accuracy: 0.9672 - val_loss: 0.4870 - val_accuracy: 0.8900
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.6917 - accuracy: 0.5333 - val_loss: 0.6890 - val_accuracy: 0.5250
    Epoch 2/10
    1800/1800 - 3s - loss: 0.6825 - accuracy: 0.5528 - val_loss: 0.6805 - val_accuracy: 0.7900
    Epoch 3/10
    1800/1800 - 3s - loss: 0.6645 - accuracy: 0.7600 - val_loss: 0.6657 - val_accuracy: 0.6550
    Epoch 4/10
    1800/1800 - 2s - loss: 0.6369 - accuracy: 0.8289 - val_loss: 0.6461 - val_accuracy: 0.8300
    Epoch 5/10
    1800/1800 - 2s - loss: 0.6021 - accuracy: 0.9206 - val_loss: 0.6225 - val_accuracy: 0.8250
    Epoch 6/10
    1800/1800 - 2s - loss: 0.5616 - accuracy: 0.9267 - val_loss: 0.5972 - val_accuracy: 0.8350
    Epoch 7/10
    1800/1800 - 4s - loss: 0.5191 - accuracy: 0.9472 - val_loss: 0.5705 - val_accuracy: 0.8500
    Epoch 8/10
    1800/1800 - 5s - loss: 0.4748 - accuracy: 0.9522 - val_loss: 0.5446 - val_accuracy: 0.8800
    Epoch 9/10
    1800/1800 - 4s - loss: 0.4328 - accuracy: 0.9667 - val_loss: 0.5186 - val_accuracy: 0.8600
    Epoch 10/10
    1800/1800 - 4s - loss: 0.3922 - accuracy: 0.9672 - val_loss: 0.4955 - val_accuracy: 0.8750
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.6923 - accuracy: 0.4994 - val_loss: 0.6906 - val_accuracy: 0.6200
    Epoch 2/10
    1800/1800 - 2s - loss: 0.6868 - accuracy: 0.7028 - val_loss: 0.6857 - val_accuracy: 0.5100
    Epoch 3/10
    1800/1800 - 2s - loss: 0.6767 - accuracy: 0.7928 - val_loss: 0.6765 - val_accuracy: 0.8500
    Epoch 4/10
    1800/1800 - 3s - loss: 0.6593 - accuracy: 0.9117 - val_loss: 0.6637 - val_accuracy: 0.8750
    Epoch 5/10
    1800/1800 - 2s - loss: 0.6361 - accuracy: 0.9289 - val_loss: 0.6480 - val_accuracy: 0.7700
    Epoch 6/10
    1800/1800 - 2s - loss: 0.6089 - accuracy: 0.9094 - val_loss: 0.6289 - val_accuracy: 0.8350
    Epoch 7/10
    1800/1800 - 3s - loss: 0.5769 - accuracy: 0.9428 - val_loss: 0.6090 - val_accuracy: 0.8700
    Epoch 8/10
    1800/1800 - 2s - loss: 0.5434 - accuracy: 0.9500 - val_loss: 0.5878 - val_accuracy: 0.8650
    Epoch 9/10
    1800/1800 - 3s - loss: 0.5096 - accuracy: 0.9411 - val_loss: 0.5669 - val_accuracy: 0.8500
    Epoch 10/10
    1800/1800 - 3s - loss: 0.4757 - accuracy: 0.9583 - val_loss: 0.5459 - val_accuracy: 0.8850
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 5s - loss: 0.5301 - accuracy: 0.7700 - val_loss: 0.3101 - val_accuracy: 0.9150
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0915 - accuracy: 0.9856 - val_loss: 0.2202 - val_accuracy: 0.9250
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0147 - accuracy: 1.0000 - val_loss: 0.2087 - val_accuracy: 0.9200
    Epoch 4/10
    1800/1800 - 1s - loss: 0.0048 - accuracy: 1.0000 - val_loss: 0.2078 - val_accuracy: 0.9250
    Epoch 5/10
    1800/1800 - 1s - loss: 0.0025 - accuracy: 1.0000 - val_loss: 0.2106 - val_accuracy: 0.9250
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.2142 - val_accuracy: 0.9300
    Epoch 7/10
    1800/1800 - 2s - loss: 9.7253e-04 - accuracy: 1.0000 - val_loss: 0.2189 - val_accuracy: 0.9300
    Epoch 8/10
    1800/1800 - 2s - loss: 6.9451e-04 - accuracy: 1.0000 - val_loss: 0.2228 - val_accuracy: 0.9250
    Epoch 9/10
    1800/1800 - 2s - loss: 5.2103e-04 - accuracy: 1.0000 - val_loss: 0.2250 - val_accuracy: 0.9200
    Epoch 10/10
    1800/1800 - 3s - loss: 4.0284e-04 - accuracy: 1.0000 - val_loss: 0.2295 - val_accuracy: 0.9250
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.5038 - accuracy: 0.7817 - val_loss: 0.2924 - val_accuracy: 0.9000
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0726 - accuracy: 0.9894 - val_loss: 0.2176 - val_accuracy: 0.9150
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0118 - accuracy: 1.0000 - val_loss: 0.2095 - val_accuracy: 0.9200
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0048 - accuracy: 1.0000 - val_loss: 0.2110 - val_accuracy: 0.9250
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0028 - accuracy: 1.0000 - val_loss: 0.2109 - val_accuracy: 0.9250
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.2116 - val_accuracy: 0.9300
    Epoch 7/10
    1800/1800 - 5s - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.2131 - val_accuracy: 0.9300
    Epoch 8/10
    1800/1800 - 4s - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.2137 - val_accuracy: 0.9300
    Epoch 9/10
    1800/1800 - 2s - loss: 7.9496e-04 - accuracy: 1.0000 - val_loss: 0.2156 - val_accuracy: 0.9300
    Epoch 10/10
    1800/1800 - 2s - loss: 6.3993e-04 - accuracy: 1.0000 - val_loss: 0.2178 - val_accuracy: 0.9250
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.5154 - accuracy: 0.7739 - val_loss: 0.2960 - val_accuracy: 0.8950
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0826 - accuracy: 0.9889 - val_loss: 0.2816 - val_accuracy: 0.9000
    Epoch 3/10
    1800/1800 - 1s - loss: 0.0132 - accuracy: 1.0000 - val_loss: 0.2050 - val_accuracy: 0.9200
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0039 - accuracy: 1.0000 - val_loss: 0.2085 - val_accuracy: 0.9150
    Epoch 5/10
    1800/1800 - 1s - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.2111 - val_accuracy: 0.9150
    Epoch 6/10
    1800/1800 - 2s - loss: 9.3081e-04 - accuracy: 1.0000 - val_loss: 0.2158 - val_accuracy: 0.9200
    Epoch 7/10
    1800/1800 - 2s - loss: 5.9669e-04 - accuracy: 1.0000 - val_loss: 0.2167 - val_accuracy: 0.9200
    Epoch 8/10
    1800/1800 - 1s - loss: 4.1287e-04 - accuracy: 1.0000 - val_loss: 0.2200 - val_accuracy: 0.9250
    Epoch 9/10
    1800/1800 - 1s - loss: 2.9374e-04 - accuracy: 1.0000 - val_loss: 0.2223 - val_accuracy: 0.9300
    Epoch 10/10
    1800/1800 - 2s - loss: 2.1005e-04 - accuracy: 1.0000 - val_loss: 0.2274 - val_accuracy: 0.9250
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.5311 - accuracy: 0.7478 - val_loss: 0.3120 - val_accuracy: 0.8850
    Epoch 2/10
    1800/1800 - 1s - loss: 0.0989 - accuracy: 0.9800 - val_loss: 0.2340 - val_accuracy: 0.9150
    Epoch 3/10
    1800/1800 - 1s - loss: 0.0143 - accuracy: 1.0000 - val_loss: 0.2271 - val_accuracy: 0.9200
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0057 - accuracy: 1.0000 - val_loss: 0.2286 - val_accuracy: 0.9100
    Epoch 5/10
    1800/1800 - 1s - loss: 0.0030 - accuracy: 1.0000 - val_loss: 0.2299 - val_accuracy: 0.9100
    Epoch 6/10
    1800/1800 - 1s - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.2319 - val_accuracy: 0.9100
    Epoch 7/10
    1800/1800 - 1s - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.2341 - val_accuracy: 0.9050
    Epoch 8/10
    1800/1800 - 1s - loss: 8.1660e-04 - accuracy: 1.0000 - val_loss: 0.2372 - val_accuracy: 0.9050
    Epoch 9/10
    1800/1800 - 2s - loss: 6.0181e-04 - accuracy: 1.0000 - val_loss: 0.2398 - val_accuracy: 0.9050
    Epoch 10/10
    1800/1800 - 1s - loss: 4.6363e-04 - accuracy: 1.0000 - val_loss: 0.2427 - val_accuracy: 0.9050
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 3s - loss: 0.5577 - accuracy: 0.7450 - val_loss: 0.3526 - val_accuracy: 0.8900
    Epoch 2/10
    1800/1800 - 1s - loss: 0.1184 - accuracy: 0.9739 - val_loss: 0.2531 - val_accuracy: 0.9200
    Epoch 3/10
    1800/1800 - 1s - loss: 0.0189 - accuracy: 0.9994 - val_loss: 0.2194 - val_accuracy: 0.9150
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0056 - accuracy: 1.0000 - val_loss: 0.2216 - val_accuracy: 0.9150
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.2188 - val_accuracy: 0.9150
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0020 - accuracy: 1.0000 - val_loss: 0.2204 - val_accuracy: 0.9150
    Epoch 7/10
    1800/1800 - 1s - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.2256 - val_accuracy: 0.9150
    Epoch 8/10
    1800/1800 - 2s - loss: 8.8732e-04 - accuracy: 1.0000 - val_loss: 0.2300 - val_accuracy: 0.9100
    Epoch 9/10
    1800/1800 - 1s - loss: 6.2414e-04 - accuracy: 1.0000 - val_loss: 0.2346 - val_accuracy: 0.9100
    Epoch 10/10
    1800/1800 - 2s - loss: 4.5349e-04 - accuracy: 1.0000 - val_loss: 0.2411 - val_accuracy: 0.8950
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.5146 - accuracy: 0.7539 - val_loss: 0.3915 - val_accuracy: 0.8250
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0938 - accuracy: 0.9778 - val_loss: 0.2979 - val_accuracy: 0.9050
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0141 - accuracy: 1.0000 - val_loss: 0.3167 - val_accuracy: 0.9050
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0052 - accuracy: 1.0000 - val_loss: 0.3369 - val_accuracy: 0.8950
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0029 - accuracy: 1.0000 - val_loss: 0.3402 - val_accuracy: 0.9050
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.3505 - val_accuracy: 0.9050
    Epoch 7/10
    1800/1800 - 2s - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.3598 - val_accuracy: 0.9000
    Epoch 8/10
    1800/1800 - 3s - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.3672 - val_accuracy: 0.9000
    Epoch 9/10
    1800/1800 - 2s - loss: 8.2438e-04 - accuracy: 1.0000 - val_loss: 0.3754 - val_accuracy: 0.9000
    Epoch 10/10
    1800/1800 - 2s - loss: 6.6326e-04 - accuracy: 1.0000 - val_loss: 0.3812 - val_accuracy: 0.9000
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 5s - loss: 0.5129 - accuracy: 0.7550 - val_loss: 0.3398 - val_accuracy: 0.8850
    Epoch 2/10
    1800/1800 - 1s - loss: 0.0943 - accuracy: 0.9817 - val_loss: 0.3190 - val_accuracy: 0.8950
    Epoch 3/10
    1800/1800 - 1s - loss: 0.0148 - accuracy: 1.0000 - val_loss: 0.3179 - val_accuracy: 0.9050
    Epoch 4/10
    1800/1800 - 1s - loss: 0.0056 - accuracy: 1.0000 - val_loss: 0.3305 - val_accuracy: 0.9050
    Epoch 5/10
    1800/1800 - 1s - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.3400 - val_accuracy: 0.9000
    Epoch 6/10
    1800/1800 - 1s - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.3514 - val_accuracy: 0.8950
    Epoch 7/10
    1800/1800 - 1s - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.3621 - val_accuracy: 0.8950
    Epoch 8/10
    1800/1800 - 1s - loss: 9.4440e-04 - accuracy: 1.0000 - val_loss: 0.3720 - val_accuracy: 0.8900
    Epoch 9/10
    1800/1800 - 1s - loss: 7.1556e-04 - accuracy: 1.0000 - val_loss: 0.3818 - val_accuracy: 0.8950
    Epoch 10/10
    1800/1800 - 1s - loss: 5.6291e-04 - accuracy: 1.0000 - val_loss: 0.3912 - val_accuracy: 0.8900
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.4834 - accuracy: 0.7672 - val_loss: 0.3254 - val_accuracy: 0.9000
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0707 - accuracy: 0.9839 - val_loss: 0.3098 - val_accuracy: 0.9000
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0136 - accuracy: 1.0000 - val_loss: 0.3162 - val_accuracy: 0.9000
    Epoch 4/10
    1800/1800 - 1s - loss: 0.0054 - accuracy: 1.0000 - val_loss: 0.3302 - val_accuracy: 0.9050
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.3455 - val_accuracy: 0.9050
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.3615 - val_accuracy: 0.9050
    Epoch 7/10
    1800/1800 - 2s - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.3807 - val_accuracy: 0.9000
    Epoch 8/10
    1800/1800 - 2s - loss: 8.3684e-04 - accuracy: 1.0000 - val_loss: 0.3965 - val_accuracy: 0.8950
    Epoch 9/10
    1800/1800 - 1s - loss: 5.9155e-04 - accuracy: 1.0000 - val_loss: 0.4103 - val_accuracy: 0.8950
    Epoch 10/10
    1800/1800 - 1s - loss: 4.3380e-04 - accuracy: 1.0000 - val_loss: 0.4224 - val_accuracy: 0.8900
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 3s - loss: 0.5461 - accuracy: 0.7122 - val_loss: 0.3417 - val_accuracy: 0.8650
    Epoch 2/10
    1800/1800 - 1s - loss: 0.1189 - accuracy: 0.9717 - val_loss: 0.3015 - val_accuracy: 0.8800
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0176 - accuracy: 1.0000 - val_loss: 0.2987 - val_accuracy: 0.9000
    Epoch 4/10
    1800/1800 - 1s - loss: 0.0058 - accuracy: 1.0000 - val_loss: 0.3220 - val_accuracy: 0.8900
    Epoch 5/10
    1800/1800 - 1s - loss: 0.0028 - accuracy: 1.0000 - val_loss: 0.3367 - val_accuracy: 0.8850
    Epoch 6/10
    1800/1800 - 1s - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.3539 - val_accuracy: 0.8850
    Epoch 7/10
    1800/1800 - 2s - loss: 9.9998e-04 - accuracy: 1.0000 - val_loss: 0.3734 - val_accuracy: 0.8800
    Epoch 8/10
    1800/1800 - 1s - loss: 6.7221e-04 - accuracy: 1.0000 - val_loss: 0.3869 - val_accuracy: 0.8800
    Epoch 9/10
    1800/1800 - 2s - loss: 4.8855e-04 - accuracy: 1.0000 - val_loss: 0.4028 - val_accuracy: 0.8800
    Epoch 10/10
    1800/1800 - 1s - loss: 3.6666e-04 - accuracy: 1.0000 - val_loss: 0.4150 - val_accuracy: 0.8800
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.5286 - accuracy: 0.7439 - val_loss: 0.3450 - val_accuracy: 0.8950
    Epoch 2/10
    1800/1800 - 1s - loss: 0.0951 - accuracy: 0.9839 - val_loss: 0.2953 - val_accuracy: 0.9000
    Epoch 3/10
    1800/1800 - 1s - loss: 0.0135 - accuracy: 1.0000 - val_loss: 0.3001 - val_accuracy: 0.9000
    Epoch 4/10
    1800/1800 - 1s - loss: 0.0052 - accuracy: 1.0000 - val_loss: 0.3116 - val_accuracy: 0.8900
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0030 - accuracy: 1.0000 - val_loss: 0.3254 - val_accuracy: 0.8900
    Epoch 6/10
    1800/1800 - 1s - loss: 0.0020 - accuracy: 1.0000 - val_loss: 0.3355 - val_accuracy: 0.8950
    Epoch 7/10
    1800/1800 - 2s - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.3476 - val_accuracy: 0.8950
    Epoch 8/10
    1800/1800 - 1s - loss: 9.9276e-04 - accuracy: 1.0000 - val_loss: 0.3574 - val_accuracy: 0.8950
    Epoch 9/10
    1800/1800 - 2s - loss: 7.5036e-04 - accuracy: 1.0000 - val_loss: 0.3683 - val_accuracy: 0.8950
    Epoch 10/10
    1800/1800 - 2s - loss: 5.7804e-04 - accuracy: 1.0000 - val_loss: 0.3786 - val_accuracy: 0.8950
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.5033 - accuracy: 0.7589 - val_loss: 0.3144 - val_accuracy: 0.8650
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0428 - accuracy: 0.9911 - val_loss: 0.3329 - val_accuracy: 0.8700
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0059 - accuracy: 0.9994 - val_loss: 0.3418 - val_accuracy: 0.8700
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0019 - accuracy: 1.0000 - val_loss: 0.3456 - val_accuracy: 0.8900
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.3517 - val_accuracy: 0.8900
    Epoch 6/10
    1800/1800 - 1s - loss: 7.4914e-04 - accuracy: 1.0000 - val_loss: 0.3591 - val_accuracy: 0.8850
    Epoch 7/10
    1800/1800 - 2s - loss: 5.4495e-04 - accuracy: 1.0000 - val_loss: 0.3661 - val_accuracy: 0.8850
    Epoch 8/10
    1800/1800 - 1s - loss: 4.1083e-04 - accuracy: 1.0000 - val_loss: 0.3724 - val_accuracy: 0.8800
    Epoch 9/10
    1800/1800 - 2s - loss: 3.2083e-04 - accuracy: 1.0000 - val_loss: 0.3770 - val_accuracy: 0.8800
    Epoch 10/10
    1800/1800 - 2s - loss: 2.5706e-04 - accuracy: 1.0000 - val_loss: 0.3816 - val_accuracy: 0.8800
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 3s - loss: 0.4973 - accuracy: 0.7622 - val_loss: 0.2766 - val_accuracy: 0.8900
    Epoch 2/10
    1800/1800 - 1s - loss: 0.0489 - accuracy: 0.9867 - val_loss: 0.2970 - val_accuracy: 0.8850
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0055 - accuracy: 1.0000 - val_loss: 0.3054 - val_accuracy: 0.8800
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0026 - accuracy: 1.0000 - val_loss: 0.3235 - val_accuracy: 0.8750
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.3338 - val_accuracy: 0.8800
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.3457 - val_accuracy: 0.8800
    Epoch 7/10
    1800/1800 - 2s - loss: 8.1879e-04 - accuracy: 1.0000 - val_loss: 0.3567 - val_accuracy: 0.8750
    Epoch 8/10
    1800/1800 - 2s - loss: 6.3182e-04 - accuracy: 1.0000 - val_loss: 0.3629 - val_accuracy: 0.8750
    Epoch 9/10
    1800/1800 - 2s - loss: 4.9654e-04 - accuracy: 1.0000 - val_loss: 0.3729 - val_accuracy: 0.8750
    Epoch 10/10
    1800/1800 - 2s - loss: 4.0160e-04 - accuracy: 1.0000 - val_loss: 0.3785 - val_accuracy: 0.8750
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.5130 - accuracy: 0.7328 - val_loss: 0.3778 - val_accuracy: 0.8250
    Epoch 2/10
    1800/1800 - 1s - loss: 0.0407 - accuracy: 0.9961 - val_loss: 0.3240 - val_accuracy: 0.8650
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0074 - accuracy: 0.9994 - val_loss: 0.3660 - val_accuracy: 0.8850
    Epoch 4/10
    1800/1800 - 1s - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.3583 - val_accuracy: 0.8950
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.3649 - val_accuracy: 0.9000
    Epoch 6/10
    1800/1800 - 1s - loss: 9.5181e-04 - accuracy: 1.0000 - val_loss: 0.3710 - val_accuracy: 0.9000
    Epoch 7/10
    1800/1800 - 2s - loss: 6.8905e-04 - accuracy: 1.0000 - val_loss: 0.3778 - val_accuracy: 0.9000
    Epoch 8/10
    1800/1800 - 1s - loss: 5.2441e-04 - accuracy: 1.0000 - val_loss: 0.3838 - val_accuracy: 0.9000
    Epoch 9/10
    1800/1800 - 1s - loss: 4.1292e-04 - accuracy: 1.0000 - val_loss: 0.3885 - val_accuracy: 0.9000
    Epoch 10/10
    1800/1800 - 1s - loss: 3.3348e-04 - accuracy: 1.0000 - val_loss: 0.3924 - val_accuracy: 0.8950
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 3s - loss: 0.4944 - accuracy: 0.7544 - val_loss: 0.2894 - val_accuracy: 0.9000
    Epoch 2/10
    1800/1800 - 2s - loss: 0.0484 - accuracy: 0.9906 - val_loss: 0.2641 - val_accuracy: 0.9000
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0061 - accuracy: 0.9994 - val_loss: 0.2858 - val_accuracy: 0.8750
    Epoch 4/10
    1800/1800 - 2s - loss: 0.0026 - accuracy: 1.0000 - val_loss: 0.2958 - val_accuracy: 0.8900
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.3066 - val_accuracy: 0.8900
    Epoch 6/10
    1800/1800 - 2s - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.3165 - val_accuracy: 0.8850
    Epoch 7/10
    1800/1800 - 1s - loss: 8.6909e-04 - accuracy: 1.0000 - val_loss: 0.3240 - val_accuracy: 0.8900
    Epoch 8/10
    1800/1800 - 1s - loss: 6.6919e-04 - accuracy: 1.0000 - val_loss: 0.3313 - val_accuracy: 0.8900
    Epoch 9/10
    1800/1800 - 1s - loss: 5.2427e-04 - accuracy: 1.0000 - val_loss: 0.3380 - val_accuracy: 0.8850
    Epoch 10/10
    1800/1800 - 1s - loss: 4.1540e-04 - accuracy: 1.0000 - val_loss: 0.3443 - val_accuracy: 0.8800
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 3s - loss: 0.4855 - accuracy: 0.7589 - val_loss: 0.3162 - val_accuracy: 0.8600
    Epoch 2/10
    1800/1800 - 1s - loss: 0.0381 - accuracy: 0.9928 - val_loss: 0.3159 - val_accuracy: 0.8700
    Epoch 3/10
    1800/1800 - 2s - loss: 0.0050 - accuracy: 1.0000 - val_loss: 0.3311 - val_accuracy: 0.8650
    Epoch 4/10
    1800/1800 - 1s - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.3432 - val_accuracy: 0.8600
    Epoch 5/10
    1800/1800 - 2s - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.3546 - val_accuracy: 0.8600
    Epoch 6/10
    1800/1800 - 1s - loss: 9.5566e-04 - accuracy: 1.0000 - val_loss: 0.3639 - val_accuracy: 0.8600
    Epoch 7/10
    1800/1800 - 2s - loss: 7.0126e-04 - accuracy: 1.0000 - val_loss: 0.3724 - val_accuracy: 0.8600
    Epoch 8/10
    1800/1800 - 1s - loss: 5.3934e-04 - accuracy: 1.0000 - val_loss: 0.3785 - val_accuracy: 0.8600
    Epoch 9/10
    1800/1800 - 1s - loss: 4.3145e-04 - accuracy: 1.0000 - val_loss: 0.3853 - val_accuracy: 0.8550
    Epoch 10/10
    1800/1800 - 1s - loss: 3.4513e-04 - accuracy: 1.0000 - val_loss: 0.3915 - val_accuracy: 0.8600
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.6927 - accuracy: 0.4994 - val_loss: 0.6911 - val_accuracy: 0.5000
    Epoch 2/10
    1800/1800 - 1s - loss: 0.6847 - accuracy: 0.5683 - val_loss: 0.6791 - val_accuracy: 0.7250
    Epoch 3/10
    1800/1800 - 1s - loss: 0.6535 - accuracy: 0.7500 - val_loss: 0.6482 - val_accuracy: 0.6300
    Epoch 4/10
    1800/1800 - 1s - loss: 0.5856 - accuracy: 0.8533 - val_loss: 0.5948 - val_accuracy: 0.7300
    Epoch 5/10
    1800/1800 - 1s - loss: 0.4885 - accuracy: 0.9144 - val_loss: 0.5306 - val_accuracy: 0.7950
    Epoch 6/10
    1800/1800 - 1s - loss: 0.3811 - accuracy: 0.9617 - val_loss: 0.4640 - val_accuracy: 0.8500
    Epoch 7/10
    1800/1800 - 1s - loss: 0.2839 - accuracy: 0.9778 - val_loss: 0.4080 - val_accuracy: 0.8900
    Epoch 8/10
    1800/1800 - 1s - loss: 0.2063 - accuracy: 0.9889 - val_loss: 0.3673 - val_accuracy: 0.9050
    Epoch 9/10
    1800/1800 - 1s - loss: 0.1486 - accuracy: 0.9950 - val_loss: 0.3459 - val_accuracy: 0.8950
    Epoch 10/10
    1800/1800 - 1s - loss: 0.1092 - accuracy: 0.9956 - val_loss: 0.3175 - val_accuracy: 0.9050
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 4s - loss: 0.6928 - accuracy: 0.4956 - val_loss: 0.6913 - val_accuracy: 0.6550
    Epoch 2/10
    1800/1800 - 1s - loss: 0.6855 - accuracy: 0.7094 - val_loss: 0.6805 - val_accuracy: 0.8400
    Epoch 3/10
    1800/1800 - 1s - loss: 0.6576 - accuracy: 0.8572 - val_loss: 0.6495 - val_accuracy: 0.8600
    Epoch 4/10
    1800/1800 - 1s - loss: 0.5877 - accuracy: 0.9283 - val_loss: 0.5894 - val_accuracy: 0.8300
    Epoch 5/10
    1800/1800 - 1s - loss: 0.4736 - accuracy: 0.9539 - val_loss: 0.5087 - val_accuracy: 0.8700
    Epoch 6/10
    1800/1800 - 1s - loss: 0.3448 - accuracy: 0.9683 - val_loss: 0.4315 - val_accuracy: 0.8900
    Epoch 7/10
    1800/1800 - 1s - loss: 0.2367 - accuracy: 0.9817 - val_loss: 0.3794 - val_accuracy: 0.8900
    Epoch 8/10
    1800/1800 - 1s - loss: 0.1618 - accuracy: 0.9883 - val_loss: 0.3420 - val_accuracy: 0.9100
    Epoch 9/10
    1800/1800 - 1s - loss: 0.1121 - accuracy: 0.9950 - val_loss: 0.3209 - val_accuracy: 0.9050
    Epoch 10/10
    1800/1800 - 1s - loss: 0.0789 - accuracy: 0.9967 - val_loss: 0.3052 - val_accuracy: 0.9000
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 3s - loss: 0.6926 - accuracy: 0.5000 - val_loss: 0.6906 - val_accuracy: 0.5000
    Epoch 2/10
    1800/1800 - 1s - loss: 0.6848 - accuracy: 0.5139 - val_loss: 0.6803 - val_accuracy: 0.5400
    Epoch 3/10
    1800/1800 - 1s - loss: 0.6566 - accuracy: 0.6322 - val_loss: 0.6520 - val_accuracy: 0.6000
    Epoch 4/10
    1800/1800 - 1s - loss: 0.5949 - accuracy: 0.7583 - val_loss: 0.6038 - val_accuracy: 0.6800
    Epoch 5/10
    1800/1800 - 1s - loss: 0.5006 - accuracy: 0.8972 - val_loss: 0.5461 - val_accuracy: 0.7700
    Epoch 6/10
    1800/1800 - 1s - loss: 0.3976 - accuracy: 0.9511 - val_loss: 0.4852 - val_accuracy: 0.8350
    Epoch 7/10
    1800/1800 - 1s - loss: 0.2963 - accuracy: 0.9817 - val_loss: 0.4203 - val_accuracy: 0.8750
    Epoch 8/10
    1800/1800 - 1s - loss: 0.2062 - accuracy: 0.9928 - val_loss: 0.3835 - val_accuracy: 0.8650
    Epoch 9/10
    1800/1800 - 1s - loss: 0.1387 - accuracy: 0.9972 - val_loss: 0.3451 - val_accuracy: 0.8800
    Epoch 10/10
    1800/1800 - 1s - loss: 0.0951 - accuracy: 0.9983 - val_loss: 0.3197 - val_accuracy: 0.9000
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 5s - loss: 0.6923 - accuracy: 0.5717 - val_loss: 0.6903 - val_accuracy: 0.7600
    Epoch 2/10
    1800/1800 - 3s - loss: 0.6825 - accuracy: 0.8072 - val_loss: 0.6770 - val_accuracy: 0.6450
    Epoch 3/10
    1800/1800 - 2s - loss: 0.6483 - accuracy: 0.8317 - val_loss: 0.6402 - val_accuracy: 0.8750
    Epoch 4/10
    1800/1800 - 1s - loss: 0.5690 - accuracy: 0.9372 - val_loss: 0.5769 - val_accuracy: 0.8700
    Epoch 5/10
    1800/1800 - 1s - loss: 0.4535 - accuracy: 0.9494 - val_loss: 0.4959 - val_accuracy: 0.8800
    Epoch 6/10
    1800/1800 - 1s - loss: 0.3307 - accuracy: 0.9672 - val_loss: 0.4285 - val_accuracy: 0.8600
    Epoch 7/10
    1800/1800 - 2s - loss: 0.2328 - accuracy: 0.9767 - val_loss: 0.3766 - val_accuracy: 0.9000
    Epoch 8/10
    1800/1800 - 2s - loss: 0.1634 - accuracy: 0.9889 - val_loss: 0.3468 - val_accuracy: 0.8850
    Epoch 9/10
    1800/1800 - 1s - loss: 0.1159 - accuracy: 0.9939 - val_loss: 0.3247 - val_accuracy: 0.8900
    Epoch 10/10
    1800/1800 - 2s - loss: 0.0839 - accuracy: 0.9967 - val_loss: 0.3063 - val_accuracy: 0.9050
    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 - 3s - loss: 0.6925 - accuracy: 0.5350 - val_loss: 0.6905 - val_accuracy: 0.8400
    Epoch 2/10
    1800/1800 - 1s - loss: 0.6842 - accuracy: 0.7383 - val_loss: 0.6802 - val_accuracy: 0.8800
    Epoch 3/10
    1800/1800 - 1s - loss: 0.6578 - accuracy: 0.8911 - val_loss: 0.6531 - val_accuracy: 0.8450
    Epoch 4/10
    1800/1800 - 1s - loss: 0.5994 - accuracy: 0.9178 - val_loss: 0.6031 - val_accuracy: 0.8750
    Epoch 5/10
    1800/1800 - 2s - loss: 0.5040 - accuracy: 0.9500 - val_loss: 0.5390 - val_accuracy: 0.8350
    Epoch 6/10
    1800/1800 - 2s - loss: 0.3940 - accuracy: 0.9583 - val_loss: 0.4636 - val_accuracy: 0.8900
    Epoch 7/10
    1800/1800 - 2s - loss: 0.2890 - accuracy: 0.9733 - val_loss: 0.4139 - val_accuracy: 0.8650
    Epoch 8/10
    1800/1800 - 2s - loss: 0.2167 - accuracy: 0.9789 - val_loss: 0.3689 - val_accuracy: 0.8950
    Epoch 9/10
    1800/1800 - 2s - loss: 0.1537 - accuracy: 0.9917 - val_loss: 0.3455 - val_accuracy: 0.8800
    Epoch 10/10
    1800/1800 - 2s - loss: 0.1140 - accuracy: 0.9961 - val_loss: 0.3204 - val_accuracy: 0.9100
                                          (mlp1, and, binary)  \
    count                                                   5   
    unique                                                  5   
    top     [0.9, 0.93, 0.92, 0.925, 0.925, 0.93, 0.93, 0....   
    freq                                                    1   

                                           (mlp1, and, count)  \
    count                                                   5   
    unique                                                  5   
    top     [0.885, 0.91, 0.9, 0.905, 0.895, 0.9, 0.9, 0.9...   
    freq                                                    1   

                                           (mlp1, and, tfidf)  \
    count                                                   5   
    unique                                                  5   
    top     [0.86, 0.88, 0.875, 0.88, 0.88, 0.87, 0.87, 0....   
    freq                                                    1   

                                            (mlp1, and, freq)  \
    count                                                   5   
    unique                                                  5   
    top     [0.755, 0.575, 0.69, 0.835, 0.795, 0.85, 0.885...   
    freq                                                    1   

                                          (mlp2, and, binary)  \
    count                                                   5   
    unique                                                  5   
    top     [0.895, 0.9, 0.92, 0.915, 0.915, 0.92, 0.92, 0...   
    freq                                                    1   

                                           (mlp2, and, count)  \
    count                                                   5   
    unique                                                  5   
    top     [0.9, 0.9, 0.9, 0.905, 0.905, 0.905, 0.9, 0.89...   
    freq                                                    1   

                                           (mlp2, and, tfidf)  \
    count                                                   5   
    unique                                                  5   
    top     [0.825, 0.865, 0.885, 0.895, 0.9, 0.9, 0.9, 0....   
    freq                                                    1   

                                            (mlp2, and, freq)  
    count                                                   5  
    unique                                                  5  
    top     [0.5, 0.54, 0.6, 0.68, 0.77, 0.835, 0.875, 0.8...  
    freq                                                    1  



![png](Multilayer%20Perceptron%20Movie%20Review%20Sentiment%20Analysis_files/Multilayer%20Perceptron%20Movie%20Review%20Sentiment%20Analysis_33_1.png)



```python
results.T.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(mlp1, and, binary)</th>
      <td>[0.915, 0.92, 0.91, 0.92, 0.92, 0.92, 0.92, 0....</td>
      <td>[0.9, 0.92, 0.935, 0.93, 0.92, 0.92, 0.925, 0....</td>
      <td>[0.9, 0.93, 0.92, 0.925, 0.925, 0.93, 0.93, 0....</td>
      <td>[0.905, 0.925, 0.94, 0.935, 0.93, 0.93, 0.93, ...</td>
      <td>[0.915, 0.91, 0.93, 0.93, 0.93, 0.92, 0.92, 0....</td>
    </tr>
    <tr>
      <th>(mlp1, and, count)</th>
      <td>[0.9, 0.89, 0.885, 0.89, 0.885, 0.89, 0.89, 0....</td>
      <td>[0.855, 0.89, 0.9, 0.9, 0.905, 0.905, 0.9, 0.9...</td>
      <td>[0.885, 0.91, 0.9, 0.905, 0.895, 0.9, 0.9, 0.9...</td>
      <td>[0.89, 0.895, 0.9, 0.9, 0.905, 0.91, 0.91, 0.9...</td>
      <td>[0.865, 0.9, 0.915, 0.905, 0.895, 0.895, 0.9, ...</td>
    </tr>
    <tr>
      <th>(mlp1, and, tfidf)</th>
      <td>[0.9, 0.9, 0.9, 0.9, 0.895, 0.895, 0.89, 0.89,...</td>
      <td>[0.89, 0.905, 0.905, 0.9, 0.895, 0.9, 0.895, 0...</td>
      <td>[0.86, 0.88, 0.875, 0.88, 0.88, 0.87, 0.87, 0....</td>
      <td>[0.895, 0.895, 0.9, 0.9, 0.895, 0.895, 0.895, ...</td>
      <td>[0.885, 0.885, 0.885, 0.89, 0.89, 0.885, 0.89,...</td>
    </tr>
    <tr>
      <th>(mlp1, and, freq)</th>
      <td>[0.715, 0.825, 0.85, 0.86, 0.87, 0.85, 0.87, 0...</td>
      <td>[0.55, 0.85, 0.825, 0.87, 0.82, 0.865, 0.845, ...</td>
      <td>[0.755, 0.575, 0.69, 0.835, 0.795, 0.85, 0.885...</td>
      <td>[0.525, 0.79, 0.655, 0.83, 0.825, 0.835, 0.85,...</td>
      <td>[0.62, 0.51, 0.85, 0.875, 0.77, 0.835, 0.87, 0...</td>
    </tr>
    <tr>
      <th>(mlp2, and, binary)</th>
      <td>[0.915, 0.925, 0.92, 0.925, 0.925, 0.93, 0.93,...</td>
      <td>[0.9, 0.915, 0.92, 0.925, 0.925, 0.93, 0.93, 0...</td>
      <td>[0.895, 0.9, 0.92, 0.915, 0.915, 0.92, 0.92, 0...</td>
      <td>[0.885, 0.915, 0.92, 0.91, 0.91, 0.91, 0.905, ...</td>
      <td>[0.89, 0.92, 0.915, 0.915, 0.915, 0.915, 0.915...</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_unraveled = pd.DataFrame()
for index, column in results.T.iterrows():
    results_unraveled[index] = [e for l in column for e in l]
results_unraveled.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>(mlp1, and, binary)</th>
      <th>(mlp1, and, count)</th>
      <th>(mlp1, and, tfidf)</th>
      <th>(mlp1, and, freq)</th>
      <th>(mlp2, and, binary)</th>
      <th>(mlp2, and, count)</th>
      <th>(mlp2, and, tfidf)</th>
      <th>(mlp2, and, freq)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.922500</td>
      <td>0.897400</td>
      <td>0.889300</td>
      <td>0.811600</td>
      <td>0.916100</td>
      <td>0.893300</td>
      <td>0.878900</td>
      <td>0.820100</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.008097</td>
      <td>0.010412</td>
      <td>0.010051</td>
      <td>0.099747</td>
      <td>0.011079</td>
      <td>0.013117</td>
      <td>0.015396</td>
      <td>0.111746</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.900000</td>
      <td>0.855000</td>
      <td>0.860000</td>
      <td>0.510000</td>
      <td>0.885000</td>
      <td>0.825000</td>
      <td>0.825000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.920000</td>
      <td>0.895000</td>
      <td>0.885000</td>
      <td>0.821250</td>
      <td>0.910000</td>
      <td>0.890000</td>
      <td>0.870000</td>
      <td>0.803750</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.922500</td>
      <td>0.900000</td>
      <td>0.890000</td>
      <td>0.850000</td>
      <td>0.917500</td>
      <td>0.895000</td>
      <td>0.880000</td>
      <td>0.870000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.930000</td>
      <td>0.905000</td>
      <td>0.895000</td>
      <td>0.870000</td>
      <td>0.925000</td>
      <td>0.900000</td>
      <td>0.890000</td>
      <td>0.890000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.940000</td>
      <td>0.915000</td>
      <td>0.905000</td>
      <td>0.890000</td>
      <td>0.930000</td>
      <td>0.905000</td>
      <td>0.900000</td>
      <td>0.910000</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_unraveled.boxplot()
plt.show()
```


![png](Multilayer%20Perceptron%20Movie%20Review%20Sentiment%20Analysis_files/Multilayer%20Perceptron%20Movie%20Review%20Sentiment%20Analysis_36_0.png)


## Test against two real reviews

The best model was the wider model, and the best encoding method was the binary encoding. Let's test it against two real reviews.


```python
model = gen_model('mlp1', trainX_dict['binary'].shape[1])
```


```python
# create tensorboard callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

H = model.fit(trainX_dict['binary'], trainY,  
                      epochs=10,
                      callbacks=[tb_callback],
                      verbose=0)
```


```python
def pos_or_neg(filename, vocab_set, model, tokenizer):
    test = []
    test.append(doc_to_line(filename, vocab_set))
    test = tokenizer.texts_to_matrix(test)
    p = model.predict(test)[0][0]
    if round(p) == 0:
        print('This was a negative review with probability:', round((1-p)*100,2),'%')
    elif round(p) == 1:
        print('This was a positive review with probability:', round((p)*100,2),'%')
```

The first test is a negative [review](https://www.sandiegoreader.com/movies/star-wars-the-rise-of-skywalker/#) of the new star wars movie, giving it 1/5 stars.


```python
pos_or_neg('negative_star_wars_review.txt', vocab_set, model, tokenizer)
```

    This was a negative review with probability: 81.43 %


The second test is a positive review of the new star wars moving giving it 3.5/4 stars.


```python
pos_or_neg('positive_star_wars_review.txt', vocab_set, model, tokenizer)
```

    This was a positive review with probability: 99.41 %


Pretty cool, it guesses right with reasonably high confidence on both reviews!
