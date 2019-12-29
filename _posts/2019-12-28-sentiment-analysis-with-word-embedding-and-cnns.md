---
layout: post
title: Sentiment Analysis with Word Embeddings and CNNs
categories: [Processed Data]
tags: [Word2Vec, GloVe, CNN, Tensorflow, Keras, Testing Harness]
---

Use word embeddings to encode text and a single layer convolutional networks to perform classification.

The text corpus used is movie reviews. Stanford's preprepared GloVe database will be used to seed our word embeddings. I'll try three different models:  
1) one without pretrained embeddings  
2) one with pretrained embeddings and learning on top of it  
3) one with pretrained embeddings and no learning on top  

Win condition: >87% accuracy on test split (87% is the upper bound for SVM and other traditional ML techniques on this data, see: http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf

Attributions:
machinelearningmastery.com DL for NLP book  

polarity dataset v2.0 ( 3.0Mb) (includes README v2.0): 1000 positive and 1000 negative processed reviews. Introduced in Pang/Lee ACL 2004. Released June 2004.

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.



## Import Libraries


```python
from os import listdir
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow.keras as tk
%load_ext tensorboard

import nltk
from nltk.corpus import stopwords
from collections import Counter
```


```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report
```

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

Now use our vocabulary to process our data


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
processed = {}
processed['trainX'] = trainX
processed['trainY'] = trainY
processed['testX'] = testX
processed['testY'] = testY
```

Next the data will be transformed to be input into a word embedding matrix and deep learning algorithms.


```python
# initiate tokenizer and fit to the training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainX)
```


```python
# encode and pad the sequences
trainX = processed['trainX']
max_length = max([len(doc.split()) for doc in trainX])
trainX = tokenizer.texts_to_sequences(trainX)
trainX = pad_sequences(trainX, maxlen=max_length, padding='post')
```


```python
testX = processed['testX']
testX = tokenizer.texts_to_sequences(testX)
testX = pad_sequences(testX, maxlen=max_length, padding='post')
```


```python
trainY, testY = processed['trainY'], processed['testY']
trainY, testY = np.array(trainY), np.array(testY)
```


```python
transformed = {}
transformed['trainX'] = trainX
transformed['trainY'] = trainY
transformed['testX'] = testX
transformed['testY'] = testY
```


```python
vocab_size = len(tokenizer.word_index)+1
print(max_length, vocab_size)
```

    1289 23549


## Build Models


```python
# Conv1D using word embeddings with no pretrained embeddings
model_np = Sequential(name='no_pretrain')

model_np.add(
    Embedding(
        len(vocab)+1,
        100,
        input_length=max_length)
)
model_np.add(Conv1D(32, 8, activation='relu'))
model_np.add(MaxPooling1D())
model_np.add(Flatten())
model_np.add(Dense(10, activation='relu'))
model_np.add(Dense(1, activation='sigmoid'))

model_np.summary()
```

    Model: "no_pretrain"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 1289, 100)         3638900   
    _________________________________________________________________
    conv1d (Conv1D)              (None, 1282, 32)          25632     
    _________________________________________________________________
    max_pooling1d (MaxPooling1D) (None, 641, 32)           0         
    _________________________________________________________________
    flatten (Flatten)            (None, 20512)             0         
    _________________________________________________________________
    dense (Dense)                (None, 10)                205130    
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 3,869,673
    Trainable params: 3,869,673
    Non-trainable params: 0
    _________________________________________________________________


Import embeddings from GloVe database


```python
# create embedding matrix
vocab = set(tokenizer.word_index.keys())
embedding_matrix = np.zeros((len(vocab)+1, 100))
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    if word in vocab:
        index = tokenizer.word_index[word]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_matrix[index] = vector
f.close()
```


```python
# Conv1D using word embeddings with pretrained embeddings and learning
model_p_l = Sequential(name='pretrain_learnings')

model_p_l.add(
    Embedding(
        len(vocab)+1,
        100,
        input_length=max_length,
        weights=[embedding_matrix],
        trainable=True)
)
model_p_l.add(Conv1D(32, 8, activation='relu'))
model_p_l.add(MaxPooling1D())
model_p_l.add(Flatten())
model_p_l.add(Dense(10, activation='relu'))
model_p_l.add(Dense(1, activation='sigmoid'))

model_p_l.summary()
```

    Model: "pretrain_learnings"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 1289, 100)         2354900   
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 1282, 32)          25632     
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 641, 32)           0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 20512)             0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                205130    
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 2,585,673
    Trainable params: 2,585,673
    Non-trainable params: 0
    _________________________________________________________________



```python
model_p_nl = Sequential(name='pretrained_nolearnings')

model_p_nl.add(
    Embedding(
        len(vocab)+1,
        100,
        input_length=max_length,
        weights=[embedding_matrix],
        trainable=False)
)
model_p_nl.add(Conv1D(32, 8, activation='relu'))
model_p_nl.add(MaxPooling1D())
model_p_nl.add(Flatten())
model_p_nl.add(Dense(10, activation='relu'))
model_p_nl.add(Dense(1, activation='sigmoid'))

model_p_nl.summary()
```

    Model: "pretrained_nolearnings"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 1289, 100)         2354900   
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 1282, 32)          25632     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 641, 32)           0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 20512)             0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 10)                205130    
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 2,585,673
    Trainable params: 230,773
    Non-trainable params: 2,354,900
    _________________________________________________________________


## Build a Testing Harness


```python
# These the models we will test
model_dict = {}
model_dict['no_pretrained'] = model_np
model_dict['pretrained_learnings'] = model_p_l
model_dict['pretrained_nolearnings'] = model_p_nl
```


```python
# create Tensorboard callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# evaluate models for various
def evaluate_models(model_dict, trainX, trainY, testX, testY, parameters):
    scores = {}
    for model_name, model in model_dict.items():
        # compile the model with the loss and optimizer passed in
        (loss, optimizer, metrics) = parameters
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        # fit to the training data
        model.fit(trainX, trainY, epochs=10, callbacks=[tb_callback])
        # evaluate the model on the test set
        _, acc = model.evaluate(testX, testY, verbose=0)
        scores[model_name] = acc
    return scores
```


```python
parameters = ('binary_crossentropy', 'adam', ['accuracy'])
trainX = transformed['trainX']
trainY = transformed['trainY']
testX = transformed['testX']
testY = transformed['testY']
scores = evaluate_models(model_dict, trainX, trainY, testX, testY, parameters)
```

    Train on 1800 samples
    Epoch 1/10
    1800/1800 [==============================] - 31s 17ms/sample - loss: 0.6875 - accuracy: 0.5489
    Epoch 2/10
    1800/1800 [==============================] - 27s 15ms/sample - loss: 0.4567 - accuracy: 0.7983
    Epoch 3/10
    1800/1800 [==============================] - 30s 17ms/sample - loss: 0.0762 - accuracy: 0.9750
    Epoch 4/10
    1800/1800 [==============================] - 25s 14ms/sample - loss: 0.0084 - accuracy: 1.0000
    Epoch 5/10
    1800/1800 [==============================] - 25s 14ms/sample - loss: 0.0024 - accuracy: 1.0000
    Epoch 6/10
    1800/1800 [==============================] - 24s 14ms/sample - loss: 0.0014 - accuracy: 1.0000
    Epoch 7/10
    1800/1800 [==============================] - 24s 13ms/sample - loss: 9.8350e-04 - accuracy: 1.0000
    Epoch 8/10
    1800/1800 [==============================] - 25s 14ms/sample - loss: 7.6414e-04 - accuracy: 1.0000
    Epoch 9/10
    1800/1800 [==============================] - 24s 13ms/sample - loss: 6.2068e-04 - accuracy: 1.0000
    Epoch 10/10
    1800/1800 [==============================] - 26s 14ms/sample - loss: 5.1969e-04 - accuracy: 1.0000
    Train on 1800 samples
    Epoch 1/10
    1800/1800 [==============================] - 27s 15ms/sample - loss: 0.6903 - accuracy: 0.5572
    Epoch 2/10
    1800/1800 [==============================] - 23s 13ms/sample - loss: 0.6490 - accuracy: 0.6172
    Epoch 3/10
    1800/1800 [==============================] - 23s 13ms/sample - loss: 0.5591 - accuracy: 0.7511
    Epoch 4/10
    1800/1800 [==============================] - 22s 12ms/sample - loss: 0.4500 - accuracy: 0.8517
    Epoch 5/10
    1800/1800 [==============================] - 22s 12ms/sample - loss: 0.3776 - accuracy: 0.9267
    Epoch 6/10
    1800/1800 [==============================] - 23s 13ms/sample - loss: 0.3310 - accuracy: 0.9633
    Epoch 7/10
    1800/1800 [==============================] - 23s 13ms/sample - loss: 0.2972 - accuracy: 0.9789
    Epoch 8/10
    1800/1800 [==============================] - 24s 13ms/sample - loss: 0.2771 - accuracy: 0.9833
    Epoch 9/10
    1800/1800 [==============================] - 22s 12ms/sample - loss: 0.2597 - accuracy: 0.9872
    Epoch 10/10
    1800/1800 [==============================] - 24s 13ms/sample - loss: 0.2476 - accuracy: 0.9889
    Train on 1800 samples
    Epoch 1/10
    1800/1800 [==============================] - 17s 10ms/sample - loss: 0.7088 - accuracy: 0.4978
    Epoch 2/10
    1800/1800 [==============================] - 15s 8ms/sample - loss: 0.6771 - accuracy: 0.5494
    Epoch 3/10
    1800/1800 [==============================] - 15s 8ms/sample - loss: 0.6016 - accuracy: 0.6311
    Epoch 4/10
    1800/1800 [==============================] - 15s 8ms/sample - loss: 0.4461 - accuracy: 0.7856
    Epoch 5/10
    1800/1800 [==============================] - 17s 9ms/sample - loss: 0.2247 - accuracy: 0.9217
    Epoch 6/10
    1800/1800 [==============================] - 17s 9ms/sample - loss: 0.1055 - accuracy: 0.9822
    Epoch 7/10
    1800/1800 [==============================] - 15s 9ms/sample - loss: 0.0306 - accuracy: 0.9989
    Epoch 8/10
    1800/1800 [==============================] - 15s 8ms/sample - loss: 0.0148 - accuracy: 1.0000
    Epoch 9/10
    1800/1800 [==============================] - 15s 8ms/sample - loss: 0.0088 - accuracy: 1.0000
    Epoch 10/10
    1800/1800 [==============================] - 15s 8ms/sample - loss: 0.0064 - accuracy: 1.0000



```python
print(scores)
```

    {'no_pretrained': 0.86, 'pretrained_learnings': 0.845, 'pretrained_nolearnings': 0.75}


## Test against two real reviews

The best model was the model without using pretrained embeddings, let's test it against some real reviews. This barely fails my win condition, might do better if I widen the dense layer. The MLP from my last project with BOW beat this.

Note: I used the smallest pretrained GloVe dataset so this might be different if I used a larger one


```python
def pos_or_neg(filename, vocab_set, model, tokenizer):
    test = []
    test.append(doc_to_line(filename, vocab_set))
    test = tokenizer.texts_to_sequences(test)
    test = pad_sequences(test, max_length)
    p = model.predict(test)[0][0]
    if round(p) == 0:
        print('This was a negative review with probability:', round((1-p)*100,2),'%')
    elif round(p) == 1:
        print('This was a positive review with probability:', round((p)*100,2),'%')
```


```python
model = model_dict['no_pretrained']
```

The first test is a negative [review](https://www.sandiegoreader.com/movies/star-wars-the-rise-of-skywalker/#) of the new star wars movie, giving it 1/5 stars.


```python
pos_or_neg('negative_star_wars_review.txt', vocab_set, model, tokenizer)
```

    This was a positive review with probability: 54.22 %


The second test is a positive review of the new star wars moving giving it 3.5/4 stars.


```python
pos_or_neg('positive_star_wars_review.txt', vocab_set, model, tokenizer)
```

    This was a positive review with probability: 67.52 %


Poor performance, much worse than the MLP with BOW. I wonder how it does if I use the pretrained embeddings.


```python
model = model_dict['pretrained_learnings']
```

The first test is a negative [review](https://www.sandiegoreader.com/movies/star-wars-the-rise-of-skywalker/#) of the new star wars movie, giving it 1/5 stars.


```python
pos_or_neg('negative_star_wars_review.txt', vocab_set, model, tokenizer)
```

    This was a negative review with probability: 66.39 %


The second test is a positive review of the new star wars moving giving it 3.5/4 stars.


```python
pos_or_neg('positive_star_wars_review.txt', vocab_set, model, tokenizer)
```

    This was a negative review with probability: 70.46 %


Still very poor... I wonder what I need to do to improve this, maybe if I don't do the vocab filtering at the first step, it seems to say that the word embeddings do better with a little less filtering.

# Try 2, no vocab filtering

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
    return alpha_words
```


```python
def doc_to_line(filename):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    return ' '.join(tokens)

def process_docs_to_lines(directory):
    lines = list()
    for filename in listdir(directory):
        if filename.startswith('cv'):
            path = directory + '/' + filename
            line = doc_to_line(path)
            lines.append(line)
    return lines
```


```python
# processing data without filtering stop words, short words, or by occurences
neg_train = process_docs_to_lines(neg_train_dir)
pos_train = process_docs_to_lines(pos_train_dir)
neg_test = process_docs_to_lines(neg_test_dir)
pos_test = process_docs_to_lines(pos_test_dir)
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
processed = {}
processed['trainX'] = trainX
processed['trainY'] = trainY
processed['testX'] = testX
processed['testY'] = testY
```

Next the data will be transformed to be input into a word embedding matrix and deep learning algorithms.


```python
# initiate tokenizer and fit to the training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainX)
```


```python
# encode and pad the sequences
trainX = processed['trainX']
max_length = max([len(doc.split()) for doc in trainX])
trainX = tokenizer.texts_to_sequences(trainX)
trainX = pad_sequences(trainX, maxlen=max_length, padding='post')
```


```python
print(max_length)
```

    2331



```python
testX = processed['testX']
testX = tokenizer.texts_to_sequences(testX)
testX = pad_sequences(testX, maxlen=max_length, padding='post')
```


```python
trainY, testY = processed['trainY'], processed['testY']
trainY, testY = np.array(trainY), np.array(testY)
```


```python
transformed = {}
transformed['trainX'] = trainX
transformed['trainY'] = trainY
transformed['testX'] = testX
transformed['testY'] = testY
```


```python
vocab = set(tokenizer.word_index.keys())
vocab_size = len(vocab)+1
print(vocab_size)
```

    36550


## Build Models


```python
# Conv1D using word embeddings with no pretrained embeddings
model_np = Sequential(name='no_pretrain')

model_np.add(
    Embedding(
        vocab_size,
        100,
        input_length=max_length)
)
model_np.add(Conv1D(32, 8, activation='relu'))
model_np.add(MaxPooling1D())
model_np.add(Flatten())
model_np.add(Dense(10, activation='relu'))
model_np.add(Dense(1, activation='sigmoid'))

model_np.summary()
```

    Model: "no_pretrain"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_4 (Embedding)      (None, 2331, 100)         3655000   
    _________________________________________________________________
    conv1d_4 (Conv1D)            (None, 2324, 32)          25632     
    _________________________________________________________________
    max_pooling1d_4 (MaxPooling1 (None, 1162, 32)          0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 37184)             0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 10)                371850    
    _________________________________________________________________
    dense_9 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 4,052,493
    Trainable params: 4,052,493
    Non-trainable params: 0
    _________________________________________________________________


Import embeddings from GloVe database


```python
# create embedding matrix
embedding_matrix = np.zeros((vocab_size, 100))
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    if word in vocab:
        index = tokenizer.word_index[word]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_matrix[index] = vector
f.close()
```


```python
# Conv1D using word embeddings with pretrained embeddings and learning
model_p_l = Sequential(name='pretrain_learnings')

model_p_l.add(
    Embedding(
        vocab_size,
        100,
        input_length=max_length,
        weights=[embedding_matrix],
        trainable=True)
)
model_p_l.add(Conv1D(32, 8, activation='relu'))
model_p_l.add(MaxPooling1D())
model_p_l.add(Flatten())
model_p_l.add(Dense(10, activation='relu'))
model_p_l.add(Dense(1, activation='sigmoid'))

model_p_l.summary()
```

    Model: "pretrain_learnings"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_5 (Embedding)      (None, 2331, 100)         3655000   
    _________________________________________________________________
    conv1d_5 (Conv1D)            (None, 2324, 32)          25632     
    _________________________________________________________________
    max_pooling1d_5 (MaxPooling1 (None, 1162, 32)          0         
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 37184)             0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 10)                371850    
    _________________________________________________________________
    dense_11 (Dense)             (None, 1)                 11        
    =================================================================
    Total params: 4,052,493
    Trainable params: 4,052,493
    Non-trainable params: 0
    _________________________________________________________________



```python
model_p_nl = Sequential(name='pretrained_nolearnings')

model_p_nl.add(
    Embedding(
        vocab_size,
        100,
        input_length=max_length,
        weights=[embedding_matrix],
        trainable=False)
)
model_p_nl.add(Conv1D(32, 8, activation='relu'))
model_p_nl.add(MaxPooling1D())
model_p_nl.add(Flatten())
model_p_nl.add(Dense(10, activation='relu'))
model_p_nl.add(Dense(1, activation='sigmoid'))

model_p_nl.summary()
```

    Model: "pretrained_nolearnings"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_6 (Embedding)      (None, 2331, 100)         3655000   
    _________________________________________________________________
    conv1d_6 (Conv1D)            (None, 2324, 32)          25632     
    _________________________________________________________________
    max_pooling1d_6 (MaxPooling1 (None, 1162, 32)          0         
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 37184)             0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 10)                371850    
    _________________________________________________________________
    dense_13 (Dense)             (None, 1)                 11        
    =================================================================
    Total params: 4,052,493
    Trainable params: 397,493
    Non-trainable params: 3,655,000
    _________________________________________________________________


## Build a Testing Harness


```python
# These the models we will test
model_dict = {}
model_dict['no_pretrained'] = model_np
model_dict['pretrained_learnings'] = model_p_l
model_dict['pretrained_nolearnings'] = model_p_nl
```


```python
# create Tensorboard callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# evaluate models for various
def evaluate_models(model_dict, trainX, trainY, testX, testY, parameters):
    scores = {}
    for model_name, model in model_dict.items():
        # compile the model with the loss and optimizer passed in
        (loss, optimizer, metrics) = parameters
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        # fit to the training data
        model.fit(trainX, trainY, epochs=10, callbacks=[tb_callback])
        # evaluate the model on the test set
        _, acc = model.evaluate(testX, testY, verbose=0)
        scores[model_name] = acc
    return scores
```


```python
parameters = ('binary_crossentropy', 'adam', ['accuracy'])
trainX = transformed['trainX']
trainY = transformed['trainY']
testX = transformed['testX']
testY = transformed['testY']
scores = evaluate_models(model_dict, trainX, trainY, testX, testY, parameters)
```

    Train on 1800 samples
    Epoch 1/10
    1800/1800 [==============================] - 50s 28ms/sample - loss: 0.6898 - accuracy: 0.5211
    Epoch 2/10
    1800/1800 [==============================] - 40s 22ms/sample - loss: 0.6381 - accuracy: 0.6228
    Epoch 3/10
    1800/1800 [==============================] - 42s 24ms/sample - loss: 0.4934 - accuracy: 0.8306
    Epoch 4/10
    1800/1800 [==============================] - 38s 21ms/sample - loss: 0.3728 - accuracy: 0.9361
    Epoch 5/10
    1800/1800 [==============================] - 43s 24ms/sample - loss: 0.3127 - accuracy: 0.9794
    Epoch 6/10
    1800/1800 [==============================] - 63s 35ms/sample - loss: 0.2852 - accuracy: 0.9933
    Epoch 7/10
    1800/1800 [==============================] - 52s 29ms/sample - loss: 0.2680 - accuracy: 0.9961
    Epoch 8/10
    1800/1800 [==============================] - 45s 25ms/sample - loss: 0.2549 - accuracy: 0.9961
    Epoch 9/10
    1800/1800 [==============================] - 37s 21ms/sample - loss: 0.2434 - accuracy: 0.9961
    Epoch 10/10
    1800/1800 [==============================] - 35s 20ms/sample - loss: 0.2328 - accuracy: 0.9961
    Train on 1800 samples
    Epoch 1/10
    1800/1800 [==============================] - 38s 21ms/sample - loss: 0.7264 - accuracy: 0.5206
    Epoch 2/10
    1800/1800 [==============================] - 36s 20ms/sample - loss: 0.6716 - accuracy: 0.5606
    Epoch 3/10
    1800/1800 [==============================] - 36s 20ms/sample - loss: 0.6305 - accuracy: 0.6167
    Epoch 4/10
    1800/1800 [==============================] - 35s 20ms/sample - loss: 0.5691 - accuracy: 0.7161
    Epoch 5/10
    1800/1800 [==============================] - 36s 20ms/sample - loss: 0.5198 - accuracy: 0.7811
    Epoch 6/10
    1800/1800 [==============================] - 35s 20ms/sample - loss: 0.4161 - accuracy: 0.8817
    Epoch 7/10
    1800/1800 [==============================] - 42s 23ms/sample - loss: 0.3487 - accuracy: 0.9383
    Epoch 8/10
    1800/1800 [==============================] - 46s 26ms/sample - loss: 0.3202 - accuracy: 0.9539
    Epoch 9/10
    1800/1800 [==============================] - 56s 31ms/sample - loss: 0.2855 - accuracy: 0.9783
    Epoch 10/10
    1800/1800 [==============================] - 53s 29ms/sample - loss: 0.2682 - accuracy: 0.9822
    Train on 1800 samples
    Epoch 1/10
    1800/1800 [==============================] - 26s 15ms/sample - loss: 0.7000 - accuracy: 0.5411
    Epoch 2/10
    1800/1800 [==============================] - 26s 14ms/sample - loss: 0.5472 - accuracy: 0.7294
    Epoch 3/10
    1800/1800 [==============================] - 36s 20ms/sample - loss: 0.3517 - accuracy: 0.8711
    Epoch 4/10
    1800/1800 [==============================] - 30s 17ms/sample - loss: 0.1878 - accuracy: 0.9589
    Epoch 5/10
    1800/1800 [==============================] - 29s 16ms/sample - loss: 0.0765 - accuracy: 0.9933
    Epoch 6/10
    1800/1800 [==============================] - 29s 16ms/sample - loss: 0.0396 - accuracy: 0.9978
    Epoch 7/10
    1800/1800 [==============================] - 29s 16ms/sample - loss: 0.0141 - accuracy: 1.0000
    Epoch 8/10
    1800/1800 [==============================] - 25s 14ms/sample - loss: 0.0079 - accuracy: 1.0000
    Epoch 9/10
    1800/1800 [==============================] - 25s 14ms/sample - loss: 0.0049 - accuracy: 1.0000
    Epoch 10/10
    1800/1800 [==============================] - 25s 14ms/sample - loss: 0.0036 - accuracy: 1.0000



```python
print(scores)
```

    {'no_pretrained': 0.855, 'pretrained_learnings': 0.725, 'pretrained_nolearnings': 0.69}


## Test against two real reviews

The best model was again no pretrained data, but in general the model did worse with no vocab filtering.


```python
def pos_or_neg(filename, vocab_set, model, tokenizer):
    test = []
    test.append(doc_to_line(filename))
    test = tokenizer.texts_to_sequences(test)
    test = pad_sequences(test, max_length)
    p = model.predict(test)[0][0]
    if round(p) == 0:
        print('This was a negative review with probability:', round((1-p)*100,2),'%')
    elif round(p) == 1:
        print('This was a positive review with probability:', round((p)*100,2),'%')
```


```python
model = model_dict['no_pretrained']
```

The first test is a negative [review](https://www.sandiegoreader.com/movies/star-wars-the-rise-of-skywalker/#) of the new star wars movie, giving it 1/5 stars.


```python
pos_or_neg('negative_star_wars_review.txt', vocab_set, model, tokenizer)
```

    This was a positive review with probability: 82.19 %


The second test is a positive review of the new star wars moving giving it 3.5/4 stars.


```python
pos_or_neg('positive_star_wars_review.txt', vocab_set, model, tokenizer)
```

    This was a positive review with probability: 99.99 %


These do in fact perform worse. If I was to do some more model tuning, I might widen the network, and perhaps tokenize the words differently, and return to a more aggressive word cleaning.

## Model Tuning

Returning to using the filtered vectors, I'm going to try increasing the size of the dense layer from 10 outputs to 32.


```python
vocab = set(tokenizer.word_index.keys())
max_length = max([len(doc) for doc in trainX])
# Conv1D using word embeddings with no pretrained embeddings
model_np = Sequential(name='no_pretrain')

model_np.add(
    Embedding(
        len(vocab)+1,
        100,
        input_length=max_length)
)
model_np.add(Conv1D(32, 8, activation='relu'))
model_np.add(MaxPooling1D())
model_np.add(Flatten())
model_np.add(Dense(32, activation='relu'))
model_np.add(Dense(1, activation='sigmoid'))

model_np.summary()
```

    Model: "no_pretrain"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_10 (Embedding)     (None, 1289, 100)         2354900   
    _________________________________________________________________
    conv1d_10 (Conv1D)           (None, 1282, 32)          25632     
    _________________________________________________________________
    max_pooling1d_10 (MaxPooling (None, 641, 32)           0         
    _________________________________________________________________
    flatten_10 (Flatten)         (None, 20512)             0         
    _________________________________________________________________
    dense_20 (Dense)             (None, 32)                656416    
    _________________________________________________________________
    dense_21 (Dense)             (None, 1)                 33        
    =================================================================
    Total params: 3,036,981
    Trainable params: 3,036,981
    Non-trainable params: 0
    _________________________________________________________________



```python
# create Tensorboard callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# compile and fit the model
model_np.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_np.fit(trainX, trainY, epochs=20, validation_data=(testX, testY), callbacks=[tb_callback])
```

    Train on 1800 samples, validate on 200 samples
    Epoch 1/20
    1800/1800 [==============================] - 27s 15ms/sample - loss: 0.6888 - accuracy: 0.5433 - val_loss: 0.6894 - val_accuracy: 0.5350
    Epoch 2/20
    1800/1800 [==============================] - 23s 13ms/sample - loss: 0.4820 - accuracy: 0.8150 - val_loss: 0.4368 - val_accuracy: 0.8300
    Epoch 3/20
    1800/1800 [==============================] - 24s 13ms/sample - loss: 0.0552 - accuracy: 0.9933 - val_loss: 0.3682 - val_accuracy: 0.8450
    Epoch 4/20
    1800/1800 [==============================] - 26s 14ms/sample - loss: 0.0057 - accuracy: 1.0000 - val_loss: 0.4041 - val_accuracy: 0.8450
    Epoch 5/20
    1800/1800 [==============================] - 24s 13ms/sample - loss: 0.0022 - accuracy: 1.0000 - val_loss: 0.4093 - val_accuracy: 0.8500
    Epoch 6/20
    1800/1800 [==============================] - 24s 13ms/sample - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.4185 - val_accuracy: 0.8500
    Epoch 7/20
    1800/1800 [==============================] - 23s 13ms/sample - loss: 9.9759e-04 - accuracy: 1.0000 - val_loss: 0.4270 - val_accuracy: 0.8450
    Epoch 8/20
    1800/1800 [==============================] - 24s 13ms/sample - loss: 7.7045e-04 - accuracy: 1.0000 - val_loss: 0.4340 - val_accuracy: 0.8450
    Epoch 9/20
    1800/1800 [==============================] - 24s 13ms/sample - loss: 6.1603e-04 - accuracy: 1.0000 - val_loss: 0.4406 - val_accuracy: 0.8500
    Epoch 10/20
    1800/1800 [==============================] - 24s 13ms/sample - loss: 4.9724e-04 - accuracy: 1.0000 - val_loss: 0.4469 - val_accuracy: 0.8500
    Epoch 11/20
    1800/1800 [==============================] - 23s 13ms/sample - loss: 4.0586e-04 - accuracy: 1.0000 - val_loss: 0.4525 - val_accuracy: 0.8500
    Epoch 12/20
    1800/1800 [==============================] - 24s 13ms/sample - loss: 3.2924e-04 - accuracy: 1.0000 - val_loss: 0.4579 - val_accuracy: 0.8450
    Epoch 13/20
     416/1800 [=====>........................] - ETA: 20s - loss: 1.7017e-04 - accuracy: 1.0000


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-153-17666d361eb5> in <module>
          5 # compile and fit the model
          6 model_np.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ----> 7 model_np.fit(trainX, trainY, epochs=20, validation_data=(testX, testY), callbacks=[tb_callback])


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        726         max_queue_size=max_queue_size,
        727         workers=workers,
    --> 728         use_multiprocessing=use_multiprocessing)
        729
        730   def evaluate(self,


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in fit(self, model, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq, **kwargs)
        322                 mode=ModeKeys.TRAIN,
        323                 training_context=training_context,
    --> 324                 total_epochs=epochs)
        325             cbks.make_logs(model, epoch_logs, training_result, ModeKeys.TRAIN)
        326


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2.py in run_one_epoch(model, iterator, execution_function, dataset_size, batch_size, strategy, steps_per_epoch, num_samples, mode, training_context, total_epochs)
        121         step=step, mode=mode, size=current_batch_size) as batch_logs:
        122       try:
    --> 123         batch_outs = execution_function(iterator)
        124       except (StopIteration, errors.OutOfRangeError):
        125         # TODO(kaftan): File bug about tf function and errors.OutOfRangeError?


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2_utils.py in execution_function(input_fn)
         84     # `numpy` translates Tensors to values in Eager mode.
         85     return nest.map_structure(_non_none_constant_value,
    ---> 86                               distributed_function(input_fn))
         87
         88   return execution_function


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/def_function.py in __call__(self, *args, **kwds)
        455
        456     tracing_count = self._get_tracing_count()
    --> 457     result = self._call(*args, **kwds)
        458     if tracing_count == self._get_tracing_count():
        459       self._call_counter.called_without_tracing()


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/def_function.py in _call(self, *args, **kwds)
        485       # In this case we have created variables on the first call, so we run the
        486       # defunned version which is guaranteed to never create variables.
    --> 487       return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
        488     elif self._stateful_fn is not None:
        489       # Release the lock early so that multiple threads can perform the call


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in __call__(self, *args, **kwargs)
       1821     """Calls a graph function specialized to the inputs."""
       1822     graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
    -> 1823     return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access
       1824
       1825   @property


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in _filtered_call(self, args, kwargs)
       1139          if isinstance(t, (ops.Tensor,
       1140                            resource_variable_ops.BaseResourceVariable))),
    -> 1141         self.captured_inputs)
       1142
       1143   def _call_flat(self, args, captured_inputs, cancellation_manager=None):


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in _call_flat(self, args, captured_inputs, cancellation_manager)
       1222     if executing_eagerly:
       1223       flat_outputs = forward_function.call(
    -> 1224           ctx, args, cancellation_manager=cancellation_manager)
       1225     else:
       1226       gradient_name = self._delayed_rewrite_functions.register()


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/function.py in call(self, ctx, args, cancellation_manager)
        509               inputs=args,
        510               attrs=("executor_type", executor_type, "config_proto", config),
    --> 511               ctx=ctx)
        512         else:
        513           outputs = execute.execute_with_cancellation(


    /anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         59     tensors = pywrap_tensorflow.TFE_Py_Execute(ctx._handle, device_name,
         60                                                op_name, inputs, attrs,
    ---> 61                                                num_outputs)
         62   except core._NotOkStatusException as e:
         63     if name is not None:


    KeyboardInterrupt:


The model isn't getting better after 7 epochs... actually it's getting worse. What if we try more Conv1D filters


```python
# Conv1D using word embeddings with no pretrained embeddings
model_np = Sequential(name='no_pretrain')

model_np.add(
    Embedding(
        len(vocab)+1,
        100,
        input_length=max_length)
)
model_np.add(Conv1D(50, 8, activation='relu'))
model_np.add(MaxPooling1D())
model_np.add(Flatten())
model_np.add(Dense(20, activation='relu'))
model_np.add(Dense(1, activation='sigmoid'))

model_np.summary()
```

    Model: "no_pretrain"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_11 (Embedding)     (None, 1289, 100)         2354900   
    _________________________________________________________________
    conv1d_11 (Conv1D)           (None, 1282, 50)          40050     
    _________________________________________________________________
    max_pooling1d_11 (MaxPooling (None, 641, 50)           0         
    _________________________________________________________________
    flatten_11 (Flatten)         (None, 32050)             0         
    _________________________________________________________________
    dense_22 (Dense)             (None, 20)                641020    
    _________________________________________________________________
    dense_23 (Dense)             (None, 1)                 21        
    =================================================================
    Total params: 3,035,991
    Trainable params: 3,035,991
    Non-trainable params: 0
    _________________________________________________________________



```python
# create Tensorboard callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# compile and fit the model
model_np.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_np.fit(trainX, trainY, epochs=10, validation_data=(testX, testY), callbacks=[tb_callback])
```

    Train on 1800 samples, validate on 200 samples
    Epoch 1/10
    1800/1800 [==============================] - 41s 23ms/sample - loss: 0.6801 - accuracy: 0.5644 - val_loss: 0.6755 - val_accuracy: 0.5650
    Epoch 2/10
    1800/1800 [==============================] - 36s 20ms/sample - loss: 0.2828 - accuracy: 0.9522 - val_loss: 0.3838 - val_accuracy: 0.8400
    Epoch 3/10
    1800/1800 [==============================] - 36s 20ms/sample - loss: 0.0143 - accuracy: 0.9972 - val_loss: 0.3397 - val_accuracy: 0.8850
    Epoch 4/10
    1800/1800 [==============================] - 41s 23ms/sample - loss: 0.0027 - accuracy: 1.0000 - val_loss: 0.3611 - val_accuracy: 0.8750
    Epoch 5/10
    1800/1800 [==============================] - 40s 22ms/sample - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.3596 - val_accuracy: 0.8900
    Epoch 6/10
    1800/1800 [==============================] - 39s 22ms/sample - loss: 6.7707e-04 - accuracy: 1.0000 - val_loss: 0.3580 - val_accuracy: 0.8900
    Epoch 7/10
    1800/1800 [==============================] - 43s 24ms/sample - loss: 4.6366e-04 - accuracy: 1.0000 - val_loss: 0.3691 - val_accuracy: 0.8900
    Epoch 8/10
    1800/1800 [==============================] - 38s 21ms/sample - loss: 3.1652e-04 - accuracy: 1.0000 - val_loss: 0.3729 - val_accuracy: 0.8900
    Epoch 9/10
    1800/1800 [==============================] - 39s 21ms/sample - loss: 2.3980e-04 - accuracy: 1.0000 - val_loss: 0.3759 - val_accuracy: 0.8850
    Epoch 10/10
    1800/1800 [==============================] - 38s 21ms/sample - loss: 1.8581e-04 - accuracy: 1.0000 - val_loss: 0.3785 - val_accuracy: 0.8900





    <tensorflow.python.keras.callbacks.History at 0x1a4359c630>



## Test against two real reviews

This version with 50 filters and 20 nodes at the fully connected layer beats my win condition.


```python
def pos_or_neg(filename, vocab_set, model, tokenizer):
    test = []
    test.append(doc_to_line(filename, vocab_set))
    test = tokenizer.texts_to_sequences(test)
    test = pad_sequences(test, max_length)
    p = model.predict(test)[0][0]
    if round(p) == 0:
        print('This was a negative review with probability:', round((1-p)*100,2),'%')
    elif round(p) == 1:
        print('This was a positive review with probability:', round((p)*100,2),'%')
```

The first test is a negative [review](https://www.sandiegoreader.com/movies/star-wars-the-rise-of-skywalker/#) of the new star wars movie, giving it 1/5 stars.


```python
pos_or_neg('negative_star_wars_review.txt', vocab_set, model_np, tokenizer)
```

    This was a negative review with probability: 65.68 %


The second test is a positive review of the new star wars moving giving it 3.5/4 stars.


```python
pos_or_neg('positive_star_wars_review.txt', vocab_set, model_np, tokenizer)
```

    This was a positive review with probability: 98.65 %


Definitely the best of the CNNs with word embeddings both statistically and against this eye test.
