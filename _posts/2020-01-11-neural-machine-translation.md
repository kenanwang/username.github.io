---
layout: post
title: Neural Machine Translation
categories: [Processed Data]
tags:
---
This project translates text from German to English. It uses LSTMs, it is trained and tested on a small corpus. 

Thanks to machinelearningmastery.com from the guide to this.

## Import Libraries


```python
import numpy as np
import tensorflow.keras as tk
import datetime
import re
import string
import pickle
from unicodedata import normalize
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, RepeatVector, TimeDistributed, Dense
```


```python
%load_ext tensorboard
```

## Data Engineering


```python
# load the data
def load_doc(path):
    file = open(path, encoding='utf-8')
    text = file.read()
    file.close()
    return text
```


```python
path = 'deu-eng/deu.txt'
data = load_doc(path)
```


```python
data[:10]
```




    'Hi.\tHallo!'




```python
def to_pairs(data):
    lines = data.strip().split('\n')
    pairs = [line.split('\t')[:2] for line in lines]
    return pairs
```


```python
pairs = to_pairs(data)
```


```python
pairs[:5]
```




    [['Hi.', 'Hallo!'],
     ['Hi.', 'Grüß Gott!'],
     ['Run!', 'Lauf!'],
     ['Wow!', 'Potzdonner!'],
     ['Wow!', 'Donnerwetter!']]




```python
# clean the pairs of unuseable text, lowercase and change to ASCII
def clean_pairs(pairs):
    cleaned=list()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    for pair in pairs:
        clean_pair = list()
        for phrase in pair:
            phrase = normalize('NFD', phrase).encode('ascii', 'ignore')
            phrase = phrase.decode('UTF-8')
            phrase = phrase.split()
            phrase = [word.lower() for word in phrase]
            phrase = [re_punc.sub('', word) for word in phrase]
            phrase = [re_print.sub('', word) for word in phrase]
            phrase = [word for word in phrase if word.isalpha()]
            clean_pair.append(' '.join(phrase))
        cleaned.append(clean_pair)
    return np.array(cleaned)
```


```python
cleaned_pairs = clean_pairs(pairs)
cleaned_pairs[:5]
```




    array([['hi', 'hallo'],
           ['hi', 'gru gott'],
           ['run', 'lauf'],
           ['wow', 'potzdonner'],
           ['wow', 'donnerwetter']], dtype='<U527')




```python
#save the data
pickle.dump(cleaned_pairs, open('cleaned_pairs.pkl', 'wb'))
```

### Transform data for training


```python
# take out a subsample of data for training and testing
n_pairs = 10000
reduced_data = cleaned_pairs[:n_pairs, :]
np.random.shuffle(reduced_data)
train, test = reduced_data[:9000], reduced_data[9000:]
```


```python
# create tokenizers
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
```


```python
# get the max length for use in defining model
def get_max_length(lines):
    return max(len(line.split()) for line in lines)
```


```python
eng_tokenizer = create_tokenizer(train[:,0])
eng_vocab_dim = len(eng_tokenizer.word_index) + 1
eng_length = get_max_length(train[:,0])
print('eng_vocab_dim:', eng_vocab_dim)
print('eng_length:', eng_length)
ger_tokenizer = create_tokenizer(train[:,1])
ger_vocab_dim = len(ger_tokenizer.word_index) + 1
ger_length = get_max_length(train[:,1])
print('ger_vocab_dim', ger_vocab_dim)
print('ger_max_length', ger_length)
```

    eng_vocab_dim: 2121
    eng_length: 5
    ger_vocab_dim 3381
    ger_max_length 9



```python
# change sequences to their tokenizer index, and pad the sequences
def encode_sequences(tokenizer, length, lines):
    out = tokenizer.texts_to_sequences(lines)
    out = pad_sequences(out, maxlen=length, padding='post')
    return out
```


```python
def encode_output(sequences, vocab_dim):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_dim)
        ylist.append(encoded)
    y = np.array(ylist)
    print('before', y.shape)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_dim)
    print('after', y.shape)
    return y
```


```python
# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:,1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:,0])
trainY = encode_output(trainY, eng_vocab_dim)
# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:,1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:,0])
testY = encode_output(testY, eng_vocab_dim)
```

    before (9000, 5, 2121)
    after (9000, 5, 2121)
    before (1000, 5, 2121)
    after (1000, 5, 2121)


## Build model


```python
def build_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, 128, input_length=src_timesteps, mask_zero=True))
    model.add(Bidirectional(LSTM(n_units)))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units*2, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model
```


```python
model = build_model(ger_vocab_dim, eng_vocab_dim, ger_length, eng_length, 128)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 9, 128)            432768    
    _________________________________________________________________
    bidirectional (Bidirectional (None, 256)               263168    
    _________________________________________________________________
    repeat_vector (RepeatVector) (None, 5, 256)            0         
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 5, 256)            525312    
    _________________________________________________________________
    time_distributed (TimeDistri (None, 5, 2121)           545097    
    =================================================================
    Total params: 1,766,345
    Trainable params: 1,766,345
    Non-trainable params: 0
    _________________________________________________________________


## Train model


```python
# set callbacks
checkpoint = tk.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(trainX, trainY,
         epochs=30,
         batch_size=64,
         validation_data=(testX, testY),
         callbacks=[checkpoint, tb_callback])
```

    Train on 9000 samples, validate on 1000 samples
    Epoch 1/30
    8960/9000 [============================>.] - ETA: 0s - loss: 4.3039
    Epoch 00001: val_loss improved from inf to 3.22506, saving model to model.h5
    9000/9000 [==============================] - 113s 13ms/sample - loss: 4.2997 - val_loss: 3.2251
    Epoch 2/30
    8960/9000 [============================>.] - ETA: 0s - loss: 3.2780
    Epoch 00002: val_loss improved from 3.22506 to 3.05955, saving model to model.h5
    9000/9000 [==============================] - 53s 6ms/sample - loss: 3.2776 - val_loss: 3.0595
    Epoch 3/30
    8960/9000 [============================>.] - ETA: 0s - loss: 3.1319
    Epoch 00003: val_loss improved from 3.05955 to 2.94987, saving model to model.h5
    9000/9000 [==============================] - 60s 7ms/sample - loss: 3.1312 - val_loss: 2.9499
    Epoch 4/30
    8960/9000 [============================>.] - ETA: 0s - loss: 2.9390
    Epoch 00004: val_loss improved from 2.94987 to 2.75444, saving model to model.h5
    9000/9000 [==============================] - 82s 9ms/sample - loss: 2.9386 - val_loss: 2.7544
    Epoch 5/30
    8960/9000 [============================>.] - ETA: 0s - loss: 2.6988
    Epoch 00005: val_loss improved from 2.75444 to 2.57748, saving model to model.h5
    9000/9000 [==============================] - 66s 7ms/sample - loss: 2.6984 - val_loss: 2.5775
    Epoch 6/30
    8960/9000 [============================>.] - ETA: 0s - loss: 2.4913
    Epoch 00006: val_loss improved from 2.57748 to 2.45836, saving model to model.h5
    9000/9000 [==============================] - 49s 5ms/sample - loss: 2.4918 - val_loss: 2.4584
    Epoch 7/30
    8960/9000 [============================>.] - ETA: 0s - loss: 2.3299
    Epoch 00007: val_loss improved from 2.45836 to 2.36843, saving model to model.h5
    9000/9000 [==============================] - 64s 7ms/sample - loss: 2.3297 - val_loss: 2.3684
    Epoch 8/30
    8960/9000 [============================>.] - ETA: 0s - loss: 2.1815
    Epoch 00008: val_loss improved from 2.36843 to 2.27782, saving model to model.h5
    9000/9000 [==============================] - 75s 8ms/sample - loss: 2.1818 - val_loss: 2.2778
    Epoch 9/30
    8960/9000 [============================>.] - ETA: 0s - loss: 2.0404
    Epoch 00009: val_loss improved from 2.27782 to 2.19430, saving model to model.h5
    9000/9000 [==============================] - 52s 6ms/sample - loss: 2.0407 - val_loss: 2.1943
    Epoch 10/30
    8960/9000 [============================>.] - ETA: 0s - loss: 1.9031
    Epoch 00010: val_loss improved from 2.19430 to 2.12312, saving model to model.h5
    9000/9000 [==============================] - 48s 5ms/sample - loss: 1.9032 - val_loss: 2.1231
    Epoch 11/30
    8960/9000 [============================>.] - ETA: 0s - loss: 1.7800
    Epoch 00011: val_loss improved from 2.12312 to 2.05453, saving model to model.h5
    9000/9000 [==============================] - 51s 6ms/sample - loss: 1.7799 - val_loss: 2.0545
    Epoch 12/30
    8960/9000 [============================>.] - ETA: 0s - loss: 1.6636
    Epoch 00012: val_loss improved from 2.05453 to 2.00121, saving model to model.h5
    9000/9000 [==============================] - 50s 6ms/sample - loss: 1.6639 - val_loss: 2.0012
    Epoch 13/30
    8960/9000 [============================>.] - ETA: 0s - loss: 1.5553
    Epoch 00013: val_loss improved from 2.00121 to 1.96518, saving model to model.h5
    9000/9000 [==============================] - 46s 5ms/sample - loss: 1.5555 - val_loss: 1.9652
    Epoch 14/30
    8960/9000 [============================>.] - ETA: 0s - loss: 1.4547
    Epoch 00014: val_loss improved from 1.96518 to 1.91886, saving model to model.h5
    9000/9000 [==============================] - 48s 5ms/sample - loss: 1.4543 - val_loss: 1.9189
    Epoch 15/30
    8960/9000 [============================>.] - ETA: 0s - loss: 1.3571
    Epoch 00015: val_loss improved from 1.91886 to 1.88586, saving model to model.h5
    9000/9000 [==============================] - 45s 5ms/sample - loss: 1.3566 - val_loss: 1.8859
    Epoch 16/30
    8960/9000 [============================>.] - ETA: 0s - loss: 1.2611
    Epoch 00016: val_loss improved from 1.88586 to 1.85778, saving model to model.h5
    9000/9000 [==============================] - 48s 5ms/sample - loss: 1.2609 - val_loss: 1.8578
    Epoch 17/30
    8960/9000 [============================>.] - ETA: 0s - loss: 1.1725
    Epoch 00017: val_loss improved from 1.85778 to 1.81554, saving model to model.h5
    9000/9000 [==============================] - 46s 5ms/sample - loss: 1.1722 - val_loss: 1.8155
    Epoch 18/30
    8960/9000 [============================>.] - ETA: 0s - loss: 1.0851
    Epoch 00018: val_loss improved from 1.81554 to 1.80638, saving model to model.h5
    9000/9000 [==============================] - 48s 5ms/sample - loss: 1.0853 - val_loss: 1.8064
    Epoch 19/30
    8960/9000 [============================>.] - ETA: 0s - loss: 1.0044
    Epoch 00019: val_loss improved from 1.80638 to 1.77076, saving model to model.h5
    9000/9000 [==============================] - 43s 5ms/sample - loss: 1.0049 - val_loss: 1.7708
    Epoch 20/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.9255
    Epoch 00020: val_loss improved from 1.77076 to 1.75050, saving model to model.h5
    9000/9000 [==============================] - 49s 5ms/sample - loss: 0.9262 - val_loss: 1.7505
    Epoch 21/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.8529
    Epoch 00021: val_loss improved from 1.75050 to 1.74760, saving model to model.h5
    9000/9000 [==============================] - 47s 5ms/sample - loss: 0.8530 - val_loss: 1.7476
    Epoch 22/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.7838
    Epoch 00022: val_loss improved from 1.74760 to 1.71741, saving model to model.h5
    9000/9000 [==============================] - 45s 5ms/sample - loss: 0.7841 - val_loss: 1.7174
    Epoch 23/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.7161
    Epoch 00023: val_loss improved from 1.71741 to 1.70361, saving model to model.h5
    9000/9000 [==============================] - 42s 5ms/sample - loss: 0.7164 - val_loss: 1.7036
    Epoch 24/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.6591
    Epoch 00024: val_loss improved from 1.70361 to 1.69800, saving model to model.h5
    9000/9000 [==============================] - 48s 5ms/sample - loss: 0.6592 - val_loss: 1.6980
    Epoch 25/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.6018
    Epoch 00025: val_loss improved from 1.69800 to 1.69750, saving model to model.h5
    9000/9000 [==============================] - 65s 7ms/sample - loss: 0.6016 - val_loss: 1.6975
    Epoch 26/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.5497
    Epoch 00026: val_loss did not improve from 1.69750
    9000/9000 [==============================] - 50s 6ms/sample - loss: 0.5499 - val_loss: 1.7041
    Epoch 27/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.5015
    Epoch 00027: val_loss improved from 1.69750 to 1.68882, saving model to model.h5
    9000/9000 [==============================] - 48s 5ms/sample - loss: 0.5019 - val_loss: 1.6888
    Epoch 28/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.4600
    Epoch 00028: val_loss did not improve from 1.68882
    9000/9000 [==============================] - 44s 5ms/sample - loss: 0.4599 - val_loss: 1.6930
    Epoch 29/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.4188
    Epoch 00029: val_loss did not improve from 1.68882
    9000/9000 [==============================] - 42s 5ms/sample - loss: 0.4189 - val_loss: 1.6950
    Epoch 30/30
    8960/9000 [============================>.] - ETA: 0s - loss: 0.3861
    Epoch 00030: val_loss did not improve from 1.68882
    9000/9000 [==============================] - 43s 5ms/sample - loss: 0.3862 - val_loss: 1.6908





    <tensorflow.python.keras.callbacks.History at 0x1a46d80cd0>



## Evaluate model


```python
model = tk.models.load_model('model.h5')
```


```python
# function to take a model the target tokenizer and a source phrase and predict a translation
def predict_sequence(model, tokenizer, source):
    ix_to_word = dict((i,w) for w,i in tokenizer.word_index.items())
    prediction = model.predict(source, verbose=0)[0]
    classification = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in classification:
        if i == 0:
            break
        target.append(ix_to_word[i])
    return ' '.join(target)
```


```python
# function to predict phrases against a corpus of phrases and calculate BLEU Scores for the translations against a reference
def evaluate_model(model, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        source = source.reshape((1,source.shape[0]))
        translation = predict_sequence(model, eng_tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        # for the first ten iteme print the source, the target text, and then the predicted text
        if i < 10:
            print(f'src={raw_src}, target={raw_target}, predicted={translation}')
        actual.append([raw_target.split()])
        predicted.append(translation.split())
    # calculate the BLEU Scores
    BLEU = {
        1 : corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)),
        2 : corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)),
        3 : corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0)),
        4 : corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    }
    return BLEU
```


```python
train_BLEU = evaluate_model(model, trainX, train)
print(f'Train BLEU: {train_BLEU}')
test_BLEU = evaluate_model(model, testX, test)
print(f'Test BLEU: {test_BLEU}')
```

    src=ich bin beschaftigt, target=im busy, predicted=im busy busy
    src=krahen sind schwarz, target=crows are black, predicted=crows are black
    src=gehst du, target=are you going, predicted=are you come
    src=ich muss es versuchen, target=i have to try, predicted=i can to
    src=ich werde leben, target=ill live, predicted=ill will
    src=ich furchte nichts, target=i fear nothing, predicted=i dont nothing
    src=jetzt fuhle ich es, target=i feel it now, predicted=i i it works
    src=lassen sie es tom tun, target=let tom do it, predicted=let tom do it
    src=geht es tom gut, target=is tom well, predicted=did tom ok
    src=tom mag wein, target=tom likes wine, predicted=tom likes wine
    Train BLEU: {1: 0.8524946644757209, 2: 0.7921264396531609, 3: 0.6736230757963649, 4: 0.38682243318774046}
    src=tom ist hellwach, target=toms alert, predicted=toms is
    src=tom ist weg, target=toms gone, predicted=tom is gone
    src=du bist ganz reizend, target=youre sweet, predicted=youre great
    src=er ist nicht versichert, target=hes uninsured, predicted=hes kidding
    src=ist tom hier, target=is tom here, predicted=is tom here
    src=ich habe tom geglaubt, target=i believed tom, predicted=i watched tom
    src=eintritt verboten, target=keep out, predicted=dont hurt
    src=das kann man in der pfeife rauchen, target=its useless, predicted=its hurt it
    src=es ist ein hinterhalt, target=its an ambush, predicted=its is
    src=ich werde schon zurechtkommen, target=ill be fine, predicted=ill manage
    Test BLEU: {1: 0.5157759775011558, 2: 0.3810208722968439, 3: 0.26187942227213623, 4: 0.12474391941112985}
