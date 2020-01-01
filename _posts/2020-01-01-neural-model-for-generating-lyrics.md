---
layout: post
title: Neural Model for Generating Beatles Lyrics
categories: [End to End Projects]
tags: [RNN, LSTM, Language Model]
---

[<img src="https://www.dw.com/image/39219505_303.jpg">]

This project uses LSTMs in Tensorflow Keras to build word based language models for the song lyrics by a chosen author (I'm going to try The Beatles).

The lyrics come from a [Kaggle dataset](https://www.kaggle.com/mousehead/songlyrics) of lyrics scraped from lyricsfreak.com.

The model will have two layers of LSTMs and we will try generating text after various levels of training.

## Import Libraries


```python
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
%matplotlib inline

import re
import string
import pickle
import datetime
import random
```


```python
import tensorflow.keras as tk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```

## Data Engineering


```python
df = pd.read_csv('songdata.csv')
df.head()
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
      <th>artist</th>
      <th>song</th>
      <th>link</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ABBA</td>
      <td>Ahe's My Kind Of Girl</td>
      <td>/a/abba/ahes+my+kind+of+girl_20598417.html</td>
      <td>Look at her face, it's a wonderful face  \nAnd...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ABBA</td>
      <td>Andante, Andante</td>
      <td>/a/abba/andante+andante_20002708.html</td>
      <td>Take it easy with me, please  \nTouch me gentl...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABBA</td>
      <td>As Good As New</td>
      <td>/a/abba/as+good+as+new_20003033.html</td>
      <td>I'll never know why I had to go  \nWhy I had t...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBA</td>
      <td>Bang</td>
      <td>/a/abba/bang_20598415.html</td>
      <td>Making somebody happy is a question of give an...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABBA</td>
      <td>Bang-A-Boomerang</td>
      <td>/a/abba/bang+a+boomerang_20002668.html</td>
      <td>Making somebody happy is a question of give an...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Build a function to extract the songs from a specific artist
def get_songs(artist, df):
    songs = df['text'][df['artist']==artist]
    return songs
```


```python
# Testing above function
print(len(get_songs('The Beatles', df)))
```

    178



```python
# I want to preserve the new lines, and these punctuation: .,?!() as they seem relevant to many lyrics
def reformat_song(song):
    song = song.replace('\n', ' \n ').replace('.','\.').replace(',', '\,').replace('?','\?').replace('!','\!').replace('(', '( ').replace(')', ' )')
    return song
```


```python
test_song = 'this, is a test\nfor sure! (yeah?)'
print(reformat_song(test_song))
```

    this\, is a test
     for sure\! ( yeah\? )



```python
def reformat_songs(series):
    reformatted = series.apply(reformat_song)
    return reformatted
```


```python
# these are the beatles songs which we will use for our example
beatles_songs = reformat_songs(get_songs('The Beatles', df))
print(beatles_songs.shape)
print(beatles_songs.head())
```

    (178,)
    1198    Well\, if your hands start a-clappin'   \n And...
    1199    Words are flowing out like   \n Endless rain i...
    1200    Whenever I want you around\, yeah   \n All I g...
    1201    I give her all my love   \n That's all I do   ...
    1202    You tell me that you've got everything you wan...
    Name: text, dtype: object


### Transform data for ML


```python
# we need to build a vocabulary and I want to include new lines, and this punctuation: .,?!-()
default_filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
to_filter = default_filters.replace('\n','').replace('.','').replace(',', '').replace('!','').replace('?','').replace('-','').replace('(','').replace(')','').replace("''", '')
tokenizer = Tokenizer(filters=to_filter)
tokenizer.fit_on_texts(beatles_songs)
vocab = set(tokenizer.word_index.keys())
vocab_size = len(vocab)
print('Vocab Size:', vocab_size)
vocab_dim = vocab_size+1 # our ML algorithms will need an additional index because no words will get mapped to 0
```

    Vocab Size: 2361



```python
print(list(vocab)[:20])
```

    ['log', 'maid', 'oww', 'portuguese', 'hanging', 'king', 'her', 'ye-ye-yeh', 'united', "orphan's", 'besame', 'would', 'man', 'seems', 'dreams', 'spoken', 'your', 'teaser', 'innocence', 'doors']



```python
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
```


```python
# producing sequences of max_length words with 1 output word
def gen_Xy(songs, tokenizer, max_length):
    sequences = []
    for song in songs:
        # encode words to integer values
        encoded = tokenizer.texts_to_sequences([song])[0]
        # generate sequences of length max_length + 1 to produce input and output values
        for i in range(max_length, len(encoded)-1):
            seq = encoded[i-max_length:i+1]
            sequences.append(seq)
    sequences = np.array(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]
    assert(X.shape[1]==max_length)
    y = to_categorical(y, num_classes=(len(tokenizer.word_index)+1))
    return X, y
```


```python
# Producing my X and y matrices
max_length = 5
X, y = gen_Xy(beatles_songs, tokenizer, 5)
print(X.shape, y.shape)
```

    (39241, 5) (39241, 2362)


## Build Model


```python
#Model with an embedding layer of 50 nodes, and two LSTM layers of 64 nodes
model = Sequential()
model.add(Embedding(vocab_dim, 32, input_length=max_length))
model.add(LSTM(64, dropout=.2, return_sequences=True))
model.add(LSTM(64, dropout=.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(vocab_dim, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
plot_model(model, to_file='model.png', show_shapes=True)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 5, 32)             75584     
    _________________________________________________________________
    lstm (LSTM)                  (None, 5, 64)             24832     
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 64)                33024     
    _________________________________________________________________
    dense (Dense)                (None, 64)                4160      
    _________________________________________________________________
    dense_1 (Dense)              (None, 2362)              153530    
    =================================================================
    Total params: 291,130
    Trainable params: 291,130
    Non-trainable params: 0
    _________________________________________________________________
    Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.



```python
model.save('model.h5')
```

## Train model and Test Text Generation


```python
# this generates text from seed text using the model
def gen_text(model, tokenizer, seed_text, max_length, n_words):
    ix_to_words = dict([(i, c) for c, i in tokenizer.word_index.items()])
    text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([text])[0]
        padded = pad_sequences([encoded], maxlen=max_length, truncating='pre')
        y_hat = model.predict_classes(padded, verbose=0)
        new_word = ix_to_words[int(y_hat)]
        text += ' ' + new_word
    return text
```


```python
# this is to generate random seed text for the above function
def gen_rand_seq(X, tokenizer):
    ix_to_words = dict([(i, c) for c, i in tokenizer.word_index.items()])
    random.seed=123
    index = random.randrange(len(X))
    seq = X[index]
    words = [ix_to_words[i] for i in seq]
    return ' '.join(words)
```


```python
# no training
seed_text = gen_rand_seq(X, tokenizer)
print(gen_text(model, tokenizer, seed_text, max_length, 10))
```

    bay-ee-a-by )
     yay , junior 'you religion biding biding outside outside mm-mm-mm-di-di-di nineteen contempt


Even with no training, the output is vaguely Beatlesesque because of the limited vocabulary.


```python
# First ten epochs of training

#TB callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# fit model for 10 epochs
model.fit(X, y, batch_size=64, epochs=10, callbacks=[tb_callback])
# save model
model.save('model.h5')

# generate some text
seed_text = gen_rand_seq(X, tokenizer)
print(gen_text(model, tokenizer, seed_text, max_length, 10))
```

    Train on 39241 samples
    Epoch 1/10
       64/39241 [..............................] - ETA: 1:21:03 - loss: 7.7672 - accuracy: 0.0000e+00WARNING:tensorflow:Method (on_train_batch_end) is slow compared to the batch update (0.628713). Check your callbacks.
    39241/39241 [==============================] - 15s 377us/sample - loss: 5.5989 - accuracy: 0.1476
    Epoch 2/10
    39241/39241 [==============================] - 7s 174us/sample - loss: 5.3189 - accuracy: 0.1481
    Epoch 3/10
    39241/39241 [==============================] - 7s 175us/sample - loss: 5.1326 - accuracy: 0.1493
    Epoch 4/10
    39241/39241 [==============================] - 7s 177us/sample - loss: 4.8481 - accuracy: 0.1678
    Epoch 5/10
    39241/39241 [==============================] - 7s 175us/sample - loss: 4.5845 - accuracy: 0.1860
    Epoch 6/10
    39241/39241 [==============================] - 7s 182us/sample - loss: 4.3995 - accuracy: 0.1980
    Epoch 7/10
    39241/39241 [==============================] - 7s 183us/sample - loss: 4.2553 - accuracy: 0.2049
    Epoch 8/10
    39241/39241 [==============================] - 7s 183us/sample - loss: 4.1270 - accuracy: 0.2142
    Epoch 9/10
    39241/39241 [==============================] - 8s 192us/sample - loss: 4.0105 - accuracy: 0.2225
    Epoch 10/10
    39241/39241 [==============================] - 7s 182us/sample - loss: 3.9039 - accuracy: 0.2321
    see how they smile like the way

     i got a way




After 10 epochs we're getting some new lines and some repetition (which is vaguely lyrical):
```
see how they smile like the way

i got a way
```


```python
# Next 20 epochs of training

#TB callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# fit model for 20 epochs
model.fit(X, y, batch_size=64, epochs=20, callbacks=[tb_callback])
# save model
model.save('model.h5')

# generate some text
seed_text = gen_rand_seq(X, tokenizer)
print(gen_text(model, tokenizer, seed_text, max_length, 10))
```

    Train on 39241 samples
    Epoch 1/20
    39241/39241 [==============================] - 7s 179us/sample - loss: 3.8045 - accuracy: 0.2403
    Epoch 2/20
    39241/39241 [==============================] - 7s 175us/sample - loss: 3.7076 - accuracy: 0.2507
    Epoch 3/20
    39241/39241 [==============================] - 7s 172us/sample - loss: 3.6149 - accuracy: 0.2578
    Epoch 4/20
    39241/39241 [==============================] - 7s 172us/sample - loss: 3.5314 - accuracy: 0.2655
    Epoch 5/20
    39241/39241 [==============================] - 7s 171us/sample - loss: 3.4431 - accuracy: 0.2741
    Epoch 6/20
    39241/39241 [==============================] - 7s 172us/sample - loss: 3.3703 - accuracy: 0.2819
    Epoch 7/20
    39241/39241 [==============================] - 7s 172us/sample - loss: 3.3001 - accuracy: 0.2907
    Epoch 8/20
    39241/39241 [==============================] - 7s 174us/sample - loss: 3.2283 - accuracy: 0.3009
    Epoch 9/20
    39241/39241 [==============================] - 7s 172us/sample - loss: 3.1655 - accuracy: 0.3085
    Epoch 10/20
    39241/39241 [==============================] - 7s 174us/sample - loss: 3.1054 - accuracy: 0.3183
    Epoch 11/20
    39241/39241 [==============================] - 7s 176us/sample - loss: 3.0529 - accuracy: 0.3223
    Epoch 12/20
    39241/39241 [==============================] - 7s 175us/sample - loss: 2.9913 - accuracy: 0.3325
    Epoch 13/20
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.9538 - accuracy: 0.3380
    Epoch 14/20
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.8975 - accuracy: 0.3462
    Epoch 15/20
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.8541 - accuracy: 0.3536
    Epoch 16/20
    39241/39241 [==============================] - 7s 178us/sample - loss: 2.8181 - accuracy: 0.3591
    Epoch 17/20
    39241/39241 [==============================] - 7s 174us/sample - loss: 2.7763 - accuracy: 0.3682
    Epoch 18/20
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.7470 - accuracy: 0.3701
    Epoch 19/20
    39241/39241 [==============================] - 7s 174us/sample - loss: 2.7053 - accuracy: 0.3798
    Epoch 20/20
    39241/39241 [==============================] - 7s 174us/sample - loss: 2.6744 - accuracy: 0.3829
    hide it in a hiding in the garden of the loved
     and maybe it


After another 20 epochs (30 total) we're getting more sensible sentences:
```
hide it in a hiding in the garden of the loved
 and maybe it
```


```python
# Next 30 epochs of training

#TB callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# fit model for 30 epochs
model.fit(X, y, batch_size=64, epochs=30, callbacks=[tb_callback])
# save model
model.save('model.h5')

# generate some text
seed_text = gen_rand_seq(X, tokenizer)
print(gen_text(model, tokenizer, seed_text, max_length, 10))
```

    Train on 39241 samples
    Epoch 1/30
    39241/39241 [==============================] - 7s 175us/sample - loss: 2.6475 - accuracy: 0.3876
    Epoch 2/30
    39241/39241 [==============================] - 7s 171us/sample - loss: 2.6113 - accuracy: 0.3945
    Epoch 3/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.5854 - accuracy: 0.3953
    Epoch 4/30
    39241/39241 [==============================] - 7s 186us/sample - loss: 2.5559 - accuracy: 0.4050
    Epoch 5/30
    39241/39241 [==============================] - 7s 185us/sample - loss: 2.5301 - accuracy: 0.4073
    Epoch 6/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.5016 - accuracy: 0.4117
    Epoch 7/30
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.4830 - accuracy: 0.4173
    Epoch 8/30
    39241/39241 [==============================] - 7s 174us/sample - loss: 2.4508 - accuracy: 0.4192
    Epoch 9/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.4368 - accuracy: 0.4216
    Epoch 10/30
    39241/39241 [==============================] - 7s 177us/sample - loss: 2.4106 - accuracy: 0.4284
    Epoch 11/30
    39241/39241 [==============================] - 7s 175us/sample - loss: 2.3877 - accuracy: 0.4325
    Epoch 12/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.3699 - accuracy: 0.4355
    Epoch 13/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.3463 - accuracy: 0.4403
    Epoch 14/30
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.3246 - accuracy: 0.4429
    Epoch 15/30
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.3104 - accuracy: 0.4477
    Epoch 16/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.2950 - accuracy: 0.4496
    Epoch 17/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.2741 - accuracy: 0.4528
    Epoch 18/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.2581 - accuracy: 0.4556
    Epoch 19/30
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.2377 - accuracy: 0.4607
    Epoch 20/30
    39241/39241 [==============================] - 7s 174us/sample - loss: 2.2183 - accuracy: 0.4651
    Epoch 21/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.1985 - accuracy: 0.4709
    Epoch 22/30
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.1920 - accuracy: 0.4685
    Epoch 23/30
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.1690 - accuracy: 0.4704
    Epoch 24/30
    39241/39241 [==============================] - 7s 174us/sample - loss: 2.1547 - accuracy: 0.4769
    Epoch 25/30
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.1432 - accuracy: 0.4789
    Epoch 26/30
    39241/39241 [==============================] - 7s 174us/sample - loss: 2.1308 - accuracy: 0.4803
    Epoch 27/30
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.1217 - accuracy: 0.4841
    Epoch 28/30
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.1098 - accuracy: 0.4844
    Epoch 29/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.0981 - accuracy: 0.4875
    Epoch 30/30
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.0726 - accuracy: 0.4930
    you're missing
     nowhere man , yeah )

     well , i talk about


After another 30 epochs of training (60 total) we seem to be getting some rhythm:
```
you're missing
 nowhere man , yeah )

 well , i talk about
```


```python
# Next 40 epochs of training

#TB callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# fit model for 40 epochs
model.fit(X, y, batch_size=64, epochs=40, callbacks=[tb_callback])
# save model
model.save('model.h5')

# generate some text
seed_text = gen_rand_seq(X, tokenizer)
print(gen_text(model, tokenizer, seed_text, max_length, 10))
```

    Train on 39241 samples
    Epoch 1/40
    39241/39241 [==============================] - 7s 185us/sample - loss: 2.0748 - accuracy: 0.4934
    Epoch 2/40
    39241/39241 [==============================] - 7s 174us/sample - loss: 2.0546 - accuracy: 0.4950
    Epoch 3/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.0491 - accuracy: 0.4963
    Epoch 4/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 2.0386 - accuracy: 0.4967
    Epoch 5/40
    39241/39241 [==============================] - 7s 176us/sample - loss: 2.0329 - accuracy: 0.5014
    Epoch 6/40
    39241/39241 [==============================] - 7s 174us/sample - loss: 2.0229 - accuracy: 0.5022
    Epoch 7/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 2.0038 - accuracy: 0.5087
    Epoch 8/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.9875 - accuracy: 0.5083
    Epoch 9/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.9815 - accuracy: 0.5128
    Epoch 10/40
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.9702 - accuracy: 0.5150
    Epoch 11/40
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.9677 - accuracy: 0.5144
    Epoch 12/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.9603 - accuracy: 0.5149
    Epoch 13/40
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.9419 - accuracy: 0.5198
    Epoch 14/40
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.9343 - accuracy: 0.5230
    Epoch 15/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.9187 - accuracy: 0.5210
    Epoch 16/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.9143 - accuracy: 0.5219
    Epoch 17/40
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.9150 - accuracy: 0.5218
    Epoch 18/40
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.8998 - accuracy: 0.5275
    Epoch 19/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.8887 - accuracy: 0.5300
    Epoch 20/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.8825 - accuracy: 0.5294
    Epoch 21/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.8745 - accuracy: 0.5338
    Epoch 22/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.8629 - accuracy: 0.5350
    Epoch 23/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.8624 - accuracy: 0.5340
    Epoch 24/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.8469 - accuracy: 0.5406
    Epoch 25/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.8495 - accuracy: 0.5355
    Epoch 26/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.8369 - accuracy: 0.5395
    Epoch 27/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.8249 - accuracy: 0.5437
    Epoch 28/40
    39241/39241 [==============================] - 7s 180us/sample - loss: 1.8235 - accuracy: 0.5437
    Epoch 29/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.8192 - accuracy: 0.5445
    Epoch 30/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.8112 - accuracy: 0.5432
    Epoch 31/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.7926 - accuracy: 0.5506
    Epoch 32/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.7910 - accuracy: 0.5501
    Epoch 33/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.7898 - accuracy: 0.5522
    Epoch 34/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.7846 - accuracy: 0.5512
    Epoch 35/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.7711 - accuracy: 0.5526
    Epoch 36/40
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.7696 - accuracy: 0.5523
    Epoch 37/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.7597 - accuracy: 0.5548
    Epoch 38/40
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.7567 - accuracy: 0.5557
    Epoch 39/40
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.7470 - accuracy: 0.5573
    Epoch 40/40
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.7465 - accuracy: 0.5568
    ,
     well , i talk about boys ,
     don't ya know i mean


After another 40 epochs of training (100 total) we've got more consistent length of lines and the lines seem to relate:
```
,
 well , i talk about boys ,
 don't ya know i mean
 ```


```python
# Next 100 epochs of training

#TB callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# fit model for 100 epochs
model.fit(X, y, batch_size=64, epochs=100, callbacks=[tb_callback])
# save model
model.save('model.h5')

# generate some text
seed_text = gen_rand_seq(X, tokenizer)
print(gen_text(model, tokenizer, seed_text, max_length, 10))
```

    Train on 39241 samples
    Epoch 1/100
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.7300 - accuracy: 0.5601
    Epoch 2/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.7352 - accuracy: 0.5601
    Epoch 3/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.7241 - accuracy: 0.5656
    Epoch 4/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.7218 - accuracy: 0.5625
    Epoch 5/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.7185 - accuracy: 0.5650
    Epoch 6/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.7153 - accuracy: 0.5663
    Epoch 7/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.7007 - accuracy: 0.5693
    Epoch 8/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.7009 - accuracy: 0.5661
    Epoch 9/100
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.6961 - accuracy: 0.5667
    Epoch 10/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.6951 - accuracy: 0.5696
    Epoch 11/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6828 - accuracy: 0.5707
    Epoch 12/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6821 - accuracy: 0.5714
    Epoch 13/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.6720 - accuracy: 0.5730
    Epoch 14/100
    39241/39241 [==============================] - 7s 181us/sample - loss: 1.6776 - accuracy: 0.5759
    Epoch 15/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6722 - accuracy: 0.5730
    Epoch 16/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.6621 - accuracy: 0.5743
    Epoch 17/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6616 - accuracy: 0.5747
    Epoch 18/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6494 - accuracy: 0.5780
    Epoch 19/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.6574 - accuracy: 0.5760
    Epoch 20/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6493 - accuracy: 0.5769
    Epoch 21/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6429 - accuracy: 0.5770
    Epoch 22/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6441 - accuracy: 0.5802
    Epoch 23/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6423 - accuracy: 0.5806
    Epoch 24/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6297 - accuracy: 0.5821
    Epoch 25/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6212 - accuracy: 0.5866
    Epoch 26/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.6184 - accuracy: 0.5826
    Epoch 27/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6101 - accuracy: 0.5845
    Epoch 28/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.6135 - accuracy: 0.5833
    Epoch 29/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.6020 - accuracy: 0.5889
    Epoch 30/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6023 - accuracy: 0.5886
    Epoch 31/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.5982 - accuracy: 0.5904
    Epoch 32/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.6004 - accuracy: 0.5874
    Epoch 33/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.5879 - accuracy: 0.5911
    Epoch 34/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.5852 - accuracy: 0.5896
    Epoch 35/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.5789 - accuracy: 0.5911
    Epoch 36/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.5780 - accuracy: 0.5911
    Epoch 37/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.5773 - accuracy: 0.5923
    Epoch 38/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.5715 - accuracy: 0.5936
    Epoch 39/100
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.5666 - accuracy: 0.5957
    Epoch 40/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.5732 - accuracy: 0.5921
    Epoch 41/100
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.5677 - accuracy: 0.5950
    Epoch 42/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.5577 - accuracy: 0.5952
    Epoch 43/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.5564 - accuracy: 0.5965
    Epoch 44/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.5527 - accuracy: 0.5970
    Epoch 45/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.5548 - accuracy: 0.5972
    Epoch 46/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.5450 - accuracy: 0.6009
    Epoch 47/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.5404 - accuracy: 0.5953
    Epoch 48/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.5398 - accuracy: 0.6012
    Epoch 49/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.5431 - accuracy: 0.5986
    Epoch 50/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.5355 - accuracy: 0.6013
    Epoch 51/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.5343 - accuracy: 0.5988
    Epoch 52/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.5291 - accuracy: 0.6012
    Epoch 53/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.5224 - accuracy: 0.6030
    Epoch 54/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.5168 - accuracy: 0.6062
    Epoch 55/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.5158 - accuracy: 0.6038
    Epoch 56/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.5204 - accuracy: 0.6058
    Epoch 57/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.5095 - accuracy: 0.6085
    Epoch 58/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.5110 - accuracy: 0.6067
    Epoch 59/100
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.5134 - accuracy: 0.6056
    Epoch 60/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4980 - accuracy: 0.6103
    Epoch 61/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4981 - accuracy: 0.6079
    Epoch 62/100
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.4927 - accuracy: 0.6071
    Epoch 63/100
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.4994 - accuracy: 0.6098
    Epoch 64/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.4876 - accuracy: 0.6118
    Epoch 65/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4952 - accuracy: 0.6122
    Epoch 66/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4931 - accuracy: 0.6089
    Epoch 67/100
    39241/39241 [==============================] - 7s 184us/sample - loss: 1.4934 - accuracy: 0.6117
    Epoch 68/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4885 - accuracy: 0.6132
    Epoch 69/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4876 - accuracy: 0.6093
    Epoch 70/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4775 - accuracy: 0.6131
    Epoch 71/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.4797 - accuracy: 0.6115
    Epoch 72/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4702 - accuracy: 0.6149
    Epoch 73/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4741 - accuracy: 0.6151
    Epoch 74/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.4647 - accuracy: 0.6167
    Epoch 75/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4676 - accuracy: 0.6174
    Epoch 76/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4594 - accuracy: 0.6172
    Epoch 77/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.4638 - accuracy: 0.6191
    Epoch 78/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.4567 - accuracy: 0.6185
    Epoch 79/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4639 - accuracy: 0.6197
    Epoch 80/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4595 - accuracy: 0.6146
    Epoch 81/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4488 - accuracy: 0.6194
    Epoch 82/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4460 - accuracy: 0.6198
    Epoch 83/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4596 - accuracy: 0.6142
    Epoch 84/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4464 - accuracy: 0.6205
    Epoch 85/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4488 - accuracy: 0.6221
    Epoch 86/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4379 - accuracy: 0.6210
    Epoch 87/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4392 - accuracy: 0.6204
    Epoch 88/100
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.4367 - accuracy: 0.6222
    Epoch 89/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4298 - accuracy: 0.6218
    Epoch 90/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.4382 - accuracy: 0.6219
    Epoch 91/100
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.4276 - accuracy: 0.6233
    Epoch 92/100
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.4226 - accuracy: 0.6267
    Epoch 93/100
    39241/39241 [==============================] - 7s 181us/sample - loss: 1.4345 - accuracy: 0.6221
    Epoch 94/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.4221 - accuracy: 0.6235
    Epoch 95/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4211 - accuracy: 0.6271
    Epoch 96/100
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4210 - accuracy: 0.6244
    Epoch 97/100
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.4089 - accuracy: 0.6257
    Epoch 98/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.4053 - accuracy: 0.6284
    Epoch 99/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.4055 - accuracy: 0.6292
    Epoch 100/100
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.4094 - accuracy: 0.6271
    nations ,
     congratulations .

     all we are saying is give peace a


After another 100 epochs of training (200 total):

```
nations ,
 congratulations .

 all we are saying is give peace a
```


```python
# Next 200 epochs of training

#TB callback
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tb_callback = tk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# fit model for 200 epochs
model.fit(X, y, batch_size=64, epochs=200, callbacks=[tb_callback])
# save model
model.save('model.h5')

# generate some text
seed_text = gen_rand_seq(X, tokenizer)
print(gen_text(model, tokenizer, seed_text, max_length, 10))
```

    Train on 39241 samples
    Epoch 1/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.4081 - accuracy: 0.6275
    Epoch 2/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.4068 - accuracy: 0.6284
    Epoch 3/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4098 - accuracy: 0.6292
    Epoch 4/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4131 - accuracy: 0.6278
    Epoch 5/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.4063 - accuracy: 0.6287
    Epoch 6/200
    39241/39241 [==============================] - 7s 184us/sample - loss: 1.4016 - accuracy: 0.6285
    Epoch 7/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3951 - accuracy: 0.6298
    Epoch 8/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.3970 - accuracy: 0.6344
    Epoch 9/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.3917 - accuracy: 0.6335
    Epoch 10/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.3938 - accuracy: 0.6313
    Epoch 11/200
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.3902 - accuracy: 0.6306
    Epoch 12/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.3880 - accuracy: 0.6341
    Epoch 13/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.3869 - accuracy: 0.6354
    Epoch 14/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3763 - accuracy: 0.6346
    Epoch 15/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3848 - accuracy: 0.6324
    Epoch 16/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3869 - accuracy: 0.6343
    Epoch 17/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3827 - accuracy: 0.6312
    Epoch 18/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.3794 - accuracy: 0.6340
    Epoch 19/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3789 - accuracy: 0.6365
    Epoch 20/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.3703 - accuracy: 0.6358
    Epoch 21/200
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.3794 - accuracy: 0.6360
    Epoch 22/200
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.3833 - accuracy: 0.6303
    Epoch 23/200
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.3730 - accuracy: 0.6348
    Epoch 24/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3669 - accuracy: 0.6365
    Epoch 25/200
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.3695 - accuracy: 0.6375
    Epoch 26/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3622 - accuracy: 0.6391
    Epoch 27/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.3735 - accuracy: 0.6376
    Epoch 28/200
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.3537 - accuracy: 0.6390
    Epoch 29/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3689 - accuracy: 0.6364
    Epoch 30/200
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.3546 - accuracy: 0.6395
    Epoch 31/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3551 - accuracy: 0.6399
    Epoch 32/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.3587 - accuracy: 0.6406
    Epoch 33/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.3482 - accuracy: 0.6404
    Epoch 34/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.3485 - accuracy: 0.6425
    Epoch 35/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.3511 - accuracy: 0.6427
    Epoch 36/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.3468 - accuracy: 0.6394
    Epoch 37/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.3507 - accuracy: 0.6386
    Epoch 38/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.3464 - accuracy: 0.6434
    Epoch 39/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.3524 - accuracy: 0.6404
    Epoch 40/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.3334 - accuracy: 0.6431
    Epoch 41/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.3425 - accuracy: 0.6431
    Epoch 42/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.3532 - accuracy: 0.6406
    Epoch 43/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.3341 - accuracy: 0.6417
    Epoch 44/200
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.3409 - accuracy: 0.6438
    Epoch 45/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.3298 - accuracy: 0.6449
    Epoch 46/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.3333 - accuracy: 0.6437
    Epoch 47/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.3330 - accuracy: 0.6446
    Epoch 48/200
    39241/39241 [==============================] - 7s 185us/sample - loss: 1.3242 - accuracy: 0.6464
    Epoch 49/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.3358 - accuracy: 0.6442
    Epoch 50/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.3198 - accuracy: 0.6484
    Epoch 51/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.3230 - accuracy: 0.6470
    Epoch 52/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.3248 - accuracy: 0.6472
    Epoch 53/200
    39241/39241 [==============================] - 7s 181us/sample - loss: 1.3207 - accuracy: 0.6467
    Epoch 54/200
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.3160 - accuracy: 0.6497
    Epoch 55/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.3221 - accuracy: 0.6446
    Epoch 56/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.3186 - accuracy: 0.6460
    Epoch 57/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.3153 - accuracy: 0.6456
    Epoch 58/200
    39241/39241 [==============================] - 7s 185us/sample - loss: 1.3210 - accuracy: 0.6449
    Epoch 59/200
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.3117 - accuracy: 0.6523
    Epoch 60/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.3178 - accuracy: 0.6477
    Epoch 61/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.3193 - accuracy: 0.6458
    Epoch 62/200
    39241/39241 [==============================] - 7s 180us/sample - loss: 1.3133 - accuracy: 0.6487
    Epoch 63/200
    39241/39241 [==============================] - 7s 184us/sample - loss: 1.3067 - accuracy: 0.6508
    Epoch 64/200
    39241/39241 [==============================] - 8s 196us/sample - loss: 1.3050 - accuracy: 0.6490
    Epoch 65/200
    39241/39241 [==============================] - 8s 193us/sample - loss: 1.3119 - accuracy: 0.6476
    Epoch 66/200
    39241/39241 [==============================] - 8s 198us/sample - loss: 1.2992 - accuracy: 0.6512
    Epoch 67/200
    39241/39241 [==============================] - 7s 190us/sample - loss: 1.3045 - accuracy: 0.6498
    Epoch 68/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.3069 - accuracy: 0.6491
    Epoch 69/200
    39241/39241 [==============================] - 7s 184us/sample - loss: 1.3007 - accuracy: 0.6494
    Epoch 70/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.3082 - accuracy: 0.6502
    Epoch 71/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.3051 - accuracy: 0.6488
    Epoch 72/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.3030 - accuracy: 0.6491
    Epoch 73/200
    39241/39241 [==============================] - 7s 182us/sample - loss: 1.2946 - accuracy: 0.6527
    Epoch 74/200
    39241/39241 [==============================] - 7s 185us/sample - loss: 1.2936 - accuracy: 0.6500
    Epoch 75/200
    39241/39241 [==============================] - 7s 183us/sample - loss: 1.3009 - accuracy: 0.6533
    Epoch 76/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2928 - accuracy: 0.6532
    Epoch 77/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2792 - accuracy: 0.6544
    Epoch 78/200
    39241/39241 [==============================] - 7s 180us/sample - loss: 1.2970 - accuracy: 0.6508
    Epoch 79/200
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.2822 - accuracy: 0.6549
    Epoch 80/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2903 - accuracy: 0.6528
    Epoch 81/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2868 - accuracy: 0.6551
    Epoch 82/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2769 - accuracy: 0.6564
    Epoch 83/200
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.2820 - accuracy: 0.6570
    Epoch 84/200
    39241/39241 [==============================] - 7s 185us/sample - loss: 1.2736 - accuracy: 0.6573
    Epoch 85/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2907 - accuracy: 0.6524
    Epoch 86/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2877 - accuracy: 0.6538
    Epoch 87/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2773 - accuracy: 0.6596
    Epoch 88/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2799 - accuracy: 0.6540
    Epoch 89/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2782 - accuracy: 0.6545
    Epoch 90/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2832 - accuracy: 0.6548
    Epoch 91/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2776 - accuracy: 0.6571
    Epoch 92/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2744 - accuracy: 0.6564
    Epoch 93/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2711 - accuracy: 0.6550
    Epoch 94/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2739 - accuracy: 0.6587
    Epoch 95/200
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.2702 - accuracy: 0.6580
    Epoch 96/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2669 - accuracy: 0.6550
    Epoch 97/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.2797 - accuracy: 0.6568
    Epoch 98/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2659 - accuracy: 0.6545
    Epoch 99/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2678 - accuracy: 0.6571
    Epoch 100/200
    39241/39241 [==============================] - 7s 181us/sample - loss: 1.2703 - accuracy: 0.6568
    Epoch 101/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2617 - accuracy: 0.6626
    Epoch 102/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2699 - accuracy: 0.6555
    Epoch 103/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2628 - accuracy: 0.6598
    Epoch 104/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2658 - accuracy: 0.6572
    Epoch 105/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2628 - accuracy: 0.6583
    Epoch 106/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2588 - accuracy: 0.6605
    Epoch 107/200
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.2616 - accuracy: 0.6617
    Epoch 108/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2538 - accuracy: 0.6602
    Epoch 109/200
    39241/39241 [==============================] - 7s 171us/sample - loss: 1.2705 - accuracy: 0.6577
    Epoch 110/200
    39241/39241 [==============================] - 7s 182us/sample - loss: 1.2566 - accuracy: 0.6626
    Epoch 111/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2557 - accuracy: 0.6598
    Epoch 112/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2660 - accuracy: 0.6585
    Epoch 113/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2509 - accuracy: 0.6626
    Epoch 114/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2503 - accuracy: 0.6604
    Epoch 115/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2516 - accuracy: 0.6615
    Epoch 116/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2600 - accuracy: 0.6592
    Epoch 117/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2498 - accuracy: 0.6628
    Epoch 118/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2467 - accuracy: 0.6618
    Epoch 119/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2378 - accuracy: 0.6644
    Epoch 120/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2411 - accuracy: 0.6637
    Epoch 121/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2524 - accuracy: 0.6628
    Epoch 122/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2374 - accuracy: 0.6646
    Epoch 123/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2476 - accuracy: 0.6605
    Epoch 124/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2387 - accuracy: 0.6651
    Epoch 125/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2406 - accuracy: 0.6619
    Epoch 126/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2444 - accuracy: 0.6658
    Epoch 127/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2369 - accuracy: 0.6646
    Epoch 128/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2459 - accuracy: 0.6628
    Epoch 129/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2437 - accuracy: 0.6609
    Epoch 130/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2445 - accuracy: 0.6652
    Epoch 131/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2339 - accuracy: 0.6665
    Epoch 132/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2418 - accuracy: 0.6643
    Epoch 133/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2465 - accuracy: 0.6644
    Epoch 134/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2338 - accuracy: 0.6661
    Epoch 135/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2261 - accuracy: 0.6654
    Epoch 136/200
    39241/39241 [==============================] - 7s 182us/sample - loss: 1.2370 - accuracy: 0.6653
    Epoch 137/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2327 - accuracy: 0.6650
    Epoch 138/200
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.2264 - accuracy: 0.6694
    Epoch 139/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.2303 - accuracy: 0.6664
    Epoch 140/200
    39241/39241 [==============================] - 7s 184us/sample - loss: 1.2376 - accuracy: 0.6632
    Epoch 141/200
    39241/39241 [==============================] - 7s 183us/sample - loss: 1.2254 - accuracy: 0.6671
    Epoch 142/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.2399 - accuracy: 0.6646
    Epoch 143/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2257 - accuracy: 0.6673
    Epoch 144/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2247 - accuracy: 0.6661
    Epoch 145/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2237 - accuracy: 0.6680
    Epoch 146/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2345 - accuracy: 0.6663
    Epoch 147/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2239 - accuracy: 0.6667
    Epoch 148/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2291 - accuracy: 0.6668
    Epoch 149/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2135 - accuracy: 0.6683
    Epoch 150/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2236 - accuracy: 0.6672
    Epoch 151/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2245 - accuracy: 0.6699
    Epoch 152/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2196 - accuracy: 0.6674
    Epoch 153/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2290 - accuracy: 0.6676
    Epoch 154/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2209 - accuracy: 0.6679
    Epoch 155/200
    39241/39241 [==============================] - 7s 172us/sample - loss: 1.2154 - accuracy: 0.6690
    Epoch 156/200
    39241/39241 [==============================] - 7s 182us/sample - loss: 1.2150 - accuracy: 0.6699
    Epoch 157/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2223 - accuracy: 0.6690
    Epoch 158/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2225 - accuracy: 0.6677
    Epoch 159/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2073 - accuracy: 0.6714
    Epoch 160/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2131 - accuracy: 0.6705
    Epoch 161/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2120 - accuracy: 0.6701
    Epoch 162/200
    39241/39241 [==============================] - 7s 184us/sample - loss: 1.2063 - accuracy: 0.6712
    Epoch 163/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2082 - accuracy: 0.6701
    Epoch 164/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.2136 - accuracy: 0.6677
    Epoch 165/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.2139 - accuracy: 0.6701
    Epoch 166/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.2056 - accuracy: 0.6709
    Epoch 167/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2131 - accuracy: 0.6709
    Epoch 168/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.2030 - accuracy: 0.6725
    Epoch 169/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2087 - accuracy: 0.6706
    Epoch 170/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2022 - accuracy: 0.6731
    Epoch 171/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.2069 - accuracy: 0.6703
    Epoch 172/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.1999 - accuracy: 0.6725
    Epoch 173/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.1967 - accuracy: 0.6752
    Epoch 174/200
    39241/39241 [==============================] - 7s 181us/sample - loss: 1.2107 - accuracy: 0.6697
    Epoch 175/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2031 - accuracy: 0.6694
    Epoch 176/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2064 - accuracy: 0.6716
    Epoch 177/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2062 - accuracy: 0.6717
    Epoch 178/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.1925 - accuracy: 0.6743
    Epoch 179/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.1943 - accuracy: 0.6724
    Epoch 180/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.1968 - accuracy: 0.6756
    Epoch 181/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2000 - accuracy: 0.6722
    Epoch 182/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.1944 - accuracy: 0.6712
    Epoch 183/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.1993 - accuracy: 0.6721
    Epoch 184/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.1895 - accuracy: 0.6748
    Epoch 185/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.2068 - accuracy: 0.6672
    Epoch 186/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.1965 - accuracy: 0.6748
    Epoch 187/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.1982 - accuracy: 0.6735
    Epoch 188/200
    39241/39241 [==============================] - 7s 181us/sample - loss: 1.1971 - accuracy: 0.6734
    Epoch 189/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.2005 - accuracy: 0.6725
    Epoch 190/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.1956 - accuracy: 0.6745
    Epoch 191/200
    39241/39241 [==============================] - 7s 173us/sample - loss: 1.1905 - accuracy: 0.6742
    Epoch 192/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.1893 - accuracy: 0.6767
    Epoch 193/200
    39241/39241 [==============================] - 7s 176us/sample - loss: 1.2053 - accuracy: 0.6716
    Epoch 194/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.1842 - accuracy: 0.6762
    Epoch 195/200
    39241/39241 [==============================] - 7s 175us/sample - loss: 1.1910 - accuracy: 0.6758
    Epoch 196/200
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.1970 - accuracy: 0.6734
    Epoch 197/200
    39241/39241 [==============================] - 7s 177us/sample - loss: 1.1851 - accuracy: 0.6782
    Epoch 198/200
    39241/39241 [==============================] - 7s 174us/sample - loss: 1.1986 - accuracy: 0.6737
    Epoch 199/200
    39241/39241 [==============================] - 7s 179us/sample - loss: 1.1858 - accuracy: 0.6740
    Epoch 200/200
    39241/39241 [==============================] - 7s 178us/sample - loss: 1.1833 - accuracy: 0.6778
    in the market place
     desmond lets the children lend a hand
     molly stays


After another 200 epochs of training (400 total):
```
in the market place
 desmond lets the children lend a hand
 molly stays
```
Let's see this model in action some more in the next section.

## Sing-a-long

Let's try generating some longer batches of text:


```python
seed_text = gen_rand_seq(X, tokenizer)
print(gen_text(model, tokenizer, seed_text, max_length, 100))
```


     i ain't gonna tell you but-a one more time
     oh , keep your hands ( keep your hands ) off my bay-ee-a-by
     girl , you get it through your head
     that boy is mine
     keep your hands ( keep your hands ) off my bay-ee-a-by
     girl , you get it through your head
     that boy is mine
     keep your hands ( keep your hands ) off my bay-ee-a-by
     girl , you get it through your head
     that boy is mine
     keep your hands ( keep your hands ) off my bay-ee-a-by
     girl ,


Pretty cool, definitely sounds vaguely Beatlesesque and there are

Let's see how it does with just some random text like: 'hey, hey, sing with me'


```python
seed_text = 'hey, hey, sing with me'
print(gen_text(model, tokenizer, seed_text, max_length, 100))
```

    hey, hey, sing with me

     i don't wanna kiss you , yeah
     all i gotta do is act naturally

     well , i'll bet you i'm gonna be a big star
     might win an oscar you can never tell
     i went out to go
     and can look to me to me
     and i will sing a lullaby .

     golden slumbers ,
     fill your eyes
     smiles await you when you rise
     sleep pretty darling
     do not cry
     so i know that you will plainly see
     the biggest fool that ever


Wow. This looks like a real song.

## Notes

This network really does produce some lyrics that are Beatlesesque. A big part of this was the limited vocabulary of about 2360 words. The limited vocabulary not only outputs words that are already pretty Beatlesesque, but it allowed me to use a reasonably small LSTM (2 layers of 64 hidden nodes each).  

I wonder how this would do if we increased the vocabulary space to that of all the words in the lyricsfreak.com dataset. We would potentially be able to get more of a variation in the words (both for input and output). This would make it harder to train but possibly more portable and also allow for the same word embeddings and program to be used for multiple artists.

Potential improvements to this network would be:
* Increasing the number of nodes or number of LSTM layers
* Potentially increasing or decreasing the batch-size
* Possibly changing this LSTM to be stateful rather than being stateless
* Increasing the length of the input sequence
* Training the network to allow for partial sequences
