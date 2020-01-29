---
layout: post
title: Lyrics Generator with Simple Web App
categories: [End to End Projects]
tags:
---
![beatles](https://www.dw.com/image/39219505_303.jpg){: width="100%" style="margin:0px 0px 10px 0px"}
### Web App Demo: Lyrics Generation
Enter text like `sing with me` or `how much do I love you`, and the app will 'sing back' with Beatles-like lyrics.
<div style="padding-bottom: 0.5cm">
    <div class="card text-center bg-light">
        <div class="card-body" style="padding-bottom: 0.2cm">
            <input class="card-title form-control" type="text" id="input" name="input" placeholder="Type the beginning of a song (i.e. 'sing with me')"/>
            <button class="card-text btn btn-outline-primary" id="btn">Sing Back</button>
            <div class="spinner" id="spinner" style="display: none">
              <div class="double-bounce1"></div>
              <div class="double-bounce2"></div>
            </div>
        </div>
        <div class="card-footer bg-white">
            <pre class="card-text api-pre" style="padding-bottom: 0.2cm">
                <div class="item" id="api_input">I will sing back </div>
                <div class="item" id="api_output">like the Beatles!</div>
            </pre>
        </div>
    </div>
</div>
<script type="text/javascript">
    function api_call(input) {
        // hide button and make the spinner appear
        $('#btn').toggle();
        $('#spinner').toggle();
        $.ajax({
            url: "http://54.184.82.12/api",
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(input),
            success: function( data, textStatus, jQxhr ){
                // toggle the spinner and button
                $('#btn').toggle();
                $('#spinner').toggle();
                // fill the html for answer
                $('#api_input').html( data.input );
                $('#api_output').html( data.output );
                $("#input").val("");
            },
            error: function( jqXhr, textStatus, errorThrown ){
                $('#btn').toggle();
                $('#spinner').toggle();
                $('#api_input').html( "Sorry, the server is taking too long..." );
                $('#api_output').html( "Try again later!" );
                console.log( errorThrown );
            },
            timeout: 20000 // sets timeout to 20 seconds
        });
    }
    $( document ).ready(function() {
        // request when clicking on the button
        $('#btn').click(function() {
            // get the input data
            var input = $("#input").val();
            api_call(input);
            input = "";
    });
    });
</script>
Note: sometimes the model may get into a repetitive loop, its not perfect :-)

This project uses LSTMs in Tensorflow Keras to build word based language models for the song lyrics by a chosen artist (I'm going to try The Beatles). The lyrics come from a [Kaggle dataset](https://www.kaggle.com/mousehead/songlyrics) of lyrics scraped from lyricsfreak.com.

The model will have two layers of LSTMs with 64 hidden nodes each, and we will try generating text after various levels of training. Additionally, this project contains data preparation, model creation, and analysis of the algorithms.

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
    ...
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
    ...
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
    ...
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
    ...
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
    ...
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
    ...
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
