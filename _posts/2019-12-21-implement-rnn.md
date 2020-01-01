---
layout: post
title: Implementing Recurrent Neural Network
categories: [Implementing Algorithms]
tags: 
---
This is an implementation of a basic RNN (not bi-directional, not an LSTM) from scratch (using only numpy). This program will do character level text prediction.

Thanks to Siraj Raval for the walkthrough [here](https://www.youtube.com/watch?v=BwmddtPFWtA&t=4s) that was immensely helpful.

## Import libraries and data


```python
import numpy as np
import random
```


```python
def import_data(file):
    data = open(file, 'r').read()
    chars = list(set(data))
    char_to_ix = {ch:i for i,ch in enumerate(chars)}
    ix_to_char = {i:ch for i,ch in enumerate(chars)}
    return data, chars, char_to_ix, ix_to_char
```


```python
data, chars, char_to_ix, ix_to_char = import_data('kafka.txt')
```


```python
def encode_one_hot(text, char_to_ix):
    indices = [char_to_ix[char] for char in text]
    vectors = np.zeros((len(indices),len(char_to_ix)))
    vectors[range(len(indices)), indices] = 1
    return vectors
```


```python
def decode_one_hot(vectors, ix_to_char):
    text = ''
    for vector in vectors:
        index = np.argmax(vector)
        text += ix_to_char[index]
    return text
```


```python
def input_to_target(vectors):
    return vectors[1:,:]
```


```python
print(decode_one_hot(input_to_target(encode_one_hot(data, char_to_ix)),ix_to_char))
```

    ne morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The bedding was hardly able to cover it and seemed ready to slide off any moment. His many legs, pitifully thin compared with the size of the rest of him, waved about helplessly as he looked.

    "What's happened to me?" he thought. It wasn't a dream. His room, a proper human room although a little too small, lay peacefully between its four familiar walls. A collection of textile samples lay spread out on the table - Samsa was a travelling salesman - and above it there hung a picture that he had recently cut out of an illustrated magazine and housed in a nice, gilded frame. It showed a lady fitted out with a fur hat and fur boa who sat upright, raising a heavy fur muff that covered the whole of her lower arm towards the viewer...





```python
vectors = encode_one_hot(data, char_to_ix)
```

## Forward Prop


```python
def initialize_parameters(vectors, hidden_size, seed=1):
    np.random.seed(seed)
    (data_length, vocab_size) = vectors.shape
    Whx = np.random.randn(hidden_size, vocab_size)*0.01
    Wha = np.random.randn(hidden_size, hidden_size)*0.01
    Wya = np.random.randn(vocab_size, hidden_size)*0.01
    bh = np.zeros((hidden_size,1))
    by = np.zeros((vocab_size,1))
    parameters = (Whx, Wha, Wya, bh, by)
    dWhx = np.zeros_like(Whx)
    dWha = np.zeros_like(Wha)
    dWya = np.zeros_like(Wya)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)
    dparameters = (dWhx, dWha, dWya, dbh, dby)
    mWhx = np.zeros_like(Whx)
    mWha = np.zeros_like(Wha)
    mWya = np.zeros_like(Wya)
    mbh = np.zeros_like(bh)
    mby = np.zeros_like(by)
    memories = (mWhx, mWha, mWya, mbh, mby)
    return parameters, dparameters, memories
```


```python
def softmax(z):
    exp = np.exp(z)
    return exp/sum(exp)
```


```python
def loss(y_hat, y):
    return -np.log(y_hat[np.where(y==1)])
```


```python
def forward_step(a_t_minus_one, x, parameters):
    (Whx, Wha, Wya, bh, by) = parameters
    h = Wha.dot(a_t_minus_one) + Whx.dot(x[:,None]) + bh
    a = np.tanh(h)
    z = Wya.dot(a) + by
    y_hat = softmax(z)
    return y_hat, a
```


```python
def forward_prop(X, Y, parameters, init_a):
    A={}
    A[-1] = init_a.copy()
    Y_hat = []
    losses = []
    for t in range(len(Y)):
        x = X[t,:]
        y = Y[t,:]
        y_hat, a = forward_step(A[t-1], x, parameters)
        A[t] = a
        Y_hat.append(y_hat)
        losses.append(loss(y_hat, y))
    return A, Y_hat, losses
```

## Backward prop


```python
def backward_step(x, y, y_hat, a, a_t_minus_one, da_next, parameters, dparameters):
    (dWhx, dWha, dWya, dbh, dby) = dparameters
    (Whx, Wha, Wya, bh, by) = parameters
    dy = np.copy(y_hat) - y
    dWya += dy.dot(a.T)
    dby += dy
    da = Wya.T.dot(dy) + da_next
    dh = (1.0-a**2)*da
    dWhx += dh.dot(x.T)
    dWha += dh.dot(a_t_minus_one.T)
    dbh += dh
    da_next = Wha.T.dot(dh)
    dparameters = (dWhx, dWha, dWya, dbh, dby)
    for dparameter in dparameters:
        np.clip(dparameter, -5, 5, out=dparameter)
    return da_next, dparameters
```


```python
def backward_prop(X, Y, Y_hat, A, parameters, dparameters):
    da_next = np.zeros_like(dparameters[3])
    _, dparameters, _ = initialize_parameters(X, len(A[0]))
    for t in reversed(range(len(Y))):
        x = X[t,:][:,None]
        y = Y[t,:][:,None]
        y_hat = Y_hat[t]
        a = A[t]
        a_t_minus_one = A[t-1]
        da_next, dparameters = backward_step(x, y, y_hat, a, a_t_minus_one, da_next, parameters, dparameters)
    return parameters, dparameters
```

## Training


```python
def sequence_generator(a, parameters, char_to_ix, ix_to_char, seed_char=None, n=100):
    (Whx, Wha, Wya, bh, by) = parameters
    if seed_char is None:
        seed_char = ix_to_char[random.randrange(len(by))]
    if isinstance(seed_char,str):
        X = encode_one_hot(seed_char, char_to_ix)
        x = X[-1,:][:,None]
    else:
        x = seed_char[:,None]
    Y_hat = [x]
    for i in range(n):
        h = Whx.dot(x) + Wha.dot(a) + bh
        a = np.tanh(h)
        z = Wya.dot(a) + by
        y_hat = softmax(z)
        Y_hat.append(y_hat)
        x = (y_hat==y_hat.max()).astype(int)
    seq = decode_one_hot(Y_hat, ix_to_char)
    print(seq)
    return seq
```


```python
testing = initialize_parameters(vectors, 100, seed=123)
sequence_generator(testing[0][3], testing[0], char_to_ix, ix_to_char, seed_char='A')
```

    Aaxf1y@JuM:3;Rn0ui'y@JuM:3;Rn0ui'y@JuM:3;Rn0ui'y@JuM:3;Rn0ui'y@JuM:3;Rn0ui'y@JuM:3;Rn0ui'y@JuM:3;Rn0u





```python
def train_rnn(data, hyperparameters, seed=1, init_parameters=None, use_init_parameters=False):

    # unpack hyperparameters
    hidden_size = hyperparameters['hidden_size']
    seq_length = hyperparameters['seq_length']
    learning_rate = hyperparameters['learning_rate']
    epochs = hyperparameters['epochs']
    sample_freq = hyperparameters['sample_freq']

    # generate vectors and vocab dictionaries
    chars = list(set(data))
    char_to_ix = {ch:i for i,ch in enumerate(chars)}
    ix_to_char = {i:ch for i,ch in enumerate(chars)}
    vectors = encode_one_hot(data, char_to_ix)
    target = input_to_target(vectors)

    # initialize parameters and smooth_loss
    parameters, dparameters, memories = initialize_parameters(vectors, hidden_size, seed=seed)
    smooth_loss = np.log(len(char_to_ix))*seq_length # loss at iteration 0
    init_a = np.zeros_like(parameters[3])
    if use_init_parameters==True:
        parameters = init_parameters
    p = 0 # position to start reading at
    # forward and backward prop
    for i in range(epochs):
        if p+seq_length>len(vectors): # start from the beginning of the book again
            p = 0
            init_a = np.zeros_like(parameters[3])
        X = vectors[p:p+seq_length]
        Y = target[p:p+seq_length]
        A, Y_hat, losses = forward_prop(X, Y, parameters, init_a)
        init_a = A[seq_length-1]
        if (i==0 or (i+1)%sample_freq == 0):
            print('Iter:', i+1, 'Loss:', smooth_loss)
            sequence_generator(init_a, parameters, char_to_ix, ix_to_char, seed_char=X[-1], n=100)
            print()
        parameters, dparameters = backward_prop(X, Y, Y_hat, A, parameters, dparameters)

        # update using adagrad
        for parameter, dparameter, memory in zip(parameters, dparameters, memories):
            memory += dparameter*dparameter
            parameter -= learning_rate*dparameter/np.sqrt(memory+1e-8)
        smooth_loss = smooth_loss* 0.999 + np.sum(losses)*0.001
        p+= seq_length

    return loss, parameters    
```


```python
hyperparameters = {
    'hidden_size' : 150,
    'seq_length' : 35,
    'learning_rate' : 0.05,
    'epochs':10000,
    'sample_freq':1000
}
```


```python
loss, parameters = train_rnn(data, hyperparameters, seed=1)
```

    Iter: 1 Loss: 153.37093221358583
    eG)XEk.Q:?-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-J?2FY-

    Iter: 1000 Loss: 114.98703910985724
    n he pere he pere he pere he pere he pere he pere he pere he pere he pere he pere he pere he pere he

    Iter: 2000 Loss: 89.98119716264166
    ed and and and and and and and and and and and and and and and and and and and and and and and and an

    Iter: 3000 Loss: 78.24674308913039
     he was she the the the the the the the the the the the the the the the the the the the the the the t

    Iter: 4000 Loss: 81.04244786556373
    ; the the the the the the the the the the the the the the the the the the the the the the the the the

    Iter: 5000 Loss: 73.78070609787777
    e the the the the the the the the the the the the the the the the the the the the the the the the the

    Iter: 6000 Loss: 68.36697717802687
    e him her and her and her and her and her and her and her and her and her and her and her and her and

    Iter: 7000 Loss: 66.07226962796535
    mand his for his for his for his for his for his for his for his for his for his for his for his for

    Iter: 8000 Loss: 71.86785818263544
    's and he was to he was to he was to he was to he was to he was to he was to he was to he was to he w

    Iter: 9000 Loss: 67.36134684532409
    wat the did the did the did the did the did the did the did the did the did the did the did the did t

    Iter: 10000 Loss: 63.723388456472314
    ware the was had his father was had his father was had his father was had his father was had his fath
