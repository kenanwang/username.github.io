---
layout: post
title: Naive-Bayes Implementation
categories: [Implementing Algorithms]
tags:
---
This is an implementation of a Gaussian Naive-Bayes using only numpy using toy data. I follow this [guide](https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/) and use scikit learn for accuracy scoring and I later compare my implementation to the scikit-learn implementation.
![gnb](https://upload.wikimedia.org/wikipedia/commons/b/b4/Naive_Bayes_Classifier.gif){: width="100%" style="margin:20px 0px 10px 0px"}

## 0. Import Libraries


```python
import numpy as np
from scipy import stats
from sklearn.datasets.samples_generator import make_blobs
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


```python
# for later comparison to scikit-learn libraries
from sklearn.naive_bayes import GaussianNB
```

## I. Implement Naive-Bayes on small 2-d sample set


```python
# Generate small 2-d sample classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# Check
print(X.shape, y.shape)
print(X[:5])
print(y[:5])
```

    (100, 2) (100,)
    [[-0.79415228  2.10495117]
     [-9.15155186 -4.81286449]
     [-3.10367371  3.90202401]
     [-1.42946517  5.16850105]
     [-7.4693868  -4.20198333]]
    [0 1 0 0 1]



```python
# fit probability distribution to univariate data sample
def fit_distribution(data):
    # estimate prameters
    mu = np.mean(data)
    sigma = np.std(data)
    print(mu, sigma)
    # fit distribution
    dist = stats.norm(mu, sigma)
    return dist
```


```python
# sort data into classes
X_y0 = X[y == 0]
X_y1 = X[y == 1]
print(X_y0.shape, X_y1.shape)
```

    (50, 2) (50, 2)



```python
print(len(y[y==1]))
```

    50



```python
# Calculate priors
priory_y0 = len(X_y0)/len(X)
priory_y1 = len(X_y1)/len(X)
print(priory_y0, priory_y1)
```

    0.5 0.5



```python
print(X_y0[:5])
print(X_y0[:5, 0])
print(X_y0[:5, 1])
```

    [[-0.79415228  2.10495117]
     [-3.10367371  3.90202401]
     [-1.42946517  5.16850105]
     [-2.76017908  5.55121358]
     [-1.17104176  4.33091816]]
    [-0.79415228 -3.10367371 -1.42946517 -2.76017908 -1.17104176]
    [2.10495117 3.90202401 5.16850105 5.55121358 4.33091816]



```python
# Probability distribution functions for each X term for y==0
X1_y0 = fit_distribution(X_y0[:, 0])
X2_y0 = fit_distribution(X_y0[:, 1])

# Probability distribution functions for each X term for y==1
X1_y1 = fit_distribution(X_y1[:, 0])
X2_y1 = fit_distribution(X_y1[:, 1])
```

    -1.5632888906409914 0.787444265443213
    4.426680361487157 0.958296071258367
    -9.681177100524485 0.8943078901048118
    -3.9713794295185845 0.9308177595208521



```python
def probability(X, prior, dist1, dist2):
    return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])
```


```python
# Pick one sample to classify
Xsample, ysample = X[0], y[0]
```


```python
# Classify our one sample
py0 = probability(Xsample, priory_y0, X1_y0, X2_y0)
py1 = probability(Xsample, priory_y1, X1_y1, X2_y1)
print(f'P(y=0) | X = {Xsample}) = {py0*100:.3f}')
print(f'P(y=1) | X = {Xsample}) = {py1*100:.3f}')
print(f'Truth: y = {ysample}')
```

    P(y=0) | X = [-0.79415228  2.10495117]) = 0.348
    P(y=1) | X = [-0.79415228  2.10495117]) = 0.000
    Truth: y = 0


## II. Implement general function for Gaussian Naive-Bayes


```python
def fit_gaussian_nb(X, y):
    if len(X) == len(y):
        sample_size = len(X)
    else:
        print('len(X) must equal len(y)')
    n = X.shape[1]
    m = max(y)+1
    # Calculate Priors
    priors = [None]*m
    for i in range(m):
        priors[i] = len(y[y==i])/sample_size
    # Calculate conditional probability distribution functions
    conditional_pdfs = defaultdict(dict)
    for j in range(m):
        for i in range(n):
            Xi_yj = X[y==j][:,i]
            conditional_pdfs[i][j] = stats.norm(np.mean(Xi_yj), np.std(Xi_yj))
    return priors, conditional_pdfs
```


```python
def gnb_predictions(X_test, priors, conditional_pdfs):
    m = len(priors)
    if len(X_test.shape)==1:
        sample_size = 1
        n = X_test.shape[0]
    else:
        sample_size = len(X_test)
        n = X_test.shape[1]        
    log_prob_y_X = np.zeros(shape=(sample_size, m))
    for i in range(sample_size):
        if sample_size == 1:
            sample_i = X_test
        else:
            sample_i = X_test[i]
        for j in range(m):
            probs_X_y=[]
            for k in range(n):
                probs_X_y.append(conditional_pdfs[k][j].pdf(sample_i[k]))
            probs_X_y.append(priors[j])
            log_prob_y_X[i,j] = sum(np.log(probs_X_y))
    predictions = log_prob_y_X.argmax(axis=1)
    return predictions, log_prob_y_X
```

## III. Test Gaussian NB functions


```python
# First try sample from Part I
ex_priors, ex_conditional_pdfs = fit_gaussian_nb(X, y)
ex_predictions, ex_log_probs = gnb_predictions(Xsample, ex_priors, ex_conditional_pdfs)
print(ex_predictions, ex_log_probs)
```

    [0] [[ -5.66138661 -73.02986163]]



```python
print(np.log([py0, py1]))
```

    [ -5.66138661 -73.02986163]



```python
# Generate larger 3-d sample classification dataset, with 4 centers
X, y = make_blobs(n_samples=10000, centers=4, n_features=3, random_state=123)
# Check
print(X.shape, y.shape)
print(X[:5])
print(y[:5])
```

    (10000, 3) (10000,)
    [[10.2860709   3.08445143  1.10950203]
     [ 9.48094927  3.63848054 -0.81090591]
     [ 5.36061008 -3.56854951 -7.5313075 ]
     [ 3.34099263 -3.63460408 -4.96655885]
     [10.52164696  4.68535779 -0.98655588]]
    [2 2 0 0 2]



```python
# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=111)
```


```python
# Train Gaussian Naive-Bayes Algorithm with training data
# Produces priors and conditional probability distribution functions
priors, conditional_pdfs = fit_gaussian_nb(X_train, y_train)
```


```python
# Make predictions using Gaussian Naive-Bayes
predictions, log_probs = gnb_predictions(X_test, priors, conditional_pdfs)
```


```python
# Check accuracy
accuracy_score(y_test, predictions)
```




    1.0



Since the last test was 100% accurate, going to try to add some noise and see how we do


```python
# 4-d sample classification dataset, with 5 centers
X, y = make_blobs(n_samples=80, centers=5, n_features=4, random_state=123)
# Add a noisey 4-d sample set with 5 centers
X_noise, y_noise = make_blobs(n_samples=20, centers=5, n_features=4, random_state=1)
X = np.concatenate((X, X_noise))
y = np.concatenate((y, y_noise))
# Check
print(X.shape, y.shape)
print(X[:5])
print(y[:5])
```

    (100, 4) (100,)
    [[-0.93028238 -3.13570734 -3.49126414  4.97257839]
     [ 3.39205793 -2.63791391  8.85884676  4.01828135]
     [ 0.07490892 -0.61310518 -3.37610846  4.72430188]
     [-1.35000378 -9.38997878 -1.048982    4.40615378]
     [ 4.24365571 -5.60347876 -4.04567188  1.83353192]]
    [2 1 2 3 0]



```python
# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=111)
```


```python
# Train Gaussian Naive-Bayes Algorithm with training data
# Produces priors and conditional probability distribution functions
priors, conditional_pdfs = fit_gaussian_nb(X_train, y_train)
```


```python
# Make predictions using Gaussian Naive-Bayes
predictions, log_probs = gnb_predictions(X_test, priors, conditional_pdfs)
```


```python
# Check accuracy
accuracy_score(y_test, predictions)
```




    0.8



## IV. Compare with Scikit-learn's Gaussian Function


```python
model = GaussianNB()
```


```python
model.fit(X_train,y_train)
```




    GaussianNB(priors=None, var_smoothing=1e-09)




```python
predictions = model.predict(X_test)
```


```python
accuracy_score(y_test, predictions)
```




    0.8
