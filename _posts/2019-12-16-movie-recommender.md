---
layout: post
title: Movie Recommender
categories: [End to End Projects]
tags: [Item Based Collaborative Filtering, Cross Validation, Hyperparameter Tuning]
---
This project uses item based collaborative filtering to make movie recommendations. Because this project was unique I implemented cross-validation and hyperparameter tuning from scratch, and defined a project specific cost function.

I'll use a fairly simple correlation function to model item similarity based on similar ratings in a movie databased. The basic implementation isn't too complicated but I'll also tune the models against three metrics:
1. A cross validated score for the model's ability to predict how a user will rate movies
2. The number of movies recommended to a user that they haven't previously seen/rated
3. We will apply an eye test

These are the hyperparameters to be tuned:
* types of correlation
* min_period

Dataset citation:
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

Also thanks to Sun Dog Education for guidance in how to implement this.

## Import Libraries


```
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
%matplotlib inline

import random
```

## Importing and Preparing Data


```
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', names=['user_id', 'movie_id', 'rating'], usecols=range(3), encoding='ISO-8859-1')
ratings.head()
```

    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
      """Entry point for launching an IPython kernel.





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
      <th>user_id</th>
      <th>movie_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1193</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>661</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>914</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3408</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2355</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```
# changing the rating system so that the mean is 0
ratings['rating'] = ratings['rating'] - ratings.rating.mean()
ratings.rating.head()
```




    0    1.418436
    1   -0.581564
    2   -0.581564
    3    0.418436
    4    1.418436
    Name: rating, dtype: float64




```
by_user = ratings.pivot_table(index=['user_id'], columns=['movie_id'], values='rating')
by_user.head()
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
      <th>movie_id</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>3943</th>
      <th>3944</th>
      <th>3945</th>
      <th>3946</th>
      <th>3947</th>
      <th>3948</th>
      <th>3949</th>
      <th>3950</th>
      <th>3951</th>
      <th>3952</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.418436</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.581564</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3706 columns</p>
</div>



## Define score function

The cost function below is based on looking at the test fold and giving a +1 score for accurate predictions. I also want to penalize incorrect predictions, -2 seems like a reasonable weight since incorrect predictions probably hurt trust more than accurate predictions build them. If a user likes a movie that doesn't get recommended we'd also like to punish this slightly and if a user doesn't like a movie and it isn't recommended we'd like to reward this slightly. One way of doing this is by making the cost function a product of two numbers: an actual score and a predicted score. The actual score is +1 for liked movies, -2 for not liked movies. The predicted score is +1 for predicted movies, -0.2 for movies not predicted. This means a true negative is rewarded .4, and a false negative punished .2.


```
# this will get the cross validation scores for an algorithm
# the score will increase for correctly predicted favorable movies, it will decrease if predicting unfavorable movies
def cross_validate_corr(num_folds, user_df, corr, rand_seed):
    random.seed(rand_seed)
    # as mentioned before we're tracking two quantitative metrics
    user_scores1 = []
    user_scores2 = []
    # this will generate folds that we will reuse for each user
    fold_size = int(len(user_df.T)/num_folds)
    folds = []
    movies = list(user_df.T.index)
    for i in range(num_folds):
        fold = []
        while len(fold) < fold_size:
            rand_index = random.randrange(len(movies))
            fold.append(movies.pop(rand_index))
        folds.append(fold)
    #get a score for each user before averaging over all users
    for user_index, user in user_df.iterrows():
        # for each user evaluate metric 1 (error score) against the generated folds
        fold_scores1 = []
        fold_scores2 = []
        for fold in folds:
            score = []
            test = user.loc[fold]
            train = user.drop(fold)
            returns = recommendations_from_corr(train, corr)
            test_clean = test.dropna()
            for movie in test_clean.index:
                if test_clean[movie] > 0:
                    actual = 1
                else:
                    actual = -2
                if returns.get(movie, default=0)==0:
                    expected = -0.2
                else:
                    expected = 1
                score.append(actual * expected)
            score_sum = np.sum(score)
            fold_scores1.append(score_sum)
            fold_scores2.append(len(returns))
        user_scores1.append(np.mean(fold_scores1))
        user_scores2.append(np.mean(fold_scores2))
        if(len(user_scores1)%100 == 0):
            print(len(user_scores1), np.mean(user_scores1), np.mean(user_scores2))
    return np.mean(user_scores1), np.mean(user_scores2)
```

## Define recommendations function


```
def recommendations_from_corr(user_series, corr):
    user_ratings = user_series.dropna()
    sim_candidates = pd.Series()
    for movie in user_ratings.index:
        sims = corr[movie].dropna()
        sims = sims.map(lambda x: x * user_ratings[movie])
        sim_candidates = sim_candidates.append(sims)    
    sim_candidates = sim_candidates.groupby(sim_candidates.index).sum()
    overlap = (user_ratings.index & sim_candidates.index)
    sim_candidates.drop(labels = overlap, inplace=True)
    sim_candidates.sort_values(ascending=False, inplace=True)
    sim_candidates = sim_candidates[sim_candidates>0]
    return sim_candidates
```

## Attempt recommendation and cross validation functions

The cross-validation is taking too long with this large of a dataset, I'm going to use a smaller dataset for now.


```
small_ratings = pd.read_csv('ml-latest-small/ratings.csv', usecols=range(3))
small_ratings.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```
small_ratings['rating'] = small_ratings.rating - small_ratings.rating.mean()
small_ratings.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.498443</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>0.498443</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>0.498443</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>1.498443</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>1.498443</td>
    </tr>
  </tbody>
</table>
</div>




```
small_by_user = small_ratings.pivot_table(index=['userId'], columns=['movieId'], values='rating')
```


```
corr = small_by_user.corr(method='pearson', min_periods=50)
```


```
corr.head()
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
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>193565</th>
      <th>193567</th>
      <th>193571</th>
      <th>193573</th>
      <th>193579</th>
      <th>193581</th>
      <th>193583</th>
      <th>193585</th>
      <th>193587</th>
      <th>193609</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.330978</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.106465</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.021409</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.330978</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.016626</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9724 columns</p>
</div>




```
user_3 = small_by_user.loc[3]
```


```
print(recommendations_from_corr(user_3, corr).head(10))
```

    1527    0.940285
    208     0.815594
    316     0.694791
    344     0.524894
    10      0.420379
    111     0.408094
    733     0.402525
    3793    0.400005
    736     0.374564
    5816    0.258474
    dtype: float64



```
print(cross_validate_corr(num_folds=3, user_df=small_by_user, corr=corr, rand_seed=1))
```

    100 6.383333333333333 188.83666666666667
    200 5.571666666666666 183.69833333333332
    300 6.122444444444445 185.75888888888892
    400 6.144500000000001 184.79666666666665
    500 6.648533333333333 186.24333333333334
    600 6.695555555555555 188.48777777777775
    (6.872021857923498, 188.32950819672132)


## Testing our hyperparameters


```
# the scores from the first cross validation test were relatively stable over the course of each of the 100 users
# I'll try comparing the hyperparameters with a sample of users
small_by_user_index = list(small_by_user.index)
small_by_user_sample = []
while len(small_by_user_sample) < 100:
    rand_index = random.randrange(len(small_by_user_index))
    small_by_user_sample.append(small_by_user_index.pop(rand_index))
small_by_user_sample = small_by_user.loc[small_by_user_sample]
small_by_user_sample.shape
```




    (100, 9724)




```
method_list = ['pearson', 'spearman']
min_periods_list = [20, 100]
```


```
# generate the correlation matrices and store them into a dictionary
corr_matrices = {}
for method in method_list:
    for min_periods in min_periods_list:
        corr_matrices[(method, min_periods)] = small_by_user.corr(method=method, min_periods=min_periods)
        print(method, min_periods, 'done.')
```

    pearson 20 done.
    pearson 100 done.
    spearman 20 done.
    spearman 100 done.


The kendall correlation calculation `corr_matrices[('kendall',)] = small_by_user.corr(method='kendall')`
took too long, so I'm skipping it.


```
# score each correlation matrix with the various hyperparameters
scores = {}
for index, corr in corr_matrices.items():
    scores[index] = cross_validate_corr(num_folds=3, user_df=small_by_user_sample, corr=corr, rand_seed=1)
print(scores)
```

    100 6.953333333333334 638.9866666666667
    100 5.86533333333333 35.42333333333333
    100 6.725333333333332 642.68
    100 5.837333333333331 35.419999999999995
    {('pearson', 20): (6.953333333333334, 638.9866666666667), ('pearson', 100): (5.86533333333333, 35.42333333333333), ('spearman', 20): (6.725333333333332, 642.68), ('spearman', 100): (5.837333333333331, 35.419999999999995)}


The methods for correlation are fairly similar in score, they don't differ too much on the accuracy metric based on my cost function. They differ significantly on the second metric. Lets see how the eye test differentiates between a pearson correlation with minimum periods of 20 and 100.

## Eye test


```
small_movies = pd.read_csv('ml-latest-small/movies.csv', usecols=range(2))
small_ratings = pd.read_csv('ml-latest-small/ratings.csv', usecols=range(3))
small_ratings = pd.merge(small_movies, small_ratings)
small_ratings.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>userId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>7</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>15</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>17</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>




```
small_ratings['rating'] = small_ratings.rating - small_ratings.rating.mean()
small_ratings.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>userId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>1</td>
      <td>0.498443</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>5</td>
      <td>0.498443</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>7</td>
      <td>0.998443</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>15</td>
      <td>-1.001557</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>17</td>
      <td>0.998443</td>
    </tr>
  </tbody>
</table>
</div>




```
small_by_user = small_ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
```


```
def test_single_user(user_series, corr, test_perc=.33, rand_seed=1):
    random.seed(rand_seed)
    test_size = int(len(user_series)*test_perc)
    user_copy = user_series.copy()
    test = {}
    while len(test) < test_size:
        rand_index = random.choice(user_copy.index)
        test[rand_index] = user_copy.pop(rand_index)
    score = []
    test = pd.Series(test)
    train = user_series.drop(test.index)
    returns = recommendations_from_corr(train, corr)
    test_clean = test.dropna()
    for movie in test_clean.index:
        if test_clean[movie] > 0:
            actual = 1
        else:
            actual = -2
        if returns.get(movie, default=0)==0:
            expected = -0.2
        else:
            expected = 1
        print(movie, actual * expected)
        score.append(actual * expected)
    score = np.sum(score)
    print('score:', score)
    return score
```

Pearson 20 eye test.


```
pear_20 = small_by_user.corr(method='pearson', min_periods=20)
```


```
pear_20.head()
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
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'71 (2014)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>'Salem's Lot (2004)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9719 columns</p>
</div>




```
user_10 = small_by_user.loc[10]
```


```
returns_pear_20 = recommendations_from_corr(user_10, pear_20)
```


```
returns_pear_20.head(10)
```




    Fugitive, The (1993)                                4.984729
    Star Trek IV: The Voyage Home (1986)                4.222668
    City Slickers (1991)                                4.019699
    Sneakers (1992)                                     3.984689
    Harry Potter and the Goblet of Fire (2005)          3.956484
    Final Fantasy: The Spirits Within (2001)            3.901192
    X2: X-Men United (2003)                             3.784831
    Thank You for Smoking (2006)                        3.657521
    Harry Potter and the Order of the Phoenix (2007)    3.459060
    Mission: Impossible (1996)                          3.230248
    dtype: float64




```
test_single_user(user_10, pear_20)
```

    Wedding Date, The (2005) 0.4
    Skyfall (2012) 1
    Terminal, The (2004) -2
    Avatar (2009) 0.4
    Mona Lisa Smile (2003) -0.2
    Bourne Ultimatum, The (2007) -2
    First Daughter (2004) -0.2
    Help, The (2011) 0.4
    Notting Hill (1999) -2
    Something's Gotta Give (2003) -0.2
    Shrek (2001) 1
    Love Actually (2003) 1
    Mulan (1998) 1
    Pulp Fiction (1994) 0.4
    Match Point (2005) 0.4
    Twilight Saga: Eclipse, The (2010) 0.4
    Enough Said (2013) 0.4
    Prince & Me, The (2004) -0.2
    Mary Poppins (1964) 0.4
    Magic Mike (2012) -0.2
    Dark Knight Rises, The (2012) 1
    Best Exotic Marigold Hotel, The (2011) -0.2
    Grand Budapest Hotel, The (2014) 0.4
    Tangled Ever After (2012) -0.2
    St Trinian's 2: The Legend of Fritton's Gold (2009) 0.4
    American Beauty (1999) 0.4
    Twilight Saga: Breaking Dawn - Part 2, The (2012) 0.4
    Graduate, The (1967) -2
    27 Dresses (2008) 0.4
    Matrix, The (1999) 0.4
    Twilight (2008) -0.2
    Sixth Sense, The (1999) 0.4
    Morning Glory (2010) 0.4
    When Harry Met Sally... (1989) -2
    Rust and Bone (De rouille et d'os) (2012) 0.4
    Despicable Me 2 (2013) -0.2
    Amazing Spider-Man, The (2012) -2
    The Hundred-Foot Journey (2014) -0.2
    Made of Honor (2008) 0.4
    Chasing Liberty (2004) -0.2
    Quantum of Solace (2008) -2
    Dark Knight, The (2008) 1
    Hitch (2005) 1
    Frozen (2013) 1
    Interstellar (2014) 0.4
    Pretty One, The (2013) -0.2
    How Do You Know (2010) 0.4
    Love and Other Drugs (2010) 0.4
    Valentine's Day (2010) 0.4
    Fight Club (1999) 0.4
    Maid in Manhattan (2002) 0.4
    score: 1.2000000000000006





    1.2000000000000006



Pearson 100 eye test.


```
pear_100 = small_by_user.corr(method='pearson', min_periods=100)
```


```
pear_100.head()
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
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'71 (2014)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>'Salem's Lot (2004)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9719 columns</p>
</div>




```
user_10 = small_by_user.loc[10]
```


```
returns_pear_100 = recommendations_from_corr(user_10, pear_100)
```


```
returns_pear_100.head(15)
```




    Fugitive, The (1993)           0.542148
    Mission: Impossible (1996)     0.449540
    Lion King, The (1994)          0.423012
    Beauty and the Beast (1991)    0.394659
    Apollo 13 (1995)               0.273224
    Mask, The (1994)               0.242050
    Batman Forever (1995)          0.238954
    Mrs. Doubtfire (1993)          0.238462
    Speed (1994)                   0.233273
    True Lies (1994)               0.200027
    dtype: float64




```
test_single_user(user_10, pear_100)
```

    Wedding Date, The (2005) 0.4
    Skyfall (2012) -0.2
    Terminal, The (2004) 0.4
    Avatar (2009) 0.4
    Mona Lisa Smile (2003) -0.2
    Bourne Ultimatum, The (2007) 0.4
    First Daughter (2004) -0.2
    Help, The (2011) 0.4
    Notting Hill (1999) 0.4
    Something's Gotta Give (2003) -0.2
    Shrek (2001) -0.2
    Love Actually (2003) -0.2
    Mulan (1998) -0.2
    Pulp Fiction (1994) -2
    Match Point (2005) 0.4
    Twilight Saga: Eclipse, The (2010) 0.4
    Enough Said (2013) 0.4
    Prince & Me, The (2004) -0.2
    Mary Poppins (1964) 0.4
    Magic Mike (2012) -0.2
    Dark Knight Rises, The (2012) -0.2
    Best Exotic Marigold Hotel, The (2011) -0.2
    Grand Budapest Hotel, The (2014) 0.4
    Tangled Ever After (2012) -0.2
    St Trinian's 2: The Legend of Fritton's Gold (2009) 0.4
    American Beauty (1999) -2
    Twilight Saga: Breaking Dawn - Part 2, The (2012) 0.4
    Graduate, The (1967) 0.4
    27 Dresses (2008) 0.4
    Matrix, The (1999) -2
    Twilight (2008) -0.2
    Sixth Sense, The (1999) -2
    Morning Glory (2010) 0.4
    When Harry Met Sally... (1989) 0.4
    Rust and Bone (De rouille et d'os) (2012) 0.4
    Despicable Me 2 (2013) -0.2
    Amazing Spider-Man, The (2012) 0.4
    The Hundred-Foot Journey (2014) -0.2
    Made of Honor (2008) 0.4
    Chasing Liberty (2004) -0.2
    Quantum of Solace (2008) 0.4
    Dark Knight, The (2008) 1
    Hitch (2005) -0.2
    Frozen (2013) -0.2
    Interstellar (2014) 0.4
    Pretty One, The (2013) -0.2
    How Do You Know (2010) 0.4
    Love and Other Drugs (2010) 0.4
    Valentine's Day (2010) 0.4
    Fight Club (1999) -2
    Maid in Manhattan (2002) 0.4
    score: -2.4000000000000004





    -2.4000000000000004




```
print(len(returns_pear_20), len(returns_pear_100))
```

    371 10


Based on the Eye Test as the scores suggested it seems that both have similar accuracy, but the 100 min_periods limit doesn't give enough recommendations, 10, to be very interesting. On the other hand 371 might be too many. We might be best served picking a min_periods value that's in between, perhaps 50, we saw in our first cross validation test that that gave us an average of 188 recommendations, this seems roughly reasonable. If we scaled up the data (number of reviews, movies, users) this min_periods value would need to be tuned again. It'd be interesting to think about how to adjust tune this regularly. In another project I compared the min_periods limit to a p-value restriction on this same task: movie recommendations. min_periods seemed to work better than p-value which was susceptible to issues of low data.
