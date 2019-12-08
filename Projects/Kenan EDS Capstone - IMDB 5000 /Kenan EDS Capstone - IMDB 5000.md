---
layout: page
title: IMDB 5000 Revenue Regression
permalink: /capstone/
---
## Introduction

This project applies Machine Learning to an IMDB database of 5000 movies. The goal is to predict the revenue of a movie at a level that is helpful to a studio. Since the costs of movies differ widely I will evaluate my revenue prediction against profit margin.

In order to define a metric for this project I will estimate profit margin for the movies. This will be done assuming that the total budget for a movie can be estimated by doubling the production budget of that movie<sup>1</sup>. I will find out the standard deviation of profit margin for the movies in this dataset and attempt to predict revenue to within 1/4 of a standard deviation of the spread of the profit margin in the dataset. I assume that it would be helpful for movie executives to prevent the worst loses in their portfolio and also avoid underbudgeting strong performers, that is the reason for choosing this metric.

<sup>1</sup> https://stephenfollows.com/how-movies-make-money-hollywood-blockbusters/

## Library Imports


```python
# These are all the libraries I want to use for initial analysis. I will import scikit libraries later.
import numpy as np

import pandas as pd
pd.set_option('display.max_columns', 100)

from matplotlib import pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_style('darkgrid')
```


```python
# Scikit libraries

# For genres feature engineering
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.exceptions import NotFittedError
```

## Exploratory Analysis

Will do exploratory analysis here. Will take notes as I go:
* Some movies lost a lot of money otherwise, the distribution looks vaguely normal but the movies that lost a lot of money are making the STD very high
* Data issue, the gross figures are only profits in the US. Wikipedia states on Princess Mononoke: Princess Mononoke was the highest-grossing Japanese film of 1997, earning ¥11.3 billion in distribution receipts. It became the highest-grossing film in Japan until it was surpassed by Titanic several months later. The film earned a domestic total of ¥19.3 billion. It was the top-grossing anime film in the United States in January 2001, but despite this the film did not fare as well financially in the country when released in December 1997. It grossed 2,298,191 dollars for the first eight weeks. The IBDB database has 2,298,191 for it's gross. We will need to remove all of the non-US titles. Luckily we didn't lose too much data. 3074 vs. 3891. The data becomes very normal after removing non-US titles!
* Many of the features show pareto like distributions including: all facebook like features, all number of reviews features, movie budget, and movie gross,
* A data issue, there are 762 different kinds of genres. This is because each combination of features for example 'Action|Adventure|Fantasy|Sci-Fi'. I need to see if there is a way to seperate out these different genres, and allow movies to belong to different combinations of genres.
* Tags has the same issue as above but there are likely too many tags even when seperated to be useful.
* For some reason in the color category ' Black and White' has a space at the beginning
* We ought to combine all of the content ratings that are not G, PG, PG13, and R into an 'Other' category
* There are only 16 non-english movies, this feature should be removed to avoid overfitting
* The highest correlation of movie gross is with budget of the movie and with the number of reviews either users or critics, however the number of reviews is not something that we would have before a movie comes out so is of limited predictive value. We'll want to try predicting with and without these number of review features. The next highest correlations are with social media likes (for the movie and for the actors / directors), and with IMDB score; our estimated profit feature is not much correlated with anything but gross and IMDB score, our estimated profitability is not correlated with anything in the dataset.
* Our STD after removing duplicates, non-USA movies, and movies with no gross or budget information is 68,779,390; so we'll see if we can predict movies to within 17 M.


```python
# Importing the IMBD 5000 Database
initial_df = pd.read_csv('movie_metadata.csv')
initial_df.head()
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
      <th>color</th>
      <th>director_name</th>
      <th>num_critic_for_reviews</th>
      <th>duration</th>
      <th>director_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>actor_2_name</th>
      <th>actor_1_facebook_likes</th>
      <th>gross</th>
      <th>genres</th>
      <th>actor_1_name</th>
      <th>movie_title</th>
      <th>num_voted_users</th>
      <th>cast_total_facebook_likes</th>
      <th>actor_3_name</th>
      <th>facenumber_in_poster</th>
      <th>plot_keywords</th>
      <th>movie_imdb_link</th>
      <th>num_user_for_reviews</th>
      <th>language</th>
      <th>country</th>
      <th>content_rating</th>
      <th>budget</th>
      <th>title_year</th>
      <th>actor_2_facebook_likes</th>
      <th>imdb_score</th>
      <th>aspect_ratio</th>
      <th>movie_facebook_likes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Color</td>
      <td>James Cameron</td>
      <td>723.0</td>
      <td>178.0</td>
      <td>0.0</td>
      <td>855.0</td>
      <td>Joel David Moore</td>
      <td>1000.0</td>
      <td>760505847.0</td>
      <td>Action|Adventure|Fantasy|Sci-Fi</td>
      <td>CCH Pounder</td>
      <td>Avatar</td>
      <td>886204</td>
      <td>4834</td>
      <td>Wes Studi</td>
      <td>0.0</td>
      <td>avatar|future|marine|native|paraplegic</td>
      <td>http://www.imdb.com/title/tt0499549/?ref_=fn_t...</td>
      <td>3054.0</td>
      <td>English</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>237000000.0</td>
      <td>2009.0</td>
      <td>936.0</td>
      <td>7.9</td>
      <td>1.78</td>
      <td>33000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Color</td>
      <td>Gore Verbinski</td>
      <td>302.0</td>
      <td>169.0</td>
      <td>563.0</td>
      <td>1000.0</td>
      <td>Orlando Bloom</td>
      <td>40000.0</td>
      <td>309404152.0</td>
      <td>Action|Adventure|Fantasy</td>
      <td>Johnny Depp</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>471220</td>
      <td>48350</td>
      <td>Jack Davenport</td>
      <td>0.0</td>
      <td>goddess|marriage ceremony|marriage proposal|pi...</td>
      <td>http://www.imdb.com/title/tt0449088/?ref_=fn_t...</td>
      <td>1238.0</td>
      <td>English</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>300000000.0</td>
      <td>2007.0</td>
      <td>5000.0</td>
      <td>7.1</td>
      <td>2.35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Color</td>
      <td>Sam Mendes</td>
      <td>602.0</td>
      <td>148.0</td>
      <td>0.0</td>
      <td>161.0</td>
      <td>Rory Kinnear</td>
      <td>11000.0</td>
      <td>200074175.0</td>
      <td>Action|Adventure|Thriller</td>
      <td>Christoph Waltz</td>
      <td>Spectre</td>
      <td>275868</td>
      <td>11700</td>
      <td>Stephanie Sigman</td>
      <td>1.0</td>
      <td>bomb|espionage|sequel|spy|terrorist</td>
      <td>http://www.imdb.com/title/tt2379713/?ref_=fn_t...</td>
      <td>994.0</td>
      <td>English</td>
      <td>UK</td>
      <td>PG-13</td>
      <td>245000000.0</td>
      <td>2015.0</td>
      <td>393.0</td>
      <td>6.8</td>
      <td>2.35</td>
      <td>85000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Color</td>
      <td>Christopher Nolan</td>
      <td>813.0</td>
      <td>164.0</td>
      <td>22000.0</td>
      <td>23000.0</td>
      <td>Christian Bale</td>
      <td>27000.0</td>
      <td>448130642.0</td>
      <td>Action|Thriller</td>
      <td>Tom Hardy</td>
      <td>The Dark Knight Rises</td>
      <td>1144337</td>
      <td>106759</td>
      <td>Joseph Gordon-Levitt</td>
      <td>0.0</td>
      <td>deception|imprisonment|lawlessness|police offi...</td>
      <td>http://www.imdb.com/title/tt1345836/?ref_=fn_t...</td>
      <td>2701.0</td>
      <td>English</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>250000000.0</td>
      <td>2012.0</td>
      <td>23000.0</td>
      <td>8.5</td>
      <td>2.35</td>
      <td>164000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Doug Walker</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>131.0</td>
      <td>NaN</td>
      <td>Rob Walker</td>
      <td>131.0</td>
      <td>NaN</td>
      <td>Documentary</td>
      <td>Doug Walker</td>
      <td>Star Wars: Episode VII - The Force Awakens    ...</td>
      <td>8</td>
      <td>143</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>http://www.imdb.com/title/tt5289954/?ref_=fn_t...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>7.1</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for empty features
initial_df.isnull().sum()
```




    color                         19
    director_name                104
    num_critic_for_reviews        50
    duration                      15
    director_facebook_likes      104
    actor_3_facebook_likes        23
    actor_2_name                  13
    actor_1_facebook_likes         7
    gross                        884
    genres                         0
    actor_1_name                   7
    movie_title                    0
    num_voted_users                0
    cast_total_facebook_likes      0
    actor_3_name                  23
    facenumber_in_poster          13
    plot_keywords                153
    movie_imdb_link                0
    num_user_for_reviews          21
    language                      12
    country                        5
    content_rating               303
    budget                       492
    title_year                   108
    actor_2_facebook_likes        13
    imdb_score                     0
    aspect_ratio                 329
    movie_facebook_likes           0
    dtype: int64




```python
# Remove entries with null gross and null budget, since profit margin cannot be calculcated on those movies
initial_df.dropna(subset=['gross', 'budget'], inplace=True)
initial_df.isnull().sum()
```




    color                         2
    director_name                 0
    num_critic_for_reviews        1
    duration                      1
    director_facebook_likes       0
    actor_3_facebook_likes       10
    actor_2_name                  5
    actor_1_facebook_likes        3
    gross                         0
    genres                        0
    actor_1_name                  3
    movie_title                   0
    num_voted_users               0
    cast_total_facebook_likes     0
    actor_3_name                 10
    facenumber_in_poster          6
    plot_keywords                31
    movie_imdb_link               0
    num_user_for_reviews          0
    language                      3
    country                       0
    content_rating               51
    budget                        0
    title_year                    0
    actor_2_facebook_likes        5
    imdb_score                    0
    aspect_ratio                 75
    movie_facebook_likes          0
    dtype: int64




```python
# Drop duplications
initial_df.drop_duplicates()
initial_df.shape
```




    (3891, 28)




```python
# Create a feature estimating total budget
initial_df['total_budget'] = initial_df.budget*2

# Create a feature estimating profit
initial_df['profit'] = (initial_df.gross - initial_df.total_budget)

initial_df.profit.describe()
```




    count    3.891000e+03
    mean    -3.936556e+07
    std      4.431208e+08
    min     -2.442880e+10
    25%     -4.815942e+07
    50%     -1.499937e+07
    75%      2.001106e+06
    max      4.389357e+08
    Name: profit, dtype: float64




```python
sns.violinplot(x=initial_df.profit)
plt.xlim(-4.389357e+08, 4.389357e+08)
```

    /anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval





    (-438935700.0, 438935700.0)




![png](output_12_2.png)



```python
initial_df[initial_df.profit < -4.389357e+08]
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
      <th>color</th>
      <th>director_name</th>
      <th>num_critic_for_reviews</th>
      <th>duration</th>
      <th>director_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>actor_2_name</th>
      <th>actor_1_facebook_likes</th>
      <th>gross</th>
      <th>genres</th>
      <th>actor_1_name</th>
      <th>movie_title</th>
      <th>num_voted_users</th>
      <th>cast_total_facebook_likes</th>
      <th>actor_3_name</th>
      <th>facenumber_in_poster</th>
      <th>plot_keywords</th>
      <th>movie_imdb_link</th>
      <th>num_user_for_reviews</th>
      <th>language</th>
      <th>country</th>
      <th>content_rating</th>
      <th>budget</th>
      <th>title_year</th>
      <th>actor_2_facebook_likes</th>
      <th>imdb_score</th>
      <th>aspect_ratio</th>
      <th>movie_facebook_likes</th>
      <th>total_budget</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Color</td>
      <td>Andrew Stanton</td>
      <td>462.0</td>
      <td>132.0</td>
      <td>475.0</td>
      <td>530.0</td>
      <td>Samantha Morton</td>
      <td>640.0</td>
      <td>73058679.0</td>
      <td>Action|Adventure|Sci-Fi</td>
      <td>Daryl Sabara</td>
      <td>John Carter</td>
      <td>212204</td>
      <td>1873</td>
      <td>Polly Walker</td>
      <td>1.0</td>
      <td>alien|american civil war|male nipple|mars|prin...</td>
      <td>http://www.imdb.com/title/tt0401729/?ref_=fn_t...</td>
      <td>738.0</td>
      <td>English</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>2.637000e+08</td>
      <td>2012.0</td>
      <td>632.0</td>
      <td>6.6</td>
      <td>2.35</td>
      <td>24000</td>
      <td>5.274000e+08</td>
      <td>-4.543413e+08</td>
    </tr>
    <tr>
      <th>1016</th>
      <td>Color</td>
      <td>Luc Besson</td>
      <td>111.0</td>
      <td>158.0</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>David Bailie</td>
      <td>51.0</td>
      <td>14131298.0</td>
      <td>Adventure|Biography|Drama|History|War</td>
      <td>Paul Brooke</td>
      <td>The Messenger: The Story of Joan of Arc</td>
      <td>55889</td>
      <td>144</td>
      <td>Rab Affleck</td>
      <td>0.0</td>
      <td>cathedral|dauphin|france|trial|wartime rape</td>
      <td>http://www.imdb.com/title/tt0151137/?ref_=fn_t...</td>
      <td>390.0</td>
      <td>English</td>
      <td>France</td>
      <td>R</td>
      <td>3.900000e+08</td>
      <td>1999.0</td>
      <td>40.0</td>
      <td>6.4</td>
      <td>2.35</td>
      <td>0</td>
      <td>7.800000e+08</td>
      <td>-7.658687e+08</td>
    </tr>
    <tr>
      <th>1338</th>
      <td>Color</td>
      <td>John Woo</td>
      <td>160.0</td>
      <td>150.0</td>
      <td>610.0</td>
      <td>478.0</td>
      <td>Tony Chiu Wai Leung</td>
      <td>755.0</td>
      <td>626809.0</td>
      <td>Action|Adventure|Drama|History|War</td>
      <td>Takeshi Kaneshiro</td>
      <td>Red Cliff</td>
      <td>36894</td>
      <td>2172</td>
      <td>Wei Zhao</td>
      <td>4.0</td>
      <td>alliance|battle|china|chinese|strategy</td>
      <td>http://www.imdb.com/title/tt0425637/?ref_=fn_t...</td>
      <td>105.0</td>
      <td>Mandarin</td>
      <td>China</td>
      <td>R</td>
      <td>5.536320e+08</td>
      <td>2008.0</td>
      <td>643.0</td>
      <td>7.4</td>
      <td>2.35</td>
      <td>0</td>
      <td>1.107264e+09</td>
      <td>-1.106637e+09</td>
    </tr>
    <tr>
      <th>2323</th>
      <td>Color</td>
      <td>Hayao Miyazaki</td>
      <td>174.0</td>
      <td>134.0</td>
      <td>6000.0</td>
      <td>745.0</td>
      <td>Jada Pinkett Smith</td>
      <td>893.0</td>
      <td>2298191.0</td>
      <td>Adventure|Animation|Fantasy</td>
      <td>Minnie Driver</td>
      <td>Princess Mononoke</td>
      <td>221552</td>
      <td>2710</td>
      <td>Billy Crudup</td>
      <td>0.0</td>
      <td>anime|cult film|forest|princess|studio ghibli</td>
      <td>http://www.imdb.com/title/tt0119698/?ref_=fn_t...</td>
      <td>570.0</td>
      <td>Japanese</td>
      <td>Japan</td>
      <td>PG-13</td>
      <td>2.400000e+09</td>
      <td>1997.0</td>
      <td>851.0</td>
      <td>8.4</td>
      <td>1.85</td>
      <td>11000</td>
      <td>4.800000e+09</td>
      <td>-4.797702e+09</td>
    </tr>
    <tr>
      <th>2334</th>
      <td>Color</td>
      <td>Katsuhiro Ôtomo</td>
      <td>105.0</td>
      <td>103.0</td>
      <td>78.0</td>
      <td>101.0</td>
      <td>Robin Atkin Downes</td>
      <td>488.0</td>
      <td>410388.0</td>
      <td>Action|Adventure|Animation|Family|Sci-Fi|Thriller</td>
      <td>William Hootkins</td>
      <td>Steamboy</td>
      <td>13727</td>
      <td>991</td>
      <td>Rosalind Ayres</td>
      <td>1.0</td>
      <td>19th century|ball|boy|inventor|steam</td>
      <td>http://www.imdb.com/title/tt0348121/?ref_=fn_t...</td>
      <td>79.0</td>
      <td>Japanese</td>
      <td>Japan</td>
      <td>PG-13</td>
      <td>2.127520e+09</td>
      <td>2004.0</td>
      <td>336.0</td>
      <td>6.9</td>
      <td>1.85</td>
      <td>973</td>
      <td>4.255040e+09</td>
      <td>-4.254629e+09</td>
    </tr>
    <tr>
      <th>2740</th>
      <td>Color</td>
      <td>Tony Jaa</td>
      <td>110.0</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>Petchtai Wongkamlao</td>
      <td>64.0</td>
      <td>102055.0</td>
      <td>Action</td>
      <td>Nirut Sirichanya</td>
      <td>Ong-bak 2</td>
      <td>24570</td>
      <td>134</td>
      <td>Sarunyu Wongkrachang</td>
      <td>0.0</td>
      <td>cult film|elephant|jungle|martial arts|stylize...</td>
      <td>http://www.imdb.com/title/tt0785035/?ref_=fn_t...</td>
      <td>72.0</td>
      <td>Thai</td>
      <td>Thailand</td>
      <td>R</td>
      <td>3.000000e+08</td>
      <td>2008.0</td>
      <td>45.0</td>
      <td>6.2</td>
      <td>2.35</td>
      <td>0</td>
      <td>6.000000e+08</td>
      <td>-5.998979e+08</td>
    </tr>
    <tr>
      <th>2988</th>
      <td>Color</td>
      <td>Joon-ho Bong</td>
      <td>363.0</td>
      <td>110.0</td>
      <td>584.0</td>
      <td>74.0</td>
      <td>Kang-ho Song</td>
      <td>629.0</td>
      <td>2201412.0</td>
      <td>Comedy|Drama|Horror|Sci-Fi</td>
      <td>Doona Bae</td>
      <td>The Host</td>
      <td>68883</td>
      <td>1173</td>
      <td>Ah-sung Ko</td>
      <td>0.0</td>
      <td>daughter|han river|monster|river|seoul</td>
      <td>http://www.imdb.com/title/tt0468492/?ref_=fn_t...</td>
      <td>279.0</td>
      <td>Korean</td>
      <td>South Korea</td>
      <td>R</td>
      <td>1.221550e+10</td>
      <td>2006.0</td>
      <td>398.0</td>
      <td>7.0</td>
      <td>1.85</td>
      <td>7000</td>
      <td>2.443100e+10</td>
      <td>-2.442880e+10</td>
    </tr>
    <tr>
      <th>3005</th>
      <td>Color</td>
      <td>Lajos Koltai</td>
      <td>73.0</td>
      <td>134.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Péter Fancsikai</td>
      <td>9.0</td>
      <td>195888.0</td>
      <td>Drama|Romance|War</td>
      <td>Marcell Nagy</td>
      <td>Fateless</td>
      <td>5603</td>
      <td>11</td>
      <td>Bálint Péntek</td>
      <td>0.0</td>
      <td>bus|death|gay slur|hatred|jewish</td>
      <td>http://www.imdb.com/title/tt0367082/?ref_=fn_t...</td>
      <td>45.0</td>
      <td>Hungarian</td>
      <td>Hungary</td>
      <td>R</td>
      <td>2.500000e+09</td>
      <td>2005.0</td>
      <td>2.0</td>
      <td>7.1</td>
      <td>2.35</td>
      <td>607</td>
      <td>5.000000e+09</td>
      <td>-4.999804e+09</td>
    </tr>
    <tr>
      <th>3075</th>
      <td>Color</td>
      <td>Karan Johar</td>
      <td>20.0</td>
      <td>193.0</td>
      <td>160.0</td>
      <td>860.0</td>
      <td>John Abraham</td>
      <td>8000.0</td>
      <td>3275443.0</td>
      <td>Drama</td>
      <td>Shah Rukh Khan</td>
      <td>Kabhi Alvida Naa Kehna</td>
      <td>13998</td>
      <td>10822</td>
      <td>Preity Zinta</td>
      <td>2.0</td>
      <td>extramarital affair|fashion magazine editor|ma...</td>
      <td>http://www.imdb.com/title/tt0449999/?ref_=fn_t...</td>
      <td>264.0</td>
      <td>Hindi</td>
      <td>India</td>
      <td>R</td>
      <td>7.000000e+08</td>
      <td>2006.0</td>
      <td>1000.0</td>
      <td>6.0</td>
      <td>2.35</td>
      <td>659</td>
      <td>1.400000e+09</td>
      <td>-1.396725e+09</td>
    </tr>
    <tr>
      <th>3273</th>
      <td>Color</td>
      <td>Anurag Basu</td>
      <td>41.0</td>
      <td>90.0</td>
      <td>116.0</td>
      <td>303.0</td>
      <td>Steven Michael Quezada</td>
      <td>594.0</td>
      <td>1602466.0</td>
      <td>Action|Drama|Romance|Thriller</td>
      <td>Bárbara Mori</td>
      <td>Kites</td>
      <td>9673</td>
      <td>1836</td>
      <td>Kabir Bedi</td>
      <td>0.0</td>
      <td>casino|desert|love|suicide|tragic event</td>
      <td>http://www.imdb.com/title/tt1198101/?ref_=fn_t...</td>
      <td>106.0</td>
      <td>English</td>
      <td>India</td>
      <td>NaN</td>
      <td>6.000000e+08</td>
      <td>2010.0</td>
      <td>412.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1.200000e+09</td>
      <td>-1.198398e+09</td>
    </tr>
    <tr>
      <th>3311</th>
      <td>Color</td>
      <td>Chatrichalerm Yukol</td>
      <td>31.0</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Chatchai Plengpanich</td>
      <td>7.0</td>
      <td>454255.0</td>
      <td>Action|Adventure|Drama|History|War</td>
      <td>Sarunyu Wongkrachang</td>
      <td>The Legend of Suriyothai</td>
      <td>1666</td>
      <td>32</td>
      <td>Mai Charoenpura</td>
      <td>3.0</td>
      <td>16th century|burmese|invasion|queen|thailand</td>
      <td>http://www.imdb.com/title/tt0290879/?ref_=fn_t...</td>
      <td>47.0</td>
      <td>Thai</td>
      <td>Thailand</td>
      <td>R</td>
      <td>4.000000e+08</td>
      <td>2001.0</td>
      <td>6.0</td>
      <td>6.6</td>
      <td>1.85</td>
      <td>124</td>
      <td>8.000000e+08</td>
      <td>-7.995457e+08</td>
    </tr>
    <tr>
      <th>3423</th>
      <td>Color</td>
      <td>Katsuhiro Ôtomo</td>
      <td>150.0</td>
      <td>124.0</td>
      <td>78.0</td>
      <td>4.0</td>
      <td>Takeshi Kusao</td>
      <td>6.0</td>
      <td>439162.0</td>
      <td>Action|Animation|Sci-Fi</td>
      <td>Mitsuo Iwata</td>
      <td>Akira</td>
      <td>106160</td>
      <td>28</td>
      <td>Tesshô Genda</td>
      <td>0.0</td>
      <td>based on manga|biker gang|gifted child|post th...</td>
      <td>http://www.imdb.com/title/tt0094625/?ref_=fn_t...</td>
      <td>430.0</td>
      <td>Japanese</td>
      <td>Japan</td>
      <td>R</td>
      <td>1.100000e+09</td>
      <td>1988.0</td>
      <td>5.0</td>
      <td>8.1</td>
      <td>1.85</td>
      <td>0</td>
      <td>2.200000e+09</td>
      <td>-2.199561e+09</td>
    </tr>
    <tr>
      <th>3851</th>
      <td>Color</td>
      <td>Carlos Saura</td>
      <td>35.0</td>
      <td>115.0</td>
      <td>98.0</td>
      <td>4.0</td>
      <td>Juan Luis Galiardo</td>
      <td>341.0</td>
      <td>1687311.0</td>
      <td>Drama|Musical</td>
      <td>Mía Maestro</td>
      <td>Tango</td>
      <td>2412</td>
      <td>371</td>
      <td>Miguel Ángel Solá</td>
      <td>3.0</td>
      <td>dancer|director|love|musical filmmaking|tango</td>
      <td>http://www.imdb.com/title/tt0120274/?ref_=fn_t...</td>
      <td>40.0</td>
      <td>Spanish</td>
      <td>Spain</td>
      <td>PG-13</td>
      <td>7.000000e+08</td>
      <td>1998.0</td>
      <td>26.0</td>
      <td>7.2</td>
      <td>2.00</td>
      <td>539</td>
      <td>1.400000e+09</td>
      <td>-1.398313e+09</td>
    </tr>
    <tr>
      <th>3859</th>
      <td>Color</td>
      <td>Chan-wook Park</td>
      <td>202.0</td>
      <td>112.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>Yeong-ae Lee</td>
      <td>717.0</td>
      <td>211667.0</td>
      <td>Crime|Drama</td>
      <td>Min-sik Choi</td>
      <td>Lady Vengeance</td>
      <td>53508</td>
      <td>907</td>
      <td>Hye-jeong Kang</td>
      <td>0.0</td>
      <td>cake|christian|lesbian sex|oral sex|pregnant s...</td>
      <td>http://www.imdb.com/title/tt0451094/?ref_=fn_t...</td>
      <td>131.0</td>
      <td>Korean</td>
      <td>South Korea</td>
      <td>R</td>
      <td>4.200000e+09</td>
      <td>2005.0</td>
      <td>126.0</td>
      <td>7.7</td>
      <td>2.35</td>
      <td>4000</td>
      <td>8.400000e+09</td>
      <td>-8.399788e+09</td>
    </tr>
    <tr>
      <th>4542</th>
      <td>Color</td>
      <td>Takao Okawara</td>
      <td>107.0</td>
      <td>99.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Naomi Nishida</td>
      <td>43.0</td>
      <td>10037390.0</td>
      <td>Action|Adventure|Drama|Sci-Fi|Thriller</td>
      <td>Hiroshi Abe</td>
      <td>Godzilla 2000</td>
      <td>5442</td>
      <td>53</td>
      <td>Sakae Kimura</td>
      <td>0.0</td>
      <td>godzilla|kaiju|monster|orga|ufo</td>
      <td>http://www.imdb.com/title/tt0188640/?ref_=fn_t...</td>
      <td>140.0</td>
      <td>Japanese</td>
      <td>Japan</td>
      <td>PG</td>
      <td>1.000000e+09</td>
      <td>1999.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>2.35</td>
      <td>339</td>
      <td>2.000000e+09</td>
      <td>-1.989963e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a profitability feature
initial_df['profitability'] = initial_df.profit/initial_df.total_budget
initial_df.profitability.describe()
```




    count    3891.000000
    mean        2.126874
    std        64.811208
    min        -0.999991
    25%        -0.774477
    50%        -0.464672
    75%         0.114270
    max      3596.242767
    Name: profitability, dtype: float64




```python
initial_df.country.unique()
```




    array(['USA', 'UK', 'New Zealand', 'Canada', 'Australia', 'Germany',
           'China', 'New Line', 'France', 'Japan', 'Spain', 'Hong Kong',
           'Czech Republic', 'Peru', 'South Korea', 'India', 'Aruba',
           'Denmark', 'Belgium', 'Ireland', 'South Africa', 'Italy',
           'Romania', 'Chile', 'Netherlands', 'Hungary', 'Russia', 'Mexico',
           'Greece', 'Taiwan', 'Official site', 'Thailand', 'Iran',
           'West Germany', 'Georgia', 'Iceland', 'Brazil', 'Finland',
           'Norway', 'Argentina', 'Colombia', 'Poland', 'Israel', 'Indonesia',
           'Afghanistan', 'Sweden', 'Philippines'], dtype=object)




```python
initial_df=initial_df[initial_df.country=='USA']
print(initial_df.country.unique())
print(initial_df.shape)
```

    ['USA']
    (3074, 31)



```python
print(initial_df.profit.describe())
sns.violinplot(x=initial_df.profit)
```

    count    3.074000e+03
    mean    -2.277299e+07
    std      6.877939e+07
    min     -4.543413e+08
    25%     -4.814359e+07
    50%     -1.372147e+07
    75%      4.430857e+06
    max      4.389357e+08
    Name: profit, dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x1a16cc17b8>




![png](output_17_2.png)



```python
initial_df.hist(figsize=(9,9))
plt.show()
```


![png](output_18_0.png)



```python
initial_df.describe()
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
      <th>num_critic_for_reviews</th>
      <th>duration</th>
      <th>director_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>actor_1_facebook_likes</th>
      <th>gross</th>
      <th>num_voted_users</th>
      <th>cast_total_facebook_likes</th>
      <th>facenumber_in_poster</th>
      <th>num_user_for_reviews</th>
      <th>budget</th>
      <th>title_year</th>
      <th>actor_2_facebook_likes</th>
      <th>imdb_score</th>
      <th>aspect_ratio</th>
      <th>movie_facebook_likes</th>
      <th>total_budget</th>
      <th>profit</th>
      <th>profitability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3073.000000</td>
      <td>3074.000000</td>
      <td>3074.000000</td>
      <td>3069.000000</td>
      <td>3073.000000</td>
      <td>3.074000e+03</td>
      <td>3.074000e+03</td>
      <td>3074.000000</td>
      <td>3068.000000</td>
      <td>3074.000000</td>
      <td>3.074000e+03</td>
      <td>3074.000000</td>
      <td>3072.000000</td>
      <td>3074.000000</td>
      <td>3016.000000</td>
      <td>3074.000000</td>
      <td>3.074000e+03</td>
      <td>3.074000e+03</td>
      <td>3074.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>163.213798</td>
      <td>109.348081</td>
      <td>902.680547</td>
      <td>830.920495</td>
      <td>8197.561341</td>
      <td>5.728945e+07</td>
      <td>1.075269e+05</td>
      <td>12264.236825</td>
      <td>1.420795</td>
      <td>333.592062</td>
      <td>4.003122e+07</td>
      <td>2003.022121</td>
      <td>2164.829102</td>
      <td>6.385947</td>
      <td>2.100368</td>
      <td>9324.176643</td>
      <td>8.006243e+07</td>
      <td>-2.277299e+07</td>
      <td>2.720936</td>
    </tr>
    <tr>
      <th>std</th>
      <td>125.215125</td>
      <td>22.122647</td>
      <td>3318.949966</td>
      <td>1992.130817</td>
      <td>16673.921347</td>
      <td>7.275710e+07</td>
      <td>1.576255e+05</td>
      <td>20370.534286</td>
      <td>2.136960</td>
      <td>410.223499</td>
      <td>4.379910e+07</td>
      <td>10.007002</td>
      <td>4792.751633</td>
      <td>1.052057</td>
      <td>0.372138</td>
      <td>21746.579013</td>
      <td>8.759821e+07</td>
      <td>6.877939e+07</td>
      <td>72.892644</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>34.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.030000e+02</td>
      <td>5.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.180000e+02</td>
      <td>1920.000000</td>
      <td>0.000000</td>
      <td>1.600000</td>
      <td>1.180000</td>
      <td>0.000000</td>
      <td>4.360000e+02</td>
      <td>-4.543413e+08</td>
      <td>-0.999907</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>72.000000</td>
      <td>95.000000</td>
      <td>11.000000</td>
      <td>229.000000</td>
      <td>799.000000</td>
      <td>1.141309e+07</td>
      <td>1.846150e+04</td>
      <td>2171.500000</td>
      <td>0.000000</td>
      <td>106.000000</td>
      <td>1.000000e+07</td>
      <td>1999.000000</td>
      <td>427.000000</td>
      <td>5.800000</td>
      <td>1.850000</td>
      <td>0.000000</td>
      <td>2.000000e+07</td>
      <td>-4.814359e+07</td>
      <td>-0.718644</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>133.000000</td>
      <td>105.000000</td>
      <td>60.500000</td>
      <td>466.000000</td>
      <td>2000.000000</td>
      <td>3.379975e+07</td>
      <td>5.409850e+04</td>
      <td>4479.000000</td>
      <td>1.000000</td>
      <td>207.000000</td>
      <td>2.500000e+07</td>
      <td>2004.000000</td>
      <td>726.000000</td>
      <td>6.500000</td>
      <td>2.350000</td>
      <td>249.000000</td>
      <td>5.000000e+07</td>
      <td>-1.372147e+07</td>
      <td>-0.396683</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>221.000000</td>
      <td>119.000000</td>
      <td>234.750000</td>
      <td>723.000000</td>
      <td>13000.000000</td>
      <td>7.486365e+07</td>
      <td>1.305638e+05</td>
      <td>16800.000000</td>
      <td>2.000000</td>
      <td>397.000000</td>
      <td>5.475000e+07</td>
      <td>2010.000000</td>
      <td>1000.000000</td>
      <td>7.100000</td>
      <td>2.350000</td>
      <td>11000.000000</td>
      <td>1.095000e+08</td>
      <td>4.430857e+06</td>
      <td>0.192774</td>
    </tr>
    <tr>
      <th>max</th>
      <td>813.000000</td>
      <td>330.000000</td>
      <td>23000.000000</td>
      <td>23000.000000</td>
      <td>640000.000000</td>
      <td>7.605058e+08</td>
      <td>1.689764e+06</td>
      <td>656730.000000</td>
      <td>43.000000</td>
      <td>4667.000000</td>
      <td>3.000000e+08</td>
      <td>2016.000000</td>
      <td>137000.000000</td>
      <td>9.300000</td>
      <td>16.000000</td>
      <td>349000.000000</td>
      <td>6.000000e+08</td>
      <td>4.389357e+08</td>
      <td>3596.242767</td>
    </tr>
  </tbody>
</table>
</div>




```python
initial_df.dtypes
```




    color                         object
    director_name                 object
    num_critic_for_reviews       float64
    duration                     float64
    director_facebook_likes      float64
    actor_3_facebook_likes       float64
    actor_2_name                  object
    actor_1_facebook_likes       float64
    gross                        float64
    genres                        object
    actor_1_name                  object
    movie_title                   object
    num_voted_users                int64
    cast_total_facebook_likes      int64
    actor_3_name                  object
    facenumber_in_poster         float64
    plot_keywords                 object
    movie_imdb_link               object
    num_user_for_reviews         float64
    language                      object
    country                       object
    content_rating                object
    budget                       float64
    title_year                   float64
    actor_2_facebook_likes       float64
    imdb_score                   float64
    aspect_ratio                 float64
    movie_facebook_likes           int64
    total_budget                 float64
    profit                       float64
    profitability                float64
    dtype: object




```python
initial_df.describe(include=['object'])
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
      <th>color</th>
      <th>director_name</th>
      <th>actor_2_name</th>
      <th>genres</th>
      <th>actor_1_name</th>
      <th>movie_title</th>
      <th>actor_3_name</th>
      <th>plot_keywords</th>
      <th>movie_imdb_link</th>
      <th>language</th>
      <th>country</th>
      <th>content_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3073</td>
      <td>3074</td>
      <td>3072</td>
      <td>3074</td>
      <td>3073</td>
      <td>3074</td>
      <td>3069</td>
      <td>3055</td>
      <td>3074</td>
      <td>3071</td>
      <td>3074</td>
      <td>3052</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>1419</td>
      <td>1821</td>
      <td>656</td>
      <td>1185</td>
      <td>2993</td>
      <td>2153</td>
      <td>2974</td>
      <td>2993</td>
      <td>11</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Color</td>
      <td>Steven Spielberg</td>
      <td>Morgan Freeman</td>
      <td>Comedy</td>
      <td>Robert De Niro</td>
      <td>Halloween</td>
      <td>Anne Hathaway</td>
      <td>eighteen wheeler|illegal street racing|truck|t...</td>
      <td>http://www.imdb.com/title/tt1976009/?ref_=fn_t...</td>
      <td>English</td>
      <td>USA</td>
      <td>R</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2983</td>
      <td>23</td>
      <td>16</td>
      <td>138</td>
      <td>38</td>
      <td>3</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>3055</td>
      <td>3074</td>
      <td>1334</td>
    </tr>
  </tbody>
</table>
</div>




```python
initial_df.genres.unique()
```




    array(['Action|Adventure|Fantasy|Sci-Fi', 'Action|Adventure|Fantasy',
           'Action|Thriller', 'Action|Adventure|Sci-Fi',
           'Action|Adventure|Romance',
           'Adventure|Animation|Comedy|Family|Fantasy|Musical|Romance',
           'Action|Adventure|Western', 'Action|Adventure|Family|Fantasy',
           'Action|Adventure|Comedy|Family|Fantasy|Sci-Fi',
           'Action|Adventure|Drama|History', 'Adventure|Fantasy',
           'Adventure|Family|Fantasy', 'Drama|Romance',
           'Action|Adventure|Sci-Fi|Thriller',
           'Action|Adventure|Fantasy|Romance',
           'Adventure|Animation|Comedy|Family|Fantasy',
           'Adventure|Animation|Comedy|Family|Sport', 'Action|Crime|Thriller',
           'Action|Adventure|Horror|Sci-Fi|Thriller',
           'Action|Adventure|Fantasy|Sci-Fi|Thriller',
           'Adventure|Animation|Family|Sci-Fi',
           'Action|Comedy|Crime|Thriller', 'Animation|Drama|Family|Fantasy',
           'Action|Adventure|Drama|Romance', 'Action|Crime|Drama|Thriller',
           'Adventure|Animation|Comedy|Family',
           'Action|Adventure|Animation|Comedy|Family|Sci-Fi',
           'Adventure|Drama|Family|Mystery', 'Action|Comedy|Sci-Fi|Western',
           'Action|Adventure|Fantasy|Horror|Thriller',
           'Action|Adventure|Comedy|Sci-Fi', 'Comedy|Family|Fantasy',
           'Adventure|Animation|Comedy|Drama|Family|Fantasy',
           'Action|Adventure|Drama|Fantasy',
           'Action|Adventure|Family|Fantasy|Romance',
           'Action|Adventure|Drama|Sci-Fi',
           'Action|Adventure|Family|Mystery|Sci-Fi',
           'Action|Adventure|Animation|Comedy|Drama|Family|Sci-Fi',
           'Adventure|Animation|Comedy|Family|Sci-Fi',
           'Adventure|Animation|Family|Fantasy', 'Action|Sci-Fi',
           'Adventure|Drama|Sci-Fi', 'Drama|Fantasy|Romance',
           'Adventure|Sci-Fi', 'Action|Adventure|Drama|Thriller',
           'Action|Drama|History|Romance|War', 'Action|Drama',
           'Drama|Horror|Sci-Fi', 'Adventure|Comedy|Family|Fantasy',
           'Animation|Comedy|Family|Fantasy', 'Action|Adventure',
           'Action|Adventure|Animation|Comedy|Family',
           'Adventure|Animation|Comedy|Family|Fantasy|Musical',
           'Mystery|Thriller', 'Adventure|Animation|Comedy|Drama|Family',
           'Action|Adventure|Animation|Comedy|Family|Fantasy|Sci-Fi',
           'Comedy|Fantasy|Horror', 'Drama|Fantasy|Horror|Thriller',
           'Action|Adventure|Thriller', 'Action|Drama|Thriller', 'Adventure',
           'Action|Comedy|Fantasy|Sci-Fi',
           'Action|Adventure|Comedy|Family|Fantasy|Mystery|Sci-Fi',
           'Action|Adventure|Animation|Fantasy', 'Comedy|Crime',
           'Action|Drama|History|War',
           'Action|Adventure|Animation|Comedy|Family|Fantasy',
           'Action|Drama|Mystery|Sci-Fi',
           'Action|Adventure|Animation|Fantasy|Romance|Sci-Fi',
           'Action|Adventure|Drama|History|War',
           'Adventure|Drama|Fantasy|Romance',
           'Adventure|Drama|Thriller|Western',
           'Adventure|Animation|Comedy|Family|Western',
           'Action|Mystery|Thriller', 'Adventure|Sci-Fi|Thriller',
           'Adventure|Animation|Comedy|Family|Fantasy|Sci-Fi',
           'Action|Crime|Mystery|Thriller', 'Action|Adventure|Drama',
           'Action|Adventure|Family|Mystery',
           'Adventure|Animation|Family|Thriller',
           'Action|Animation|Comedy|Family|Sci-Fi', 'Action|Comedy|Fantasy',
           'Adventure|Animation|Comedy|Family|Musical',
           'Action|Adventure|Crime|Mystery|Thriller',
           'Action|Adventure|History', 'Action', 'Adventure|Drama|Fantasy',
           'Action|Fantasy|Thriller', 'Action|Adventure|Comedy|Crime',
           'Adventure|Mystery|Sci-Fi', 'Action|Drama|Sci-Fi|Thriller',
           'Action|Crime|Sci-Fi|Thriller', 'Action|Family|Sport',
           'Comedy|Drama|Romance', 'Action|Comedy|Romance',
           'Action|Adventure|Mystery|Sci-Fi', 'Action|Drama|War',
           'Action|Adventure|Comedy|Family|Fantasy', 'Crime|Thriller',
           'Action|Comedy|Crime|Romance|Thriller', 'Biography|Drama',
           'Action|Comedy|Crime|Sci-Fi|Thriller', 'Action|Drama|Fantasy|War',
           'Animation|Comedy|Family|Music|Western',
           'Action|Adventure|Mystery|Sci-Fi|Thriller',
           'Action|Drama|Sci-Fi|Sport', 'Action|Crime|Romance|Thriller',
           'Action|Adventure|Comedy', 'Biography|Drama|Sport',
           'Action|Mystery|Sci-Fi|Thriller',
           'Animation|Family|Fantasy|Musical|Romance',
           'Action|Adventure|Romance|Sci-Fi|Thriller', 'Action|Drama|Romance',
           'Biography|Crime|Drama|History|Romance', 'Biography|Crime|Drama',
           'Action|Comedy|Thriller', 'Comedy|Romance', 'Action|Comedy|Crime',
           'Action|Drama|Mystery|Thriller', 'Drama|Western',
           'Animation|Drama|Family|Musical|Romance',
           'Action|Adventure|Comedy|Family|Mystery',
           'Action|Romance|Thriller', 'Action|Fantasy|Horror|Mystery',
           'Biography|Comedy|Crime|Drama', 'Action|Sci-Fi|War',
           'Action|Adventure|Animation|Family|Fantasy',
           'Action|Crime|Fantasy|Romance|Thriller', 'Adventure|Comedy|Sci-Fi',
           'Action|Crime|Sport|Thriller',
           'Action|Adventure|Biography|Drama|History|Thriller',
           'Action|Comedy|Sci-Fi', 'Drama|Mystery|Thriller',
           'Action|Adventure|Fantasy|Thriller', 'Crime|Drama',
           'Animation|Comedy|Family|Sport', 'Comedy|Sci-Fi|Thriller',
           'Drama|History|War', 'Comedy',
           'Adventure|Animation|Comedy|Family|Romance',
           'Drama|Family|Fantasy|Romance', 'Drama|Fantasy|Thriller',
           'Drama|History|War|Western', 'Action|Adventure|Animation|Family',
           'Adventure|Comedy|Family|Mystery|Sci-Fi',
           'Drama|Fantasy|Horror|Mystery|Thriller',
           'Animation|Comedy|Family|Sci-Fi',
           'Adventure|Comedy|Drama|Fantasy|Romance',
           'Action|Adventure|Comedy|Crime|Thriller', 'Crime|Drama|Thriller',
           'Adventure|Animation|Family|Fantasy|Musical|War', 'Action|Comedy',
           'Crime|Drama|Mystery|Thriller',
           'Action|Adventure|Animation|Family|Fantasy|Sci-Fi',
           'Adventure|Animation|Comedy|Family|Fantasy|Music',
           'Drama|History|Thriller|War', 'Action|Animation|Comedy|Sci-Fi',
           'Comedy|Family|Fantasy|Horror|Mystery',
           'Drama|Mystery|Sci-Fi|Thriller', 'Action|Horror|Sci-Fi|Thriller',
           'Action|Adventure|Comedy|Crime|Mystery|Thriller',
           'Comedy|Drama|Sci-Fi', 'Drama|History|Sport',
           'Adventure|Drama|Romance', 'Animation|Comedy|Family|Music|Romance',
           'Animation|Comedy|Family|Fantasy|Musical|Romance',
           'Adventure|Comedy|Family', 'Action|Crime|Drama|Mystery|Thriller',
           'Action|Adventure|Comedy|Thriller',
           'Action|Adventure|Comedy|Fantasy',
           'Adventure|Comedy|Drama|Family|Fantasy',
           'Action|Comedy|Fantasy|Romance', 'Comedy|Romance|Sci-Fi',
           'Adventure|Comedy|Mystery', 'Comedy|Drama|Fantasy|Romance',
           'Action|Comedy|Family|Fantasy',
           'Action|Adventure|Fantasy|Horror|Sci-Fi',
           'Crime|Drama|History|Mystery|Thriller', 'Comedy|Drama',
           'Adventure|Drama|Sci-Fi|Thriller',
           'Adventure|Animation|Comedy|Drama|Family|Fantasy|Sci-Fi',
           'Action|Drama|Romance|Sci-Fi|Thriller', 'Comedy|Crime|Sport',
           'Comedy|Family|Fantasy|Romance',
           'Adventure|Drama|History|Romance|War', 'Comedy|Family|Sci-Fi',
           'Fantasy|Horror|Mystery|Thriller',
           'Adventure|Animation|Comedy|Family|Fantasy|Sci-Fi|Sport',
           'Adventure|Comedy|Crime|Family|Mystery', 'Drama|Sci-Fi|Thriller',
           'Action|Crime|Mystery|Romance|Thriller',
           'Action|Adventure|Comedy|Romance',
           'Adventure|Animation|Family|Western', 'Comedy|Family|Romance',
           'Action|Adventure|Family|Sci-Fi|Thriller',
           'Animation|Family|Fantasy|Music',
           'Action|Adventure|Family|Fantasy|Thriller', 'Comedy|Fantasy',
           'Action|Adventure|Comedy|Fantasy|Thriller',
           'Action|Sci-Fi|Thriller', 'Drama|History|Thriller',
           'Adventure|Animation|Family', 'Drama|Musical|Romance',
           'Adventure|Animation|Drama|Family|Musical',
           'Animation|Comedy|Family|Fantasy|Sci-Fi', 'Sci-Fi|Thriller',
           'Action|Crime|Fantasy|Thriller',
           'Comedy|Drama|Family|Music|Musical|Romance',
           'Horror|Mystery|Thriller', 'Action|Adventure|Comedy|Family|Sci-Fi',
           'Crime|Mystery|Thriller', 'Comedy|Family',
           'Biography|Comedy|Drama|History', 'Drama|Music|Musical',
           'Comedy|Crime|Music', 'Action|Drama|Thriller|War',
           'Action|Comedy|Romance|Thriller',
           'Animation|Comedy|Family|Fantasy|Mystery',
           'Comedy|Crime|Drama|Romance', 'Action|Adventure|Romance|Thriller',
           'Drama|History|Romance', 'Action|Drama|Fantasy|Romance',
           'Action|Adventure|Animation|Family|Sci-Fi',
           'Animation|Comedy|Fantasy', 'Action|Fantasy',
           'Action|Animation|Comedy|Family',
           'Action|Adventure|Comedy|Romance|Thriller', 'Action|Comedy|Sport',
           'Biography|Drama|History|War', 'Adventure|Animation|Comedy',
           'Action|Drama|Sport', 'Drama|Mystery|Romance|Thriller',
           'Adventure|Animation|Comedy|Family|Fantasy|Romance',
           'Adventure|Drama|War', 'Action|Adventure|Crime|Thriller',
           'Fantasy|Mystery|Romance|Sci-Fi|Thriller',
           'Drama|Fantasy|Mystery|Thriller',
           'Animation|Comedy|Family|Fantasy|Music',
           'Drama|Horror|Romance|Thriller', 'Drama|War', 'Drama',
           'Action|Drama|Fantasy|Horror|War',
           'Adventure|Biography|Drama|History|War', 'Action|Fantasy|Horror',
           'Comedy|Drama|Musical|Romance', 'Action|Sci-Fi|Sport',
           'Action|Adventure|Animation|Comedy|Crime|Family|Fantasy',
           'Adventure|Animation|Family|Fantasy|Musical',
           'Action|Crime|Mystery|Sci-Fi|Thriller', 'Animation|Comedy|Family',
           'Action|Comedy|Crime|Drama|Thriller', 'Biography|Drama|Thriller',
           'Comedy|Fantasy|Romance', 'Drama|Horror|Sci-Fi|Thriller',
           'Adventure|Drama|History',
           'Action|Adventure|Comedy|Romance|Thriller|Western',
           'Biography|Drama|Sport|War', 'Comedy|Drama|Family|Musical',
           'Action|Adventure|Fantasy|Horror|Sci-Fi|Thriller', 'Drama|Sport',
           'Drama|Mystery|Romance', 'Adventure|Biography|Crime|Drama|Western',
           'Comedy|Romance|Sport', 'Crime|Drama|Mystery|Thriller|Western',
           'Comedy|Sport', 'Comedy|Drama|Family',
           'Crime|Drama|Fantasy|Mystery',
           'Adventure|Animation|Biography|Drama|Family|Fantasy|Musical',
           'Drama|Romance|Western', 'Documentary|Music', 'Drama|Thriller',
           'Animation|Family|Fantasy', 'Action|Fantasy|Horror|Sci-Fi',
           'Biography|Comedy|Drama', 'Action|Horror|Sci-Fi',
           'Adventure|Comedy', 'Biography|Drama|History|Sport',
           'Comedy|Crime|Romance|Thriller', 'Comedy|Crime|Romance',
           'Drama|Fantasy|Sport', 'Adventure|Comedy|Drama|Music',
           'Action|Fantasy|Horror|Sci-Fi|Thriller',
           'Adventure|Animation|Comedy|Drama|Family|Fantasy|Romance',
           'Drama|Fantasy|Mystery|Romance|Thriller', 'Horror|Sci-Fi|Thriller',
           'Drama|Fantasy|Mystery|Romance', 'Fantasy|Horror|Mystery|Romance',
           'Adventure|Comedy|Family|Romance|Sci-Fi', 'Drama|Horror|Thriller',
           'Adventure|Drama|Romance|War',
           'Action|Adventure|Comedy|Romance|Sci-Fi',
           'Action|Biography|Drama|History|Thriller|War',
           'Adventure|Comedy|Family|Fantasy|Horror',
           'Comedy|Family|Romance|Sci-Fi', 'Action|Adventure|Thriller|War',
           'Comedy|Drama|Romance|Sport', 'Action|Comedy|Crime|Drama',
           'Action|Crime',
           'Adventure|Animation|Drama|Family|History|Musical|Romance',
           'Action|Adventure|Drama|Romance|Sci-Fi',
           'Action|Adventure|Comedy|Family|Romance',
           'Action|Adventure|Comedy|Western',
           'Biography|Drama|History|Musical',
           'Adventure|Drama|Horror|Thriller', 'Action|Drama|Sport|Thriller',
           'Comedy|Drama|Family|Fantasy',
           'Adventure|Comedy|Crime|Family|Musical',
           'Drama|Music|Musical|Romance', 'Action|Adventure|Romance|Sci-Fi',
           'Adventure|Animation|Drama|Family|Fantasy|Musical|Mystery|Romance',
           'Action|Horror|Thriller', 'Drama|History|Horror',
           'Drama|Romance|Sport', 'Comedy|Family|Musical|Romance',
           'Romance|Sci-Fi|Thriller', 'Biography|Comedy|Drama|Romance',
           'Mystery|Sci-Fi|Thriller', 'Drama|Fantasy|Horror',
           'Adventure|Comedy|Drama|Fantasy|Musical',
           'Action|Adventure|Family|Fantasy|Sci-Fi|Thriller',
           'Adventure|Comedy|Family|Fantasy|Romance|Sport',
           'Crime|Drama|Romance|Thriller', 'Adventure|Drama|Thriller',
           'Comedy|Crime|Drama|Thriller', 'Drama|Fantasy',
           'Adventure|Comedy|Drama', 'Action|Biography|Drama|History|War',
           'Adventure|Comedy|Fantasy', 'Adventure|Comedy|Crime|Drama|Family',
           'Comedy|Sci-Fi', 'Action|Adventure|Comedy|Crime|Music|Mystery',
           'Action|Crime|Drama|Sci-Fi|Thriller',
           'Action|Adventure|Comedy|Drama|War', 'Drama|Mystery|Sci-Fi',
           'Crime|Drama|Music', 'Adventure|Crime|Drama|Western',
           'Comedy|Drama|Thriller',
           'Action|Comedy|Crime|Music|Romance|Thriller',
           'Crime|Romance|Thriller', 'Action|Adventure|Drama|Sci-Fi|Thriller',
           'Action|Drama|Fantasy|Thriller|Western',
           'Action|Drama|Mystery|Thriller|War', 'Action|Comedy|Crime|Romance',
           'Action|Adventure|Family|Fantasy|Sci-Fi',
           'Adventure|Comedy|Family|Musical', 'Action|Horror',
           'Action|Adventure|Horror|Thriller', 'Comedy|Drama|Music|Romance',
           'Comedy|Family|Romance|Sport', 'Drama|Family|Fantasy',
           'Drama|Fantasy|Musical|Romance',
           'Adventure|Comedy|Family|Fantasy|Sci-Fi', 'Comedy|Musical',
           'Biography|Drama|History', 'Action|Adventure|Horror|Sci-Fi',
           'Action|Crime|Drama|Thriller|War', 'Comedy|Crime|Thriller',
           'Biography|Drama|History|Thriller',
           'Action|Adventure|Crime|Drama|Mystery|Thriller',
           'Animation|Family|Fantasy|Musical', 'Adventure|Drama|Western',
           'Drama|Horror|Mystery|Thriller', 'Drama|Music',
           'Action|Drama|Family|Sport', 'Action|Biography|Drama|Thriller|War',
           'Comedy|Drama|Sport', 'Horror|Mystery',
           'Adventure|Comedy|Sci-Fi|Western', 'Fantasy|Horror|Romance',
           'Biography|Drama|Romance', 'Adventure|Drama|Family',
           'Action|Adventure|Drama|Romance|War', 'Biography|Drama|Music',
           'Adventure|Comedy|Crime|Romance',
           'Comedy|Drama|Family|Fantasy|Romance', 'Horror',
           'Biography|Drama|Music|Musical', 'Drama|History', 'Comedy|Western',
           'Action|Adventure|Crime|Fantasy|Mystery|Thriller',
           'Adventure|Drama|Mystery', 'Biography|Crime|Drama|Music',
           'Crime|Drama|Horror|Thriller', 'Horror|Thriller',
           'Adventure|Animation|Comedy|Drama|Family|Fantasy|Musical',
           'Biography|Drama|History|Romance',
           'Action|Adventure|Comedy|Music|Thriller',
           'Adventure|Animation|Comedy|Crime|Family',
           'Comedy|Romance|Sci-Fi|Thriller', 'Comedy|Crime|Family|Romance',
           'Adventure|Animation|Comedy|Fantasy|Romance',
           'Adventure|Comedy|Drama|Romance|Thriller|War',
           'Action|Drama|Fantasy', 'Drama|Fantasy|Romance|Sci-Fi',
           'Animation|Comedy|Family|Horror|Sci-Fi',
           'Biography|Drama|Romance|Sport', 'Adventure|Drama',
           'Horror|Mystery|Sci-Fi', 'Action|Adventure|Drama|Thriller|Western',
           'Adventure|Family|Fantasy|Sci-Fi', 'Drama|Family',
           'Action|Adventure|Crime|Drama|Family|Fantasy|Romance|Thriller',
           'Biography|Comedy|Romance', 'Drama|Romance|War',
           'Adventure|Comedy|Family|Sci-Fi',
           'Biography|Drama|Family|History|Sport',
           'Biography|Comedy|Drama|History|Music', 'Fantasy|Horror',
           'Comedy|Drama|Romance|Sci-Fi', 'Action|Comedy|Sci-Fi|Thriller',
           'Comedy|Horror', 'Drama|Thriller|War', 'Comedy|Music',
           'Action|Western', 'Action|Adventure|Family|Sci-Fi',
           'Adventure|Biography|Drama|Thriller', 'Drama|Romance|War|Western',
           'Drama|Sci-Fi', 'Action|Adventure|Comedy|Drama|Thriller',
           'Drama|Music|Romance', 'Crime|Horror|Mystery|Thriller',
           'Comedy|Drama|Fantasy', 'Drama|Family|Sport',
           'Animation|Comedy|Family|Fantasy|Musical',
           'Adventure|Comedy|Crime|Drama', 'Drama|Mystery',
           'Comedy|Fantasy|Horror|Thriller',
           'Crime|Drama|Mystery|Sci-Fi|Thriller', 'Comedy|Crime|Musical',
           'Comedy|Drama|Family|Music|Romance', 'Comedy|Horror|Romance',
           'Comedy|Family|Fantasy|Sport',
           'Animation|Drama|Family|Fantasy|Musical|Romance',
           'Comedy|Horror|Musical|Sci-Fi', 'Crime|Drama|Sport',
           'Action|Adventure|Animation|Drama|Mystery|Sci-Fi|Thriller',
           'Action|Adventure|Crime|Drama|Romance', 'Action|Comedy|Horror',
           'Adventure|Horror|Thriller',
           'Biography|Crime|Drama|History|Western',
           'Action|Biography|Crime|Drama', 'Biography|Drama|Music|Romance',
           'Biography|Crime|Drama|History|Music',
           'Adventure|Animation|Comedy|Drama|Family|Musical',
           'Comedy|Drama|Music', 'Drama|Romance|Thriller',
           'Action|Fantasy|Horror|Thriller', 'Adventure|Biography',
           'Action|Comedy|Family', 'Action|Horror|Romance',
           'Action|Comedy|Crime|Music', 'Action|Crime|Drama|History|Western',
           'Comedy|Crime|Drama', 'Comedy|Family|Fantasy|Music|Romance',
           'Action|Adventure|Comedy|Sci-Fi|Thriller',
           'Action|Crime|Drama|Western',
           'Action|Adventure|Comedy|Family|Romance|Sci-Fi',
           'Comedy|Crime|Mystery|Romance', 'Adventure|Family',
           'Comedy|Drama|Family|Romance', 'Action|Drama|Music|Romance',
           'Adventure|Comedy|Family|Fantasy|Horror|Mystery',
           'Action|Biography|Drama|History|Romance|Western',
           'Adventure|Drama|Family|Fantasy', 'Biography|Drama|Family',
           'Action|Adventure|Comedy|Crime|Family|Romance|Thriller',
           'Action|Comedy|Music', 'Fantasy|Horror|Thriller',
           'Comedy|Drama|Romance|Thriller', 'Adventure|Comedy|Family|Romance',
           'Crime|Drama|History', 'Drama|Music|Mystery|Romance|Thriller',
           'Action|Adventure|Fantasy|Horror', 'Action|Drama|Western',
           'Comedy|War', 'Adventure|Comedy|Family|Fantasy|Music|Sci-Fi',
           'Adventure|Family|Fantasy|Music|Musical',
           'Adventure|Comedy|Horror|Sci-Fi', 'Horror|Sci-Fi',
           'Drama|Romance|Sci-Fi', 'Biography|Comedy|Drama|Family|Sport',
           'Action|Crime|Drama|Thriller|Western',
           'Drama|Fantasy|Romance|Thriller', 'Comedy|Family|Sport',
           'Comedy|Drama|Musical|Romance|War',
           'Animation|Comedy|Fantasy|Musical', 'Action|Comedy|Documentary',
           'Adventure|Comedy|Drama|Family|Romance',
           'Adventure|Comedy|Drama|Family|Mystery',
           'Drama|Family|Music|Romance', 'Fantasy|Romance',
           'Adventure|Animation|Family|Musical',
           'Animation|Comedy|Drama|Family|Musical',
           'Biography|Crime|Drama|History',
           'Adventure|Comedy|Fantasy|Music|Sci-Fi',
           'Action|Adventure|Drama|Mystery',
           'Comedy|Crime|Family|Mystery|Romance|Thriller',
           'Adventure|Crime|Mystery|Sci-Fi|Thriller',
           'Adventure|Biography|Drama', 'Crime|Fantasy|Horror',
           'Animation|Family|Fantasy|Mystery', 'Comedy|Family|Music|Musical',
           'Action|Crime|Drama|Sport', 'Comedy|Musical|Romance',
           'Comedy|Crime|Drama|Mystery|Romance',
           'Animation|Drama|Mystery|Sci-Fi|Thriller',
           'Adventure|Comedy|Drama|Romance', 'Biography|Comedy|Musical',
           'Adventure|Family|Fantasy|Musical',
           'Action|Adventure|Drama|History|Romance',
           'Action|Comedy|Crime|Fantasy|Horror|Mystery|Sci-Fi|Thriller',
           'Crime|Drama|Mystery', 'Adventure|Comedy|Romance',
           'Family|Fantasy|Music', 'Action|Comedy|Crime|Fantasy',
           'Action|Drama|Fantasy|Mystery|Thriller',
           'Biography|Drama|History|Music', 'Biography|Drama|Family|Sport',
           'Adventure|Fantasy|Mystery|Thriller',
           'Action|Horror|Romance|Sci-Fi|Thriller',
           'Action|Animation|Comedy|Crime|Family',
           'Drama|Family|Music|Musical', 'Drama|Family|Musical|Romance',
           'Comedy|Drama|Family|Fantasy|Sci-Fi', 'Comedy|Music|Romance',
           'Adventure|Comedy|Family|Fantasy|Musical',
           'Action|Comedy|Crime|Family', 'Comedy|Crime|Mystery',
           'Action|Crime|Sci-Fi', 'Drama|Family|Romance',
           'Adventure|Comedy|Family|Music|Romance', 'Crime|Horror|Thriller',
           'Comedy|Horror|Thriller', 'Adventure|Family|Fantasy|Romance',
           'Adventure|Fantasy|Horror|Mystery|Thriller',
           'Action|Crime|Drama|Romance|Thriller',
           'Crime|Drama|Musical|Romance', 'Family|Music|Romance',
           'Biography|Drama|History|Thriller|War', 'Fantasy|Horror|Mystery',
           'Comedy|Drama|Fantasy|Music|Romance', 'Biography|Drama|War',
           'Adventure|Comedy|Family|Sport', 'Comedy|Fantasy|Horror|Romance',
           'Adventure|Mystery|Thriller', 'Comedy|Mystery|Romance',
           'Comedy|Drama|War', 'Biography|Comedy|Drama|War', 'Comedy|Mystery',
           'Comedy|Crime|Family|Sci-Fi', 'Adventure|Family|Sci-Fi',
           'Adventure|Comedy|Romance|Sci-Fi',
           'Action|Adventure|Comedy|Fantasy|Mystery',
           'Action|Adventure|Drama|Thriller|War', 'Crime|Drama|Music|Romance',
           'Adventure|Animation|Comedy|Crime',
           'Comedy|Drama|Family|Fantasy|Musical', 'Comedy|Crime|Family',
           'Crime|Drama|Romance', 'Drama|Fantasy|Music|Romance',
           'Family|Sci-Fi', 'Drama|History|Romance|Western',
           'Action|Comedy|War', 'Comedy|Drama|Family|Sport',
           'Adventure|Comedy|Music|Sci-Fi',
           'Animation|Comedy|Family|Mystery|Sci-Fi',
           'Action|Comedy|Drama|Music', 'Adventure|Comedy|Drama|Fantasy',
           'Fantasy|Horror|Sci-Fi', 'Comedy|Horror|Sci-Fi',
           'Biography|Crime|Drama|Romance',
           'Adventure|Comedy|Drama|Romance|Sci-Fi', 'Action|Crime|Drama',
           'Adventure|Biography|Drama|War', 'Documentary|Drama',
           'Horror|Musical|Sci-Fi', 'Biography|Drama|Family|Musical|Romance',
           'Comedy|Crime|Drama|Romance|Thriller',
           'Animation|Comedy|Drama|Romance', 'Comedy|Crime|Musical|Mystery',
           'Crime|Horror', 'Western', 'Mystery|Romance|Thriller',
           'Adventure|Comedy|Drama|Family', 'Musical|Romance',
           'Documentary|Drama|War', 'Biography|Crime|Drama|Western',
           'Comedy|Family|Fantasy|Musical',
           'Crime|Drama|Musical|Romance|Thriller',
           'Fantasy|Horror|Romance|Thriller', 'Comedy|Drama|Musical',
           'Comedy|Drama|Music|Musical', 'Action|Sport',
           'Action|Comedy|Drama|Thriller',
           'Drama|Horror|Mystery|Sci-Fi|Thriller', 'Comedy|Documentary',
           'Biography|Crime|Drama|Thriller', 'Adventure|Horror',
           'Horror|Mystery|Sci-Fi|Thriller',
           'Comedy|Crime|Drama|Mystery|Thriller',
           'Biography|Crime|Drama|Mystery|Thriller',
           'Crime|Horror|Music|Thriller', 'Crime|Thriller|War',
           'Drama|Horror', 'Crime|Drama|Fantasy|Romance', 'Documentary',
           'Drama|History|Romance|War', 'Adventure|War|Western',
           'Biography|Comedy|Musical|Romance|Western',
           'Adventure|Comedy|Musical|Romance',
           'Action|Adventure|Comedy|Musical', 'Biography|Comedy|Documentary',
           'Comedy|Documentary|Music', 'Crime|Drama|History|Romance',
           'Comedy|Drama|Horror', 'Drama|Family|Western',
           'Comedy|Crime|Drama|Sci-Fi', 'Comedy|Family|Musical|Romance|Short',
           'Comedy|Documentary|War', 'Drama|Romance|Sci-Fi|Thriller',
           'Drama|Musical', 'Action|Comedy|Horror|Sci-Fi', 'Documentary|War',
           'Biography|Documentary|History',
           'Action|Adventure|Comedy|Drama|Music|Sci-Fi',
           'Crime|Drama|Film-Noir|Mystery|Thriller',
           'Comedy|Fantasy|Musical|Sci-Fi', 'Action|Biography|Drama|Sport',
           'Adventure|Comedy|Sport', 'Action|Drama|Horror|Thriller',
           'Crime|Documentary|War', 'Comedy|Fantasy|Thriller',
           'Drama|Fantasy|Sci-Fi', 'Action|Adventure|Drama|War',
           'Action|Adventure|Animation|Comedy|Fantasy|Sci-Fi',
           'Documentary|Drama|Sport', 'Documentary|History|Music',
           'Adventure|Family|Romance',
           'Adventure|Biography|Drama|Horror|Thriller',
           'Biography|Documentary|Sport', 'Documentary|Sport',
           'Action|Biography|Documentary|Sport', 'Comedy|Horror|Musical',
           'Comedy|Fantasy|Horror|Musical', 'Biography|Documentary',
           'Action|Fantasy|Horror|Mystery|Thriller', 'Thriller',
           'Animation|Comedy|Drama|Fantasy|Sci-Fi', 'Sci-Fi',
           'Adventure|Horror|Sci-Fi', 'Crime|Documentary',
           'Adventure|Documentary', 'Comedy|Crime|Drama|Horror|Thriller',
           'Comedy|Documentary|Drama', 'Romance', 'Comedy|Crime|Horror'],
          dtype=object)




```python
sns.countplot(y=initial_df.color)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a16c39048>




![png](output_23_1.png)



```python
initial_df.color.unique()
```




    array(['Color', ' Black and White', nan], dtype=object)




```python
initial_df[initial_df.color==' Black and White'].shape
```




    (90, 31)




```python
sns.countplot(y='content_rating', data=initial_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a17a78b38>




![png](output_26_1.png)



```python
initial_df[initial_df.content_rating=='Not Rated'].shape
```




    (19, 31)




```python
initial_df.content_rating.replace(to_replace=['Approved', 'X', 'Not Rated', 'M', 'Unrated', 'Passed', 'NC-17'], value='Other', inplace=True)
sns.countplot(y='content_rating', data=initial_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a179d72e8>




![png](output_28_1.png)



```python
initial_df[initial_df.language!='English']
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
      <th>color</th>
      <th>director_name</th>
      <th>num_critic_for_reviews</th>
      <th>duration</th>
      <th>director_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>actor_2_name</th>
      <th>actor_1_facebook_likes</th>
      <th>gross</th>
      <th>genres</th>
      <th>actor_1_name</th>
      <th>movie_title</th>
      <th>num_voted_users</th>
      <th>cast_total_facebook_likes</th>
      <th>actor_3_name</th>
      <th>facenumber_in_poster</th>
      <th>plot_keywords</th>
      <th>movie_imdb_link</th>
      <th>num_user_for_reviews</th>
      <th>language</th>
      <th>country</th>
      <th>content_rating</th>
      <th>budget</th>
      <th>title_year</th>
      <th>actor_2_facebook_likes</th>
      <th>imdb_score</th>
      <th>aspect_ratio</th>
      <th>movie_facebook_likes</th>
      <th>total_budget</th>
      <th>profit</th>
      <th>profitability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>484</th>
      <td>Color</td>
      <td>Martin Campbell</td>
      <td>137.0</td>
      <td>129.0</td>
      <td>258.0</td>
      <td>163.0</td>
      <td>Nick Chinlund</td>
      <td>2000.0</td>
      <td>45356386.0</td>
      <td>Action|Adventure|Western</td>
      <td>Michael Emerson</td>
      <td>The Legend of Zorro</td>
      <td>71574</td>
      <td>2864</td>
      <td>Adrian Alonso</td>
      <td>1.0</td>
      <td>california|fight|hero|mask|zorro</td>
      <td>http://www.imdb.com/title/tt0386140/?ref_=fn_t...</td>
      <td>244.0</td>
      <td>Spanish</td>
      <td>USA</td>
      <td>PG</td>
      <td>75000000.0</td>
      <td>2005.0</td>
      <td>277.0</td>
      <td>5.9</td>
      <td>2.35</td>
      <td>951</td>
      <td>150000000.0</td>
      <td>-104643614.0</td>
      <td>-0.697624</td>
    </tr>
    <tr>
      <th>811</th>
      <td>Black and White</td>
      <td>John Dahl</td>
      <td>81.0</td>
      <td>132.0</td>
      <td>131.0</td>
      <td>242.0</td>
      <td>Clayne Crawford</td>
      <td>11000.0</td>
      <td>10166502.0</td>
      <td>Action|Drama|War</td>
      <td>James Franco</td>
      <td>The Great Raid</td>
      <td>18209</td>
      <td>12133</td>
      <td>Paolo Montalban</td>
      <td>0.0</td>
      <td>american|lieutenant colonel|mission|rescue|sol...</td>
      <td>http://www.imdb.com/title/tt0326905/?ref_=fn_t...</td>
      <td>183.0</td>
      <td>Filipino</td>
      <td>USA</td>
      <td>R</td>
      <td>80000000.0</td>
      <td>2005.0</td>
      <td>298.0</td>
      <td>6.7</td>
      <td>2.35</td>
      <td>0</td>
      <td>160000000.0</td>
      <td>-149833498.0</td>
      <td>-0.936459</td>
    </tr>
    <tr>
      <th>1236</th>
      <td>Color</td>
      <td>Mel Gibson</td>
      <td>283.0</td>
      <td>139.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>Dalia Hernández</td>
      <td>708.0</td>
      <td>50859889.0</td>
      <td>Action|Adventure|Drama|Thriller</td>
      <td>Rudy Youngblood</td>
      <td>Apocalypto</td>
      <td>236000</td>
      <td>848</td>
      <td>Jonathan Brewer</td>
      <td>0.0</td>
      <td>jaguar|mayan|solar eclipse|tribe|village</td>
      <td>http://www.imdb.com/title/tt0472043/?ref_=fn_t...</td>
      <td>1043.0</td>
      <td>Maya</td>
      <td>USA</td>
      <td>R</td>
      <td>40000000.0</td>
      <td>2006.0</td>
      <td>78.0</td>
      <td>7.8</td>
      <td>1.85</td>
      <td>14000</td>
      <td>80000000.0</td>
      <td>-29140111.0</td>
      <td>-0.364251</td>
    </tr>
    <tr>
      <th>1866</th>
      <td>Color</td>
      <td>Mel Gibson</td>
      <td>406.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>113.0</td>
      <td>Maia Morgenstern</td>
      <td>260.0</td>
      <td>499263.0</td>
      <td>Drama</td>
      <td>Christo Jivkov</td>
      <td>The Passion of the Christ</td>
      <td>179235</td>
      <td>705</td>
      <td>Hristo Shopov</td>
      <td>0.0</td>
      <td>anti semitism|cult film|grindhouse|suffering|t...</td>
      <td>http://www.imdb.com/title/tt0335345/?ref_=fn_t...</td>
      <td>2814.0</td>
      <td>Aramaic</td>
      <td>USA</td>
      <td>R</td>
      <td>30000000.0</td>
      <td>2004.0</td>
      <td>252.0</td>
      <td>7.1</td>
      <td>2.35</td>
      <td>13000</td>
      <td>60000000.0</td>
      <td>-59500737.0</td>
      <td>-0.991679</td>
    </tr>
    <tr>
      <th>2259</th>
      <td>Color</td>
      <td>Marc Forster</td>
      <td>201.0</td>
      <td>128.0</td>
      <td>395.0</td>
      <td>161.0</td>
      <td>Shaun Toub</td>
      <td>283.0</td>
      <td>15797907.0</td>
      <td>Drama</td>
      <td>Mustafa Haidari</td>
      <td>The Kite Runner</td>
      <td>68119</td>
      <td>904</td>
      <td>Khalid Abdalla</td>
      <td>0.0</td>
      <td>afghanistan|based on novel|boy|friend|kite</td>
      <td>http://www.imdb.com/title/tt0419887/?ref_=fn_t...</td>
      <td>230.0</td>
      <td>Dari</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>20000000.0</td>
      <td>2007.0</td>
      <td>206.0</td>
      <td>7.6</td>
      <td>2.35</td>
      <td>0</td>
      <td>40000000.0</td>
      <td>-24202093.0</td>
      <td>-0.605052</td>
    </tr>
    <tr>
      <th>2863</th>
      <td>Color</td>
      <td>Clint Eastwood</td>
      <td>251.0</td>
      <td>141.0</td>
      <td>16000.0</td>
      <td>78.0</td>
      <td>Kazunari Ninomiya</td>
      <td>378.0</td>
      <td>13753931.0</td>
      <td>Drama|History|War</td>
      <td>Yuki Matsuzaki</td>
      <td>Letters from Iwo Jima</td>
      <td>132149</td>
      <td>751</td>
      <td>Shidô Nakamura</td>
      <td>0.0</td>
      <td>blood splatter|general|island|japan|world war two</td>
      <td>http://www.imdb.com/title/tt0498380/?ref_=fn_t...</td>
      <td>316.0</td>
      <td>Japanese</td>
      <td>USA</td>
      <td>R</td>
      <td>19000000.0</td>
      <td>2006.0</td>
      <td>85.0</td>
      <td>7.9</td>
      <td>2.35</td>
      <td>5000</td>
      <td>38000000.0</td>
      <td>-24246069.0</td>
      <td>-0.638054</td>
    </tr>
    <tr>
      <th>2890</th>
      <td>Color</td>
      <td>Angelina Jolie Pitt</td>
      <td>110.0</td>
      <td>127.0</td>
      <td>11000.0</td>
      <td>116.0</td>
      <td>Nikola Djuricko</td>
      <td>306.0</td>
      <td>301305.0</td>
      <td>Drama|Romance|War</td>
      <td>Jelena Jovanova</td>
      <td>In the Land of Blood and Honey</td>
      <td>31414</td>
      <td>796</td>
      <td>Branko Djuric</td>
      <td>0.0</td>
      <td>bosnian war|church|emaciation|soldier|violence</td>
      <td>http://www.imdb.com/title/tt1714209/?ref_=fn_t...</td>
      <td>180.0</td>
      <td>Bosnian</td>
      <td>USA</td>
      <td>R</td>
      <td>13000000.0</td>
      <td>2011.0</td>
      <td>164.0</td>
      <td>4.3</td>
      <td>2.35</td>
      <td>0</td>
      <td>26000000.0</td>
      <td>-25698695.0</td>
      <td>-0.988411</td>
    </tr>
    <tr>
      <th>3086</th>
      <td>Color</td>
      <td>Christopher Cain</td>
      <td>43.0</td>
      <td>111.0</td>
      <td>58.0</td>
      <td>258.0</td>
      <td>Taylor Handley</td>
      <td>482.0</td>
      <td>1066555.0</td>
      <td>Drama|History|Romance|Western</td>
      <td>Jon Gries</td>
      <td>September Dawn</td>
      <td>2618</td>
      <td>1526</td>
      <td>Trent Ford</td>
      <td>0.0</td>
      <td>massacre|mormon|settler|utah|wagon train</td>
      <td>http://www.imdb.com/title/tt0473700/?ref_=fn_t...</td>
      <td>111.0</td>
      <td>NaN</td>
      <td>USA</td>
      <td>R</td>
      <td>11000000.0</td>
      <td>2007.0</td>
      <td>362.0</td>
      <td>5.8</td>
      <td>1.85</td>
      <td>411</td>
      <td>22000000.0</td>
      <td>-20933445.0</td>
      <td>-0.951520</td>
    </tr>
    <tr>
      <th>3455</th>
      <td>Color</td>
      <td>Siddharth Anand</td>
      <td>16.0</td>
      <td>153.0</td>
      <td>5.0</td>
      <td>60.0</td>
      <td>Mary Goggin</td>
      <td>532.0</td>
      <td>872643.0</td>
      <td>Comedy|Family|Romance</td>
      <td>Saif Ali Khan</td>
      <td>Ta Ra Rum Pum</td>
      <td>2909</td>
      <td>902</td>
      <td>Vic Aviles</td>
      <td>3.0</td>
      <td>comeback|family relationships|marriage|new yor...</td>
      <td>http://www.imdb.com/title/tt0833553/?ref_=fn_t...</td>
      <td>37.0</td>
      <td>Hindi</td>
      <td>USA</td>
      <td>NaN</td>
      <td>6000000.0</td>
      <td>2007.0</td>
      <td>249.0</td>
      <td>5.4</td>
      <td>NaN</td>
      <td>108</td>
      <td>12000000.0</td>
      <td>-11127357.0</td>
      <td>-0.927280</td>
    </tr>
    <tr>
      <th>3614</th>
      <td>Color</td>
      <td>Matt Piedmont</td>
      <td>133.0</td>
      <td>84.0</td>
      <td>4.0</td>
      <td>546.0</td>
      <td>Adrian Martinez</td>
      <td>8000.0</td>
      <td>5895238.0</td>
      <td>Comedy|Western</td>
      <td>Will Ferrell</td>
      <td>Casa de mi Padre</td>
      <td>17169</td>
      <td>10123</td>
      <td>Luis E. Carazo</td>
      <td>1.0</td>
      <td>absurd humor|drug lord|mexico|ranch|spaghetti ...</td>
      <td>http://www.imdb.com/title/tt1702425/?ref_=fn_t...</td>
      <td>70.0</td>
      <td>Spanish</td>
      <td>USA</td>
      <td>R</td>
      <td>6000000.0</td>
      <td>2012.0</td>
      <td>806.0</td>
      <td>5.5</td>
      <td>2.35</td>
      <td>9000</td>
      <td>12000000.0</td>
      <td>-6104762.0</td>
      <td>-0.508730</td>
    </tr>
    <tr>
      <th>3731</th>
      <td>Color</td>
      <td>Bille Woodruff</td>
      <td>9.0</td>
      <td>106.0</td>
      <td>23.0</td>
      <td>467.0</td>
      <td>Cameron Mills</td>
      <td>1000.0</td>
      <td>17382982.0</td>
      <td>Drama|Thriller</td>
      <td>Boris Kodjoe</td>
      <td>Addicted</td>
      <td>5975</td>
      <td>2840</td>
      <td>Sharon Leal</td>
      <td>0.0</td>
      <td>adultery|attraction|lust|obsession|temptation</td>
      <td>http://www.imdb.com/title/tt2205401/?ref_=fn_t...</td>
      <td>33.0</td>
      <td>Spanish</td>
      <td>USA</td>
      <td>R</td>
      <td>5000000.0</td>
      <td>2014.0</td>
      <td>694.0</td>
      <td>5.2</td>
      <td>1.85</td>
      <td>0</td>
      <td>10000000.0</td>
      <td>7382982.0</td>
      <td>0.738298</td>
    </tr>
    <tr>
      <th>3931</th>
      <td>Color</td>
      <td>Ron Fricke</td>
      <td>115.0</td>
      <td>102.0</td>
      <td>330.0</td>
      <td>0.0</td>
      <td>Balinese Tari Legong Dancers</td>
      <td>48.0</td>
      <td>2601847.0</td>
      <td>Documentary|Music</td>
      <td>Collin Alfredo St. Dic</td>
      <td>Samsara</td>
      <td>22457</td>
      <td>48</td>
      <td>Puti Sri Candra Dewi</td>
      <td>0.0</td>
      <td>hall of mirrors|mont saint michel france|palac...</td>
      <td>http://www.imdb.com/title/tt0770802/?ref_=fn_t...</td>
      <td>69.0</td>
      <td>None</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>4000000.0</td>
      <td>2011.0</td>
      <td>0.0</td>
      <td>8.5</td>
      <td>2.35</td>
      <td>26000</td>
      <td>8000000.0</td>
      <td>-5398153.0</td>
      <td>-0.674769</td>
    </tr>
    <tr>
      <th>4110</th>
      <td>Color</td>
      <td>Michael Landon Jr.</td>
      <td>5.0</td>
      <td>87.0</td>
      <td>84.0</td>
      <td>331.0</td>
      <td>Kevin Gage</td>
      <td>702.0</td>
      <td>252726.0</td>
      <td>Drama|Family|Western</td>
      <td>William Morgan Sheppard</td>
      <td>Love's Abiding Joy</td>
      <td>1289</td>
      <td>2715</td>
      <td>Brianna Brown</td>
      <td>0.0</td>
      <td>19th century|faith|mayor|ranch|sheriff</td>
      <td>http://www.imdb.com/title/tt0785025/?ref_=fn_t...</td>
      <td>18.0</td>
      <td>NaN</td>
      <td>USA</td>
      <td>PG</td>
      <td>3000000.0</td>
      <td>2006.0</td>
      <td>366.0</td>
      <td>7.2</td>
      <td>NaN</td>
      <td>76</td>
      <td>6000000.0</td>
      <td>-5747274.0</td>
      <td>-0.957879</td>
    </tr>
    <tr>
      <th>4207</th>
      <td>Color</td>
      <td>Alex Rivera</td>
      <td>47.0</td>
      <td>90.0</td>
      <td>8.0</td>
      <td>35.0</td>
      <td>Jacob Vargas</td>
      <td>426.0</td>
      <td>75727.0</td>
      <td>Drama|Romance|Sci-Fi|Thriller</td>
      <td>Leonor Varela</td>
      <td>Sleep Dealer</td>
      <td>5699</td>
      <td>862</td>
      <td>Tenoch Huerta</td>
      <td>0.0</td>
      <td>computer|future|mexican immigrant|network|wilh...</td>
      <td>http://www.imdb.com/title/tt0804529/?ref_=fn_t...</td>
      <td>40.0</td>
      <td>Spanish</td>
      <td>USA</td>
      <td>PG-13</td>
      <td>2500000.0</td>
      <td>2008.0</td>
      <td>399.0</td>
      <td>5.9</td>
      <td>1.85</td>
      <td>0</td>
      <td>5000000.0</td>
      <td>-4924273.0</td>
      <td>-0.984855</td>
    </tr>
    <tr>
      <th>4463</th>
      <td>Color</td>
      <td>Ham Tran</td>
      <td>15.0</td>
      <td>135.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>Kieu Chinh</td>
      <td>51.0</td>
      <td>638951.0</td>
      <td>Drama</td>
      <td>Long Nguyen</td>
      <td>Journey from the Fall</td>
      <td>775</td>
      <td>83</td>
      <td>Cat Ly</td>
      <td>2.0</td>
      <td>1970s|1980s|nonlinear timeline|rescue|vietnam war</td>
      <td>http://www.imdb.com/title/tt0433398/?ref_=fn_t...</td>
      <td>19.0</td>
      <td>Vietnamese</td>
      <td>USA</td>
      <td>R</td>
      <td>1592000.0</td>
      <td>2006.0</td>
      <td>24.0</td>
      <td>7.4</td>
      <td>1.85</td>
      <td>100</td>
      <td>3184000.0</td>
      <td>-2545049.0</td>
      <td>-0.799324</td>
    </tr>
    <tr>
      <th>4505</th>
      <td>Color</td>
      <td>Tom Sanchez</td>
      <td>1.0</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Antonio Arrué</td>
      <td>3.0</td>
      <td>3830.0</td>
      <td>Comedy|Drama</td>
      <td>Nataniel Sánchez</td>
      <td>The Knife of Don Juan</td>
      <td>27</td>
      <td>5</td>
      <td>Juan Carlos Montoya</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>http://www.imdb.com/title/tt1349485/?ref_=fn_t...</td>
      <td>1.0</td>
      <td>Spanish</td>
      <td>USA</td>
      <td>NaN</td>
      <td>1200000.0</td>
      <td>2013.0</td>
      <td>2.0</td>
      <td>7.2</td>
      <td>NaN</td>
      <td>75</td>
      <td>2400000.0</td>
      <td>-2396170.0</td>
      <td>-0.998404</td>
    </tr>
    <tr>
      <th>4796</th>
      <td>Color</td>
      <td>Richard Glatzer</td>
      <td>69.0</td>
      <td>90.0</td>
      <td>25.0</td>
      <td>138.0</td>
      <td>Jesse Garcia</td>
      <td>231.0</td>
      <td>1689999.0</td>
      <td>Drama</td>
      <td>Emily Rios</td>
      <td>Quinceañera</td>
      <td>3675</td>
      <td>771</td>
      <td>Alicia Sixtos</td>
      <td>1.0</td>
      <td>15th birthday|birthday|gay|party|security guard</td>
      <td>http://www.imdb.com/title/tt0451176/?ref_=fn_t...</td>
      <td>48.0</td>
      <td>Spanish</td>
      <td>USA</td>
      <td>R</td>
      <td>400000.0</td>
      <td>2006.0</td>
      <td>200.0</td>
      <td>7.1</td>
      <td>2.35</td>
      <td>426</td>
      <td>800000.0</td>
      <td>889999.0</td>
      <td>1.112499</td>
    </tr>
    <tr>
      <th>4958</th>
      <td>Black and White</td>
      <td>Harry F. Millarde</td>
      <td>1.0</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Johnnie Walker</td>
      <td>2.0</td>
      <td>3000000.0</td>
      <td>Crime|Drama</td>
      <td>Stephen Carr</td>
      <td>Over the Hill to the Poorhouse</td>
      <td>5</td>
      <td>4</td>
      <td>Mary Carr</td>
      <td>1.0</td>
      <td>family relationships|gang|idler|poorhouse|thief</td>
      <td>http://www.imdb.com/title/tt0011549/?ref_=fn_t...</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>USA</td>
      <td>NaN</td>
      <td>100000.0</td>
      <td>1920.0</td>
      <td>2.0</td>
      <td>4.8</td>
      <td>1.33</td>
      <td>0</td>
      <td>200000.0</td>
      <td>2800000.0</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>5035</th>
      <td>Color</td>
      <td>Robert Rodriguez</td>
      <td>56.0</td>
      <td>81.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>Peter Marquardt</td>
      <td>121.0</td>
      <td>2040920.0</td>
      <td>Action|Crime|Drama|Romance|Thriller</td>
      <td>Carlos Gallardo</td>
      <td>El Mariachi</td>
      <td>52055</td>
      <td>147</td>
      <td>Consuelo Gómez</td>
      <td>0.0</td>
      <td>assassin|death|guitar|gun|mariachi</td>
      <td>http://www.imdb.com/title/tt0104815/?ref_=fn_t...</td>
      <td>130.0</td>
      <td>Spanish</td>
      <td>USA</td>
      <td>R</td>
      <td>7000.0</td>
      <td>1992.0</td>
      <td>20.0</td>
      <td>6.9</td>
      <td>1.37</td>
      <td>0</td>
      <td>14000.0</td>
      <td>2026920.0</td>
      <td>144.780000</td>
    </tr>
  </tbody>
</table>
</div>




```python
correlations=initial_df.corr()
```


```python
# Increase the figsize to 10 x 9
plt.figure(figsize=(10,9))

# Plot heatmap of correlations
sns.heatmap(correlations, annot=True, cmap='RdBu_r', )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a179d7b38>




![png](output_31_1.png)



```python
sns.violinplot(initial_df.budget)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a170c1fd0>




![png](output_32_1.png)



```python
sns.violinplot(initial_df.gross)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a17666898>




![png](output_33_1.png)


## Data Cleaning

I will create a new DataFrame and clean the data on that by: removing duplicates, removing entries without budget or gross values, addressing missing data, dropping non-US movies (which have incorrect gross values), addressing sparse data.


```python
df = initial_df = pd.read_csv('movie_metadata.csv')
df.shape
```




    (5043, 28)




```python
# Remove duplicates and entries without budget or gross
df.dropna(subset=['gross', 'budget'], inplace=True)
df.drop_duplicates()
print(df.shape)
print(df.isnull().sum())
```

    (3891, 28)
    color                         2
    director_name                 0
    num_critic_for_reviews        1
    duration                      1
    director_facebook_likes       0
    actor_3_facebook_likes       10
    actor_2_name                  5
    actor_1_facebook_likes        3
    gross                         0
    genres                        0
    actor_1_name                  3
    movie_title                   0
    num_voted_users               0
    cast_total_facebook_likes     0
    actor_3_name                 10
    facenumber_in_poster          6
    plot_keywords                31
    movie_imdb_link               0
    num_user_for_reviews          0
    language                      3
    country                       0
    content_rating               51
    budget                        0
    title_year                    0
    actor_2_facebook_likes        5
    imdb_score                    0
    aspect_ratio                 75
    movie_facebook_likes          0
    dtype: int64



```python
# Replace null values of categorical values:
df.color.fillna('Missing', inplace=True)
df.actor_2_name.fillna('Missing', inplace=True)
df.actor_1_name.fillna('Missing', inplace=True)
df.actor_3_name.fillna('Missing', inplace=True)
df.plot_keywords.fillna('Missing', inplace=True)
df.content_rating.fillna('Missing', inplace=True)
df.aspect_ratio.fillna('Missing', inplace=True)
df.language.fillna('Missing', inplace=True)
print(df.isnull().sum())
```

    color                         0
    director_name                 0
    num_critic_for_reviews        1
    duration                      1
    director_facebook_likes       0
    actor_3_facebook_likes       10
    actor_2_name                  0
    actor_1_facebook_likes        3
    gross                         0
    genres                        0
    actor_1_name                  0
    movie_title                   0
    num_voted_users               0
    cast_total_facebook_likes     0
    actor_3_name                  0
    facenumber_in_poster          6
    plot_keywords                 0
    movie_imdb_link               0
    num_user_for_reviews          0
    language                      0
    country                       0
    content_rating                0
    budget                        0
    title_year                    0
    actor_2_facebook_likes        5
    imdb_score                    0
    aspect_ratio                  0
    movie_facebook_likes          0
    dtype: int64



```python
# Fill missing data for numerical features
df['num_critic_for_reviews_missing'] = df.num_critic_for_reviews.isnull().astype(int)
df.num_critic_for_reviews.fillna(0, inplace=True)

df['duration_missing'] = df.duration.isnull().astype(int)
df.duration.fillna(0, inplace=True)

df['actor_1_facebook_likes_missing'] = df.actor_1_facebook_likes.isnull().astype(int)
df.actor_1_facebook_likes.fillna(0, inplace=True)

df['actor_2_facebook_likes_missing'] = df.actor_2_facebook_likes.isnull().astype(int)
df.actor_2_facebook_likes.fillna(0, inplace=True)

df['actor_3_facebook_likes_missing'] = df.actor_3_facebook_likes.isnull().astype(int)
df.actor_3_facebook_likes.fillna(0, inplace=True)

df['facenumber_in_poster_missing'] = df.facenumber_in_poster.isnull().astype(int)
df.facenumber_in_poster.fillna(0, inplace=True)

print(df.isnull().sum())
```

    color                             0
    director_name                     0
    num_critic_for_reviews            0
    duration                          0
    director_facebook_likes           0
    actor_3_facebook_likes            0
    actor_2_name                      0
    actor_1_facebook_likes            0
    gross                             0
    genres                            0
    actor_1_name                      0
    movie_title                       0
    num_voted_users                   0
    cast_total_facebook_likes         0
    actor_3_name                      0
    facenumber_in_poster              0
    plot_keywords                     0
    movie_imdb_link                   0
    num_user_for_reviews              0
    language                          0
    country                           0
    content_rating                    0
    budget                            0
    title_year                        0
    actor_2_facebook_likes            0
    imdb_score                        0
    aspect_ratio                      0
    movie_facebook_likes              0
    num_critic_for_reviews_missing    0
    duration_missing                  0
    actor_1_facebook_likes_missing    0
    actor_2_facebook_likes_missing    0
    actor_3_facebook_likes_missing    0
    facenumber_in_poster_missing      0
    dtype: int64



```python
df.dtypes
```




    color                              object
    director_name                      object
    num_critic_for_reviews            float64
    duration                          float64
    director_facebook_likes           float64
    actor_3_facebook_likes            float64
    actor_2_name                       object
    actor_1_facebook_likes            float64
    gross                             float64
    genres                             object
    actor_1_name                       object
    movie_title                        object
    num_voted_users                     int64
    cast_total_facebook_likes           int64
    actor_3_name                       object
    facenumber_in_poster              float64
    plot_keywords                      object
    movie_imdb_link                    object
    num_user_for_reviews              float64
    language                           object
    country                            object
    content_rating                     object
    budget                            float64
    title_year                        float64
    actor_2_facebook_likes            float64
    imdb_score                        float64
    aspect_ratio                       object
    movie_facebook_likes                int64
    num_critic_for_reviews_missing      int64
    duration_missing                    int64
    actor_1_facebook_likes_missing      int64
    actor_2_facebook_likes_missing      int64
    actor_3_facebook_likes_missing      int64
    facenumber_in_poster_missing        int64
    dtype: object




```python
# Remove any non-US films and also remove country column
df=df[df.country=='USA']
df.drop(columns=['country'], inplace=True)
print(df.shape)
print(df.columns)
```

    (3074, 33)
    Index(['color', 'director_name', 'num_critic_for_reviews', 'duration',
           'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',
           'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name',
           'movie_title', 'num_voted_users', 'cast_total_facebook_likes',
           'actor_3_name', 'facenumber_in_poster', 'plot_keywords',
           'movie_imdb_link', 'num_user_for_reviews', 'language', 'content_rating',
           'budget', 'title_year', 'actor_2_facebook_likes', 'imdb_score',
           'aspect_ratio', 'movie_facebook_likes',
           'num_critic_for_reviews_missing', 'duration_missing',
           'actor_1_facebook_likes_missing', 'actor_2_facebook_likes_missing',
           'actor_3_facebook_likes_missing', 'facenumber_in_poster_missing'],
          dtype='object')


    /anaconda3/lib/python3.7/site-packages/pandas/core/frame.py:3697: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)



```python
# Replace sparse content_rating features with 'Other'
df.content_rating.replace(to_replace=['Approved', 'X', 'Not Rated', 'M', 'Unrated', 'Passed', 'NC-17'], value='Other', inplace=True)
sns.countplot(y='content_rating', data=df)
```

    /anaconda3/lib/python3.7/site-packages/pandas/core/generic.py:5890: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._update_inplace(new_data)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a1841ba58>




![png](output_42_2.png)



```python
# Dropping Language Column because everything besides english is sparse so I don't want this feature to cause overfitting
df.drop(['language'], axis=1, inplace=True)
print(df.shape)
print(df.columns)
```

    (3074, 32)
    Index(['color', 'director_name', 'num_critic_for_reviews', 'duration',
           'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',
           'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name',
           'movie_title', 'num_voted_users', 'cast_total_facebook_likes',
           'actor_3_name', 'facenumber_in_poster', 'plot_keywords',
           'movie_imdb_link', 'num_user_for_reviews', 'content_rating', 'budget',
           'title_year', 'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio',
           'movie_facebook_likes', 'num_critic_for_reviews_missing',
           'duration_missing', 'actor_1_facebook_likes_missing',
           'actor_2_facebook_likes_missing', 'actor_3_facebook_likes_missing',
           'facenumber_in_poster_missing'],
          dtype='object')


## Feature Engineering

Some feature engineering possibilities:
* Need to create dummy features for the categories
* See if I can create movie genre features from the current movie genre's feature which is organized poorly - done
* Possibly see if there is some way to seperate out big budget smaller budget movies - can't think of anything that wouldn't be accounted for already by budget
* Maybe keep the most popular directors and actors, that way we don't increase the dimensionality too much but we keep some actor and director information


```python
# Creating more useful movie genre feature with a list of the genres
df['genres_list'] = df.genres.str.split('|')
df.genres_list.head()
```

    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy






    0    [Action, Adventure, Fantasy, Sci-Fi]
    1            [Action, Adventure, Fantasy]
    3                      [Action, Thriller]
    5             [Action, Adventure, Sci-Fi]
    6            [Action, Adventure, Romance]
    Name: genres_list, dtype: object




```python
s = df['genres_list']

mlb = MultiLabelBinarizer()

genres_list_df = pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=df.index)

genres_list_df.head()
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
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>History</th>
      <th>Horror</th>
      <th>Music</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
abt = pd.concat([df, genres_list_df], axis=1)
abt.head()
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
      <th>color</th>
      <th>director_name</th>
      <th>num_critic_for_reviews</th>
      <th>duration</th>
      <th>director_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>actor_2_name</th>
      <th>actor_1_facebook_likes</th>
      <th>gross</th>
      <th>genres</th>
      <th>actor_1_name</th>
      <th>movie_title</th>
      <th>num_voted_users</th>
      <th>cast_total_facebook_likes</th>
      <th>actor_3_name</th>
      <th>facenumber_in_poster</th>
      <th>plot_keywords</th>
      <th>movie_imdb_link</th>
      <th>num_user_for_reviews</th>
      <th>content_rating</th>
      <th>budget</th>
      <th>title_year</th>
      <th>actor_2_facebook_likes</th>
      <th>imdb_score</th>
      <th>aspect_ratio</th>
      <th>movie_facebook_likes</th>
      <th>num_critic_for_reviews_missing</th>
      <th>duration_missing</th>
      <th>actor_1_facebook_likes_missing</th>
      <th>actor_2_facebook_likes_missing</th>
      <th>actor_3_facebook_likes_missing</th>
      <th>facenumber_in_poster_missing</th>
      <th>genres_list</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>History</th>
      <th>Horror</th>
      <th>Music</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Color</td>
      <td>James Cameron</td>
      <td>723.0</td>
      <td>178.0</td>
      <td>0.0</td>
      <td>855.0</td>
      <td>Joel David Moore</td>
      <td>1000.0</td>
      <td>760505847.0</td>
      <td>Action|Adventure|Fantasy|Sci-Fi</td>
      <td>CCH Pounder</td>
      <td>Avatar</td>
      <td>886204</td>
      <td>4834</td>
      <td>Wes Studi</td>
      <td>0.0</td>
      <td>avatar|future|marine|native|paraplegic</td>
      <td>http://www.imdb.com/title/tt0499549/?ref_=fn_t...</td>
      <td>3054.0</td>
      <td>PG-13</td>
      <td>237000000.0</td>
      <td>2009.0</td>
      <td>936.0</td>
      <td>7.9</td>
      <td>1.78</td>
      <td>33000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[Action, Adventure, Fantasy, Sci-Fi]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Color</td>
      <td>Gore Verbinski</td>
      <td>302.0</td>
      <td>169.0</td>
      <td>563.0</td>
      <td>1000.0</td>
      <td>Orlando Bloom</td>
      <td>40000.0</td>
      <td>309404152.0</td>
      <td>Action|Adventure|Fantasy</td>
      <td>Johnny Depp</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>471220</td>
      <td>48350</td>
      <td>Jack Davenport</td>
      <td>0.0</td>
      <td>goddess|marriage ceremony|marriage proposal|pi...</td>
      <td>http://www.imdb.com/title/tt0449088/?ref_=fn_t...</td>
      <td>1238.0</td>
      <td>PG-13</td>
      <td>300000000.0</td>
      <td>2007.0</td>
      <td>5000.0</td>
      <td>7.1</td>
      <td>2.35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[Action, Adventure, Fantasy]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Color</td>
      <td>Christopher Nolan</td>
      <td>813.0</td>
      <td>164.0</td>
      <td>22000.0</td>
      <td>23000.0</td>
      <td>Christian Bale</td>
      <td>27000.0</td>
      <td>448130642.0</td>
      <td>Action|Thriller</td>
      <td>Tom Hardy</td>
      <td>The Dark Knight Rises</td>
      <td>1144337</td>
      <td>106759</td>
      <td>Joseph Gordon-Levitt</td>
      <td>0.0</td>
      <td>deception|imprisonment|lawlessness|police offi...</td>
      <td>http://www.imdb.com/title/tt1345836/?ref_=fn_t...</td>
      <td>2701.0</td>
      <td>PG-13</td>
      <td>250000000.0</td>
      <td>2012.0</td>
      <td>23000.0</td>
      <td>8.5</td>
      <td>2.35</td>
      <td>164000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[Action, Thriller]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Color</td>
      <td>Andrew Stanton</td>
      <td>462.0</td>
      <td>132.0</td>
      <td>475.0</td>
      <td>530.0</td>
      <td>Samantha Morton</td>
      <td>640.0</td>
      <td>73058679.0</td>
      <td>Action|Adventure|Sci-Fi</td>
      <td>Daryl Sabara</td>
      <td>John Carter</td>
      <td>212204</td>
      <td>1873</td>
      <td>Polly Walker</td>
      <td>1.0</td>
      <td>alien|american civil war|male nipple|mars|prin...</td>
      <td>http://www.imdb.com/title/tt0401729/?ref_=fn_t...</td>
      <td>738.0</td>
      <td>PG-13</td>
      <td>263700000.0</td>
      <td>2012.0</td>
      <td>632.0</td>
      <td>6.6</td>
      <td>2.35</td>
      <td>24000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[Action, Adventure, Sci-Fi]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Color</td>
      <td>Sam Raimi</td>
      <td>392.0</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>4000.0</td>
      <td>James Franco</td>
      <td>24000.0</td>
      <td>336530303.0</td>
      <td>Action|Adventure|Romance</td>
      <td>J.K. Simmons</td>
      <td>Spider-Man 3</td>
      <td>383056</td>
      <td>46055</td>
      <td>Kirsten Dunst</td>
      <td>0.0</td>
      <td>sandman|spider man|symbiote|venom|villain</td>
      <td>http://www.imdb.com/title/tt0413300/?ref_=fn_t...</td>
      <td>1902.0</td>
      <td>PG-13</td>
      <td>258000000.0</td>
      <td>2007.0</td>
      <td>11000.0</td>
      <td>6.2</td>
      <td>2.35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[Action, Adventure, Romance]</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
abt.dtypes
```




    color                              object
    director_name                      object
    num_critic_for_reviews            float64
    duration                          float64
    director_facebook_likes           float64
    actor_3_facebook_likes            float64
    actor_2_name                       object
    actor_1_facebook_likes            float64
    gross                             float64
    genres                             object
    actor_1_name                       object
    movie_title                        object
    num_voted_users                     int64
    cast_total_facebook_likes           int64
    actor_3_name                       object
    facenumber_in_poster              float64
    plot_keywords                      object
    movie_imdb_link                    object
    num_user_for_reviews              float64
    content_rating                     object
    budget                            float64
    title_year                        float64
    actor_2_facebook_likes            float64
    imdb_score                        float64
    aspect_ratio                       object
    movie_facebook_likes                int64
    num_critic_for_reviews_missing      int64
    duration_missing                    int64
    actor_1_facebook_likes_missing      int64
    actor_2_facebook_likes_missing      int64
    actor_3_facebook_likes_missing      int64
    facenumber_in_poster_missing        int64
    genres_list                        object
    Action                              int64
    Adventure                           int64
    Animation                           int64
    Biography                           int64
    Comedy                              int64
    Crime                               int64
    Documentary                         int64
    Drama                               int64
    Family                              int64
    Fantasy                             int64
    Film-Noir                           int64
    History                             int64
    Horror                              int64
    Music                               int64
    Musical                             int64
    Mystery                             int64
    Romance                             int64
    Sci-Fi                              int64
    Short                               int64
    Sport                               int64
    Thriller                            int64
    War                                 int64
    Western                             int64
    dtype: object




```python
abt.drop(columns=['genres', 'genres_list'], inplace=True)
abt.columns
```




    Index(['color', 'director_name', 'num_critic_for_reviews', 'duration',
           'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',
           'actor_1_facebook_likes', 'gross', 'actor_1_name', 'movie_title',
           'num_voted_users', 'cast_total_facebook_likes', 'actor_3_name',
           'facenumber_in_poster', 'plot_keywords', 'movie_imdb_link',
           'num_user_for_reviews', 'content_rating', 'budget', 'title_year',
           'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio',
           'movie_facebook_likes', 'num_critic_for_reviews_missing',
           'duration_missing', 'actor_1_facebook_likes_missing',
           'actor_2_facebook_likes_missing', 'actor_3_facebook_likes_missing',
           'facenumber_in_poster_missing', 'Action', 'Adventure', 'Animation',
           'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
           'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical',
           'Mystery', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War',
           'Western'],
          dtype='object')




```python
top_30_directors_df = abt.groupby(['director_name']).size().reset_index(name ='Count').sort_values('Count').tail(30)
top_30_directors=list(top_30_directors_df.director_name)
print(top_30_directors)
```

    ['Francis Ford Coppola', 'M. Night Shyamalan', 'Dennis Dugan', 'John McTiernan', 'Bobby Farrelly', 'Richard Linklater', 'Oliver Stone', 'Kevin Smith', 'Sam Raimi', 'Tony Scott', 'David Fincher', 'Rob Cohen', 'Rob Reiner', 'Robert Rodriguez', 'John Carpenter', 'Shawn Levy', 'Ron Howard', 'Wes Craven', 'Michael Bay', 'Barry Levinson', 'Robert Zemeckis', 'Ridley Scott', 'Renny Harlin', 'Woody Allen', 'Steven Soderbergh', 'Spike Lee', 'Tim Burton', 'Martin Scorsese', 'Clint Eastwood', 'Steven Spielberg']



```python
top_30_actors_df = abt.groupby(['actor_1_name']).size().reset_index(name ='Count').sort_values('Count').tail(30)
top_30_actors = list(top_30_actors_df.actor_1_name)
print(top_30_actors)
```

    ['Julia Roberts', 'Brad Pitt', 'Paul Walker', 'Joseph Gordon-Levitt', 'Hugh Jackman', 'Matthew McConaughey', 'Liam Neeson', 'Gerard Butler', 'Leonardo DiCaprio', 'Channing Tatum', 'Dwayne Johnson', 'Will Smith', 'Morgan Freeman', 'Kevin Spacey', 'Will Ferrell', 'Tom Cruise', 'Keanu Reeves', 'Steve Buscemi', 'Tom Hanks', 'Robin Williams', 'Robert Downey Jr.', 'Bill Murray', 'Harrison Ford', 'Bruce Willis', 'Matt Damon', 'Nicolas Cage', 'Denzel Washington', 'J.K. Simmons', 'Johnny Depp', 'Robert De Niro']



```python
abt.loc[~abt['director_name'].isin(top_30_directors), 'director_name'] = np.nan
abt.head(10)
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
      <th>color</th>
      <th>director_name</th>
      <th>num_critic_for_reviews</th>
      <th>duration</th>
      <th>director_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>actor_2_name</th>
      <th>actor_1_facebook_likes</th>
      <th>gross</th>
      <th>actor_1_name</th>
      <th>movie_title</th>
      <th>num_voted_users</th>
      <th>cast_total_facebook_likes</th>
      <th>actor_3_name</th>
      <th>facenumber_in_poster</th>
      <th>plot_keywords</th>
      <th>movie_imdb_link</th>
      <th>num_user_for_reviews</th>
      <th>content_rating</th>
      <th>budget</th>
      <th>title_year</th>
      <th>actor_2_facebook_likes</th>
      <th>imdb_score</th>
      <th>aspect_ratio</th>
      <th>movie_facebook_likes</th>
      <th>num_critic_for_reviews_missing</th>
      <th>duration_missing</th>
      <th>actor_1_facebook_likes_missing</th>
      <th>actor_2_facebook_likes_missing</th>
      <th>actor_3_facebook_likes_missing</th>
      <th>facenumber_in_poster_missing</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>History</th>
      <th>Horror</th>
      <th>Music</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Color</td>
      <td>NaN</td>
      <td>723.0</td>
      <td>178.0</td>
      <td>0.0</td>
      <td>855.0</td>
      <td>Joel David Moore</td>
      <td>1000.0</td>
      <td>760505847.0</td>
      <td>CCH Pounder</td>
      <td>Avatar</td>
      <td>886204</td>
      <td>4834</td>
      <td>Wes Studi</td>
      <td>0.0</td>
      <td>avatar|future|marine|native|paraplegic</td>
      <td>http://www.imdb.com/title/tt0499549/?ref_=fn_t...</td>
      <td>3054.0</td>
      <td>PG-13</td>
      <td>237000000.0</td>
      <td>2009.0</td>
      <td>936.0</td>
      <td>7.9</td>
      <td>1.78</td>
      <td>33000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Color</td>
      <td>NaN</td>
      <td>302.0</td>
      <td>169.0</td>
      <td>563.0</td>
      <td>1000.0</td>
      <td>Orlando Bloom</td>
      <td>40000.0</td>
      <td>309404152.0</td>
      <td>Johnny Depp</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>471220</td>
      <td>48350</td>
      <td>Jack Davenport</td>
      <td>0.0</td>
      <td>goddess|marriage ceremony|marriage proposal|pi...</td>
      <td>http://www.imdb.com/title/tt0449088/?ref_=fn_t...</td>
      <td>1238.0</td>
      <td>PG-13</td>
      <td>300000000.0</td>
      <td>2007.0</td>
      <td>5000.0</td>
      <td>7.1</td>
      <td>2.35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Color</td>
      <td>NaN</td>
      <td>813.0</td>
      <td>164.0</td>
      <td>22000.0</td>
      <td>23000.0</td>
      <td>Christian Bale</td>
      <td>27000.0</td>
      <td>448130642.0</td>
      <td>Tom Hardy</td>
      <td>The Dark Knight Rises</td>
      <td>1144337</td>
      <td>106759</td>
      <td>Joseph Gordon-Levitt</td>
      <td>0.0</td>
      <td>deception|imprisonment|lawlessness|police offi...</td>
      <td>http://www.imdb.com/title/tt1345836/?ref_=fn_t...</td>
      <td>2701.0</td>
      <td>PG-13</td>
      <td>250000000.0</td>
      <td>2012.0</td>
      <td>23000.0</td>
      <td>8.5</td>
      <td>2.35</td>
      <td>164000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Color</td>
      <td>NaN</td>
      <td>462.0</td>
      <td>132.0</td>
      <td>475.0</td>
      <td>530.0</td>
      <td>Samantha Morton</td>
      <td>640.0</td>
      <td>73058679.0</td>
      <td>Daryl Sabara</td>
      <td>John Carter</td>
      <td>212204</td>
      <td>1873</td>
      <td>Polly Walker</td>
      <td>1.0</td>
      <td>alien|american civil war|male nipple|mars|prin...</td>
      <td>http://www.imdb.com/title/tt0401729/?ref_=fn_t...</td>
      <td>738.0</td>
      <td>PG-13</td>
      <td>263700000.0</td>
      <td>2012.0</td>
      <td>632.0</td>
      <td>6.6</td>
      <td>2.35</td>
      <td>24000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Color</td>
      <td>Sam Raimi</td>
      <td>392.0</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>4000.0</td>
      <td>James Franco</td>
      <td>24000.0</td>
      <td>336530303.0</td>
      <td>J.K. Simmons</td>
      <td>Spider-Man 3</td>
      <td>383056</td>
      <td>46055</td>
      <td>Kirsten Dunst</td>
      <td>0.0</td>
      <td>sandman|spider man|symbiote|venom|villain</td>
      <td>http://www.imdb.com/title/tt0413300/?ref_=fn_t...</td>
      <td>1902.0</td>
      <td>PG-13</td>
      <td>258000000.0</td>
      <td>2007.0</td>
      <td>11000.0</td>
      <td>6.2</td>
      <td>2.35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Color</td>
      <td>NaN</td>
      <td>324.0</td>
      <td>100.0</td>
      <td>15.0</td>
      <td>284.0</td>
      <td>Donna Murphy</td>
      <td>799.0</td>
      <td>200807262.0</td>
      <td>Brad Garrett</td>
      <td>Tangled</td>
      <td>294810</td>
      <td>2036</td>
      <td>M.C. Gainey</td>
      <td>1.0</td>
      <td>17th century|based on fairy tale|disney|flower...</td>
      <td>http://www.imdb.com/title/tt0398286/?ref_=fn_t...</td>
      <td>387.0</td>
      <td>PG</td>
      <td>260000000.0</td>
      <td>2010.0</td>
      <td>553.0</td>
      <td>7.8</td>
      <td>1.85</td>
      <td>29000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Color</td>
      <td>NaN</td>
      <td>635.0</td>
      <td>141.0</td>
      <td>0.0</td>
      <td>19000.0</td>
      <td>Robert Downey Jr.</td>
      <td>26000.0</td>
      <td>458991599.0</td>
      <td>Chris Hemsworth</td>
      <td>Avengers: Age of Ultron</td>
      <td>462669</td>
      <td>92000</td>
      <td>Scarlett Johansson</td>
      <td>4.0</td>
      <td>artificial intelligence|based on comic book|ca...</td>
      <td>http://www.imdb.com/title/tt2395427/?ref_=fn_t...</td>
      <td>1117.0</td>
      <td>PG-13</td>
      <td>250000000.0</td>
      <td>2015.0</td>
      <td>21000.0</td>
      <td>7.5</td>
      <td>2.35</td>
      <td>118000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Color</td>
      <td>NaN</td>
      <td>673.0</td>
      <td>183.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>Lauren Cohan</td>
      <td>15000.0</td>
      <td>330249062.0</td>
      <td>Henry Cavill</td>
      <td>Batman v Superman: Dawn of Justice</td>
      <td>371639</td>
      <td>24450</td>
      <td>Alan D. Purwin</td>
      <td>0.0</td>
      <td>based on comic book|batman|sequel to a reboot|...</td>
      <td>http://www.imdb.com/title/tt2975590/?ref_=fn_t...</td>
      <td>3018.0</td>
      <td>PG-13</td>
      <td>250000000.0</td>
      <td>2016.0</td>
      <td>4000.0</td>
      <td>6.9</td>
      <td>2.35</td>
      <td>197000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Color</td>
      <td>NaN</td>
      <td>434.0</td>
      <td>169.0</td>
      <td>0.0</td>
      <td>903.0</td>
      <td>Marlon Brando</td>
      <td>18000.0</td>
      <td>200069408.0</td>
      <td>Kevin Spacey</td>
      <td>Superman Returns</td>
      <td>240396</td>
      <td>29991</td>
      <td>Frank Langella</td>
      <td>0.0</td>
      <td>crystal|epic|lex luthor|lois lane|return to earth</td>
      <td>http://www.imdb.com/title/tt0348150/?ref_=fn_t...</td>
      <td>2367.0</td>
      <td>PG-13</td>
      <td>209000000.0</td>
      <td>2006.0</td>
      <td>10000.0</td>
      <td>6.1</td>
      <td>2.35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Color</td>
      <td>NaN</td>
      <td>313.0</td>
      <td>151.0</td>
      <td>563.0</td>
      <td>1000.0</td>
      <td>Orlando Bloom</td>
      <td>40000.0</td>
      <td>423032628.0</td>
      <td>Johnny Depp</td>
      <td>Pirates of the Caribbean: Dead Man's Chest</td>
      <td>522040</td>
      <td>48486</td>
      <td>Jack Davenport</td>
      <td>2.0</td>
      <td>box office hit|giant squid|heart|liar's dice|m...</td>
      <td>http://www.imdb.com/title/tt0383574/?ref_=fn_t...</td>
      <td>1832.0</td>
      <td>PG-13</td>
      <td>225000000.0</td>
      <td>2006.0</td>
      <td>5000.0</td>
      <td>7.3</td>
      <td>2.35</td>
      <td>5000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
abt.loc[~abt['actor_1_name'].isin(top_30_actors), 'actor_1_name'] = np.nan
abt.head(10)
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
      <th>color</th>
      <th>director_name</th>
      <th>num_critic_for_reviews</th>
      <th>duration</th>
      <th>director_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>actor_2_name</th>
      <th>actor_1_facebook_likes</th>
      <th>gross</th>
      <th>actor_1_name</th>
      <th>movie_title</th>
      <th>num_voted_users</th>
      <th>cast_total_facebook_likes</th>
      <th>actor_3_name</th>
      <th>facenumber_in_poster</th>
      <th>plot_keywords</th>
      <th>movie_imdb_link</th>
      <th>num_user_for_reviews</th>
      <th>content_rating</th>
      <th>budget</th>
      <th>title_year</th>
      <th>actor_2_facebook_likes</th>
      <th>imdb_score</th>
      <th>aspect_ratio</th>
      <th>movie_facebook_likes</th>
      <th>num_critic_for_reviews_missing</th>
      <th>duration_missing</th>
      <th>actor_1_facebook_likes_missing</th>
      <th>actor_2_facebook_likes_missing</th>
      <th>actor_3_facebook_likes_missing</th>
      <th>facenumber_in_poster_missing</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>History</th>
      <th>Horror</th>
      <th>Music</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Color</td>
      <td>NaN</td>
      <td>723.0</td>
      <td>178.0</td>
      <td>0.0</td>
      <td>855.0</td>
      <td>Joel David Moore</td>
      <td>1000.0</td>
      <td>760505847.0</td>
      <td>NaN</td>
      <td>Avatar</td>
      <td>886204</td>
      <td>4834</td>
      <td>Wes Studi</td>
      <td>0.0</td>
      <td>avatar|future|marine|native|paraplegic</td>
      <td>http://www.imdb.com/title/tt0499549/?ref_=fn_t...</td>
      <td>3054.0</td>
      <td>PG-13</td>
      <td>237000000.0</td>
      <td>2009.0</td>
      <td>936.0</td>
      <td>7.9</td>
      <td>1.78</td>
      <td>33000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Color</td>
      <td>NaN</td>
      <td>302.0</td>
      <td>169.0</td>
      <td>563.0</td>
      <td>1000.0</td>
      <td>Orlando Bloom</td>
      <td>40000.0</td>
      <td>309404152.0</td>
      <td>Johnny Depp</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>471220</td>
      <td>48350</td>
      <td>Jack Davenport</td>
      <td>0.0</td>
      <td>goddess|marriage ceremony|marriage proposal|pi...</td>
      <td>http://www.imdb.com/title/tt0449088/?ref_=fn_t...</td>
      <td>1238.0</td>
      <td>PG-13</td>
      <td>300000000.0</td>
      <td>2007.0</td>
      <td>5000.0</td>
      <td>7.1</td>
      <td>2.35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Color</td>
      <td>NaN</td>
      <td>813.0</td>
      <td>164.0</td>
      <td>22000.0</td>
      <td>23000.0</td>
      <td>Christian Bale</td>
      <td>27000.0</td>
      <td>448130642.0</td>
      <td>NaN</td>
      <td>The Dark Knight Rises</td>
      <td>1144337</td>
      <td>106759</td>
      <td>Joseph Gordon-Levitt</td>
      <td>0.0</td>
      <td>deception|imprisonment|lawlessness|police offi...</td>
      <td>http://www.imdb.com/title/tt1345836/?ref_=fn_t...</td>
      <td>2701.0</td>
      <td>PG-13</td>
      <td>250000000.0</td>
      <td>2012.0</td>
      <td>23000.0</td>
      <td>8.5</td>
      <td>2.35</td>
      <td>164000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Color</td>
      <td>NaN</td>
      <td>462.0</td>
      <td>132.0</td>
      <td>475.0</td>
      <td>530.0</td>
      <td>Samantha Morton</td>
      <td>640.0</td>
      <td>73058679.0</td>
      <td>NaN</td>
      <td>John Carter</td>
      <td>212204</td>
      <td>1873</td>
      <td>Polly Walker</td>
      <td>1.0</td>
      <td>alien|american civil war|male nipple|mars|prin...</td>
      <td>http://www.imdb.com/title/tt0401729/?ref_=fn_t...</td>
      <td>738.0</td>
      <td>PG-13</td>
      <td>263700000.0</td>
      <td>2012.0</td>
      <td>632.0</td>
      <td>6.6</td>
      <td>2.35</td>
      <td>24000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Color</td>
      <td>Sam Raimi</td>
      <td>392.0</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>4000.0</td>
      <td>James Franco</td>
      <td>24000.0</td>
      <td>336530303.0</td>
      <td>J.K. Simmons</td>
      <td>Spider-Man 3</td>
      <td>383056</td>
      <td>46055</td>
      <td>Kirsten Dunst</td>
      <td>0.0</td>
      <td>sandman|spider man|symbiote|venom|villain</td>
      <td>http://www.imdb.com/title/tt0413300/?ref_=fn_t...</td>
      <td>1902.0</td>
      <td>PG-13</td>
      <td>258000000.0</td>
      <td>2007.0</td>
      <td>11000.0</td>
      <td>6.2</td>
      <td>2.35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Color</td>
      <td>NaN</td>
      <td>324.0</td>
      <td>100.0</td>
      <td>15.0</td>
      <td>284.0</td>
      <td>Donna Murphy</td>
      <td>799.0</td>
      <td>200807262.0</td>
      <td>NaN</td>
      <td>Tangled</td>
      <td>294810</td>
      <td>2036</td>
      <td>M.C. Gainey</td>
      <td>1.0</td>
      <td>17th century|based on fairy tale|disney|flower...</td>
      <td>http://www.imdb.com/title/tt0398286/?ref_=fn_t...</td>
      <td>387.0</td>
      <td>PG</td>
      <td>260000000.0</td>
      <td>2010.0</td>
      <td>553.0</td>
      <td>7.8</td>
      <td>1.85</td>
      <td>29000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Color</td>
      <td>NaN</td>
      <td>635.0</td>
      <td>141.0</td>
      <td>0.0</td>
      <td>19000.0</td>
      <td>Robert Downey Jr.</td>
      <td>26000.0</td>
      <td>458991599.0</td>
      <td>NaN</td>
      <td>Avengers: Age of Ultron</td>
      <td>462669</td>
      <td>92000</td>
      <td>Scarlett Johansson</td>
      <td>4.0</td>
      <td>artificial intelligence|based on comic book|ca...</td>
      <td>http://www.imdb.com/title/tt2395427/?ref_=fn_t...</td>
      <td>1117.0</td>
      <td>PG-13</td>
      <td>250000000.0</td>
      <td>2015.0</td>
      <td>21000.0</td>
      <td>7.5</td>
      <td>2.35</td>
      <td>118000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Color</td>
      <td>NaN</td>
      <td>673.0</td>
      <td>183.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>Lauren Cohan</td>
      <td>15000.0</td>
      <td>330249062.0</td>
      <td>NaN</td>
      <td>Batman v Superman: Dawn of Justice</td>
      <td>371639</td>
      <td>24450</td>
      <td>Alan D. Purwin</td>
      <td>0.0</td>
      <td>based on comic book|batman|sequel to a reboot|...</td>
      <td>http://www.imdb.com/title/tt2975590/?ref_=fn_t...</td>
      <td>3018.0</td>
      <td>PG-13</td>
      <td>250000000.0</td>
      <td>2016.0</td>
      <td>4000.0</td>
      <td>6.9</td>
      <td>2.35</td>
      <td>197000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Color</td>
      <td>NaN</td>
      <td>434.0</td>
      <td>169.0</td>
      <td>0.0</td>
      <td>903.0</td>
      <td>Marlon Brando</td>
      <td>18000.0</td>
      <td>200069408.0</td>
      <td>Kevin Spacey</td>
      <td>Superman Returns</td>
      <td>240396</td>
      <td>29991</td>
      <td>Frank Langella</td>
      <td>0.0</td>
      <td>crystal|epic|lex luthor|lois lane|return to earth</td>
      <td>http://www.imdb.com/title/tt0348150/?ref_=fn_t...</td>
      <td>2367.0</td>
      <td>PG-13</td>
      <td>209000000.0</td>
      <td>2006.0</td>
      <td>10000.0</td>
      <td>6.1</td>
      <td>2.35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Color</td>
      <td>NaN</td>
      <td>313.0</td>
      <td>151.0</td>
      <td>563.0</td>
      <td>1000.0</td>
      <td>Orlando Bloom</td>
      <td>40000.0</td>
      <td>423032628.0</td>
      <td>Johnny Depp</td>
      <td>Pirates of the Caribbean: Dead Man's Chest</td>
      <td>522040</td>
      <td>48486</td>
      <td>Jack Davenport</td>
      <td>2.0</td>
      <td>box office hit|giant squid|heart|liar's dice|m...</td>
      <td>http://www.imdb.com/title/tt0383574/?ref_=fn_t...</td>
      <td>1832.0</td>
      <td>PG-13</td>
      <td>225000000.0</td>
      <td>2006.0</td>
      <td>5000.0</td>
      <td>7.3</td>
      <td>2.35</td>
      <td>5000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
abt.dtypes
```




    color                              object
    director_name                      object
    num_critic_for_reviews            float64
    duration                          float64
    director_facebook_likes           float64
    actor_3_facebook_likes            float64
    actor_2_name                       object
    actor_1_facebook_likes            float64
    gross                             float64
    actor_1_name                       object
    movie_title                        object
    num_voted_users                     int64
    cast_total_facebook_likes           int64
    actor_3_name                       object
    facenumber_in_poster              float64
    plot_keywords                      object
    movie_imdb_link                    object
    num_user_for_reviews              float64
    content_rating                     object
    budget                            float64
    title_year                        float64
    actor_2_facebook_likes            float64
    imdb_score                        float64
    aspect_ratio                       object
    movie_facebook_likes                int64
    num_critic_for_reviews_missing      int64
    duration_missing                    int64
    actor_1_facebook_likes_missing      int64
    actor_2_facebook_likes_missing      int64
    actor_3_facebook_likes_missing      int64
    facenumber_in_poster_missing        int64
    Action                              int64
    Adventure                           int64
    Animation                           int64
    Biography                           int64
    Comedy                              int64
    Crime                               int64
    Documentary                         int64
    Drama                               int64
    Family                              int64
    Fantasy                             int64
    Film-Noir                           int64
    History                             int64
    Horror                              int64
    Music                               int64
    Musical                             int64
    Mystery                             int64
    Romance                             int64
    Sci-Fi                              int64
    Short                               int64
    Sport                               int64
    Thriller                            int64
    War                                 int64
    Western                             int64
    dtype: object




```python
movie_titles_df = abt.movie_title
movie_titles_df
```




    0                                            Avatar 
    1          Pirates of the Caribbean: At World's End 
    3                             The Dark Knight Rises 
    5                                       John Carter 
    6                                      Spider-Man 3 
    7                                           Tangled 
    8                           Avengers: Age of Ultron 
    10               Batman v Superman: Dawn of Justice 
    11                                 Superman Returns 
    13       Pirates of the Caribbean: Dead Man's Chest 
    14                                  The Lone Ranger 
    15                                     Man of Steel 
    16         The Chronicles of Narnia: Prince Caspian 
    17                                     The Avengers 
    18      Pirates of the Caribbean: On Stranger Tides 
    19                                   Men in Black 3 
    21                           The Amazing Spider-Man 
    22                                       Robin Hood 
    23              The Hobbit: The Desolation of Smaug 
    24                               The Golden Compass 
    26                                          Titanic 
    27                       Captain America: Civil War 
    28                                       Battleship 
    29                                   Jurassic World 
    31                                     Spider-Man 2 
    32                                       Iron Man 3 
    33                              Alice in Wonderland 
    35                              Monsters University 
    36              Transformers: Revenge of the Fallen 
    37                  Transformers: Age of Extinction 
                                ...                     
    4941                                     Roger & Me 
    4947                           Your Sister's Sister 
    4955                              Facing the Giants 
    4956                                    The Gallows 
    4958                 Over the Hill to the Poorhouse 
    4959                              Hollywood Shuffle 
    4962                   The Lost Skeleton of Cadavra 
    4964                                  Cheap Thrills 
    4971                     The Last House on the Left 
    4973                                             Pi 
    4975                                       20 Dates 
    4977                                  Super Size Me 
    4978                                         The FP 
    4979                                Happy Christmas 
    4984                          The Brothers McMullen 
    4987                                 Tiny Furniture 
    4997                              George Washington 
    4998                    Smiling Fish & Goat on Fire 
    5004                        The Legend of God's Gun 
    5008                                         Clerks 
    5009                                 Pink Narcissus 
    5012                                       Sabotage 
    5015                                        Slacker 
    5021                                The Puffy Chair 
    5023                               Breaking Upwards 
    5025                                 Pink Flamingos 
    5033                                         Primer 
    5035                                    El Mariachi 
    5037                                      Newlyweds 
    5042                              My Date with Drew 
    Name: movie_title, Length: 3074, dtype: object




```python
abt.drop(columns=['actor_2_name', 'actor_3_name', 'plot_keywords', 'movie_imdb_link', 'title_year', 'facenumber_in_poster', 'facenumber_in_poster_missing', 'movie_title'], inplace=True)
abt.head()
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
      <th>color</th>
      <th>director_name</th>
      <th>num_critic_for_reviews</th>
      <th>duration</th>
      <th>director_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>actor_1_facebook_likes</th>
      <th>gross</th>
      <th>actor_1_name</th>
      <th>num_voted_users</th>
      <th>cast_total_facebook_likes</th>
      <th>num_user_for_reviews</th>
      <th>content_rating</th>
      <th>budget</th>
      <th>actor_2_facebook_likes</th>
      <th>imdb_score</th>
      <th>aspect_ratio</th>
      <th>movie_facebook_likes</th>
      <th>num_critic_for_reviews_missing</th>
      <th>duration_missing</th>
      <th>actor_1_facebook_likes_missing</th>
      <th>actor_2_facebook_likes_missing</th>
      <th>actor_3_facebook_likes_missing</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>History</th>
      <th>Horror</th>
      <th>Music</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Color</td>
      <td>NaN</td>
      <td>723.0</td>
      <td>178.0</td>
      <td>0.0</td>
      <td>855.0</td>
      <td>1000.0</td>
      <td>760505847.0</td>
      <td>NaN</td>
      <td>886204</td>
      <td>4834</td>
      <td>3054.0</td>
      <td>PG-13</td>
      <td>237000000.0</td>
      <td>936.0</td>
      <td>7.9</td>
      <td>1.78</td>
      <td>33000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Color</td>
      <td>NaN</td>
      <td>302.0</td>
      <td>169.0</td>
      <td>563.0</td>
      <td>1000.0</td>
      <td>40000.0</td>
      <td>309404152.0</td>
      <td>Johnny Depp</td>
      <td>471220</td>
      <td>48350</td>
      <td>1238.0</td>
      <td>PG-13</td>
      <td>300000000.0</td>
      <td>5000.0</td>
      <td>7.1</td>
      <td>2.35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Color</td>
      <td>NaN</td>
      <td>813.0</td>
      <td>164.0</td>
      <td>22000.0</td>
      <td>23000.0</td>
      <td>27000.0</td>
      <td>448130642.0</td>
      <td>NaN</td>
      <td>1144337</td>
      <td>106759</td>
      <td>2701.0</td>
      <td>PG-13</td>
      <td>250000000.0</td>
      <td>23000.0</td>
      <td>8.5</td>
      <td>2.35</td>
      <td>164000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Color</td>
      <td>NaN</td>
      <td>462.0</td>
      <td>132.0</td>
      <td>475.0</td>
      <td>530.0</td>
      <td>640.0</td>
      <td>73058679.0</td>
      <td>NaN</td>
      <td>212204</td>
      <td>1873</td>
      <td>738.0</td>
      <td>PG-13</td>
      <td>263700000.0</td>
      <td>632.0</td>
      <td>6.6</td>
      <td>2.35</td>
      <td>24000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Color</td>
      <td>Sam Raimi</td>
      <td>392.0</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>4000.0</td>
      <td>24000.0</td>
      <td>336530303.0</td>
      <td>J.K. Simmons</td>
      <td>383056</td>
      <td>46055</td>
      <td>1902.0</td>
      <td>PG-13</td>
      <td>258000000.0</td>
      <td>11000.0</td>
      <td>6.2</td>
      <td>2.35</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.countplot(y=abt.aspect_ratio)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a18674eb8>




![png](output_58_1.png)



```python
abt.aspect_ratio.replace(to_replace=[1.78, 2.0, 2.2, 2.39, 2.24, 1.66, 1.5, 1.77, 2.4, 2.76, 1.33, 1.18, 2.55, 1.75, 16.0], value='Other', inplace=True)
sns.countplot(y=abt.aspect_ratio)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a189a14a8>




![png](output_59_1.png)



```python
abt.dtypes
```




    color                              object
    director_name                      object
    num_critic_for_reviews            float64
    duration                          float64
    director_facebook_likes           float64
    actor_3_facebook_likes            float64
    actor_1_facebook_likes            float64
    gross                             float64
    actor_1_name                       object
    num_voted_users                     int64
    cast_total_facebook_likes           int64
    num_user_for_reviews              float64
    content_rating                     object
    budget                            float64
    actor_2_facebook_likes            float64
    imdb_score                        float64
    aspect_ratio                       object
    movie_facebook_likes                int64
    num_critic_for_reviews_missing      int64
    duration_missing                    int64
    actor_1_facebook_likes_missing      int64
    actor_2_facebook_likes_missing      int64
    actor_3_facebook_likes_missing      int64
    Action                              int64
    Adventure                           int64
    Animation                           int64
    Biography                           int64
    Comedy                              int64
    Crime                               int64
    Documentary                         int64
    Drama                               int64
    Family                              int64
    Fantasy                             int64
    Film-Noir                           int64
    History                             int64
    Horror                              int64
    Music                               int64
    Musical                             int64
    Mystery                             int64
    Romance                             int64
    Sci-Fi                              int64
    Short                               int64
    Sport                               int64
    Thriller                            int64
    War                                 int64
    Western                             int64
    dtype: object




```python
abt = pd.get_dummies(abt)
```


```python
abt.head(10)
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
      <th>num_critic_for_reviews</th>
      <th>duration</th>
      <th>director_facebook_likes</th>
      <th>actor_3_facebook_likes</th>
      <th>actor_1_facebook_likes</th>
      <th>gross</th>
      <th>num_voted_users</th>
      <th>cast_total_facebook_likes</th>
      <th>num_user_for_reviews</th>
      <th>budget</th>
      <th>actor_2_facebook_likes</th>
      <th>imdb_score</th>
      <th>movie_facebook_likes</th>
      <th>num_critic_for_reviews_missing</th>
      <th>duration_missing</th>
      <th>actor_1_facebook_likes_missing</th>
      <th>actor_2_facebook_likes_missing</th>
      <th>actor_3_facebook_likes_missing</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Biography</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Fantasy</th>
      <th>Film-Noir</th>
      <th>History</th>
      <th>Horror</th>
      <th>Music</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Short</th>
      <th>Sport</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
      <th>color_ Black and White</th>
      <th>color_Color</th>
      <th>color_Missing</th>
      <th>director_name_Barry Levinson</th>
      <th>director_name_Bobby Farrelly</th>
      <th>director_name_Clint Eastwood</th>
      <th>director_name_David Fincher</th>
      <th>director_name_Dennis Dugan</th>
      <th>director_name_Francis Ford Coppola</th>
      <th>...</th>
      <th>director_name_Sam Raimi</th>
      <th>director_name_Shawn Levy</th>
      <th>director_name_Spike Lee</th>
      <th>director_name_Steven Soderbergh</th>
      <th>director_name_Steven Spielberg</th>
      <th>director_name_Tim Burton</th>
      <th>director_name_Tony Scott</th>
      <th>director_name_Wes Craven</th>
      <th>director_name_Woody Allen</th>
      <th>actor_1_name_Bill Murray</th>
      <th>actor_1_name_Brad Pitt</th>
      <th>actor_1_name_Bruce Willis</th>
      <th>actor_1_name_Channing Tatum</th>
      <th>actor_1_name_Denzel Washington</th>
      <th>actor_1_name_Dwayne Johnson</th>
      <th>actor_1_name_Gerard Butler</th>
      <th>actor_1_name_Harrison Ford</th>
      <th>actor_1_name_Hugh Jackman</th>
      <th>actor_1_name_J.K. Simmons</th>
      <th>actor_1_name_Johnny Depp</th>
      <th>actor_1_name_Joseph Gordon-Levitt</th>
      <th>actor_1_name_Julia Roberts</th>
      <th>actor_1_name_Keanu Reeves</th>
      <th>actor_1_name_Kevin Spacey</th>
      <th>actor_1_name_Leonardo DiCaprio</th>
      <th>actor_1_name_Liam Neeson</th>
      <th>actor_1_name_Matt Damon</th>
      <th>actor_1_name_Matthew McConaughey</th>
      <th>actor_1_name_Morgan Freeman</th>
      <th>actor_1_name_Nicolas Cage</th>
      <th>actor_1_name_Paul Walker</th>
      <th>actor_1_name_Robert De Niro</th>
      <th>actor_1_name_Robert Downey Jr.</th>
      <th>actor_1_name_Robin Williams</th>
      <th>actor_1_name_Steve Buscemi</th>
      <th>actor_1_name_Tom Cruise</th>
      <th>actor_1_name_Tom Hanks</th>
      <th>actor_1_name_Will Ferrell</th>
      <th>actor_1_name_Will Smith</th>
      <th>content_rating_G</th>
      <th>content_rating_Missing</th>
      <th>content_rating_Other</th>
      <th>content_rating_PG</th>
      <th>content_rating_PG-13</th>
      <th>content_rating_R</th>
      <th>aspect_ratio_1.37</th>
      <th>aspect_ratio_1.85</th>
      <th>aspect_ratio_2.35</th>
      <th>aspect_ratio_Missing</th>
      <th>aspect_ratio_Other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>723.0</td>
      <td>178.0</td>
      <td>0.0</td>
      <td>855.0</td>
      <td>1000.0</td>
      <td>760505847.0</td>
      <td>886204</td>
      <td>4834</td>
      <td>3054.0</td>
      <td>237000000.0</td>
      <td>936.0</td>
      <td>7.9</td>
      <td>33000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>302.0</td>
      <td>169.0</td>
      <td>563.0</td>
      <td>1000.0</td>
      <td>40000.0</td>
      <td>309404152.0</td>
      <td>471220</td>
      <td>48350</td>
      <td>1238.0</td>
      <td>300000000.0</td>
      <td>5000.0</td>
      <td>7.1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>813.0</td>
      <td>164.0</td>
      <td>22000.0</td>
      <td>23000.0</td>
      <td>27000.0</td>
      <td>448130642.0</td>
      <td>1144337</td>
      <td>106759</td>
      <td>2701.0</td>
      <td>250000000.0</td>
      <td>23000.0</td>
      <td>8.5</td>
      <td>164000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>462.0</td>
      <td>132.0</td>
      <td>475.0</td>
      <td>530.0</td>
      <td>640.0</td>
      <td>73058679.0</td>
      <td>212204</td>
      <td>1873</td>
      <td>738.0</td>
      <td>263700000.0</td>
      <td>632.0</td>
      <td>6.6</td>
      <td>24000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>392.0</td>
      <td>156.0</td>
      <td>0.0</td>
      <td>4000.0</td>
      <td>24000.0</td>
      <td>336530303.0</td>
      <td>383056</td>
      <td>46055</td>
      <td>1902.0</td>
      <td>258000000.0</td>
      <td>11000.0</td>
      <td>6.2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>324.0</td>
      <td>100.0</td>
      <td>15.0</td>
      <td>284.0</td>
      <td>799.0</td>
      <td>200807262.0</td>
      <td>294810</td>
      <td>2036</td>
      <td>387.0</td>
      <td>260000000.0</td>
      <td>553.0</td>
      <td>7.8</td>
      <td>29000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>635.0</td>
      <td>141.0</td>
      <td>0.0</td>
      <td>19000.0</td>
      <td>26000.0</td>
      <td>458991599.0</td>
      <td>462669</td>
      <td>92000</td>
      <td>1117.0</td>
      <td>250000000.0</td>
      <td>21000.0</td>
      <td>7.5</td>
      <td>118000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>673.0</td>
      <td>183.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>15000.0</td>
      <td>330249062.0</td>
      <td>371639</td>
      <td>24450</td>
      <td>3018.0</td>
      <td>250000000.0</td>
      <td>4000.0</td>
      <td>6.9</td>
      <td>197000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>434.0</td>
      <td>169.0</td>
      <td>0.0</td>
      <td>903.0</td>
      <td>18000.0</td>
      <td>200069408.0</td>
      <td>240396</td>
      <td>29991</td>
      <td>2367.0</td>
      <td>209000000.0</td>
      <td>10000.0</td>
      <td>6.1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>313.0</td>
      <td>151.0</td>
      <td>563.0</td>
      <td>1000.0</td>
      <td>40000.0</td>
      <td>423032628.0</td>
      <td>522040</td>
      <td>48486</td>
      <td>1832.0</td>
      <td>225000000.0</td>
      <td>5000.0</td>
      <td>7.3</td>
      <td>5000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 115 columns</p>
</div>



## Algorithm Selection

We will use a linear regression algorithm with Lasso, Ridge, and Elastic Net regularization. We'll also use two tree ensemble algorithms: random forests and boosted trees. These are the best common algorithms for regression tasks.

## Model Training


```python
# Split features from target variable, and split training and test data.

y = abt.gross
X = abt.drop('gross', axis=1)
print(y.shape, X.shape)

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1234)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (3074,) (3074, 114)
    (2459, 114) (615, 114) (2459,) (615,)



```python
# Make a pipelines dictionary for the five algorithms selected, including Standardization in the pipelines

pipelines = {
    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=123)),
    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=123)),
    'enet' : make_pipeline(StandardScaler(), ElasticNet(random_state=123)),
    'rf' : make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123))
}
```


```python
# Create hyperparameters dictionary for Lasso Regression
lasso_hyperparameters = {
    'lasso__alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
}
```


```python
# Create hyperparameters dictionary for Ridge Regression
ridge_hyperparameters = {
    'ridge__alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
}
```


```python
# Create hyperparameters dictionary for Elastic Net Regression
enet_hyperparameters = {
    'elasticnet__alpha': [0.0001, 0.001, 0.1, 1, 5, 10],
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]
}
```


```python
# Create hyperparameters dictionary for Random Forest Regression
rf_hyperparameters = {
    'randomforestregressor__n_estimators' : [100, 200],
    'randomforestregressor__max_features' : ['auto', 'sqrt', 0.5, 0.33, 0.2]
}
```


```python
# Create hyperparameters dictionary for Gradient Boosting Regression
gb_hyperparameters = {
    'gradientboostingregressor__n_estimators' : [100, 200],
    'gradientboostingregressor__learning_rate' : [0.02, 0.05, 0.1, 0.2, 0.5],
    'gradientboostingregressor__max_depth' : [1, 3, 5]
}
```


```python
# Create hyperparameters dictionary for all five algorithms
hyperparameters = {
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters,
    'lasso' : lasso_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'enet' : enet_hyperparameters
}
```


```python
# Create dictionary of fitted models
fitted_models = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted.')
```

    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)
    /anaconda3/lib/python3.7/site-packages/sklearn/linear_model/coordinate_descent.py:492: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    lasso has been fitted.


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)


    ridge has been fitted.


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)
    /anaconda3/lib/python3.7/site-packages/sklearn/linear_model/coordinate_descent.py:492: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    enet has been fitted.


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)


    rf has been fitted.


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)


    gb has been fitted.



```python
# Check that all items in fitted_models are the correct type
for name, model in fitted_models.items():
    print(name, type(model))
```

    lasso <class 'sklearn.model_selection._search.GridSearchCV'>
    ridge <class 'sklearn.model_selection._search.GridSearchCV'>
    enet <class 'sklearn.model_selection._search.GridSearchCV'>
    rf <class 'sklearn.model_selection._search.GridSearchCV'>
    gb <class 'sklearn.model_selection._search.GridSearchCV'>



```python
# Check that all items in fitted_models were fitted
for name, model in fitted_models.items():
    try:
        model.predict(X_test)
        print(name, 'has can be predicted.')
    except NotFittedError as e:
        print(repr(e))
```

    lasso has can be predicted.
    ridge has can be predicted.
    enet has can be predicted.
    rf has can be predicted.
    gb has can be predicted.


    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)



```python
for name, model in fitted_models.items():
    print(name, model.best_score_)
```

    lasso 0.6105838668132211
    ridge 0.610530518624387
    enet 0.6106410986449954
    rf 0.7054995315513697
    gb 0.7188774223628239



```python
for name, model in fitted_models.items():
    pred=model.predict(X_test)
    print(name)
    print('---------')
    print('R^2:', r2_score(y_test, pred))
    print('MAE:', mean_absolute_error(y_test,pred))
    print()
```

    lasso
    ---------
    R^2: 0.6542008006921576
    MAE: 29005703.711320646

    ridge
    ---------
    R^2: 0.6533121366602359
    MAE: 29022969.691347323

    enet
    ---------
    R^2: 0.6535463184121977
    MAE: 29014032.898831517

    rf
    ---------
    R^2: 0.7126563011741076
    MAE: 24716511.824715447

    gb
    ---------
    R^2: 0.7172318667124713
    MAE: 24404568.83419173



    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)



```python
# Plotting gb predictions against actuals
gb_pred = fitted_models['gb'].predict(X_test)
plt.scatter(gb_pred, y_test)
plt.xlabel('predicted by gb')
plt.ylabel('actual')
plt.show()
```

    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)



![png](output_79_1.png)


## Insights & Analysis

The gradient boosting algorithm was the best model. It has an R^2 score of ~72%, pretty good, against both the testing and training data. It predicted movie gross to within ~24M, our goal was to predict scores to within 1/4 of the standard deviation of the profit of movies (~69), a win condition of ~17M. Let's take a look at the winning algorithm to see what we might learn about it. Also maybe we can tune the model a bit more to get under the win condition. Right now we are predicting to within 35 percent of one standard deviation of estimated profit margin.


```python
fitted_models['gb'].best_estimator_
```




    Pipeline(memory=None,
         steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('gradientboostingregressor', GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.05, loss='ls', max_depth=5, max_features=None,
                 max_leaf_nodes=None, m...123, subsample=1.0, tol=0.0001,
                 validation_fraction=0.1, verbose=0, warm_start=False))])




```python
# Since the best values ended up being a learning_rate of 0.05, and a max_depth of 5, I will try a few more values nearby
gb_hyperparameters_ft = {
    'gradientboostingregressor__n_estimators' : [100, 200],
    'gradientboostingregressor__learning_rate' : [0.033, 0.05, 0.066, 0.75],
    'gradientboostingregressor__max_depth' : [4, 5, 7, 9]
}

gb_model = GridSearchCV(pipelines['gb'], gb_hyperparameters_ft, cv=10, n_jobs=-1)
gb_model.fit(X_train, y_train)
gb_pred_ft = gb_model.predict(X_test)
print('R^2:', r2_score(y_test, gb_pred_ft))
print('MAE:', mean_absolute_error(y_test, gb_pred_ft))
print(gb_model.best_estimator_)
```

    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)


    R^2: 0.7147880977701286
    MAE: 24403582.191616513
    Pipeline(memory=None,
         steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('gradientboostingregressor', GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.066, loss='ls', max_depth=5,
                 max_features=None, max_leaf_nodes=None,
    ...     subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
                 warm_start=False))])


    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)


Not a huge improvement, nonetheless we're able to predict movie prices to within a fraction of a standard deviation of estimated profit margin, this should still help, and we were able to capture an R^2 score of 72% based on basic information about the movie including the name of the director and cast, director and cast facebook likes, and critical information.

Some of these wouldn't be available before a movie came out so that makes the model less useful. I wonder if it is possible to predict gross to within one standard deviation even if we remove critical information: imdb score, number of user reviews, and number of critical reviews.


```python
abt_pre = abt.drop(columns = ['num_critic_for_reviews', 'num_voted_users', 'num_user_for_reviews', 'imdb_score'])
X_pre = abt_pre.drop(columns = ['gross'])
y_pre = abt.gross

(X_pre_train, X_pre_test, y_pre_train, y_pre_test) = train_test_split(X_pre, y_pre, test_size=0.2, random_state=1234)
print(X_pre_train.shape, X_pre_test.shape, y_pre_train.shape, y_pre_test.shape)
```

    (2459, 110) (615, 110) (2459,) (615,)



```python
fitted_models_pre = {}

for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    model.fit(X_pre_train, y_pre_train)
    fitted_models_pre[name] = model
    print(name, 'has been fitted.')
```

    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)
    /anaconda3/lib/python3.7/site-packages/sklearn/linear_model/coordinate_descent.py:492: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    lasso has been fitted.


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)


    ridge has been fitted.


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)
    /anaconda3/lib/python3.7/site-packages/sklearn/linear_model/coordinate_descent.py:492: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    enet has been fitted.


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)


    rf has been fitted.


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /anaconda3/lib/python3.7/site-packages/sklearn/base.py:465: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, y, **fit_params).transform(X)


    gb has been fitted.



```python
for name, model in fitted_models_pre.items():
    print(name, type(model))
```

    lasso <class 'sklearn.model_selection._search.GridSearchCV'>
    ridge <class 'sklearn.model_selection._search.GridSearchCV'>
    enet <class 'sklearn.model_selection._search.GridSearchCV'>
    rf <class 'sklearn.model_selection._search.GridSearchCV'>
    gb <class 'sklearn.model_selection._search.GridSearchCV'>



```python
for name, model in fitted_models_pre.items():
    try:
        model.predict(X_pre_test)
        print(name, 'has can be predicted.')
    except NotFittedError as e:
        print(repr(e))
```

    lasso has can be predicted.
    ridge has can be predicted.
    enet has can be predicted.
    rf has can be predicted.
    gb has can be predicted.


    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)



```python
for name, model in fitted_models_pre.items():
    pred=model.predict(X_pre_test)
    print(name)
    print('---------')
    print('R^2:', r2_score(y_pre_test, pred))
    print('MAE:', mean_absolute_error(y_pre_test,pred))
    print()
```

    lasso
    ---------
    R^2: 0.49931838103619997
    MAE: 34377721.31137128

    ridge
    ---------
    R^2: 0.4984129637794107
    MAE: 34358785.48021498

    enet
    ---------
    R^2: 0.49865728288734834
    MAE: 34359875.78074347

    rf
    ---------
    R^2: 0.5809882590723332
    MAE: 30897302.40704065

    gb
    ---------
    R^2: 0.5891104892117891
    MAE: 30378141.617756207



    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)
    /anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:331: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      Xt = transform.transform(Xt)


Even without the critical information we were able to predict movie gross to within ~30M, less than half the standard deviation of estimated movie profti.
