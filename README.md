# RasmusAU-PHBS_MLF_2018

## Author
|Github account |Student ID |Name |
|:----- |:----- |:----- |
|[RasmusAU](https://github.com/RasmusAU) |1802010222 |Rasmus Behnk |

## Goal of the project
* Knowing many of the features before making a movie, e.g. movie director, duration, genre, actors, budget etc., can be used to predict the IMDb score and its gross earnings.
* The goal of this project is to identify the features needed to predict what makes a good movie, defined as ranking highly on the IMDb score, using various methods of machine learning, e.g. logistic regression, KNN, SVM, random forest, bagging and AdaBoosting, based on the book "Python Machine Learning" by Sebastian Raschka. 
* Furthermore the project can show whether the gross earnings will exceed the budget of a movie, hence identifying whether the movie will be profitable or not, and hence if it should be realized or not.
* From the best scoring algorithm one can identify the features to make a good scoring movie.

## Brief description of data
|Attribute |Description |
|:----- |:----- |
|'color' |Whether the movie is in black/white or color |
|'director_name' |Name of director |
|'num_critic_for_reviews' |Number of critics for review |
|'duration' |Duration |
|'director_facebook_likes' |Likes of director's Facebook page |
|'actor_3_facebook_likes' |Likes of actor no. 3's Facebook page |
|'actor_2_name' |Name of actor no. 2 |
|'actor_1_facebook_likes' |Likes of actor no. 1's Facebook page |
|'gross' |Movie's gross earnings |
|'genres' |Movie's genre(s) |
|'actor_1_name' |Name of actor no. 1 |
|'movie_title' |Movie's title |
|'num_voted_users' |Number of voted users |
|'cast_total_facebook_likes' |Total cast's likes on Facebook |
|'actor_3_name' |Name of actor no. 3 |
|'facenumber_in_poster' |Faces of actor's in poster |
|'plot_keywords' |Keywords |
|'movie_imdb_link' |Link to movie |
|'num_user_for_reviews' |Number of users for review |
|'language' |Language |
|'country' |Country of origin |
|'content_rating' |Movie's content rating |
|'budget' |Budget |
|'title_year' |Year of release |
|'actor_2_facebook_likes' |Likes of actor no. 2's Facebook page |
|'imdb_score' |IMDb score |
|'movie_facebook_likes' |Likes of movie's Facebook page |


* Original data consists of 5043 movies and 28 features (5043x28 matrix).
* Data is on IMDb movie ratings contraining features on e.g. director, duration,, genres, actors, language, budget, IMDb score etc. (see [data](data) for full list of features).
* Data is pulled from www.kaggle.com - https://www.kaggle.com/kevalm/movie-imdb that is originally scraped from IMDb's homepage from 20/02-2018.

<img src="https://github.com/RasmusAU/RasmusAU-PHBS_MLF_2018/blob/master/data/Description.png" width="700">

## Data processing
* Drop the meaningless columns
 * Dropping non-numeric columns that are not useful. E.g. director name, actors names.
* Deal with the NAN in data
  * I use mean imputation to estimate the missing values for the features where it makes sense. Mean imputation is a common interpolation technique, where I replace the missing values (NAN) with the mean value of the entire feature column. This is done to minimize the loss of information, and an alternative instead of dropping observations.
  * E.g. If the year of the movie's release is missing, then I drop the observation, since it does not makes sense to estimate this year based on other movies.
* I examine the IMDb scores of movies in the period 2000-2018, hence movies from earlier than 2000 have been dropped. This is done to account for changes in movie budgets, the entrence of the internet, and improvement of technology.

## Data after processing
The mean of the 2041 included movies' IMDb's ratings in the period 2000-2018 is approximately 6.30 and with a standard deviation of approximately 0.99. The distribution looks as the following:
<img src="https://github.com/RasmusAU/RasmusAU-PHBS_MLF_2018/blob/master/data/distribution_imdb_score_done.png" width="400">

The following is a correlation map showing the correlation between the different features:
<img src="https://github.com/RasmusAU/RasmusAU-PHBS_MLF_2018/blob/master/data/correlation_map.png" width="700">

## Partitioning the dataset into training and test dataset
Using SKLearn and test_size=0.4 the dataset is slit into a training dataset and a test dataset.

## Standardization of data
Data is standardized using SKLearn.

## Feature selection
<img src="https://github.com/RasmusAU/RasmusAU-PHBS_MLF_2018/blob/master/data/Feature_importance_RF.png" width="400">
Number of features that meet this threshold criterion: 8
* actor_2_name                   0.135532
* director_name                  0.099928
* num_critic_for_reviews         0.094924
* genres                         0.093145
* actor_1_name                   0.087181
* duration                       0.071415
* director_facebook_likes        0.071251
* actor_1_facebook_likes         0.070462

## Learning curves to assess bias/variance problems 
To improve performance, I look at learning and validation curves:

<img src="https://github.com/RasmusAU/RasmusAU-PHBS_MLF_2018/blob/master/data/Learning_curves.png" width="400">
Based on the learning curves above it is clear that the variance is low, given me no indication of overfitting the data. Hence the model is not too complex for the dataset.
However the bias it relatively high, indicating underfitting, i.e. my model suffers from low performance for unseen data, since the model is not complex enough to capture the patterns in the training data.
In order to adress the problem of the high degree of bias, and hence find a nice bias-variance tradeoff I will try to tune the complexity of the model using regularization. 

## Validation curves to assess over- and underfitting
<img src="https://github.com/RasmusAU/RasmusAU-PHBS_MLF_2018/blob/master/data/Validation_curves.png" width="400">
Validation curves vary the model parameters values instead of plotting training and test accuracies as functions. Also here I get a high bias.

## Models
### Logistic regression
The first model is the simple Logistic regression. It generates a  multi-class model with linear weights, most directly comparable  to  the  feature  weights  given  by linear regression.

Logistic regression = 0.4639

### Support Vector Machine using grid search
Optimization of the hyper-parameter  C  was  done using  grid  search. Grid  search is  exhaustive search through a manually specified subset of the hyper-parameter space. 

SVM = 0.5122

I see that using the exhaustive grid search, a popular hyperparameter optimization technique, improves the model's performance.

### K-nearest neighbor
Using majority voting, the KNN model finds the nearest specified k samples in the training dataset and use majority voting of these samples to classify the new data point.

KNN = 0.4002 (K=100)

### Random forest
I use decision trees for classification. Entropy is a measure of impurity to determine which feature split maximizes the
Information Gain (IG).

Entropy accuracy = 0.4480

The Gini impurity is a criterion that minimizes the probability of misclassification:

Gini accuracy = 0.4308

## Conclusion
The most significant features were found to be the actors names, the name of the director, genre and the number of vritics for review.
The best model to represent the movie features is the Support Vector Machine using grid search.
Not possible to estimate the IMDb ratings well based on the data available on IMDb. This may be caused by multiple factors e.g.:
* Dataset from Kaggle is insufficient and/or not correct.
* Not complex enough features, as indicated by the learning curve, I have high bias.

Further research could include a sentiment analysis of the comments from IMDb, news and social medias impact (coverage of movie premiere).

## References:
* "Python Machine Learning" by Sebastian Raschka.
