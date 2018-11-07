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
* Original data consists of 5043 movies and 28 features (5043x28 matrix).
* Data is on IMDb movie ratings contraining features on e.g. director, duration,, genres, actors, language, budget, IMDb score etc. (see [data](data) for full list of features).
* Data is pulled from www.kaggle.com - https://www.kaggle.com/kevalm/movie-imdb that is originally scraped from IMDb's homepage from 20/02-2018.
* Data will be checked by scraping IMDb's homepage myself.

<img src="https://github.com/RasmusAU/RasmusAU-PHBS_MLF_2018/blob/master/data/Dataset.png" width="400">
<img src="https://github.com/RasmusAU/RasmusAU-PHBS_MLF_2018/blob/master/data/df_head.png" width="400">

## Data processing
* Drop the meaningless columns
** Dropping non-numeric columns that are not useful.
* Deal with the NAN in data
  * I use mean imputation to estimate the missing values for the features where it makes sense. Mean imputation is a common interpolation technique, where I replace the missing values (NAN) with the mean value of the entire feature column. This is done to minimize the loss of information, and an alternative instead of dropping observations.
  * E.g. If the year of the movie's release is missing, then I drop the observation, since it does not makes sense to estimate this year based on other movies.
* I examine the IMDb scores of movies in the period 2002-2018, hence movies from earlier than 2000 have been dropped. This is done to account for changes in movie budgets, the entrence of the internet, and improvement of technology.

## Data after processing
The mean of all included movies' IMDb's ratings in the period 2000-2018 is approximately 6.31. The distribution looks as the following:
<img src="https://github.com/RasmusAU/RasmusAU-PHBS_MLF_2018/blob/master/data/distribution_imdb_score_done.png" width="400">

The following is a correlation map showing the correlation between the different features:
<img src="https://github.com/RasmusAU/RasmusAU-PHBS_MLF_2018/blob/master/data/correlation_map.png" width="400">

## References:
* "Python Machine Learning" by Sebastian Raschka.
