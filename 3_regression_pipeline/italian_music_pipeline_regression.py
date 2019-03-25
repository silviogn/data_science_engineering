import warnings
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import *

warnings.simplefilter("ignore")

#Loads the DataFrame from the CSV. The CSV was generated from the JSON documents.
df_italian_music = pd.read_csv(r"../0_data/italian_music.csv")


#Replaces the None values for Not a Number value (NaN)
df_italian_music[df_italian_music == 'None'] = np.nan


#Drop the Non a Number values to sanitize the DataFrame

#This method drops the rows with null/Nan values
df_italian_music = df_italian_music.dropna()


#Encodes the values that are not number. Due sci-kit algorithms does not support strings 
lb_maker = LabelEncoder()

fields_to_encode = ['root.artist.region', 'root.artist.genre', 'root.lyrics', 'root.song', 'root.artist.name',
                    'root.artist.artist_id', 'root.id_song']
for name in fields_to_encode:
    df_italian_music[name] = lb_maker.fit_transform(df_italian_music[name])


#Creates the target feature called y
y = df_italian_music['root.musical_features.danceability']

#drop the target feature from the data frame an create X
X = df_italian_music.drop(['root.musical_features.danceability', 'Unnamed: 0'], axis=1).values

# Scale the data to increase the accuracy
# http://scikit-learn.org/stable/modules/preprocessing.html
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
# Often a model will make some assumptions about the distribution
# or scale of your features. Standardization is a way to make your data
# fit these assumptions and improve the algorithm's performance.
# Scaling is a method for standarization of the data.
# X = scale(X)

standard_scaler = StandardScaler()
standard_scaler.fit(X)
X = standard_scaler.transform(X)


#Feature Selection Phase
#Based scikit-learn documentation: http://scikit-learn.org/stable/modules/feature_selection.html
print("Shape of the dataset X BEFORE dimensionality reduction {}".format(X.shape))

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X = model.transform(X)

print("Shape of the dataset X AFTER dimensionality reduction {}".format(X.shape))


#First step of model validation

#Intatiates the Random Forestest Classifier
linear_regression = LinearRegression()

# Perform 10-fold CV
tef_fold_cross_validation_scores = cross_val_score(linear_regression, X, y, cv=10)

print("Tend Folds Stratified Cross Validation Results:{}".format(tef_fold_cross_validation_scores))
print("Tend Folds Stratified Cross Validation Mean Result: {}".format(np.mean(tef_fold_cross_validation_scores)))


# Get metrics such as Recall, Precicion, Accuracy and F1 Score
# Create training and test sets

linear_regression = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the classifier to the training data
linear_regression.fit(X_train, y_train)

y_predicted = linear_regression.predict(X_test)

# Métricas obtenidas de http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

# MAE output is non-negative floating point. The best value is 0.0.
print("Mean Absolute Error: {}".format(mean_absolute_error(y_test, y_predicted)))
# A non-negative floating point value (the best value is 0.0), or an array of floating point values, one for each individual target.
print("Mean Squared Error: {}".format(mean_squared_error(y_test, y_predicted)))
# A non-negative floating point value (the best value is 0.0), or an array of floating point values, one for each individual target.
print("Mean Squared Log Error: {}".format(mean_squared_log_error(y_test, y_predicted)))
# A positive floating point value (the best value is 0.0).
print("Median Absolute Error: {}".format(median_absolute_error(y_test, y_predicted)))

#Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
# A constant model that always predicts the expected value of y, disregarding the input features,
# would get a R^2 score of 0.0.
print("r2_score: {}".format(r2_score(y_test, y_predicted)))
