import warnings
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import *
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


warnings.simplefilter("ignore")

#Loads the DataFrame from the CSV. The CSV was generated from the JSON documents.
df_italian_ds  = pd.read_csv(r"../0_data/italian_music.csv")

#Replaces the None values for Not a Number value (NaN)
df_italian_ds[df_italian_ds == 'None'] = np.nan

#Drop the Non a Number values to sanitize the DataFrame

#Drops the rows with null/Nan values
df_italian_ds  = df_italian_ds.dropna()

#Encodes the values that are not number. Due sci-kit algorithms does not support strings 
lb_maker = LabelEncoder()

fields_to_encode = ['root.artist.region', 'root.artist.genre', 'root.lyrics', 'root.song', 'root.artist.name',
                    'root.artist.artist_id', 'root.id_song']
for name in fields_to_encode:
    df_italian_ds[name] = lb_maker.fit_transform(df_italian_ds
    [name])


#Creates the Target Feature called y
y = df_italian_ds['root.artist.region']

#drop the target feature from the data frame an create X, containing the features to the model induction. 
X = df_italian_ds.drop(['root.artist.region', 'Unnamed: 0'], axis=1).values


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
random_forests = RandomForestClassifier(n_estimators=50)


# Perform 10-fold CV
tef_fold_cross_validation_scores = cross_val_score(random_forests, X, y, cv=50)

print("Tend Folds Stratified Cross Validation Results:{}".format(tef_fold_cross_validation_scores))
print("Tend Folds Stratified Cross Validation Mean Result: {}".format(np.mean(tef_fold_cross_validation_scores)))


# Get metrics such as Recall, Precicion, Accuracy and F1 Score
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the classifier to the training data
random_forests.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_predicted = random_forests.predict(X_test)

# Generate the Classification Report
print("Classification Results Report using train and test splited data:")
print(classification_report(y_test, y_predicted))


# Look for the improvement of the model searching the bes hyperparameter in this case the number of classifiers
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

random_forests = RandomForestClassifier()

random_forests_cv = GridSearchCV(random_forests, param_grid, cv=10)

# Fit it to the training data
random_forests_cv.fit(X, y)

# Print the optimal parameters and best score
print("Tuned Random Forests Parameter: {}".format(random_forests_cv.best_params_))
print("Tuned Random Forests Accuracy: {}".format(random_forests_cv.best_score_))
