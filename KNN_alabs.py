import pandas as pd
import numpy as np
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

# Configure logging
logging.basicConfig(
    filename='Knn.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load dataset
data = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\Machine Learning\self study\titanic knn\tested.csv")

# Drop unnecessary columns
data.drop(columns=['PassengerId', 'Survived', 'Cabin', 'Name', 'Ticket'], inplace=True)

# Process numerical and categorical data
df_num = data.select_dtypes(include=['int', 'float'])
df_cat = data.select_dtypes(include=['object'])

# Fill missing values
df_num = df_num.apply(lambda x: x.fillna(x.mean()), axis=0)  # Fill numerical with mean
df_cat = df_cat.apply(lambda x: x.fillna(x.mode()[0]), axis=0)  # Fill categorical with mode

# Clip numerical values to remove outliers
df_num = df_num.apply(lambda x: x.clip(upper=x.quantile(0.995), lower=x.quantile(0.005)), axis=0)

# Convert categorical variables to dummy/one-hot encoding
df_cat = pd.get_dummies(df_cat, drop_first=True).astype(int)

# Combine processed data
df_processed = pd.concat([df_num, df_cat], axis=1)

# Round 'Age' to 2 decimal places if it exists
if 'Age' in df_processed.columns:
    df_processed['Age'] = df_processed['Age'].round(2)

# Define features and target
x = df_processed.drop('Pclass', axis=1)
y = df_processed['Pclass']

# Split data into train and test sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=123)

# Perform GridSearchCV
knn = KNeighborsClassifier()
grid_params = {'n_neighbors': list(range(3, 10))}
knn_clf = GridSearchCV(knn, grid_params, cv=10, scoring='accuracy')
knn_clf.fit(train_x, train_y.astype(int))

logging.info(f"Best parameters (GridSearchCV): {knn_clf.best_params_}")

# Train model with best parameters from GridSearchCV
k = KNeighborsClassifier(n_neighbors=knn_clf.best_params_['n_neighbors'])
k.fit(train_x, train_y.astype(int))

# Evaluate the model
train_acc = accuracy_score(train_y.astype(int), k.predict(train_x))
test_acc = accuracy_score(test_y.astype(int), k.predict(test_x))
logging.info(f"Train accuracy: {train_acc}")
logging.info(f"Test accuracy: {test_acc}")

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(knn, grid_params, n_jobs=-1, cv=10, random_state=123, n_iter=5, scoring='accuracy')
random_search.fit(train_x, train_y.astype(int))
logging.info(f"Best parameters (RandomizedSearchCV): {random_search.best_params_}")

# Perform BayesSearchCV
bayes_params = {'n_neighbors': (2, 100)}  # Corrected the range
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
bayes_search = BayesSearchCV(estimator=KNeighborsClassifier(), search_spaces=bayes_params, cv=cv, n_jobs=-1)
bayes_search.fit(train_x, train_y.astype(int))

logging.info(f"Best score (BayesSearchCV): {bayes_search.best_score_}")
logging.info(f"Best parameters (BayesSearchCV): {bayes_search.best_params_}")
