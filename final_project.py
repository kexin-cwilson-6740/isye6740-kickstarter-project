import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from sklearn import preprocessing
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score

file = "data/ks-projects-201801.csv"

raw = pd.read_csv(file)

# Initial correlation matrix
corrmat = raw.corr()
top_corr_features = corrmat.index
plt.figure()
g=sns.heatmap(raw[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

def clean_data(data):

    df = data.copy()

    # Remove any rows where there are NaN values
    df = df.dropna()

    #Id is not needed
    df = df.loc[:, df.columns != 'ID']

    # Curency and Country should not be needed
    df = df.loc[:, df.columns != 'currency']
    df = df.loc[:, df.columns != 'country']

    # There are some redundant categories, let's remove 'goal', 'pledged', and 'usd pledged'
    df = df.loc[:, df.columns != 'goal']
    df = df.loc[:, df.columns != 'pledged']
    df = df.loc[:, df.columns != 'usd pledged']

    # Add difference between launched and deadline as project_length_days and remove launched and deadline
    df['project_length_days'] = (pd.to_datetime(df['deadline']) - pd.to_datetime(df['launched'])).dt.days
    df = df.loc[:, df.columns != 'launched']
    df = df.loc[:, df.columns != 'deadline']

    # Let's set the 'category' and 'main_category' into integer values for feature selection purposes.
    le = preprocessing.LabelEncoder()
    df['category'] = le.fit_transform(df['category'])
    df['main_category'] = le.fit_transform(df['main_category'])


    # Possible 'state' values
    # 'failed', 'canceled', 'successful', 'live', 'suspended'
    # 'suspended', 'undefined', and 'live' don't seem to fit into 'successful' or 'fail' classes, so we will remove them
    # We can make the 'live' rows as part of a final test set to see which projects our models think will be successful.

    # For 'state' let's combine 'cancelled' into 'failed'. Let's also transform 'successful' to 1 and 'failed' to 0
    df = df.replace({'state': {'successful': 1, 'failed': 0, 'canceled': 0}})

    # Remove rows where state is 'live', 'undefined', 'suspended'
    df = df.loc[(df['state'] == 1) | (df['state'] == 0)]
    df = df.astype({'state' : 'int64'})

    # Remove any outliers in specific columns ('usd_goal_real')
    df = df[(np.abs(stats.zscore(df['usd_goal_real'])) < 3)]

    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure()
    g=sns.heatmap(corrmat,annot=True)
    plt.show()

    # From corrmat, we see that backers and usd_pledged_real are highly correlated, so we can either choose one, or
    # combine them in some way, or even remove both. If a project is just starting fresh, it won't have any backers
    # or pledged. So it wouldn't make sense to make a prediction based off of that.

    return df


data = clean_data(raw)

# Can do some analysis on the name to see if there any keywords that may indicate success
names_df = data[['name', 'state']]

x_train = data.loc[:, (data.columns != 'state') & (data.columns != 'name')]
y_train = data['state']

# Feature selection
fs = SelectKBest(score_func=mutual_info_classif, k='all')
fs.fit(x_train, y_train)
x_train_fs = fs.transform(x_train)
#print(x_train_fs)

# From this we can see that 'backers' and 'usd_pledged_real' are the best features to predict success, which makes sense.
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()

# What if those are unavailable? Let's say because we are wanting to analyze new projects where there would be no backers or
# pledged amounts yet.
x_train = x_train.loc[:, (x_train.columns != 'backers') & (x_train.columns != 'usd_pledged_real')]

fs = SelectKBest(score_func=mutual_info_classif, k='all')
fs.fit(x_train, y_train)
x_train_fs = fs.transform(x_train)
print(x_train_fs)

# Now 'category' and 'usd_goal_real' seem to be the top two features, in that order
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()

#pca = PCA(n_components=2)
#pca.fit(x_train)
#print(pca.transform(x))



df_num = data.drop(['name'],axis=1)

# Define the feature matrix X and target variable y
y = df_num['state']
y=y.astype('int')
X = df_num.drop(['state'], axis=1)
X2=X

# Downsample dataset
# Get the number of rows in the original dataset
total_rows = X2.shape[0]
sample_size = 5000
# Generate a random sample of row indices
sample_indices = np.random.choice(total_rows, size=sample_size, replace=False)

# Use the sample indices to select the corresponding rows from the original dataset

df_X = pd.DataFrame(X2)
df_y = pd.DataFrame(y)

sampled_df = df_X.iloc[sample_indices]
sampled_y = df_y.iloc[sample_indices]

sampled_y = sampled_y.values.reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(sampled_df, sampled_y, test_size=0.25, random_state=42)


# Define SVM model with a linear kernel - note: SVM is taking too long to compute
#model = SVC(kernel='linear',probability=False)


# fitting neural network

# define the weights:
reg_strengths = [0.001,0.01,0.1,1,10]
models = []
for strength in reg_strengths:
    model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, alpha=strength)
    models.append(model)

# train the model and record error:
n_epochs = 100
n_classes = np.unique(y_train)
train_error = np.zeros((len(reg_strengths), n_epochs))
test_error = np.zeros((len(reg_strengths), n_epochs))
for i, model in enumerate(models):
    for j in range(n_epochs):
        model.partial_fit(X_train, y_train,classes=n_classes)
        train_error[i, j] = 1-model.score(X_train, y_train)
        test_pred = model.predict(X_test)
        test_error[i, j] = 1-accuracy_score(y_test, test_pred)

plt.figure(figsize=(10, 6))
#fig, axs = plt.subplots(6)
#fig, ax = plt.subplots()

for i, strength in enumerate(reg_strengths):
    plt.plot(np.arange(n_epochs), train_error[i], label='Train (decay={})'.format(strength))
    plt.plot(np.arange(n_epochs), test_error[i], label='Test (decay={})'.format(strength))
    plt.ylabel('Error')
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

# use alpha = 1
# redefine the models and search for best hidden layer parameter:

models = []
for unit in range(100):
    model = MLPClassifier(hidden_layer_sizes=(unit+1,), activation='relu', max_iter=10000, alpha=1)
    models.append(model)

test_errors = np.zeros(100)
for i, model in enumerate(models):
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_errors[i] = 1-accuracy_score(y_test, test_pred)
plt.figure(figsize=(10, 6))

plt.plot(test_errors)
plt.ylabel('Error')
plt.xlabel('Unit')
plt.show()

# Based on the plots above, the parameters for neural networks are:
# alpha = 1
# hidden layer = 5

# fit the Random Forest model
rf = RandomForestClassifier(random_state = 42)

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = model.score(test_features,test_labels)
    #print("predictions are", predictions)
    print(model,'Model Performance')
    print('Accuracy = {:0.4f}.'.format(accuracy))

    return accuracy

base_model = RandomForestClassifier(random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

gnb = GaussianNB()
rf = RandomForestClassifier()
lgr = LogisticRegression(random_state=0)
nn = MLPClassifier(alpha=1, hidden_layer_sizes=5,max_iter=100)

from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 6),sharey=True)

common_params = {
    "X": X_train,
    "y": y_train,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}

for ax_idx, estimator in enumerate([gnb,lgr,rf,nn]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"{estimator.__class__.__name__}")
    ax[ax_idx].set_xlabel("Training samples")

# Linear regression
y_lr = X['usd_pledged_real']
x_lr = X.drop(['usd_pledged_real'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(x_lr, y_lr, test_size=0.25, random_state=42)

regr = LinearRegression().fit(X_train, y_train)
y_pred = regr.predict(X_test)
# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
