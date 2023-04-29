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

pca = PCA(n_components=2)
pca.fit(x_train)
#print(pca.transform(x))