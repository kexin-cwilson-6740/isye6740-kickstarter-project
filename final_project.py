import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn import preprocessing
file = "data/ks-projects-201801.csv"

raw = pd.read_csv(file)

# Initial correlation matrix
corrmat = raw.corr()
top_corr_features = corrmat.index
plt.figure()
g=sns.heatmap(raw[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.show()

def clean_data(data):

    df = data.copy()
    
    #Id is not needed
    df = df.loc[:, df.columns != 'ID']

    # There are some redundant categories, let's remove 'goal', 'pledged', and 'usd pledged'
    df = df.loc[:, df.columns != 'goal']
    df = df.loc[:, df.columns != 'pledged']
    df = df.loc[:, df.columns != 'usd pledged']
    
    # Add difference between launched and deadline as project_length_days
    df['project_length_days'] = (pd.to_datetime(df['deadline']) - pd.to_datetime(df['launched'])).dt.days

    # Let's set the 'category' and 'main_category' into integer values for feature selection purposes.
    le = preprocessing.LabelEncoder()
    df['category'] = le.fit_transform(df['category'])
    df['main_category'] = le.fit_transform(df['main_category'])

    # For 'state' let's combine 'cancelled' into 'failed'. Let's also transform 'successful' to 1 and 'failed' to 0
    df = df.replace({'state': {'successful': 1, 'failed': 0, 'canceled': 0}})

    # Remove rows where state is 'live', 'undefined', 'suspended'
    df = df.loc[(df['state'] == 1) | (df['state'] == 0)]

    return df


data = clean_data(raw)

# Can do some analysis on the name to see if there any keywords that may indicate success
names_df = data[['name', 'state']]

x = data.loc[:, data.columns != 'state']
y = data['state']

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure()
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()