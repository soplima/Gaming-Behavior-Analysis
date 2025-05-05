#%%
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# %%
df = pd.read_csv('../data/online_gaming_behavior_dataset.csv')
pd.set_option('display.max_rows', None)
# %%
df.head()
# %%
df.info()
# %%
df['Gender'].value_counts()
# %%
#! transforming gender info nummerical data to do clustering
df['Gender_Group'] = df['Gender'].map({'Male': 0, 'Female': 1})
# %%
df['Gender'].value_counts()
# %%
# %%
df['Location'].value_counts()
# %%
#! transforming location info nummerical data to do clustering
df['Location_Group'] = df['Location'].map({'Other': 0, 'Europe': 1, 'Asia': 2, 'USA': 3})
# %%
df['Location'].value_counts()
# %%
df['GameGenre'].value_counts()
# %%
#! transforming Game Genre info nummerical data to do clustering
df['GameGenre_Group'] = df['GameGenre'].map({'Sports': 0, 'Action': 1, 'Strategy': 2, 'Simulation': 3, 'RPG': 4})
df['GameGenre_Group'].value_counts()
# %%
df['GameDifficulty'].value_counts()
# %%
#! transforming Game Difficulty info nummerical data to do clustering
df['GameDifficulty_Group'] = df['GameDifficulty'].map({'Easy': 0, 'Medium': 1, 'Hard': 2})
df['GameDifficulty_Group'].value_counts()
# %%
df['EngagementLevel'].value_counts()
# %%
#! transforming Engagement Level info nummerical data to do clustering
df['EngagementLevel_Group'] = df['EngagementLevel'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['EngagementLevel_Group'].value_counts()
# %%
#! transforming age into age bands of 5
df['Age'].value_counts()
df['Age'].describe()
bins = [0, 19, 24, 29, 34, 39, 44, 49]
labels = [0, 1, 2, 3, 4, 5, 6]

df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
df['Age_Group'].isna().sum()
# %%
#! transforming Player Level into bands of 10
df['PlayerLevel'].describe()
bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 100]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

df['PlayerLevel_Group'] = pd.cut(df['PlayerLevel'], bins=bins, labels=labels, right=True, include_lowest=True)
# %%
#! transforming Play Time Hours into bands of 10
df['PlayTimeHours'].describe()
bins = [0, 4, 8, 12, 16, 20, 24]
labels = [0, 1, 2, 3, 4, 5]

df['PlayTimeHours_Group'] = pd.cut(df['PlayTimeHours'], bins=bins, labels=labels, right=True, include_lowest=True)

df.head()


#! transforming Sessions Per Week into bands of 10
#%%
df['SessionsPerWeek'].value_counts()
#%%
df['SessionsPerWeek'].describe()
# %%
bins = [0, 5, 10, 15, 20]
labels = [0, 1, 2, 3]

df['SessionsPerWeek_Group'] = pd.cut(df['SessionsPerWeek'], bins=bins, labels=labels, right=True, include_lowest=True)
df.head()
#%%
df['SessionsPerWeek_Group'].value_counts()
#%%
#! transforming AvgSessionDurationMinutes into bands of 30 MIN
df['AvgSessionDurationMinutes'].describe()

bins = [0, 30, 60, 90, 120, 150, 180]
labels = [0, 1, 2, 3, 4, 5]

df['AvgSessionDurationMinutes_Group'] = pd.cut(df['AvgSessionDurationMinutes'], bins=bins, labels=labels, right=True, include_lowest=True)
df.head()
#%%
df['AchievementsUnlocked'].describe()

bins = [0, 10, 20, 30, 40, 50]
labels = [0, 1, 2, 3, 4]

df['AchievementsUnlocked_Group'] = pd.cut(df['AchievementsUnlocked'], bins=bins, labels=labels, right=True, include_lowest=True)
df.head()

#*All the data has been discretized

df.drop(columns=['PlayerID', 'Age', 'Gender', 
                'Location', 'GameGenre', 'PlayTimeHours',
                'GameDifficulty', 'SessionsPerWeek',
       'AvgSessionDurationMinutes', 'PlayerLevel',
       'AchievementsUnlocked','EngagementLevel'
]).to_csv('../data/processed_data.csv', index=False)

# %%
df.columns
# %%
