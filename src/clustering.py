#%%
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sb
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('../data/processed_data.csv')
pd.set_option('display.max_rows', None)
# %%
df.head(20)
# %%
df.isna().sum()
# %%
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)
# %%
scaled_df = pd.DataFrame(x_scaled, columns=df.columns)
scaled_df.head(10)

# %%
#!Elbow method is used to determin the right amount of clusters. It is possible to 
#!identify this number at the inflexion point

inertia = []
k_range = range(1, 15)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)
    print(f'k = {k}, inertia = {kmeans.inertia_:.2f}')
    
#putting more cluster numbers to find the number when is still significant 
#lower the inertia means tigher clusters
# find the point wghere more clusteress returns elbow

#%%

plt.plot(range(1, 15), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.title('Elbow Method For Optimal K')
plt.show()
# %%
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(x_scaled)
# %%
df['Cluster'] = cluster_labels
# %%
#!Doing PCA to visualize in 2D
pca = PCA(n_components=2)
pca_components = pca.fit_transform(x_scaled)

# %%
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
pca_df['cluster'] = cluster_labels
# %%
plt.figure(figsize=(20, 16))
sb.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster', palette='Set2', s=60)
plt.title('Cluster Visuzlized via PCA')
plt.show()
# %%

#!Analisando dados de cada cluster 
cluster_labels = kmeans.labels_
df['cluster'] = cluster_labels

print(df['cluster'].value_counts())
# cluster
# 3    8330
# 4    8290
# 2    8041
# 0    7867
# 1    7506

# %%
cluster_summary = df.groupby('cluster').mean()
print(cluster_summary)
# %%
df['Cluster'] = cluster_labels  # Adiciona os rótulos dos clusters ao DataFrame original

# Escalonamento, mas sem a coluna 'Cluster'
scaled_df = pd.DataFrame(scaler.fit_transform(df.drop(columns=['Cluster'])), columns=df.drop(columns=['Cluster']).columns)

# Reverte a escala dos dados (sem os labels) no DataFrame escalonado
original_values = scaler.inverse_transform(scaled_df)

# Cria o DataFrame original com os dados revertidos
original_df = pd.DataFrame(original_values, columns=df.drop(columns=['Cluster']).columns)

# Adiciona os rótulos dos clusters de volta ao DataFrame revertido
original_df['Cluster'] = df['Cluster']

# Agora você pode comparar os valores revertidos com os clusters
print(original_df.head())  
# %%
#!gender distribution in clusters
gender_distribution = df.groupby(['cluster', 'Gender_Group']).size().unstack().fillna(0)
print(gender_distribution)
# %%
plt.figure(figsize=(10, 6))
sb.countplot(data=df, x='Gender_Group', hue='cluster')
plt.title('Gender distribution')
plt.show()
# %%
#!gender distribution in clusters
game_purchases = df.groupby(['cluster', 'Gender_Group', 'InGamePurchases']).size().reset_index(name='count')

# Pivot to prepare for stacked bar chart
pivot_df = game_purchases.pivot_table(index=['cluster', 'Gender_Group'], columns='InGamePurchases', values='count', fill_value=0)

# Plot
pivot_df.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('In-Game Purchases by Cluster and Gender')
plt.xlabel('Cluster and Gender')
plt.ylabel('Number of Users')
plt.legend(title='InGamePurchases')
plt.tight_layout()
plt.show()
# %%
df['Gender'] = df['Gender_Group'].map({0: 'Male', 1: 'Female'})
df['Location'] = df['Location_Group'].map({0: 'Other', 1: 'Europe', 2: 'Asia', 3: 'USA'})
df['GameGenre'] = df['GameGenre_Group'].map({0: 'Sports', 1: 'Action', 2: 'Strategy', 3: 'Simulation', 4: 'RPG'})
df['GameDifficulty'] = df['GameDifficulty_Group'].map({0: 'Easy', 1: 'Medium', 2: 'Hard'})
df['EngagementLevel'] = df['EngagementLevel_Group'].map({0: 'Low', 1: 'Medium', 2: 'High'})
age_ranges = {
    0: '15-19', 1: '20-24', 2: '25-29', 3: '30-34',
    4: '35-39', 5: '40-44', 6: '45-49'
}
df['AgeRange'] = df['Age_Group'].map(age_ranges)
level_ranges = {
    0: '0-9', 1: '10-19', 2: '20-29', 3: '30-39', 4: '40-49',
    5: '50-59', 6: '60-69', 7: '70-79', 8: '80-89', 9: '90-100'
}
df['PlayerLevelRange'] = df['PlayerLevel_Group'].map(level_ranges)
playtime_ranges = {
    0: '0-4h', 1: '5-8h', 2: '9-12h', 3: '13-16h', 4: '17-20h', 5: '21-24h'
}
df['PlayTimeRange'] = df['PlayTimeHours_Group'].map(playtime_ranges)
sessions_ranges = {
    0: '0-5', 1: '6-10', 2: '11-15', 3: '16-20'
}
df['SessionsPerWeekRange'] = df['SessionsPerWeek_Group'].map(sessions_ranges)
duration_ranges = {
    0: '0-30min', 1: '31-60min', 2: '61-90min', 3: '91-120min',
    4: '121-150min', 5: '151-180min'
}
df['SessionDurationRange'] = df['AvgSessionDurationMinutes_Group'].map(duration_ranges)
achievements_ranges = {
    0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50'
}
df['AchievementsRange'] = df['AchievementsUnlocked_Group'].map(achievements_ranges)

# %%
print(df[['Gender', 'Location', 'GameGenre', 'AgeRange', 'PlayTimeRange', 'cluster']].head())

# %%
# Porcentagem de cada gênero por cluster
gender_counts = df.groupby(['cluster', 'Gender']).size()
gender_dist = gender_counts.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("Gender Distribution by cluster (%):\n", gender_dist)

# Distribuição percentual de Localização por cluster
location_counts = df.groupby(['cluster', 'Location']).size()
location_dist = location_counts.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nLocation Distribution by cluster (%):\n", location_dist)

InGame_Purchases_count = df.groupby(['cluster', 'InGamePurchases']).size()
Purchases_dist = InGame_Purchases_count.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nPurchases Distribution by cluster (%):\n", Purchases_dist)

gameGenre_count = df.groupby(['cluster', 'GameGenre']).size()
gameGenre_dist = gameGenre_count.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nGame Genre Distribution by cluster (%):\n", gameGenre_dist)

GameDifficulty_count = df.groupby(['cluster', 'GameDifficulty']).size()
GameDifficulty_dist = GameDifficulty_count.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nGame Difficulty Distribution by cluster (%):\n", GameDifficulty_dist)

engagementLevel_count = df.groupby(['cluster', 'EngagementLevel']).size()
engagementLevel_dist = engagementLevel_count.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nEngagement Level Distribution by cluster (%):\n", engagementLevel_dist)

age_count = df.groupby(['cluster', 'AgeRange']).size()
age_dist = age_count.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nAge Distribution by cluster (%):\n", age_dist)

playerLevel_count = df.groupby(['cluster', 'PlayerLevelRange']).size()
playerLevel_dist = playerLevel_count.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nPlayer Level Distribution by cluster (%):\n", playerLevel_dist)

playTimeHours_count = df.groupby(['cluster', 'PlayTimeRange']).size()
playTimeHours_dist = playTimeHours_count.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nPlay Time Hours Distribution by cluster (%):\n", playTimeHours_dist)

SessionsPerWeek_count = df.groupby(['cluster', 'SessionsPerWeekRange']).size()
SessionsPerWeek_dist = SessionsPerWeek_count.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nSessions Per Week Distribution by cluster (%):\n", SessionsPerWeek_dist)

AvgSessionDurationMinutes_count = df.groupby(['cluster', 'SessionDurationRange']).size()
AvgSessionDurationMinutes_dist = AvgSessionDurationMinutes_count.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nAvg Session Duration Minutes Distribution by cluster (%):\n", AvgSessionDurationMinutes_dist)

AchievementsUnlocked_count = df.groupby(['cluster', 'AchievementsRange']).size()
AchievementsUnlocked_dist = AchievementsUnlocked_count.groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack()
print("\nAchievements Unlocked Distribution by cluster (%):\n", AchievementsUnlocked_dist)
# %%

#!plot engagement per Cluster

engagementLevel_dist.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
plt.title('Engagement level distribution per cluster')
plt.xlabel('Cluster')
plt.ylabel('Percentage (%)')
plt.legend(title='Engagement Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# %%
