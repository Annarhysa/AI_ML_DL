import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Create a sample student performance dataset
data = {
    'Math': [80, 85, 92, 45, 70, 78, 60, 92, 55, 88],
    'English': [75, 90, 85, 50, 60, 70, 65, 95, 50, 80],
    'Science': [90, 88, 78, 60, 70, 75, 80, 85, 65, 92]
}

df = pd.DataFrame(data)

#Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

#Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(data_scaled)

#Display the clusters
print("Clusters:")
print(df[['Math', 'English', 'Science', 'cluster']])

inertia = kmeans.inertia_

#Display results
print("\nInertia (Within-Cluster Sum of Squares):", inertia)

#clusters
plt.scatter(df['Math'], df['English'], c=df['cluster'], cmap='viridis')
plt.title('K-Means Clustering of Student Performance')
plt.xlabel('Math')
plt.ylabel('English')
plt.show()