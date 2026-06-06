# ============================================
# CUSTOMER SEGMENTATION USING K-MEANS
# UNSUPERVISED LEARNING PROJECT
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

# ============================================
# LOAD DATASET
# ============================================

print("Loading Dataset...")

df = pd.read_csv("dataset/Mall_Customers.csv")

print("\nDataset Loaded Successfully!\n")

# Display first 5 rows
print(df.head())

# Dataset Information
print("\nDataset Info:\n")
print(df.info())

# Check Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# ============================================
# DATA VISUALIZATION
# ============================================

print("\nGenerating Visualizations...")

# Income vs Spending Plot
plt.figure(figsize=(8,5))

sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    data=df
)

plt.title("Customer Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.savefig("outputs/customer_distribution.png")

plt.show()

# ============================================
# SELECT FEATURES
# ============================================

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ============================================
# FIND OPTIMAL NUMBER OF CLUSTERS
# ELBOW METHOD
# ============================================

print("\nFinding Optimal Clusters using Elbow Method...")

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8,5))

plt.plot(range(1, 11), wcss, marker='o')

plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

plt.savefig("outputs/elbow_method.png")

plt.show()

# ============================================
# APPLY K-MEANS
# ============================================

print("\nApplying K-Means Clustering...")

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

# Predict Clusters
y_kmeans = kmeans.fit_predict(X)

print("\nClustering Completed!")

# ============================================
# ADD CLUSTER COLUMN
# ============================================

df['Cluster'] = y_kmeans

print("\nClustered Dataset:\n")
print(df.head())

# ============================================
# VISUALIZE CLUSTERS
# ============================================

print("\nGenerating Cluster Visualization...")

plt.figure(figsize=(10,7))

# Cluster 1
plt.scatter(
    X.iloc[y_kmeans == 0, 0],
    X.iloc[y_kmeans == 0, 1],
    s=100,
    label='Cluster 1'
)

# Cluster 2
plt.scatter(
    X.iloc[y_kmeans == 1, 0],
    X.iloc[y_kmeans == 1, 1],
    s=100,
    label='Cluster 2'
)

# Cluster 3
plt.scatter(
    X.iloc[y_kmeans == 2, 0],
    X.iloc[y_kmeans == 2, 1],
    s=100,
    label='Cluster 3'
)

# Cluster 4
plt.scatter(
    X.iloc[y_kmeans == 3, 0],
    X.iloc[y_kmeans == 3, 1],
    s=100,
    label='Cluster 4'
)

# Cluster 5
plt.scatter(
    X.iloc[y_kmeans == 4, 0],
    X.iloc[y_kmeans == 4, 1],
    s=100,
    label='Cluster 5'
)

# Plot Centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='black',
    label='Centroids'
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')

plt.legend()

plt.savefig("outputs/customer_clusters.png")

plt.show()

# ============================================
# SAVE CLUSTERED DATASET
# ============================================

df.to_csv("outputs/clustered_customers.csv", index=False)

print("\nClustered Dataset Saved!")

# ============================================
# FINAL MESSAGE
# ============================================

print("\n===================================")
print("PROJECT EXECUTED SUCCESSFULLY")
print("===================================")

print("\nOutput Files Saved in outputs/ Folder")