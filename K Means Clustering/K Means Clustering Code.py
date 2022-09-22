import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Getting Data
college_data = pd.read_csv("College_Data", index_col = 0)

print(college_data.head())
print()
print(college_data.describe())
print()
print(college_data.info())
print()

# EDA
plt.figure(figsize=(10,6))
sns.scatterplot(x = "Room.Board", y = "Grad.Rate", data = college_data, hue = "Private")

plt.figure(figsize=(10,6))
sns.scatterplot(x = "Outstate", y = "F.Undergrad", data = college_data, hue = "Private")

g = sns.FacetGrid(college_data, hue = "Private", palette = 'coolwarm', height = 6, aspect = 2)
g = g.map(plt.hist, 'Outstate', bins = 20, alpha = 0.7)

g = sns.FacetGrid(college_data, hue="Private", palette='coolwarm', height=6, aspect=2)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)

print("College with more than 100% Grad Rate:")
print(college_data[college_data["Grad.Rate"]>100])
print()
college_data["Grad.Rate"]["Cazenovia College"] = 100

g = sns.FacetGrid(college_data,hue="Private",palette='coolwarm',height=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

# Training Model
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(college_data.drop("Private", axis=1))

print("Cluster Centers:")
print(kmeans.cluster_centers_)
print()

# Evaluation

# New Column called Cluster. 1 for Private, 0 for Public
def converter(value):
    return (value=="Yes")*1

college_data["Cluster"] = college_data["Private"].apply(converter)
print(college_data.head())
print()

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(college_data["Cluster"], kmeans.labels_))
print()
print(classification_report(college_data["Cluster"], kmeans.labels_))
print()






























