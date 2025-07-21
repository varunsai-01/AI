import pandas as pd
df = pd.read_csv('path.csv')
df.head()
#upload file
import numpy as np
from scipy import stats

data = df['Score']  # We’re analyzing the “Score” column

mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)
std_dev = np.std(data)
variance = np.var(data)
percentile_25 = np.percentile(data, 25)
percentile_75 = np.percentile(data, 75)

# Print all results
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"25th Percentile: {percentile_25}")
print(f"75th Percentile: {percentile_75}")


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.figure(figsize=(6,4))
sns.histplot(data, kde=True)
plt.title('Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# Boxplot
plt.figure(figsize=(4,4))
sns.boxplot(y=data)
plt.title('Boxplot of Scores')
plt.ylabel('Score')
plt.show()

# Pairplot for all numerical features
sns.pairplot(df)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()
