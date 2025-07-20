# K NEAREST NEIGHBOUR

# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print(os.listdir())
data = pd.read_csv('Social_Network_Ads.csv')
print(data.head(10))

# split your dataset
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

# prepare data for test and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Move to KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(X_train, y_train)

# get your predictor
y_predictor = classifier.predict(X_test)

# In classifiers, you will be creating a confusion matrix a lot
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predictor)

# plotting a graph
X_point, Y_point = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_point[:, 0].min() - 1, stop=X_point[:, 0].max() + 1, step=0.01),
    np.arange(start=X_point[:, 1].min() - 1, stop=X_point[:, 1].max() + 1, step=0.01)
)
plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75, cmap=ListedColormap(('green', 'blue'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_point)):
    plt.scatter(
        X_point[y_point == j, 0], X_point[y_point == j, 1],
        c=ListedColormap(('green', 'blue'))(i), label=j
    )

plt.title('K-NN Training set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# repeat plotting for the test set
X_point, y_point = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_point[:, 0].min() - 1, stop=X_point[:, 0].max() + 1, step=0.01),
    np.arange(start=X_point[:, 1].min() - 1, stop=X_point[:, 1].max() + 1, step=0.01)
)

plt.contourf(
    X1, X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75, cmap=ListedColormap(('green', 'blue'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_point)):
    plt.scatter(
        X_point[y_point == j, 0], X_point[y_point == j, 1],
        c=ListedColormap(('green', 'blue'))(i), label=j
    )

plt.title('K-NN Test set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()






# DECISION TREE

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Load dataset
data = pd.read_csv('groupStudy.csv')
print(data.head(10))  # Always good to confirm data loaded correctly

# Extract features and labels
X = data.iloc[:, 1:2].values    # Hours
y = data.iloc[:, 2].values      # Marks

# Fit model
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# Predict for a new value
y_predicted = regressor.predict([[20]])
print("Predicted marks for 20 hours of study:", y_predicted[0])

