import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

df = pd.read_csv('wine.data', header=None,
                 names=['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                        'Flavanoids',
                        'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                        'OD280/OD315 of diluted wines', 'Proline'])
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
X = scale(df.drop('Class', axis=1))
y = df.Class
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    res = cross_val_score(knn, X, y, cv=kfold)
    print(np.mean(res), k)
