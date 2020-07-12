import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

df_train = pd.read_csv('salary-train.csv')
df_test = pd.read_csv('salary-test-mini.csv')

X_train = df_train.drop(['SalaryNormalized'], axis=1)
X_train['FullDescription'] = X_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
X_train = X_train.apply(lambda x: x.astype(str).str.lower())
y_train = df_train['SalaryNormalized']

X_test = df_test.drop(['SalaryNormalized'], axis=1)
y_test = df_test['SalaryNormalized']

X_train['LocationNormalized'].fillna('nan', inplace=True)
X_train['ContractTime'].fillna('nan', inplace=True)

tfid = TfidfVectorizer(min_df=5)
X_train_vect = tfid.fit_transform(X_train['FullDescription'])
X_test_vect = tfid.transform(X_test['FullDescription'])

enc = DictVectorizer()
X_train_categ = enc.fit_transform(X_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(X_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_for_train = hstack([X_train_vect, X_train_categ])
X_for_test = hstack([X_test_vect, X_test_categ])

clf = Ridge(alpha=1)
clf.fit(X_for_train, y_train)
print(clf.predict(X_for_test))
