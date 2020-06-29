from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import pandas as pd


data = pd.read_csv('titanic.csv')
data = data.drop(['PassengerId', 'Name', 'SibSp', 'Parch',
                  'Ticket', 'Cabin', 'Embarked'], axis=1)
label = LabelEncoder()
dicts = {}
label.fit(data.Sex.drop_duplicates())
dicts['Sex'] = list(label.classes_)
data.Sex = label.transform(data.Sex)

X = data[['Pclass', 'Fare', 'Age', 'Sex']]
X = pd.get_dummies(X)
X = X.fillna({'Age': X.Age.mean()})
y = data.Survived

clf = tree.DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
importances = pd.DataFrame({'features': X.columns, 'feature_importances': clf.feature_importances_}).sort_values('feature_importances', ascending=False)
print(importances)
