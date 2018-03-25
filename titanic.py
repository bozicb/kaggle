import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer

imputer = Imputer()

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = train.Survived
#feature_cols = ['Pclass','Sex','Age','Fare','Cabin','Embarked']
feature_cols = ['Age','Fare']
X_train = train[feature_cols]
X_test = test[feature_cols]

imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()
cols_with_missing = (col for col in X_train.columns if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
imputed_X_train_plus = imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = imputer.transform(imputed_X_test_plus)

model = RandomForestRegressor()
model.fit(imputed_X_train_plus,y_train)

predictions = np.around(model.predict(imputed_X_test_plus))

submission = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':predictions})
submission.to_csv('submission.csv',index=False)
