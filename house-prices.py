import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

imputer = Imputer()

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = train.SalePrice
X_train = pd.get_dummies(train)
X_test = pd.get_dummies(test)
X_train, X_test = X_train.align(X_test,join='inner',axis=1)

imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()
cols_with_missing = (col for col in X_train.columns if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
imputed_X_train_plus = imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = imputer.fit_transform(imputed_X_test_plus)

model = XGBRegressor(n_estimators=1000,learning_rate=.05)
model.fit(imputed_X_train_plus,y_train,early_stopping_rounds=5,eval_set=[(imputed_X_train_plus, y_train)])

predictions = model.predict(imputed_X_test_plus)



my_submission = pd.DataFrame({'Id':test.Id,'SalePrice':predictions})
my_submission.to_csv('submission.csv',index=False)

