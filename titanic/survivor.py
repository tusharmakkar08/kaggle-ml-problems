# __author__ = 'tusharmakkar08'
import numpy
import pandas

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import linear_model


def convert_data(x_var_data):
    x_var_data['Sex'] = x_var_data['Sex'].astype('category')
    x_var_data['Sex'] = x_var_data['Sex'].cat.reorder_categories(['male', 'female'], ordered=True)
    x_var_data['Sex'] = x_var_data['Sex'].cat.codes
    x_var_data['Age'] = x_var_data['Age'].fillna(numpy.median(x_var_data['Age'].fillna(0)))
    x_var_data['Fare'] = x_var_data['Fare'].fillna(numpy.median(x_var_data['Fare'].fillna(0)))

    x_var_data['Embarked'] = x_var_data['Embarked'].astype('category')
    x_var_data['Embarked'] = x_var_data['Embarked'].cat.reorder_categories(['C', 'Q', 'S'], ordered=True)
    x_var_data['Embarked'] = x_var_data['Embarked'].cat.codes

    designation = []
    various_designations = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Don.', 'Col.', 'Dr.', 'Countess.', 'Jonkheer.',
                            'Rev.', 'Capt.', 'Mlle.', 'Major.', 'Lady.', 'Mme.', 'Sir.', 'Ms.', 'Dona.']
    for value in x_var_data['Name']:
        flag_false = True
        for cnt, i in enumerate(various_designations):
            if i in value:
                flag_false = False
                designation.append(cnt)
                break
        if flag_false:
            designation.append(len(various_designations))
    x_var_data['Name_len'] = x_var_data['Name'].apply(lambda x: len(x))
    x_var_data['Name'] = designation
    x_var_data['Cabin'] = pandas.isnull(x_var_data['Cabin'])
    x_var_data['FamSize'] = numpy.where((x_var_data['SibSp'] + x_var_data['Parch']) == 0, 0,
                                        numpy.where((x_var_data['SibSp'] + x_var_data['Parch']) <= 3, 1, 2))
    return x_var_data


x_data = pandas.DataFrame.from_csv('train.csv')
x_data = convert_data(x_data)
y_data = x_data['Survived']
x_data.pop('Ticket')
x_data.pop('SibSp')
x_data.pop('Parch')
x_data.pop('Age')
# print(x_data.corr())
x_data.pop('Survived')

logistic = linear_model.LogisticRegression()

# Cross Validation score
cross_scores = cross_val_score(logistic, x_data, y_data, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (cross_scores.mean(), cross_scores.std() * 2))

# Normal breaking test size
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=0)
logistic.fit(X_train, y_train)
print("Training data result %s" % logistic.score(X_train, y_train))
print("Testing data result %s" % logistic.score(X_test, y_test))

random_forest = RandomForestClassifier(**{'criterion': 'gini', 'min_samples_leaf': 1,
                                          'min_samples_split': 12, 'n_estimators': 50})

# param_grid = {"criterion": ["gini", "entropy"],
#               "min_samples_leaf": [1, 5, 10], "min_samples_split": [2, 4, 10, 12, 16],
#               "n_estimators": [50, 100, 400, 700, 1000]}
#
# gs = GridSearchCV(estimator=random_forest, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
#
# gs = gs.fit(X_train, y_train)
#
# print(gs.best_score_)
# print(gs.best_params_)
# print(gs.cv_results_)

# Cross Validation score
cross_scores = cross_val_score(random_forest, x_data, y_data, cv=10)
print("Accuracy RF: %0.2f (+/- %0.2f)" % (cross_scores.mean(), cross_scores.std() * 2))

# Normal breaking test size
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=0)
random_forest.fit(X_train, y_train)
print("Training data RF result %s" % random_forest.score(X_train, y_train))
print("Testing data RF result %s" % random_forest.score(X_test, y_test))
x_data_test = pandas.DataFrame.from_csv('test.csv')
x_data_test = convert_data(x_data_test)
x_data_test.pop('Ticket')
x_data_test.pop('SibSp')
x_data_test.pop('Parch')
x_data_test.pop('Age')
y_pred = logistic.predict(x_data_test)
pandas.DataFrame({
    'PassengerId': x_data_test.index,
    'Survived': y_pred,
}).set_index("PassengerId").to_csv('titanic_pred.csv')

y_pred = random_forest.predict(x_data_test)
pandas.DataFrame({
    'PassengerId': x_data_test.index,
    'Survived': y_pred,
}).set_index("PassengerId").to_csv('titanic_pred_rf.csv')
xgb = xgb.XGBClassifier(**{'base_score': 0.85, 'colsample_bytree': 0.8, 'learning_rate': 0.025,
                           'max_depth': 12, 'n_estimators': 200, 'subsample': 0.8})

# param_grid = {
#         'max_depth': [3, 6, 9, 12],
#         'subsample': [0.8, 0.9, 1.0],
#         'colsample_bytree': [0.8, 0.9, 1.0],
#         'n_estimators': [100, 200, 300, 500],
#         'learning_rate': [0.025, 0.05, 0.1],
#         'base_score': [0.25, 0.5, 0.75, 0.85],
#     }
# gs = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
# gs = gs.fit(X_train, y_train)
# print(gs.best_params_)

cross_scores = cross_val_score(xgb, x_data, y_data, cv=10)
print("Accuracy xgb: %0.2f (+/- %0.2f)" % (cross_scores.mean(), cross_scores.std() * 2))
xgb.fit(X_train, y_train)
print("Training data xgb result %s" % xgb.score(X_train, y_train))
print("Testing data xgb result %s" % xgb.score(X_test, y_test))
y_pred = xgb.predict(x_data_test)
pandas.DataFrame({
    'PassengerId': x_data_test.index,
    'Survived': y_pred,
}).set_index("PassengerId").to_csv('titanic_pred_xgb.csv')

gbm = GradientBoostingClassifier(**{'learning_rate': 0.1, 'max_depth': 4, 'max_features': 1.0, 'min_samples_leaf': 20})
# gb_grid_params = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
#                   'max_depth': [4, 6, 8],
#                   'min_samples_leaf': [20, 50, 100, 150],
#                   'max_features': [1.0, 0.3, 0.1]}
# gs = GridSearchCV(estimator=gbm, param_grid=gb_grid_params, scoring='accuracy', cv=3, n_jobs=-1)
# gs = gs.fit(X_train, y_train)
# print(gs.best_params_)

cross_scores = cross_val_score(gbm, x_data, y_data, cv=10)
print("Accuracy extra gbm: %0.2f (+/- %0.2f)" % (cross_scores.mean(), cross_scores.std() * 2))
gbm.fit(X_train, y_train)
print("Training data gbm result %s" % gbm.score(X_train, y_train))
print("Testing data gbm result %s" % gbm.score(X_test, y_test))
y_pred = gbm.predict(x_data_test)
pandas.DataFrame({
    'PassengerId': x_data_test.index,
    'Survived': y_pred,
}).set_index("PassengerId").to_csv('titanic_pred_gbm.csv')

ensemble_classifier = VotingClassifier([('rf', random_forest), ('xgb', xgb), ('gbm', gbm)],
                                       voting='soft', weights=[2, 1, 3])
cross_scores = cross_val_score(ensemble_classifier, x_data, y_data, cv=10)
print("Accuracy ensemble: %0.2f (+/- %0.2f)" % (cross_scores.mean(), cross_scores.std() * 2))

ensemble_classifier.fit(X_train, y_train)
print("Training data ensemble result %s" % ensemble_classifier.score(X_train, y_train))
print("Testing data ensemble result %s" % ensemble_classifier.score(X_test, y_test))
y_pred = ensemble_classifier.predict(x_data_test)
pandas.DataFrame({
    'PassengerId': x_data_test.index,
    'Survived': y_pred,
}).set_index("PassengerId").to_csv('titanic_pred_ensemble.csv')
