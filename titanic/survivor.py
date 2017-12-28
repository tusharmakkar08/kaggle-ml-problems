# __author__ = 'tusharmakkar08'
import numpy
import pandas

from sklearn.model_selection import train_test_split, cross_val_score

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

    tickets = []
    for value in x_var_data['Ticket']:
        try:
            int(value)
            tickets.append(0)
        except ValueError:
            tickets.append(1)
    x_var_data['Ticket'] = tickets

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
    x_var_data['Name'] = designation
    x_var_data['Cabin'] = pandas.isna(x_var_data['Cabin'])
    return x_var_data

x_data = pandas.DataFrame.from_csv('train.csv')
x_data = convert_data(x_data)
y_data = x_data['Survived']
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


x_data_test = pandas.DataFrame.from_csv('test.csv')
x_data_test = convert_data(x_data_test)
y_pred = logistic.predict(x_data_test)
pandas.DataFrame({
    'PassengerId': x_data_test.index,
    'Survived': y_pred,
}).set_index("PassengerId").to_csv('titanic_pred.csv')
