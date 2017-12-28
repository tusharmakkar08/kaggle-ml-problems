# __author__ = 'tusharmakkar08'
import numpy
import pandas

from sklearn import linear_model, decomposition, datasets

train_data = pandas.DataFrame.from_csv('train.csv')

y_train = train_data['Survived']
train_data['Sex'] = train_data['Sex'].astype('category')
train_data['Sex'] = train_data['Sex'].cat.reorder_categories(['male', 'female'], ordered=True)
train_data['Sex'] = train_data['Sex'].cat.codes

train_data['Age'] = train_data['Age'].fillna(numpy.median(train_data['Age'].fillna(0)))

train_data['Embarked'] = train_data['Embarked'].astype('category')
train_data['Embarked'] = train_data['Embarked'].cat.reorder_categories(['C', 'Q', 'S'], ordered=True)
train_data['Embarked'] = train_data['Embarked'].cat.codes

tickets = []
for value in train_data['Ticket']:
    try:
        int(value)
        tickets.append(0)
    except ValueError:
        tickets.append(1)
train_data['Ticket'] = tickets

designation = []
various_designations = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Don.', 'Col.', 'Dr.', 'Countess.', 'Jonkheer.',
                        'Rev.', 'Capt.', 'Mlle.', 'Major.', 'Lady.', 'Mme.', 'Sir.', 'Ms.']
for value in train_data['Name']:
    for cnt, i in enumerate(various_designations):
        if i in value:
            designation.append(cnt)
            break
train_data['Name'] = designation
train_data['Cabin'] = pandas.isna(train_data['Cabin'])
train_data.pop('Survived')
x_train = train_data

logistic = linear_model.LogisticRegression()
logistic.fit(x_train, y_train)

print(logistic.score(x_train, y_train))

