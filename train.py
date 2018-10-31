import pickle
from ensemble import AdaBoostClassifier
from feature import feature1, feature2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import datetime

if __name__ == "__main__":
    # extract the features
    feature1()
    feature2()

    face_features = []
    nonface_features = []

    train_size = 160
    test_size = 40

    print('Loading Training data ...')
    with open('face2.pkl', 'rb') as face:
        for i in range(train_size + test_size):
            face_features.append(pickle.load(face))
    X_train = face_features[:train_size]
    with open('nonface2.pkl', 'rb') as nonface:
        for i in range(train_size + test_size):
            nonface_features.append(pickle.load(nonface))
    X_train.extend(nonface_features[:train_size])
    print('Training data loading finish')
    
    
    y_train = [1 for _ in range(train_size)]
    y_train.extend([-1 for _ in range(train_size)])

    ada = AdaBoostClassifier(DecisionTreeClassifier, 5)

    # train model
    print('Fitting model ...')
    begin_time = datetime.datetime.now()
    ada.fit(X_train, y_train)
    end_time = datetime.datetime.now()
    print('Fit model finish, Training Time cost is: ', end_time - begin_time)
    del X_train, y_train

    # load test data
    print('Loading Test data ...')
    X_test = face_features[train_size:train_size+test_size]
    X_test.extend(nonface_features[train_size:train_size+test_size])
    y_test = [1 for _ in range(test_size)]
    y_test.extend([-1 for _ in range(test_size)])
    print('Test data loading finish')

    # model predict
    print('Model predicting ...')
    begin_time = datetime.datetime.now()
    y_pred = ada.predict(X_test)
    end_time = datetime.datetime.now()
    print('Predicting Time cost is: ', end_time - begin_time)
    print('************************************')
    print(classification_report(y_test, y_pred, target_names=['face', 'non face']))
