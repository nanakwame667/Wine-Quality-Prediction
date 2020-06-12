import time
import pandas as pd

# models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_csv_dataset(csv_path=None, delimiter=','):
    df = pd.read_csv(csv_path, delimiter=delimiter)
    return df


def pre_process_dataset(samples, labels, test_ratio=0.3, random_state=0, scaler=StandardScaler()):
    x_train, x_test, y_train, y_test = train_test_split(samples, labels,
                                                        test_size=test_ratio,
                                                        random_state=random_state)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def train_model(model, x_train, y_train, shuffle_data=True, **kwargs):
    if model == 'LinearRegression':
        model = LinearRegression()
    elif model == 'LogisticRegression':
        model = LogisticRegression()
    elif model == 'RandomForest':
        model = RandomForestClassifier()
    elif model == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif model == 'SVM' or model == 'SVC':
        model = SVC()

    if shuffle_data:
        x_train, y_train = shuffle(x_train, y_train)

    return model.fit(x_train, y_train, **kwargs)


def test_model(model, x_test, y_test, process_result=None):
    result = model.predict(x_test)
    if process_result is not None:
        result = process_result(result)

    if y_test is not None:
        cm = confusion_matrix(y_test, result)
        asc = accuracy_score(y_test, result)
        return result, cm, asc

    return result


def timed_func(func):
    start_time = time.time()
    result = func()
    end_time = time.time()
    time_used = end_time - start_time
    return result, time_used
