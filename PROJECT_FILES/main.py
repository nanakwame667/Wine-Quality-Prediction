from utils import load_csv_dataset, pre_process_dataset, train_model, timed_func, test_model

VERBOSE = True

# loading dataset
df = load_csv_dataset('dataset/winequality-white.csv', delimiter=';')

# independent set
X_df = df.iloc[:, :-1]  # get columns 0 TO 11 for each row

# depedent set
Y_df = df['quality']

# preprocessing datatset with StandardScalar
x_train, x_test, y_train, y_test = pre_process_dataset(X_df, Y_df)

# five base models


def display(line):
    if VERBOSE:
        print(line)


def main():
    for model in ['LinearRegression', 'LogisticRegression', 'RandomForest', 'DecisionTree', 'SVM']:

        display(f'Training model {model}')

        def func():
            # training model
            return train_model(model, x_train, y_train)

        trained_model, training_time = timed_func(func)
        display(f'time used: {training_time}')

        display(f'Testing model')
        results, confusion_matrix, accuracy_score = test_model(trained_model, x_test, y_test,
                                                               lambda predictions: [int(i) for i in predictions])
        display(f'confusion_matrix: \n{confusion_matrix}')
        display(f'accuracy_score: {accuracy_score}\n\n')


if __name__ == '__main__':
    main()
