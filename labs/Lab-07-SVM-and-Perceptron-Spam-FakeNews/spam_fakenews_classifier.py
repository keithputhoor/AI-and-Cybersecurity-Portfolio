import pandas as pd
import numpy as np

def using_perseptron():

    print("FAKE NEWS DETECTION WITH PERCEPTRON")
    df_perc = pd.read_csv('fake_news_perceptron.csv')
    y = df_perc.iloc[:, 0].values
    y = np.where(y == 'fake', -1, 1) # -1 fake, 1 real
    X = df_perc.iloc[:, [1,2]].values

    real = 0
    fake = 0
    for i in y:
        if i == 1:
            real = real+1
        else:
            fake = fake+1

    print(f"Perceptron Dataset Shape: {df_perc.shape}")
    print(f"Class Distribution - Real: {real}, Fake: {fake}")


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=30)

    from sklearn.linear_model import Perceptron
    p = Perceptron(max_iter = 50, eta0 = 0.1, random_state = 30)
    p.fit(X_train, y_train)

    y_pred = p.predict(X_test)

    from sklearn.metrics import accuracy_score
    print("\nPerceptron Results:")
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


def using_svm():
    print("\nFAKE NEWS DETECTION WITH SVM")

    df_svm = pd.read_csv('fake_news_svm.csv')

    y = df_svm.iloc[:, 0].values
    y = np.where(y == 'fake', -1, 1) # -1 fake, 1 real
    X = df_svm.iloc[:, [1,4]].values

    real = 0
    fake = 0
    for i in y:
        if i == 1:
            real = real + 1
        else:
            fake = fake + 1

    print(f"SVM Dataset Shape: {df_svm.shape}")
    print(f"Class Distribution - Real: {real}, Fake: {fake}")

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 30)

    from sklearn.svm import SVC
    svm = SVC(kernel = 'linear', C = 1.0, random_state = 30)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    from sklearn.metrics import accuracy_score
    print("\nSVM Linear Kernel Results:")
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


using_perseptron()
using_svm()