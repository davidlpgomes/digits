from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file


def main():
    X, y = load_svmlight_file('features.txt')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    neigh = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

    print('Fitting knn')
    neigh.fit(X_train, y_train)

    print('Predicting...')
    y_pred = neigh.predict(X_test)

    print('Accuracy: ',  neigh.score(X_test, y_test))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
