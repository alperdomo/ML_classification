import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score

def define_strategy(X_train, y_train):
    plt = matplotlib.pyplot.gcf()
    plt.set_size_inches(12, 8)
    strats = ['stratified', 'most_frequent', 'prior', 'uniform', 'constant']
    train_dummy_scores = {}

    for clfs in strats:
        if clfs == 'constant':
            dummy_clf = DummyClassifier(strategy = clfs, random_state = 0, constant = 0)
        else:
            dummy_clf = DummyClassifier(strategy = clfs, random_state = 0)
        dummy_clf.fit(X_train, y_train)
        score = dummy_clf.score(X_train, y_train)
        train_dummy_scores[clfs] = score

    values = list(train_dummy_scores.values())
    ax = sns.stripplot(strats, values);
    ax.set(xlabel ='strategy', ylabel ='training score')
    plt.savefig('../plots/strategies.jpg')
    plt.clf()


def build_dummy():
    dummy_clf = DummyClassifier(strategy = 'most_frequent', random_state = 0)
    dummy_clf.fit(X_train, y_train)
    dummy_clf.score(X_train, y_train)
    return dummy_clf


def simple_regression(training_cols):
    model = LogisticRegression(C=0.1, max_iter = 1000)
    model.fit(X_train[training_cols], y_train)
    score = model.score(X_train[training_cols], y_train)
    model.predict(X_train[training_cols])
    coef = model.coef_
    intercept = model.intercept_
    return coef, intercept, score, model

def accuracy(X_train, training_cols):
    y_pred_dummy = dummy_clf.predict(X_train) ## NOT SURE THE DUMMY IS NEEDED
    y_pred_lr = model.predict(X_train[training_cols])
    accuracy = accuracy_score(y_train, y_pred_lr)
    return accuracy, y_pred_lr


def confusion_matrix_(model, training_cols, X_train, y_train):
    titles_options = [("without_normalization", None),
                      ("Normalized", 'true')]
    class_names = training_cols
    for i in range(0, 2):
        titles = titles_options[i]
        if i == 0:
            disp = plot_confusion_matrix(model, X_train[columns], y_train,
                                         cmap=plt.cm.Blues,
                                         normalize=None)
        else:
            disp = plot_confusion_matrix(model, X_train[columns], y_train,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
        name = (titles_options[i][0], "confussion_matrix")
        name = "_".join(name)
        disp.ax_.set_title(name)
        print(name)
        print(disp.confusion_matrix)
        plt.savefig('../plots/'+name, format ="jpg")


def precision_recall(y_train, y_pred_lr):
    precision = precision_score(y_train, y_pred_lr)
    recall = recall_score(y_train, y_pred_lr)
    return precision, recall

def cross_validation(model, X_train, y_train, split, score_type):
    cross_scores = cross_val_score(model, X_train, y_train, cv=int(split), \
                                scoring=score_type)
    cross_mean = cross_scores.mean().round(3)
    cross_std = cross_scores.std()
    return cross_mean, cross_std



data = pd.read_csv("../data/train_featured.csv")
y = data['Survived']
X = data.iloc[:, 2:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
columns = [ 'Mrs', 'Miss', 'Age_ranges', 'Pclass', 'Sex', 'SibSp', 'qbin_Age2', 'Parch', 'Cabin_C']

if __name__ == "__main__":
    define_strategy(X_train, y_train)
    dummy_clf = build_dummy()
    coef, intercept, score, model = simple_regression(columns)
    print("Simple regression results:\nscore: " + str(score), \
      "\ncoef: " + str(coef), "\nintercept: " + str(intercept))
    confusion_matrix_(model, columns, X_train, y_train)
    accuracy, y_pred_lr = accuracy(X_train, columns)
    print("The accuracy of the simple regression model is: " , accuracy)
    precision, recall = precision_recall(y_train, y_pred_lr)
    print("The precision of the simple regression model is: ", precision, "\n", \
          "and the recall is: ", recall)
    crossval_mean, crossval_std = cross_validation(model, X_train, y_train, 5, \
                                  "accuracy")
    print("The mean for the cross valiadion score is: ", crossval_mean, "\n", \
              "and the SD is: ", crossval_std)
