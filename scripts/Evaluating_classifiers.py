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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
    y_pred_dummy = dummy_clf.predict(X_train)
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


def precision_recall_F1(y_train, y_pred_lr):
    precision = precision_score(y_train, y_pred_lr)
    recall = recall_score(y_train, y_pred_lr)
    F1 = 2 * (precision*recall)/(precision+recall)
    return precision, recall, F1

def cross_validation(model, X_train, y_train, split, score_type):
    cross_scores = cross_val_score(model, X_train, y_train, cv=int(split), \
                                scoring=score_type)
    cross_mean = cross_scores.mean().round(3)
    cross_std = cross_scores.std()
    return cross_mean, cross_std

def hyperparam_optimization(X_train, y_train):
    model_rf = RandomForestClassifier(n_estimators = 100, max_depth = 3, max_features = 3, min_samples_split = 2)
    param_grid = {
    'n_estimators': [1, 3, 10, 20, 50, 100],
    'max_depth':[1, 3, 5, 10, None]
    }
    gridcv = GridSearchCV(model_rf, param_grid = param_grid )
    gridcv.fit(X_train, y_train)
    columns = ['mean_test_score', 'std_test_score', 'mean_fit_time', \
            'param_max_depth', 'param_n_estimators']
    results_gridcv = pd.DataFrame(gridcv.cv_results_)
    results_gridcv[columns].sort_values('mean_test_score', ascending=False)
    random_forest = RandomForestClassifier(n_estimators=100, max_depth = 10, max_features =3)
    random_forest.fit(X_train, y_train)
    y_prediction = random_forest.predict(X_train)
    random_forest.score(X_train, y_train)
    acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
    return results_gridcv, acc_random_forest, random_forest


def select_features(rf_model, X_train, X_test, features):
    relevance_features = pd.DataFrame({"Feature": X_train.columns, \
                        "Relevance": rf_model.feature_importances_}\
                                      ).sort_values(by="Relevance", \
                                                    ascending=False)
    best_ = relevance_features["Feature"].values[:int(features)]
    X_train = X_train[best_]
    X_test = X_test[best_]
    return relevance_features, X_train, X_test, best_

def run_test_data(pred, best_, rf_model, pred_ids, file_name):
    pred = pred[best_]
    y_pred = rf_model.predict(pred)
    kaggle = pd.DataFrame({"PassengerId":pred_ids,  "Survived":y_pred})
    kaggle.to_csv('../data/' + file_name, index=False)


data = pd.read_csv("../data/train_featured.csv")
y = data['Survived']
X = data.iloc[:, 2:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
columns = [ 'Mrs', 'Miss', 'Age_ranges', 'Pclass', 'Sex', 'SibSp', 'qbin_Age2', 'Parch', 'Cabin_C']
pred_ids = pd.read_csv("../data/pred_ids.csv")
pred_ids = pred_ids["PassengerId"].to_list()
test_data =  pd.read_csv("../data/test_featured.csv")


if __name__ == "__main__":
    define_strategy(X_train, y_train)
    dummy_clf = build_dummy()
    coef, intercept, score, model = simple_regression(columns)
    print("Simple regression results:\nscore: " + str(score), \
      "\ncoef: " + str(coef), "\nintercept: " + str(intercept))
    confusion_matrix_(model, columns, X_train, y_train)
    accuracy, y_pred_lr = accuracy(X_train, columns)
    print("The accuracy of the simple regression model is: " , accuracy)
    precision, recall, F1 = precision_recall_F1(y_train, y_pred_lr)
    print("The precision of the simple regression model is: ", precision, "\n", \
          "the recall is: ", recall, " and the F1 score is : ", F1  )
    crossval_mean, crossval_std = cross_validation(model, X_train, y_train, 5, \
                                  "accuracy")
    print("The mean for the cross valiadion score is: ", crossval_mean, "\n", \
              "and the SD is: ", crossval_std)
    gridcv_res, random_forest_res, rf_model = hyperparam_optimization(X_train, \
                                              y_train)
    print("The gridcv results are : ", gridcv_res, "\n", \
              "and the random forest accuracy is: ", random_forest_res)
    relevance_features, X_train_2, X_test_2, best_ = select_features(rf_model,\
                                            X_train, X_test, 8)
    gridcv_res, random_forest_res, rf_model = hyperparam_optimization(X_train_2,\
                                            y_train)
    run_test_data(test_data, best_, rf_model, pred_ids, "kaggle_submission.csv")
