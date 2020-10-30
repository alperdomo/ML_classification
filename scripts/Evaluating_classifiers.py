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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def cross_validation_models(X_train, y_train, splits):
    random_state = 42
    cross_val_results = []
    cross_val_means = []
    cross_val_std = []
    models = []
    K_fold = StratifiedKFold(n_splits=splits)
    models.append(LogisticRegression(random_state = random_state))
    models.append(RandomForestClassifier(random_state=random_state))
    models.append(KNeighborsClassifier())
    models.append(SVC(random_state=random_state))
    models.append(GradientBoostingClassifier(random_state=random_state))
    models.append(LinearDiscriminantAnalysis())
    for i in models :
        cross_val_results.append(cross_val_score(i, X_train, y_train,
                                          scoring = "accuracy",
                                                 cv = K_fold, n_jobs=4))
    for results in cross_val_results:
        cross_val_means.append(results.mean())
        cross_val_std.append(results.std())
    cross_val_df = pd.DataFrame(
        {"Cross_Val_Means":cross_val_means,
         "Cross_Val_Errors": cross_val_std,
         "Algorithms":[
             "Logistic_Regression", "Random_Forest",
             "K_Neighbors", "SVC", "Gradient_Boosting",
             "LinearDiscriminantAnalysis"
            ]
        })
    plt.figure(figsize=(9, 11))
    cross_val_plot = sns.barplot("Cross_Val_Means", "Algorithms", \
                            data = cross_val_df,
                    palette="colorblind", orient = "h", \
                                **{'xerr':cross_val_std})
    cross_val_plot.set_xlabel("Accuracy (mean)")
    cross_val_plot = cross_val_plot.set_title("Cross validation scores")


    plt.savefig('../plots/Cross_val_algorithms.jpg', format ="jpg", \
                bbox_inches = 'tight')
    plt.clf()
    return cross_val_df

def hyper_opt_logistic(X_train, y_train, splits):
    K_fold = StratifiedKFold(n_splits = splits)
    LR_model = LogisticRegression()
    lR_parameters = {
        "penalty" : ["l2"],
        "C" :[0.01, 0.1, 1, 10, 100],
        "intercept_scaling": [1, 2, 3, 4],
        "tol" : [0.0001,0.0002,0.0003],
        "max_iter": [100,200,300],
        "solver":['liblinear'],
        "verbose":[1]
    }
    grid_LR_model = GridSearchCV(LR_model, param_grid = lR_parameters, cv=K_fold,
                         scoring="accuracy", n_jobs= 5, verbose = 1)
    grid_LR_model.fit(X_train, y_train)
    LR_model_best = grid_LR_model.best_estimator_
    best_score_LR = grid_LR_model.best_score_
    return best_score_LR, LR_model_best


def hyper_opt_RanForest(X_train, y_train, splits):
    K_fold = StratifiedKFold(n_splits = splits)
    RF_model = RandomForestClassifier()
    rf_parameters = {
        "max_depth": [3, 6, 9, 12, None],
        "min_samples_split": [2, 6, 20],
        "min_samples_leaf": [1, 4, 16],
        "n_estimators" :[100,200,300,400],
        "criterion": ["gini"]
        }
    grid_RF = GridSearchCV(RF_model, param_grid = rf_parameters, cv=K_fold,
                         scoring="accuracy", n_jobs= 5, verbose = 1)

    grid_RF.fit(X_train, y_train)
    RF_model_best = grid_RF.best_estimator_
    best_score_RF = grid_RF.best_score_
    return best_score_RF, RF_model_best


def hyper_opt_KNN(X_train, y_train, splits):
    K_fold = StratifiedKFold(n_splits = splits)
    KNN_model = KNeighborsClassifier()#Use GridSearch

    KNN_parameters = {
        "leaf_size" : list(range(1,50)),
        "n_neighbors" : list(range(1,30)),
        "p" : [1,2]
    }

    grid_KNN_model = GridSearchCV(KNN_model,
                                  param_grid = KNN_parameters,
                                  cv=K_fold,
                                  scoring="accuracy",
                                  n_jobs= 5,
                                  verbose = 1
                                 )
    grid_KNN_model.fit(X_train, y_train)
    KNN_model_best = grid_KNN_model.best_estimator_
    best_score_KNN = grid_KNN_model.best_score_
    return best_score_KNN, KNN_model_best


def hyper_opt_SVMC(X_train, y_train, splits):
    K_fold = StratifiedKFold(n_splits = splits)
    SVMC_model = SVC(probability=True)
    svmc_parameters = {
        'C': [1, 10, 50, 100, 200, 300],
        'kernel': ['rbf'],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
    }

    grid_SVMC_model = GridSearchCV(SVMC_model, param_grid = svmc_parameters, cv = K_fold,
                          scoring="accuracy", n_jobs= -1, verbose = 1)

    grid_SVMC_model.fit(X_train,y_train)
    SVMC_model_best = grid_SVMC_model.best_estimator_
    best_score_SVMC = grid_SVMC_model.best_score_
    return best_score_SVMC, SVMC_model_best


def hyper_opt_GBC(X_train, y_train, splits):
    K_fold = StratifiedKFold(n_splits = splits)
    GB_model = GradientBoostingClassifier()
    GB_parameters = {
                  'loss' : ["deviance"],
                  'n_estimators' : [100,200,300],
                  'learning_rate': [0.1, 0.05, 0.01, 0.001],
                  'max_depth': [4, 8,16],
                  'min_samples_leaf': [100,150,250],
                  'max_features': [0.3, 0.1]
                  }
    gridGB_model = GridSearchCV(GB_model, param_grid = GB_parameters, cv=K_fold,
                         scoring="accuracy", n_jobs= 5, verbose = 1)

    gridGB_model.fit(X_train,y_train)
    GB_model_best = gridGB_model.best_estimator_
    best_score_GBC = gridGB_model.best_score_
    return best_score_GBC, GB_model_best


def hyper_opt_LDA(X_train, y_train, splits):
    K_fold = StratifiedKFold(n_splits = splits)
    LDA_model= LinearDiscriminantAnalysis()
    lda_parameters= {"solver" : ["svd"],
                  "tol" : [0.0001,0.0002,0.0003]}

    grid_LDA_model = GridSearchCV(LDA_model, param_grid = lda_parameters, cv=K_fold,
                         scoring="accuracy", n_jobs= 5, verbose = 1)

    grid_LDA_model.fit(X_train,y_train)
    LDA_model_best = grid_LDA_model.best_estimator_
    best_score_LDA = grid_LDA_model.best_score_
    return best_score_LDA, LDA_model_best


def general_hyper(X_train, y_train, splits):
    best_LR, LR_mbest = hyper_opt_logistic(X_train, y_train, splits)
    best_RF, RF_mbest = hyper_opt_RanForest(X_train, y_train, splits)
    best_KNN, KNN_mbest = hyper_opt_KNN(X_train, y_train, splits)
    best_SVMC, SVMC_mbest = hyper_opt_SVMC(X_train, y_train, splits)
    best_GBC, GBC_mbest = hyper_opt_GBC(X_train, y_train, splits)
    best_score_LDA, LDA_mbest = hyper_opt_LDA(X_train, y_train, splits)
    return best_LR, LR_mbest, best_RF, RF_mbest, best_KNN, KNN_mbest, \
           best_SVMC, SVMC_mbest, best_score_GBC, GBC_mbest, \
           best_score_LDA, LDA_mbest

def hyperparam_optimization(X_train, y_train, splits):
    K_fold = StratifiedKFold(n_splits = splits)
    model_rf = RandomForestClassifier(n_estimators = 100, max_depth = 3, \
                                      max_features = 3, min_samples_split = splits)
    rf_parameters = {
        "max_depth": [3, 6, 9, 12, None],
        "min_samples_split": [2, 6, 20],
        "min_samples_leaf": [1, 4, 16],
        "n_estimators" :[100,200,300,400],
        "criterion": ["gini"]
        }

    grid_RF = GridSearchCV(RF_model, param_grid = rf_parameters, cv=K_fold,
                         scoring="accuracy", n_jobs= 3, verbose = 1)
    grid_RF.fit(X_train, y_train)
    columns = ['mean_test_score', 'std_test_score', 'mean_fit_time', \
            'param_max_depth', 'param_n_estimators']
    results_gridcv = pd.DataFrame(grid_RF.cv_results_)
    results_gridcv[columns].sort_values('mean_test_score', ascending=False)
    random_forest = RandomForestClassifier(n_estimators=100, max_depth = 10,\
                                           max_features =3)
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
      "\ncoef: " + str(coef), "\nintercept: " + str(intercept), "\n")
    confusion_matrix_(model, columns, X_train, y_train)
    accuracy, y_pred_lr = accuracy(X_train, columns)
    print("\nThe accuracy of the simple regression model is: " , accuracy, "\n")
    precision, recall, F1 = precision_recall_F1(y_train, y_pred_lr)
    print("The precision of the simple regression model is: ", precision, "\n", \
          "the recall is: ", recall, " and the F1 score is : ", F1 , "\n")
    crossval_mean, crossval_std = cross_validation(model, X_train, y_train, 5, \
                                  "accuracy", "\n")
    print("The mean for the cross validation score is: ", crossval_mean, "\n", \
              "and the SD is: ", crossval_std, "\n")
    cross_val_df = cross_validation_models(X_train, y_train, 3)
    print("This are the cross valifation results for multiple models", \
         cross_val_df.head(6), "\n")
    print("\n\nRunning general function for hyperparameters optimization\n\n")
    LR_best, LR_model, RF_best, RF_model, KNN_best, KNN_model, SVMC_best, \
    SVMC_model, GBC_best, GBC_model, LDA_best, LDA_model = \
                        general_hyper(X_train, y_train, 3)
    print("\n\n Running all 6 MODELS on unseen Test data\n\n")
    LR_model.fit(X_train, y_train)
    print('Logistic Regresion training score: ', \
            LR_model.score(X_train, y_train).round(3))
    print('Logistic Regresion test score: ', \
            LR_model.score(X_test, y_test).round(3))
    RF_model.fit(X_train, y_train)
    print('Random Forest training score: ', \
            RF_model.score(X_train, y_train).round(3))
    print('Random Forest test score: ', \
            RF_model.score(X_test, y_test).round(3))
    KNN_model.fit(X_train, y_train)
    print('K neigbors training score: ', \
            KNN_model.score(X_train, y_train).round(3))
    print('K neigbors test score: ', \
            KNN_model.score(X_test, y_test).round(3))
    SVMC_model.fit(X_train, y_train)
    print('Support Vector Machine training score: ', \
            SVMC_model.score(X_train, y_train).round(3))
    print('Support Vector Machine test score: ', \
            SVMC_model.score(X_test, y_test).round(3))

    GBC_model.fit(X_train, y_train)
    print('Gradient Boosting training score: ', \
            GBC_model.score(X_train, y_train).round(3))
    print('Gradient Boosting test score : ', \
            GBC_model.score(X_test, y_test).round(3))
    LDA_model.fit(X_train, y_train)
    print('Linear Discriminant Analysis  training score: ', \
            LDA_model.score(X_train, y_train).round(3))
    print('Linear Discriminant Analysis  test score: ', \
            LDA_model.score(X_test, y_test).round(3))

    gridcv_res, random_forest_res, rf_model = hyperparam_optimization(X_train, \
                                              y_train)
    print("The gridcv results are : ", gridcv_res, "\n", \
              "and the random forest accuracy is: ", random_forest_res, "\n")
    relevance_features, X_train_2, X_test_2, best_ = select_features(rf_model,\
                                            X_train, X_test, 8)
    gridcv_res, random_forest_res, rf_model = hyperparam_optimization(X_train_2,\
                                            y_train)
    run_test_data(test_data, best_, rf_model, pred_ids, "kaggle_submission.csv")
