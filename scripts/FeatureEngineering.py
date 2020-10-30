import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cabin_names(dataframe, data_type):
    """
    Hoteconodes the cabin names

    Parameters:
    dataframe: Data from the Titanic dataset. Either training or test data
    data_type: training, test
    ----------
    """

    dataframe['Embarked'] = dataframe['Embarked'].replace(to_replace = 'C',
                            value = int(0), regex =True)
    dataframe['Embarked'] = dataframe['Embarked'].replace(to_replace = 'S',
                            value = int(1), regex =True)
    dataframe['Embarked'] = dataframe['Embarked'].replace(to_replace = 'Q',
                            value = int(2), regex =True)
    dataframe['Cabin'] = dataframe['Cabin'].fillna(0).astype(str).str[0]
    onehot = pd.get_dummies(dataframe['Cabin'])
    if data_type == "training":
        onehot.columns = ['unk_Cabin', 'Cabin_A','Cabin_B','Cabin_C', \
                          'Cabin_D', 'Cabin_E', 'Cabin_F','Cabin_G', 'Cabin_T']
        dataframe = pd.concat([dataframe, onehot], axis = 1)
    else:
        onehot.columns = ['unk_Cabin', 'Cabin_A','Cabin_B','Cabin_C', \
                          'Cabin_D', 'Cabin_E', 'Cabin_F','Cabin_G']
        dataframe = pd.concat([dataframe, onehot], axis = 1)
    return dataframe


def name_to_titles(dataframe):
    """
    Hotecondes names of passengers to only their title

    Parameters:
    ----------
    dataframe: training dataset Titanic
    """
    dataframe['Name'] = dataframe['Name'].replace(to_replace = ['\.*.*Capt.*',
                        '\.*.*Countess.*',  '\.*.*Lady.*', '\.*.*Col.*',
                        '\.*.*Dr.*', '\.*.*Don.*', '\.*.*Major.*',
                        '\.*.*Sir.*', '\.*.*Dona.*', '\.*.*Jonkheer.*',
                        '\.*.*Rev.*'], value = 0, regex = True)
    dataframe['Name'] = dataframe['Name'].replace(to_replace = ['\.*.*Mlle.*',
                        '\.*.*Mlle.*'], value = 'Miss', regex=True)
    dataframe['Name'] = dataframe['Name'].replace(to_replace = ['\.*.*Miss.*',
                        '\.*.*Miss.*'], value = 'Miss', regex=True)
    dataframe['Name'] = dataframe['Name'].replace(to_replace = ['\.*.*Ms.*',
                        '\.*.*Miss.*'], value = 'Miss', regex=True)
    dataframe['Name'] = dataframe['Name'].replace(to_replace = '\.*.*Mme.*',
                        value = 'Mistress', regex=True)
    dataframe['Name'] = dataframe['Name'].replace(to_replace = '\.*.*Mrs.*',
                        value = 'Mistress', regex=True)
    dataframe['Name'] = dataframe['Name'].replace(to_replace = '\.*.*Master.*',
                        value = 'Master', regex=True)
    dataframe['Name'] = dataframe['Name'].replace(to_replace = '\.*.*Mr. .*',
                        value = 'Mr', regex=True)
    dataframe['Name'] = dataframe['Name'].fillna('uncommon')
    onehot = pd.get_dummies(dataframe['Name'])
    onehot.columns=['Master', 'Miss','Mrs','Mr','uncommon']
    dataframe = pd.concat([dataframe, onehot], axis = 1)
    return dataframe


def fares(dataframe):
    """
    Hotecondes the fares into five different groups

    Parameters:
    ----------
    dataframe: training dataset Titanic
    """
    bins_fares = [-1, 7.92, 14.5, 32, 100, np.inf]
    names_fares = [0, 1, 2, 3, 4]
    dataframe['Fare_ranges'] = pd.cut(dataframe['Fare'], bins_fares,
                               labels = names_fares)
    return dataframe, bins_fares


def gender(dataframe):
    """
    Hotencodes gender to either male or female

    Parameters:
    ----------
    dataframe: training dataset Titanic
    """
    dataframe.loc[(dataframe['Sex'] == 'male'), 'Sex'] = 0
    dataframe.loc[(dataframe['Sex'] == 'female'), 'Sex'] = 1
    return dataframe


def ages(dataframe):
    """
    Hotencodes ages into four different groups using pandas qbins function

    Parameters:
    ----------
    dataframe: training dataset Titanic
    """
    dataframe['Age'] = dataframe['Age'].fillna(dataframe['Age'].mean())
    qbins = pd.qcut(dataframe['Age'], q = 4)
    qbins = pd.get_dummies(qbins)
    qbins.columns = ['qbin_Age1', 'qbin_Age2', 'qbin_Age3', 'qbin_Age4']
    dataframe = pd.concat([dataframe, qbins], axis =1)
    return dataframe


def age_per_class(dataframe, bins_fares):
    """
    Hotecondes ages per travelling class into five different groups

    Parameters:
    ----------
    dataframe: training dataset Titanic
    """
    dataframe['int_Age*Pclass'] = dataframe['Age'] * dataframe['Pclass']
    qbins = pd.qcut(dataframe['int_Age*Pclass'], q=4)
    qbins = pd.get_dummies(qbins)
    qbins.columns=['qbin_SexClass1', 'qbin_SexClass2', 'qbin_SexClass3', 'qbin_SexClass4']
    dataframe = pd.concat([dataframe, qbins], axis = 1)
    bins_Ages = [-1, 22, 29, 35, 60, np.inf]
    names_fares = [1, 2, 3, 4, 5]
    dataframe['Age_ranges'] =  pd.cut(dataframe['Age'], bins_fares, labels=names_fares)
    return dataframe


def plot_distribution_titles(dataframe):
    """
    Plot the distribution of passengers based on their title

    Parameters:
    ----------
    dataframe: training dataset Titanic with the names modified to titles
    """
    data = dataframe.groupby(['Survived', 'Name'])[['Name']].count().unstack()
    data.plot.bar(figsize = (12, 8))
    plt.savefig('../plots/distribution_titles', format="svg")


def clear_engineer(dataframe, columns_drop, data_type):
    """
    clear dataframe for a list containing the name of columns that will not be
    used as classifiers for the evaluation and prediction

    Parameters:
    ----------
    dataframe: Data from the Titanic dataset. Either training or test data
    data_type: training, test
    """
    if data_type == "training":
        dataframe.drop(columns_drop, axis = 1, inplace = True)
        dataframe.to_csv('../data/train_featured.csv', index=False)
    else:
        columns_drop = [value for value in columns_drop if value != "Cabin_T"]
        columns_drop += ["PassengerId"]
        dataframe.drop(columns_drop, axis = 1, inplace = True)
        dataframe.to_csv('../data/test_featured.csv', index=False)
    return dataframe

def engineer_test_kaggle(test_data):
    pred = test_data
    pred = cabin_names(pred, "test")
    pred =  name_to_titles(pred)
    pred, pred_bins_fares = fares(pred)
    pred =  gender(pred)
    pred =  ages(pred)
    pred =  age_per_class(pred, pred_bins_fares)
    pred =  clear_engineer(pred, columns_drop, "test")

data = pd.read_csv('../data/train.csv')
columns_drop = ['Age', 'Name', 'Fare',  'Ticket', 'Cabin', 'Embarked', 'int_Age*Pclass', "Cabin_T"]
test_data  = pd.read_csv('../data/test.csv')

if __name__ == "__main__":
    data = cabin_names(data, "training")
    data = name_to_titles(data)
    data, bins_fares = fares(data)
    data = gender(data)
    data = ages(data)
    data = age_per_class(data, bins_fares)
    plot_distribution_titles(data)
    clear_engineer(data, columns_drop, "training")
    engineer_test_kaggle(test_data)
