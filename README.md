# ML_classification_titanic
There are many different machine learning (ML) systems. Based on their purposes and features, they are classified in different types: Superivised-, unsupervised-, semisupervised-, and reinforcement- learning (Geron, 2019).

In the case of Supervised ML, a typical task is classification. In such ML task an algorithm is trained, "fed", with the data and the desired solutions.

This image represents a  basic pipeline for a supervised ML Automation and evaluation

![](images/Data_Types_titanic.jpg)

In the ML_classification_titanic we are using a very popular dataset from Kaggle. Here, we used the training set from the `Titanic: Machine learning from Disaster` from Kaggle to fine six of the most important supervised ML algorythms:
- Logistic Regression
- RandomForestClassifier: A meta estimator  to fit a n number of decision tree classifiers on various sub-samples of the dataset
- K neighbors
- Support Vector Machines
- Gradient Boosting
- Linear Discriminant Analysis

`The pipeline includes the following steps`

- Data exploratory analysis

- Feature engineering

- Evaluating classifiers:
  * Strategy definition
  * Accuracy
  * Precision
  * Recall
  * F1 scores

- Hyperparameters optimization using Exhaustive search over specified parameters for estimators (GridSearchCV):
  * Logistic Regression
  * RandomForestClassifier: A meta estimator  to fit a n number of decision tree classifiers on various sub-samples of the dataset
  * Support Vector Machines
  * K neighbors
  * Gradient Boosting
  * Linear Discriminant Analysis

- How different ML models perform on unseen data: Predicting on the test data.  


References:
Aurélien Géron. 2019. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition.O'Reilly Media
