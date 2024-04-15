# Import some "simple" classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def initiate_models():
    """
    Create instances from Support Vector, Decision Tree, Gaussian Naive Bayes, Logistic Regression,
    Random Forest and K-Nearest Neighbors classifiers from the Scikit-Learn library.
    """
    svm = SVC(kernel='rbf')
    tree = DecisionTreeClassifier(random_state = 0)
    NB = GaussianNB()
    logreg = LogisticRegression(random_state = 0)
    rf = RandomForestClassifier(criterion = 'entropy', random_state = 0)
    knn = KNeighborsClassifier(n_neighbors=7)
    
    return [svm, tree, NB, logreg, rf, knn], ['svm', 'tree', 'NB', \
        'logistic', 'rf', 'knn']

def param_grid():
    """
    Initializes a simple parameter grid for the Scikit-Learn classifiers for GridSearchCV.
    """
    svm_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}
    tree_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
    NB_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
    log_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
    rf_grid = {'max_depth': [10, 20, 30], 'n_estimators': [5,10, 15]}
    knn_grid = {'n_neighbors': np.arange(3, 10)}
    return [svm_grid, tree_grid, NB_grid, log_grid, rf_grid, knn_grid]