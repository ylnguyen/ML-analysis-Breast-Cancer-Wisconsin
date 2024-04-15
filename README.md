# Python: Exploring features + Machine Learning analysis in the Breast Cancer Wisconsin dataset

The open-source Diagnostic Wisconsin Breast cancer dataset contains features of the cell nuclei of malignant and benign tumor cells. The features are computed based on Whole Slide Images of a breast mass. The project's primary challenge is to determine whether a sample is malignant or benign, based on the characteristics in the dataset.

To approach this challenge, first data preprocessing are required. For the model, simple classifiers from ```scikit-learn``` were included, namely the Support Vector Machine, Decision Tree, Gaussian Naive Bayes, Logistic Regression and K-Nearest Neighbors classifiers. Additional steps were taken to improve the model robustness: nested cross validation with stratified splits, statistical-based feature selection and hyperparameter optimization using GridSearchCV.

The motivation behind this project was to demonstrate the steps of a machine learning analysis for medical data.