Diabetes Classification using Machine Learning


Project Overview
This project focuses on classifying diabetes status using various machine learning algorithms. The dataset contains medical information from patients, with the goal of predicting whether a patient is diabetic or not. The project involves data preprocessing, augmentation, and the application of multiple classification models to find the most accurate one for this binary classification problem.

Dataset
The dataset used in this project is the Diabetes Dataset. It includes patient medical records with various features and a binary label indicating diabetes status.
Input Data: 8 continuous attributes including glucose level, blood pressure, BMI, and others.
Output: A binary classification label ('Diabetic' or 'Non-Diabetic').

Project Structure
data/: Contains the dataset (diabetes.csv).
notebooks/: Jupyter notebooks for data exploration and visualization (if applicable).
src/: Main Python scripts with implementations for data processing, augmentation, and model evaluation.
README.md: Project documentation.
requirements.txt: List of dependencies.

Key Steps
Data Loading & Inspection:
Loaded the dataset and performed initial exploration to understand its structure and check for missing values and duplicates.
Data Preprocessing:
Class Distribution Analysis: Evaluated the distribution of classes in the dataset.
Data Cleaning: Removed outliers based on Z-scores.
Data Augmentation: Added Gaussian noise to augment the dataset.
Feature Scaling: Applied StandardScaler to normalize the features.
Model Selection:

Applied several classification models:
Logistic Regression
Support Vector Machine (SVC)
K-Nearest Neighbors (KNN)
Random Forest
Gaussian Naive Bayes
Bernoulli Naive Bayes
Decision Tree
Gradient Boosting
AdaBoost
Extra Trees
Model Evaluation:

Evaluated models using accuracy, precision, recall, and F1-score on both training and test datasets.
Used StratifiedKFold and GridSearchCV for hyperparameter tuning and model comparison.
Prediction:

Developed a system to make predictions using the best-performing model.


Results
This project implements several machine learning models to predict whether an individual has diabetes based on health-related data. The models are evaluated based on various metrics such as accuracy, precision, recall, and F1-Score. The dataset used is the "Pima Indians Diabetes Database."

Models Evaluated
The following machine learning models were evaluated:

Logistic Regression
Support Vector Classifier (SVC)
K-Nearest Neighbors (KNN)
Random Forest Classifier
Gaussian Naive Bayes
Bernoulli Naive Bayes
Decision Tree Classifier
Gradient Boosting Classifier
AdaBoost Classifier
Extra Trees Classifier

Best Model
The best-performing model based on test accuracy is the Extra Trees Classifier, with a test accuracy of 97.1%.

Results Summary
Below are the best hyperparameters for each model, cross-validation accuracy, test accuracy, precision, recall, and F1-score:

Model	Best Parameters	CV Accuracy	Training Accuracy	Test Accuracy	Precision	Recall	F1-Score
Logistic Regression	{'C': 0.1}	78.73%	79.09%	78.62%	71.62%	58.24%	64.24%
Support Vector Classifier	{'C': 10, 'kernel': 'rbf'}	82.00%	91.82%	83.33%	74.73%	74.73%	74.73%
K-Nearest Neighbors	{'n_neighbors': 5}	78.18%	85.73%	81.16%	74.07%	65.93%	69.77%
Random Forest Classifier	{'max_depth': None, 'n_estimators': 50}	88.82%	100.0%	93.12%	90.91%	87.91%	89.39%
Gaussian Naive Bayes	Default	75.91%	76.18%	75.00%	61.70%	63.74%	62.70%
Bernoulli Naive Bayes	Default	72.45%	73.00%	73.91%	60.22%	61.54%	60.87%
Decision Tree Classifier	{'max_depth': 20}	85.00%	100.0%	89.49%	86.90%	80.22%	83.43%
Gradient Boosting Classifier	{'learning_rate': 0.1, 'n_estimators': 100}	81.82%	91.00%	81.88%	77.33%	63.74%	69.88%
AdaBoost Classifier	{'n_estimators': 50}	78.73%	84.73%	78.26%	70.67%	58.24%	63.86%
Extra Trees Classifier	{'max_depth': None, 'n_estimators': 50}	92.09%	100.0%	97.10%	95.60%	95.60%	95.60%

Conclusion
The Extra Trees Classifier outperformed all other models with the highest test accuracy of 97.1%, making it the best model for predicting diabetes in this dataset. Other models such as Random Forest and Support Vector Classifier also performed well, with test accuracies above 80%.

For future work, additional techniques such as model ensembling, more advanced hyperparameter tuning, or using different data augmentation techniques could further improve performance.

Data augmentation improved performance across several models.
Scaling had a significant positive impact on specific algorithms.

How to Use
Train the Models:
Use the compare_models() function in src/ to train and evaluate models. This function performs hyperparameter tuning and compares models based on their accuracy.
Make Predictions:
Load the saved scaler and model using joblib.
Prepare input data in the same format as the training data.
Use the best model to make predictions.

Project Highlights
Comprehensive Data Analysis: Includes exploration, outlier detection, and class distribution analysis.
Data Augmentation: Implemented Gaussian noise addition to enhance model robustness.
Feature Scaling: Applied normalization to improve model performance.
Model Evaluation: Compared multiple models using cross-validation and various metrics.
Predictive System: Developed for real-time predictions using the trained models.

Future Improvements
Hyperparameter Tuning: Further fine-tuning with advanced techniques such as GridSearchCV or RandomizedSearchCV.
Cross-Validation: Continue using cross-validation to ensure model robustness.
Feature Engineering: Explore additional features or transformations to improve model accuracy.
Advanced Augmentation: Experiment with more sophisticated data augmentation methods.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

Acknowledgments
Dataset: Diabetes dataset from Kaggle.
Libraries: Utilizes open-source libraries such as scikit-learn, NumPy, Pandas, Matplotlib, and Seaborn.
Feel free to customize or adjust any section to better fit your project's details or structure.

