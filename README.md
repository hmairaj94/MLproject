# Student Performance Prediction Using Machine Learning

## Project Overview

This project aims to predict students’ academic performance by analyzing how various factors influence test scores. Understanding these relationships can help educators and institutions implement proactive measures to support students who may be at risk of underperforming. By leveraging machine learning, this project provides insights into the impact of demographic and socio-economic factors on student outcomes, allowing for data-driven interventions.

## Problem Statement

Academic performance is influenced by multiple factors, including personal demographics, family background, and external support systems. The goal of this project is to predict student test scores based on variables such as gender, ethnicity, parental level of education, lunch type, and test preparation course completion. Insights from this analysis can help educators identify key factors that influence performance and assist in designing strategies to improve student outcomes.

## Dataset

### Dataset Source
- **Kaggle Dataset**: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)

### Dataset Description
The dataset consists of information on students' demographic backgrounds and their scores in three core subjects: math, reading, and writing. The dataset contains the following columns:
- **gender**: Gender of the student (Male/Female)
- **race/ethnicity**: Group designation for race/ethnicity (e.g., Group A, Group B)
- **parental level of education**: Highest level of education achieved by the student’s parents
- **lunch**: Type of lunch provided (standard or free/reduced)
- **test preparation course**: Completion status of the test preparation course (completed or not completed)
- **math score**: Student’s score in mathematics
- **reading score**: Student’s score in reading
- **writing score**: Student’s score in writing

## Project Workflow

### 1. Data Preprocessing
- **Data Cleaning**: Checked for any missing values, anomalies, or duplicates in the dataset and handled them accordingly.
- **Feature Encoding**: Converted categorical variables (such as gender, race/ethnicity, lunch, etc.) into numerical representations using techniques like one-hot encoding to make the data suitable for machine learning models.
- **Standardization**: Standardized numerical features to ensure they are on a similar scale, improving model performance and convergence.

### 2. Exploratory Data Analysis (EDA)
Performed in-depth EDA to understand relationships and patterns within the dataset:
- **Correlation Analysis**: Examined correlations between independent variables (e.g., parental education, lunch type) and dependent variables (test scores).
- **Visualizations**: Used histograms, box plots, and scatter plots to visualize distributions, outliers, and relationships between variables.
- **Insights**: Noted key insights, such as the impact of test preparation on scores and variations in scores based on parental education and lunch type.

### 3. Model Building
Various machine learning models were applied to predict student scores and identify significant factors:
- **Regression Models**: Built multiple regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor) to predict continuous test scores.
- **Classification Models (Optional)**: Converted test scores into categories (e.g., "low," "average," "high") and used classification models like Logistic Regression and Support Vector Machine (SVM) for classification tasks.
- **Hyperparameter Tuning**: Optimized models by tuning hyperparameters using techniques like grid search and cross-validation.

### 4. Model Evaluation
Evaluated model performance using several metrics:
- **Regression Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) for regression models.
- **Classification Metrics (if applicable)**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix for classification models.
- **Model Comparison**: Compared the performance of different models to select the best one based on accuracy and interpretability.

### 5. Feature Importance Analysis
- **Feature Importance**: Analyzed feature importance to identify the factors that most significantly impact student performance.
- **SHAP Values (if applicable)**: Used SHAP values to interpret model predictions, explaining how individual features contribute to predictions.

## Results and Insights
- **Top Predictors**: Identified key predictors of student performance, such as parental level of education, test preparation course completion, and lunch type.
- **Model Performance**: The best-performing model achieved an accuracy of [X]% with a Mean Absolute Error (MAE) of [Y] on the test set (replace with actual results).
- **Insights for Educators**: Suggested actionable insights, such as focusing resources on students from specific backgrounds or supporting students who haven’t completed test preparation courses, as these factors were associated with lower scores.

## Conclusion
This project demonstrates how machine learning can be used to predict student performance based on socio-economic and demographic factors. By identifying at-risk students, educational institutions can proactively design interventions to improve academic outcomes. The analysis also sheds light on the factors that most influence student success, providing valuable insights for targeted educational strategies.

## Future Work
- **Additional Features**: Collect additional data on factors like attendance, socio-economic status, and mental health to further enhance predictions.
- **Advanced Models**: Experiment with advanced models such as Gradient Boosting or Neural Networks for potentially better accuracy.
- **Explainability**: Incorporate more interpretable models or explainability tools like LIME to ensure model transparency for educational stakeholders.

## Technologies and Tools Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, SHAP (optional)
- **Environment**: Jupyter Notebook or any Python IDE

## Screenshot
![Screenshot (68)](https://github.com/user-attachments/assets/b9a90ac0-24b0-42b2-830d-4e2165f282bc)
