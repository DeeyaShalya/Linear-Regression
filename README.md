# Linear-Regression
Welcome to the Performance Impact Prediction project! This repository contains a machine learning solution that predicts the performance impact on students based on various factors related to mobile phone usage and health. The project leverages both linear and logistic regression models to provide accurate predictions. It includes comprehensive data preprocessing steps, such as handling missing values and encoding categorical variables, ensuring the models are trained on clean and well-structured data. Additionally, a user-friendly graphical user interface (GUI) is implemented using Tkinter, allowing users to easily input data and obtain predictions without needing extensive technical knowledge.

The dataset used in this project encompasses a wide range of features, including age, gender, daily mobile phone usage, health risks, symptoms, and more. Through detailed data visualizations, such as bar charts and pie charts, the project provides insights into the distribution and relationships of these features. This repository is structured to facilitate ease of use, with clear instructions for cloning the repository, installing dependencies, and running the scripts. By integrating data analysis, machine learning, and an intuitive GUI, this project serves as a robust example of applying AI and data analytics to real-world problems, making it an excellent showcase for job placement and furthering one's career in the field of data science and AI.

Machine Learning Algorithms
Linear Regression
Linear regression is a fundamental machine learning algorithm used to model the relationship between a dependent variable and one or more independent variables. The goal is to fit a linear equation to the observed data. This algorithm is ideal for predicting continuous outcomes, such as the performance impact on students in this project. In our code, we use the LinearRegression class from the sklearn.linear_model module to train the model on our dataset. The mean squared error (MSE) metric is used to evaluate the model's performance, indicating how well the model's predictions match the actual data.

Why Each Coding Step is Used
Loading the Dataset: The dataset is loaded using pandas to facilitate easy manipulation and analysis of the data.
Data Preprocessing: We strip extra spaces from column names and handle missing values by dropping rows with any missing data, ensuring the dataset is clean.
Encoding Categorical Variables: LabelEncoder is used to convert categorical variables into numerical values, making them suitable for machine learning algorithms.
Feature and Target Selection: Features relevant to predicting the performance impact are selected, and the target variable is defined.
Train-Test Split: The data is split into training and testing sets using train_test_split to evaluate the model's performance on unseen data.
Model Training: The linear regression model is trained on the training data using the fit method.
Model Evaluation: The model's predictions on the test data are compared to the actual values using the MSE metric.
GUI Implementation: Tkinter is used to create a user-friendly interface for inputting data and displaying predictions, making the project accessible to non-technical users.
By following these steps, the project ensures a robust and accurate machine learning solution for predicting the performance impact on students based on mobile phone usage and health factors.
