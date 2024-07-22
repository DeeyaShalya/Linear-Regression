import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk

# Load the dataset
file_path = r"E:\PROJECT - AI & ML\Linear and Logistic Regression\Impact_of_Mobile_Phone_on_Students_Health.csv"
df = pd.read_csv(file_path)

# Strip extra spaces from column names
df.columns = df.columns.str.strip()

# Handle missing values (if any)
df = df.dropna()

# Encode categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Features and target variable
X = df.drop(columns='Health rating')
y = df['Health rating']

# Define features and target
features = ['Age', 'Mobile Phone', 'Mobile phone use for education', 'Daily usages', 'Usage distraction', 'Attention span', 'Health Risks', 'Symptom frequency', 'Health precautions', 'Health rating']
target = 'Performance impact'

# Selecting relevant features for the regression model
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Evaluate the linear regression model's performance
y_pred_lin = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lin)
print(f'Linear Regression Mean Squared Error: {mse}')

# Train a logistic regression model if target is binary
log_reg = None
if len(y.unique()) == 2:
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_log)
    print(f'Logistic Regression Accuracy: {accuracy}')

# Function to predict performance impact
def predict_performance_impact():
    user_input = {}
    for feature in features:
        value = feature_vars[feature].get()
        if feature in label_encoders:
            if value in label_encoders[feature].classes_:
                value = label_encoders[feature].transform([value])[0]
            else:
                result_label.config(text=f"Error: '{value}' is not a valid label for '{feature}'. Please enter a valid value.")
                return
        else:
            value = float(value)
        user_input[feature] = value
    
    user_input_df = pd.DataFrame([user_input])
    lin_prediction = lin_reg.predict(user_input_df)[0]
    result = f'Linear Regression Prediction: {lin_prediction}'
    if log_reg is not None:
        log_prediction = log_reg.predict(user_input_df)[0]
        result += f'\nLogistic Regression Prediction: {log_prediction}'
    
    result_label.config(text=result)

# Create UI with Tkinter
root = tk.Tk()
root.title("Performance Impact Prediction")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

feature_vars = {}
for i, feature in enumerate(features):
    ttk.Label(frame, text=feature).grid(row=i, column=0, sticky=tk.W)
    if feature in label_encoders:
        feature_vars[feature] = tk.StringVar()
        values = list(label_encoders[feature].classes_)
        ttk.Combobox(frame, textvariable=feature_vars[feature], values=values).grid(row=i, column=1, sticky=(tk.W, tk.E))
    else:
        feature_vars[feature] = tk.DoubleVar()
        ttk.Entry(frame, textvariable=feature_vars[feature]).grid(row=i, column=1, sticky=(tk.W, tk.E))

ttk.Button(frame, text="Predict", command=predict_performance_impact).grid(row=len(features), column=0, columnspan=2, sticky=(tk.W, tk.E))

result_label = ttk.Label(frame, text="")
result_label.grid(row=len(features)+1, column=0, columnspan=2, sticky=(tk.W, tk.E))

# Display dataset summary
print("Dataset Summary:")
print(df.describe(include='all'))

# Display the head and tail of the dataset
print("\nDataset Head:")
print(df.head())

print("\nDataset Tail:")
print(df.tail())

# Show training data information
print("\nTraining Data:")
print(X_train.head())

# Graphical representation of the dataset
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

root.mainloop()