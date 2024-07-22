# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import ipywidgets as widgets
from IPython.display import display, clear_output
from tabulate import tabulate

import pandas as pd
file_path = r'E:\PROJECT - AI & ML\Linear and Logistic Regression\Impact_of_Mobile_Phone_on_Students_Health.csv'
def new_func(file_path):
    return pd.read_csv(file_path)

data = new_func(file_path)

# Display the dataset summary
print("Dataset Summary:")
print(data.describe(include='all'))

# Display the head and tail of the dataset
print("\nDataset Head:")
print(data.head())

print("\nDataset Tail:")
print(data.tail())

# Data Preprocessing
# Drop non-numeric and irrelevant columns
data = data.drop(columns=['Names'])

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the data into features and target variable
X = data.drop(columns=['Performance impact'])  # Assuming 'Performance impact' is the target
y = data['Performance impact']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lin)
print(f'Linear Regression Mean Squared Error: {mse}')

# Logistic Regression (only if the target is binary)
log_reg = None
if len(y.unique()) == 2:
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_log)
    print(f'Logistic Regression Accuracy: {accuracy}')

# Function for user input prediction
def predict_from_input(model_type, input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    if model_type == 'linear':
        prediction = lin_reg.predict(input_df)
        return prediction[0]
    elif model_type == 'logistic' and log_reg is not None:
        prediction = log_reg.predict(input_df)
        return prediction[0]

# Function to create a structured tabular output
def create_output_table(model_type, input_data, prediction):
    output_data = input_data.copy()
    output_data['Prediction'] = prediction
    table = pd.DataFrame([output_data])
    print(tabulate(table, headers='keys', tablefmt='pretty'))

# GUI for user input
def create_gui():
    input_widgets = {}
    for column in X.columns:
        if column in label_encoders:
            le = label_encoders[column]
            input_widgets[column] = widgets.Dropdown(
                options=list(le.classes_),
                description=column,
            )
        else:
            input_widgets[column] = widgets.FloatText(description=column)
    
    display_widgets = widgets.VBox(list(input_widgets.values()))
    
    output = widgets.Output()
    
    def on_button_clicked(b):
        with output:
            clear_output()
            input_data = {col: input_widgets[col].value for col in X.columns}
            for col in input_data:
                if col in label_encoders:
                    input_data[col] = label_encoders[col].transform([input_data[col]])[0]
            
            lin_prediction = predict_from_input('linear', input_data)
            print("Linear Regression Prediction:")
            create_output_table('Linear Regression', input_data, lin_prediction)
            
            if log_reg is not None:
                log_prediction = predict_from_input('logistic', input_data)
                print("\nLogistic Regression Prediction:")
                create_output_table('Logistic Regression', input_data, log_prediction)
    
    predict_button = widgets.Button(description="Predict")
    predict_button.on_click(on_button_clicked)
    
    display(display_widgets, predict_button, output)

# Display dataset head and tail
def display_dataset_info():
    print("\nDataset Head:")
    print(tabulate(data.head(), headers='keys', tablefmt='pretty'))
    
    print("\nDataset Tail:")
    print(tabulate(data.tail(), headers='keys', tablefmt='pretty'))
    
    print("\nTraining Data:")
    print(tabulate(X_train.head(), headers='keys', tablefmt='pretty'))

display_dataset_info()
# Create the GUI
create_gui()
