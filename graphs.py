import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'E:/PROJECT - AI & ML/Linear and Logistic Regression/Impact_of_Mobile_Phone_on_Students_Health.csv'
df = pd.read_csv(file_path)

# Strip any leading or trailing spaces from the column names
df.columns = df.columns.str.strip()

# Display dataset head and tail
print("Dataset Head:")
print(df.head())
print("\nDataset Tail:")
print(df.tail())

# Bar Chart 1: Distribution of 'Age'
plt.figure(figsize=(10, 6))
df['Age'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Pie Chart 1: Distribution of 'Gender'
plt.figure(figsize=(8, 8))
df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
plt.title('Distribution of Gender')
plt.ylabel('')
plt.show()

# Bar Chart 2: Distribution of 'Mobile phone use for education'
plt.figure(figsize=(10, 6))
df['Mobile phone use for education'].value_counts().plot(kind='bar', color='lightcoral')
plt.title('Distribution of Mobile Phone Use for Education')
plt.xlabel('Mobile Phone Use for Education')
plt.ylabel('Count')
plt.show()

# Pie Chart 2: Distribution of 'Helpful for studying'
plt.figure(figsize=(8, 8))
df['Helpful for studying'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightyellow'])
plt.title('Distribution of Helpful for Studying')
plt.ylabel('')
plt.show()

# Bar Chart 3: Distribution of 'Daily usages'
plt.figure(figsize=(10, 6))
df['Daily usages'].value_counts().plot(kind='bar', color='lightseagreen')
plt.title('Distribution of Daily Usages')
plt.xlabel('Daily Usages')
plt.ylabel('Count')
plt.show()

# Pie Chart 3: Distribution of 'Performance impact'
plt.figure(figsize=(8, 8))
df['Performance impact'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightseagreen', 'lightpink'])
plt.title('Distribution of Performance Impact')
plt.ylabel('')
plt.show()

# Bar Chart 4: Distribution of 'Usage distraction'
plt.figure(figsize=(10, 6))
df['Usage distraction'].value_counts().plot(kind='bar', color='plum')
plt.title('Distribution of Usage Distraction')
plt.xlabel('Usage Distraction')
plt.ylabel('Count')
plt.show()

# Pie Chart 4: Distribution of 'Health Risks'
plt.figure(figsize=(8, 8))
df['Health Risks'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['plum', 'lightsalmon'])
plt.title('Distribution of Health Risks')
plt.ylabel('')
plt.show()

# Bar Chart 5: Distribution of 'Symptoms'
plt.figure(figsize=(10, 6))
df['Symptoms'].value_counts().plot(kind='bar', color='lightsteelblue')
plt.title('Distribution of Symptoms')
plt.xlabel('Symptoms')
plt.ylabel('Count')
plt.show()

# Pie Chart 5: Distribution of 'Precautions'
plt.figure(figsize=(8, 8))
df['Precautions'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightsteelblue', 'lightcyan'])
plt.title('Distribution of Precautions')
plt.ylabel('')
plt.show()
