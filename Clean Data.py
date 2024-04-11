import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib

def clean_data(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Modify the values
    df['Identified_Skills'] = df['Identified_Skills'].str.replace('python', "'python'")
    df['Identified_Skills'] = df['Identified_Skills'].str.replace("learning'", "'machine learning'")
    df['Identified_Skills'] = df['Identified_Skills'].str.replace('aws', "'aws'")

    # Extract years of experience and add them to the new column
    pattern = r"(\d+)\s*years"
    df['Actual_Years_of_Experience'] = df['Job Description'].apply(lambda x: int(re.findall(pattern, x)[0]) if re.findall(pattern, x) else np.nan)

    # Drop rows with missing years of experience
    df.dropna(subset=['Actual_Years_of_Experience'], inplace=True)

    return df

def update_skills(skills, mlb):
    unique_skills = mlb.classes_
    missing_skills = set(skills) - set(unique_skills)
    if missing_skills:
        # Add specific missing skills to the MultiLabelBinarizer classes
        mlb.classes_ = np.concatenate((unique_skills, list(missing_skills)))

def train_model(df, desired_skills):
    # Encode skills
    mlb = MultiLabelBinarizer()
    skills_encoded = mlb.fit_transform(df['Identified_Skills'].apply(lambda x: re.findall(r"[\w']+|[.,!?;]", x)))
    df_skills = pd.DataFrame(skills_encoded, columns=mlb.classes_)

    # Update skills to include all possible skills including the desired skills
    update_skills(desired_skills, mlb)

    # Encode location
    location_encoder = LabelEncoder()
    df['Location'] = location_encoder.fit_transform(df['Location'])

    # Concatenate features
    X = pd.concat([df[['Location', 'Actual_Years_of_Experience']], df_skills], axis=1)
    y_salary = df['Salary']
    y_title = df['Title']

    # Split data into train and test sets
    X_train, X_test, y_salary_train, y_salary_test, y_title_train, y_title_test = train_test_split(X, y_salary, y_title, test_size=0.2, random_state=42)

    # Train linear regression model for salary prediction
    salary_model = LinearRegression()
    salary_model.fit(X_train, y_salary_train)

    # Train logistic regression model for title prediction
    title_model = LogisticRegression(solver='liblinear')  # Use 'liblinear' solver
    title_model.fit(X_train, y_title_train)

    return salary_model, title_model, location_encoder, mlb, list(X.columns)

# After training the models


def predict_salary(user_input, location_encoder, mlb, salary_model, feature_names):
    # Process user input for salary prediction
    user_location = location_encoder.transform([user_input['Location']])[0]
    user_skills = mlb.transform([user_input['Skills']])
    user_years_of_experience = user_input['Years of Experience']

    # Create DataFrame for prediction
    user_data = pd.DataFrame({
        'Location': user_location,
        'Actual_Years_of_Experience': user_years_of_experience
    }, index=[0])
    user_skills_df = pd.DataFrame(user_skills, columns=mlb.classes_)
    user_data = pd.concat([user_data, user_skills_df], axis=1)  # Include encoded skills

    # Reorder columns to match the order during training
    user_data = user_data[feature_names]

    # Predict salary
    predicted_salary = salary_model.predict(user_data)
    return predicted_salary[0]

def predict_title(user_input, location_encoder, mlb, title_model, feature_names):
    # Process user input for title prediction
    user_location = location_encoder.transform([user_input['Location']])[0]
    user_skills = mlb.transform([user_input['Skills']])
    user_years_of_experience = user_input['Years of Experience']

    # Create DataFrame for prediction
    user_data = pd.DataFrame({
        'Location': user_location,
        'Actual_Years_of_Experience': user_years_of_experience
    }, index=[0])
    user_skills_df = pd.DataFrame(user_skills, columns=mlb.classes_)
    user_data = pd.concat([user_data, user_skills_df], axis=1)  # Include encoded skills

    # Reorder columns to match the order during training
    user_data = user_data[feature_names]

    # Predict title
    predicted_title = title_model.predict(user_data)
    return predicted_title[0]

# File path
file_path = "Modified_Data_and_Years_Of_Experience.csv"

# Clean data
df = clean_data(file_path)

# Define the desired skills list including 'aws', 'machine learning', and 'python'
desired_skills = ['python', 'machine learning', 'aws', ...]  # Add other skills as needed

# Train models
salary_model, title_model, location_encoder, mlb, feature_names = train_model(df, desired_skills)

# Example usage for salary prediction
user_input_salary = {
    'Location': 'CA',
    'Skills': ['python', 'machine learning', 'aws'],
    'Years of Experience': 5
}

# Example usage for title prediction
user_input_title = {
    'Location': 'CA',
    'Skills': ['python', 'machine learning', 'aws'],
    'Years of Experience': 5
}

# Predict salary
predicted_salary = predict_salary(user_input_salary, location_encoder, mlb, salary_model, feature_names)
print("Predicted salary:", predicted_salary)

# Predict title
predicted_title = predict_title(user_input_title, location_encoder, mlb, title_model, feature_names)
print("Predicted title:", predicted_title)

# File paths for saving models
salary_model_path = "salary_prediction_model.pkl"
title_model_path = "title_prediction_model.pkl"
mlb_path = "mlb.pkl"

# Save models
joblib.dump(salary_model, salary_model_path)
joblib.dump(title_model, title_model_path)
joblib.dump(mlb, mlb_path)

print("Models saved successfully.")

# For LinearRegression model (salary_model)
print("LinearRegression model coefficients:", salary_model.coef_)
print("LinearRegression model intercept:", salary_model.intercept_)

# For LogisticRegression model (title_model)
print("LogisticRegression model coefficients:", title_model.coef_)
print("LogisticRegression model intercept:", title_model.intercept_)


#### How to run the script without dealing with the ampersand error
### Open Command Prompt (Windows) or Terminal (Mac/Linux):
###     On Windows, you can do this by searching for "Command Prompt" in the Start menu.
###     On Mac, you can find Terminal in the Utilities folder within the Applications folder.
###     On Linux, you can typically find Terminal in the Applications menu or by searching for "Terminal".
### 
### Navigate to the Directory Containing Your Python Script:
### 
###     Use the cd command followed by the path to the directory where your Python script is located. For example:
### 
###     bash
### 
###     cd C:/Users/szerp/Downloads/AI4ALL_2nd_Attempt
### 
###     This command changes the current directory to the specified directory.
### 
### Execute Your Python Script:
### 
###     Once you're in the correct directory, you can execute your Python script using the Python interpreter.
###     Use the following command to execute your script:
### 
###     arduino
### 
###         "C:/Users/szerp/anaconda3/python.exe" "Clean Data_and_Model_MAIN copy.py"
### 
###         Replace "Clean Data_and_Model_MAIN copy.py" with the name of your Python script if it's different.
###         This command runs the Python interpreter (python.exe) located at the specified path and tells it to execute your script.
### 
###     Review Output:
###         After executing the command, your Python script should run, and any output or error messages will be displayed in the Command Prompt or Terminal window.
### 
### Following these steps should allow you to run your Python script without encountering the ampersand error. If you have any further questions or encounter any issues, feel free to ask!






### Write down Model expected behavior criteria/behavior guardrails
### Expected behavior from common sense: eyeballing the data and intuitively suggesting something like predicted salary should increase/not decrease with more years of experience or adding more skills
### Coastal locations should be higher in salary than remote.
### If model does not meet it, come up with hypothesis talking about why it does not meet the criteria
### Verify if years of experience are being interpreted by model as number/numerical value