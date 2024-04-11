import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
import ast  # Import the ast module

# Load the data
file_path = r"C:\Users\szerp\OneDrive\Documents\GitHub\AI4ALL_Project\Modified_Data_and_Years_Of_Experience.csv"
df = pd.read_csv(file_path)

# Safely convert 'Identified_Skills' from string to list
df['Identified_Skills'] = df['Identified_Skills'].apply(lambda x: ast.literal_eval(x))

# Proceed with the MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df_skills_encoded = mlb.fit_transform(df['Identified_Skills'])
df_skills = pd.DataFrame(df_skills_encoded, columns=mlb.classes_)

# Encoding location
location_encoder = LabelEncoder()
df['Location'] = location_encoder.fit_transform(df['Location'])

# Preparing features and labels
X = pd.concat([df[['Location', 'Actual_Years_of_Experience']], df_skills], axis=1)
y_salary = df['Salary']
y_title = df['Title']

# Splitting data
X_train, X_test, y_salary_train, y_salary_test, y_title_train, y_title_test = train_test_split(
    X, y_salary, y_title, test_size=0.2, random_state=42)

# Training models
salary_model = LinearRegression()
salary_model.fit(X_train, y_salary_train)

title_model = LogisticRegression(max_iter=10000)  # Increase max_iter for convergence
title_model.fit(X_train, y_title_train)

# Saving models and encoders
joblib.dump(location_encoder, 'location_encoder.pkl')
joblib.dump(mlb, 'mlb.pkl')
joblib.dump(salary_model, 'salary_prediction_model.pkl')
joblib.dump(title_model, 'title_prediction_model.pkl')

print("Encoders and models saved successfully.")

# Prediction and testing functionality
def make_predictions(location, skills, years_of_experience):
    location_encoded = location_encoder.transform([location])[0]
    skills_encoded = mlb.transform([skills])
    user_data = pd.DataFrame(np.hstack(([[location_encoded, years_of_experience]], skills_encoded)), columns=['Location', 'Actual_Years_of_Experience'] + list(mlb.classes_))
    predicted_salary = salary_model.predict(user_data)[0]
    predicted_title = title_model.predict(user_data)[0]
    return predicted_salary, predicted_title

# Test cases
def run_tests():
    test_cases = [
        ("NY", ["python", "machine learning"], 3),
        ("CA", ["aws", "python", "data analysis"], 5),
        ("TX", ["project management", "leadership"], 2)
    ]
    
    for idx, (location, skills, years) in enumerate(test_cases, 1):
        predicted_salary, predicted_title = make_predictions(location, skills, years)
        print(f"Test Case {idx}: Location: {location}, Skills: {skills}, Years: {years}")
        print(f"Predicted Salary: {predicted_salary}, Predicted Title: {predicted_title}\n")


# Run the tests
run_tests()

# Assuming your MultiLabelBinarizer was saved as 'mlb.pkl'
mlb = joblib.load('mlb.pkl')

# Print the skills the model was trained on
print("Skills the model was trained on:")
for skill in mlb.classes_:
    print(skill)

def test_experience_impact(location, skills, start_experience, end_experience, step=1):
    """
    Tests the impact of years of experience on the predicted salary.
    """
    results = []
    for experience in range(start_experience, end_experience + 1, step):
        predicted_salary, _ = make_predictions(location, skills, experience)
        results.append((experience, predicted_salary))
    return results

# Example test execution
if __name__ == "__main__":
    location = "CA"
    skills = ["python", "machine learning"]
    start_experience = 1
    end_experience = 20
    results = test_experience_impact(location, skills, start_experience, end_experience)
    for experience, salary in results:
        print(f"Experience: {experience}, Predicted Salary: {salary}")


def test_title_and_salary_impact(locations, skills_sets, years_of_experience):
    """
    Tests the impact of different skills, locations, and years of experience on the predicted job title and salary.
    Args:
        locations (list): List of locations to test.
        skills_sets (list of lists): List of skill sets to test.
        years_of_experience (list): List of years of experience to test.
    
    Returns:
        dict: A dictionary with the test parameters as keys and the predicted titles and salaries as values.
    """
    results = {}
    for location in locations:
        for skills in skills_sets:
            for years in years_of_experience:
                predicted_salary, predicted_title = make_predictions(location, skills, years)
                test_case = f"Location: {location}, Skills: {skills}, Years: {years}"
                results[test_case] = {'Predicted Title': predicted_title, 'Predicted Salary': predicted_salary}
    return results

# Example test execution
if __name__ == "__main__":
    locations = ["CA", "NY", "Remote", "FL"]
    skills_sets = [["python", "sql"], ["aws", "docker", "machine learning"]]
    years_of_experience_list = [1, 5, 10]

    impact_results = test_title_and_salary_impact(locations, skills_sets, years_of_experience_list)
    for test_case, outcomes in impact_results.items():
        print(f"{test_case} -> Predicted Title: {outcomes['Predicted Title']}, Predicted Salary: {outcomes['Predicted Salary']}")


def test_salary_realism(location, skills, years_of_experience, expected_salary_range):
    """
    Tests if the predicted salary falls within an expected range for given parameters.
    Args:
        location (str): The location to test.
        skills (list): The skills to test.
        years_of_experience (int): The years of experience to test.
        expected_salary_range (tuple): A tuple containing the minimum and maximum expected salary.
    
    Returns:
        bool: True if the predicted salary falls within the expected range, False otherwise.
    """
    predicted_salary, _ = make_predictions(location, skills, years_of_experience)
    return expected_salary_range[0] <= predicted_salary <= expected_salary_range[1]

if __name__ == "__main__":
    location = "Remote"
    skills = ["python", "machine learning", "sql"]
    years_of_experience = 10
    expected_salary_range = (150000, 180000)  # Range estimated by ChatGPT "For a Data Scientist working remotely with 10 years of experience, and knowledge in Python, Machine Learning, and SQL, the expected salary range could be approximately $150,000 to $180,000 according to the information found on Remotely​ (Remotely)​. This range is indicative of a Senior Data Scientist role and reflects the high demand and value of experienced professionals in this field."

    realism_test_result = test_salary_realism(location, skills, years_of_experience, expected_salary_range)
    print(f"Salary realism test {'passed' if realism_test_result else 'failed'}.")

