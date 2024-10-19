import pandas as pd
import numpy as np
# Set a random seed for reproducibility
np.random.seed(42)
# Number of students
num_students = 3000

# Generate random data for each feature
data = {
    'student_id': range(1, num_students + 1),
    'cgpa': np.random.uniform(2.0, 10.0, num_students),  # CGPA between 2.0 and 10.0
    'hackathons': np.random.randint(0, 6, num_students),  # Number of hackathons (0 to 5)
    'papers_presented': np.random.randint(0, 11, num_students),  # Papers presented (0 to 10)
    'core_courses': np.random.randint(3, 8, num_students),  # Core courses (3 to 7)
    'teacher_assistance': np.random.randint(0, 6, num_students),  # Teacher assistance (0 to 5)
    'sem_consistency': np.random.uniform(0.0, 1.0, num_students)  # Sem consistency between 0 and 1
}

# Create a DataFrame
student_data = pd.DataFrame(data)

# Save to CSV file
student_data.to_csv('student_data.csv', index=False)

# Display the first few rows of the dataset
#print(student_data.head())
print(student_data.to_string())





import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
def get_top_students():
 data = pd.read_csv('student_data.csv')


 # Ensure 'data' is a DataFrame
if isinstance(data, dict):
    data = pd.DataFrame(data)


# Normalize the data
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data.iloc[:, 1:]), columns=data.columns[1:])
normalized_data.insert(0, 'student_id', data['student_id'])




# Define weights for each factor
weights = {
    'cgpa': 0.4,
    'hackathons': 0.15,
    'papers_presented': 0.15,
    'core_courses': 0.15,
    'teacher_assistance': 0.1,
    'sem_consistency': 0.05
}

# Calculate the weighted sum for each student
normalized_data['total_score'] = (
    normalized_data['cgpa'] * weights['cgpa'] +
    normalized_data['hackathons'] * weights['hackathons'] +
    normalized_data['papers_presented'] * weights['papers_presented'] +
    normalized_data['core_courses'] * weights['core_courses'] +
    normalized_data['teacher_assistance'] * weights['teacher_assistance'] +
    normalized_data['sem_consistency'] * weights['sem_consistency']
)

# Rank students based on total score
normalized_data['rank'] = normalized_data['total_score'].rank(ascending=False)

# Label the top 3 students as 1 (top performers) and others as 0
normalized_data['label'] = 0
normalized_data.loc[normalized_data['rank'] <= 3, 'label'] = 1

# Drop unnecessary columns for training
X = normalized_data.drop(columns=['student_id', 'total_score', 'rank', 'label'])
y = normalized_data['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
(classification_report(y_test, y_pred))


# Predict the probabilities for each student
normalized_data['probability'] = rf_classifier.predict_proba(X)[:, 0]  # Use the probability of being a top performer (class 1)


# After predicting probabilities
normalized_data['probability'] = rf_classifier.predict_proba(X)[:, 0]

# Create a combined score by weighing both the total_score and probability
normalized_data['combined_score'] = (
    normalized_data['total_score'] * 0.5 +  # Weight for the weighted score
    normalized_data['probability'] * 0.5    # Weight for the probability
)

# Rank students based on the combined score
normalized_data['predicted_rank'] = normalized_data['combined_score'].rank(ascending=False, method='first')

# Sort by predicted rank and select the top 3 students
top_students = normalized_data.sort_values(by='predicted_rank').head(3)

# Display the top 3 predicted students with unique ranks
print("Top 3 Predicted Students:")
print(top_students[['student_id', 'combined_score', 'predicted_rank']])


