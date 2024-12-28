from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import pandas as pd

# load the dataset and prepare features
file_path = input("Please enter the path to your CSV file: ") 
data = pd.read_csv(file_path)

attributes = ['credit_lines_outstanding', 'debt_to_income_ratio', 'payment_to_income_ratio', 'years_employed', 'fico_score']
data['payment_to_income_ratio'] = data['loan_amt_outstanding'] / data['income']
data['debt_to_income_ratio'] = data['total_debt_outstanding'] / data['income']

# initialize and train the logistic regression model
model = LogisticRegression(random_state=0, solver='liblinear', tol=1e-5, max_iter=10000)
model.fit(data[attributes], data['default'])

# calculate expected financial loss based on the trained model
def calculate_expected_loss(data_row):
    data_row['payment_to_income_ratio'] = data_row['loan_amt_outstanding'] / data_row['income']
    data_row['debt_to_income_ratio'] = data_row['total_debt_outstanding'] / data_row['income']
    probability_of_default = model.predict_proba(data_row[attributes].values.reshape(1, -1))
    return probability_of_default[0, 1] * data_row['loan_amt_outstanding'] * 0.1
