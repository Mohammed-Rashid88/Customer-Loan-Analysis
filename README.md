# Customer-Loan-Analysis
**This project involves building a logistic regression model to predict the probability of loan defaults based on various borrower attributes. It also includes a function to calculate the expected financial loss based on the predicted probability of default.**

**Requirements:**

Python 3.x

scikit-learn library

pandas library

numpy library

**You can install the required libraries using pip:**

_pip install scikit-learn pandas numpy_

**Files:**

**Loan_Data.csv:** A dataset containing loan information and borrower attributes.

**Columns:** loan_amt_outstanding, income, total_debt_outstanding, fico_score, default

**Usage:**

**Load the dataset:** The script loads loan data from a CSV file.

**Train the model:** A logistic regression model is trained using the specified features: credit_lines_outstanding, debt_to_income_ratio, payment_to_income_ratio, years_employed, and fico_score.

**Calculate Expected Loss: **The calculate_expected_loss function calculates the expected financial loss based on the predicted probability of default for each loan.
