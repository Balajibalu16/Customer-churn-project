CUSTOMER CHURN PREDICTION PROJECT
=================================

FILES
-----
1. customer_churn_dataset_1000_rows.csv : 1,000 synthetic customer records.
2. customer_churn_dashboard.xlsx        : Excel dashboard + raw data + summary sheets.
3. customer_churn_prediction.py         : Python machine learning training/evaluation program.
4. customer_churn_project_README.txt    : This guide.

DATASET COLUMNS
---------------
CustomerID          Unique customer ID (exclude from ML features).
Gender              Customer gender.
SeniorCitizen       1 = senior citizen, 0 = not senior citizen.
Partner             Whether customer has a partner.
Dependents          Whether customer has dependents.
TenureMonths        Number of months as a customer.
PhoneService        Whether phone service is used.
MultipleLines       Multiple phone lines status.
InternetService     Fiber optic, DSL, or no internet service.
OnlineSecurity      Online security subscription status.
OnlineBackup        Online backup subscription status.
DeviceProtection    Device protection subscription status.
TechSupport         Tech support subscription status.
StreamingTV         Streaming TV subscription status.
StreamingMovies     Streaming movies subscription status.
Contract            Month-to-month, one year, or two year.
PaperlessBilling    Paperless billing status.
PaymentMethod       Customer payment method.
MonthlyCharges      Monthly service charges.
TotalCharges        Approximate total charges over tenure.
Churn               Target label: Yes = churned, No = retained.

DASHBOARD METRICS
-----------------
Total customers: 1000
Churned customers: 185
Retained customers: 815
Churn rate: 18.5%
Average monthly charges: 70.74
Average tenure: 36.6 months

RUN PYTHON MODEL
----------------
1. Keep the CSV and Python file in the same folder.
2. Install Python 3.10+ if needed.
3. Open Command Prompt/Terminal in that folder.
4. Run:
   pip install pandas scikit-learn matplotlib joblib
5. Run:
   python customer_churn_prediction.py

IMPORTANT
---------
This is synthetic data generated for educational use, not real customer records.
Model metrics are only a demonstration and should not be treated as real business
performance. For a real project, use a verified real-world dataset and validate
the model on future/unseen customers.
