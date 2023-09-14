from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Load the trained SVM model
with open('SVM.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

# Load the scaler fitted in the Jupyter Notebook
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input data from the HTML form
        retained_earnings_to_total_assets = float(request.form['retained_earnings_to_total_assets'])
        sales_ratio = float(request.form['sales_ratio'])
        profit_to_financial_expenses = float(request.form['profit_to_financial_expenses'])
        current_assets_to_long_term_liabilities = float(request.form['current_assets_to_long_term_liabilities'])
        profit_to_sales = float(request.form['profit_to_sales'])
        retained_earnings_to_total_assets_2 = float(request.form['retained_earnings_to_total_assets_2'])
        liabilities_to_profit_and_depreciation = float(request.form['liabilities_to_profit_and_depreciation'])
        profit_and_depreciation_to_liabilities = float(request.form['profit_and_depreciation_to_liabilities'])
        sales_growth_ratio = float(request.form['sales_growth_ratio'])
        profit_to_total_assets_2 = float(request.form['profit_to_total_assets_2'])

        # Create a numpy array with the input data containing the selected features
        input_data = np.array([[
            retained_earnings_to_total_assets,
            sales_ratio,
            profit_to_financial_expenses,
            current_assets_to_long_term_liabilities,
            profit_to_sales,
            retained_earnings_to_total_assets_2,
            liabilities_to_profit_and_depreciation,
            profit_and_depreciation_to_liabilities,
            sales_growth_ratio,
            profit_to_total_assets_2,
        ]])

        
        # Create a StandardScaler instance
        scaler = StandardScaler()

        # Fit and transform the input features
        input_features_non_bankrupt_scaled = scaler.fit_transform(input_data)

        # Make a prediction using the trained SVM model
        prediction_non_bankrupt = svm_classifier.predict(input_data)

        # Interpret the prediction
        if prediction_non_bankrupt == 1:
            result_non_bankrupt = "bankrupt"
        else:
            result_non_bankrupt = "not bankrupt"



        return render_template('index.html', prediction_text=result_non_bankrupt)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained SVM model
with open('SVM.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

# Load the scaler fitted in the Jupyter Notebook
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define your input features for a non-bankrupt company
input_features_non_bankrupt = pd.DataFrame(
    {
    'retained_earnings_to_total_assets': [0.9],
    'sales_ratio': [0.8],
    'profit_to_financial_expenses': [0.7],
    'current_assets_to_long_term_liabilities': [0.9],
    'profit_to_sales': [0.9],
    'retained_earnings_to_total_assets_2': [0.8],
    'liabilities_to_profit_and_depreciation': [0.7],
    'profit_and_depreciation_to_liabilities': [0.9],
    'sales_growth_ratio': [0.8],
    'profit_to_total_assets_2': [0.8],
}




)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit and transform the input features
input_features_non_bankrupt_scaled = scaler.fit_transform(input_features_non_bankrupt)

# Make a prediction using the trained SVM model
prediction_non_bankrupt = svm_classifier.predict(input_features_non_bankrupt)

# Interpret the prediction
if prediction_non_bankrupt == 1:
    result_non_bankrupt = "bankrupt"
else:
    result_non_bankrupt = "not bankrupt"

# Print the result for the non-bankrupt company
print(f"The company is predicted to be {result_non_bankrupt}.")

'''