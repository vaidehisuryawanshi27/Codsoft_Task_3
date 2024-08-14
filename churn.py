import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the best model and preprocessor
pipeline = joblib.load('best_churn_model.pkl')
preprocessor = joblib.load('scaler.pkl')

def predict_churn():
    try:
        # Get user input values
        credit_score = int(credit_score_var.get())
        geography = geography_var.get()
        gender = gender_var.get()
        age = int(age_var.get())
        tenure = int(tenure_var.get())
        balance = float(balance_var.get())
        num_products = int(num_products_var.get())
        has_cr_card = has_cr_card_var.get()
        is_active_member = is_active_member_var.get()
        estimated_salary = float(estimated_salary_var.get())

        # Convert categorical data to numerical
        gender_encoded = {'Male': 0, 'Female': 1}[gender]
        geography_encoded = {'France': 0, 'Spain': 1, 'Germany': 2}[geography]
        has_cr_card_encoded = {'No': 0, 'Yes': 1}[has_cr_card]
        is_active_member_encoded = {'No': 0, 'Yes': 1}[is_active_member]

        # Create input data dictionary
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography_encoded,
            'Gender': gender_encoded,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': has_cr_card_encoded,
            'IsActiveMember': is_active_member_encoded,
            'EstimatedSalary': estimated_salary
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess input data using the preprocessor
        input_data_transformed = preprocessor.transform(input_df)
        
        # Predict churn
        prediction = pipeline.predict(input_data_transformed)
        prediction_proba = pipeline.predict_proba(input_data_transformed)

        result = "Customer will leave" if prediction[0] == 1 else "Customer will stay"
        prob_stay = prediction_proba[0][0]
        prob_leave = prediction_proba[0][1]

        # Update result labels
        result_label.config(text=f"Prediction: {result}", font=("Helvetica", 20))
        prob_stay_label.config(text=f"Probability of staying: {prob_stay:.2f}")
        prob_leave_label.config(text=f"Probability of leaving: {prob_leave:.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Initialize the main window
root = tk.Tk()
root.title("Customer Churn Prediction")
root.geometry("800x600")  # Set a fixed window size

# Create main frame
frame = tk.Frame(root, bg='lightgrey', padx=20, pady=20)
frame.pack(fill='both', expand=True)

# Create input widgets frame
input_frame = tk.Frame(frame, bg='lightgrey')
input_frame.pack(side='left', fill='y', padx=20)

# Create result widgets frame
result_frame = tk.Frame(frame, bg='lightgrey')
result_frame.pack(side='right', fill='both', expand=True)

# Define variables
credit_score_var = tk.IntVar(value=650)
geography_var = tk.StringVar(value='France')
gender_var = tk.StringVar(value='Male')
age_var = tk.IntVar(value=30)
tenure_var = tk.IntVar(value=1)
balance_var = tk.DoubleVar(value=0.0)
num_products_var = tk.IntVar(value=1)
has_cr_card_var = tk.StringVar(value='No')
is_active_member_var = tk.StringVar(value='Yes')
estimated_salary_var = tk.DoubleVar(value=50000.0)

# Create input widgets
tk.Label(input_frame, text="Credit Score", bg='lightgrey').grid(row=0, column=0, padx=10, pady=5, sticky='w')
tk.Scale(input_frame, from_=0, to=850, orient='horizontal', variable=credit_score_var, length=300).grid(row=0, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Geography", bg='lightgrey').grid(row=1, column=0, padx=10, pady=5, sticky='w')
geography_menu = tk.OptionMenu(input_frame, geography_var, 'France', 'Spain', 'Germany')
geography_menu.config(width=15)
geography_menu.grid(row=1, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Gender", bg='lightgrey').grid(row=2, column=0, padx=10, pady=5, sticky='w')
gender_menu = tk.OptionMenu(input_frame, gender_var, 'Male', 'Female')
gender_menu.config(width=15)
gender_menu.grid(row=2, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Age", bg='lightgrey').grid(row=3, column=0, padx=10, pady=5, sticky='w')
tk.Scale(input_frame, from_=18, to=100, orient='horizontal', variable=age_var, length=300).grid(row=3, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Tenure (in years)", bg='lightgrey').grid(row=4, column=0, padx=10, pady=5, sticky='w')
tk.Scale(input_frame, from_=0, to=10, orient='horizontal', variable=tenure_var, length=300).grid(row=4, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Balance", bg='lightgrey').grid(row=5, column=0, padx=10, pady=5, sticky='w')
tk.Scale(input_frame, from_=-100000, to=200000, orient='horizontal', variable=balance_var, length=300).grid(row=5, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Number of Products", bg='lightgrey').grid(row=6, column=0, padx=10, pady=5, sticky='w')
tk.Scale(input_frame, from_=1, to=4, orient='horizontal', variable=num_products_var, length=300).grid(row=6, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Has Credit Card", bg='lightgrey').grid(row=7, column=0, padx=10, pady=5, sticky='w')
has_cr_card_menu = tk.OptionMenu(input_frame, has_cr_card_var, 'No', 'Yes')
has_cr_card_menu.config(width=15)
has_cr_card_menu.grid(row=7, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Is Active Member", bg='lightgrey').grid(row=8, column=0, padx=10, pady=5, sticky='w')
is_active_member_menu = tk.OptionMenu(input_frame, is_active_member_var, 'No', 'Yes')
is_active_member_menu.config(width=15)
is_active_member_menu.grid(row=8, column=1, padx=10, pady=5)

tk.Label(input_frame, text="Estimated Salary", bg='lightgrey').grid(row=9, column=0, padx=10, pady=5, sticky='w')
tk.Scale(input_frame, from_=0, to=200000, orient='horizontal', variable=estimated_salary_var, length=300).grid(row=9, column=1, padx=10, pady=5)

# Prediction button
predict_button = tk.Button(input_frame, text="Predict", command=predict_churn, bg='blue', fg='white')
predict_button.grid(row=10, column=0, columnspan=2, padx=10, pady=20)

# Result labels
result_label = tk.Label(result_frame, text="Prediction: ", bg='lightgrey', font=("Helvetica", 20))
result_label.pack(padx=10, pady=10)

prob_stay_label = tk.Label(result_frame, text="Probability of staying: ", bg='lightgrey', font=("Helvetica", 14))
prob_stay_label.pack(padx=10, pady=5)

prob_leave_label = tk.Label(result_frame, text="Probability of leaving: ", bg='lightgrey', font=("Helvetica", 14))
prob_leave_label.pack(padx=10, pady=5)

# Run the application
root.mainloop()
