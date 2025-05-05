import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load dataset from CSV
df = pd.read_csv("UMALAY_Dataset.csv")

# Drop ID column and separate features/target
X = df.drop(columns=["ID", "Avails_Membership"])
y = df["Avails_Membership"]

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Map Yes/No to binary
def yes_no_to_binary(value):
    return 1 if value == "Yes" else 0

# Prediction function
def predict_membership():
    try:
        age = yes_no_to_binary(age_var.get())
        job = yes_no_to_binary(job_var.get())
        income = yes_no_to_binary(income_var.get())

        input_data = np.array([[age, job, income]])
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        result = "YES" if prediction == 1 else "NO"
        messagebox.showinfo("Prediction", f"Will avail membership? {result}\nProbability: {prob:.2f}")

    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

# Tkinter GUI setup
root = tk.Tk()
root.title("Membership Prediction")

# Labels
tk.Label(root, text="Are you above the age of 25?").grid(row=0, column=0, padx=10, pady=5)
tk.Label(root, text="Do you have a job?").grid(row=1, column=0, padx=10, pady=5)
tk.Label(root, text="Is your income above $1000?").grid(row=2, column=0, padx=10, pady=5)

# Dropdowns
options = ["Yes", "No"]
age_var = tk.StringVar(value="Yes")
job_var = tk.StringVar(value="Yes")
income_var = tk.StringVar(value="Yes")

tk.OptionMenu(root, age_var, *options).grid(row=0, column=1)
tk.OptionMenu(root, job_var, *options).grid(row=1, column=1)
tk.OptionMenu(root, income_var, *options).grid(row=2, column=1)

# Predict button
tk.Button(root, text="Predict", command=predict_membership).grid(row=3, column=0, columnspan=2, pady=10)

root.mainloop()
